"""Supervised Training for FractalDecoder.

This module trains the FractalDecoder to map Log-Signatures to Wavelet
coefficients using ground-truth fBM data. The decoder learns to "hallucinate"
high-frequency details from the geometric summary (signature) of rough paths.

Training Pipeline:
1. Generate ground-truth fBM paths (H ≈ 0.1)
2. Compute Log-Signatures (via signatory) as input features
3. Compute Ground-Truth Wavelet coefficients as targets
4. Train decoder with MSE loss between predicted and true coefficients

Usage:
    python fractalsig/train_decoder.py --n_samples 10000 --epochs 100
"""

import argparse
import json
import logging
from pathlib import Path

import iisignature
import numpy as np
import pywt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fractalsig.data_gen import generate_rough_paths
from fractalsig.decoder import FractalDecoder

log = logging.getLogger(__name__)


class RoughPathDataset(Dataset):
    """Dataset of rough paths with pre-computed signatures and wavelet coefficients.

    Pre-computes:
    1. Log-Signatures: Compact geometric summaries (input to decoder)
    2. Wavelet Coefficients: Multi-scale decomposition (target for decoder)

    Both inputs and targets are normalized to zero mean/unit variance. When
    `train_stats` is provided, normalization is applied with the supplied
    statistics instead of recomputing — this is how val/test splits stay on
    the same scale as the training split.

    Attributes:
        paths: Input paths of shape (n_samples, seq_len, n_channels).
        log_signatures: Pre-computed log-signatures (n_samples, logsig_dim).
        wavelet_coeffs: Flattened wavelet coefficients (n_samples, coeff_dim).
        logsig_stats: Dict with 'mean' and 'std' for log-signatures.
        coeff_stats: Dict with 'mean' and 'std' for wavelet coefficients.
    """

    def __init__(
        self,
        paths: torch.Tensor,
        sig_depth: int = 4,
        wavelet: str = "db4",
        level: int | None = None,
        train_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> None:
        """Initialize the dataset from a tensor of paths.

        Args:
            paths: Tensor of shape (n_samples, seq_len, n_channels).
            sig_depth: Depth of log-signature computation.
            wavelet: Wavelet family for decomposition.
            level: Wavelet decomposition level (auto if None).
            train_stats: Optional dict with keys 'logsig' and 'coeff', each a
                stats dict with 'mean' and 'std'. When provided, the dataset
                is normalized with these stats instead of recomputing — used
                to keep val/test on the train split's scale.
        """
        super().__init__()

        self.paths = paths
        self.n_samples, self.seq_len, self.n_channels = (
            int(paths.shape[0]),
            int(paths.shape[1]),
            int(paths.shape[2]),
        )
        self.sig_depth = sig_depth
        self.wavelet = wavelet

        if level is None:
            wavelet_obj = pywt.Wavelet(wavelet)
            max_level = pywt.dwt_max_level(self.seq_len, wavelet_obj.dec_len)
            self.level = max(1, min(max_level, 6))
        else:
            self.level = level

        log.info("Pre-computing log-signatures...")
        self.log_signatures = self._compute_log_signatures()

        log.info("Pre-computing wavelet coefficients...")
        self.wavelet_coeffs = self._compute_wavelet_coeffs()

        log.info("Normalizing data...")
        if train_stats is None:
            self.log_signatures, self.logsig_stats = self._normalize(self.log_signatures)
            self.wavelet_coeffs, self.coeff_stats = self._normalize(self.wavelet_coeffs)
        else:
            self.logsig_stats = train_stats["logsig"]
            self.coeff_stats = train_stats["coeff"]
            self.log_signatures = self._apply_stats(self.log_signatures, self.logsig_stats)
            self.wavelet_coeffs = self._apply_stats(self.wavelet_coeffs, self.coeff_stats)

        log.info(f"Dataset ready: logsig_dim={self.log_signatures.shape[1]}, "
              f"coeff_dim={self.wavelet_coeffs.shape[1]}")

    def _compute_log_signatures(self) -> torch.Tensor:
        """Compute log-signatures for all paths using iisignature.

        iisignature is a stable numpy-based library for signature computation.
        Results are converted to torch tensors for training.
        """
        # Add time augmentation for signature computation
        # Shape: (batch, seq_len, n_channels + 1)
        batch_size = self.paths.shape[0]
        time = torch.linspace(0, 1, self.seq_len).view(1, -1, 1).expand(batch_size, -1, -1)
        augmented_paths = torch.cat([time, self.paths], dim=-1)

        # Convert to numpy for iisignature
        paths_np = augmented_paths.numpy()
        n_channels_aug = paths_np.shape[2]

        # Prepare iisignature (precompute basis)
        s = iisignature.prepare(n_channels_aug, self.sig_depth)

        # Compute log-signatures for each path
        logsigs = []
        for i in range(batch_size):
            logsig = iisignature.logsig(paths_np[i], s)
            logsigs.append(logsig)

        return torch.from_numpy(np.array(logsigs)).float()

    def _compute_wavelet_coeffs(self) -> torch.Tensor:
        """Compute flattened wavelet coefficients for all paths.

        Performs DWT on each channel and concatenates all coefficient
        arrays into a single flat vector per sample.
        """
        all_coeffs = []

        for i in range(self.n_samples):
            sample_coeffs = []
            for c in range(self.n_channels):
                signal = self.paths[i, :, c].numpy()
                coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
                flat = np.concatenate([c.flatten() for c in coeffs])
                sample_coeffs.append(flat)
            all_coeffs.append(np.concatenate(sample_coeffs))

        return torch.from_numpy(np.array(all_coeffs)).float()

    def _normalize(
        self,
        tensor: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Normalize tensor to zero mean/unit variance.

        Args:
            tensor: Input tensor (n_samples, dim).

        Returns:
            Tuple of (normalized tensor, stats dict with mean/std).
        """
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-8)

        normalized = (tensor - mean) / std

        stats = {"mean": mean.squeeze(0), "std": std.squeeze(0)}
        return normalized, stats

    @staticmethod
    def _apply_stats(
        tensor: torch.Tensor,
        stats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply pre-computed normalization stats to a tensor."""
        mean = stats["mean"].unsqueeze(0)
        std = torch.clamp(stats["std"].unsqueeze(0), min=1e-8)
        return (tensor - mean) / std

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.log_signatures[idx], self.wavelet_coeffs[idx]

    def save_stats(self, filepath: str) -> None:
        """Save normalization statistics to file."""
        stats = {
            "logsig_mean": self.logsig_stats["mean"].numpy().tolist(),
            "logsig_std": self.logsig_stats["std"].numpy().tolist(),
            "coeff_mean": self.coeff_stats["mean"].numpy().tolist(),
            "coeff_std": self.coeff_stats["std"].numpy().tolist(),
            "logsig_dim": self.log_signatures.shape[1],
            "coeff_dim": self.wavelet_coeffs.shape[1],
            "seq_len": self.seq_len,
            "n_channels": self.n_channels,
            "wavelet": self.wavelet,
            "level": self.level,
            "sig_depth": self.sig_depth,
        }
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)
        log.info(f"Saved normalization stats to {filepath}")


def train(
    n_samples: int = 10000,
    seq_len: int = 256,
    n_channels: int = 1,
    H: float = 0.1,
    sig_depth: int = 4,
    hidden_dim: int = 256,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    patience: int = 20,
    checkpoint_dir: str = "checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int | None = 42,
) -> dict[str, object]:
    """Train the FractalDecoder on ground-truth fBM data.

    Generates `n_samples` rough paths once, splits them into train/val,
    pre-computes signatures and wavelet coefficients on each split, and
    trains the decoder. The best checkpoint is selected by validation
    loss; training stops early after `patience` epochs without improvement.

    Args:
        n_samples: Total number of paths (split into train/val).
        seq_len: Sequence length.
        n_channels: Number of channels.
        H: Hurst parameter.
        sig_depth: Log-signature depth.
        hidden_dim: Decoder hidden dimension.
        batch_size: Training batch size.
        epochs: Maximum number of training epochs.
        lr: Peak learning rate for OneCycleLR.
        val_frac: Fraction of samples held out for validation.
        patience: Stop after this many epochs without val-loss improvement.
        checkpoint_dir: Directory for saving checkpoints.
        device: Training device.
        seed: Random seed.

    Returns:
        Dict with keys: best_val_loss, final_train_loss, epochs_trained,
        history (dict of train_loss / val_loss lists).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")

    log.info(f"Generating {n_samples} rough paths (H={H}, seq_len={seq_len})...")
    all_paths = generate_rough_paths(
        n_paths=n_samples,
        seq_len=seq_len,
        n_channels=n_channels,
        H=H,
        seed=seed,
        standardize=True,
    )

    n_val = max(1, int(round(n_samples * val_frac)))
    n_train = n_samples - n_val
    if n_train < 1:
        raise ValueError(
            f"val_frac={val_frac} leaves no training samples (n_samples={n_samples})"
        )
    perm = torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed or 0))
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train_paths = all_paths[train_idx]
    val_paths = all_paths[val_idx]

    train_dataset = RoughPathDataset(
        paths=train_paths,
        sig_depth=sig_depth,
    )
    val_dataset = RoughPathDataset(
        paths=val_paths,
        sig_depth=sig_depth,
        wavelet=train_dataset.wavelet,
        level=train_dataset.level,
        train_stats={
            "logsig": train_dataset.logsig_stats,
            "coeff": train_dataset.coeff_stats,
        },
    )

    stats_path = checkpoint_path / "normalization_stats.json"
    train_dataset.save_stats(str(stats_path))

    pin = device == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    input_dim = train_dataset.log_signatures.shape[1]
    model = FractalDecoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_seq_len=seq_len,
        out_channels=n_channels,
        wavelet=train_dataset.wavelet,
        level=train_dataset.level,
    ).to(device)

    log.info(f"\nModel: {model.get_num_params():,} parameters")
    log.info(f"Input dim: {input_dim}, Output coeff dim: {model.total_coeff_dim}")
    log.info(f"Train/Val: {n_train}/{n_val} samples (val_frac={val_frac})")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.3,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    avg_train_loss = float("nan")
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    log.info(f"Starting training for up to {epochs} epochs (patience={patience})...")

    pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    last_epoch = -1
    for epoch in pbar:
        last_epoch = epoch
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for logsig_batch, coeff_batch in train_loader:
            logsig_batch = logsig_batch.to(device)
            coeff_batch = coeff_batch.to(device)

            optimizer.zero_grad()
            pred_coeffs = model.mlp(logsig_batch)
            loss = criterion(pred_coeffs, coeff_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(1, n_batches)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        with torch.no_grad():
            for logsig_batch, coeff_batch in val_loader:
                logsig_batch = logsig_batch.to(device)
                coeff_batch = coeff_batch.to(device)
                pred_coeffs = model.mlp(logsig_batch)
                val_loss_total += criterion(pred_coeffs, coeff_batch).item()
                val_batches += 1
        avg_val_loss = val_loss_total / max(1, val_batches)
        history["val_loss"].append(avg_val_loss)

        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            "Train": f"{avg_train_loss:.5f}",
            "Val": f"{avg_val_loss:.5f}",
            "Best": f"{best_val_loss:.5f}",
            "LR": f"{current_lr:.2e}",
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "train_loss": avg_train_loss,
                    "config": {
                        "input_dim": input_dim,
                        "hidden_dim": hidden_dim,
                        "seq_len": seq_len,
                        "n_channels": n_channels,
                        "wavelet": train_dataset.wavelet,
                        "level": train_dataset.level,
                    },
                },
                checkpoint_path / "fractal_decoder_best.pth",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no val improvement for {patience} epochs)"
                )
                break

    log.info(f"Training complete. Best val loss: {best_val_loss:.6f}")
    log.info(f"Model saved to: {checkpoint_path / 'fractal_decoder_best.pth'}")

    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "final_train_loss": avg_train_loss,
        "best_val_loss": best_val_loss,
        "epochs_trained": last_epoch + 1,
        "history": history,
    }


def main() -> None:
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train FractalDecoder on ground-truth fBM data"
    )
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of training samples")
    parser.add_argument("--seq_len", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--n_channels", type=int, default=1,
                        help="Number of channels")
    parser.add_argument("--H", type=float, default=0.1,
                        help="Hurst parameter")
    parser.add_argument("--sig_depth", type=int, default=6,
                        help="Log-signature depth")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Decoder hidden dimension")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    train(
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        n_channels=args.n_channels,
        H=args.H,
        sig_depth=args.sig_depth,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
