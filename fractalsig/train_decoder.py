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

    This dataset generates fBM paths and pre-computes:
    1. Log-Signatures: Compact geometric summaries (input to decoder)
    2. Wavelet Coefficients: Multi-scale decomposition (target for decoder)

    Both inputs and targets are normalized to zero mean/unit variance for
    stable training. The normalization statistics are saved for inference.

    Attributes:
        paths: Generated fBM paths (n_samples, seq_len, n_channels).
        log_signatures: Pre-computed log-signatures (n_samples, logsig_dim).
        wavelet_coeffs: Flattened wavelet coefficients (n_samples, coeff_dim).
        logsig_stats: Dict with 'mean' and 'std' for log-signatures.
        coeff_stats: Dict with 'mean' and 'std' for wavelet coefficients.
    """

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        n_channels: int = 1,
        H: float = 0.1,
        sig_depth: int = 4,
        wavelet: str = "db4",
        level: int | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            n_samples: Number of paths to generate.
            seq_len: Length of each path.
            n_channels: Number of channels per path.
            H: Hurst parameter for fBM generation.
            sig_depth: Depth of log-signature computation.
            wavelet: Wavelet family for decomposition.
            level: Wavelet decomposition level (auto if None).
            device: Device for signature computation.
            seed: Random seed for reproducibility.
        """
        super().__init__()

        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.sig_depth = sig_depth
        self.wavelet = wavelet
        self.device = device

        # Determine wavelet level
        if level is None:
            wavelet_obj = pywt.Wavelet(wavelet)
            max_level = pywt.dwt_max_level(seq_len, wavelet_obj.dec_len)
            self.level = max(1, min(max_level, 6))
        else:
            self.level = level

        log.info(f"Generating {n_samples} rough paths (H={H}, seq_len={seq_len})...")
        self.paths = generate_rough_paths(
            n_paths=n_samples,
            seq_len=seq_len,
            n_channels=n_channels,
            H=H,
            seed=seed,
            standardize=True,
        )

        log.info("Pre-computing log-signatures...")
        self.log_signatures = self._compute_log_signatures()

        log.info("Pre-computing wavelet coefficients...")
        self.wavelet_coeffs = self._compute_wavelet_coeffs()

        # Normalize inputs and targets
        log.info("Normalizing data...")
        self.log_signatures, self.logsig_stats = self._normalize(self.log_signatures)
        self.wavelet_coeffs, self.coeff_stats = self._normalize(self.wavelet_coeffs)

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
        std = torch.clamp(std, min=1e-8)  # Avoid division by zero

        normalized = (tensor - mean) / std

        stats = {"mean": mean.squeeze(0), "std": std.squeeze(0)}
        return normalized, stats

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
    checkpoint_dir: str = "checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int | None = 42,
) -> dict[str, float]:
    """Train the FractalDecoder on ground-truth fBM data.

    Args:
        n_samples: Number of training samples.
        seq_len: Sequence length.
        n_channels: Number of channels.
        H: Hurst parameter.
        sig_depth: Log-signature depth.
        hidden_dim: Decoder hidden dimension.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        lr: Learning rate.
        checkpoint_dir: Directory for saving checkpoints.
        device: Training device.
        seed: Random seed.

    Returns:
        Dict with training metrics (final loss, best loss).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Create dataset
    dataset = RoughPathDataset(
        n_samples=n_samples,
        seq_len=seq_len,
        n_channels=n_channels,
        H=H,
        sig_depth=sig_depth,
        device=device,
        seed=seed,
    )

    # Save normalization stats
    stats_path = checkpoint_path / "normalization_stats.json"
    dataset.save_stats(str(stats_path))

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )

    # Initialize model
    input_dim = dataset.log_signatures.shape[1]
    model = FractalDecoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_seq_len=seq_len,
        out_channels=n_channels,
        wavelet=dataset.wavelet,
        level=dataset.level,
    ).to(device)

    log.info(f"\nModel: {model.get_num_params():,} parameters")
    log.info(f"Input dim: {input_dim}, Output coeff dim: {model.total_coeff_dim}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # OneCycleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.3,
    )

    # Training loop
    best_loss = float("inf")
    history: dict[str, list[float]] = {"train_loss": []}

    log.info(f"Starting training for {epochs} epochs...")

    pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for logsig_batch, coeff_batch in dataloader:
            logsig_batch = logsig_batch.to(device)
            coeff_batch = coeff_batch.to(device)

            optimizer.zero_grad()

            # Forward: predict wavelet coefficients directly
            # We need to get the MLP output before waverec
            pred_coeffs = model.mlp(logsig_batch)

            # Loss on coefficient space
            loss = criterion(pred_coeffs, coeff_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)

        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            "Loss": f"{avg_loss:.5f}",
            "LR": f"{current_lr:.2e}",
            "Best": f"{best_loss:.5f}"
        })

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": {
                        "input_dim": input_dim,
                        "hidden_dim": hidden_dim,
                        "seq_len": seq_len,
                        "n_channels": n_channels,
                        "wavelet": dataset.wavelet,
                        "level": dataset.level,
                    },
                },
                checkpoint_path / "fractal_decoder_best.pth",
            )
            # Only log detailed info occasionally to avoid spam
            if (epoch + 1) % 50 == 0:
                 log.debug(f"New best model saved at epoch {epoch+1} (Loss: {best_loss:.6f})")

    log.info(f"Training complete. Best loss: {best_loss:.6f}")
    log.info(f"Model saved to: {checkpoint_path / 'fractal_decoder_best.pth'}")

    # Clear GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "final_loss": avg_loss,
        "best_loss": best_loss,
        "epochs_trained": epochs,
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
