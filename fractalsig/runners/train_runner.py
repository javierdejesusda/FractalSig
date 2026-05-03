"""Unified training loop usable by every baseline and our method.

Each baseline implements a small `Trainer` protocol exposing:
  * model: nn.Module
  * train_step(batch) -> dict[str, float]    (must contain key "loss" — a scalar Tensor)
  * eval_step(batch)  -> dict[str, float]    (same)

The runner owns the optimizer, scheduler, gradient clipping, val loop, early
stopping, checkpointing, and optional wandb logging.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)


class Trainer(Protocol):
    model: torch.nn.Module

    def train_step(self, batch: Any) -> dict[str, Any]: ...
    def eval_step(self, batch: Any) -> dict[str, Any]: ...


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 20
    val_every: int = 1
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    run_name: str = "run"
    wandb_project: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def fit(
    trainer: Trainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Train `trainer` with cosine schedule, val-based early stopping, wandb logging.

    Args:
        trainer: An instance implementing the `Trainer` protocol.
        train_loader: PyTorch dataloader over the training split.
        val_loader: PyTorch dataloader over the validation split.
        cfg: TrainConfig.

    Returns:
        Dict with `best_val_loss`, `best_epoch`, and `epochs_trained`.
    """
    device = torch.device(cfg.device)
    trainer.model.to(device)

    opt = torch.optim.AdamW(
        trainer.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    run = None
    if cfg.wandb_project is not None:
        import wandb
        run = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))

    ckpt_dir = Path(cfg.checkpoint_dir) / cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    epoch = 0

    for epoch in tqdm(range(cfg.epochs), desc=f"train[{cfg.run_name}]"):
        trainer.model.train()
        train_losses: dict[str, float] = {}
        for batch in train_loader:
            opt.zero_grad()
            metrics = trainer.train_step(batch)
            loss = metrics["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), cfg.grad_clip)
            opt.step()
            for k, v in metrics.items():
                train_losses[k] = train_losses.get(k, 0.0) + (
                    v.item() if torch.is_tensor(v) else float(v)
                )
        for k in train_losses:
            train_losses[k] /= max(1, len(train_loader))
        sched.step()

        if (epoch + 1) % cfg.val_every == 0:
            trainer.model.train(False)
            val_losses: dict[str, float] = {}
            with torch.no_grad():
                for batch in val_loader:
                    m = trainer.eval_step(batch)
                    for k, v in m.items():
                        val_losses[k] = val_losses.get(k, 0.0) + (
                            v.item() if torch.is_tensor(v) else float(v)
                        )
            for k in val_losses:
                val_losses[k] /= max(1, len(val_loader))

            vl = val_losses.get("loss", float("inf"))
            if vl < best_val:
                best_val = vl
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(trainer.model.state_dict(), ckpt_dir / "best.pt")
            else:
                epochs_no_improve += 1

            if run is not None:
                run.log(
                    {
                        **{f"train/{k}": v for k, v in train_losses.items()},
                        **{f"val/{k}": v for k, v in val_losses.items()},
                        "epoch": epoch,
                        "lr": sched.get_last_lr()[0],
                    }
                )

            if epochs_no_improve >= cfg.patience:
                log.info(
                    "early stop at epoch %d (no val improvement for %d)",
                    epoch,
                    cfg.patience,
                )
                break

    summary = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_trained": epoch + 1,
    }
    with open(ckpt_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if run is not None:
        run.finish()

    return summary
