"""Smoke tests for fractalsig.runners.train_runner."""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fractalsig.runners.train_runner import TrainConfig, fit


class _LinearTrainer:
    """Minimal Trainer for smoke testing the runner."""

    def __init__(self):
        self.model = torch.nn.Linear(4, 1)
        self.loss_fn = torch.nn.MSELoss()

    def train_step(self, batch):
        x, y = batch
        return {"loss": self.loss_fn(self.model(x), y)}

    def eval_step(self, batch):
        x, y = batch
        return {"loss": self.loss_fn(self.model(x), y)}


@pytest.mark.smoke
def test_fit_runs_end_to_end(tmp_path):
    x = torch.randn(64, 4)
    y = (x.sum(dim=1, keepdim=True) > 0).float()
    ds = TensorDataset(x, y)
    train_loader = DataLoader(ds, batch_size=8)
    val_loader = DataLoader(ds, batch_size=8)
    cfg = TrainConfig(
        epochs=3,
        checkpoint_dir=tmp_path,
        run_name="t",
        wandb_project=None,
        device="cpu",
    )
    out = fit(_LinearTrainer(), train_loader, val_loader, cfg)
    assert out["best_val_loss"] < 10
    assert out["epochs_trained"] == 3
    assert (tmp_path / "t" / "best.pt").exists()
    assert (tmp_path / "t" / "summary.json").exists()


@pytest.mark.smoke
def test_fit_early_stops_when_loss_flat(tmp_path):
    """If the model never improves val, the runner stops after `patience` epochs."""
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)

    cfg = TrainConfig(
        epochs=100,
        patience=2,
        checkpoint_dir=tmp_path,
        run_name="es",
        wandb_project=None,
        device="cpu",
        lr=0.0,
    )
    out = fit(_LinearTrainer(), loader, loader, cfg)
    assert out["epochs_trained"] < 10
