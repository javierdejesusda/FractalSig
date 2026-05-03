"""Smoke tests for fractalsig.train_decoder.

Covers train/val split, val-loss-based checkpointing, and early stopping.
"""
from __future__ import annotations

import json

import pytest
import torch

from fractalsig.train_decoder import RoughPathDataset, train


@pytest.mark.smoke
def test_dataset_accepts_prebuilt_paths():
    """RoughPathDataset can be constructed from a tensor of paths directly."""
    paths = torch.randn(8, 32, 1)
    ds = RoughPathDataset(paths=paths, sig_depth=2, level=2)
    assert len(ds) == 8
    assert ds.seq_len == 32
    assert ds.n_channels == 1
    logsig, coeffs = ds[0]
    assert logsig.ndim == 1
    assert coeffs.ndim == 1


@pytest.mark.smoke
def test_dataset_shares_train_stats():
    """Val dataset normalized with train stats has the same shape outputs."""
    train_paths = torch.randn(12, 32, 1)
    val_paths = torch.randn(4, 32, 1)
    train_ds = RoughPathDataset(paths=train_paths, sig_depth=2, level=2)
    val_ds = RoughPathDataset(
        paths=val_paths,
        sig_depth=2,
        level=2,
        train_stats={
            "logsig": train_ds.logsig_stats,
            "coeff": train_ds.coeff_stats,
        },
    )
    assert val_ds.logsig_stats is train_ds.logsig_stats
    assert val_ds.coeff_stats is train_ds.coeff_stats
    assert val_ds.log_signatures.shape[1] == train_ds.log_signatures.shape[1]
    assert val_ds.wavelet_coeffs.shape[1] == train_ds.wavelet_coeffs.shape[1]


@pytest.mark.smoke
def test_train_three_epochs_decreases_val_loss(tmp_path):
    """A 3-epoch run on a tiny fBM dataset should reduce val loss."""
    out = train(
        n_samples=32,
        seq_len=32,
        n_channels=1,
        H=0.1,
        sig_depth=2,
        hidden_dim=16,
        batch_size=8,
        epochs=3,
        lr=1e-3,
        patience=10,
        val_frac=0.25,
        checkpoint_dir=str(tmp_path),
        device="cpu",
        seed=42,
    )
    assert "val_loss" in out["history"]
    assert len(out["history"]["val_loss"]) >= 1
    assert out["history"]["val_loss"][-1] < out["history"]["val_loss"][0]
    assert out["best_val_loss"] == min(out["history"]["val_loss"])
    assert (tmp_path / "fractal_decoder_best.pth").exists()
    stats_path = tmp_path / "normalization_stats.json"
    assert stats_path.exists()
    with stats_path.open() as f:
        stats = json.load(f)
    assert "logsig_dim" in stats and stats["logsig_dim"] > 0


@pytest.mark.smoke
def test_train_early_stops_when_val_flat(tmp_path):
    """With lr=0 the val loss never improves; training should stop early."""
    out = train(
        n_samples=32,
        seq_len=32,
        n_channels=1,
        H=0.1,
        sig_depth=2,
        hidden_dim=16,
        batch_size=8,
        epochs=50,
        lr=0.0,
        patience=2,
        val_frac=0.25,
        checkpoint_dir=str(tmp_path),
        device="cpu",
        seed=0,
    )
    assert out["epochs_trained"] < 10
