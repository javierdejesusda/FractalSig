"""Smoke tests for fractalsig.data_gen."""
from __future__ import annotations

import pytest
import torch

from fractalsig.data_gen import generate_rough_paths


@pytest.mark.smoke
def test_generate_returns_correct_shape():
    paths = generate_rough_paths(n_paths=4, seq_len=64, n_channels=1, H=0.1, seed=0)
    assert paths.shape == (4, 64, 1)
    assert paths.dtype == torch.float32


@pytest.mark.smoke
def test_generate_is_deterministic_under_seed():
    a = generate_rough_paths(n_paths=2, seq_len=32, n_channels=1, H=0.1, seed=42)
    b = generate_rough_paths(n_paths=2, seq_len=32, n_channels=1, H=0.1, seed=42)
    assert torch.allclose(a, b)


@pytest.mark.smoke
def test_generate_standardized_by_default():
    p = generate_rough_paths(n_paths=512, seq_len=128, n_channels=1, H=0.1, seed=0)
    assert abs(p.mean().item()) < 0.1
    assert abs(p.std().item() - 1.0) < 0.1


@pytest.mark.smoke
def test_generate_rejects_invalid_hurst():
    with pytest.raises(ValueError):
        generate_rough_paths(n_paths=1, seq_len=8, n_channels=1, H=1.5)


@pytest.mark.smoke
def test_generate_supports_smooth_hurst():
    """H > 0.5 (smooth fBM) must work too — needed for Hurst-sweep ablation."""
    p = generate_rough_paths(n_paths=4, seq_len=64, n_channels=1, H=0.7, seed=0)
    assert p.shape == (4, 64, 1)
    assert torch.isfinite(p).all()
