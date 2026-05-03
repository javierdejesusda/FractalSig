"""Tests for the Burgers turbulence dataset."""
from __future__ import annotations

import numpy as np
import pytest

from fractalsig.datasets.turbulence_burgers import TurbulenceBurgers, _simulate_burgers


@pytest.mark.smoke
def test_simulator_returns_correct_shape():
    arr = _simulate_burgers(n_paths=2, seq_len=64, n_steps=20, seed=0)
    assert arr.shape == (2, 64, 1)


@pytest.mark.smoke
def test_simulator_finite():
    arr = _simulate_burgers(n_paths=2, seq_len=64, n_steps=20, seed=0)
    assert np.isfinite(arr).all()


@pytest.mark.smoke
def test_simulator_deterministic():
    a = _simulate_burgers(n_paths=2, seq_len=32, n_steps=10, seed=3)
    b = _simulate_burgers(n_paths=2, seq_len=32, n_steps=10, seed=3)
    np.testing.assert_array_equal(a, b)


@pytest.mark.smoke
def test_dataset_split_sizes(tmp_path, monkeypatch):
    """A trimmed dataset (small n_steps via the cache) for fast tests."""
    monkeypatch.chdir(tmp_path)

    cache = tmp_path / "data" / "turbulence_burgers_L32.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, _simulate_burgers(n_paths=18, seq_len=32, n_steps=20, seed=0))

    tr = TurbulenceBurgers("train", seq_len=32, n_train=10, n_val=4, n_test=4)
    assert tr[0].shape == (32, 1)
    assert len(tr) == 10


@pytest.mark.smoke
def test_dataset_registered():
    from fractalsig.registries import DATASETS
    assert DATASETS.get_or_raise("turbulence_burgers") is TurbulenceBurgers
