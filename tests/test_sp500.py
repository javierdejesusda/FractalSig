"""Tests for the SP500/rough-Bergomi vol surrogate."""
from __future__ import annotations

import pytest

from fractalsig.datasets.sp500_intraday import SP500Intraday, _simulate_rbergomi


@pytest.mark.smoke
def test_simulator_returns_correct_shape():
    arr = _simulate_rbergomi(n_paths=4, seq_len=64, H=0.1, seed=0)
    assert arr.shape == (4, 64, 1)


@pytest.mark.smoke
def test_simulator_deterministic_under_seed():
    a = _simulate_rbergomi(n_paths=2, seq_len=32, seed=7)
    b = _simulate_rbergomi(n_paths=2, seq_len=32, seed=7)
    import numpy as np
    np.testing.assert_array_equal(a, b)


@pytest.mark.smoke
def test_dataset_split_sizes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = SP500Intraday("train", seq_len=64, n_train=20, n_val=4, n_test=4)
    assert tr[0].shape == (64, 1)
    assert len(tr) == 20


@pytest.mark.smoke
def test_dataset_registered():
    from fractalsig.registries import DATASETS
    assert DATASETS.get_or_raise("sp500_intraday") is SP500Intraday
