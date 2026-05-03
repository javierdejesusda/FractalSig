"""Tests for fractalsig.datasets.synthetic_fbm.SyntheticFBM."""
from __future__ import annotations

import pytest

from fractalsig.datasets.synthetic_fbm import SyntheticFBM


@pytest.mark.smoke
def test_split_sizes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = SyntheticFBM("train", seq_len=64, n_train=10, n_val=4, n_test=4)
    va = SyntheticFBM("val", seq_len=64, n_train=10, n_val=4, n_test=4)
    te = SyntheticFBM("test", seq_len=64, n_train=10, n_val=4, n_test=4)
    assert len(tr) == 10
    assert len(va) == 4
    assert len(te) == 4
    assert tr[0].shape == (64, 1)


@pytest.mark.smoke
def test_standardization_uses_train_stats(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr = SyntheticFBM("train", seq_len=64, n_train=200, n_val=20, n_test=20)
    assert abs(tr.data.mean().item()) < 0.1
    assert abs(tr.data.std().item() - 1.0) < 0.1


@pytest.mark.smoke
def test_invalid_split_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="split must be one of"):
        SyntheticFBM("dev", seq_len=64, n_train=10, n_val=4, n_test=4)


@pytest.mark.smoke
def test_registered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from fractalsig.registries import DATASETS
    cls = DATASETS.get_or_raise("synthetic_fbm")
    assert cls is SyntheticFBM


@pytest.mark.smoke
def test_cache_reused_on_second_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = SyntheticFBM("train", seq_len=64, n_train=10, n_val=4, n_test=4, seed=7)
    b = SyntheticFBM("train", seq_len=64, n_train=10, n_val=4, n_test=4, seed=999)
    assert a.data.shape == b.data.shape
    assert (a.spec.cache_path).exists()
