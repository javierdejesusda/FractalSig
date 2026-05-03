"""Tests for the CHB-MIT EEG dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fractalsig.datasets.eeg_chbmit import EEGCHBMIT, _slice_windows


@pytest.mark.smoke
def test_slice_windows_drops_remainder():
    x = np.arange(100, dtype=np.float32)
    out = _slice_windows(x, 30)
    assert out.shape == (3, 30, 1)
    assert out[0, 0, 0] == 0.0
    assert out[2, -1, 0] == 89.0


@pytest.mark.smoke
def test_dataset_registered():
    from fractalsig.registries import DATASETS
    assert DATASETS.get_or_raise("eeg_chbmit") is EEGCHBMIT


@pytest.mark.smoke
def test_dataset_split_sizes(tmp_path, monkeypatch):
    """Stub the cache so we don't need the real EDF for this fast smoke test."""
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "data" / "eeg_chbmit_L64.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, np.random.RandomState(0).randn(50, 64, 1).astype(np.float32))
    tr = EEGCHBMIT("train", seq_len=64, n_train=10, n_val=4, n_test=4)
    assert tr[0].shape == (64, 1)
    assert len(tr) == 10


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("data/raw/chb01_01.edf").exists(),
    reason="raw EDF not downloaded; run scripts/download_eeg.py",
)
def test_loads_from_real_edf():
    """Only runs when the EDF has been fetched; verifies end-to-end ingest."""
    d = EEGCHBMIT("train", seq_len=512, n_train=100, n_val=10, n_test=10)
    assert d[0].shape == (512, 1)
    assert d.data.dtype.is_floating_point
