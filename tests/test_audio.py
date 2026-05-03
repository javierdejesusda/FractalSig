"""Tests for the ESC-50 audio dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fractalsig.datasets.audio_esc50 import AudioESC50


@pytest.mark.smoke
def test_dataset_registered():
    from fractalsig.registries import DATASETS
    assert DATASETS.get_or_raise("audio_esc50") is AudioESC50


@pytest.mark.smoke
def test_dataset_split_sizes_with_stub_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "data" / "audio_esc50_L64.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, np.random.RandomState(0).randn(50, 64, 1).astype(np.float32))
    tr = AudioESC50("train", seq_len=64, n_train=10, n_val=4, n_test=4)
    assert tr[0].shape == (64, 1)
    assert len(tr) == 10


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("data/raw/esc50/ESC-50-master/audio").exists(),
    reason="ESC-50 not downloaded; run scripts/download_esc50.py",
)
def test_loads_from_real_audio():
    d = AudioESC50("train", seq_len=1024, n_train=200, n_val=20, n_test=20)
    assert d[0].shape == (1024, 1)
