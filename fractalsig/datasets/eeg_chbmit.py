"""EEG windows from PhysioNet CHB-MIT (chb01_01.edf).

We extract one channel from the 23-channel pediatric recording and slice
it into non-overlapping windows of `seq_len` samples at the native 256 Hz.
EEG signals show fractal/self-similar behavior; the broadband Hurst
exponent typically falls in 0.6-0.8.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fractalsig.datasets.base import (
    DatasetSpec,
    SignalDataset,
    _split_arr,
    standardize,
)
from fractalsig.registries import DATASETS

RAW = Path("data/raw/chb01_01.edf")


def _load_edf_channel(path: Path, channel_idx: int = 0) -> np.ndarray:
    import pyedflib

    with pyedflib.EdfReader(str(path)) as f:
        x = f.readSignal(channel_idx).astype(np.float32)
    return x


def _slice_windows(x: np.ndarray, seq_len: int) -> np.ndarray:
    n = x.shape[0] // seq_len
    return x[: n * seq_len].reshape(n, seq_len, 1)


@DATASETS.register("eeg_chbmit")
class EEGCHBMIT(SignalDataset):
    """One-channel CHB-MIT EEG windows."""

    def __init__(
        self,
        split: str,
        seq_len: int = 512,
        n_train: int = 1400,
        n_val: int = 200,
        n_test: int = 200,
    ):
        spec = DatasetSpec(
            name="eeg_chbmit",
            seq_len=seq_len,
            n_channels=1,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            cache_path=Path("data") / f"eeg_chbmit_L{seq_len}.npy",
            hurst_estimated=0.7,
        )
        super().__init__(spec, split)

    def _load(self) -> torch.Tensor:
        cache = self.spec.cache_path
        if cache.exists():
            arr = np.load(cache)
        else:
            if not RAW.exists():
                raise FileNotFoundError(
                    f"Run scripts/download_eeg.py to fetch {RAW} (~42 MB) first."
                )
            raw = _load_edf_channel(RAW, channel_idx=0)
            arr = _slice_windows(raw, self.spec.seq_len)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, arr)

        n_total = self.spec.n_train + self.spec.n_val + self.spec.n_test
        if arr.shape[0] < n_total:
            raise RuntimeError(
                f"EEG only has {arr.shape[0]} windows; reduce splits or extend EDF source"
            )
        tr, va, te = _split_arr(arr, self.spec.n_train, self.spec.n_val, self.spec.n_test)
        tr, stats = standardize(tr)
        va, _ = standardize(va, stats)
        te, _ = standardize(te, stats)
        chosen = {"train": tr, "val": va, "test": te}[self.split]
        return torch.from_numpy(chosen).float()
