"""ESC-50 environmental sound clips, resampled to 8 kHz, sliced into windows."""
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

ROOT = Path("data/raw/esc50/ESC-50-master/audio")


def _build_cache(seq_len: int, n_files: int = 600) -> np.ndarray:
    import librosa

    windows: list[np.ndarray] = []
    wavs = sorted(ROOT.glob("*.wav"))[:n_files]
    if not wavs:
        raise FileNotFoundError(
            f"No WAV files at {ROOT}; run scripts/download_esc50.py first."
        )
    for wav in wavs:
        y, _ = librosa.load(wav, sr=8000, mono=True)
        n = y.shape[0] // seq_len
        if n == 0:
            continue
        windows.append(y[: n * seq_len].astype(np.float32).reshape(n, seq_len, 1))
    return np.concatenate(windows, axis=0)


@DATASETS.register("audio_esc50")
class AudioESC50(SignalDataset):
    """Sliced 8 kHz ESC-50 audio clips."""

    def __init__(
        self,
        split: str,
        seq_len: int = 1024,
        n_train: int = 1500,
        n_val: int = 200,
        n_test: int = 200,
    ):
        spec = DatasetSpec(
            name="audio_esc50",
            seq_len=seq_len,
            n_channels=1,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            cache_path=Path("data") / f"audio_esc50_L{seq_len}.npy",
        )
        super().__init__(spec, split)

    def _load(self) -> torch.Tensor:
        cache = self.spec.cache_path
        if not cache.exists():
            arr = _build_cache(self.spec.seq_len)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, arr)
        else:
            arr = np.load(cache)

        n_total = self.spec.n_train + self.spec.n_val + self.spec.n_test
        if arr.shape[0] < n_total:
            raise RuntimeError(
                f"audio_esc50: only {arr.shape[0]} windows; reduce splits or seq_len"
            )
        tr, va, te = _split_arr(arr, self.spec.n_train, self.spec.n_val, self.spec.n_test)
        tr, stats = standardize(tr)
        va, _ = standardize(va, stats)
        te, _ = standardize(te, stats)
        chosen = {"train": tr, "val": va, "test": te}[self.split]
        return torch.from_numpy(chosen).float()
