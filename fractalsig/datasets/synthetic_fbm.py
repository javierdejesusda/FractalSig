"""Synthetic fractional-Brownian-motion dataset with configurable Hurst exponent."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fractalsig.data_gen import generate_rough_paths
from fractalsig.datasets.base import (
    DatasetSpec,
    SignalDataset,
    _split_arr,
    standardize,
)
from fractalsig.registries import DATASETS


@DATASETS.register("synthetic_fbm")
class SyntheticFBM(SignalDataset):
    """Synthetic fBM paths with configurable Hurst exponent.

    Used both as the canonical training dataset (H=0.1) and as the substrate
    for the Hurst-sweep ablation (H in {0.05, 0.1, 0.2, 0.3, 0.5, 0.7}).
    """

    def __init__(
        self,
        split: str,
        seq_len: int = 256,
        hurst: float = 0.1,
        n_train: int = 4000,
        n_val: int = 500,
        n_test: int = 500,
        seed: int = 0,
    ):
        self._hurst = hurst
        self._seed = seed
        spec = DatasetSpec(
            name=f"fbm_H{hurst:.2f}",
            seq_len=seq_len,
            n_channels=1,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            cache_path=Path("data") / f"synthetic_fbm_H{hurst:.2f}_L{seq_len}.npy",
            hurst_estimated=hurst,
        )
        super().__init__(spec, split)

    def _load(self) -> torch.Tensor:
        n_total = self.spec.n_train + self.spec.n_val + self.spec.n_test
        cache = self.spec.cache_path
        if cache.exists():
            arr = np.load(cache)
            if arr.shape[0] < n_total:
                raise RuntimeError(
                    f"cache {cache} has {arr.shape[0]} paths, need {n_total}; delete to rebuild"
                )
        else:
            cache.parent.mkdir(parents=True, exist_ok=True)
            paths = generate_rough_paths(
                n_paths=n_total,
                seq_len=self.spec.seq_len,
                n_channels=1,
                H=self._hurst,
                seed=self._seed,
                standardize=False,
            )
            arr = paths.numpy().astype(np.float32)
            np.save(cache, arr)

        tr, va, te = _split_arr(arr, self.spec.n_train, self.spec.n_val, self.spec.n_test)
        tr, stats = standardize(tr)
        va, _ = standardize(va, stats)
        te, _ = standardize(te, stats)
        chosen = {"train": tr, "val": va, "test": te}[self.split]
        return torch.from_numpy(chosen).float()
