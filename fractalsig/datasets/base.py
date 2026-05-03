"""Abstract base class for all signal datasets in FractalSig."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata about a dataset."""

    name: str
    seq_len: int
    n_channels: int
    n_train: int
    n_val: int
    n_test: int
    cache_path: Path
    hurst_estimated: float | None = None


class SignalDataset(Dataset, ABC):
    """Common base for every dataset the sweep can use.

    Subclasses must implement `_load(self) -> torch.Tensor` returning the
    tensor for the requested split. The base class holds the spec and split
    name; sequencing, shuffling, and standardization helpers live in this
    module so subclasses do not duplicate them.
    """

    spec: DatasetSpec
    data: torch.Tensor

    def __init__(self, spec: DatasetSpec, split: str):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be one of train/val/test, got {split!r}")
        self.spec = spec
        self.split = split
        self.data = self._load()

    @abstractmethod
    def _load(self) -> torch.Tensor:
        """Load (and cache) the dataset, return the tensor for self.split."""

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _split_arr(
    x: np.ndarray, n_train: int, n_val: int, n_test: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sequential split — never reorder time series across split boundaries."""
    needed = n_train + n_val + n_test
    if x.shape[0] < needed:
        raise ValueError(f"have {x.shape[0]} samples, need {needed}")
    tr = x[:n_train]
    va = x[n_train : n_train + n_val]
    te = x[n_train + n_val : n_train + n_val + n_test]
    return tr, va, te


def standardize(
    x: np.ndarray, stats: dict | None = None
) -> tuple[np.ndarray, dict]:
    """Per-channel z-score; if `stats` provided, reuse them (test-set hygiene).

    Returns (standardized_array, stats_dict). Pass the returned stats to
    standardize() on the val/test splits to avoid leaking validation info
    into training.
    """
    if stats is None:
        mu = x.mean(axis=(0, 1), keepdims=True)
        sd = x.std(axis=(0, 1), keepdims=True).clip(min=1e-8)
        stats = {"mean": mu, "std": sd}
    else:
        mu, sd = stats["mean"], stats["std"]
    return (x - mu) / sd, stats
