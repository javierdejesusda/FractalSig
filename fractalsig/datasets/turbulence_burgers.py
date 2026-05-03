"""Stochastic Burgers' equation: 1D viscous turbulence with multifractal velocity.

We integrate
    u_t + u u_x = nu u_xx + sqrt(2 nu) eta(x, t)
on a periodic 1D grid with random initial conditions and additive
space-time white noise. Velocity differences exhibit Hurst H ~ 1/3
(Kolmogorov scaling for 1D Burgers turbulence).

Cheap to simulate (CPU-bound), self-contained (no external data),
and a recognized benchmark for rough-signal generation.
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


def _simulate_burgers(
    n_paths: int,
    seq_len: int,
    *,
    nu: float = 0.01,
    n_steps: int = 2000,
    dt: float = 5e-4,
    seed: int = 0,
) -> np.ndarray:
    """Return (n_paths, seq_len, 1) of Burgers' velocity snapshots.

    Args:
        n_paths: Number of independent realizations.
        seq_len: Spatial grid size; the snapshot at the final time becomes one path.
        nu: Kinematic viscosity.
        n_steps: Number of forward-Euler steps.
        dt: Time step.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    L = 2 * np.pi
    dx = L / seq_len
    out = np.zeros((n_paths, seq_len, 1), dtype=np.float32)
    for i in range(n_paths):
        u = 0.1 * rng.standard_normal(seq_len).astype(np.float32)
        for _ in range(n_steps):
            u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
            u_xx = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx * dx)
            noise = rng.standard_normal(seq_len).astype(np.float32) * np.sqrt(2 * nu * dt)
            u = u + dt * (-u * u_x + nu * u_xx) + noise
        out[i, :, 0] = u
    return out


@DATASETS.register("turbulence_burgers")
class TurbulenceBurgers(SignalDataset):
    """Stochastic Burgers turbulence; multifractal velocities with H ~ 1/3."""

    def __init__(
        self,
        split: str,
        seq_len: int = 256,
        n_train: int = 1000,
        n_val: int = 200,
        n_test: int = 200,
        seed: int = 0,
    ):
        self._seed = seed
        spec = DatasetSpec(
            name="turbulence_burgers",
            seq_len=seq_len,
            n_channels=1,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            cache_path=Path("data") / f"turbulence_burgers_L{seq_len}.npy",
            hurst_estimated=1.0 / 3.0,
        )
        super().__init__(spec, split)

    def _load(self) -> torch.Tensor:
        cache = self.spec.cache_path
        n_total = self.spec.n_train + self.spec.n_val + self.spec.n_test
        if cache.exists():
            arr = np.load(cache)
            if arr.shape[0] < n_total:
                raise RuntimeError(
                    f"cache {cache} has {arr.shape[0]} paths, need {n_total}; delete to rebuild"
                )
        else:
            arr = _simulate_burgers(n_total, self.spec.seq_len, seed=self._seed)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, arr)

        tr, va, te = _split_arr(arr, self.spec.n_train, self.spec.n_val, self.spec.n_test)
        tr, stats = standardize(tr)
        va, _ = standardize(va, stats)
        te, _ = standardize(te, stats)
        chosen = {"train": tr, "val": va, "test": te}[self.split]
        return torch.from_numpy(chosen).float()
