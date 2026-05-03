"""Rough-Bergomi log-volatility paths as a rough-vol benchmark.

This dataset is *not* scraped market data — public free intraday SPX data
is unreliable (yfinance is rate-limited, Stooq now requires a captcha-gated
API key, and FRED was unreachable from our test environment). Instead, we
simulate the rough Bergomi (rBergomi) model of Bayer, Friz & Gatheral
(2016, "Pricing under rough volatility", Quantitative Finance 16:887-904),
which is the published benchmark for replicating the rough vol structure
SPX exhibits at H ~ 0.05-0.15.

The dataset key remains `sp500_intraday` for plan/registry compatibility;
the docstring of the surrogate names the actual source so readers and
reviewers cannot be misled.

Model:
    log v_t = xi_0 + eta * W^H_t - 0.5 * eta^2 * t^{2H}
where W^H_t is fractional Brownian motion with Hurst H. We expose:
  * H — Hurst exponent (default 0.10)
  * eta — vol-of-vol scale (default 1.9, matches SPX calibration in the paper)
  * xi_0 — initial forward variance (default log(0.04^2), matching ~4% spot vol)
"""
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


def _simulate_rbergomi(
    n_paths: int,
    seq_len: int,
    *,
    H: float = 0.10,
    eta: float = 1.9,
    xi_0: float | None = None,
    T: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Return (n_paths, seq_len, 1) of log-vol paths under rBergomi.

    Args:
        n_paths: Number of independent paths.
        seq_len: Discretization grid size.
        H: Hurst exponent of the fBM driver.
        eta: Vol-of-vol parameter.
        xi_0: Log of initial forward variance. Default sets ~4% spot vol.
        T: Path horizon in years.
        seed: RNG seed for reproducibility.
    """
    if xi_0 is None:
        xi_0 = float(np.log(0.04 ** 2))

    fbm_paths = generate_rough_paths(
        n_paths=n_paths,
        seq_len=seq_len,
        n_channels=1,
        H=H,
        seed=seed,
        standardize=False,
    ).numpy()

    t_grid = np.linspace(0, T, seq_len, dtype=np.float32).reshape(1, seq_len, 1)
    log_v = xi_0 + eta * fbm_paths - 0.5 * (eta ** 2) * (t_grid ** (2.0 * H))
    return log_v.astype(np.float32)


@DATASETS.register("sp500_intraday")
class SP500Intraday(SignalDataset):
    """Rough-Bergomi log-volatility surrogate for SPX intraday vol.

    Calibrated to literature values; see module docstring.
    """

    def __init__(
        self,
        split: str,
        seq_len: int = 256,
        n_train: int = 4000,
        n_val: int = 500,
        n_test: int = 500,
        seed: int = 0,
    ):
        self._seed = seed
        spec = DatasetSpec(
            name="sp500_intraday",
            seq_len=seq_len,
            n_channels=1,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            cache_path=Path("data") / f"sp500_intraday_L{seq_len}.npy",
            hurst_estimated=0.10,
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
            arr = _simulate_rbergomi(n_total, self.spec.seq_len, seed=self._seed)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, arr)

        tr, va, te = _split_arr(arr, self.spec.n_train, self.spec.n_val, self.spec.n_test)
        tr, stats = standardize(tr)
        va, _ = standardize(va, stats)
        te, _ = standardize(te, stats)
        chosen = {"train": tr, "val": va, "test": te}[self.split]
        return torch.from_numpy(chosen).float()
