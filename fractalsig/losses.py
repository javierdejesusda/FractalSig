"""Loss functions for FractalSig."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class ScaleWeightedMSE(nn.Module):
    """MSE on flat wavelet coefficients with per-scale energy weighting.

    `pywt.wavedec` returns coefficient arrays in the order
    [cA_n, cD_n, cD_{n-1}, ..., cD_1]. cA_n is the lowest-frequency
    approximation; cD_1 holds the finest detail (highest frequency). Each
    segment is weighted by ``2**(beta * level_idx)``, where ``level_idx``
    is the array's index in the pywt list — so higher-frequency detail
    bands receive exponentially larger weight than the smooth approximation.
    The full weight vector is rescaled to ``mean(weights) == 1``, leaving
    the overall magnitude comparable to a plain MSE.

    For multi-channel signals the per-channel pattern is repeated
    ``n_channels`` times, matching the layout produced by
    ``RoughPathDataset`` (which concatenates per-channel coefficient
    vectors per sample).

    Attributes:
        weights: 1-D buffer of length ``sum(coeff_lengths) * n_channels``.
    """

    weights: torch.Tensor

    def __init__(
        self,
        coeff_lengths: Sequence[int],
        beta: float = 1.0,
        n_channels: int = 1,
    ) -> None:
        """Initialize the weighted-MSE loss.

        Args:
            coeff_lengths: Per-channel segment lengths in pywt order
                ``[cA_n, cD_n, ..., cD_1]``.
            beta: Exponent controlling high-frequency emphasis. ``beta=0``
                reduces to plain MSE; larger ``beta`` concentrates loss
                on detail bands.
            n_channels: Number of channels concatenated per target sample.

        Raises:
            ValueError: If ``beta`` is negative or any segment length is
                non-positive.
        """
        super().__init__()
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {n_channels}")
        if any(length <= 0 for length in coeff_lengths):
            raise ValueError(f"all segment lengths must be positive: {list(coeff_lengths)}")

        per_segment = [
            torch.full((int(length),), 2.0 ** (beta * level_idx))
            for level_idx, length in enumerate(coeff_lengths)
        ]
        per_channel = torch.cat(per_segment)
        full = per_channel.repeat(n_channels)
        self.register_buffer("weights", full / full.mean())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE between flat coefficient vectors.

        Args:
            pred: Predicted coefficients of shape (batch, total_dim).
            target: Target coefficients of shape (batch, total_dim).

        Returns:
            Scalar weighted-MSE loss.
        """
        return (self.weights * (pred - target) ** 2).mean()
