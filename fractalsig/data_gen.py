"""Robust data generation module for Fractional Brownian Motion.

This module provides utilities for generating rough paths with Hurst parameter
H ≈ 0.1, specifically designed for neural network training.
"""

import warnings
from typing import Optional

import numpy as np
import torch
from fbm import FBM


def generate_single_fbm(
    seq_len: int,
    H: float,
    method: str = "daviesharte"
) -> np.ndarray:
    """Generate a single Fractional Brownian Motion path.

    Args:
        seq_len: Length of the sequence to generate.
        H: Hurst parameter (0 < H < 1). Values near 0.1 represent rough paths.
        method: Generation method ('daviesharte' or 'cholesky').

    Returns:
        A numpy array of shape (seq_len,) containing the fBm path.

    Raises:
        ValueError: If generation fails with both methods.
    """
    fbm_generator = FBM(n=seq_len - 1, hurst=H, method=method)
    return fbm_generator.fbm()


def generate_rough_paths(
    n_paths: int,
    seq_len: int,
    n_channels: int,
    H: float = 0.1,
    seed: Optional[int] = None,
    standardize: bool = True
) -> torch.Tensor:
    """Generate robust Fractional Brownian Motion paths with rough volatility.

    This function generates fBm paths suitable for rough volatility modeling.
    For low Hurst parameters (H ≈ 0.1), the standard Davies-Harte method often
    fails due to negative eigenvalues in the circulant matrix. This function
    implements an automatic fallback to the Cholesky method when necessary.

    Args:
        n_paths: Number of paths (batch size) to generate.
        seq_len: Length of each sequence.
        n_channels: Number of independent channels per path.
        H: Hurst parameter (0 < H < 1). Default is 0.1 for rough volatility.
            H < 0.5: Anti-persistent (rough) paths.
            H = 0.5: Standard Brownian motion.
            H > 0.5: Persistent (smooth) paths.
        seed: Optional random seed for reproducibility.
        standardize: If True, standardize paths to zero mean and unit variance
            per channel. Essential for neural network stability.

    Returns:
        A torch.FloatTensor of shape (n_paths, seq_len, n_channels) containing
        the generated fBm paths.

    Raises:
        ValueError: If H is not in the valid range (0, 1).

    Example:
        >>> paths = generate_rough_paths(n_paths=100, seq_len=256, n_channels=1)
        >>> paths.shape
        torch.Size([100, 256, 1])
    """
    if not 0 < H < 1:
        raise ValueError(f"Hurst parameter H must be in (0, 1), got {H}")

    if seed is not None:
        np.random.seed(seed)

    paths = np.zeros((n_paths, seq_len, n_channels), dtype=np.float32)

    # Track method used for logging
    used_cholesky = False

    for i in range(n_paths):
        for c in range(n_channels):
            # Attempt Davies-Harte first (O(n log n)), fallback to Cholesky (O(n^3))
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    path = generate_single_fbm(seq_len, H, method="daviesharte")
            except (ValueError, UserWarning, RuntimeWarning):
                # Davies-Harte fails for H ≈ 0.1 due to negative eigenvalues
                # Fallback to exact Cholesky decomposition
                if not used_cholesky:
                    used_cholesky = True
                path = generate_single_fbm(seq_len, H, method="cholesky")

            paths[i, :, c] = path

    if used_cholesky:
        print(f"Note: Using Cholesky method for H={H} "
              "(Davies-Harte failed due to negative eigenvalues)")

    # Standardize per channel for neural network stability
    if standardize:
        for c in range(n_channels):
            channel_data = paths[:, :, c]
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            if std > 1e-8:  # Avoid division by zero
                paths[:, :, c] = (channel_data - mean) / std

    return torch.from_numpy(paths).float()


if __name__ == "__main__":
    paths = generate_rough_paths(n_paths=5, seq_len=256, n_channels=2, H=0.1, seed=42)
    print(f"Generated paths shape: {paths.shape}")
    print(f"Mean: {paths.mean().item():.4f}, Std: {paths.std().item():.4f}")