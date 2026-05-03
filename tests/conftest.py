"""Shared pytest fixtures for the FractalSig test suite."""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed torch and numpy for every test."""
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


@pytest.fixture
def tiny_fbm_dataset():
    """Returns 16 fBM paths of length 32, channel 1."""
    from fractalsig.data_gen import generate_rough_paths
    return generate_rough_paths(n_paths=16, seq_len=32, n_channels=1, H=0.1, seed=0)


@pytest.fixture
def small_decoder():
    """Tiny FractalDecoder for fast tests."""
    from fractalsig.decoder import FractalDecoder
    return FractalDecoder(input_dim=8, hidden_dim=16, output_seq_len=32, out_channels=1, level=2)
