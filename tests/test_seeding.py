"""Tests for fractalsig.seeding."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from fractalsig.seeding import set_global_seed


@pytest.mark.smoke
def test_same_seed_same_torch_tensor():
    set_global_seed(123)
    a = torch.randn(5)
    set_global_seed(123)
    b = torch.randn(5)
    assert torch.allclose(a, b)


@pytest.mark.smoke
def test_same_seed_same_numpy_array():
    set_global_seed(7)
    a = np.random.randn(5)
    set_global_seed(7)
    b = np.random.randn(5)
    np.testing.assert_array_equal(a, b)


@pytest.mark.smoke
def test_different_seeds_different_output():
    set_global_seed(0)
    a = torch.randn(5)
    set_global_seed(1)
    b = torch.randn(5)
    assert not torch.allclose(a, b)


@pytest.mark.smoke
def test_pythonhashseed_env_set():
    set_global_seed(42)
    assert os.environ["PYTHONHASHSEED"] == "42"
