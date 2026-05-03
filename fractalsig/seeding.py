"""Centralized determinism setup for reproducible experiments."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, *, cudnn_deterministic: bool = True) -> None:
    """Seed every RNG and configure cuDNN for reproducibility.

    Args:
        seed: Integer seed applied to python, numpy, torch (cpu and cuda).
        cudnn_deterministic: If True, force cuDNN deterministic algorithms
            (slower, but bit-exact across runs on the same GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
