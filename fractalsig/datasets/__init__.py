"""FractalSig datasets package.

Importing this package registers all built-in datasets in
fractalsig.registries.DATASETS so they can be looked up by name.
"""
from __future__ import annotations

from fractalsig.datasets import (
    eeg_chbmit,  # noqa: F401
    sp500_intraday,  # noqa: F401
    synthetic_fbm,  # noqa: F401
    turbulence_burgers,  # noqa: F401
)
