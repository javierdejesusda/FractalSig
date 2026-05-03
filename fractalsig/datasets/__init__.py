"""FractalSig datasets package.

Importing this package registers all built-in datasets in
fractalsig.registries.DATASETS so they can be looked up by name.
"""
from __future__ import annotations

from fractalsig.datasets import synthetic_fbm  # noqa: F401  (registers via decorator)
