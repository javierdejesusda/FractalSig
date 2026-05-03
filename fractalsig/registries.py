"""String → class registries for datasets, baselines, and metrics.

This avoids dynamic imports scattered through the codebase. Adding a new
dataset or baseline is a one-line decorator on its module:

    from fractalsig.registries import DATASETS

    @DATASETS.register("my_dataset")
    class MyDataset(SignalDataset):
        ...

The sweep runner then resolves the string to the class:

    cls = DATASETS.get_or_raise("my_dataset")
    instance = cls("train")
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=type)


class _Registry(dict[str, type]):
    """Decorator-based string→class registry."""

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator: register a class under `name`.

        Raises:
            KeyError: If `name` is already registered.
        """
        def deco(cls: T) -> T:
            if name in self:
                raise KeyError(f"{name!r} already registered as {self[name]!r}")
            self[name] = cls
            return cls
        return deco

    def get_or_raise(self, name: str) -> type:
        """Look up `name` or raise with the available alternatives.

        Raises:
            KeyError: If `name` is not registered.
        """
        if name not in self:
            raise KeyError(f"{name!r} not registered. Available: {sorted(self)}")
        return self[name]


DATASETS: _Registry = _Registry()
BASELINES: _Registry = _Registry()
METRICS: _Registry = _Registry()
