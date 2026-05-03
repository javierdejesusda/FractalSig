"""Tests for fractalsig.registries."""
from __future__ import annotations

import pytest

from fractalsig.registries import DATASETS, _Registry


@pytest.mark.smoke
def test_register_and_lookup():
    r = _Registry()

    @r.register("foo")
    class Foo:
        pass

    assert r.get_or_raise("foo") is Foo


@pytest.mark.smoke
def test_double_register_raises():
    r = _Registry()

    @r.register("foo")
    class Foo:
        pass

    with pytest.raises(KeyError, match="already registered"):
        @r.register("foo")
        class Bar:
            pass


@pytest.mark.smoke
def test_unknown_lookup_raises():
    r = _Registry()
    with pytest.raises(KeyError, match="not registered"):
        r.get_or_raise("nope")


@pytest.mark.smoke
def test_global_singletons_present():
    """The DATASETS/BASELINES/METRICS singletons must be importable."""
    from fractalsig.registries import BASELINES, METRICS
    assert isinstance(DATASETS, _Registry)
    assert isinstance(BASELINES, _Registry)
    assert isinstance(METRICS, _Registry)
