"""Verifies the iisignature integration we depend on."""
from __future__ import annotations

import iisignature
import numpy as np
import pytest


@pytest.mark.smoke
def test_logsig_shape_for_dim2_depth4():
    s = iisignature.prepare(2, 4)
    x = np.cumsum(np.random.RandomState(0).randn(64, 2), axis=0)
    out = iisignature.logsig(x, s)
    expected_dim = iisignature.logsiglength(2, 4)
    assert out.shape == (expected_dim,)


@pytest.mark.smoke
def test_logsig_invariant_to_translation():
    """Signatures of (x + c) and x are equal: signatures depend only on increments."""
    s = iisignature.prepare(2, 3)
    x = np.cumsum(np.random.RandomState(0).randn(32, 2), axis=0)
    a = iisignature.logsig(x, s)
    b = iisignature.logsig(x + 17.0, s)
    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.smoke
def test_cosx2_supports_logsigtosig_roundtrip():
    """The COSX2 prepare method must allow logsig -> sig conversion."""
    s = iisignature.prepare(2, 3, "COSX2")
    x = np.cumsum(np.random.RandomState(0).randn(16, 2), axis=0)
    logsig = iisignature.logsig(x, s)
    sig = iisignature.logsigtosig(logsig, s)
    assert sig.shape == (iisignature.siglength(2, 3),)
    assert np.isfinite(sig).all()


@pytest.mark.smoke
def test_logsig_grows_with_depth():
    """logsiglength should be monotone in depth."""
    lens = [iisignature.logsiglength(2, d) for d in (2, 3, 4, 5)]
    assert lens == sorted(lens)
    assert all(b > a for a, b in zip(lens, lens[1:]))
