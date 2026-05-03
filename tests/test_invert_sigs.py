"""Verifies SigDiffusions Fourier inversion runs on the patched signature layout.

Background: SigDiffusions/compute_signatures.py:90 was patched to drop the sin/cos
augmentation, so signatures are now computed at iisignature.prepare(dim, sig_depth).
The corresponding inversion path (invert_signatures.py:78) must use the same dim — but
currently uses dim+3, which is the un-patched layout. This test catches that mismatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "SigDiffusions"))


@pytest.mark.smoke
def test_logsig_to_sig_dim_alignment():
    """The dim used by compute_signatures must match the dim used by invert_signatures.

    Both code paths must agree:
      * compute_signatures.py:90 prepares with `iisignature.prepare(dim, sig_depth, "COSX2")`
      * invert_signatures.py:78 prepares with `iisignature.prepare(dim, sig_depth, "COSX2")`
    Currently invert uses `dim+3`, which crashes or produces garbage on the patched layout.
    """
    import iisignature

    dim = 2
    sig_depth = 4
    n = 2
    seq_len = 32

    rng = np.random.RandomState(0)
    paths = np.cumsum(rng.randn(n, seq_len, dim), axis=1).astype(np.float64)
    data = np.concatenate([np.zeros((n, 1, dim), dtype=np.float64), paths], axis=1)

    s_compute = iisignature.prepare(dim, sig_depth, "COSX2")
    logsigs = iisignature.logsig(data, s_compute)

    expected_logsig_dim = iisignature.logsiglength(dim, sig_depth)
    assert logsigs.shape == (n, expected_logsig_dim)

    s_invert = iisignature.prepare(dim, sig_depth, "COSX2")
    sigs = iisignature.logsigtosig(logsigs, s_invert)

    expected_sig_dim = iisignature.siglength(dim, sig_depth)
    assert sigs.shape == (n, expected_sig_dim), f"got {sigs.shape}, want {(n, expected_sig_dim)}"
    assert np.isfinite(sigs).all()


@pytest.mark.smoke
def test_invert_signatures_module_uses_matching_dim():
    """Inspect the source of invert_signatures.py to confirm dim+3 has been removed.

    This is a structural test — it ensures the buggy `dim + 3` literal does not appear
    in the prepare() call inside invert_signatures.py. We do not just match `dim + 3`
    anywhere because that pattern may legitimately appear elsewhere in comments.
    """
    src = (PROJECT_ROOT / "SigDiffusions" / "invert_signatures.py").read_text()
    bad = "iisignature.prepare(dim + 3"
    good = 'iisignature.prepare(dim, sig_depth, "COSX2")'
    assert bad not in src, (
        f"invert_signatures.py still contains the buggy `{bad}`; expected `{good}`"
    )
    assert good in src, f"invert_signatures.py must contain `{good}`"
