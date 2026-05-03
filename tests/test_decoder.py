"""Smoke tests for fractalsig.decoder.FractalDecoder."""
from __future__ import annotations

import pytest
import torch

from fractalsig.decoder import FractalDecoder


@pytest.mark.smoke
def test_forward_shape():
    m = FractalDecoder(input_dim=24, hidden_dim=32, output_seq_len=64, out_channels=1)
    x = torch.randn(8, 24)
    y = m(x)
    assert y.shape == (8, 64, 1)


@pytest.mark.smoke
def test_gradients_flow():
    m = FractalDecoder(input_dim=24, hidden_dim=32, output_seq_len=64, out_channels=1)
    x = torch.randn(2, 24)
    m(x).sum().backward()
    params_with_grad = [
        p for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert len(params_with_grad) == sum(1 for _ in m.parameters())


@pytest.mark.smoke
@pytest.mark.parametrize("wavelet", ["db4", "sym8", "coif3", "haar"])
def test_supports_multiple_wavelets(wavelet):
    m = FractalDecoder(
        input_dim=8, hidden_dim=16, output_seq_len=64, out_channels=1, wavelet=wavelet
    )
    y = m(torch.randn(2, 8))
    assert y.shape == (2, 64, 1)


@pytest.mark.smoke
def test_param_count_reported():
    m = FractalDecoder(input_dim=24, hidden_dim=32, output_seq_len=64, out_channels=1)
    n = m.get_num_params()
    assert n > 0
    assert n == sum(p.numel() for p in m.parameters() if p.requires_grad)


@pytest.mark.smoke
def test_extra_repr_contains_key_fields():
    m = FractalDecoder(input_dim=24, hidden_dim=32, output_seq_len=64, out_channels=1)
    s = m.extra_repr()
    for key in ("input_dim", "hidden_dim", "output_seq_len", "wavelet"):
        assert key in s
