"""Smoke tests for fractalsig.losses."""
from __future__ import annotations

import pytest
import torch

from fractalsig.losses import ScaleWeightedMSE


@pytest.mark.smoke
def test_weights_average_to_unity():
    loss = ScaleWeightedMSE([4, 4, 8, 16], beta=1.0)
    assert abs(loss.weights.mean().item() - 1.0) < 1e-5


@pytest.mark.smoke
def test_zero_when_pred_equals_target():
    loss = ScaleWeightedMSE([4, 8], beta=1.0)
    x = torch.randn(3, 12)
    assert loss(x, x).item() < 1e-12


@pytest.mark.smoke
def test_higher_freq_dominates():
    """A residual in the finest detail band must cost more than one in cA_n.

    pywt order is [cA_n, cD_n, ..., cD_1], so positions in the last segment
    correspond to the highest frequency band and must receive the largest
    weight.
    """
    loss = ScaleWeightedMSE([4, 4, 8], beta=1.0)
    pred = torch.zeros(1, 16)
    tgt_low = torch.zeros(1, 16)
    tgt_low[0, 0] = 1.0
    tgt_high = torch.zeros(1, 16)
    tgt_high[0, 15] = 1.0
    assert loss(pred, tgt_high) > loss(pred, tgt_low)


@pytest.mark.smoke
def test_beta_zero_reduces_to_plain_mse():
    """beta=0 collapses all weights to 1, recovering vanilla MSE."""
    loss = ScaleWeightedMSE([4, 4, 8], beta=0.0)
    pred = torch.randn(2, 16)
    target = torch.randn(2, 16)
    expected = ((pred - target) ** 2).mean()
    assert abs(loss(pred, target).item() - expected.item()) < 1e-6


@pytest.mark.smoke
def test_multichannel_weights_repeat_per_channel_pattern():
    """For n_channels > 1, the per-channel weight pattern repeats verbatim."""
    loss = ScaleWeightedMSE([4, 4, 8], beta=1.0, n_channels=2)
    assert loss.weights.shape == (32,)
    assert torch.allclose(loss.weights[:16], loss.weights[16:])


@pytest.mark.smoke
def test_negative_beta_rejected():
    with pytest.raises(ValueError):
        ScaleWeightedMSE([4, 4, 8], beta=-1.0)


@pytest.mark.smoke
def test_gradients_flow_through_loss():
    loss = ScaleWeightedMSE([4, 4, 8], beta=1.0)
    pred = torch.randn(2, 16, requires_grad=True)
    target = torch.randn(2, 16)
    loss(pred, target).backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0
