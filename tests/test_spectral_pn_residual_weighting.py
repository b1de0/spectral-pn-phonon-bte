"""The TIME-WEIGHTED residual: make the tail visible to the objective."""
import numpy as np
import pytest
import torch

from pinn_bte.models.spectral_pn import residual_time_weight

pytestmark = pytest.mark.usefixtures("default_float64")


def test_weight_never_reads_the_reference():
    """DATA-FREE BY SIGNATURE. The whole PN claim is that the reference solution
    does not enter the loss and appears only in post-processing.
    """
    import inspect
    params = set(inspect.signature(residual_time_weight).parameters)
    assert params == {"amplitude", "floor"}, (
        f"unexpected signature {params}: the weight may see the model's own "
        "amplitude and a floor, and nothing else -- in particular no A_ref, no "
        "t_ref, no reference rate")


def test_weight_is_detached():
    """The weight is a CONSTANT in the backward pass."""
    a = torch.linspace(1.0, 1e-4, 32, dtype=torch.float64, requires_grad=True)
    w = residual_time_weight(a, floor=1e-3)
    assert not w.requires_grad, "the weight must not carry a gradient"
    assert w.grad_fn is None


def test_weight_is_the_reciprocal_amplitude_above_the_floor():
    """Above the floor a given RELATIVE error must cost the same at every time,
    which is exactly w = 1/|A|.
    """
    a = torch.tensor([1.0, 0.5, 0.1, 0.01], dtype=torch.float64)
    w = residual_time_weight(a, floor=1e-3)
    np.testing.assert_allclose(w.numpy(), [1.0, 2.0, 10.0, 100.0], rtol=1e-12)


def test_floor_caps_the_amplification():
    """The floor IS the maximum amplification, and that is the whole content of
    the one new constant. Below it the weight must stop growing.
    """
    a = torch.tensor([1e-3, 1e-4, 1e-8, 0.0], dtype=torch.float64)
    w = residual_time_weight(a, floor=1e-3)
    np.testing.assert_allclose(w.numpy(), [1e3, 1e3, 1e3, 1e3], rtol=1e-12)


def test_sign_of_the_amplitude_is_irrelevant():
    """A(t) changes sign at the ballistic rungs and that is REAL physics there."""
    a = torch.tensor([0.4, -0.4], dtype=torch.float64)
    w = residual_time_weight(a, floor=1e-6)
    assert w[0].item() == pytest.approx(w[1].item())
    assert (w > 0).all()


def test_weight_is_finite_through_a_zero_crossing():
    """The amplitude passes through exactly zero at the ballistic pole."""
    a = torch.tensor([1e-2, 1e-6, 0.0, -1e-6, -1e-2], dtype=torch.float64)
    w = residual_time_weight(a, floor=1e-4)
    assert torch.isfinite(w).all()
    assert w.max().item() == pytest.approx(1e4)


def test_floor_must_be_positive():
    with pytest.raises(ValueError, match="floor"):
        residual_time_weight(torch.ones(4, dtype=torch.float64), floor=0.0)


def test_it_removes_the_dynamic_range_that_hides_the_tail():
    """The defect in one assertion."""
    a = torch.tensor([1.0, 1e-3], dtype=torch.float64)      # head, tail
    rel_err = 0.01                                          # 1 % at both ends
    res = a * rel_err                                       # residual ~ amplitude

    unweighted = (res ** 2)
    assert unweighted[0] / unweighted[1] == pytest.approx(1e6)

    w = residual_time_weight(a, floor=1e-4)
    weighted = (res * w) ** 2
    assert weighted[0] / weighted[1] == pytest.approx(1.0, rel=1e-12)


def test_the_floor_is_the_only_free_parameter():
    """A constant with no sensitivity measurement is a tuned constant."""
    a = torch.tensor([1.0, 1e-3, 1e-5], dtype=torch.float64)
    w2, w3, w4 = (residual_time_weight(a, floor=f) for f in (1e-2, 1e-3, 1e-4))
    assert w2.max() < w3.max() < w4.max()
    # and they must agree wherever the amplitude is above every floor
    for w in (w2, w3, w4):
        assert w[0].item() == pytest.approx(1.0)
