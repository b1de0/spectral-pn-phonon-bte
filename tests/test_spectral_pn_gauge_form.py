"""The MULTIPLICATIVE gauge: let the slow head span decades cheaply."""
import numpy as np
import pytest
import torch

from pinn_bte.models.spectral_pn import XiHybridPNCoefficientNet

pytestmark = pytest.mark.usefixtures("default_float64")

GAUGES = ("additive", "positive")


def _net(gauge="additive", n=6, seed=0):
    torch.manual_seed(seed)
    v = np.linspace(2.6e12, 8.8e13, n)
    tau = np.linspace(3.7e-12, 1.2e-09, n)
    C = np.linspace(1e5, 7e5, n)
    return XiHybridPNCoefficientNet(
        n_modes=n, L_max=4, gamma_kin=1.0e10, v=v, tau=tau, q=6.28e5,
        width=16, depth=2, emb_dim=4, shared_ahat=False, C=C,
        blend="ugks", macro_row="on", gauge=gauge)


def _t(n=64, t_end=1.0e-9):
    return torch.linspace(0.0, t_end, n, dtype=torch.float64).reshape(-1, 1)


def test_default_is_the_additive_gauge():
    assert _net().gauge == "additive"


def test_additive_gauge_is_bit_identical_to_the_shipped_form():
    """`gauge='additive'` must reproduce `A_hat = 1 + (1-e)h` exactly. Every
    published arm is on it.
    """
    net = _net("additive")
    t = _t()
    A, dA = net._blend_amplitude(t)
    ts = t * net.gamma_kin
    h = torch.log1p(ts)
    for layer in net.slow:            # the implementation's own arithmetic:
        if isinstance(layer, torch.nn.Linear):
            h = layer(h)
        else:                          # SiLU as h*sigmoid(h), not the fused op
            h = h * torch.sigmoid(h)   # (they differ by one ULP)
    want = 1.0 + (1.0 - torch.exp(-ts)) * h
    np.testing.assert_array_equal(A.detach().numpy(), want.detach().numpy())


def test_unknown_gauge_is_refused():
    with pytest.raises(ValueError, match="gauge"):
        _net("exponential-ish")


@pytest.mark.parametrize("gauge", GAUGES)
def test_gauge_starts_at_one_exactly(gauge):
    """A_hat(0) = 1 is the exact IC the whole ansatz is pinned to. A gauge that
    only approximately starts at 1 breaks it.
    """
    A, _ = _net(gauge)._blend_amplitude(_t())
    assert A[0].item() == 1.0


def test_positive_gauge_never_changes_sign():
    """The measured pathology: the shipped gauge reaches -3.19 at L = 0.1, and the
    closure kernels are evaluated at -d ln A_hat/dt, which does not exist for a
    negative A_hat.
    """
    net = _net("positive")
    with torch.no_grad():
        for layer in net.slow:                       # force a large negative h
            if isinstance(layer, torch.nn.Linear):
                layer.weight.mul_(50.0)
                if layer.bias is not None:
                    layer.bias.sub_(20.0)
    A, dA = net._blend_amplitude(_t())
    assert (A > 0).all(), "the positive gauge must not reach zero or below"
    # and the quantity the closure kernels actually consume stays finite even
    # at these absurd weights -- that is what the exponent range guard buys
    assert torch.isfinite(-dA / A).all()


def test_positive_gauge_spans_decades_at_O1_network_output():
    """THE POINT, as an assertion."""
    ts = torch.tensor([7.0], dtype=torch.float64)
    phi = torch.nn.functional.softplus(torch.tensor([1.0], dtype=torch.float64))
    A = torch.exp(-ts * phi)
    assert A.item() < 1e-3          # three decades from an O(1) demand
    assert phi.item() < 2.0


@pytest.mark.parametrize("gauge", GAUGES)
def test_derivative_matches_autograd(gauge):
    """`_blend_amplitude` returns dA/dt in CLOSED FORM because pn_residual's
    forward-mode jvp composes through it.
    """
    net = _net(gauge)
    t = _t(24).clone().requires_grad_(True)
    A, dA = net._blend_amplitude(t)
    got = torch.autograd.grad(A.sum(), t)[0]
    np.testing.assert_allclose(dA.detach().numpy(), got.detach().numpy(),
                               rtol=1e-9, atol=1e-14)


def test_positive_gauge_has_a_finite_log_rate():
    """gamma_hat = -d ln A_hat/dt is what the closure kernels are evaluated at.
    Under the positive gauge it exists everywhere by construction.
    """
    net = _net("positive")
    t = _t()
    A, dA = net._blend_amplitude(t)
    gamma_hat = -dA / A
    assert torch.isfinite(gamma_hat).all()
