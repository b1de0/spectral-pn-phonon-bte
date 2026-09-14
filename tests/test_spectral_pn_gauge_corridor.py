"""The RATE CORRIDOR gauge: bound the gauge from BOTH sides, no constant."""
import numpy as np
import pytest
import torch

from pinn_bte.models.spectral_pn import XiHybridPNCoefficientNet, rate_corridor

pytestmark = pytest.mark.usefixtures("default_float64")


def _modes(n=8, q=6.283e-4):
    """A mode set in the SHIPPED unit system, spanning the real optical thickness."""
    v = np.linspace(2.6e12, 8.8e13, n)
    tau = np.linspace(3.7e-12, 1.2e-09, n)
    C = np.linspace(1e5, 7e5, n)
    return v, tau, C, q


def _net(gauge="additive", n=8, seed=0):
    torch.manual_seed(seed)
    v, tau, C, q = _modes(n)
    return XiHybridPNCoefficientNet(
        n_modes=n, L_max=4, gamma_kin=1.0e10, v=v, tau=tau, q=q,
        width=16, depth=2, emb_dim=4, shared_ahat=False, C=C,
        blend="ugks", macro_row="on", gauge=gauge)


def _t(n=64, t_end=1.0e-9):
    return torch.linspace(0.0, t_end, n, dtype=torch.float64).reshape(-1, 1)


def test_corridor_is_a_pure_function_of_the_mode_set():
    """DATA-FREE BY SIGNATURE. If a bound could read the reference, the paper's
    central claim would be false by construction.
    """
    import inspect
    assert set(inspect.signature(rate_corridor).parameters) == {"v", "tau", "q"}


def test_lower_bound_is_the_slowest_suppressed_mode():
    v, tau, C, q = _modes()
    lo, hi = rate_corridor(v, tau, q)
    xi = q * v * tau
    S = 3.0 * (xi - np.arctan(xi)) / xi ** 3
    want = float(((q ** 2) * v ** 2 * tau * S / 3.0).min())
    assert lo == pytest.approx(want, rel=1e-12)


def test_upper_bound_is_the_fastest_kinetic_mode():
    v, tau, C, q = _modes()
    lo, hi = rate_corridor(v, tau, q)
    assert hi == pytest.approx(float((1.0 / tau + q * v).max()), rel=1e-12)


def test_corridor_is_ordered_and_positive():
    lo, hi = rate_corridor(*_modes()[:2], _modes()[3])
    assert 0.0 < lo < hi


def test_corridor_brackets_the_diffusive_limit_where_that_limit_exists():
    """At Kn << 1 the collective rate tends to q^2*kappa/C, and the corridor must
    contain it: that is the limit the paper's Fourier rung certifies.
    """
    v, tau, C, q = _modes()
    for scale in (1e-1, 1e-2, 1e-3):          # xi_max 6.6 -> 0.66 -> 0.066
        qs = q * scale
        lo, hi = rate_corridor(v, tau, qs)
        g_diff = (qs ** 2) * float(np.sum(C * v ** 2 * tau)) / 3.0 / float(np.sum(C))
        assert lo <= g_diff <= hi, f"diffusive limit outside corridor at q*{scale}"


def test_corridor_excludes_the_fourier_expression_where_it_is_unphysical():
    v, tau, C, q = _modes()                    # xi_max = 66, hyper-ballistic
    lo, hi = rate_corridor(v, tau, q)
    g_fourier = (q ** 2) * float(np.sum(C * v ** 2 * tau)) / 3.0 / float(np.sum(C))
    assert g_fourier > hi


def test_default_is_still_the_additive_gauge():
    assert _net().gauge == "additive"


def test_corridor_gauge_starts_at_one_exactly():
    """A_hat(0) = 1 is the exact IC the ansatz is pinned to."""
    A, _ = _net("corridor")._blend_amplitude(_t())
    assert A[0].item() == 1.0


@pytest.mark.parametrize("push", [+60.0, -60.0])
def test_gauge_rate_stays_inside_the_corridor_under_absurd_weights(push):
    """THE POINT. The additive gauge drifts to 2056x inflation and the
    multiplicative one drifts to 2.35e-12; both are the loss being indifferent
    to the gauge.
    """
    net = _net("corridor")
    with torch.no_grad():
        for layer in net.slow:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.mul_(50.0)
                if layer.bias is not None:
                    layer.bias.add_(push)
    t = _t()
    A, dA = net._blend_amplitude(t)
    live = (A > 1e-300).squeeze()
    gamma_hat = (-dA / A)[live][1:]
    v, tau, C, q = _modes()
    lo, hi = rate_corridor(v, tau, q)
    assert (gamma_hat >= lo * (1 - 1e-12)).all()
    assert (gamma_hat <= hi * (1 + 1e-12)).all()
    assert (A > 0).all()
    assert torch.isfinite(A).all()


def test_the_corridor_is_wide_enough_to_be_a_bound_not_a_target():
    """A bound that pins the answer is supervision wearing a bound's clothes."""
    lo, hi = rate_corridor(*_modes()[:2], _modes()[3])
    assert hi / lo > 100.0


def test_derivative_matches_autograd():
    """pn_residual's forward-mode jvp composes through the closed-form dA/dt; if
    they disagree every residual on this path is silently wrong.
    """
    net = _net("corridor")
    t = _t(24).clone().requires_grad_(True)
    A, dA = net._blend_amplitude(t)
    got = torch.autograd.grad(A.sum(), t)[0]
    np.testing.assert_allclose(dA.detach().numpy(), got.detach().numpy(),
                               rtol=1e-9, atol=1e-14)


def test_gradient_is_alive_across_the_corridor():
    """sigmoid saturates. If the gradient vanished at the bounds the corridor
    would be a trap rather than a constraint, so check it is finite and nonzero
    at a generic interior point.
    """
    net = _net("corridor")
    t = _t(16)
    lo, _ = rate_corridor(*_modes()[:2], _modes()[3])
    A, _ = net._blend_amplitude(_t(16, t_end=3.0 / lo))
    A.sum().backward()
    g = [p.grad.abs().max().item() for p in net.slow.parameters() if p.grad is not None]
    assert g and max(g) > 0.0 and np.isfinite(max(g))
