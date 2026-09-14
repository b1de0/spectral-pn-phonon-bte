"""The ugks blend: default byte-identical, kernel limits, t=0 flux exactly
zero, safeguards against NaN.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_bte.models.spectral_pn import (
    XiHybridPNCoefficientNet,
    duhamel_free_coefficients,
    legendre_streaming_matrix,
    pn_amplitude,
    pn_residual,
    solve_pn_ode,
    ugks_resolvent_kernels,
)
from pinn_bte.physics.ttg_dispersion import _J_moments_axisym

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.usefixtures("default_float64")


# --------------------------------------------------------------- fixtures
def _gamma_kin(v, tau, C, q):
    return float(q ** 2 * torch.sum(C * v ** 2 * tau) / 3.0 / torch.sum(C))


def _si_case(Nk=8, L_um=1.0, T_ref=300.0):
    """Full-Si mixed-band case (CE + plain + Duhamel all populated at L=1)."""
    from pinn_bte.physics.ttg_dispersion import phonon_modes
    v, tau, C = phonon_modes(Nk, T_ref)
    C = C / C.mean()
    q = 2 * np.pi / (L_um * 1e4)
    return (torch.tensor(v), torch.tensor(tau), torch.tensor(C), float(q))


def _gray_three():
    v = torch.tensor([0.1, 3.0, 30.0])
    tau = torch.tensor([1.0, 1.0, 1.0])
    C = torch.tensor([1.0, 1.0, 1.0])
    return v, tau, C, 1.0


def _blend_net(v, tau, C, q, seed=0, L_max=8, width=32, depth=3, **kw):
    torch.manual_seed(seed)
    return XiHybridPNCoefficientNet(
        n_modes=len(v), L_max=L_max, gamma_kin=_gamma_kin(v, tau, C, q),
        v=v.numpy(), tau=tau.numpy(), q=q, width=width, depth=depth,
        blend="ugks", **kw).double()


class _InjectedNet(XiHybridPNCoefficientNet):
    """Test/harness injection point: override the amplitude head with a prescribed
    (A_hat(t), dA_hat/dt) pair.
    """

    def set_injection(self, fn):
        self._inj_fn = fn

    def _blend_amplitude(self, t2):
        return self._inj_fn(t2)


def _injected_net(v, tau, C, q, fn, seed=0, L_max=8):
    torch.manual_seed(seed)
    net = _InjectedNet(
        n_modes=len(v), L_max=L_max, gamma_kin=_gamma_kin(v, tau, C, q),
        v=v.numpy(), tau=tau.numpy(), q=q, width=16, depth=2,
        blend="ugks").double()
    net.set_injection(fn)
    return net


# ============================== B0: default path byte-identical (T0c-style)
def test_blend_default_is_byte_identical():
    v, tau, C, q = _si_case(Nk=8, L_um=1.0)
    gk = _gamma_kin(v, tau, C, q)
    torch.manual_seed(3)
    net0 = XiHybridPNCoefficientNet(len(v), 8, gk, v.numpy(), tau.numpy(), q,
                                    width=32, depth=3, shared_ahat=True,
                                    C=C.numpy(), carrier_energy=True).double()
    torch.manual_seed(3)
    net1 = XiHybridPNCoefficientNet(len(v), 8, gk, v.numpy(), tau.numpy(), q,
                                    width=32, depth=3, shared_ahat=True,
                                    C=C.numpy(), carrier_energy=True,
                                    blend="banded").double()
    for name in ("ce_mask", "duhamel_mask", "carrier_mask", "wce", "epsl"):
        assert torch.equal(getattr(net0, name), getattr(net1, name)), name
    assert net0.macro_licensed == net1.macro_licensed
    t = torch.linspace(0.0, 4.0, 25) * 1e-8
    c0, s0 = net0.coefficients(t, v, tau, C, q)
    c1, s1 = net1.coefficients(t, v, tau, C, q)
    assert torch.equal(c0, c1) and torch.equal(s0, s1)
    r0, carr0 = pn_residual(net0, t, v, tau, C, q, ap_scaling=True,
                            return_carrier=True)
    r1, carr1 = pn_residual(net1, t, v, tau, C, q, ap_scaling=True,
                            return_carrier=True)
    assert torch.equal(r0, r1) and torch.equal(carr0, carr1)


# ======================================== B0b: loud rejection of bad combos
def test_blend_invalid_value_rejected():
    v, tau, C, q = _gray_three()
    with pytest.raises(ValueError, match="blend"):
        XiHybridPNCoefficientNet(3, 8, _gamma_kin(v, tau, C, q),
                                 v.numpy(), tau.numpy(), q, blend="maznev")


def test_blend_rejects_carrier_machinery():
    v, tau, C, q = _gray_three()
    with pytest.raises(AssertionError):
        XiHybridPNCoefficientNet(3, 8, _gamma_kin(v, tau, C, q),
                                 v.numpy(), tau.numpy(), q,
                                 shared_ahat=True, C=C.numpy(),
                                 blend="ugks")


def test_blend_t0_flux_exactly_zero():
    """Energy IC exact: c0(0) = C."""
    v, tau, C, q = _gray_three()
    net = _blend_net(v, tau, C, q, seed=3)
    t0 = torch.zeros(1)
    with torch.no_grad():
        c, s = net.coefficients(t0, v, tau, C, q)
    assert int(torch.count_nonzero(s[0])) == 0, "t=0 flux must be EXACTLY 0"
    assert torch.allclose(c[0, :, 0], C, atol=1e-13)
    assert float(c[0, :, 1:].abs().max()) < 1e-13


# =========================== B2: gamma=0 kernel limits (ugks_limits Claim 1)
def test_blend_kernels_gamma0_limits():
    xi = torch.tensor([1e-3, 0.01, 0.1, 0.653, 1.0, 3.0, 10.0, 65.3,
                       653.0, 6532.0], dtype=torch.float64)
    y = torch.zeros(1, len(xi), dtype=torch.float64)
    K0, K1 = ugks_resolvent_kernels(y, xi)
    K0_ref = torch.arctan(xi) / xi
    K1_ref = 3.0 * (1.0 - torch.arctan(xi) / xi) / xi     # = xi * A(xi)
    assert torch.allclose(K0[0], K0_ref, rtol=1e-12, atol=0)
    assert torch.allclose(K1[0], K1_ref, rtol=1e-10, atol=1e-15)


# =================== B2b: torch port == shipped oracle incl. a<0 (physical)
def test_blend_kernels_match_shipped_oracle():
    tau = np.ones(6)
    xi = np.array([0.05, 0.5, 1.0, 3.0, 30.0, 6532.0])
    for g in (-0.5, 0.3, 0.99, 1.0, 1.5, 2.44):
        with np.errstate(divide="ignore", invalid="ignore"):
            # the oracle's own b->0 series arm divides by a = 1-g*tau even
            # when unselected (harmless; xi >> 1e-10 here)
            J0, J1, _ = _J_moments_axisym(g, tau, xi)
        y = torch.full((1, 6), float(g), dtype=torch.float64)
        K0, K1 = ugks_resolvent_kernels(
            y, torch.as_tensor(xi, dtype=torch.float64))
        assert np.allclose(K0[0].numpy(), np.real(J0), rtol=1e-12,
                           atol=1e-15), f"K0 mismatch at gamma*tau={g}"
        assert np.allclose(K1[0].numpy(), -np.imag(3.0 * J1), rtol=1e-12,
                           atol=1e-15), f"K1 mismatch at gamma*tau={g}"


# ================================= B3: smooth safeguards — adversarial A_hat
@pytest.mark.parametrize("label,fn", [
    ("tiny", lambda t2: (torch.full_like(t2, 1e-300),
                         torch.full_like(t2, -1e5))),
    ("at-delta-scale", lambda t2: (torch.full_like(t2, 1e-4),
                                   torch.full_like(t2, -1e8))),
    ("zero-over-zero", lambda t2: (torch.zeros_like(t2),
                                   torch.zeros_like(t2))),
    ("negative", lambda t2: (torch.full_like(t2, -0.5),
                             torch.full_like(t2, 2.0))),
    ("oscillating", lambda t2: (torch.sin(3e9 * t2),
                                3e9 * torch.cos(3e9 * t2))),
])
def test_blend_safeguard_no_nan(label, fn):
    v, tau, C, q = _si_case(Nk=8, L_um=1.0)
    net = _injected_net(v, tau, C, q, fn, seed=1)
    t = torch.linspace(0.0, 4.0, 17) * 1e-8
    c, s = net.coefficients(t, v, tau, C, q)
    assert torch.isfinite(c).all() and torch.isfinite(s).all(), label
    res = pn_residual(net, t, v, tau, C, q, ap_scaling=True)
    assert torch.isfinite(res).all(), label
    (res ** 2).mean().backward()
    for p in net.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), label


def test_blend_safeguard_no_nan_wild_slow_head():
    """Same, through the REAL amplitude head with adversarially large slow weights
    (the full production path incl. the dual-number derivative).
    """
    v, tau, C, q = _si_case(Nk=8, L_um=0.01)   # ballistic rung
    net = _blend_net(v, tau, C, q, seed=2)
    with torch.no_grad():
        for p in net.slow.parameters():
            p.mul_(50.0)
    t = torch.linspace(0.0, 1.0, 17) * 1e-10
    res = pn_residual(net, t, v, tau, C, q, ap_scaling=True)
    assert torch.isfinite(res).all()
    (res ** 2).mean().backward()
    for p in net.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


# =========================== B4: dual-number slow-head derivative == autograd
def test_blend_amplitude_derivative_matches_autograd():
    v, tau, C, q = _si_case(Nk=8, L_um=1.0)
    net = _blend_net(v, tau, C, q, seed=5)
    t2 = (torch.linspace(0.0, 3.0, 33, dtype=torch.float64)
          .reshape(-1, 1) * 1e-8).requires_grad_(True)
    A, dA = net._blend_amplitude(t2)
    dA_auto = torch.autograd.grad(A.sum(), t2)[0]
    scale = float(dA_auto.detach().abs().max())
    assert float((dA - dA_auto).detach().abs().max()) < 1e-12 * max(scale, 1.0)


# ================== B5: truncation-edge correction GLOBAL (doc §2, risk 2)
def test_blend_edge_correction_global():
    v = torch.tensor([0.5, 30.0])
    tau = torch.tensor([1.0, 1.0])
    C = torch.tensor([1.0, 1.0])
    q, L_max = 1.0, 2
    net = _injected_net(v, tau, C, q,
                        lambda t2: (torch.zeros_like(t2),
                                    torch.zeros_like(t2)), L_max=L_max)
    with torch.no_grad():
        net.trunk[-1].weight.zero_()
        net.trunk[-1].bias.zero_()
    t = torch.linspace(0.05, 2.0, 31, dtype=torch.float64)
    res = pn_residual(net, t, v, tau, C, q, ap_scaling=False).detach()
    # rows: [c_0..c_Lmax, s_0..s_Lmax]; l>=1 rows of BOTH parities
    rows_l_ge1 = torch.cat([res[..., 1:L_max + 1],
                            res[..., L_max + 2:]], dim=-1)
    worst = float(rows_l_ge1.abs().max())
    assert worst < 1e-10, f"free part not edge-corrected globally: {worst:.2e}"
    # mutation guard: the would-be (uncorrected) edge term is NOT small
    with torch.no_grad():
        c_hi, s_hi = duhamel_free_coefficients(t, v, tau, C, q, L_max + 1)
    T_edge = float(legendre_streaming_matrix(L_max + 1)[L_max, L_max + 1])
    scale = ((1.0 / tau + q * v) * C).reshape(1, -1)
    edge = (v * q).reshape(1, -1) * T_edge * s_hi[..., L_max + 1] / scale
    assert float(edge.abs().max()) > 1e3 * worst, \
        "edge coupling trivially small — the test cannot see the correction"


# =================================== B6: macro row licensing (doc §5)
def test_blend_macro_licensed_on_physical_condition():
    v, tau, C, q = _si_case(Nk=8, L_um=1.0)     # mixed: ce.all() is False
    gk = _gamma_kin(v, tau, C, q)
    torch.manual_seed(0)
    banded = XiHybridPNCoefficientNet(len(v), 8, gk, v.numpy(), tau.numpy(),
                                      q, width=16, depth=2,
                                      shared_ahat=True, C=C.numpy()).double()
    assert not banded.macro_licensed
    blend = _blend_net(v, tau, C, q, width=16, depth=2)
    assert blend.macro_licensed


# ================================================= B7: trainer + CLI plumbing
def _tiny_blend_cfg(**overrides):
    from pinn_bte.training.transient.spectral_pn_trainer import SpectralPNConfig
    cfg = dict(
        v=np.array([0.1, 3.0, 30.0]), tau=np.ones(3), C=np.ones(3),
        q=1.0, t_end=6.0, L_max=4, width=16, depth=2,
        epochs=3, lr=2e-3, nt=41, seed=0, device="cpu", dtype="float64",
        formulation="xihybrid", blend="ugks", label="blend-smoke",
    )
    cfg.update(overrides)
    return SpectralPNConfig(**cfg)


def test_trainer_blend_records_provenance(tmp_path):
    """cfg.blend reaches the net and is recorded in the npz (provenance: every CLI
    param must be reconstructable from the run artifact).
    """
    from pinn_bte.training.transient.spectral_pn_trainer import train_spectral_pn
    res = train_spectral_pn(_tiny_blend_cfg(outdir=tmp_path, run_id="b"))
    assert np.isfinite(res["loss_final"])
    data = np.load(tmp_path / "b_results.npz", allow_pickle=True)
    assert str(data["blend"]) == "ugks"
    res2 = train_spectral_pn(_tiny_blend_cfg(blend="banded",
                                             outdir=tmp_path, run_id="b2"))
    data2 = np.load(tmp_path / "b2_results.npz", allow_pickle=True)
    assert str(data2["blend"]) == "banded"


def test_trainer_blend_forbidden_combos():
    from pinn_bte.training.transient.spectral_pn_trainer import train_spectral_pn
    with pytest.raises(AssertionError):
        train_spectral_pn(_tiny_blend_cfg(formulation="plain"))
    with pytest.raises(AssertionError):
        train_spectral_pn(_tiny_blend_cfg(shared_ahat=True,
                                          carrier_energy=True))
    with pytest.raises(AssertionError):
        train_spectral_pn(_tiny_blend_cfg(carrier_row=True))


def test_cli_blend_flag():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_spectral_pn_cli_test",
        REPO / "experiments" / "transient" / "run_spectral_pn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.parse_args(["--L", "1.0"]).blend == "banded"
    args = mod.parse_args(["--L", "1.0", "--formulation", "xihybrid",
                           "--blend", "ugks"])
    assert args.blend == "ugks"
    with pytest.raises(SystemExit):        # ugks requires xihybrid
        mod.parse_args(["--L", "1.0", "--blend", "ugks"])
    with pytest.raises(SystemExit):        # ugks excludes the carrier flags
        mod.parse_args(["--L", "1.0", "--formulation", "xihybrid",
                        "--blend", "ugks", "--carrier-row"])


# ============================================== B8: training smoke (3 bands)
def test_blend_training_smoke_three_band_spectrum():
    from pinn_bte.training.transient.spectral_pn_trainer import train_spectral_pn
    v, tau, C, q = _gray_three()
    t_ref = np.linspace(0.0, 6.0, 241)
    A_ref = solve_pn_ode(v.numpy(), tau.numpy(), C.numpy(), q=q,
                         t=t_ref, L_max=40)
    cfg = _tiny_blend_cfg(epochs=800, width=48, depth=3, L_max=8,
                          nt=241, t_grid="uniform",
                          t_ref=t_ref, A_ref=A_ref)
    res = train_spectral_pn(cfg)
    assert res["loss_final"] < 0.05 * res["loss_initial"], \
        f"residual did not drop: {res['loss_initial']:.2e} -> {res['loss_final']:.2e}"
    err = float(np.abs(res["A"] - np.interp(res["t"], t_ref, A_ref)).max())
    assert err < 0.15, f"amplitude off by {err:.3f} after smoke training"
