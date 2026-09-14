"""Xi-composed hybrid PN net: per-mode formulation selection by the KNOWN
dimensionless number xi_m = q v_m tau_m (deterministic physics weighting — not
learned; the CE budget-drift at L=0.1 showed gradient descent picks the wrong
basin when given that freedom).
"""
import numpy as np
import pytest
import torch

from pinn_bte.models.spectral_pn import (
    CEPNCoefficientNet,
    PNCoefficientNet,
    XiHybridPNCoefficientNet,
    pn_amplitude,
    pn_residual,
    solve_pn_ode,
)

# float64 confined to this module's tests (no global leak)
pytestmark = pytest.mark.usefixtures("default_float64")


def _gray_three():
    """Three gray modes, one per band at (xi_split=1, xi_duh=10): xi = 0.1 (CE), 3
    (plain), 30 (Duhamel) at q = 1.
    """
    v = torch.tensor([0.1, 3.0, 30.0])
    tau = torch.tensor([1.0, 1.0, 1.0])
    C = torch.tensor([1.0, 1.0, 1.0])
    return v, tau, C, 1.0


def _gamma_kin(v, tau, C, q):
    return float(q ** 2 * torch.sum(C * v ** 2 * tau) / 3.0 / torch.sum(C))


def _hybrid(v, tau, C, q, xi_split, xi_duh=10.0, seed=0, width=32, depth=3):
    torch.manual_seed(seed)
    return XiHybridPNCoefficientNet(
        n_modes=len(v), L_max=8, gamma_kin=_gamma_kin(v, tau, C, q),
        v=v.numpy(), tau=tau.numpy(), q=q,
        width=width, depth=depth, xi_split=xi_split,
        xi_duh=xi_duh).double()


def test_all_duhamel_limit_equals_duhamel_plain():
    """Both thresholds below every xi_m: reproduce the Duhamel-enriched plain net
    exactly (same trunk/emb weights).
    """
    v, tau, C, q = _gray_three()
    hyb = _hybrid(v, tau, C, q, xi_split=1e-6, xi_duh=1e-6)
    ref = PNCoefficientNet(n_modes=3, L_max=8, width=32, depth=3,
                           duhamel=True).double()
    ref.trunk.load_state_dict(hyb.trunk.state_dict())
    ref.emb.load_state_dict(hyb.emb.state_dict())
    t = torch.linspace(0.0, 3.0, 41)
    c_h, s_h = hyb.coefficients(t, v, tau, C, q)
    c_r, s_r = ref.coefficients(t, v, tau, C, q)
    assert torch.allclose(c_h, c_r, atol=1e-12)
    assert torch.allclose(s_h, s_r, atol=1e-12)


def test_all_plain_limit_equals_plain():
    """xi_split below and xi_duh above every xi_m: reproduce the PLAIN net exactly
    — the new middle band.
    """
    v, tau, C, q = _gray_three()
    hyb = _hybrid(v, tau, C, q, xi_split=1e-6, xi_duh=1e6)
    ref = PNCoefficientNet(n_modes=3, L_max=8, width=32, depth=3,
                           duhamel=False).double()
    ref.trunk.load_state_dict(hyb.trunk.state_dict())
    ref.emb.load_state_dict(hyb.emb.state_dict())
    t = torch.linspace(0.0, 3.0, 41)
    c_h, s_h = hyb.coefficients(t, v, tau, C, q)
    c_r, s_r = ref.coefficients(t, v, tau, C, q)
    assert torch.allclose(c_h, c_r, atol=1e-12)
    assert torch.allclose(s_h, s_r, atol=1e-12)
    r_h = pn_residual(hyb, t, v, tau, C, q)
    r_r = pn_residual(ref, t, v, tau, C, q)
    assert torch.allclose(r_h, r_r, atol=1e-12)


def test_all_diffusive_limit_equals_ce():
    """xi_split above every xi_m: reproduce the CE net exactly (same
    trunk/emb/slow weights), residual included.
    """
    v, tau, C, q = _gray_three()
    hyb = _hybrid(v, tau, C, q, xi_split=1e6)
    ref = CEPNCoefficientNet(n_modes=3, L_max=8,
                             gamma_kin=_gamma_kin(v, tau, C, q),
                             v=v.numpy(), tau=tau.numpy(), q=q,
                             width=32, depth=3).double()
    ref.trunk.load_state_dict(hyb.trunk.state_dict())
    ref.emb.load_state_dict(hyb.emb.state_dict())
    ref.slow.load_state_dict(hyb.slow.state_dict())
    t = torch.linspace(0.0, 3.0, 41)
    c_h, s_h = hyb.coefficients(t, v, tau, C, q)
    c_r, s_r = ref.coefficients(t, v, tau, C, q)
    assert torch.allclose(c_h, c_r, atol=1e-12)
    assert torch.allclose(s_h, s_r, atol=1e-12)
    r_h = pn_residual(hyb, t, v, tau, C, q, ap_scaling=True)
    r_r = pn_residual(ref, t, v, tau, C, q, ap_scaling=True)
    assert torch.allclose(r_h, r_r, atol=1e-12)


def test_ic_exact_three_bands():
    v, tau, C, q = _gray_three()
    hyb = _hybrid(v, tau, C, q, xi_split=1.0, xi_duh=10.0, seed=3)
    t0 = torch.zeros(1)
    with torch.no_grad():
        c, s = hyb.coefficients(t0, v, tau, C, q)
    assert torch.allclose(c[0, :, 0], C, atol=1e-12)
    for m in (1, 2):                     # plain and Duhamel bands
        assert float(c[0, m, 1:].abs().max()) < 1e-12
        assert float(s[0, m, :].abs().max()) < 1e-12
    xi0 = float(q * v[0] * tau[0])       # CE band: slaved flux O(xi)
    assert float(s[0, 0, :].abs().max()) <= 1.1 * xi0 * float(C[0])


def test_band_masks_and_duhamel_correction():
    """Masks land one mode per band; the l = L_max+1 Duhamel residual correction
    applies ONLY to the Duhamel band.
    """
    v, tau, C, q = _gray_three()
    hyb = _hybrid(v, tau, C, q, xi_split=1.0, xi_duh=10.0, seed=5)
    assert hyb.ce_mask.tolist() == [True, False, False]
    assert hyb.duhamel_mask.tolist() == [False, False, True]
    t = torch.linspace(0.0, 2.0, 21)
    res = pn_residual(hyb, t, v, tau, C, q)
    assert res.shape == (21, 3, 2 * (8 + 1))


def test_training_smoke_three_band_spectrum():
    """600 Adam steps on the three-band gray system: the residual drops and A(t)
    lands near the exact expm reference — one formulation covering all three
    regimes at once, no supervision.
    """
    v, tau, C, q = _gray_three()
    torch.manual_seed(0)
    hyb = _hybrid(v, tau, C, q, xi_split=1.0, xi_duh=10.0,
                  width=48, depth=3)
    opt = torch.optim.Adam(hyb.parameters(), lr=2e-3)
    t_coll = torch.linspace(0.0, 6.0, 241)
    loss0 = None
    # 1000 steps: the xi=30 mode's winding correction needs a bit more
    # budget than the two-band smoke did (600 -> err 0.105; 1000 -> <0.08)
    for _ in range(1000):
        opt.zero_grad()
        res = pn_residual(hyb, t_coll, v, tau, C, q, ap_scaling=True)
        loss = (res ** 2).mean()
        loss.backward()
        opt.step()
        if loss0 is None:
            loss0 = float(loss)
    assert float(loss) < 0.05 * loss0, "residual did not drop"
    with torch.no_grad():
        A = pn_amplitude(hyb, t_coll, tau, C, v=v, q=q).numpy()
    # exact reference needs the winding harmonics of the xi=30 mode
    A_ref = solve_pn_ode(v.numpy(), tau.numpy(), C.numpy(), q=q,
                         t=t_coll.numpy(), L_max=40)
    err = np.abs(A - A_ref).max()
    assert err < 0.12, f"amplitude off by {err:.3f} after smoke training"


def _mixed_slow():
    v = torch.tensor([0.1, 3.0])
    tau = torch.tensor([1.0, 1.0])
    C = torch.tensor([1.0, 1.0])
    return v, tau, C, 1.0


def _shared(v, tau, C, q, seed=0, width=32, depth=3, **kw):
    torch.manual_seed(seed)
    return XiHybridPNCoefficientNet(
        n_modes=len(v), L_max=8, gamma_kin=_gamma_kin(v, tau, C, q),
        v=v.numpy(), tau=tau.numpy(), q=q, width=width, depth=depth,
        xi_split=1.0, xi_duh=10.0, shared_ahat=True,
        C=C.numpy(), **kw).double()


def test_shared_ahat_closure_identity():
    """With the Duhamel band empty, the tau-weighted closure of the coefficients
    must equal A_hat exactly at any time, for random untrained weights — the
    structural anti-freeze property.
    """
    v, tau, C, q = _mixed_slow()
    hyb = _shared(v, tau, C, q, seed=7)
    t = torch.linspace(0.0, 4.0, 33)
    with torch.no_grad():
        c, _ = hyb.coefficients(t, v, tau, C, q)
        That = (c[..., 0] / tau.reshape(1, -1)).sum(1) / (C / tau).sum()
        ts = t.reshape(-1, 1) * hyb.gamma_kin
        A_hat = (1.0 + (1.0 - torch.exp(-ts))
                 * hyb.slow(torch.log1p(ts)))[:, 0]
    assert torch.allclose(That, A_hat, atol=1e-12)
    with torch.no_grad():
        energy = c[..., 0].sum(1)
    assert torch.allclose(energy, C.sum() * A_hat, atol=1e-12)


def test_shared_ahat_all_ce_limit_unchanged():
    """All-CE spectrum: shared_ahat must coincide with CEPNCoefficientNet (its l=0
    already has exactly this structure).
    """
    v, tau, C, q = _gray_three()
    hyb = _shared(v, tau, C, q, seed=1)
    hyb2 = _hybrid(v, tau, C, q, xi_split=1e6, seed=1)
    ref = CEPNCoefficientNet(n_modes=3, L_max=8,
                             gamma_kin=_gamma_kin(v, tau, C, q),
                             v=v.numpy(), tau=tau.numpy(), q=q,
                             width=32, depth=3).double()
    ref.trunk.load_state_dict(hyb.trunk.state_dict())
    ref.emb.load_state_dict(hyb.emb.state_dict())
    ref.slow.load_state_dict(hyb.slow.state_dict())
    hyb_allce = XiHybridPNCoefficientNet(
        n_modes=3, L_max=8, gamma_kin=_gamma_kin(v, tau, C, q),
        v=v.numpy(), tau=tau.numpy(), q=q, width=32, depth=3,
        xi_split=1e6, xi_duh=1e7, shared_ahat=True).double()
    hyb_allce.trunk.load_state_dict(ref.trunk.state_dict())
    hyb_allce.emb.load_state_dict(ref.emb.state_dict())
    hyb_allce.slow.load_state_dict(ref.slow.state_dict())
    t = torch.linspace(0.0, 3.0, 41)
    c_h, s_h = hyb_allce.coefficients(t, v, tau, C, q)
    c_r, s_r = ref.coefficients(t, v, tau, C, q)
    assert torch.allclose(c_h, c_r, atol=1e-12)
    assert torch.allclose(s_h, s_r, atol=1e-12)


def test_shared_ahat_defeats_the_freeze():
    v, tau, C, q = _mixed_slow()
    torch.manual_seed(0)
    hyb = _shared(v, tau, C, q, width=48, depth=3)
    opt = torch.optim.Adam(hyb.parameters(), lr=2e-3)
    t_coll = torch.linspace(0.0, 6.0, 241)
    loss0 = None
    for _ in range(1000):
        opt.zero_grad()
        res = pn_residual(hyb, t_coll, v, tau, C, q, ap_scaling=True)
        loss = (res ** 2).mean()
        loss.backward()
        opt.step()
        if loss0 is None:
            loss0 = float(loss)
    assert float(loss) < 0.05 * loss0
    with torch.no_grad():
        A = pn_amplitude(hyb, t_coll, tau, C, v=v, q=q).numpy()
    A_ref = solve_pn_ode(v.numpy(), tau.numpy(), C.numpy(), q=q,
                         t=t_coll.numpy(), L_max=24)
    tt = t_coll.numpy()
    g = np.trapezoid(np.clip(A, 0, None), tt)
    g_ref = np.trapezoid(np.clip(A_ref, 0, None), tt)
    ratio = g_ref / g   # integral-rate ratio ~ gamma_eff/gamma_ref
    assert 0.8 < ratio < 1.25, f"collective rate off: {ratio:.3f}"
    assert np.abs(A - A_ref).max() < 0.12


def test_shared_ahat_energy_dominance_rule():
    """Rule: active iff f_C(CE+plain) > 0.9."""
    # duh band owns 70% of the energy -> carrier must deactivate
    v = torch.tensor([0.1, 30.0])
    tau = torch.tensor([1.0, 1.0])
    C = torch.tensor([0.3, 0.7])
    q = 1.0
    hyb = _shared(v, tau, C, q, seed=11)
    assert not bool(hyb.shared_active)
    ref = _hybrid(v, tau, C, q, xi_split=1.0, xi_duh=10.0, seed=11)
    ref.load_state_dict(hyb.state_dict())
    t = torch.linspace(0.0, 3.0, 21)
    c_h, s_h = hyb.coefficients(t, v, tau, C, q)
    c_r, s_r = ref.coefficients(t, v, tau, C, q)
    assert torch.allclose(c_h, c_r, atol=1e-12)
    assert torch.allclose(s_h, s_r, atol=1e-12)
    # slow bands own all the energy -> carrier active
    v2 = torch.tensor([0.1, 3.0])
    hyb2 = _shared(v2, tau, C, q, seed=11)
    assert bool(hyb2.shared_active)


def test_macro_license_pure_ce_carrier_only():
    """The macro row is licensed only when the active carrier is pure CE (flux
    fully slaved); a plain mode in the carrier revokes it.
    """
    tau = torch.tensor([1.0, 1.0])
    C = torch.tensor([1.0, 1.0])
    q = 1.0
    pure_ce = _shared(torch.tensor([0.1, 0.5]), tau, C, q)
    assert bool(pure_ce.macro_licensed)
    with_plain = _shared(torch.tensor([0.1, 3.0]), tau, C, q)
    assert not bool(with_plain.macro_licensed)


def _mixed_slow_het():
    """Heterogeneous-tau version of the freezing configuration: two CE modes
    (xi=0.1, 0.3) + one plain mode (xi=3) with tau spanning 60x.
    """
    v = torch.tensor([2.0, 0.6, 1.0])
    tau = torch.tensor([0.05, 0.5, 3.0])
    C = torch.tensor([1.0, 1.0, 0.5])
    return v, tau, C, 1.0


def test_carrier_energy_identity_splits_roles():
    v, tau, C, q = _mixed_slow_het()
    hyb = _shared(v, tau, C, q, seed=7, carrier_energy=True)
    t = torch.linspace(0.0, 4.0, 33)
    with torch.no_grad():
        c, _ = hyb.coefficients(t, v, tau, C, q)
        ts = t.reshape(-1, 1) * hyb.gamma_kin
        A_hat = (1.0 + (1.0 - torch.exp(-ts))
                 * hyb.slow(torch.log1p(ts)))[:, 0]
        energy = c[..., 0].sum(1)
        That = (c[..., 0] / tau.reshape(1, -1)).sum(1) / (C / tau).sum()
    assert torch.allclose(energy, C.sum() * A_hat, atol=1e-12)
    assert not torch.allclose(That, A_hat, atol=1e-6)


def test_carrier_row_equals_macro_on_full_subset():
    """When the carrier subset is ALL modes (no Duhamel band), the carrier row is
    algebraically the macro row (same collision cancellation, same collective
    normalization).
    """
    v, tau, C, q = _mixed_slow_het()
    hyb = _shared(v, tau, C, q, seed=3)
    assert hyb.carrier_mask.all()
    t = torch.linspace(0.0, 6.0, 41)
    res_c, carrier = pn_residual(hyb, t, v, tau, C, q,
                                 ap_scaling=True, return_carrier=True)
    res_m, macro = pn_residual(hyb, t, v, tau, C, q,
                               ap_scaling=True, return_macro=True)
    assert torch.allclose(res_c, res_m, atol=1e-12)
    assert torch.allclose(carrier, macro, atol=1e-12)


def test_carrier_energy_off_is_v4_identical():
    v, tau, C, q = _mixed_slow_het()
    hyb_v4 = _shared(v, tau, C, q, seed=5)
    hyb_off = _shared(v, tau, C, q, seed=5, carrier_energy=False)
    t = torch.linspace(0.0, 4.0, 21)
    c4, s4 = hyb_v4.coefficients(t, v, tau, C, q)
    c0, s0 = hyb_off.coefficients(t, v, tau, C, q)
    assert torch.allclose(c4, c0, atol=1e-15)
    assert torch.allclose(s4, s0, atol=1e-15)


def test_carrier_row_closes_het_tau_miniature():
    v, tau, C, q = _mixed_slow_het()
    torch.manual_seed(0)
    hyb = _shared(v, tau, C, q, width=48, depth=3, carrier_energy=True)
    opt = torch.optim.Adam(hyb.parameters(), lr=2e-3)
    g_true = 0.1095
    t_coll = torch.linspace(0.0, 7.0 / g_true, 241)
    for _ in range(3000):
        opt.zero_grad()
        res, carrier = pn_residual(hyb, t_coll, v, tau, C, q,
                                   ap_scaling=True, return_carrier=True)
        loss = (res ** 2).mean() + (carrier ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        A = pn_amplitude(hyb, t_coll, tau, C, v=v, q=q).numpy()
    A_ref = solve_pn_ode(v.numpy(), tau.numpy(), C.numpy(), q=q,
                         t=t_coll.numpy(), L_max=24)
    tt = t_coll.numpy()
    g = np.trapezoid(np.clip(A, 0, None), tt)
    g_ref = np.trapezoid(np.clip(A_ref, 0, None), tt)
    ratio = g_ref / g
    assert 0.85 < ratio < 1.15, f"collective rate off: {ratio:.3f}"


def test_carrier_row_finite_when_carrier_empty():
    v = torch.tensor([3.0, 30.0])       # xi = 3 (plain), 30 (Duhamel)
    tau = torch.tensor([1.0, 1.0])
    C = torch.tensor([1.0, 1.0])
    q = 1.0
    hyb = _shared(v, tau, C, q, carrier_energy=True)
    assert not hyb.shared_active        # f_C(slow) = 0.5 < 0.9
    assert not bool(hyb.carrier_mask.any())   # CE band empty
    t = torch.linspace(0.0, 3.0, 17)
    res, carrier = pn_residual(hyb, t, v, tau, C, q,
                               ap_scaling=True, return_carrier=True)
    assert torch.isfinite(res).all()
    assert torch.isfinite(carrier).all()
    assert float(carrier.abs().max()) == 0.0
    loss = (res ** 2).mean() + (carrier ** 2).mean()
    loss.backward()
    for p in hyb.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
