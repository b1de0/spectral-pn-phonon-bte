"""Matrix-exponential P_N solver with H harmonics: phase basis, tau(T) rate
field, H=2 reproduces the two-harmonic solver, dt Richardson.
"""
import numpy as np
import pytest

from pinn_bte.physics.ttg_pn2q import (
    _HW as _HW_2Q,
    _phase_basis as _phase_basis_2q,
    _rate_field as _rate_field_2q,
    solve_pn2q_decay,
)
from pinn_bte.physics.ttg_pnhq import (
    TauGridEdgeError,
    _hw,
    _phase_basis_h,
    _rate_field_h,
    solve_pnHq_decay,
)


# Appendix-A production config
PROD = dict(A0_K=75.0, Nk=20, L_max=8, ic_2c=-1.0, n_phi=32, n_decay=4.0,
            T_lo=140.0, T_hi=460.0, nT=160)
CHEAP = dict(A0_K=30.0, Nk=6, L_max=4, ic_2c=-1.0, n_phi=16, n_decay=2.0,
             T_lo=140.0, T_hi=460.0, nT=160)


def test_phase_basis_h2_matches_2q():
    for n_phi in (8, 16, 32):
        B2 = _phase_basis_2q(n_phi)
        BH = _phase_basis_h(n_phi, H=2)
        assert BH.shape == B2.shape
        assert np.array_equal(BH, B2), "phase basis not bit-identical at H=2"
    assert np.array_equal(_hw(2), _HW_2Q)


def test_phase_basis_h3_h4_rows_and_orthogonality():
    for H in (3, 4):
        n_phi = 64
        B = _phase_basis_h(n_phi, H)
        assert B.shape == (2 * H + 1, n_phi)
        w = _hw(H)
        # real-Fourier orthonormality under the projection weights
        G = (w[:, None] * B) @ B.T / n_phi
        assert np.allclose(G, np.eye(2 * H + 1), atol=1e-13)


def test_rate_field_h2_matches_2q_inrange():
    from pinn_bte.physics.ttg_dispersion import phonon_modes
    from pinn_bte.physics.ttg_dom import _tables

    _, tau0, _ = phonon_modes(6, 300.0)
    Tg, tauT, _, _, _ = _tables(6, 300.0, T_lo=140.0, T_hi=460.0, nT=160)
    B = _phase_basis_2q(16)
    dT_h = np.array([5.0, 30.0, -3.0, -30.0, 2.0])       # in-range drive
    for model in ("local", "frozen"):
        g2 = _rate_field_2q(dT_h, tau0, tauT, Tg, 300.0, model, B)
        gH = _rate_field_h(dT_h, tau0, tauT, Tg, 300.0, model, B)
        assert np.array_equal(gH, g2), f"rate field differs ({model})"


def test_rate_field_loud_edge_assert():
    from pinn_bte.physics.ttg_dispersion import phonon_modes
    from pinn_bte.physics.ttg_dom import _tables

    _, tau0, _ = phonon_modes(6, 300.0)
    Tg, tauT, _, _, _ = _tables(6, 300.0, T_lo=180.0, T_hi=420.0, nT=160)
    B = _phase_basis_2q(16)
    dT_cold = np.array([0.0, 75.0, 0.0, -75.0, 0.0])
    with pytest.raises(TauGridEdgeError):
        _rate_field_h(dT_cold, tau0, tauT, Tg, 300.0, "local", B)
    # frozen path never consults the table -> no raise
    _rate_field_h(dT_cold, tau0, tauT, Tg, 300.0, "frozen", B)


def test_solver_raises_on_default_grid_with_asym_drive():
    """The Appendix-A drive on the DEFAULT 180/420 grid must fail LOUDLY
    (solve_pn2q_decay silently clamps here — a silent-clamp trap).
    """
    with pytest.raises(TauGridEdgeError):
        solve_pnHq_decay(10.0, H=2, A0_K=75.0, Nk=6, L_max=4, ic_2c=-1.0,
                         n_phi=16, n_decay=1.0)          # defaults: 180/420


@pytest.mark.parametrize("tau_model", ["local", "frozen"])
def test_h2_reproduces_pn2q_bit_level(tau_model):
    g2, tr2 = solve_pn2q_decay(10.0, tau_model=tau_model, return_trace=True,
                               **CHEAP)
    gH, trH = solve_pnHq_decay(10.0, H=2, tau_model=tau_model,
                               return_trace=True, **CHEAP)
    assert gH == g2, f"gamma not bit-identical: {gH!r} vs {g2!r}"
    assert np.array_equal(trH["A"], tr2["A"]), "A(t) trace not bit-identical"
    assert np.array_equal(trH["t"], tr2["t"])
    assert np.array_equal(trH["dTh"][3][1:], tr2["dT2c"][1:])
    assert trH["dTh"][3][0] == CHEAP["ic_2c"]


def test_dt_scale_richardson():
    g1, tr1 = solve_pnHq_decay(10.0, H=2, tau_model="local",
                               return_trace=True, **CHEAP)
    gh, trh = solve_pnHq_decay(10.0, H=2, tau_model="local", dt_scale=0.5,
                               return_trace=True, **CHEAP)
    assert abs(trh["nsteps"] - 2 * tr1["nsteps"]) <= 1
    assert gh != g1                                       # dt actually moved
    assert abs(gh - g1) / g1 < 5e-3                       # O(dt) splitting error


@pytest.mark.parametrize("H", [3, 4])
def test_frozen_higher_harmonics_stay_zero(H):
    gH, tr = solve_pnHq_decay(10.0, H=H, tau_model="frozen",
                              return_trace=True, **CHEAP)
    for h in range(5, 2 * H + 1):
        assert tr["dTh"][h][0] == 0.0
        assert np.max(np.abs(tr["dTh"][h])) < 1e-12, f"row {h} leaked"
    # dt-matched consistency with H=2 (dt_scale=H/2 restores the H=2 CFL dt)
    g2, tr2 = solve_pnHq_decay(10.0, H=2, tau_model="frozen",
                               return_trace=True, **CHEAP)
    gm, trm = solve_pnHq_decay(10.0, H=H, tau_model="frozen",
                               dt_scale=H / 2.0, return_trace=True, **CHEAP)
    assert trm["nsteps"] == tr2["nsteps"], "CFL bookkeeping drifted"
    assert abs(gm - g2) / g2 < 1e-12, f"frozen H={H} != H=2 at matched dt"


def test_local_h3_carries_3q_content():
    _, tr = solve_pnHq_decay(10.0, H=3, tau_model="local",
                             return_trace=True, **CHEAP)
    assert np.max(np.abs(tr["dTh"][5])) > 1e-8, "3q cos row never populated"


@pytest.mark.slow
def test_h2_reproduces_pinned_appendix_a_targets():
    """The new instrument must reproduce the RECORDED numbers."""
    from pinn_bte.physics.ttg_dispersion import shipped_mode_source

    with shipped_mode_source():
        gl10 = solve_pnHq_decay(10.0, H=2, tau_model="local", **PROD)
        gf10 = solve_pnHq_decay(10.0, H=2, tau_model="frozen", **PROD)
        assert f"{gl10:.6e}" == "2.841392e+07"
        assert f"{gf10:.6e}" == "2.781393e+07"
        gl100 = solve_pnHq_decay(100.0, H=2, tau_model="local", **PROD)
        gf100 = solve_pnHq_decay(100.0, H=2, tau_model="frozen", **PROD)
    assert f"{gl100:.6e}" == "3.068442e+05"
    assert f"{gf100:.6e}" == "2.962037e+05"
    assert abs((gl100 - gf100) / gf100 * 100.0 - 3.5923) < 5e-4
