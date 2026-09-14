""": the invariants `fig:mode_spectrum` asserts, as tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

sys.path.insert(0, str(REPO / "scripts"))
import p101_mode_spectrum as MS  # noqa: E402

def test_the_module_self_test_passes():
    assert MS.self_test()

# --------------------------------------------------------------------------- #
# as missing the k-space Jacobian dw = v dk -- as a labelled debt.  These two
# tests are what the re-pin earns: the first would have caught the debt
# becoming permanent (the drawn set silently diverging from the production
# comb), the second is the DIMENSIONAL axis the legacy weight fails by
# construction (its C is per-unit-omega, so its kappa is ~1e-37 in SI).  Same
# class as tests/test_mode_source_capacity_contract.py, applied to this
# figure's own source.
# --------------------------------------------------------------------------- #

def test_the_mode_source_is_the_production_joint_comb():
    """The drawn mode set must BE the comb production trains and scores under
    (`optical_reservoir.joint_modes`: 60 measure-corrected acoustic modes plus
    one lumped optical reservoir), bit-identical -- not a copy that agrees
    today."""
    from pinn_bte.physics.comb_routing import comb_modes

    vj, tj, Cj = comb_modes(MS.COMB, MS.NK, MS.T_REF)
    v, tau, C = MS.modes()
    assert v.shape == vj.shape, (v.shape, vj.shape)
    assert np.array_equal(v, vj)
    assert np.array_equal(tau, tj)
    assert np.array_equal(C, Cj)

def test_the_mode_source_satisfies_the_capacity_contract():
    """The dimensional axis: the weights the figure draws must imply silicon's
    own kappa and heat capacity in SI, checked against numbers from OUTSIDE
    this repo (142-156 W/(m K); rho c_p).  A ratio battery cannot see a wrong
    weight; this can, and does -- the legacy pin fails it by ~38 orders."""
    from tests.test_mode_source_capacity_contract import (SI_KAPPA_BAND,
                                                          comb_kappa_si)
    from pinn_bte.physics.optical_reservoir import SI_RHO_CP_300K

    v, tau, C = MS.modes()
    kappa = comb_kappa_si(v, tau, C)
    assert SI_KAPPA_BAND[0] <= kappa <= SI_KAPPA_BAND[1], kappa
    assert abs(float(np.sum(C)) / SI_RHO_CP_300K - 1.0) < 0.05

def test_the_shipped_kernel_is_the_angular_moment_it_is_drawn_as():
    """K0 from `ugks_resolvent_kernels` vs adaptive quadrature of
    <1/(1 + i xi mu)>_mu.  The whole figure is a picture of this kernel."""
    for L in MS.LADDER_UM:
        xi = MS.xi_of(L)
        assert np.abs(MS.kernels(xi)[0] - MS.k0_by_quadrature(xi)).max() < 1e-10

def test_the_spectrum_translates_rigidly():
    """xi_m = q v_m tau_m and only q moves with the period, so the spectrum's
    position scales exactly as 1/L and its WIDTH does not move at all.  This is
    what licenses the caption's 'translating rigidly' and the single width."""
    span = MS.xi_span_decades()
    ref = MS.xi_of(1.0)
    for L in MS.LADDER_UM:
        assert np.allclose(MS.xi_of(L) * L, ref, rtol=0, atol=1e-12)
        assert abs(np.log10(MS.xi_of(L).max() / MS.xi_of(L).min()) - span) < 1e-12

@pytest.mark.parametrize("kind", ["C", "closure"])
def test_the_composition_is_monotone_and_has_no_step(kind):
    """The claim the figure exists to make: one formula, no threshold.  A
    routing cut would show up as a jump; 400 log-spaced periods over six
    decades would not miss one."""
    Ls = np.logspace(-3, 3, 400)
    f = np.array([MS.slaved_share(L, kind) for L in Ls])
    assert np.all(np.diff(f) > 0), f"{kind}: not monotone in L"
    # largest single step, on a grid 400 points across six decades
    assert np.diff(f).max() < 0.02, f"{kind}: step {np.diff(f).max():.3f}"

@pytest.mark.parametrize("kind", ["C", "closure"])
def test_both_limits_are_reached_by_the_same_formula(kind):
    assert MS.slaved_share(1e-4, kind) < 1e-2
    assert MS.slaved_share(1e4, kind) > 1 - 1e-5

def test_the_drawn_mode_set_is_the_one_production_trains_on():
    """`build_si_case` is what every pinned run was built from.  L = 1 um only;
    see the module docstring for why one rung covers the ladder."""
    v, tau, C = MS.modes()
    v2, tau2, C2, q2 = MS.case_from_trainer(1.0)
    assert np.array_equal(v, v2) and np.array_equal(tau, tau2)
    assert np.array_equal(C, C2) and q2 == MS.q_of(1.0)

def test_the_figure_prints_what_the_manuscript_holds():
    """Every numeral the figure and the caption print, pinned to 1e-12 on the
    sum-rule mode set: a drift of the module moves the figure and this test
    together."""
    assert np.isclose(MS.xi_weighted_span_decades(), 2.1094606214363867, rtol=0, atol=1e-12)
    assert np.allclose([MS.share_above_cut(L, 10.0) for L in MS.PANEL_UM],
                       [0.2573860987730836, 0.00038232955231997073, 0.0], rtol=0, atol=1e-12)
    assert np.allclose(
        [MS.slaved_share(L) for L in MS.LADDER_UM],
        [0.20947811023955007, 0.5836986335801453, 0.7285650261997363,
         0.7909052285534739, 0.829502580887572, 0.8677237115087653,
         0.9065720060501721, 0.9937268061610998, 0.9999219581819159],
        rtol=0, atol=1e-12)
    assert np.allclose([MS.xi_geomean(L) for L in MS.PANEL_UM],
                       [1.7560089568085644, 0.1756008956808564, 0.017560089568085635],
                       rtol=0, atol=1e-12)
    assert np.isclose(MS.xi_span_decades(), 3.3694926551476887, rtol=0, atol=1e-12)
    assert np.isclose(MS.reservoir_share(), 0.42339379060254456, rtol=0, atol=1e-12)


def test_the_reservoir_share_and_its_complement_are_consistent():
    """The stem adoption : panel (a) draws the optical reservoir as a per-period STEM whose length is proportional to its share of the total heat capacity, excluded from the KDE fill, and prints that share and the acoustic complement in two annotation lines. Both numerals must be claim-owned (`pn.modespec.reservoir.share.C`) and must reconcile exactly with the joint comb the figure draws: the share IS C[reservoir]/sum(C), the reservoir IS the last mode (argmin of the MFP), and the two elements."""

    share = MS.reservoir_share()
    v, tau, C = MS.modes()
    assert int(np.argmin(v * tau)) == v.size - 1     # the reservoir is last
    assert share == float(C[-1] / C.sum())
    assert 0.42 < share < 0.43
    import make_mode_spectrum_figure as fig
    num = fig.Numbers()
    assert num.text("pn.modespec.reservoir.share.C", 0, "pct1fig") == "42.3%"
    assert num.text("pn.modespec.reservoir.share.C", 1, "pct1fig") == "57.7%"


# --------------------------------------------------------------------------- #
# shipped a FALSE SENTENCE in vector text -- "retired hard cuts, either falls
# inside the spectrum" -- and nothing could catch it: the nine tests above
# so a green gate could never evaluate an English claim.  These assert the
# CLAIMS the figure's prose makes, not just the values it prints.
# --------------------------------------------------------------------------- #

def test_a_hard_cut_lands_differently_at_different_periods():
    """The corrected sentence, as an assertion. If a future edit reinstates "either cut falls inside the spectrum at every period", this fails."""
    assert MS.cut_position(1.0, 1.0) == "interior"
    assert MS.cut_position(1.0, 10.0) == "interior"
    # at L=0.1 the cut slices the optical reservoir off the acoustic block
    assert MS.cut_position(0.1, 1.0) == "interior"
    xi01 = MS.xi_of(0.1)
    assert xi01.min() < 1.0 < np.sort(xi01)[1]     # exactly one mode below
    # at the fully ballistic period the xi=1 rule is BELOW the whole spectrum
    assert MS.cut_position(0.01, 1.0) == "below"
    assert MS.xi_of(0.01).min() > 1.0
    # at the diffusive end the xi=10 rule is ABOVE the whole spectrum
    assert MS.cut_position(10.0, 10.0) == "above"
    assert MS.xi_of(10.0).max() < 10.0

@pytest.mark.parametrize("L,cut", [(0.01, 1.0), (0.1, 1.0), (1.0, 1.0),
                                   (1.0, 10.0), (10.0, 10.0)])
def test_share_above_cut_is_strictly_interior_only_where_claimed(L, cut):
    """Per (cut, period): the weight fraction above the cut is strictly inside
    (0, 1) exactly when the cut is called interior, and pinned at an endpoint
    otherwise.  This is the invariant the four false sites violated."""
    f = MS.share_above_cut(L, cut)
    if MS.cut_position(L, cut) == "interior":
        assert 0.0 < f < 1.0
    elif MS.cut_position(L, cut) == "below":
        # the weights are normalised in float, so the "everything" end is
        # 1 - O(eps), not the literal 1.0 -- and 0.9999999999999999 IS the
        # store element that shipped under the sentence "the cut falls inside
        # the spectrum". Assert the endpoint, not a float equality.
        assert 1.0 - f < 1e-12
    else:
        assert f == 0.0

def test_the_drawn_support_does_not_outrun_the_modes():
    """The 0.13-decade drawing kernel puts visible density below the lowest real mode; on the legacy comb that smoothing crossed the xi = 1 rule at L = 0.1 um -- manufacturing evidence for the printed sentence -- so the figure clips each fill to the true mode support. This asserts the clip is what the modes say, and that the unclipped kernel still outruns them (so the test fails if someone removes the clip believing it was unnecessary). On the joint comb the support's low edge IS the optical reservoir, which really does sit below the xi = 1 rule at L = 0.1 um -- that crossing is a mode, not smoothing, and since the stem adoption  it is drawn as a per-period STEM at its exact xi rather than inside the KDE fill, which is acoustic-only and clipped to the acoustic."""
    import numpy as np
    lo, hi = MS.support(0.1)
    assert (lo, hi) == (float(MS.xi_of(0.1).min()), float(MS.xi_of(0.1).max()))
    assert lo == float(MS.xi_of(0.1)[-1])   # the optical mode is the low edge
    assert lo < 1.0                         # and it really is below the rule
    grid, dens = MS.spectral_density(0.1, grid=np.linspace(-4, 5, 1400))
    visible = 10.0 ** grid[dens > 0.01 * dens.max()]
    assert visible.min() < lo               # unclipped, the kernel outruns it

def test_the_weighted_span_is_much_narrower_than_the_mode_set_span():
    """Both widths are printed because they differ materially, and quoting
    only the unweighted one invites a reader to measure the drawn hump against
    a number too large.  Joint comb: 3.37 vs 2.12 decades, a factor 1.6 (the
    legacy comb's 2.2x shrank because the optical reservoir -- 43.1% of the
    heat capacity at the set's low edge -- pulls the weighted 5% quantile onto
    itself)."""
    assert MS.xi_weighted_span_decades() < 0.7 * MS.xi_span_decades()
    assert MS.xi_span_decades() - MS.xi_weighted_span_decades() > 1.0

def test_the_y_dependence_is_quoted_over_the_drawn_periods_not_one_rung():
    """The caveat's own numbers, on the joint comb.  At L = 0.01 um the
    y-dependent weighted mean is NEGATIVE (K0(y, xi) is a resolvent, not a
    share) -- so panel (b)'s [0,1] "share" reading and its monotonicity belong
    to the y = 0 form, and no site may size this caveat with one rung alone.
    On the legacy comb the sign flip sat at the DRAWN L = 0.1 um; the re-pin
    moved it to the undrawn L = 0.01 um, and the drawn periods now carry
    finite same-sign corrections (0.58 -> 0.56 at 0.1 um, 0.86 -> 0.92 at
    1 um)."""
    lo, mid, hi = (MS.slaved_share_ydep(L) for L in MS.PANEL_UM)
    assert MS.slaved_share_ydep(0.01) < 0.0 < MS.slaved_share(0.01)
    assert 0.0 < lo < MS.slaved_share(0.1)      # shifted down, no sign flip
    assert mid > 0.9 and hi > 0.99
    # non-monotone over the ladder: it falls from L = 0.1 to L = 0.3
    assert MS.slaved_share_ydep(0.3) < lo
