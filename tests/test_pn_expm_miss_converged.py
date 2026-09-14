"""Tests for the C22 full re-pin machinery (converged-dt expm-miss Nk sweep)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pn_expm_miss_converged import (  # noqa: E402
    ARM_DEPTHS,
    AUDIT_GAMMAS,
    CROSSING_REL_TOL,
    NK_ALL,
    RATIO_MAX,
    arm_limit,
    build_job_list,
    frozen_richardson,
    ladder_depth_gate,
    miss_bracket,
    monotone_tail_start,
    run_rung,
    two_tail_bracket,
    two_term_limit,
)

H135_L100_LOCAL = [3.068442e5, 3.066796e5, 3.063990e5, 3.060790e5, 3.057592e5,
                   3.054664e5, 3.052239e5, 3.050451e5, 3.049280e5, 3.048584e5,
                   3.048200e5]
H135_L100_FROZEN = [2.962037e5, 2.963734e5, 2.964594e5, 2.965028e5, 2.965246e5,
                    2.965355e5, 2.965409e5, 2.965437e5, 2.965450e5, 2.965457e5]
H135_L10_LOCAL = [2.841392e7, 2.847637e7, 2.849465e7, 2.849447e7, 2.848907e7,
                  2.848394e7, 2.848048e7, 2.847846e7, 2.847737e7, 2.847680e7]
H135_L10_FROZEN = [2.781393e7, 2.790363e7, 2.794909e7, 2.797197e7, 2.798345e7,
                   2.798920e7, 2.799208e7, 2.799352e7, 2.799424e7]

# REAL campaign ladders (data/runs/logs/c22_repin_outputs), L=100 symmetric-control
# arm.  This arm does NOT converge: its increment CROSSES ZERO where the O(dt)
# and the first-order drift then RESUMES, negative.  At Nk>=30 the crossing has
# already happened inside the campaign ladder (last increment negative); at
# Nk=20 it happens exactly at the finest rung, where the last increment is a
# near-zero +0.0029 Hz -- 267x smaller than its predecessor.
# NOTE: full double precision on purpose -- the last two Nk=20 rungs differ by
# 2.9e-3 Hz, which a %.7e round-trip destroys (it fakes a zero increment).
CAMP_L100_SYM_NK160 = [368432.75773663097, 368629.1231920244, 368726.9749581855,
                       368774.52493846056, 368796.54250993935,
                       368805.79000205704, 368808.9740009342,
                       368809.6270176131, 368809.4708074283]
CAMP_L100_SYM_NK20 = [297488.9946019481, 297661.49054336856, 297747.9383482338,
                      297790.17786394013, 297809.9165728963,
                      297818.37918613123, 297821.45573938836,
                      297822.23921768414, 297822.24215351086]
CAMP_L100_SYM_NK30 = [323524.3390488102, 323705.6467279371, 323796.3070936151,
                      323840.510873251, 323861.094772634, 323869.8509532679,
                      323872.97063107474, 323873.7086547232, 323873.65466716886]
# L=10 symmetric control: a HEALTHY first-order ladder that merely wobbles just
# ABOVE ratio 2 (r_last = 2.0035, the largest benign ratio in the whole
# 42-ladder campaign).  The domain guard must not fire here.
CAMP_L10_SYM_NK20 = [27898301.280829653, 27988299.432989992, 28033770.66378268,
                     28056544.001181733, 28067893.501483236,
                     28073540.795376487, 28076352.3548861,
                     28077753.994912412, 28078453.578440398]
# L=10 asymmetric (headline) arm, campaign depth ds=1/512.
CAMP_L10_LOCAL_NK20 = [28413919.942600965, 28476370.34092232, 28494652.739209894,
                       28494471.734827958, 28489066.19632993,
                       28483943.398720324, 28480479.72531254,
                       28478459.937324703, 28477367.37018336,
                       28476798.818814293]

# data/runs/logs/c22_audit_rungs/).
EXT_L100_SYM_NK20_DS512 = AUDIT_GAMMAS[(100.0, 20, "localsym", 512)]
EXT_L100_SYM_NK30_DS512 = AUDIT_GAMMAS[(100.0, 30, "localsym", 512)]
AUDIT_L10_LOCAL_NK20_DS1024 = AUDIT_GAMMAS[(10.0, 20, "local", 1024)]
# The SUPERSEDED two-tail band the old code returned for CAMP_L100_SYM_NK20.
SUPERSEDED_SYM_NK20_BRACKET = (297822.24216455326, 297822.2450893376)

def test_two_tail_bracket_exact_on_geometric_sequence():
    """A perfectly first-order sequence (increments halving) has a bracket of
    zero width at the true limit."""
    g_inf, K = 5.0e5, 1.0e3
    gammas = [g_inf - K * 0.5 ** n for n in range(8)]
    lo, hi, mid, half = two_tail_bracket(gammas)
    assert abs(mid - g_inf) < 1e-6 * g_inf
    assert half < 1e-9 * g_inf
    assert lo <= mid <= hi

def test_two_tail_bracket_reproduces_h135_l100_local():
    lo, hi, mid, half = two_tail_bracket(H135_L100_LOCAL)
    # published: [3.047726e5, 3.047816e5] (rounded to 0.1 Hz)
    assert lo == pytest.approx(3.047726e5, abs=1.0)
    assert hi == pytest.approx(3.047816e5, abs=1.0)
    assert half == pytest.approx((hi - lo) / 2)

def test_monotone_tail_skips_l10_sign_change():
    incs = list(np.diff(H135_L10_LOCAL))
    start = monotone_tail_start(incs)
    # increments: +62450, +18280, -180, -5400, -5130, -3460, -2020, -1090, -570
    # the shrinking same-sign suffix begins at the -5400 entry (index 3)
    assert start == 3

def test_two_tail_bracket_reproduces_h135_l10_local():
    lo, hi, mid, half = two_tail_bracket(H135_L10_LOCAL)
    # published: [2.847618e7, 2.847623e7]
    assert lo == pytest.approx(2.847618e7, abs=100.0)
    assert hi == pytest.approx(2.847623e7, abs=100.0)

def test_frozen_richardson_reproduces_h135_limits():
    assert frozen_richardson(H135_L100_FROZEN) == pytest.approx(2.965464e5, abs=1.0)
    assert frozen_richardson(H135_L10_FROZEN) == pytest.approx(2.799496e7, abs=100.0)

def test_miss_bracket_reproduces_h135_repin_candidates():
    """The published converged misses: +2.7755 +/- 0.0015 % (L=100),
    +1.7190 +/- 0.0001 % (L=10)."""
    m_lo, m_hi, m_mid, m_half = miss_bracket(H135_L100_LOCAL, H135_L100_FROZEN)
    assert m_mid == pytest.approx(2.7755, abs=0.002)
    assert m_half == pytest.approx(0.0015, abs=0.001)
    m_lo, m_hi, m_mid, m_half = miss_bracket(H135_L10_LOCAL, H135_L10_FROZEN)
    assert m_mid == pytest.approx(1.7190, abs=0.001)

def test_arm_limit_prefers_two_tail_and_is_byte_identical_to_it():
    """Extrapolation is PREFERRED wherever an in-domain shrinking tail exists:
    on every healthy arm, arm_limit must return the two_tail_bracket numbers
    unchanged."""
    for gammas in (H135_L100_LOCAL, H135_L100_FROZEN, H135_L10_LOCAL,
                   H135_L10_FROZEN, CAMP_L10_SYM_NK20, CAMP_L10_LOCAL_NK20):
        lo, hi, mid, half, method = arm_limit(gammas)
        assert method == "two-tail"
        assert (lo, hi, mid, half) == two_tail_bracket(gammas)

# ------------------------------------------------------------------ C2 fix
def test_two_tail_bracket_refuses_ratio_outside_its_domain():
    """C2.  The two-tail bracket IS the statement 'the increment ratio stays
    in [r_last, 2]'.  When r_last > 2 that interval is inverted and the band
    collapses onto the finest rung -- the zero-crossing signature.  Measured
    on the real L=100 Nk=20 symmetric ladder: r_last = 266.9, and the band it
    used to return EXCLUDED the truth by 108 half-widths (see the
    extended-ladder test below).  It must refuse."""
    incs = np.diff(CAMP_L100_SYM_NK20)
    assert incs[-2] / incs[-1] == pytest.approx(266.87, rel=1e-3)
    with pytest.raises(ValueError, match="ratio"):
        two_tail_bracket(CAMP_L100_SYM_NK20)

def test_two_tail_bracket_domain_guard_leaves_benign_ratios_alone():
    """The guard must not over-tighten: a genuine first-order ladder wobbles
    about 2 from BOTH sides.  L=10 localsym (r_last = 2.0035) is the largest
    benign ratio in the whole 42-ladder campaign; RATIO_MAX keeps a margin
    over it and still refuses the crossing ladder 100x above."""
    incs = np.diff(CAMP_L10_SYM_NK20)
    r_benign = incs[-2] / incs[-1]
    assert 2.0 < r_benign < RATIO_MAX
    lo, hi, mid, half, method = arm_limit(CAMP_L10_SYM_NK20)
    assert method == "two-tail"
    assert lo < mid < hi

def test_extended_l100_sym_ladder_falsifies_the_saturation_claim():
    """C2 REGRESSION (physicist sign-off sec.4). The L=100 localsym ladder was justified as 'already converged to its floor'. It is NOT: the increment crosses zero and the first-order drift RESUMES,."""
    for camp, deeper, blowup in ((CAMP_L100_SYM_NK20, EXT_L100_SYM_NK20_DS512, 50.0),
                                 (CAMP_L100_SYM_NK30, EXT_L100_SYM_NK30_DS512, 3.5)):
        incs = np.diff(camp + [deeper])
        assert incs[-3] > 0 and incs[-1] < 0, "the increment must cross zero"
        # the drift RESUMES: the post-crossing increment is far LARGER
        assert abs(incs[-1]) > blowup * abs(incs[-2])
        # ... so |inc_last| is NOT a converged floor: a 'saturated' band built
        # from the campaign's last increments would keep shrinking while the
        # true residual grows.
        assert abs(incs[-1]) / abs(camp[-1]) > 5e-7

    # the SUPERSEDED two-tail band excluded the truth by >100 half-widths
    old_lo, old_hi = SUPERSEDED_SYM_NK20_BRACKET
    half = 0.5 * (old_hi - old_lo)
    assert EXT_L100_SYM_NK20_DS512 < old_lo
    assert (old_lo - EXT_L100_SYM_NK20_DS512) / half == pytest.approx(108.5, abs=0.5)

def test_crossing_band_contains_the_deeper_rung():
    """The replacement treatment must be HONEST where the old one was not:
    the band computed from the campaign ladder alone must contain the fresh
    ds=1/512 rung that the old two-tail band excluded."""
    lo, hi, mid, half, method = arm_limit(CAMP_L100_SYM_NK20)
    assert method == "crossing"
    assert lo <= EXT_L100_SYM_NK20_DS512 <= hi
    lo30, hi30, _, _, method30 = arm_limit(CAMP_L100_SYM_NK30)
    assert method30 == "crossing"
    assert lo30 <= EXT_L100_SYM_NK30_DS512 <= hi30

def test_crossing_band_is_a_model_spread_not_a_floor_claim():
    """On the EXTENDED ladder the band is the outward hull of the admissible
    continuations (finest rung / p=1 tail / 2-term A*x+B*x^2 elimination /
    twice that tail) -- a spread over models, never a claim that the ladder
    converged.  It is centred on the 2-term limit with half-width = the whole
    2-term tail."""
    ext = CAMP_L100_SYM_NK20 + [EXT_L100_SYM_NK20_DS512]
    lo, hi, mid, half, method = arm_limit(ext)
    assert method == "crossing"
    assert hi == ext[-1]                       # no continuation goes upward
    assert mid == pytest.approx(two_term_limit(ext))
    assert half == pytest.approx(abs(ext[-1] - two_term_limit(ext)))
    assert lo < ext[-1] + (ext[-1] - ext[-2]) < hi   # p=1 tail lies inside
    assert half > 0.1                          # NOT the 0.0015 Hz of the old band

def test_crossing_band_survives_leaving_the_finest_rung_out():
    """Honesty test in the physicist's own style (sec.3.3a), adapted to the
    crossing regime: the band from the SHALLOWER ladder must contain the
    deeper ladder's limit estimate.  A 3-model hull FAILS this (the anchor
    moves by a whole increment per halving while the modelled tail barely
    changes, so the deeper limit escapes by 1.3 half-widths) -- which is why
    the band carries the 2x-tail candidate.  The interval itself is NOT
    claimed to nest; the limit ESTIMATE is what must stay contained."""
    ext = CAMP_L100_SYM_NK20 + [EXT_L100_SYM_NK20_DS512]
    lo_c, hi_c, mid_c, half_c, _ = arm_limit(CAMP_L100_SYM_NK20)
    lo_e, hi_e, mid_e, half_e, _ = arm_limit(ext)
    assert lo_c <= mid_e <= hi_c, "shallower band must contain the deeper limit"
    assert lo_c <= ext[-1] <= hi_c, "... and the deeper rung itself"
    # the 3-model hull would NOT have: its low end was 1.29 half-widths short
    hull3_lo = min(CAMP_L100_SYM_NK20[-1],
                   2 * CAMP_L100_SYM_NK20[-1] - CAMP_L100_SYM_NK20[-2],
                   two_term_limit(CAMP_L100_SYM_NK20))
    assert hull3_lo > mid_e

def test_two_term_limit_is_exact_on_a_quadratic_ladder():
    """gamma(x) = g_inf + A x + B x^2 sampled at x = 4h, 2h, h is inverted
    exactly -- including the opposite-sign case that produces the crossing."""
    g_inf, A, B, h = 3.0e5, 0.4, -0.05, 1.0
    g = [g_inf + A * x + B * x * x for x in (4 * h, 2 * h, h)]
    assert two_term_limit(g) == pytest.approx(g_inf, abs=1e-9)

def test_arm_limit_still_refuses_when_unconverged_and_no_tail():
    """'s lesson stays enforced: a ladder that is neither extrapolable NOR in a small-residual crossing must RAISE, never be silently 'limited'."""
    with pytest.raises(ValueError):
        arm_limit([1.0, 2.0, 4.0, 8.0])                # growing increments
    # shrinking-then-growing, last increment 10% of the value: drift too big
    with pytest.raises(ValueError):
        arm_limit([100.0, 110.0, 114.0, 130.0])

def test_crossing_fallback_needs_both_no_bracket_and_small_drift():
    """The fallback is gated on the DRIFT, not merely on the refusal."""
    g = [1.0, 2.0, 4.0, 8.0]
    assert abs(g[-1] - g[-2]) / abs(g[-1]) > CROSSING_REL_TOL
    with pytest.raises(ValueError):
        arm_limit(g)

def test_all_l100_sym_arms_get_one_treatment():
    """After the C2 fix no sweep mixes treatments: every L=100 symmetric arm
    is `crossing` (they all sit in the same cancellation regime), so the
    Nk-sweep column is internally uniform."""
    for ladder in (CAMP_L100_SYM_NK20, CAMP_L100_SYM_NK30, CAMP_L100_SYM_NK160):
        assert arm_limit(ladder)[4] == "crossing"

def test_l10_extended_local_bracket_nests_inside_the_campaign_one():
    """Durable capture of the physicist's third fresh solve
    (L=10/Nk=20/local/ds=1/1024): the headline bracket is HONEST -- deepening
    the ladder lands strictly INSIDE it, and the increment ratio climbs
    towards 2 without ever crossing it (no cancellation on this arm)."""
    camp, ext = CAMP_L10_LOCAL_NK20, CAMP_L10_LOCAL_NK20 + [AUDIT_L10_LOCAL_NK20_DS1024]
    lo_c, hi_c, mid_c, _ = two_tail_bracket(camp)
    lo_e, hi_e, mid_e, half_e = two_tail_bracket(ext)
    assert lo_c <= lo_e and hi_e <= hi_c            # nested => honest
    assert (hi_c - lo_c) > 3.5 * (hi_e - lo_e)      # and 4x tighter
    r_c = np.diff(camp)[-2] / np.diff(camp)[-1]
    r_e = np.diff(ext)[-2] / np.diff(ext)[-1]
    assert 1.92 < r_c < r_e < 2.0
    assert r_c == pytest.approx(1.92167, abs=1e-4)
    assert r_e == pytest.approx(1.96010, abs=1e-4)
    # the quoted headline moves by +0.000024 pp (0.28 half-widths, UPWARD) --
    # i.e. the campaign band contains it, and the mid is low-biased as the
    # sign-off says.  +1.7190 +/- 0.0001 stands.
    frozen = 27994961.887886                        # gamma_frozen(dt->0), Nk=20
    miss_c = (mid_c - frozen) / frozen * 100.0
    miss_e = (mid_e - frozen) / frozen * 100.0
    half_c = (hi_c - lo_c) / 2.0 / frozen * 100.0
    assert miss_c == pytest.approx(1.719038, abs=2e-6)
    assert miss_e == pytest.approx(1.719062, abs=2e-6)
    assert 0 < miss_e - miss_c < half_c

def test_two_tail_bracket_requires_shrinking_tail():
    with pytest.raises(ValueError):
        two_tail_bracket([1.0, 2.0, 4.0, 8.0])  # growing increments
    with pytest.raises(ValueError):
        two_tail_bracket([1.0, 2.0])  # too short

def test_job_list_uniform_across_nk():
    """: every Nk gets the IDENTICAL ladder (same L, arm, ds set)."""
    jobs = build_job_list()
    per_nk = {}
    for j in jobs:
        per_nk.setdefault(j["Nk"], set()).add((j["L_um"], j["arm"], j["ds_den"]))
    assert set(per_nk) == set(NK_ALL)
    ladders = list(per_nk.values())
    assert all(lad == ladders[0] for lad in ladders)
    l100 = {(arm, max(d for (L, a, d) in ladders[0] if L == 100.0 and a == arm))
            for arm in ("local", "frozen", "localsym")}
    # L=100 localsym extended 256 -> 512 by the C1 fix, at ALL SEVEN Nk
    assert l100 == {("local", 1024), ("frozen", 512), ("localsym", 512)}
    l10 = {(arm, max(d for (L, a, d) in ladders[0] if L == 10.0 and a == arm))
           for arm in ("local", "frozen", "localsym")}
    assert l10 == {("local", 512), ("frozen", 256), ("localsym", 256)}

def test_arm_depths_cover_both_lengths():
    assert set(ARM_DEPTHS) == {100.0, 10.0}

def test_ladder_depth_gate_catches_a_short_or_ragged_ladder():
    """C1's failure mode, mechanised: the L=100 symmetric miss was quoted off
    a ladder that stopped exactly ON the increment's zero-crossing.  The
    report must refuse to assemble unless EVERY (L, Nk, arm) ladder is the
    complete dyadic set 1..ARM_DEPTHS -- no shallower (under-resolved) and no
    deeper (non-uniform across Nk)."""
    full = {(L, nk, arm): {2 ** i: 0.0 for i in range(int(np.log2(dep)) + 1)}
            for L, arms in ARM_DEPTHS.items() for arm, dep in arms.items()
            for nk in NK_ALL}
    assert ladder_depth_gate(full)["pass"]

    short = {k: dict(v) for k, v in full.items()}
    short[(100.0, 20, "localsym")].pop(512)                 # the C1 ladder
    g = ladder_depth_gate(short)
    assert not g["pass"] and g["violations"] == [["100/Nk20/localsym", 256, 512]]

    deep = {k: dict(v) for k, v in full.items()}
    deep[(10.0, 20, "local")][1024] = 0.0                   # non-uniform vs Nk>=30
    assert not ladder_depth_gate(deep)["pass"]

    ragged = {k: dict(v) for k, v in full.items()}
    ragged[(10.0, 30, "frozen")].pop(16)                    # hole in the middle
    assert not ladder_depth_gate(ragged)["pass"]

def test_audit_gammas_are_the_ones_the_tests_pin():
    """`runs/` is gitignored, so AUDIT_GAMMAS + the inline ladders ARE the
    durable record of the three fresh sign-off solves.  Keep the two in
    lockstep."""
    assert set(AUDIT_GAMMAS) == {(100.0, 20, "localsym", 512),
                                 (100.0, 30, "localsym", 512),
                                 (10.0, 20, "local", 1024)}
    assert EXT_L100_SYM_NK20_DS512 == 297822.08355874615
    assert EXT_L100_SYM_NK30_DS512 == 323873.45365197206
    assert AUDIT_L10_LOCAL_NK20_DS1024 == 28476508.75633468

def test_audit_gammas_match_the_stored_rungs():
    """Opportunistic cross-check: every AUDIT_GAMMAS entry that HAS a rung
    JSON on this machine must match it BIT-EXACTLY (the solver is
    deterministic).  Skipped on a fresh clone -- `runs/` is gitignored."""
    repo = Path(__file__).resolve().parents[1]
    seen = 0
    for (L, nk, arm, den), gamma in AUDIT_GAMMAS.items():
        for sub in ("c22_repin_outputs", "c22_ext_localsym", "c22_audit_rungs"):
            f = repo / "runs" / "logs" / sub / f"rung_L{L:g}_Nk{nk}_{arm}_ds{den}.json"
            if f.exists():
                assert json.load(open(f))["gamma"] == gamma, f
                seen += 1
    if not seen:
        pytest.skip("no local campaign artifacts (runs/ is gitignored)")

@pytest.mark.slow
def test_run_rung_provenance_and_pinned_gamma(tmp_path):
    """A ds=1 rung at Nk=20/L=10 must record every parameter (provenance rule)
    and reproduce the pinned Appendix-A gamma_local bit-for-bit."""
    # bare `default` is the measure-only acoustic comb (gamma 58591763.37,
    out = run_rung(L_um=10.0, nk=20, arm="local", ds_den=1, out_dir=tmp_path,
                   mode_source="shipped")
    d = json.load(open(out))
    # T_ref and ic_2s are solver DEFAULTS, not CFG entries -- record them too,
    for key in ("L_um", "Nk", "arm", "ds_den", "ds", "gamma", "n_decay", "H",
                "n_phi", "ic_2c", "ic_2s", "A0_K", "L_max", "T_ref", "T_lo",
                "T_hi", "nT", "tau_model", "dt", "nsteps", "wall_s"):
        assert key in d, f"provenance missing: {key}"
    # BIT-EXACT against the campaign rung (data/runs/logs/c22_repin_outputs) AND
    # against the pinned old-dt path: passing T_ref/ic_2s explicitly must not
    # perturb gamma by one ulp.
    assert d["gamma"] == 28413919.942600965
    assert d["ic_2c"] == -1.0 and d["tau_model"] == "local"
    assert (d["T_lo"], d["T_hi"], d["nT"]) == (140.0, 460.0, 160)
    assert (d["T_ref"], d["ic_2s"]) == (300.0, 0.0)
