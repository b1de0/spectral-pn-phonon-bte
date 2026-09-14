"""The whole-curve decay-rate estimator is |A|, not clip(A,0)."""
from pathlib import Path

import numpy as np
import pytest

from pinn_bte.physics.decay_metrics import (
    GAMMA_ESTIMATORS,
    GAMMA_ESTIMATOR_RULED,
    amplitude_sign_report,
    integral_gamma,
)

REPO = Path(__file__).resolve().parents[1]

# Ladder-resolved (data/ladders.yaml corridor 1d arms), NEVER newest-on-disk.
PINNED_1D = {
    "0.01": "ladder_L0.01_1d",
    "0.1": "ladder_L0.1_1d",
    "0.3": "ladder_L0.3_1d",
    "0.5": "ladder_L0.5_1d",
    "0.7": "ladder_L0.7_1d",
    "1.0": "ladder_L1.0_1d",
    "1.5": "ladder_L1.5_1d",
    "10.0": "ladder_L10.0_1d",
    "100.0": "ladder_L100.0_1d",
}

#: The one period whose amplitude changes sign inside its own scoring window
#: (the DOM reference itself crosses there).
MOVERS = ("0.01",)


def _npz(key: str):
    arm = PINNED_1D[key]
    hits = sorted((REPO / "data/runs/pn_corridor_ladder_60k" / arm).glob(
        "*_results.npz"))
    assert len(hits) == 1, hits
    return hits[0]


NEED_RUNS = not all(
    (REPO / "data/runs/pn_corridor_ladder_60k" / a).exists()
    for a in PINNED_1D.values())
pinned = pytest.mark.skipif(NEED_RUNS,
                            reason="archived 1D ladder not on this machine")


# ============================================== (a) the ruled default is |A|
def test_the_ruled_default_estimator_is_absA():
    """Asserted on the CONSTANT and on the behaviour, so a silent flip of either
    is caught.
    """
    assert GAMMA_ESTIMATOR_RULED == "absA"
    assert set(GAMMA_ESTIMATORS) == {"absA", "clip", "raw"}
    t = np.array([0.0, 1.0, 2.0])
    A = np.array([2.0, 1.0, -1.0])
    assert integral_gamma(t, A) == integral_gamma(t, A, estimator="absA")
    assert integral_gamma(t, A) != integral_gamma(t, A, estimator="clip")


def test_an_unknown_estimator_raises_rather_than_defaulting():
    t = np.array([0.0, 1.0])
    A = np.array([1.0, 0.5])
    with pytest.raises(ValueError, match="estimator"):
        integral_gamma(t, A, estimator="abs")          # near-miss spelling


# ======================== (c)+(d) the arithmetic, pinned by hand on 3 points
def test_all_three_estimators_agree_bitwise_on_a_sign_definite_trace():
    """THE EQUIVALENCE THAT MAKES SIX OF NINE RUNGS IMMUNE. A >= 0 everywhere =>
    |A| == max(A,0) == A pointwise, so the three integrals are the SAME
    floating-point sum.
    """
    t = np.array([0.0, 1.0, 2.0])
    A = np.array([2.0, 1.0, 0.5])
    g = {e: integral_gamma(t, A, estimator=e) for e in GAMMA_ESTIMATORS}
    assert g["absA"] == g["clip"] == g["raw"]
    assert g["absA"] == 2.0 / 2.25


def test_the_three_estimators_split_on_a_sign_changing_trace():
    """t = [0, 1, 2], A = [2, 1, -1] (A(0) = 2)"""
    t = np.array([0.0, 1.0, 2.0])
    A = np.array([2.0, 1.0, -1.0])
    assert integral_gamma(t, A, estimator="clip") == 1.0
    assert integral_gamma(t, A, estimator="absA") == 0.8
    assert integral_gamma(t, A, estimator="raw") == pytest.approx(4.0 / 3.0)
    # ordering is structural, not incidental: area_raw <= area_clip <= area_abs
    assert (integral_gamma(t, A, "raw") > integral_gamma(t, A, "clip")
            > integral_gamma(t, A, "absA"))


# =========================================== (e) the sign report, by hand
def test_sign_report_arithmetic_is_hand_computable():
    """t = [0, 1, 2, 3, 4], A = [1, 1, -1, 1, 1]"""
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    A = np.array([1.0, 1.0, -1.0, 1.0, 1.0])
    r = amplitude_sign_report(t, A)

    assert r.sign_change is True
    assert r.n_negative == 1 and r.n_samples == 5
    assert r.A_min == -1.0
    assert r.negative_time_fraction == 0.25
    assert r.negative_sample_fraction == 0.2
    assert r.gamma_clip != r.gamma_absA


def test_sign_report_on_a_sign_definite_trace_is_inert():
    t = np.array([0.0, 1.0, 2.0])
    A = np.array([2.0, 1.0, 0.5])
    r = amplitude_sign_report(t, A)
    assert r.sign_change is False
    assert r.n_negative == 0
    assert r.negative_time_fraction == 0.0
    assert r.negative_sample_fraction == 0.0
    assert r.gamma_clip == r.gamma_absA           # bitwise
    assert r.clip_minus_absA_rel == 0.0


# ===================== (b) the archived ladder: own-estimator is byte-exact
@pinned
@pytest.mark.parametrize("key", list(PINNED_1D))
def test_own_estimator_reproduces_the_stored_gamma_eff_bit_exactly(key):
    """THE REGRESSION GUARD."""
    d = np.load(_npz(key), allow_pickle=True)
    est = str(d["gamma_estimator"])
    got = integral_gamma(d["t"], d["A"], estimator=est)
    assert got == float(d["gamma_eff"]), (
        f"L={key}um: {est} {got!r} != stored gamma_eff "
        f"{float(d['gamma_eff'])!r} — the archived artifact is no longer "
        f"reproducible by its own estimator")


@pinned
@pytest.mark.parametrize("key", [k for k in PINNED_1D if k not in MOVERS])
def test_absA_is_bit_identical_to_clip_where_the_trace_never_turns(key):
    """Eight of nine periods: A_min > 0, so the estimator choice moves NOTHING
    there. Measured, not assumed — that is the whole point of enumerating them.
    """
    d = np.load(_npz(key), allow_pickle=True)
    A = d["A"]
    assert A.min() > 0.0, f"L={key}um is not sign-definite: A_min={A.min()!r}"
    assert (integral_gamma(d["t"], A, estimator="absA")
            == integral_gamma(d["t"], A, estimator="clip")
            == float(d["gamma_eff"]))


@pinned
@pytest.mark.parametrize("key", MOVERS)
def test_the_estimators_split_on_the_sign_changing_period(key):
    """At the ballistic period the trace crosses zero, so clip and |A| are two
    different functionals there — and the clip one is FASTER (it deletes
    anti-phase area from the denominator).
    """
    d = np.load(_npz(key), allow_pickle=True)
    A = d["A"]
    assert A.min() < 0.0, f"L={key}um no longer changes sign"
    g_abs = integral_gamma(d["t"], A, estimator="absA")
    g_clip = integral_gamma(d["t"], A, estimator="clip")
    assert g_abs == float(d["gamma_eff"])      # the stored convention is |A|
    assert g_clip > g_abs


@pinned
def test_the_measured_sign_census_of_the_pinned_ladder():
    """The census itself, so a future artifact swap is caught: exactly one of nine
    periods changes sign, and the negative fractions are the measured ones.
    """
    census = {}
    for key in PINNED_1D:
        d = np.load(_npz(key), allow_pickle=True)
        census[key] = amplitude_sign_report(d["t"], d["A"])

    assert {k for k, r in census.items() if r.sign_change} == set(MOVERS)
    assert census["0.01"].n_negative == 258 and census["0.01"].n_samples == 1200
    # time measure (denominator = window length t[-1]-t[0]), as measured
    assert census["0.01"].negative_time_fraction == pytest.approx(0.215179,
                                                                  abs=5e-7)


# ============================= (f) the reference's own reference trace is SIGNED
@pinned
def test_the_stored_reference_trace_is_signed():
    """THE PREMISE, ASSERTED. `ttg_amplitude_curve` returns 'A signed and
    normalised' (its own docstring) — the abs lives only inside
    `ttg_decay_spectral`.
    """
    signed = {k for k in PINNED_1D
              if np.load(_npz(k), allow_pickle=True)["A_ref"].min() < 0.0}
    assert signed == {"0.01"}, signed
