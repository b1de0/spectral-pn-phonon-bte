"""The whole-curve decay-rate estimator, its denominator conventions, and the
sign census of an amplitude trace.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The three amplitude transforms. Names are recorded
#: VERBATIM in artifacts, so treat them as a public vocabulary.
GAMMA_ESTIMATORS = ("absA", "clip", "raw")

GAMMA_ESTIMATOR_RULED = "absA"

GAMMA_ESTIMATOR_LEGACY = "clip"

DENOMINATOR_RULE_SAME_WINDOW = "same-window-arbiter-native"
DENOMINATOR_RULE_ARBITER_WINDOW = "arbiter-own-ndecay-window"
DENOMINATOR_RULE_RUN_GRID = "same-window-run-grid-interp"

DENOMINATOR_RULES = (DENOMINATOR_RULE_SAME_WINDOW,
                     DENOMINATOR_RULE_ARBITER_WINDOW,
                     DENOMINATOR_RULE_RUN_GRID)

#: The ruled default.
DENOMINATOR_RULE_RULED = DENOMINATOR_RULE_SAME_WINDOW
DENOMINATOR_RULE_LEGACY = DENOMINATOR_RULE_ARBITER_WINDOW


def amplitude_transform(A: np.ndarray, estimator: str) -> np.ndarray:
    """`g(A)` for the named estimator -- the integrand of `integral_gamma`."""
    if estimator == "absA":
        return np.abs(A)
    if estimator == "clip":
        return np.clip(A, 0.0, None)
    if estimator == "raw":
        return A
    raise ValueError(
        f"unknown gamma estimator {estimator!r}; expected one of "
        f"{GAMMA_ESTIMATORS}")


def integral_gamma(t: np.ndarray, A: np.ndarray,
                   estimator: str = GAMMA_ESTIMATOR_RULED) -> float:
    """gamma_eff = A(0) / int g(A) dt, with g selected by `estimator`."""
    t = np.asarray(t, dtype=float)
    A = np.asarray(A, dtype=float)
    return float(A[0] / np.trapezoid(amplitude_transform(A, estimator), t))


def _midpoint_weights(t: np.ndarray) -> np.ndarray:
    """Per-sample time measure summing EXACTLY to ``t[-1] - t[0]``."""
    t = np.asarray(t, dtype=float)
    if t.size < 2:
        return np.zeros_like(t)
    w = np.empty_like(t)
    w[0] = 0.5 * (t[1] - t[0])
    w[-1] = 0.5 * (t[-1] - t[-2])
    w[1:-1] = 0.5 * (t[2:] - t[:-2])
    return w


@dataclass(frozen=True)
class AmplitudeSignReport:
    """Did the clip bite on this trace, and by how much?"""

    sign_change: bool
    A_min: float
    n_negative: int
    n_samples: int
    #: denominator = window length t[-1]-t[0] (midpoint-interval measure)
    negative_time_fraction: float
    #: denominator = sample count len(t)
    negative_sample_fraction: float
    gamma_absA: float
    gamma_clip: float
    #: (gamma_clip - gamma_absA) / gamma_absA — denominator in the name
    clip_minus_absA_rel: float


def amplitude_sign_report(t: np.ndarray, A: np.ndarray) -> AmplitudeSignReport:
    """Sign census of A over its scoring window + both estimator readings."""
    t = np.asarray(t, dtype=float)
    A = np.asarray(A, dtype=float)
    neg = A < 0.0
    w = _midpoint_weights(t)
    span = float(t[-1] - t[0])
    g_abs = integral_gamma(t, A, estimator="absA")
    g_clip = integral_gamma(t, A, estimator="clip")
    return AmplitudeSignReport(
        sign_change=bool(neg.any()),
        A_min=float(A.min()),
        n_negative=int(neg.sum()),
        n_samples=int(A.size),
        negative_time_fraction=(float(w[neg].sum() / span) if span > 0.0
                                else 0.0),
        negative_sample_fraction=float(neg.sum() / A.size),
        gamma_absA=g_abs,
        gamma_clip=g_clip,
        clip_minus_absA_rel=float((g_clip - g_abs) / g_abs),
    )


def denominator_provenance(estimator: str, rule: str, t_end_s: float,
                           n_samples: int, gamma_ref_hz: float,
                           **extra) -> dict:
    """The block an artefact must carry so its denominator is RECONSTRUCTIBLE."""
    if estimator not in GAMMA_ESTIMATORS:
        raise ValueError(f"unknown gamma estimator {estimator!r}; expected one "
                         f"of {GAMMA_ESTIMATORS}")
    if rule not in DENOMINATOR_RULES:
        raise ValueError(f"unknown denominator rule {rule!r}; expected one of "
                         f"{DENOMINATOR_RULES}")
    return dict(gamma_ref_estimator=estimator, gamma_ref_rule=rule,
                gamma_ref_t_end_s=float(t_end_s),
                gamma_ref_n_samples=int(n_samples),
                gamma_ref_hz=float(gamma_ref_hz), **extra)
