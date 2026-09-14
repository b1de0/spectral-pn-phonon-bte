"""Provenance of the decay-rate estimator: sign census, npz labels and the
battery's dispatch on the stored estimator.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from pn_shape_battery import gate, npz_estimator  # noqa: E402

from pinn_bte.physics.decay_metrics import (
    GAMMA_ESTIMATOR_RULED,
    amplitude_sign_report,
    integral_gamma,
)
from pinn_bte.training.transient.spectral_pn_trainer import (
    SpectralPNConfig,
    _sign_provenance,
    build_si_case,
    train_spectral_pn,
)

pytestmark = pytest.mark.usefixtures("default_float64")

SIGN_KEYS = ("gamma_estimator", "gamma_eff_clip", "gamma_clip_minus_absA_rel",
             "amp_sign_change", "amp_A_min", "amp_n_negative", "amp_n_samples",
             "amp_negative_time_frac", "amp_negative_sample_frac")


@pytest.fixture(scope="module")
def case():
    return build_si_case(10.0, Nk=6, n_decay_times=6.0, geometry="1d")


def _cfg_1q(case, **kw):
    base = dict(
        v=case.v, tau=case.tau, C=case.C, q=case.q, t_end=case.t_end,
        L_max=4, width=16, depth=2, emb_dim=4, epochs=8, lr=2e-3,
        lr_cosine=True, nt=60, t_grid="uniform", seed=42, device="cpu",
        dtype="float64", t_ref=case.t_ref, A_ref=case.A_ref,
        gamma_ref=case.gamma_dom_hz, Nk=6, T_ref=300.0, log_every=1,
        formulation="xihybrid", shared_ahat=True,
        carrier_energy=True, carrier_row=True, xi_split=1.0, xi_duh=10.0,
    )
    base.update(kw)
    return SpectralPNConfig(**base)


def test_sign_provenance_fires_on_a_sign_changing_trace():
    """t = [0,1,2], A = [2,1,-1]: clip area 2.0 -> 1.0, |A| area 2.5 -> 0.8."""
    t = np.array([0.0, 1.0, 2.0])
    A = np.array([2.0, 1.0, -1.0])
    p = _sign_provenance(t, A)

    assert p["gamma_estimator"] == GAMMA_ESTIMATOR_RULED == "absA"
    assert p["amp_sign_change"] is True
    assert p["amp_A_min"] == -1.0
    assert p["amp_n_negative"] == 1 and p["amp_n_samples"] == 3
    assert p["gamma_eff_clip"] == 1.0
    # (clip - absA)/absA = (1.0 - 0.8)/0.8 = 0.25 — denominator NAMED
    assert p["gamma_clip_minus_absA_rel"] == pytest.approx(0.25)


def test_sign_provenance_is_inert_on_a_sign_definite_trace():
    t = np.array([0.0, 1.0, 2.0])
    A = np.array([2.0, 1.0, 0.5])
    p = _sign_provenance(t, A)
    assert p["amp_sign_change"] is False
    assert p["gamma_eff_clip"] == integral_gamma(t, A, estimator="absA")
    assert p["gamma_clip_minus_absA_rel"] == 0.0


# ================================ (a)-(d) both trainer paths, in the artifact
CFGS = {"1q": _cfg_1q}


@pytest.mark.parametrize("path", ["1q"])
def test_npz_records_the_estimator_and_the_sign_census(tmp_path, case, path):
    cfg = CFGS[path](case, outdir=tmp_path, run_id=f"est_{path}")
    r = train_spectral_pn(cfg)
    d = np.load(tmp_path / f"est_{path}_results.npz", allow_pickle=True)

    for k in SIGN_KEYS:                                   # (a)
        assert k in d.files, f"{path}: npz lost provenance key {k!r}"
        assert k in r, f"{path}: metrics dict lost {k!r}"
    assert str(d["gamma_estimator"]) == "absA"

    t, A = d["t"], d["A"]
    assert float(d["gamma_eff"]) == integral_gamma(t, A, "absA")     # (b)
    assert float(d["gamma_eff_clip"]) == integral_gamma(t, A, "clip")  # (c)

    fresh = amplitude_sign_report(t, A)                              # (d)
    assert bool(d["amp_sign_change"]) == fresh.sign_change
    assert float(d["amp_A_min"]) == fresh.A_min
    assert int(d["amp_n_negative"]) == fresh.n_negative
    assert int(d["amp_n_samples"]) == fresh.n_samples == len(t)
    assert float(d["amp_negative_time_frac"]) == fresh.negative_time_fraction
    assert float(d["amp_negative_sample_frac"]) == \
        fresh.negative_sample_fraction


@pytest.mark.parametrize("path", ["1q"])
def test_gamma_eff_is_the_ruled_estimator_not_the_legacy_one(case, path):
    """On a sign-definite fit the two coincide bitwise, so this also documents
    WHEN the change is a no-op.
    """
    r = train_spectral_pn(CFGS[path](case))
    t, A = r["t"], r["A"]
    assert r["gamma_eff"] == integral_gamma(t, A, "absA")
    assert (r["gamma_eff"] == r["gamma_eff_clip"]) == (A.min() >= 0.0)


@pytest.mark.parametrize("path", ["1q"])
def test_a_min_shape_metric_and_amp_a_min_are_the_same_reading(case, path):
    """`A_min` (shape set, present only with a reference) and `amp_A_min`
    (unconditional provenance) must never disagree — two names for one number
    is how conventions drift apart.
    """
    r = train_spectral_pn(CFGS[path](case))
    assert r["amp_A_min"] == r["A_min"]


def _synthetic_npz(estimator: str | None, sign_change: bool) -> dict:
    t = np.linspace(0.0, 1.0, 201)
    A_ref = np.exp(-3.0 * t)
    A = A_ref - (0.05 if sign_change else 0.0)
    A = A / A[0]
    scored = estimator or "clip"
    d = {"t": t, "A": A, "t_ref": t, "A_ref": A_ref,
         "gamma_eff": integral_gamma(t, A, estimator=scored)}
    if estimator is not None:
        d["gamma_estimator"] = np.asarray(estimator)
    return d


def test_legacy_npz_without_the_key_is_scored_as_clip():
    d = _synthetic_npz(None, sign_change=True)
    assert d["A"].min() < 0.0, "the fixture must exercise the divergence"
    assert npz_estimator(d) == "clip"
    g = gate({**d, "rmse": np.nan, "max_abs_dA": np.nan})
    assert g["gamma_eff_ok"] is True
    assert g["gamma_estimator"] == "clip"


def test_new_npz_declaring_absA_is_scored_as_absA():
    d = _synthetic_npz("absA", sign_change=True)
    g = gate({**d, "rmse": np.nan, "max_abs_dA": np.nan})
    assert g["gamma_eff_ok"] is True
    assert g["gamma_estimator"] == "absA"


def test_the_gate_still_disagrees_when_the_label_lies():
    """MUTATION: an agreeing gate is evidence only if it can disagree. Label a
    clip-scored record as 'absA' on a sign-CHANGING trace and it must go red.
    """
    d = _synthetic_npz(None, sign_change=True)          # clip-scored value
    d["gamma_estimator"] = np.asarray("absA")           # ... mislabelled
    g = gate({**d, "rmse": np.nan, "max_abs_dA": np.nan})
    assert g["gamma_eff_ok"] is False


def test_a_sign_definite_record_passes_under_every_label():
    """The six immune rungs, in miniature: with A >= 0 the label cannot matter,
    which is exactly why the seam stayed invisible for so long.
    """
    for label in (None, "clip", "absA", "raw"):
        d = _synthetic_npz(label, sign_change=False)
        assert d["A"].min() > 0.0
        g = gate({**d, "rmse": np.nan, "max_abs_dA": np.nan})
        assert g["gamma_eff_ok"] is True, label


def test_the_sign_census_survives_the_logger_being_on(case):
    off = train_spectral_pn(_cfg_1q(case, gamma_log_every=0))
    on = train_spectral_pn(_cfg_1q(case, gamma_log_every=1))
    for k in SIGN_KEYS + ("gamma_eff",):
        assert on[k] == off[k], f"{k}: {on[k]!r} vs {off[k]!r}"
    assert torch.equal(torch.random.get_rng_state(),
                       torch.random.get_rng_state())
