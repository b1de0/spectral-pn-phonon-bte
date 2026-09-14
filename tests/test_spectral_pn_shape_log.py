"""Shape-metric logging: opt-in, byte-identical trajectory, metrics exact on
hand-computed curves.
"""
import numpy as np
import pytest
import torch

from pinn_bte.training.transient.spectral_pn_trainer import (
    SHAPE_NAMES,
    SpectralPNConfig,
    _integral_gamma,
    _shape_metrics,
    build_si_case,
    train_spectral_pn,
)

pytestmark = pytest.mark.usefixtures("default_float64")


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


CFGS = {"1q": _cfg_1q}


@pytest.mark.parametrize("path", ["1q"])
def test_shape_history_defaults_off(case, path):
    """Opt-in: the shape curve rides gamma_log_every, whose default is 0."""
    cfg = CFGS[path](case)
    assert cfg.gamma_log_every == 0
    r = train_spectral_pn(cfg)
    assert r["shape_history"] == []


@pytest.mark.parametrize("path", ["1q"])
def test_shape_logging_leaves_trajectory_byte_identical(case, path):
    cfg = CFGS[path]
    r_off = train_spectral_pn(cfg(case, gamma_log_every=0))
    r_on = train_spectral_pn(cfg(case, gamma_log_every=1))

    assert r_on["history"] == r_off["history"], (
        f"loss trajectory perturbed by the shape logger:\n"
        f"  off={r_off['history']}\n  on ={r_on['history']}")
    assert r_on["loss_final"] == r_off["loss_final"], \
        f"{r_on['loss_final']!r} vs {r_off['loss_final']!r}"
    assert r_on["gamma_eff"] == r_off["gamma_eff"], \
        f"{r_on['gamma_eff']!r} vs {r_off['gamma_eff']!r}"
    assert r_on["gamma_ratio"] == r_off["gamma_ratio"], \
        f"{r_on['gamma_ratio']!r} vs {r_off['gamma_ratio']!r}"
    assert np.array_equal(r_on["A"], r_off["A"]), "published A perturbed"
    # the published SHAPE numbers must also survive the logger being on
    for k in ("rmse", "max_abs_dA"):
        assert r_on[k] == r_off[k], f"{k}: {r_on[k]!r} vs {r_off[k]!r}"


@pytest.mark.parametrize("path", ["1q"])
def test_shape_logging_consumes_no_rng(case, path):
    """Assert the invariant directly on the generator state."""
    cfg = CFGS[path]

    def _rng_state_after(gamma_log_every):
        train_spectral_pn(cfg(case, gamma_log_every=gamma_log_every))
        return torch.random.get_rng_state()

    assert torch.equal(_rng_state_after(1), _rng_state_after(0)), \
        "the shape logger advanced the global torch RNG"


@pytest.mark.parametrize("path", ["1q"])
def test_inloop_shape_at_final_epoch_equals_postloop_metrics(case, path):
    cfg = CFGS[path](case, gamma_log_every=1)
    r = train_spectral_pn(cfg)
    sh = np.asarray(r["shape_history"], dtype=float)

    assert sh.shape == (cfg.epochs, 1 + len(SHAPE_NAMES)), sh.shape
    assert sh[-1, 0] == cfg.epochs - 1
    cols = {n: 1 + i for i, n in enumerate(SHAPE_NAMES)}
    assert sh[-1, cols["rmse"]] == r["rmse"], (
        f"in-loop rmse is NOT the published estimator: "
        f"{sh[-1, cols['rmse']]!r} vs {r['rmse']!r}")
    assert sh[-1, cols["max_abs_dA"]] == r["max_abs_dA"], (
        f"in-loop max_abs_dA is NOT the published estimator: "
        f"{sh[-1, cols['max_abs_dA']]!r} vs {r['max_abs_dA']!r}")
    # the shape-only readings are published at the endpoint too
    for n in ("int_signed_dA", "int_abs_dA", "A_min", "max_dA_fwd"):
        assert sh[-1, cols[n]] == r[n], f"{n}: {sh[-1, cols[n]]!r} vs {r[n]!r}"


@pytest.mark.parametrize("path", ["1q"])
def test_shape_history_rides_the_gamma_cadence(case, path):
    """One amplitude evaluation feeds both, so the two curves must sit on the SAME
    epoch grid — a reader may join them column-wise without re-aligning.
    """
    cfg = CFGS[path](case, epochs=8, gamma_log_every=3)
    r = train_spectral_pn(cfg)
    gh = np.asarray(r["gamma_history"], dtype=float)
    sh = np.asarray(r["shape_history"], dtype=float)
    assert list(sh[:, 0]) == [0, 3, 6, 7]
    assert np.array_equal(sh[:, 0], gh[:, 0])


# ==================================================== (d) provenance in npz
@pytest.mark.parametrize("path", ["1q"])
def test_shape_curve_and_names_recorded_in_npz(tmp_path, case, path):
    """The curve is the deliverable; the names make the npz self-describing so no
    reader ever hardcodes a column order.
    """
    cfg = CFGS[path](case, gamma_log_every=2, outdir=tmp_path,
                     run_id=f"sh_{path}")
    train_spectral_pn(cfg)
    data = np.load(tmp_path / f"sh_{path}_results.npz", allow_pickle=True)

    sh = data["shape_history"]
    assert sh.ndim == 2 and sh.shape[1] == 1 + len(SHAPE_NAMES)
    assert sh[-1, 0] == cfg.epochs - 1
    assert list(data["shape_names"]) == list(SHAPE_NAMES)
    # endpoint scalars land as their own keys, matching the curve's last row
    for i, n in enumerate(SHAPE_NAMES):
        assert float(data[n]) == pytest.approx(float(sh[-1, 1 + i]), rel=1e-12)
    # every pre-existing key survives (the pinned-npz reader must not break)
    for k in ("t", "A", "loss_history", "gamma_history", "gamma_eff",
              "gamma_ratio", "rmse", "max_abs_dA", "component_history",
              "formulation", "shared_ahat"):
        assert k in data.files
    assert data["gamma_history"].shape[1] == 3, \
        "gamma_history widened — existing readers reshape(-1, 3)"


@pytest.mark.parametrize("path", ["1q"])
def test_shape_log_off_records_empty_well_shaped_curve(tmp_path, case, path):
    """OFF still records an empty, well-shaped curve — downstream readers never
    have to branch on the key's absence.
    """
    cfg = CFGS[path](case, outdir=tmp_path, run_id=f"shoff_{path}")
    train_spectral_pn(cfg)
    data = np.load(tmp_path / f"shoff_{path}_results.npz", allow_pickle=True)
    assert data["shape_history"].shape == (0, 1 + len(SHAPE_NAMES))
    assert list(data["shape_names"]) == list(SHAPE_NAMES)


# ================================= (e) the arithmetic, pinned by hand
def test_shape_metrics_exact_on_a_hand_computed_curve():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    A_ref = np.ones(5)
    A = np.array([1.0, 2.0, 1.0, 0.0, 1.0])

    rmse, max_abs, int_signed, int_abs, a_min, max_fwd = _shape_metrics(
        t, A, A_ref)

    assert rmse == pytest.approx(np.sqrt(2.0 / 5.0))   # mean([0,1,0,1,0])
    assert max_abs == pytest.approx(1.0)
    assert int_signed == pytest.approx(0.0, abs=1e-15)
    assert int_abs == pytest.approx(2.0)
    assert a_min == pytest.approx(0.0)
    assert max_fwd == pytest.approx(1.0)               # diff = [1,-1,-1,1]

    # the cancellation ratio: 0 => the area error is entirely invisible to
    # gamma_ratio while the curve is wrong by 2 t-units of |dA|
    assert abs(int_signed) / int_abs == pytest.approx(0.0, abs=1e-15)


def test_shape_metrics_flags_negative_excursion_and_monotone_decay():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    A_ref = np.array([1.0, 0.5, 0.25, 0.125])

    # strictly decaying, never negative => clip inert, monotone certified
    good = _shape_metrics(t, A_ref.copy(), A_ref)
    assert good[4] == pytest.approx(0.125)   # A_min > 0
    assert good[5] < 0.0                     # max forward diff < 0 => monotone

    # dips below zero => the published estimator would clip it away
    bad = _shape_metrics(t, np.array([1.0, 0.5, -0.2, 0.1]), A_ref)
    assert bad[4] == pytest.approx(-0.2), "A_min failed to see the excursion"
    assert bad[5] > 0.0, "max_dA_fwd failed to see the non-monotone rebound"


def test_shape_catches_right_integral_wrong_curve():
    """THE point of the track, and the user's directive made executable."""
    t = np.linspace(0.0, 10.0, 4001)
    A_ref = np.exp(-t)
    # areas over [0,inf): 0.5/10 + 0.5/(10/19) = 0.05 + 0.95 = 1.0 == int A_ref
    A = 0.5 * np.exp(-10.0 * t) + 0.5 * np.exp(-(10.0 / 19.0) * t)

    assert A[0] == pytest.approx(1.0)        # same numerator as the reference
    g_ratio = _integral_gamma(t, A) / _integral_gamma(t, A_ref)
    rmse, max_abs, int_signed, int_abs, _, _ = _shape_metrics(t, A, A_ref)

    assert abs(g_ratio - 1.0) < 0.01, g_ratio          # measured 0.0049
    # the shape: catastrophically wrong, by every reading
    assert rmse > 0.05, rmse                           # measured 0.0717
    assert max_abs > 0.2, max_abs                      # measured 0.3018
    assert int_abs >= abs(int_signed) > 0.0, (int_abs, int_signed)
    cancellation = abs(int_signed) / int_abs
    assert 0.0 <= cancellation < 0.05, (               # measured 0.0118
        f"the cancellation ratio should expose that gamma_ratio={g_ratio:.4f} "
        f"is an artifact of two errors cancelling; got {cancellation:.4f}")


def test_A0_is_pinned_at_one_so_an_A0_error_term_would_be_a_dead_metric():
    case = build_si_case(10.0, Nk=6, n_decay_times=6.0, geometry="1d")
    for cfg_fn in (_cfg_1q,):
        r = train_spectral_pn(cfg_fn(case, epochs=2))
        assert r["A"][0] == 1.0, (
            f"A(0) is no longer pinned at 1.0 ({r['A'][0]!r}) — gamma_eff is no "
            f"longer a pure functional of the area, and the omitted A0 error "
            f"term is now a live diagnostic")
        assert "A0_err" not in r
