"""gamma_ratio logging on the single-harmonic path: opt-in, byte-identical
trajectory, recorded in the npz.
"""
import numpy as np
import pytest
import torch

from pinn_bte.training.transient.spectral_pn_trainer import (
    SpectralPNConfig,
    build_si_case,
    train_spectral_pn,
)

pytestmark = pytest.mark.usefixtures("default_float64")


@pytest.fixture(scope="module")
def case():
    return build_si_case(10.0, Nk=6, n_decay_times=6.0, geometry="1d")


def _cfg(case, **kw):
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


# ===================================================== (a) default is OFF
def test_gamma_log_every_defaults_off(case):
    """The flag is opt-in: default 0 => no curve, current behaviour exactly."""
    cfg = _cfg(case)
    assert cfg.gamma_log_every == 0
    r = train_spectral_pn(cfg)
    assert r["gamma_history"] == []


def test_gamma_logging_leaves_trajectory_byte_identical(case):
    r_off = train_spectral_pn(_cfg(case, gamma_log_every=0))
    r_on = train_spectral_pn(_cfg(case, gamma_log_every=1))

    assert r_on["history"] == r_off["history"], (
        f"loss trajectory perturbed by the logger:\n"
        f"  off={r_off['history']}\n  on ={r_on['history']}")
    assert r_on["loss_final"] == r_off["loss_final"], \
        f"{r_on['loss_final']!r} vs {r_off['loss_final']!r}"
    assert r_on["gamma_eff"] == r_off["gamma_eff"], \
        f"{r_on['gamma_eff']!r} vs {r_off['gamma_eff']!r}"
    assert r_on["gamma_ratio"] == r_off["gamma_ratio"], \
        f"{r_on['gamma_ratio']!r} vs {r_off['gamma_ratio']!r}"
    assert np.array_equal(r_on["A"], r_off["A"]), "published A perturbed"


def test_gamma_logging_consumes_no_rng(case):
    """Assert the invariant directly on the generator state instead of relying on
    that coincidence.
    """
    def _rng_state_after(gamma_log_every):
        train_spectral_pn(_cfg(case, gamma_log_every=gamma_log_every))
        return torch.random.get_rng_state()

    assert torch.equal(_rng_state_after(1), _rng_state_after(0)), \
        "the gamma logger advanced the global torch RNG"


def test_inloop_gamma_at_final_epoch_equals_postloop_gamma_ratio(case):
    cfg = _cfg(case, gamma_log_every=1)
    r = train_spectral_pn(cfg)
    gh = np.asarray(r["gamma_history"], dtype=float)

    assert gh.shape == (cfg.epochs, 3), gh.shape
    assert gh[-1, 0] == cfg.epochs - 1
    assert gh[-1, 1] == r["gamma_ratio"], (
        f"in-loop gamma is NOT the published estimator: "
        f"{gh[-1, 1]!r} vs {r['gamma_ratio']!r}")


def test_gamma_history_tracks_the_logged_loss(case):
    """The curve carries its own loss column so a budget curve can be read against
    the loss without re-aligning two different epoch grids.
    """
    cfg = _cfg(case, gamma_log_every=1)
    r = train_spectral_pn(cfg)
    gh = np.asarray(r["gamma_history"], dtype=float)
    losses = np.asarray(r["history"], dtype=float)
    assert np.array_equal(gh[:, 0], losses[:, 0])
    assert np.array_equal(gh[:, 2], losses[:, 1])
    assert np.isfinite(gh[:, 1]).all()


def test_gamma_log_every_cadence_is_honoured(case):
    """Cadence N samples epochs 0, N, 2N, ... plus the final epoch (so the curve
    always ends on the published number, whatever N divides).
    """
    cfg = _cfg(case, epochs=8, gamma_log_every=3)
    gh = np.asarray(train_spectral_pn(cfg)["gamma_history"], dtype=float)
    assert list(gh[:, 0]) == [0, 3, 6, 7]


# ============================================== (d) provenance in the npz
def test_gamma_curve_and_flag_recorded_in_npz(tmp_path, case):
    cfg = _cfg(case, gamma_log_every=2, outdir=tmp_path, run_id="gl1q_prov")
    train_spectral_pn(cfg)
    data = np.load(tmp_path / "gl1q_prov_results.npz", allow_pickle=True)
    assert int(data["gamma_log_every"]) == 2
    gh = data["gamma_history"]
    assert gh.ndim == 2 and gh.shape[1] == 3
    assert gh[-1, 0] == cfg.epochs - 1
    assert float(gh[-1, 1]) == pytest.approx(float(data["gamma_ratio"]), rel=1e-12)
    # existing payload keys unchanged (the pinned-npz reader must not break)
    for k in ("t", "A", "loss_history", "gamma_eff", "gamma_ratio", "rmse",
              "duhamel", "slow_head", "formulation", "shared_ahat"):
        assert k in data.files


def test_gamma_log_off_records_empty_curve(tmp_path, case):
    """OFF still records the flag (provenance) and an empty, well-shaped curve —
    downstream readers never have to branch on the key's absence.
    """
    cfg = _cfg(case, outdir=tmp_path, run_id="gl1q_off")
    train_spectral_pn(cfg)
    data = np.load(tmp_path / "gl1q_off_results.npz", allow_pickle=True)
    assert int(data["gamma_log_every"]) == 0
    assert data["gamma_history"].shape == (0, 3)


# ======================================= the runner exposes it on the 1q path
def test_runner_accepts_gamma_log_every_on_1q():
    from experiments.transient.run_spectral_pn import parse_args
    args = parse_args(["--L", "1.0", "--gamma-log-every", "250"])
    assert args.gamma_log_every == 250


def test_runner_still_rejects_negative_gamma_log_every():
    from experiments.transient.run_spectral_pn import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--L", "1.0", "--gamma-log-every", "-1"])
