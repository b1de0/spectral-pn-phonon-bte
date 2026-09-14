"""Snapshot selection ('final' vs 'min-loss'): byte-identical trajectory,
shipped weights match the reported loss, npz provenance.
"""
import numpy as np
import pytest
import torch

from pinn_bte.training.transient.spectral_pn_trainer import (
    SpectralPNConfig,
    _collocation_times,
    build_si_case,
    train_spectral_pn,
)

pytestmark = pytest.mark.usefixtures("default_float64")


@pytest.fixture(scope="module")
def case():
    return build_si_case(10.0, Nk=6, n_decay_times=6.0, geometry="1d")


def _cfg(case, **kw):
    """At lr=0.01 / 8 epochs this history BOUNCES (interior minimum at epoch 3,
    final ~2.7x worse) — the synthetic bouncing history the flag exists for.
    """
    base = dict(
        v=case.v, tau=case.tau, C=case.C, q=case.q, t_end=case.t_end,
        L_max=4, width=16, depth=2, emb_dim=4, epochs=8, lr=0.01,
        lr_cosine=True, nt=60, t_grid="uniform", seed=42, device="cpu",
        dtype="float64", t_ref=case.t_ref, A_ref=case.A_ref,
        gamma_ref=case.gamma_dom_hz, Nk=6, T_ref=300.0, log_every=1,
        formulation="xihybrid", shared_ahat=True,
        carrier_energy=True, carrier_row=True, xi_split=1.0, xi_duh=10.0,
    )
    base.update(kw)
    return SpectralPNConfig(**base)


def _tensors(cfg):
    to = lambda a: torch.as_tensor(np.asarray(a, dtype=float),  # noqa: E731
                                   dtype=torch.float64)
    return to(cfg.v), to(cfg.tau), to(cfg.C)


def _recompute_loss_1q(cfg, pt_path):
    """Rebuild the trainer's net EXACTLY as train_spectral_pn does, load the
    shipped weights, and recompute the trainer's own loss expression on the
    training collocation grid.
    """
    from pinn_bte.models.spectral_pn import XiHybridPNCoefficientNet, pn_residual

    v, tau, C = _tensors(cfg)
    gamma_kin = (cfg.q ** 2 * float(np.sum(np.asarray(cfg.C) *
                                           np.asarray(cfg.v) ** 2 *
                                           np.asarray(cfg.tau)))
                 / 3.0 / float(np.sum(cfg.C)))
    net = XiHybridPNCoefficientNet(
        n_modes=len(cfg.v), L_max=cfg.L_max, gamma_kin=gamma_kin,
        v=cfg.v, tau=cfg.tau, q=cfg.q, width=cfg.width, depth=cfg.depth,
        emb_dim=cfg.emb_dim, xi_split=cfg.xi_split, xi_duh=cfg.xi_duh,
        shared_ahat=cfg.shared_ahat, C=cfg.C,
        carrier_energy=cfg.carrier_energy, blend=cfg.blend,
    ).to(dtype=torch.float64)
    net.load_state_dict(torch.load(pt_path))
    t_coll = torch.as_tensor(_collocation_times(cfg), dtype=torch.float64)
    res, carrier = pn_residual(net, t_coll, v, tau, C, q=cfg.q,
                               ap_scaling=True, return_carrier=True)
    return float(((res ** 2).mean() + (carrier ** 2).mean()).detach()), net


def test_snapshot_selection_defaults_final(case):
    cfg = _cfg(case)
    assert cfg.snapshot_selection == "final"


def test_default_matches_explicit_final_byte_identical(tmp_path, case):
    """T0c-style: the flag OMITTED == 'final' EXPLICIT, bit-for-bit — same npz
    values (every key), same .pt bytes. The default path must be HEAD.
    """
    train_spectral_pn(_cfg(case, outdir=tmp_path / "a", run_id="dflt"))
    train_spectral_pn(_cfg(case, outdir=tmp_path / "b", run_id="dflt",
                           snapshot_selection="final"))
    da = np.load(tmp_path / "a" / "dflt_results.npz", allow_pickle=True)
    db = np.load(tmp_path / "b" / "dflt_results.npz", allow_pickle=True)
    assert sorted(da.files) == sorted(db.files)
    for k in da.files:
        a, b = np.asarray(da[k]), np.asarray(db[k])
        if a.dtype.kind == "f":
            assert np.array_equal(a, b, equal_nan=True), k
        else:
            assert np.array_equal(a, b), k
    pa = (tmp_path / "a" / "net_pn_dflt.pt").read_bytes()
    pb = (tmp_path / "b" / "net_pn_dflt.pt").read_bytes()
    assert pa == pb, "saved weights differ between omitted and explicit 'final'"


def test_trainer_rejects_unknown_snapshot_selection(case):
    with pytest.raises(ValueError, match="snapshot_selection"):
        train_spectral_pn(_cfg(case, snapshot_selection="best-rmse"))


# ============================ (b) pure observation: trajectory + RNG inert
def test_min_loss_leaves_trajectory_byte_identical(case):
    """The non-negotiable (spec requirement 3): min-loss tracking must not change
    the training trajectory AT ALL — bitwise on the loss history and the
    in-loop gamma/shape curves.
    """
    r_off = train_spectral_pn(_cfg(case, gamma_log_every=1))
    r_on = train_spectral_pn(_cfg(case, gamma_log_every=1,
                                  snapshot_selection="min-loss"))
    assert r_on["history"] == r_off["history"], (
        f"loss trajectory perturbed by the snapshot tracker:\n"
        f"  off={r_off['history']}\n  on ={r_on['history']}")
    assert r_on["loss_final"] == r_off["loss_final"]
    assert r_on["gamma_history"] == r_off["gamma_history"]
    assert r_on["shape_history"] == r_off["shape_history"]


def test_min_loss_consumes_no_rng(case):
    """The tracker must draw ZERO RNG (state_dict clone only)."""
    def _rng_state_after(**kw):
        train_spectral_pn(_cfg(case, **kw))
        return torch.random.get_rng_state()

    assert torch.equal(_rng_state_after(snapshot_selection="min-loss"),
                       _rng_state_after()), \
        "the snapshot tracker advanced the global torch RNG"


# ================= (c) the bouncing history ships the INTERIOR snapshot
def test_min_loss_ships_interior_snapshot_on_bouncing_history(tmp_path, case):
    """Decisive check: reload the shipped .pt and recompute the trainer's own loss
    — it must reproduce best_loss bitwise (the final state's loss would not).
    """
    cfg_off = _cfg(case, outdir=tmp_path / "off", run_id="b")
    r_off = train_spectral_pn(cfg_off)
    losses = np.asarray(r_off["history"], dtype=float)[:, 1]
    am = int(np.argmin(losses))
    # precondition: the fixture really bounces (else this test is vacuous)
    assert am != cfg_off.epochs - 1, (
        f"fixture no longer bounces (argmin={am}); retune lr/epochs")

    cfg_on = _cfg(case, outdir=tmp_path / "on", run_id="b",
                  snapshot_selection="min-loss")
    r_on = train_spectral_pn(cfg_on)
    assert r_on["best_epoch"] == am
    assert r_on["best_loss"] == losses[am]
    assert r_on["best_loss"] < r_on["loss_final"]

    got, _ = _recompute_loss_1q(cfg_on, tmp_path / "on" / "net_pn_b.pt")
    assert got == r_on["best_loss"], (
        f"shipped .pt is not the best-loss snapshot: recomputed {got!r} "
        f"vs best_loss {r_on['best_loss']!r}")
    # and it is NOT the final state
    pt_on = (tmp_path / "on" / "net_pn_b.pt").read_bytes()
    pt_off = (tmp_path / "off" / "net_pn_b.pt").read_bytes()
    assert pt_on != pt_off, "min-loss shipped the final weights"


def test_min_loss_published_metrics_describe_shipped_weights(tmp_path, case):
    """Self-consistent artifacts: the npz A/gamma must be evaluated AT the shipped
    snapshot (an npz whose metrics describe a different state than its .pt
    would be a provenance trap).
    """
    from pinn_bte.models.spectral_pn import pn_amplitude

    cfg = _cfg(case, outdir=tmp_path, run_id="sc",
               snapshot_selection="min-loss")
    train_spectral_pn(cfg)
    data = np.load(tmp_path / "sc_results.npz", allow_pickle=True)
    _, net = _recompute_loss_1q(cfg, tmp_path / "net_pn_sc.pt")
    v, tau, C = _tensors(cfg)
    te = torch.as_tensor(data["t"], dtype=torch.float64)
    with torch.no_grad():
        A = pn_amplitude(net, te, tau, C, v=v, q=cfg.q).numpy()
    assert np.array_equal(A, data["A"]), \
        "npz A does not describe the shipped snapshot"


# =============== (d) selection restricted to the EXISTING logging cadence
def test_min_loss_selects_on_logging_cadence_only(tmp_path, case):
    """No extra evaluations: with log_every=2 the global minimum (epoch 3) is
    OFF-cadence and must NOT be selected — the tracker sees only the logged
    epochs {0, 2, 4, 6, 7}.
    """
    losses = np.asarray(
        train_spectral_pn(_cfg(case))["history"], dtype=float)[:, 1]
    n = len(losses)
    am_global = int(np.argmin(losses))
    coarse = sorted({it for it in range(n) if it % 2 == 0} | {n - 1})
    am_coarse = min(coarse, key=lambda it: losses[it])
    # precondition: the cadence distinction is real for this fixture
    assert am_global not in coarse, "fixture degenerate; retune log_every"

    r_coarse = train_spectral_pn(_cfg(case, log_every=2,
                                      snapshot_selection="min-loss"))
    assert r_coarse["best_epoch"] == am_coarse, \
        "tracker selected an off-cadence epoch (extra evaluation?)"

    fine = sorted(set(coarse) | {it for it in range(n) if it % 3 == 0})
    am_fine = min(fine, key=lambda it: losses[it])
    assert am_fine == am_global  # epoch 3 is on the gamma cadence
    r_fine = train_spectral_pn(_cfg(case, log_every=2, gamma_log_every=3,
                                    snapshot_selection="min-loss"))
    assert r_fine["best_epoch"] == am_fine


# ========================================== (e) provenance in the npz
def test_npz_records_selection_keys_min_loss(tmp_path, case):
    cfg = _cfg(case, outdir=tmp_path, run_id="p",
               snapshot_selection="min-loss")
    r = train_spectral_pn(cfg)
    data = np.load(tmp_path / "p_results.npz", allow_pickle=True)
    assert str(data["snapshot_selection"]) == "min-loss"
    assert int(data["best_epoch"]) == r["best_epoch"]
    assert float(data["best_loss"]) == r["best_loss"]
    assert float(data["final_loss"]) == r["loss_final"]
    hist = data["loss_history"]
    assert float(data["final_loss"]) == hist[-1, 1]
    assert float(data["best_loss"]) <= float(data["final_loss"])


def test_npz_records_selection_keys_final_mode(tmp_path, case):
    """OFF still records the keys (provenance; readers never branch on absence):
    the shipped snapshot IS the final epoch.
    """
    cfg = _cfg(case, outdir=tmp_path, run_id="pf")
    train_spectral_pn(cfg)
    data = np.load(tmp_path / "pf_results.npz", allow_pickle=True)
    assert str(data["snapshot_selection"]) == "final"
    assert int(data["best_epoch"]) == cfg.epochs - 1
    assert float(data["best_loss"]) == float(data["final_loss"])
    # existing payload keys unchanged (pinned-npz readers must not break)
    for k in ("t", "A", "loss_history", "gamma_eff", "gamma_ratio", "rmse",
              "formulation", "blend", "gamma_log_every"):
        assert k in data.files


def test_runner_defaults_snapshot_final():
    from experiments.transient.run_spectral_pn import parse_args
    args = parse_args(["--L", "1.0"])
    assert args.snapshot_selection == "final"


def test_runner_accepts_min_loss():
    from experiments.transient.run_spectral_pn import parse_args
    args = parse_args(["--L", "1.0", "--snapshot-selection", "min-loss"])
    assert args.snapshot_selection == "min-loss"


def test_runner_rejects_unknown_snapshot_selection():
    from experiments.transient.run_spectral_pn import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--L", "1.0", "--snapshot-selection", "best-rmse"])


@pytest.mark.parametrize("geometry", ["1d", "2d"])
def test_runner_wires_min_loss_end_to_end(tmp_path, geometry):
    """The tracker is geometry-agnostic by construction (it reads only the
    cadences, the loss tensor and net.state_dict(); geometry enters solely via
    the DOM reference), and this pins it.
    """
    from experiments.transient.run_spectral_pn import main
    outdir = main(["--L", "10.0", "--nk", "6", "--lmax", "4", "--width", "16",
                   "--depth", "2", "--epochs", "3", "--nt", "60",
                   "--t-grid", "uniform", "--n-decay-times", "6.0",
                   "--device", "cpu", "--log-every", "1",
                   "--geometry", geometry,
                   "--snapshot-selection", "min-loss",
                   "--outdir", str(tmp_path)])
    npz = sorted(outdir.glob("*_results.npz"))
    assert len(npz) == 1
    data = np.load(npz[0], allow_pickle=True)
    assert str(data["snapshot_selection"]) == "min-loss"
    assert str(data["geometry"]) == geometry
    assert "best_epoch" in data.files and "final_loss" in data.files
