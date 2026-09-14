"""Gray-TTG data-free control."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pinn_bte.physics.decay_metrics import (  # noqa: E402
    DENOMINATOR_RULE_SAME_WINDOW,
    GAMMA_ESTIMATOR_RULED,
    integral_gamma,
)
from pinn_bte.physics.gray_ttg import (  # noqa: E402
    GRAY_XI_CRITICAL,
    build_gray_case,
    gray_amplitude_curve,
    gray_dominant_gamma,
    gray_gamma_same_window,
    gray_window_gamma,
)


class TestGrayArbiterTrace:
    def test_matches_compute_analytical_amplitude(self):
        """Round-off agreement with the repo's exact gray reference."""
        from pinn_bte.utils.plotting_transient import (
            compute_analytical_amplitude,
        )
        for xi in (0.6, 4.0):
            t, A = gray_amplitude_curve(xi, t_end_tau=6.0, nsteps=1200)
            A_ref = compute_analytical_amplitude(t, xi)
            assert np.abs(A - A_ref).max() < 1e-10

    def test_dt_convergence(self):
        xi = 0.6
        t_end = 7.0 / gray_window_gamma(xi)
        g = {n: gray_gamma_same_window(xi, t_end_tau=t_end, nsteps=n)
             for n in (12000, 24000, 48000)}
        order = (g[24000] - g[12000]) / (g[48000] - g[24000])
        assert order == pytest.approx(4.0, rel=0.05)
        assert abs(g[12000] / g[48000] - 1.0) < 5e-5

    def test_float64(self):
        t, A = gray_amplitude_curve(0.6, t_end_tau=1.0, nsteps=100)
        assert t.dtype == np.float64 and A.dtype == np.float64
        assert A[0] == 1.0


class TestGrayDominantGammaAndWindow:
    def test_pole_closed_form(self):
        """Collective pole of 1 - F(s)/tau: s* = -1 + xi/tan(xi)."""
        for xi in (0.3, 0.6, 1.0, 1.5):
            assert gray_dominant_gamma(xi) == pytest.approx(
                1.0 - xi / np.tan(xi), rel=0, abs=0)

    def test_fourier_limit(self):
        """xi -> 0 must recover Fourier: gamma -> q^2 * (v^2 tau / 3) = xi^2/3."""
        for xi in (0.01, 0.05):
            assert gray_dominant_gamma(xi) == pytest.approx(
                xi * xi / 3.0, rel=1e-3)

    def test_no_pole_beyond_critical(self):
        """arctan(xi/(s+1)) = xi has no solution for xi >= pi/2: the discrete
        collective mode vanishes (Collins/Hua-Minnich transition).
        """
        assert GRAY_XI_CRITICAL == pytest.approx(np.pi / 2.0, rel=0, abs=0)
        for xi in (np.pi / 2.0, 4.0, 36.0):
            assert gray_dominant_gamma(xi) is None

    def test_window_gamma_fallback(self):
        """Window rate = pole where it exists, else q*<v> = xi — the same fallback
        ttg_dom uses (`q * sum(C v)/sum(C)`) restated for one gray mode.
        """
        assert gray_window_gamma(0.6) == pytest.approx(
            1.0 - 0.6 / np.tan(0.6), rel=0, abs=0)
        assert gray_window_gamma(4.0) == 4.0
        assert gray_window_gamma(36.0) == 36.0

    def test_case_window_is_n_decay_over_gamma(self):
        case = build_gray_case(0.6, n_decay=7)
        assert case.t_end_tau == pytest.approx(
            7.0 / (1.0 - 0.6 / np.tan(0.6)), rel=1e-12)
        case4 = build_gray_case(4.0, n_decay=7)
        assert case4.t_end_tau == pytest.approx(7.0 / 4.0, rel=1e-12)


class TestSameWindowDenominator:
    def test_self_score_is_exactly_one(self):
        """The reference re-scored on its own native trace must give ratio 1.0
        BIT-EXACTLY — the property the convention exists to guarantee.
        """
        case = build_gray_case(0.6, n_decay=7)
        g_num = integral_gamma(case.t_ref, case.A_ref,
                               estimator=GAMMA_ESTIMATOR_RULED)
        assert g_num / case.gamma_ref_per_tau == 1.0

    def test_provenance_block(self):
        case = build_gray_case(0.6, n_decay=7, nsteps=12000)
        p = case.provenance
        assert p["gamma_ref_estimator"] == GAMMA_ESTIMATOR_RULED
        assert p["gamma_ref_rule"] == DENOMINATOR_RULE_SAME_WINDOW
        assert p["gamma_ref_n_samples"] == 12001
        assert p["time_unit"] == "tau"
        assert p["gamma_ref_hz"] == case.gamma_ref_per_tau
        assert case.t_ref.size == 12001
        assert case.A_ref[0] == 1.0

    def test_sign_facts_at_zhou_operating_points(self):
        assert build_gray_case(0.6).A_ref.min() > 0.0
        assert build_gray_case(4.0).A_ref.min() < 0.0
        assert build_gray_case(36.0).A_ref.min() < 0.0

    def test_explicit_window_override(self):
        """`t_end_tau` explicit (the Zhou 3-tau fidelity arm) bypasses the n_decay
        rule but records both.
        """
        case = build_gray_case(0.6, n_decay=7, t_end_tau=3.0)
        assert case.t_end_tau == 3.0
        assert case.t_ref[-1] == pytest.approx(3.0, rel=1e-12)
        # A(3 tau) at xi=0.6 — Zhou's full published window sees only ~22%
        # of the decay (measured 0.7777); pin loosely as an reference fact.
        assert case.A_ref[-1] == pytest.approx(0.7777, abs=2e-3)


class TestAmplitudeExtraction:
    def test_recovers_pure_cosine_amplitude(self):
        from experiments.transient.run_gray_ttg_control import amp_cos
        x = np.linspace(0.0, 1.0, 64, endpoint=False)
        t = np.linspace(0.0, 5.0, 41)
        a = np.exp(-0.3 * t)
        T = a[:, None] * np.cos(2 * np.pi * x)[None, :]
        assert np.abs(amp_cos(T, x) - a).max() < 1e-12

    def test_orthogonal_to_higher_harmonics(self):
        from experiments.transient.run_gray_ttg_control import amp_cos
        x = np.linspace(0.0, 1.0, 64, endpoint=False)
        T = (0.7 * np.cos(2 * np.pi * x) + 0.2 * np.cos(4 * np.pi * x)
             + 0.1)[None, :]
        assert amp_cos(T, x)[0] == pytest.approx(0.7, abs=1e-12)


TINY = ["--xi", "0.6", "--epochs", "3", "--nx", "8", "--nt-train", "8",
        "--ns", "4", "--eval-nt", "41", "--eval-nx", "16",
        "--arbiter-nsteps", "400", "--seed", "42", "--device", "cpu",
        "--no-wandb"]

REQUIRED_NPZ_KEYS = [
    # battery contract (pn_rows reads these four + gamma_dom_hz + rmse)
    "t", "A", "t_ref", "A_ref", "gamma_dom_hz", "gamma_ratio", "rmse",
    "gamma_eff",
    # shape battery (mirrors spectral_pn_trainer SHAPE_NAMES)
    "max_abs_dA", "int_signed_dA", "int_abs_dA", "A_min", "max_dA_fwd",
    # sign census
    "gamma_estimator", "gamma_eff_clip", "gamma_clip_minus_absA_rel",
    "amp_sign_change", "amp_A_min", "amp_n_negative", "amp_n_samples",
    "amp_negative_time_frac", "amp_negative_sample_frac",
    # denominator provenance
    "gamma_ref_estimator", "gamma_ref_rule", "gamma_ref_t_end_s",
    "gamma_ref_n_samples", "gamma_ref_hz", "time_unit",
    "xi", "Kn", "n_decay", "window_mode", "window_tau", "epochs", "lr",
    "seed", "nx", "nt_train", "ns", "eval_nt", "eval_nx", "arbiter_nsteps",
    "residual_chunks",
    "net_arch", "hidden_dim", "num_layers", "snapshot_selection",
    "gamma_dominant_per_tau", "gamma_window_per_tau",
    # field leg (H-B's "renders the field plausibly")
    "T_field", "x_field", "field_rmse",
    "loss_history", "loss_component_names",
]


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("gray_ttg_tiny")
    proc = subprocess.run(
        [sys.executable, "-u",
         str(REPO / "experiments/transient/run_gray_ttg_control.py"),
         *TINY, "--outdir", str(outdir)],
        capture_output=True, text=True, timeout=600,
        env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "",
             "HOME": str(outdir), "WANDB_MODE": "disabled"},
        cwd=str(REPO))
    assert proc.returncode == 0, f"runner failed:\n{proc.stdout}\n{proc.stderr}"
    assert f"[gray-ttg] saved: {outdir}" in proc.stdout
    npz = list(outdir.glob("*_results.npz"))
    params = list(outdir.glob("*_params.json"))
    assert len(npz) == 1, f"expected exactly one results npz, got {npz}"
    assert len(params) == 1
    return dict(outdir=outdir, npz=np.load(npz[0], allow_pickle=True),
                params=json.loads(params[0].read_text()))


class TestRunnerArtifact:
    def test_npz_keys_complete(self, tiny_run):
        missing = [k for k in REQUIRED_NPZ_KEYS
                   if k not in tiny_run["npz"].files]
        assert not missing, f"npz missing keys: {missing}"

    def test_battery_rescore_reproduces_stored_ratio(self, tiny_run):
        """The EXACT expression `p086_cell_d_ledger.pn_rows` applies must
        reproduce the stored gamma_ratio bit-exactly from the traces.
        """
        d = tiny_run["npz"]
        t, A = np.asarray(d["t"]), np.asarray(d["A"])
        t_ref, A_ref = np.asarray(d["t_ref"]), np.asarray(d["A_ref"])
        g = integral_gamma(t, A, estimator=GAMMA_ESTIMATOR_RULED)
        g_ref = integral_gamma(t_ref, A_ref, estimator=GAMMA_ESTIMATOR_RULED)
        assert g / g_ref == float(d["gamma_ratio"])
        assert g == float(d["gamma_eff"])
        assert g_ref == float(d["gamma_ref_hz"])

    def test_reference_is_native_grid(self, tiny_run):
        d = tiny_run["npz"]
        assert int(d["gamma_ref_n_samples"]) == np.asarray(d["t_ref"]).size
        assert np.asarray(d["t_ref"]).size == 400 + 1
        assert float(np.asarray(d["A_ref"])[0]) == 1.0
        assert str(d["time_unit"]) == "tau"

    def test_windows_agree(self, tiny_run):
        """Numerator and denominator windows must END at the same time — the
        two-window mismatch is exactly what the same-window rule removes.
        """
        d = tiny_run["npz"]
        assert float(np.asarray(d["t"])[-1]) == pytest.approx(
            float(np.asarray(d["t_ref"])[-1]), rel=1e-12)
        assert float(d["gamma_ref_t_end_s"]) == pytest.approx(
            float(d["window_tau"]), rel=1e-12)

    def test_params_json_records_every_cli_param(self, tiny_run):
        from experiments.transient.run_gray_ttg_control import parse_args
        dests = set(vars(parse_args(TINY + ["--outdir", "x"])).keys())
        recorded = set(tiny_run["params"]["args"].keys())
        missing = dests - recorded
        assert not missing, f"params json missing CLI params: {missing}"

    def test_params_json_derived_values(self, tiny_run):
        derived = tiny_run["params"]["derived"]
        for key in ("t_end_tau", "nt_train_effective", "gamma_ref_per_tau",
                    "gamma_window_per_tau", "run_id", "git_sha"):
            assert key in derived, f"derived params missing {key}"

    def test_field_leg_shapes(self, tiny_run):
        d = tiny_run["npz"]
        T = np.asarray(d["T_field"])
        assert T.shape == (41, 16)
        assert np.asarray(d["x_field"]).shape == (16,)
        assert np.isfinite(float(d["field_rmse"]))

    def test_identity_scalars(self, tiny_run):
        d = tiny_run["npz"]
        assert float(d["xi"]) == 0.6
        assert int(d["seed"]) == 42
        assert int(d["epochs"]) == 3
        assert str(d["net_arch"]) == "mlp"
        assert int(d["hidden_dim"]) == 30 and int(d["num_layers"]) == 5
        assert str(d["snapshot_selection"]) == "final"
        assert float(d["Kn"]) == pytest.approx(0.6 / (2 * np.pi), rel=1e-12)


class TestRunnerDeterminism:
    def test_same_seed_same_amplitude(self, tmp_path):
        """CPU + fixed seed => the full A(t) trace reproduces bit-exactly
        (full-batch training, no sampling, no dropout).
        """
        from experiments.transient.run_gray_ttg_control import run_control
        argv = [a for a in TINY if a != "--no-wandb"] + ["--no-wandb"]
        r1 = run_control(argv + ["--outdir", str(tmp_path / "a")])
        r2 = run_control(argv + ["--outdir", str(tmp_path / "b")])
        assert np.array_equal(r1["A"], r2["A"])
        assert r1["gamma_ratio"] == r2["gamma_ratio"]

    def test_residual_chunks_recorded(self, tiny_run):
        assert int(tiny_run["npz"]["residual_chunks"]) == 1

    def test_nt_train_auto_rule(self):
        from experiments.transient.run_gray_ttg_control import resolve_nt_train
        # Zhou fidelity arm: 3 tau -> exactly Zhou's 60
        assert resolve_nt_train(0, 3.0) == 61  # ceil(60)+1
        # primary arm: 56.9187 tau -> 1140 points, dt <= 0.05 tau
        t_end = 7.0 / gray_window_gamma(0.6)
        nt = resolve_nt_train(0, t_end)
        assert nt == int(np.ceil(t_end / 0.05)) + 1
        assert t_end / (nt - 1) <= 0.05
        # short ballistic windows: floor at 60
        assert resolve_nt_train(0, 7.0 / 4.0) == 60
        assert resolve_nt_train(0, 7.0 / 36.0) == 60
        # explicit value passes through verbatim
        assert resolve_nt_train(8, 56.9) == 8


def _tiny_trainer(tmp_path, K=None, seed=7, Nx=10, Nt=12, Ns=4, epochs=1):
    """Trainer on a tiny CPU mesh; identical weights for identical seed."""
    import torch
    from pinn_bte.training.transient.gray.gray_1d import Transient1DGrayTrainer
    Transient1DGrayTrainer._wandb_enabled = False
    torch.manual_seed(seed)
    kw = {} if K is None else dict(residual_chunks=K)
    return Transient1DGrayTrainer(
        eta=0.6, window_tau=3.0, epochs=epochs, learning_rate=1e-3,
        Nx=Nx, Nt=Nt, Ns=Ns, device=torch.device("cpu"),
        output_dir=tmp_path, **kw)


def _expanded(tr):
    """The exact tensor expansion `_training_loop` performs (same ops, same order
    — transformed from the trainer's own data, never recreated).
    """
    x = tr.data['x'].repeat(1, tr.Ns).reshape(-1, 1)
    t = tr.data['t'].repeat(1, tr.Ns).reshape(-1, 1)
    mu = tr.data['mu'].repeat(tr.Nx * tr.Nt, 1)
    w = tr.data['w']
    mub = tr.data['mu'].repeat(tr.Nt, 1)
    mui = tr.data['mu'].repeat(tr.Nx, 1)
    tb = tr.data['tb'].repeat(1, tr.Ns).reshape(-1, 1)
    xi_ = tr.data['xi'].repeat(1, tr.Ns).reshape(-1, 1)
    return x, t, mu, w, mub, mui, tb, xi_


def _to_double(tr):
    """Cast nets + mesh data to float64 IN PLACE for identity testing."""
    tr.net0.double()
    tr.net1.double()
    tr.data = {k: v.double() for k, v in tr.data.items()}
    return tr


class TestResidualChunking:
    def test_default_is_one_and_never_routes_through_chunked(self, tmp_path,
                                                             monkeypatch):
        """(a) K=1 = the EXISTING code path. The chunked machinery is made to
        explode; a default-config train() must complete without touching it.
        """
        from pinn_bte.training.transient.gray.gray_1d import (
            Transient1DGrayTrainer,
        )
        tr = _tiny_trainer(tmp_path, K=None, Nx=6, Nt=6, Ns=4, epochs=2)
        assert tr.residual_chunks == 1

        def boom(*a, **k):
            raise AssertionError("chunked path reached at K=1")
        monkeypatch.setattr(Transient1DGrayTrainer, "_build_residual_chunks",
                            boom)
        monkeypatch.setattr(Transient1DGrayTrainer, "_chunked_backward", boom)
        tr.train()   # would raise if K=1 routed through the new code

    def test_chunk_partition_covers_and_aligns(self, tmp_path):
        import torch
        tr = _tiny_trainer(tmp_path, K=7)
        x, t, mu, *_ = _expanded(tr)
        chunks = tr._build_residual_chunks(x, t, mu)
        assert len(chunks) == 7
        rows = 0
        wsum = 0.0
        for xc, tc, mc, wc in chunks:
            assert xc.shape == tc.shape == mc.shape
            assert xc.shape[0] % tr.Ns == 0
            rows += xc.shape[0]
            wsum += wc
        assert rows == x.shape[0]
        assert wsum == pytest.approx(1.0, rel=0, abs=1e-15)
        assert torch.equal(torch.cat([c[0] for c in chunks]), x)
        assert torch.equal(torch.cat([c[2] for c in chunks]), mu)

    def test_chunked_loss_equals_fullbatch_epoch0_float64(self, tmp_path):
        """(b) epoch-0 identity at the coordinator's 1e-12 bar, in FLOAT64: every
        component of the K=4 accumulated loss equals the full-batch value to <=
        1e-12 relative.
        """
        tr1 = _to_double(_tiny_trainer(tmp_path / "f", K=None))
        tr4 = _to_double(_tiny_trainer(tmp_path / "c", K=4))
        full = [float(v.detach()) for v in tr1.compute_loss(*_expanded(tr1))]
        x, t, mu, w, mub, mui, tb, xi_ = _expanded(tr4)
        chunks = tr4._build_residual_chunks(x, t, mu)
        acc = [float(v) for v in
               tr4._chunked_backward(chunks, w, mub, mui, tb, xi_)]
        for lf, lc in zip(full, acc):
            assert lc == pytest.approx(lf, rel=1e-12)

    def test_chunked_loss_equals_fullbatch_epoch0_float32(self, tmp_path):
        tr1 = _tiny_trainer(tmp_path / "f", K=None)
        tr4 = _tiny_trainer(tmp_path / "c", K=4)
        full = [float(v.detach()) for v in tr1.compute_loss(*_expanded(tr1))]
        x, t, mu, w, mub, mui, tb, xi_ = _expanded(tr4)
        chunks = tr4._build_residual_chunks(x, t, mu)
        acc = [float(v) for v in
               tr4._chunked_backward(chunks, w, mub, mui, tb, xi_)]
        for lf, lc in zip(full, acc):
            assert lc == pytest.approx(lf, rel=2e-6)

    def test_unequal_chunks_use_point_share_not_mean_of_means(self, tmp_path):
        """(c) K=7 over 120 points => unequal chunks (18 vs 17)."""
        tr1 = _to_double(_tiny_trainer(tmp_path / "f", K=None))
        tr7 = _to_double(_tiny_trainer(tmp_path / "c", K=7))
        full = [float(v.detach()) for v in tr1.compute_loss(*_expanded(tr1))]
        x, t, mu, w, mub, mui, tb, xi_ = _expanded(tr7)
        chunks = tr7._build_residual_chunks(x, t, mu)
        sizes = {c[0].shape[0] for c in chunks}
        assert len(sizes) > 1, "test needs UNEQUAL chunks to have teeth"
        acc = [float(v) for v in
               tr7._chunked_backward(chunks, w, mub, mui, tb, xi_)]
        assert acc[0] == pytest.approx(full[0], rel=1e-12)
        assert acc[1] == pytest.approx(full[1], rel=1e-12)
        # the bug this test exists to catch, computed explicitly:
        naive = [np.mean([float(tr7.compute_loss(xc, tc, mc, w, mub, mui,
                                                 tb, xi_)[i].detach())
                          for xc, tc, mc, _ in chunks]) for i in (0, 1)]
        assert abs(naive[0] / full[0] - 1.0) > 1e-10
        assert abs(naive[1] / full[1] - 1.0) > 1e-10

    def test_trained_trajectory_agrees_after_200_epochs(self, tmp_path):
        """(b, trajectory) 200 epochs of Adam from identical init: the K=4 and K=1
        A(t) traces agree to <= 1e-9 max-abs.
        """
        import torch
        from experiments.transient.run_gray_ttg_control import amp_cos
        A = {}
        x_eval = np.linspace(0.0, 1.0, 16, endpoint=False)
        t_norm = np.linspace(0.0, 1.0, 41)
        xg, tg = np.meshgrid(x_eval, t_norm)
        grid = np.stack([xg.ravel(), tg.ravel()], axis=1)
        for label, K in (("k1", None), ("k4", 4)):
            tr = _to_double(_tiny_trainer(tmp_path / label, K=K, epochs=200))
            tr.train()
            tr.net1.eval()
            with torch.no_grad():
                T = tr.net1(torch.tensor(grid, dtype=torch.float64)
                            ).numpy().reshape(41, 16)
            A[label] = amp_cos(T, x_eval)
        assert np.abs(A["k1"] - A["k4"]).max() <= 1e-9

    def test_invalid_chunk_counts_refused(self, tmp_path):
        with pytest.raises(ValueError):
            _tiny_trainer(tmp_path, K=0)
        with pytest.raises(ValueError):
            _tiny_trainer(tmp_path / "b", K=121)   # > Nx*Nt = 120 points
