"""Spectral-PN training harness: configuration, training smoke and archived outputs."""
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn_bte.training.transient.spectral_pn_trainer import (
    SpectralPNConfig,
    build_si_case,
    train_spectral_pn,
)


def _gray_config(**overrides):
    """Tiny gray single-mode problem with the exact Volterra reference."""
    from pinn_bte.utils.plotting_transient import compute_analytical_amplitude
    t_ref = np.linspace(0.0, 6.0, 601)
    A_ref = compute_analytical_amplitude(t_ref, 0.6)
    cfg = dict(
        v=np.array([1.0]), tau=np.array([1.0]), C=np.array([1.0]),
        q=0.6, t_end=6.0, L_max=8, width=48, depth=3,
        epochs=600, lr=2e-3, nt=241, seed=0,
        device="cpu", dtype="float64",
        t_ref=t_ref, A_ref=A_ref, label="gray-smoke",
    )
    cfg.update(overrides)
    return SpectralPNConfig(**cfg)


def test_build_si_case_window_and_reference():
    from pinn_bte.physics.ttg_dispersion import shipped_mode_source
    with shipped_mode_source():
        case = build_si_case(L_um=1.0, Nk=20, n_decay_times=7.0)
    assert case.v.shape == case.tau.shape == case.C.shape == (60,)
    assert (case.v > 0).all() and (case.tau > 0).all() and (case.C > 0).all()
    # q of the 1 um grating in the mode units (Angstrom-based)
    assert np.isclose(case.q, 2 * np.pi / 1e4)
    # window = n_decay / gamma_dom, ~97 tau_ref at L=1.0 (known DOM fact)
    tau_ref = case.C.sum() / (case.C / case.tau).sum()
    assert 90.0 < case.t_end / tau_ref < 105.0
    # reference trace: normalized, starts at 1, spans the window
    assert abs(case.A_ref[0] - 1.0) < 1e-6
    assert case.t_ref[0] == 0.0
    assert np.isclose(case.t_ref[-1], case.t_end, rtol=0.06)


def test_trainer_smoke_gray_residual_drop_and_amplitude():
    res = train_spectral_pn(_gray_config())
    assert res["loss_final"] < 0.02 * res["loss_initial"], "residual did not drop"
    assert res["max_abs_dA"] < 0.08
    assert res["gamma_ratio"] == pytest.approx(1.0, abs=0.12)


def test_trainer_writes_npz_artifact(tmp_path):
    cfg = _gray_config(epochs=30, outdir=tmp_path, run_id="tst0001",
                       extra_meta=dict(tau_ref=1.0, Nk=1, L_um=0.0))
    res = train_spectral_pn(cfg)
    npz = tmp_path / "tst0001_results.npz"
    assert npz.exists()
    d = np.load(npz, allow_pickle=False)
    for key in ("t", "A", "A_ref", "t_ref", "loss_history",
                "gamma_eff", "gamma_ratio", "rmse", "max_abs_dA",
                "L_max", "label", "nt_actual",
                "tau_ref", "Nk", "L_um"):
        assert key in d.files, key
    assert d["A"].shape == d["t"].shape
    assert np.isclose(float(d["gamma_ratio"]), res["gamma_ratio"])
    # the log-mix grid dedups t_end -> record the ACTUAL collocation count
    assert 0 < int(d["nt_actual"]) <= cfg.nt
    # amplitude is normalized (exact IC -> A(0) = 1 by construction)
    assert abs(d["A"][0] - 1.0) < 1e-9


def test_trainer_seed_reproducible():
    r1 = train_spectral_pn(_gray_config(epochs=40, seed=7))
    r2 = train_spectral_pn(_gray_config(epochs=40, seed=7))
    assert np.isclose(r1["loss_final"], r2["loss_final"], rtol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_trainer_cuda_smoke():
    res = train_spectral_pn(_gray_config(epochs=40, device="cuda",
                                         dtype="float32"))
    assert np.isfinite(res["loss_final"])
    assert res["loss_final"] < res["loss_initial"]


def _cli():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_spectral_pn", Path(__file__).parent.parent / "experiments"
        / "transient" / "run_spectral_pn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stiff_diffusive_case():
    from pinn_bte.models.spectral_pn import solve_pn_ode
    from pinn_bte.physics.ttg_dispersion import phonon_modes
    v, tau, C = phonon_modes(4, 300.0)
    q = 2 * np.pi / (10.0 * 1e4)
    gamma_kin = q ** 2 * np.sum(C * v ** 2 * tau) / 3.0 / np.sum(C)
    t_end = 7.0 / gamma_kin
    t_ref = np.linspace(0.0, t_end, 801)
    A_ref = solve_pn_ode(v, tau, C, q=q, t=t_ref, L_max=8)
    return v, tau, C, q, t_end, t_ref, A_ref


def test_ce_formulation_rescues_diffusive_and_stays_uniform():
    v, tau, C, q, t_end, t_ref, A_ref = _stiff_diffusive_case()
    base = dict(v=v, tau=tau, C=C, q=q, t_end=t_end, L_max=8,
                width=48, depth=3, epochs=600, lr=2e-3, nt=201,
                t_grid="uniform", seed=0, device="cpu", dtype="float64",
                t_ref=t_ref, A_ref=A_ref)
    frozen = train_spectral_pn(SpectralPNConfig(**base))
    ce = train_spectral_pn(SpectralPNConfig(**base, formulation="ce"))
    assert frozen["gamma_ratio"] < 0.6
    assert ce["gamma_ratio"] > 0.7, ce["gamma_ratio"]
    assert ce["rmse"] < 0.5 * frozen["rmse"]


def test_ce_formulation_flag_plumbing(tmp_path):
    mod = _cli()
    run_dir = mod.main(["--L", "1.0", "--epochs", "3", "--nt", "64",
                        "--width", "16", "--depth", "2", "--lmax", "4",
                        "--device", "cpu", "--dtype", "float32",
                        "--formulation", "ce", "--outdir", str(tmp_path),
                        "--log-every", "1"])
    d = np.load(next(Path(run_dir).glob("*_results.npz")))
    assert str(d["formulation"]) == "ce"


def test_ap_slow_head_rescues_deep_diffusive_freeze():
    v, tau, C, q, t_end, t_ref, A_ref = _stiff_diffusive_case()
    base = dict(v=v, tau=tau, C=C, q=q, t_end=t_end, L_max=8,
                width=48, depth=3, epochs=600, lr=2e-3, nt=201,
                t_grid="uniform", seed=0, device="cpu", dtype="float64",
                t_ref=t_ref, A_ref=A_ref)
    frozen = train_spectral_pn(SpectralPNConfig(**base))
    rescued = train_spectral_pn(SpectralPNConfig(
        **base, ap_scaling=True, slow_head=True))
    assert frozen["gamma_ratio"] < 0.6, (
        f"expected freeze without the fix, got {frozen['gamma_ratio']:.3f}")
    assert rescued["gamma_ratio"] > 0.75, (
        f"AP+slow-head did not rescue: {rescued['gamma_ratio']:.3f}")
    assert rescued["rmse"] < 0.5 * frozen["rmse"]


def test_ap_slow_head_off_is_default_regression():
    r_default = train_spectral_pn(_gray_config(epochs=40, seed=3))
    r_off = train_spectral_pn(_gray_config(epochs=40, seed=3,
                                           ap_scaling=False,
                                           slow_head=False))
    assert np.isclose(r_default["loss_final"], r_off["loss_final"],
                      rtol=1e-12)


def test_duhamel_free_coefficients_match_spherical_bessel():
    from scipy.special import spherical_jn
    from pinn_bte.models.spectral_pn import duhamel_free_coefficients
    import torch as th
    v = th.tensor([3.0e12], dtype=th.float64)
    tau = th.tensor([2e-11], dtype=th.float64)
    C = th.tensor([0.7], dtype=th.float64)
    q = 2 * np.pi / 1e2                       # deep-ballistic: vq*tau ~ 3.8e3
    t = th.linspace(0.0, 5e-12, 7, dtype=th.float64)
    L_top = 9
    c_f, s_f = duhamel_free_coefficients(t, v, tau, C, q, L_top)
    x = (v * q * t.reshape(-1, 1)).numpy()[:, 0]
    damp = np.exp(-t.numpy() / tau.numpy()[0])
    for l in range(L_top + 1):
        jl = spherical_jn(l, x)
        if l % 2 == 0:
            ref_c = 0.7 * (2 * l + 1) * (-1) ** (l // 2) * jl * damp
            assert np.abs(c_f[:, 0, l].numpy() - ref_c).max() < 1e-10, l
            assert np.abs(s_f[:, 0, l].numpy()).max() < 1e-10, l
        else:
            ref_s = 0.7 * (2 * l + 1) * (-1) ** ((l - 1) // 2) * jl * damp
            assert np.abs(s_f[:, 0, l].numpy() - ref_s).max() < 1e-10, l
            assert np.abs(c_f[:, 0, l].numpy()).max() < 1e-10, l


def test_duhamel_net_exact_ic_and_flag_plumbing(tmp_path):
    """duhamel=True: free part replaces the static IC (j_0(0)=1 -> exact cosine
    IC), gate keeps t=0 exact; CLI flag lands in the npz.
    """
    import torch as th
    from pinn_bte.models.spectral_pn import PNCoefficientNet
    net = PNCoefficientNet(n_modes=1, L_max=6, width=8, depth=2,
                           duhamel=True).double()
    v = th.tensor([1.0], dtype=th.float64)
    tau = th.tensor([1.0], dtype=th.float64)
    C = th.tensor([2.5], dtype=th.float64)
    c, s = net.coefficients(th.zeros(1, dtype=th.float64), v, tau, C, q=3.0)
    assert abs(float(c[0, 0, 0]) - 2.5) < 1e-12
    assert float(th.abs(c[0, 0, 1:]).max()) < 1e-12
    assert float(th.abs(s).max()) < 1e-12
    mod = _cli()
    run_dir = mod.main(["--L", "0.01", "--epochs", "3", "--nt", "64",
                        "--width", "16", "--depth", "2", "--lmax", "4",
                        "--device", "cpu", "--dtype", "float32",
                        "--duhamel", "--outdir", str(tmp_path),
                        "--log-every", "1"])
    d = np.load(next(Path(run_dir).glob("*_results.npz")))
    assert bool(d["duhamel"])


def test_duhamel_rescues_deep_ballistic():
    from pinn_bte.physics.ttg_dom import ttg_amplitude_curve, ttg_decay_spectral
    from pinn_bte.physics.ttg_dispersion import phonon_modes
    v, tau, C = phonon_modes(4, 300.0)
    L_um = 0.01
    q = 2 * np.pi / (L_um * 1e4)
    g_dom = ttg_decay_spectral(L_um, "1d", Nk=4)
    t_end = 7.0 / g_dom
    t_ref, A_ref = ttg_amplitude_curve(L_um, "1d", Nk=4, t_end_s=t_end)
    base = dict(v=v, tau=tau, C=C, q=q, t_end=t_end, L_max=8,
                width=48, depth=3, epochs=800, lr=2e-3, nt=241,
                t_grid="log-mix", seed=0, device="cpu", dtype="float64",
                t_ref=t_ref, A_ref=A_ref, gamma_ref=g_dom)
    plain = train_spectral_pn(SpectralPNConfig(**base))
    duh = train_spectral_pn(SpectralPNConfig(**base, duhamel=True))
    assert abs(plain["gamma_ratio"] - 1.0) > 0.15, "plain PN unexpectedly ok"
    assert abs(duh["gamma_ratio"] - 1.0) < 0.15, duh["gamma_ratio"]
    assert duh["rmse"] < 0.05
    assert duh["rmse"] < 0.25 * plain["rmse"]


def test_cli_full_si_integration(tmp_path):
    """3-epoch end-to-end: full-Si case build -> train -> npz on disk."""
    mod = _cli()
    run_dir = mod.main(["--L", "1.0", "--epochs", "3", "--nt", "64",
                        "--width", "16", "--depth", "2", "--lmax", "4",
                        "--device", "cpu", "--dtype", "float32",
                        "--outdir", str(tmp_path), "--log-every", "1"])
    npzs = list(Path(run_dir).glob("*_results.npz"))
    assert len(npzs) == 1
    d = np.load(npzs[0])
    assert d["A"].shape == d["t"].shape
    assert np.isfinite(float(d["gamma_ratio"]))
    assert str(d["label"]) .startswith("spn-L1.0")


def test_cli_ap_flags_and_2d_geometry(tmp_path):
    mod = _cli()
    run_dir = mod.main(["--L", "1.0", "--epochs", "3", "--nt", "64",
                        "--width", "16", "--depth", "2", "--lmax", "4",
                        "--device", "cpu", "--dtype", "float32",
                        "--geometry", "2d", "--ap-scaling", "--slow-head",
                        "--outdir", str(tmp_path), "--log-every", "1"])
    d = np.load(next(Path(run_dir).glob("*_results.npz")))
    assert bool(d["ap_scaling"]) and bool(d["slow_head"])
    assert str(d["geometry"]) == "2d"


def test_build_si_case_2d_matches_1d():
    """TTG 1d == 2d at the DOM level (uniform Omega_x marginal)."""
    c1 = build_si_case(1.0, Nk=10, geometry="1d")
    c2 = build_si_case(1.0, Nk=10, geometry="2d")
    assert np.isclose(c2.gamma_dom_hz, c1.gamma_dom_hz, rtol=1e-4)
    assert np.abs(np.interp(c1.t_ref, c2.t_ref, c2.A_ref)
                  - c1.A_ref).max() < 1e-5


def test_xihybrid_formulation_flag_plumbing(tmp_path):
    mod = _cli()
    run_dir = mod.main(["--L", "1.0", "--epochs", "3", "--nt", "64",
                        "--width", "16", "--depth", "2", "--lmax", "4",
                        "--device", "cpu", "--dtype", "float32",
                        "--formulation", "xihybrid", "--xi-split", "2.5",
                        "--xi-duh", "7.5", "--shared-ahat",
                        "--outdir", str(tmp_path), "--log-every", "1"])
    d = np.load(next(Path(run_dir).glob("*_results.npz")))
    assert str(d["formulation"]) == "xihybrid"
    assert float(d["xi_split"]) == 2.5
    assert float(d["xi_duh"]) == 7.5
    assert bool(d["shared_ahat"])


def test_xihybrid_trainer_smoke_mixed_spectrum():
    """Two gray modes straddling the split (xi = 0.1 and 10): one formulation
    covers both regimes; residual drops, amplitude lands on the exact expm
    reference.
    """
    from pinn_bte.models.spectral_pn import solve_pn_ode
    v = np.array([0.1, 10.0]); tau = np.array([1.0, 1.0])
    C = np.array([1.0, 1.0])
    t_ref = np.linspace(0.0, 6.0, 601)
    A_ref = solve_pn_ode(v, tau, C, q=1.0, t=t_ref, L_max=24)
    res = train_spectral_pn(SpectralPNConfig(
        v=v, tau=tau, C=C, q=1.0, t_end=6.0, L_max=8, width=48, depth=3,
        epochs=600, lr=2e-3, nt=241, seed=0, device="cpu", dtype="float64",
        t_ref=t_ref, A_ref=A_ref, label="xihybrid-smoke",
        formulation="xihybrid", xi_split=1.0))
    assert res["loss_final"] < 0.05 * res["loss_initial"]
    assert res["max_abs_dA"] < 0.08
