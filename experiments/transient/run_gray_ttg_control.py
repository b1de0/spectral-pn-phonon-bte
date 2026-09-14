#!/usr/bin/env python3
"""Gray-BTE sampled-residual control on the transient thermal grating: trains
the gray 1D trainer at a chosen xi and scores its amplitude against the exact
gray reference.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from pinn_bte.physics.decay_metrics import (
    GAMMA_ESTIMATOR_RULED,
    integral_gamma,
)
from pinn_bte.physics.gray_ttg import (
    ZHOU_DT_TRAIN_MAX_TAU,
    build_gray_case,
    gray_gamma_same_window,
    gray_window_gamma,
)
from pinn_bte.training.transient.spectral_pn_trainer import (
    SHAPE_NAMES,
    _shape_metrics,
    _sign_provenance,
)
from pinn_bte.training.transient.gray.gray_1d import Transient1DGrayTrainer

NET_ARCH = "mlp"
HIDDEN_DIM = 30
NUM_LAYERS = 5
SNAPSHOT_SELECTION = "final"   # trainer saves the final-epoch state last


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--xi", type=float, required=True,
                   help="rarefaction parameter xi = 2*pi*Kn (Zhou: 0.6/4/36)")
    p.add_argument("--window", choices=("ndecay", "zhou"), default="ndecay",
                   help="'ndecay': t_end = n_decay/gamma_window(xi) (repo "
                        "convention); 'zhou': explicit --window-tau "
                        "(Zhou's published training window)")
    p.add_argument("--n-decay", type=float, default=7.0,
                   help="window depth in units of 1/gamma_window (repo "
                        "convention: 7)")
    p.add_argument("--window-tau", type=float, default=None,
                   help="explicit window in tau units (REQUIRED with "
                        "--window zhou, REFUSED with --window ndecay)")
    p.add_argument("--epochs", type=int, default=20000,
                   help="Adam epochs (control budget: 20000)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam learning rate (Zhou Methods: 1e-3; the "
                        "trainer's own default 4e-3 is NOT used)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nx", type=int, default=60,
                   help="spatial training points (Zhou Table 1: 60)")
    p.add_argument("--nt-train", type=int, default=0,
                   help="time training points; 0 = auto rule "
                        "max(60, ceil(t_end/0.05)+1): never coarser in "
                        "absolute dt than Zhou's coarsest published grid, "
                        "never fewer per window than Zhou's 60")
    p.add_argument("--ns", type=int, default=16,
                   help="Gauss-Legendre angular points (Zhou Table 1: 16)")
    p.add_argument("--eval-nt", type=int, default=801,
                   help="uniform evaluation grid for A(t) (>= 801 keeps the "
                        "run-grid quadrature question inert)")
    p.add_argument("--eval-nx", type=int, default=64,
                   help="spatial evaluation points (endpoint=False, "
                        "periodic cos projection)")
    p.add_argument("--arbiter-nsteps", type=int, default=12000,
                   help="native arbiter grid (house convention: 12000 steps "
                        "= 12001 samples)")
    p.add_argument("--residual-chunks", type=int, default=1,
                   help="gradient-accumulation chunks K for the interior "
                        "residual (OOM remedy: 1,094,400 rows "
                        "needed > 14.31 GiB at K=1 on the 15.44 GiB card). "
                        "Exact-arithmetic-equivalent to full batch (chunk "
                        "means weighted by point share); K=1 (default) is "
                        "the pre-existing code path untouched")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", default=None,
                   help="output directory used VERBATIM (queue passes the "
                        "arm dir); default runs/transient/gray_ttg_control/"
                        "<timestamp>")
    p.add_argument("--no-wandb", action="store_true",
                   help="disable W&B tracking (tests / offline queue)")
    p.add_argument("--tag", default="",
                   help="free-form arm label recorded in params/npz")
    return p.parse_args(argv)


def resolve_nt_train(nt_train: int, t_end_tau: float) -> int:
    """The auto-Nt rule (0 = auto), or the explicit value verbatim."""
    if nt_train:
        return int(nt_train)
    return max(60, int(np.ceil(t_end_tau / ZHOU_DT_TRAIN_MAX_TAU)) + 1)


def amp_cos(T: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fundamental cosine amplitude of T (Nt, Nx): 2*<T cos(2 pi x)>
    (x periodic, endpoint=False).
    """
    return 2.0 * np.mean(T * np.cos(2.0 * np.pi * x)[None, :], axis=1)


def _evaluate_T(trainer: Transient1DGrayTrainer, t_norm: np.ndarray,
                x: np.ndarray) -> np.ndarray:
    """T(x, t) from net1 on the (t_norm, x) product grid -> (Nt, Nx)."""
    trainer.net1.eval()
    xg, tg = np.meshgrid(x, t_norm)                       # (Nt, Nx) each
    inp = torch.tensor(
        np.stack([xg.ravel(), tg.ravel()], axis=1), dtype=torch.float32,
        device=trainer.device)
    with torch.no_grad():
        T = trainer.net1(inp).cpu().numpy().reshape(len(t_norm), len(x))
    return T.astype(np.float64)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            capture_output=True, timeout=10).stdout.strip() or "unknown"
    except Exception:                                     # noqa: BLE001
        return "unknown"


def run_control(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)

    # ---- window ----------------------------------------------------------
    if args.window == "zhou" and args.window_tau is None:
        raise SystemExit("--window zhou requires an explicit "
                         "--window-tau (Zhou's Table 1 value, e.g. 3.0)")
    if args.window == "ndecay" and args.window_tau is not None:
        raise SystemExit("--window-tau with --window ndecay is ambiguous "
                         "(the window is n_decay/gamma by rule); refusing "
                         "rather than guessing which one you meant")

    # the window rule itself: what the ndecay arms train on, and what the
    # arbiter's own-window rate below is keyed to on EVERY arm.
    t_end_ndecay = float(args.n_decay) / gray_window_gamma(args.xi)
    t_end_tau = (float(args.window_tau) if args.window == "zhou"
                 else t_end_ndecay)

    case = build_gray_case(args.xi, n_decay=args.n_decay,
                           t_end_tau=t_end_tau, nsteps=args.arbiter_nsteps)
    gamma_dom = gray_gamma_same_window(args.xi, t_end_ndecay,
                                       nsteps=args.arbiter_nsteps)

    nt_train = resolve_nt_train(args.nt_train, t_end_tau)

    outdir = Path(args.outdir) if args.outdir else (
        REPO_ROOT / "runs/transient/gray_ttg_control"
        / datetime.now().strftime("%Y%m%d_%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- trainer (house gray trainer, UNTOUCHED) -------------------------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wandb_was = Transient1DGrayTrainer._wandb_enabled
    if args.no_wandb:
        Transient1DGrayTrainer._wandb_enabled = False
    try:
        trainer = Transient1DGrayTrainer(
            eta=args.xi, window_tau=t_end_tau, epochs=args.epochs,
            learning_rate=args.lr, Nx=args.nx, Nt=nt_train, Ns=args.ns,
            device=device, output_dir=outdir,
            residual_chunks=args.residual_chunks)
        t0 = time.time()
        loss_history = trainer.train()
        wall_s = time.time() - t0

        # ---- evaluation (final-epoch state; dense uniform grid) ----------
        x_eval = np.linspace(0.0, 1.0, args.eval_nx, endpoint=False)
        t_norm = np.linspace(0.0, 1.0, args.eval_nt)
        T = _evaluate_T(trainer, t_norm, x_eval)
    finally:
        Transient1DGrayTrainer._wandb_enabled = wandb_was

    t_tau = t_norm * t_end_tau
    A = amp_cos(T, x_eval)
    A_ref_i = np.interp(t_tau, case.t_ref, case.A_ref)

    gamma_eff = integral_gamma(t_tau, A, estimator=GAMMA_ESTIMATOR_RULED)
    gamma_ratio = gamma_eff / case.gamma_ref_per_tau
    shape = dict(zip(SHAPE_NAMES, _shape_metrics(t_tau, A, A_ref_i)))
    sign_prov = _sign_provenance(t_tau, A)
    field_rmse = float(np.sqrt(np.mean(
        (T - A_ref_i[:, None] * np.cos(2.0 * np.pi * x_eval)[None, :]) ** 2)))

    loss_hist = np.asarray(loss_history, dtype=float)     # (epochs, 6)
    run_id = trainer.run_id

    assert case.provenance["xi"] == args.xi
    prov = {k: v for k, v in case.provenance.items() if k != "xi"}

    results = dict(
        t=t_tau, A=A, t_ref=case.t_ref, A_ref=case.A_ref,
        A_ref_interp=A_ref_i, T_field=T, x_field=x_eval,
        gamma_eff=gamma_eff, gamma_ratio=gamma_ratio,
        gamma_dom_hz=gamma_dom, field_rmse=field_rmse, **shape, **sign_prov,
        **prov,
        gamma_dom_window_n_decay=float(args.n_decay),
        gamma_dominant_per_tau=(np.nan if case.gamma_dominant_per_tau is None
                                else case.gamma_dominant_per_tau),
        gamma_window_per_tau=case.gamma_window_per_tau,
        xi=args.xi, Kn=case.Kn, n_decay=args.n_decay,
        window_mode=args.window, window_tau=t_end_tau,
        epochs=args.epochs, lr=args.lr, seed=args.seed,
        nx=args.nx, nt_train=nt_train, ns=args.ns,
        eval_nt=args.eval_nt, eval_nx=args.eval_nx,
        arbiter_nsteps=args.arbiter_nsteps,
        residual_chunks=args.residual_chunks,
        net_arch=NET_ARCH, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        snapshot_selection=SNAPSHOT_SELECTION, tag=args.tag,
        model="gray", trainer="Transient1DGrayTrainer",
        loss_history=loss_hist,
        loss_component_names=np.asarray(
            trainer.loss_names + ["total"], dtype=object),
        wall_s=wall_s,
    )
    np.savez(outdir / f"{run_id}_results.npz", **results)

    params = dict(
        args=vars(args),
        derived=dict(
            t_end_tau=t_end_tau, nt_train_effective=nt_train,
            gamma_ref_per_tau=case.gamma_ref_per_tau,
            gamma_dominant_per_tau=case.gamma_dominant_per_tau,
            gamma_window_per_tau=case.gamma_window_per_tau,
            gamma_dom_own_window_per_tau=gamma_dom,
            gamma_eff_per_tau=gamma_eff, gamma_ratio=gamma_ratio,
            rmse=shape["rmse"], field_rmse=field_rmse,
            A0=float(A[0]), A_min=float(A.min()),
            Kn=case.Kn, run_id=run_id, outdir=str(outdir),
            device=str(device), wall_s=wall_s, git_sha=_git_sha(),
            dt_train_tau=t_end_tau / (nt_train - 1),
            zhou_dt_train_max_tau=ZHOU_DT_TRAIN_MAX_TAU,
        ),
    )
    (outdir / f"{run_id}_params.json").write_text(json.dumps(params, indent=2))

    print(f"[gray-ttg] xi={args.xi} window={args.window} "
          f"t_end={t_end_tau:.4f} tau  nt_train={nt_train}  "
          f"gamma_eff={gamma_eff:.6f}/tau  "
          f"gamma_ref={case.gamma_ref_per_tau:.6f}/tau  "
          f"gamma_ratio={gamma_ratio:.4f}  rmse={shape['rmse']:.5f}  "
          f"field_rmse={field_rmse:.5f}  A0={A[0]:.4f}", flush=True)
    print(f"[gray-ttg] saved: {outdir}", flush=True)
    return results


if __name__ == "__main__":
    run_control()
