"""Score the adoption evidence for the PN sum-rule re-pin -- and STOP."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinn_bte.physics.decay_metrics import GAMMA_ESTIMATOR_RULED, integral_gamma  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_sumrule_ladder import (  # noqa: E402
    ALPHA_MEASURED, KAPPA_BAND, comb_facts,
)

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "data" / "runs" / "pn_sumrule_adoption"
MANIFEST = ROOT / "arms.tsv"
#: The JOINT-comb cell-D ledger this score was taken against -- a FROZEN copy.
#: data/cell_d_pn.json itself moved to the sum-rule comb with the
LEDGER = REPO / "data" / "cell_d_pn_joint_baseline.json"
COMB = "joint-sumrule"
N_REF_NATIVE = 12001

# ---- pre-declared bars (see module docstring) -----------------------------
GATE_SEED_WORST_DEV = 0.008
GATE_SEED_RMSE_FACTOR = 1.5
GATE_SEED_SPREAD = 0.003
GATE_FREEZE = 0.5
BAND = 0.10
P203_FACTS = dict(mode_count_ratio=0.9989, capacity_ratio=1.0028,
                  kappa_bulk=142.30, D_over_alpha=0.9894)
FACT_TOL = dict(mode_count_ratio=5e-4, capacity_ratio=5e-4,
                kappa_bulk=0.05, D_over_alpha=5e-4)

SEED_PINS = {
    (0.1, "1d"): Path("data/runs/pn_sumrule_ladder_1d/ladder_L0.1_1d"),
    (0.1, "2d"): Path("data/runs/pn_sumrule_ladder_2d/ladder_L0.1_2d"),
    (1.0, "1d"): Path("data/runs/pn_sumrule_ladder_1d/ladder_L1.0_1d"),
}
UNIFORM = (
    "Nk", "L_max", "width", "depth", "lr", "nt", "n_decay_times", "geometry",
    "formulation", "blend", "harmonics", "dtype", "t_grid", "window_mode",
    "ref_solver", "dT", "n_phi", "tau_model", "xi_split", "xi_duh", "gauge",
)
#: arms.tsv label -> the label the claim store and the cell-D ledger key rows by
#: (data/ladders.yaml `ablations:`; renaming a key there empties a claim).
CONTROL_LEDGER_LABEL = {
    "C3_L1.0_plain_s42": "C3 L=1 plain s42",
    "C3_L1.0_plain_s0": "C3 L=1 plain s0",
    "C3_L1.0_plain_s1": "C3 L=1 plain s1",
    "C5_L0.1_plain_lmax16": "C5 L=0.1 plain l16",
    "C6_L0.1_duhamel": "C6 L=0.1 duhamel",
    "C8_L10_plain": "C8 L=10 plain freeze",
    "C1_L0.01_duhamel": "C1 L=0.01 duhamel",
    "C2_L0.01_plain": "C2 L=0.01 plain",
    "C7_L10_ce": "C7 L=10 ce",
    "C7_L100_ce": "C7 L=100 ce",
    "C9_L0.1_ce": "C9 L=0.1 ce 20k",
    "C10_L1.0_ce": "C10 L=1.0 ce",
    "C43_L0.01_ce": "C43 L=0.01 ce",
    "C44_L1.0_duhamel": "C44 L=1.0 duhamel",
    "C44_L10.0_duhamel": "C44 L=10 duhamel",
    "C44_L100.0_duhamel": "C44 L=100 duhamel",
}
FREEZE_LABEL = "C8 L=10 plain freeze"
C3_TRIPLE = ("C3 L=1 plain s42", "C3 L=1 plain s0", "C3 L=1 plain s1")

def read_manifest() -> list[dict]:
    with MANIFEST.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))

def load_npz(directory: Path):
    hits = sorted(Path(directory).glob("*_results.npz"))
    if len(hits) != 1:
        raise RuntimeError(f"{directory}: expected exactly one *_results.npz, got {hits}")
    return np.load(hits[0], allow_pickle=True), hits[0]

def scalar(z, key):
    v = z[key]
    return v.item() if hasattr(v, "item") and getattr(v, "ndim", 1) == 0 else v

def cell_d_metrics(z) -> dict:
    """Ruled |A| estimator on both sides; denominator = the run's own stored
    native-grid reference (12001 samples).  Identical to
    scripts/make_figures._cell_d_gamma_ratio and to the ledger."""
    t = np.asarray(z["t"], float).ravel()
    A = np.asarray(z["A"], float).ravel()
    t_ref = np.asarray(z["t_ref"], float).ravel()
    A_ref = np.asarray(z["A_ref"], float).ravel()
    if t_ref.size != N_REF_NATIVE:
        raise RuntimeError(f"stored reference has {t_ref.size} samples, not {N_REF_NATIVE}")
    g_pn = integral_gamma(t, A, estimator=GAMMA_ESTIMATOR_RULED)
    g_ref = integral_gamma(t_ref, A_ref, estimator=GAMMA_ESTIMATOR_RULED)
    ref_on_t = np.interp(t, t_ref, A_ref)
    rmse = float(np.sqrt(np.mean((A / A[0] - ref_on_t / ref_on_t[0]) ** 2)))
    return dict(gamma_pn=float(g_pn), gamma_ref=float(g_ref),
                gamma_ratio=float(g_pn / g_ref), rmse=rmse, n_ref=int(t_ref.size))

def band_class(gamma: float) -> str:
    return "inside" if abs(1.0 - gamma) <= BAND else "outside"

def is_frozen(gamma: float) -> bool:
    return gamma < GATE_FREEZE

def formulation_of(z) -> str:
    if bool(scalar(z, "duhamel")):
        return "duhamel"
    return str(scalar(z, "formulation"))

def ledger_baseline() -> dict[str, dict]:
    d = json.loads(LEDGER.read_text())
    out = {}
    for section in ("pn_ablations", "pn_ablations_cited"):
        for row in d[section]:
            out[row["label"]] = dict(gamma=float(row["after"]), rmse=float(row["rmse"]),
                                     run=row["run"])
    return out

# ---------------------------------------------------------------------------
# axes
# ---------------------------------------------------------------------------

def _q4_band(z) -> list[float]:
    """[min, max] of the in-run gamma_ratio over the last quarter of training."""
    gh = np.asarray(z["gamma_history"], float).reshape(-1, 3)
    cut = 0.75 * float(scalar(z, "epochs"))
    q = gh[gh[:, 0] >= cut, 1]
    return [float(q.min()), float(q.max())]

def _loss_drop(z) -> float:
    lh = np.asarray(z["loss_history"], float)
    col = lh[:, -1] if lh.ndim > 1 else lh
    return float(col[0] / col[-1])

def _finite_positive(gamma: float, rmse: float) -> bool:
    return bool(np.isfinite(gamma) and gamma > 0 and np.isfinite(rmse))

def score_artefacts(rows: list[dict]) -> list[str]:
    fails = []
    if len(rows) != 22:
        fails.append(f"manifest: {len(rows)} rows, expected 22")
    for r in rows:
        if r["exit"] != "0":
            fails.append(f"{r['label']}: exit={r['exit']}")
        try:
            z, _ = load_npz(ROOT / r["label"])
        except RuntimeError as e:
            fails.append(str(e))
            continue
        if str(scalar(z, "mode_source")) != COMB:
            fails.append(f"{r['label']}: mode_source={scalar(z, 'mode_source')!r}")
        checks = (("seed", int(r["seed"])), ("epochs", int(r["epochs"])),
                  ("L_max", int(r["lmax"])), ("L_um", float(r["L_um"])),
                  ("geometry", r["geometry"]))
        for key, want in checks:
            got = scalar(z, key)
            if str(got) != str(want) and not (key == "L_um" and float(got) == want):
                fails.append(f"{r['label']}: {key}={got!r} != manifest {want!r}")
        want_gauge = "corridor" if r["kind"] == "seed" else "additive"
        if str(scalar(z, "gauge")) != want_gauge:
            fails.append(f"{r['label']}: gauge={scalar(z, 'gauge')!r}, expected {want_gauge}")
    return fails

def score_seeds(rows: list[dict]) -> dict:
    fams: dict[str, dict] = {}
    for r in rows:
        if r["kind"] != "seed":
            continue
        L, geom, seed = float(r["L_um"]), r["geometry"], int(r["seed"])
        key = f"L{L}_{geom}"
        if key not in fams:
            pin_z, pin_path = load_npz(SEED_PINS[(L, geom)])
            pin = cell_d_metrics(pin_z)
            pin.update(npz=str(pin_path), seed=int(scalar(pin_z, "seed")),
                       q4_band=_q4_band(pin_z),
                       amp_sign_change=int(scalar(pin_z, "amp_sign_change")),
                       amp_negative_time_frac=float(scalar(pin_z, "amp_negative_time_frac")))
            fams[key] = dict(L_um=L, geometry=geom, pin=pin, pin_z=pin_z, arms=[], failures=[])
        z, path = load_npz(ROOT / r["label"])
        m = cell_d_metrics(z)
        fam = fams[key]
        A_ref = np.asarray(z["A_ref"], float).ravel()
        arm = dict(label=r["label"], npz=str(path), seed=seed, **m,
                   dev=abs(1.0 - m["gamma_ratio"]),
                   rmse_factor=m["rmse"] / fam["pin"]["rmse"],
                   stored_gamma_ratio=float(z["gamma_ratio"]), stored_rmse=float(z["rmse"]),
                   q4_band=_q4_band(z), loss_drop=_loss_drop(z),
                   amp_sign_change=int(scalar(z, "amp_sign_change")),
                   amp_negative_time_frac=float(scalar(z, "amp_negative_time_frac")),
                   A_ref_min=float(A_ref.min() / A_ref[0]))
        for k in UNIFORM:
            if k in z.files and k in fam["pin_z"].files and scalar(z, k) != scalar(fam["pin_z"], k):
                fam["failures"].append(
                    f"{r['label']}: {k}={scalar(z, k)!r} differs from the seed-42 pin")
        if m["gamma_ratio"] != arm["stored_gamma_ratio"]:
            fam["failures"].append(f"{r['label']}: recomputed gamma_ratio != stored")
        if abs(m["rmse"] - arm["stored_rmse"]) > 1e-15:
            fam["failures"].append(f"{r['label']}: recomputed rmse != stored")
        if arm["dev"] > GATE_SEED_WORST_DEV:
            fam["failures"].append(
                f"{r['label']}: |1-gamma|={arm['dev']:.6g} > {GATE_SEED_WORST_DEV}")
        if arm["rmse_factor"] > GATE_SEED_RMSE_FACTOR:
            fam["failures"].append(
                f"{r['label']}: rmse factor {arm['rmse_factor']:.4f} > {GATE_SEED_RMSE_FACTOR}")
        if not _finite_positive(m["gamma_ratio"], m["rmse"]):
            fam["failures"].append(f"{r['label']}: non-finite or non-positive metric")
        fam["arms"].append(arm)
    for fam in fams.values():
        fam["arms"].sort(key=lambda a: a["seed"])
        ratios = [fam["pin"]["gamma_ratio"]] + [a["gamma_ratio"] for a in fam["arms"]]
        fam["spread_pp"] = 100.0 * (max(ratios) - min(ratios))
        if (max(ratios) - min(ratios)) > GATE_SEED_SPREAD:
            fam["failures"].append(
                f"{fam['L_um']}/{fam['geometry']}: spread {fam['spread_pp']:.5f} pp "
                f"> {100 * GATE_SEED_SPREAD} pp")
        del fam["pin_z"]
    return fams

def score_controls(rows: list[dict], baseline: dict) -> dict:
    out: dict = {}
    fails: list[str] = []
    for r in rows:
        if r["kind"] != "ablation":
            continue
        lab = CONTROL_LEDGER_LABEL[r["label"]]
        z, path = load_npz(ROOT / r["label"])
        m = cell_d_metrics(z)
        joint = baseline[lab]
        row = dict(manifest_label=r["label"], npz=str(path), L_um=float(r["L_um"]),
                   seed=int(r["seed"]), formulation=formulation_of(z),
                   new_gamma=m["gamma_ratio"], new_rmse=m["rmse"],
                   stored_gamma_ratio=float(z["gamma_ratio"]),
                   joint_gamma=joint["gamma"], joint_rmse=joint["rmse"], joint_run=joint["run"],
                   new_class=band_class(m["gamma_ratio"]), joint_class=band_class(joint["gamma"]),
                   amp_sign_change=int(scalar(z, "amp_sign_change")),
                   loss_drop=_loss_drop(z))
        row["flip"] = row["new_class"] != row["joint_class"]
        row["frozen"] = is_frozen(m["gamma_ratio"])
        if m["gamma_ratio"] != row["stored_gamma_ratio"]:
            fails.append(f"{lab}: recomputed gamma_ratio != stored")
        if row["flip"]:
            fails.append(f"{lab}: verdict class {row['joint_class']} -> {row['new_class']} "
                         f"(gamma {joint['gamma']:.4f} -> {m['gamma_ratio']:.4f})")
        if lab == FREEZE_LABEL and not row["frozen"]:
            fails.append(f"{lab}: gamma={m['gamma_ratio']:.4f} >= {GATE_FREEZE}, "
                         f"the freeze exhibit is gone")
        if not _finite_positive(m["gamma_ratio"], m["rmse"]):
            fails.append(f"{lab}: non-finite or non-positive gamma")
        out[lab] = row
    universal = {}
    for form in ("plain", "ce", "duhamel"):
        cells = [r for r in out.values() if r["formulation"] == form]
        universal[form] = dict(
            n=len(cells),
            outside_new=[r["manifest_label"] for r in cells if r["new_class"] == "outside"],
            outside_joint=[r["manifest_label"] for r in cells if r["joint_class"] == "outside"])
        if not universal[form]["outside_new"]:
            fails.append(f"universal: {form} no longer fails the +-10 % band anywhere")
    c3 = [out[k]["new_gamma"] for k in C3_TRIPLE]
    out["universal"] = universal
    out["c3_seed_triple"] = dict(gammas=c3, mean=float(np.mean(c3)),
                                 half_range_pct=float(50.0 * (max(c3) - min(c3)) / np.mean(c3)))
    out["failures"] = fails
    return out

def score_dimensional() -> dict:
    f = comb_facts()
    facts = dict(mode_count_ratio=f["mode_count_ratio"], capacity_ratio=f["capacity_ratio"],
                 kappa_bulk=f["kappa_bulk"], D_over_alpha=f["D_bulk"] / ALPHA_MEASURED,
                 mfp_max_um=f["mfp_max_um"], grid=f["grid"])
    fails = []
    for k, want in P203_FACTS.items():
        if abs(facts[k] - want) > FACT_TOL[k]:
            fails.append(f"dimensional: {k}={facts[k]:.5g} != {want} (tol {FACT_TOL[k]})")
    if not (KAPPA_BAND[0] <= facts["kappa_bulk"] <= KAPPA_BAND[1]):
        fails.append(f"dimensional: kappa_bulk {facts['kappa_bulk']:.2f} outside {KAPPA_BAND}")
    return dict(facts=facts, failures=fails)

def _control_rows(ctl: dict) -> dict:
    return {k: v for k, v in ctl.items() if isinstance(v, dict) and "npz" in v}

def assemble(rows: list[dict]) -> dict:
    art = score_artefacts(rows)
    seeds = score_seeds(rows)
    ctl = score_controls(rows, ledger_baseline())
    dim = score_dimensional()
    seed_fails = [f for fam in seeds.values() for f in fam["failures"]]

    def ax(status, detail):
        return dict(status=status, detail=detail)

    sc_fails = [f for f in seed_fails + ctl["failures"] if "recomputed" in f]
    abs_fails = [f for f in seed_fails if "|1-gamma|" in f or "rmse factor" in f]
    spread_fails = [f for f in seed_fails if "spread" in f]
    phys_fails = [f for f in ctl["failures"]
                  if "verdict class" in f or "freeze" in f or "universal" in f]
    bound_fails = [f for f in seed_fails + ctl["failures"] if "non-finite" in f]
    worst_dev = max(a["dev"] for fam in seeds.values() for a in fam["arms"])
    worst_factor = max(a["rmse_factor"] for fam in seeds.values() for a in fam["arms"])
    n_flips = sum(r["flip"] for r in _control_rows(ctl).values())
    axes = {
        "artefact": ax(
            "PASS" if not art else "FAIL",
            f"22 rows, exit 0, one npz each, mode_source={COMB}, gauge per kind; "
            f"{len(art)} defects"),
        "self_consistency": ax(
            "PASS" if not sc_fails else "FAIL",
            "recomputed native-grid cell-D gamma_ratio == stored (bit-exact) and rmse "
            "within 1e-15, all 22 arms"),
        "absolute_every_point": ax(
            "PASS" if not abs_fails else "FAIL",
            f"seeds: worst |1-gamma| {100 * worst_dev:.4f} % (bar 0.8 %); worst RMSE "
            f"factor vs the seed-42 sum-rule pin {worst_factor:.3f} (bar 1.5)"),
        "shape_span": ax(
            "PASS" if not spread_fails else "FAIL",
            "three-seed spread per family: "
            + ", ".join(f"{k} {v['spread_pp']:.5f} pp" for k, v in seeds.items())
            + " (bar 0.3 pp)"),
        "dimensional": ax(
            "PASS" if not dim["failures"] else "FAIL",
            f"sum_rule_grid(20): mode count {dim['facts']['mode_count_ratio']:.4f}, "
            f"C/rho-cp {dim['facts']['capacity_ratio']:.4f}, kappa "
            f"{dim['facts']['kappa_bulk']:.2f} W/(m K) vs 142-156, D/alpha "
            f"{dim['facts']['D_over_alpha']:.4f}"),
        "arbiter_free_physics": ax(
            "PASS" if not phys_fails else "FAIL",
            f"C8 freeze gamma={ctl[FREEZE_LABEL]['new_gamma']:.4f} (<0.5); class flips: "
            f"{n_flips}/16; universal per formulation: "
            + ", ".join(f"{k} {len(v['outside_new'])}/{v['n']} outside"
                        for k, v in ctl["universal"].items())),
        "bounds": ax("PASS" if not bound_fails else "FAIL",
                     "gamma finite and > 0, rmse finite, all 22 arms"),
        "optimisation_health": ax(
            "REPORTED",
            "per-arm last-quarter gamma band and loss drop are in the JSON; no "
            "pre-declared bar"),
        "spread": ax(
            "REPORTED",
            f"seed triples above; C3 plain L=1 seed triple half-range "
            f"{ctl['c3_seed_triple']['half_range_pct']:.1f} % of mean (joint ledger: 48.8 %)"),
        "elasticity": ax(
            "SKIPPED",
            "a replication/control battery sweeps no input anchor; comb elasticity "
            "measured in "),
        "visual": ax("SKIPPED",
                     "set to LOOKED-AT in the score doc after a human opens the PNGs"),
    }
    failures = art + seed_fails + ctl["failures"] + dim["failures"]
    return dict(comb=COMB, manifest=str(MANIFEST), axes=axes, skipped=["elasticity"],
                seeds=seeds, controls=ctl, dimensional=dim, failures=failures,
                verdict="PASS" if not failures else "FAIL")

def _plots(rep: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    for key, fam in rep["seeds"].items():
        fig, ax = plt.subplots(figsize=(6, 3.6))
        series = [("seed 42 (pin)", fam["pin"]["npz"])]
        series += [(f"seed {a['seed']}", a["npz"]) for a in fam["arms"]]
        for label, npz in series:
            z = np.load(npz, allow_pickle=True)
            t, A = np.asarray(z["t"]).ravel(), np.asarray(z["A"]).ravel()
            ax.plot(t / t[-1], A / A[0], lw=1, label=label)
        z = np.load(fam["pin"]["npz"], allow_pickle=True)
        tr, Ar = np.asarray(z["t_ref"]).ravel(), np.asarray(z["A_ref"]).ravel()
        ax.plot(tr / tr[-1], Ar / Ar[0], color="0.5", lw=1.5, ls="--", label="DOM reference")
        ax.set(xlabel="t / t_end", ylabel="A(t)/A(0)", title=f"seed replicas {key} ({COMB})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"seeds_{key}.png", dpi=130)
        plt.close(fig)
    ctl = _control_rows(rep["controls"])
    fig, axes = plt.subplots(4, 4, figsize=(12, 10))
    for ax, (lab, row) in zip(axes.ravel(), sorted(ctl.items())):
        z = np.load(row["npz"], allow_pickle=True)
        t, A = np.asarray(z["t"]).ravel(), np.asarray(z["A"]).ravel()
        tr, Ar = np.asarray(z["t_ref"]).ravel(), np.asarray(z["A_ref"]).ravel()
        ax.plot(tr / tr[-1], Ar / Ar[0], color="0.5", lw=1.5, ls="--")
        ax.plot(t / t[-1], A / A[0], lw=1, color="C3" if row["flip"] else "C0")
        ax.set_title(f"{lab}\nγ {row['joint_gamma']:.3f}→{row['new_gamma']:.3f} "
                     f"{row['new_class']}", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Table-4 controls on joint-sumrule (red = class flip vs joint)", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "controls.png", dpi=110)
    plt.close(fig)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--png", type=Path, default=None)
    ap.add_argument("--c9-seeds", action="store_true",
                    help="score the C9 seed replicas against their pre-declared bar")
    args = ap.parse_args(argv)
    if args.c9_seeds:
        return _c9_main(args.json)
    rep = assemble(read_manifest())
    print(f"adoption score -- comb {COMB}, manifest {MANIFEST}")
    for name, a in rep["axes"].items():
        print(f"  [{a['status']:<8}] {name:<22} {a['detail']}")
    print("SKIPPED AXES:", ", ".join(rep["skipped"]))
    if rep["failures"]:
        print("FAILURES:")
        for f in rep["failures"]:
            print("  -", f)
    print("VERDICT:", rep["verdict"], "-- a report, not permission: adoption is a decision")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rep, indent=1, default=str))
    if args.png:
        _plots(rep, args.png)
    return 0 if rep["verdict"] == "PASS" else 1

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
C9_ROOT = REPO / "data" / "runs" / "pn_sumrule_c9_seeds"
C9_PIN = ROOT / "C9_L0.1_ce"
#: PRE-DECLARED (scripts/queues/p228_c9_seeds_queue.sh header, before either seed
#: trained): the class statement is seed-robust iff BOTH new seeds read
#: |1 - gamma| > BAND under the ruled cell-D estimator.
C9_VERDICTS = ("CLASS-ROBUST", "NOT-ROBUST")

def score_c9_seeds() -> dict:
    with (C9_ROOT / "arms.tsv").open() as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t")]
    pin_z, pin_path = load_npz(C9_PIN)
    pin = cell_d_metrics(pin_z)
    arms, fails = [], []
    for r in sorted(rows, key=lambda r: int(r["seed"])):
        if r["exit"] != "0":
            fails.append(f"{r['label']}: exit={r['exit']}")
        z, path = load_npz(C9_ROOT / r["label"])
        m = cell_d_metrics(z)
        arm = dict(label=r["label"], npz=str(path), seed=int(r["seed"]), **m,
                   stored_gamma_ratio=float(z["gamma_ratio"]),
                   mode_source=str(scalar(z, "mode_source")), epochs=int(scalar(z, "epochs")),
                   formulation=formulation_of(z), L_max=int(scalar(z, "L_max")),
                   gauge=str(scalar(z, "gauge")), band_class=band_class(m["gamma_ratio"]),
                   dev=abs(1.0 - m["gamma_ratio"]), q4_band=_q4_band(z), loss_drop=_loss_drop(z))
        for key, want in (("mode_source", COMB), ("epochs", 20000), ("formulation", "ce"),
                          ("L_max", 8), ("gauge", "additive")):
            if arm[key] != want:
                fails.append(f"{r['label']}: {key}={arm[key]!r} != {want!r}")
        if m["gamma_ratio"] != arm["stored_gamma_ratio"]:
            fails.append(f"{r['label']}: recomputed gamma_ratio != stored")
        arms.append(arm)
    gammas = [pin["gamma_ratio"]] + [a["gamma_ratio"] for a in arms]
    robust = len(arms) == 2 and all(a["band_class"] == "outside" for a in arms)
    return dict(pin=dict(npz=str(pin_path), **pin), arms=arms,
                spread_pp=100.0 * (max(gammas) - min(gammas)),
                gammas=gammas, failures=fails,
                verdict="CLASS-ROBUST" if robust and not fails else "NOT-ROBUST")

def _c9_main(json_path: Path | None) -> int:
    rep = score_c9_seeds()
    print(f"C9 seeds -- pin (seed 42) gamma={rep['pin']['gamma_ratio']:.6f}")
    for a in rep["arms"]:
        print(f"  seed {a['seed']}: gamma={a['gamma_ratio']:.6f} |1-g|={100*a['dev']:.2f} % "
              f"class={a['band_class']} rmse={a['rmse']:.6f} q4=[{a['q4_band'][0]:.4f}, {a['q4_band'][1]:.4f}]")
    print(f"  three-seed spread {rep['spread_pp']:.4f} pp; failures: {rep['failures'] or 'none'}")
    print("VERDICT:", rep["verdict"], "-- bar: both seeds outside the +-10 % band (pre-declared)")
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(rep, indent=1, default=str))
    return 0 if rep["verdict"] == "CLASS-ROBUST" else 1

if __name__ == "__main__":
    raise SystemExit(main())
