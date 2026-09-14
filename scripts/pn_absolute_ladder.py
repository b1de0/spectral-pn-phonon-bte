#!/usr/bin/env python3
"""PN's ABSOLUTE physical observable: the suppression ladder S(L), the effective conductivity kappa_eff(L) in W m^-1 K^-1, and the effective diffusivity D_eff(L)."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from pinn_bte.physics.decay_metrics import (
    DENOMINATOR_RULE_SAME_WINDOW,
    GAMMA_ESTIMATOR_RULED,
    integral_gamma,
)
from pinn_bte.physics.fullbz_modes import SI_C_VOLUMETRIC_300K, SI_KAPPA_BULK_300K
from pinn_bte.physics.comb_routing import comb_context, comb_grid, comb_modes
from pinn_bte.physics.optical_reservoir import PRIMITIVE_CELLS_PER_M3, joint_modes
from pinn_bte.physics.ttg_dispersion import (
    K_NORM_HI_SUM_RULE, K_NORM_LO_PRODUCTION, fourier_gamma, phonon_modes, sum_rule_grid,
)

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "data" / "runs"
OUT = REPO / "data" / "pn_absolute_ladder.json"

# ---------------------------------------------------------------------------
# WHICH LADDER.  Resolved from data/ladders.yaml -- the ONE declaration the
# claim store, the figure generators and the cell-D ledger all read -- never
# hard-coded here.  A hard-coded root is exactly how figA/figB/figC were left
# The uniformity CONTRACT (epochs/seed/snapshot) comes from `ladder_meta`, so a
# future re-pin flips one file and this writer follows or refuses loudly.
# ---------------------------------------------------------------------------
_LADDERS = yaml.safe_load((REPO / "data" / "ladders.yaml").read_text())
LADDER_SOURCE = _LADDERS["ladder_source"]
COMB = _LADDERS["combs"][LADDER_SOURCE]
if COMB not in ("joint", "joint-sumrule"):
    raise SystemExit(
        f"ladder {LADDER_SOURCE!r} rides the {COMB!r} comb; every anchor below "
        f"(the comb's capacity, kappa, D_bulk) is the joint comb's own on its "
        f"declared grid, so an absolute ladder on another comb is a scale "
        f"error, not a parameter choice.")
#: The declared comb's k-grid kwargs ({} for joint, sum_rule_grid for
GRID = comb_grid(COMB, 20)
_META = _LADDERS["ladder_meta"][LADDER_SOURCE]

#: All 18 rung directories of the ACTIVE ladder, keyed (L_um, geometry).
_RUNG_DIRS = {(float(L), geom): REPO / rel
              for geom in ("1d", "2d")
              for L, rel in _LADDERS["ladders"][LADDER_SOURCE][geom].items()}
#: One run root per campaign: corridor60k trained both geometries under one
#: its own arms.tsv.  More than two roots is a ladder assembled from parts,
#: which this writer refuses (uniform_params_within_experiment_type).
LADDER_ROOTS = sorted({p.parent for p in _RUNG_DIRS.values()})
if not 1 <= len(LADDER_ROOTS) <= 2:
    raise SystemExit(f"ladder {LADDER_SOURCE!r} spans {len(LADDER_ROOTS)} run roots "
                     f"({[str(r) for r in LADDER_ROOTS]}); one per campaign geometry "
                     f"at most, each carrying its own arms.tsv.")
MANIFESTS = [root / "arms.tsv" for root in LADDER_ROOTS]
LADDER_ROOT, MANIFEST = LADDER_ROOTS[0], MANIFESTS[0]

M_PER_UM = 1e-6
ANG_PER_M = 1e10
NK = 20
T_REF = 300.0
EPOCHS_LADDER = int(_META["epochs"])
SEED_LADDER = int(_META["seed"])
SNAPSHOT_LADDER = str(_META["snapshot"])

#: The nine rungs, in ladder order.  Also the row order of the JSON sections.
LADDER_L_UM = (0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 10.0, 100.0)

#: Row LABELS, declared explicitly rather than formatted from the float.
#: "%g" % 1.0 is "1", not "1.0" -- and the store's `_rungs`, the previous
#: ledger and the companion study's published ladder all key on "1.0", so a formatted label
#: would silently fail to join and drop that rung from every cross-check.
#: (It did, until this line existed.)
LADDER_LABEL = dict(zip(LADDER_L_UM,
                        ("0.01", "0.1", "0.3", "0.5", "0.7", "1.0", "1.5",
                         "10", "100")))

#: Experimental bulk Si conductivity band at 300 K -- an EXTERNAL standard.
KAPPA_BAND = (142.0, 156.0)

#: The companion study's published S_dom ladder (its earlier mode set), for the
#: cross-paper gap report only.
#: at L = 0.01/0.1/1/10/100 um, Nk=20, n_decay=12, cell-D window rule, and --
#: the part that now matters -- the SHIPPED mode weight.  Quoted here only to
#: MEASURE the cross-paper gap the adoption opened; it is not an agreement
#: check any more, and it must not be turned back into one until the companion study re-pins.
LOBE_S_DOM_SHIPPED = {"0.01": 0.0164, "0.1": 0.1078, "1.0": 0.4073,
                      "10": 0.9417, "100": 0.9975}

#: Diagnostic arms. NEVER a ladder rung -- see the module docstring.
PROBE_ARMS = (
    ("L0.3_2d_60k", "pn_joint_weight_2d_budget_probe/budget_L0.3_2d_ep60000_s42"),
    ("L0.1_2d_60k", "pn_joint_weight_2d_budget_probe/budget_L0.1_2d_ep60000_s42"),
    ("L0.1_2d_100k", "pn_joint_weight_2d_settle_probe/settle_L0.1_2d_ep100000_s42"),
    ("L0.3_2d_20k_seed1", "pn_joint_weight_2d_budget_probe/seed_L0.3_2d_ep20000_s1"),
    ("L0.1_2d_20k_seed1", "pn_joint_weight_2d_budget_probe/seed_L0.1_2d_ep20000_s1"),
)

def q_si(L_um: float) -> float:
    """Grating wavevector q = 2 pi / L in 1/m (L in um).  == the companion study's q_si."""
    return 2.0 * np.pi / (L_um * M_PER_UM)

def _npz(run_dir: Path):
    hits = sorted(glob.glob(str(run_dir / "*_results.npz")))
    if len(hits) != 1:
        raise FileNotFoundError(f"{run_dir}: expected 1 *_results.npz, got {hits}")
    return hits[0], np.load(hits[0], allow_pickle=True)

def arms() -> dict:
    """{(L_um, geometry): Path} -- the paper's declaration, manifest-verified."""
    manifest = {}
    for path in MANIFESTS:
        lines = [ln for ln in path.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
        if lines and lines[0].startswith("L_um\t"):
            cols = lines[0].split("\t")
            for ln in lines[1:]:
                row = dict(zip(cols, ln.split("\t")))
                if row.get("exit", "0") != "0":
                    raise SystemExit(f"{path}: arm {row['L_um']} {row['geometry']} "
                                     f"recorded exit={row['exit']}; not a pin")
                manifest[(float(row["L_um"]), row["geometry"])] = REPO / row["outdir"]
        else:
            for ln in lines:
                L, geom, d = ln.split("\t")
                manifest[(float(L), geom)] = REPO / d
    if manifest != _RUNG_DIRS:
        only_m = sorted(k for k in manifest if k not in _RUNG_DIRS)
        only_l = sorted(k for k in _RUNG_DIRS if k not in manifest)
        moved = sorted(k for k in manifest
                       if k in _RUNG_DIRS and manifest[k] != _RUNG_DIRS[k])
        raise SystemExit(
            f"{[str(m) for m in MANIFESTS]} disagree with ladders.yaml[{LADDER_SOURCE!r}]: "
            f"manifest-only {only_m}, ladder-only {only_l}, moved {moved}. "
            f"A pin two records disagree about is not a pin.")
    return dict(_RUNG_DIRS)

def comb_anchors() -> dict:
    """The joint comb's own capacity, conductivity and diffusivity at Nk=20.

    NOTHING here is external.  `sum C` is the comb's capacity; kappa is its own
    RTA conductivity; D_bulk is their quotient.  rho c_p and the literature
    kappa are carried alongside as CHECKS, never as anchors.
    """
    v, tau, C = comb_modes(COMB, NK, T_REF)
    C_tot = float(np.sum(C))
    kappa_bulk = float(np.sum(C * (v / ANG_PER_M) ** 2 * tau) / 3.0)
    D_bulk = kappa_bulk / C_tot
    # The theorem the legacy weight violated by 2.12x: the DOM operator's own
    # q -> 0 limit must BE the comb's kappa/C.  Measured, not assumed.
    with comb_context(COMB, NK, T_REF):
        D_from_operator = fourier_gamma(1.0, Nk=NK, T_ref=T_REF) / q_si(1.0) ** 2
    return dict(n_modes=int(len(v)), C_tot=C_tot, kappa_bulk=kappa_bulk,
                D_bulk=D_bulk, D_from_operator=D_from_operator,
                capacity_vs_rho_cp=C_tot / SI_C_VOLUMETRIC_300K,
                operator_identity_rel=abs(D_from_operator - D_bulk) / D_bulk)

def ladder(geometry: str, arm_dirs: dict, anch: dict) -> list:
    """The three quantities on every rung of one arm, from its pinned runs."""
    out = []
    for L_um in LADDER_L_UM:
        path, d = _npz(arm_dirs[(L_um, geometry)])
        assert int(d["Nk"]) == NK, f"{path}: Nk={int(d['Nk'])} != {NK}"
        assert int(d["epochs"]) == EPOCHS_LADDER, f"{path}: non-uniform budget"
        assert int(d["seed"]) == SEED_LADDER, f"{path}: non-uniform seed"
        assert str(d["mode_source"]) == COMB, f"{path}: wrong comb (declared {COMB!r})"
        assert str(d["geometry"]) == geometry, f"{path}: wrong geometry"
        assert str(d["snapshot_selection"]) == SNAPSHOT_LADDER, (
            f"{path}: snapshot {d['snapshot_selection']!r} != declared "
            f"{SNAPSHOT_LADDER!r} -- mixing min-loss and final-epoch arms in "
            f"one ladder is a budget mix in disguise ")

        # Numerator: the network's own trace, ruled |A| estimator, run window.
        g_pn = integral_gamma(np.asarray(d["t"]), np.asarray(d["A"]),
                              estimator=GAMMA_ESTIMATOR_RULED)
        # Denominator arm: the arbiter re-scored on the SAME window, native grid.
        g_dom = integral_gamma(np.asarray(d["t_ref"]), np.asarray(d["A_ref"]),
                               estimator=GAMMA_ESTIMATOR_RULED)
        q2 = q_si(L_um) ** 2
        D_bulk = anch["D_bulk"]
        out.append(dict(
            label=LADDER_LABEL[L_um], L_um=L_um,
            run=str(arm_dirs[(L_um, geometry)].relative_to(REPO)),
            geometry=str(d["geometry"]), nt=int(np.asarray(d["t"]).size),
            n_ref=int(np.asarray(d["t_ref"]).size),
            t_end_s=float(d["t_end"]), n_decay=float(d["n_decay_times"]),
            epochs=int(d["epochs"]), seed=int(d["seed"]),
            best_epoch=int(d["best_epoch"]), best_loss=float(d["best_loss"]),
            gamma_pn_hz=g_pn, gamma_dom_hz=g_dom,
            # (1) suppression -- S = D_eff/D_bulk on the comb's own D_bulk
            S_pn=g_pn / q2 / D_bulk, S_dom=g_dom / q2 / D_bulk,
            # (2) absolute conductivity, on the comb's OWN capacity
            kappa_eff_pn=anch["C_tot"] * g_pn / q2,
            kappa_eff_dom=anch["C_tot"] * g_dom / q2,
            # (3) effective diffusivity
            D_eff_pn=g_pn / q2, D_eff_dom=g_dom / q2,
            D_bulk=D_bulk,
            gamma_ratio=g_pn / g_dom,
            # the trainer's OWN in-run number, computed independently at train
            # time -- the bit-exactness axis below
            gamma_ratio_in_run=float(d["gamma_ratio"]),
            rel_miss=abs(g_pn / g_dom - 1.0),
        ))
    return out

def _kappa_C(Nk: int, **grid) -> tuple:
    v, tau, C = joint_modes(Nk=Nk, T_ref=T_REF, **grid)
    return float(np.sum(C * (v / ANG_PER_M) ** 2 * tau) / 3.0), float(np.sum(C))

#: The full Brillouin zone, for the truncation half of the convergence story.
#: The production comb deliberately truncates k_norm to [0.05, 0.95] (a pinned
#: 300 K choice, tests/test_ttm_cryo_gaps.py), and that truncation -- not the
#: mode count -- is what carries kappa below the experimental band at high Nk.
FULLBZ_GRID = dict(k_lo=1e-3, k_hi=1.0)
#: centre; k_hi is the comb's own radius (the sum-rule radius for joint-sumrule,
#: the full sphere for joint).
FULLZONE_GRID = (dict(k_lo=1e-3, k_hi=K_NORM_HI_SUM_RULE, quadrature="midpoint")
                 if COMB == "joint-sumrule" else FULLBZ_GRID)

def nk_sweep(nks=(10, 20, 40, 80, 160, 400, 800)) -> list:
    """kappa_bulk and D_bulk of the JOINT comb vs Nk -- the convergence
    disclosure.  Both are the comb's own, so this is now a statement about one
    self-consistent object rather than about a model number times a borrowed
    constant.

    Two columns on purpose.  On the PRODUCTION grid kappa drifts DOWN with Nk
    and crosses just under the band; on the FULL BZ it converges inside it.
    Reporting only the first would misattribute a k-range truncation to the
    mode count -- which is the exact error the pre-adoption text made in the
    opposite direction, so it is worth not repeating with the sign flipped.
    """
    rows = []
    for nk in nks:
        kap, C_tot = _kappa_C(nk, **comb_grid(COMB, nk))
        kap_full, C_full = _kappa_C(nk, **FULLZONE_GRID)
        rows.append(dict(label=str(nk), Nk=nk, D_bulk=kap / C_tot,
                         kappa_bulk=kap, C_tot=C_tot,
                         kappa_bulk_fullbz=kap_full, C_tot_fullbz=C_full))
    return rows

def grid_variants(Nk: int = NK) -> list:
    """THE DISCLOSED SEARCH ORDER ( rule, adopted 2 of ): the k-grid variants tried before the sum-rule comb was adopted, at the production Nk, each with its mode count (vs 3 per primitive cell), capacity (vs rho c_p) and kappa. The adopted one is LAST; it was chosen by the sum rule and the floor anchor, never by which kappa lands in the band. Comb-independent: it describes the joint comb's grid family, whichever grid."""
    KB = 1.380649e-23
    variants = [
        ("endpoint_full", dict()),
        ("endpoint_sumrule_radius", dict(k_hi=K_NORM_HI_SUM_RULE)),
        ("midpoint_full", dict(k_lo=K_NORM_LO_PRODUCTION, k_hi=0.95, quadrature="midpoint")),
        ("midpoint_sumrule_unanchored", dict(k_lo=K_NORM_LO_PRODUCTION, k_hi=K_NORM_HI_SUM_RULE,
                                             quadrature="midpoint")),
        ("midpoint_sumrule_anchored", sum_rule_grid(Nk)),
    ]
    rows = []
    for label, grid in variants:
        v, tau, C = joint_modes(Nk=Nk, T_ref=T_REF, **grid)
        # mode count via the T -> infinity limit of C/k_B per mode (equipartition),
        # the same trick as scripts/score_sumrule_ladder.comb_facts
        n_modes = float(np.sum(phonon_modes(Nk=Nk, T_ref=4000.0, measure=True, **grid)[2])) / KB
        kap = float(np.sum(C * (v / ANG_PER_M) ** 2 * tau) / 3.0)
        rows.append(dict(label=label, Nk=Nk, grid=grid, kappa_bulk=kap,
                         C_tot=float(np.sum(C)),
                         capacity_ratio=float(np.sum(C)) / SI_C_VOLUMETRIC_300K,
                         mode_count_ratio=n_modes / (3.0 * PRIMITIVE_CELLS_PER_M3),
                         mfp_max_um=float(((v[:-1] / ANG_PER_M) * tau[:-1]).max() * 1e6),
                         adopted=(label == "midpoint_sumrule_anchored")))
    return rows

def kappa_vs_nk_unanchored(nks=(20, 40, 80, 160)) -> list:
    """kappa_bulk of the midpoint sum-rule grid WITHOUT the floor anchor: the
    wander the anchor removes (its lowest sampled mode moves with Nk and
    dominates kappa).  Disclosed beside the anchored sweep."""
    grid = dict(k_lo=K_NORM_LO_PRODUCTION, k_hi=K_NORM_HI_SUM_RULE, quadrature="midpoint")
    return [dict(label=str(nk), Nk=nk, value=_kappa_C(nk, **grid)[0]) for nk in nks]

def gamma_ratio_band(z, last_fraction: float = 0.25) -> dict:
    """min/max/mean of the in-run gamma_ratio over the last quarter of training.

    The reproducible way to quote a rung that has NOT settled: one draw from a
    wandering trajectory is not a measurement, and the spread is the honest
    error bar.  NOT capped -- a 3.5% deviation is real, not the sub-1%
    numerical bump the reporting convention allows capping.
    """
    gh = np.asarray(z["gamma_history"])
    if gh.ndim != 2 or gh.shape[0] < 8:
        raise ValueError("gamma_history too short to band")
    cut = (1.0 - last_fraction) * float(z["epochs"])
    q = gh[gh[:, 0] >= cut, 1]
    return dict(n=int(q.size), min=float(q.min()), max=float(q.max()),
                mean=float(q.mean()),
                half_spread=float((q.max() - q.min()) / 2.0))

#: The PREVIOUS pinning's per-rung ledger, kept only so the re-pin can state
#: what moved.  Its `after` column is the RULED cell-D convention; its `before`
#: Both are carried because they disagree about which rung was worst, and
PUBLISHED_LEDGER = REPO / "data" / "p086_cell_d_ledger.json"

def published_comparison() -> list:
    """What the SHIPPED-comb ladder said, rung by rung, under both conventions.

    Bound rather than typed into the prose: the re-pin's honesty rests on this
    comparison, and a hand-copied comparison number is how a superseded
    convention gets quoted beside a live one.
    """
    led = json.loads(PUBLISHED_LEDGER.read_text())
    rows = []
    for tag in ("pn_1d", "pn_2d"):
        for r in led[tag]:
            rows.append(dict(
                label=f"{tag}_{r['label']}", geometry=tag[-2:],
                L_um=float(r["label"]),
                rel_miss_celld=abs(1.0 - float(r["after"])),
                rel_miss_arbiterwindow=abs(1.0 - float(r["before"])),
                run=r["run"]))
    for tag in ("1d", "2d"):
        sub = [r for r in rows if r["geometry"] == tag and "worst" not in r["label"]]
        wc = max(sub, key=lambda r: r["rel_miss_celld"])
        wa = max(sub, key=lambda r: r["rel_miss_arbiterwindow"])
        rows.append(dict(label=f"worst_celld_{tag}", geometry=tag,
                         L_um=wc["L_um"],
                         rel_miss_celld=wc["rel_miss_celld"],
                         rel_miss_arbiterwindow=wc["rel_miss_arbiterwindow"],
                         run=wc["run"]))
        rows.append(dict(label=f"worst_arbiterwindow_{tag}", geometry=tag,
                         L_um=wa["L_um"],
                         rel_miss_celld=wa["rel_miss_celld"],
                         rel_miss_arbiterwindow=wa["rel_miss_arbiterwindow"],
                         run=wa["run"]))
    return rows

#: The rung whose decay is NOT a single exponential, and which therefore needs
#: its physics disclosed separately from its numerics.
NONEXP_RUNG_L_UM = 0.1

def nonexponentiality(L_um: float = NONEXP_RUNG_L_UM) -> dict:
    """Is the decay at this period a single exponential? Measured, not assumed."""
    from pn_L01_wander_table import rung, signed_trace

    modes = comb_modes(COMB, NK, T_REF)
    d = rung(modes, L_um)
    t, A = signed_trace(L_um, *modes, d["t_end"], 12000)
    g_loc = -np.gradient(np.log(np.abs(A) + 1e-300), t) / d["g"]
    i_early = int(0.005 * (len(t) - 1))
    i_mid = (len(t) - 1) // 2
    return dict(label=f"L{L_um:g}_nonexponentiality", L_um=L_um,
                root_over_continuum=float(d["root_over_continuum"]),
                gloc_early=float(g_loc[i_early]),
                gloc_mid=float(g_loc[i_mid]),
                gloc_edge=float(d["gloc_edge"]),
                t_zero_over_t_end=float(d["t_zero"]),
                xi_median=float(d["xi_med"]),
                t_early_over_t_end=float(t[i_early] / d["t_end"]))

def disclosures(one_d: list, two_d: list) -> list:
    """The DIAGNOSTIC arms, and the bands that make an unsettled rung quotable.

    These are the numbers the manuscript's disclosure paragraphs are bound to.
    Every one is read off a pinned npz; none is a ladder rung.
    """
    rows = []
    for label, rel in PROBE_ARMS:
        path, z = _npz(RUNS / rel)
        row = dict(label=label, run=rel, L_um=float(z["L_um"]),
                   geometry=str(z["geometry"]), epochs=int(z["epochs"]),
                   seed=int(z["seed"]), mode_source=str(z["mode_source"]),
                   best_epoch=int(z["best_epoch"]),
                   best_loss=float(z["best_loss"]),
                   gamma_ratio=float(z["gamma_ratio"]),
                   rel_miss=abs(float(z["gamma_ratio"]) - 1.0))
        try:
            row["q4_band"] = gamma_ratio_band(z)
        except (ValueError, KeyError):
            row["q4_band"] = None
        # FLAT copies as well: the claim store's `cell_d_field` reads one named
        # column of one row, so a number that only exists inside a nested dict
        # cannot be bound -- and an unbindable number is one that gets typed by
        # hand into the manuscript, which is the failure this file prevents.
        for k in ("min", "max", "mean", "half_spread", "n"):
            row[f"q4_{k}"] = row["q4_band"][k] if row["q4_band"] else float("nan")
        rows.append(row)

    # The ladder's own worst rungs, derived rather than typed beside the table.
    for tag, rung_rows in (("1d", one_d), ("2d", two_d)):
        worst = max(rung_rows, key=lambda r: r["rel_miss"])
        rows.append(dict(label=f"ladder_worst_{tag}", run=worst["run"],
                         L_um=worst["L_um"], geometry=tag,
                         epochs=worst["epochs"], seed=worst["seed"],
                         mode_source=COMB, best_epoch=worst["best_epoch"],
                         best_loss=worst["best_loss"],
                         gamma_ratio=worst["gamma_ratio"],
                         rel_miss=worst["rel_miss"], q4_band=None,
                         **{f"q4_{k}": float("nan")
                            for k in ("min", "max", "mean", "half_spread", "n")}))
    return rows

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

def checks(one_d: list, two_d: list, anch: dict) -> list:
    res = []

    def ok(name, cond, detail=""):
        res.append(dict(name=name, passed=bool(cond), detail=detail))

    # AXIS 0 -- the comb is a physical object: its capacity is a capacity, its
    # conductivity is silicon's, and the operator's diffusive limit is its own
    # kappa/C.  None of this was true of the weight this ladder replaces.
    ok("comb: kappa_bulk inside the experimental band 142-156",
       KAPPA_BAND[0] <= anch["kappa_bulk"] <= KAPPA_BAND[1],
       f"{anch['kappa_bulk']:.3f} W/(m K)")
    ok("comb: DOM q->0 limit == kappa/C to 1e-12",
       anch["operator_identity_rel"] < 1e-12,
       f"rel dev {anch['operator_identity_rel']:.2e}")
    ok("comb: sum C within 5% of rho c_p (EMERGENT, not fitted)",
       abs(anch["capacity_vs_rho_cp"] - 1.0) < 0.05,
       f"sum C / rho c_p = {anch['capacity_vs_rho_cp']:.4f}")

    for tag, rows in (("1d", one_d), ("2d", two_d)):
        S_pn = [r["S_pn"] for r in rows]
        S_dom = [r["S_dom"] for r in rows]
        # AXIS 1 -- monotone in L (physics: bigger grating is more diffusive)
        ok(f"{tag}: S_pn strictly increasing in L",
           all(b > a for a, b in zip(S_pn, S_pn[1:])),
           f"{S_pn[0]:.4f} -> {S_pn[-1]:.4f}")
        ok(f"{tag}: S_dom strictly increasing in L",
           all(b > a for a, b in zip(S_dom, S_dom[1:])))
        # AXIS 2 -- physically bounded
        ok(f"{tag}: S_pn <= 1 (within 1% numerical bump)",
           max(S_pn) <= 1.01, f"max {max(S_pn):.4f}")
        ok(f"{tag}: S_dom <= 1", max(S_dom) <= 1.0, f"max {max(S_dom):.6f}")
        # AXIS 3 -- diffusive anchor recovered with NO supervision
        ok(f"{tag}: S_pn(100um) within 1% of the Fourier limit",
           abs(S_pn[-1] - 1.0) < 0.01, f"S={S_pn[-1]:.4f}")
        # AXIS 3b -- Fourier recovery in SI: kappa_eff(100um)/kappa_bulk >= 0.99
        # then this read "kappa_eff(100um) inside 142-156 W/(m K)", which tests a
        # FINITE-L value against the BULK band and rejects the arbiter itself on
        # the sum-rule comb (kappa_bulk = 142.30 sits 0.2 % above the band's lower
        # edge; S_dom(100) = 0.9941 puts even the reference at 141.47).  The two
        # physical statements it conflated are checked separately: kappa_bulk in
        # band (below, the dimensional axis) and the diffusive limit recovered
        # here.  Old reading on the corridor60k/joint pin: 143.70 / 143.42.
        k100 = rows[-1]["kappa_eff_pn"]
        ok(f"{tag}: kappa_eff(100um)/kappa_bulk >= 0.99 (Fourier recovery; "
           f"kappa_eff = {k100:.3f} W/(m K))",
           k100 / anch["kappa_bulk"] >= 0.99, f"{k100 / anch['kappa_bulk']:.4f}")
        # AXIS 3c -- ballistic must be ballistic
        ok(f"{tag}: S_pn(0.01um) <= 0.05", S_pn[0] <= 0.05, f"S={S_pn[0]:.4f}")
        # AXIS 4 -- D_bulk is L-independent (an identity, must be exact)
        Db = [r["D_bulk"] for r in rows]
        ok(f"{tag}: D_bulk L-independent to 1e-12",
           (max(Db) - min(Db)) / np.mean(Db) < 1e-12,
           f"{np.mean(Db):.6e} m^2/s")
        # AXIS 5 -- the three quantities are ONE quantity: S = D_eff/D_bulk =
        #           kappa_eff/kappa_bulk.  If this ever fails, two anchors got
        #           mixed.
        worst = max(abs(r["S_pn"] - r["D_eff_pn"] / r["D_bulk"]) for r in rows)
        ok(f"{tag}: S == D_eff/D_bulk exactly", worst < 1e-12, f"max dev {worst:.2e}")
        worst = max(abs(r["S_pn"] - r["kappa_eff_pn"] / anch["kappa_bulk"])
                    for r in rows)
        ok(f"{tag}: S == kappa_eff/kappa_bulk exactly", worst < 1e-12,
           f"max dev {worst:.2e}")
        # AXIS 6 -- the re-integration reproduces the TRAINER's own in-run
        # gamma_ratio bit-for-bit.  Independent code path, computed at train
        # time; if this drifts, the estimator or the window has moved.
        worst = max(abs(r["gamma_ratio"] - r["gamma_ratio_in_run"]) for r in rows)
        ok(f"{tag}: gamma_ratio reproduces the run's own value bit-for-bit",
           worst == 0.0, f"max dev {worst:.2e}")
        # AXIS 9 -- optimisation health: uniform budget and seed (the snapshot
        # rule is asserted per arm in ladder() against the declared contract).
        ok(f"{tag}: uniform budget across all nine rungs",
           len({r["epochs"] for r in rows}) == 1
           and len({r["seed"] for r in rows}) == 1,
           f"epochs={sorted({r['epochs'] for r in rows})} "
           f"seed={sorted({r['seed'] for r in rows})}")

    # AXIS 7 -- the arbiter's 1D and 2D arms are the SAME kinetic problem for a
    # grating in x (ttg_dispersion C16 marginal identity).  The PN arms are
    # trained INDEPENDENTLY, so their agreement is a result, not an identity.
    d_arb = max(abs(a["S_dom"] - b["S_dom"]) / a["S_dom"]
                for a, b in zip(one_d, two_d))
    res.append(dict(name="arbiter 1d vs 2d S_dom agree (<2%)", passed=d_arb < 0.02,
                    detail=f"max rel dev {d_arb:.2e}"))
    d_pn = max(abs(a["S_pn"] - b["S_pn"]) / a["S_pn"]
               for a, b in zip(one_d, two_d))
    # The 12% bar was sized for the JOINT ladder's 2D L = 0.3 miss of 0.107 --
    # to remove.  On the corridor ladder the worst |1-gamma| is 0.0055 and this
    # axis sits far inside the bar; the bar is kept at the value it was
    # DECLARED at rather than re-fitted to the data that happens to be pinned.
    res.append(dict(name="PN 1d vs 2d S agree (<12%) -- independently trained",
                    passed=d_pn < 0.12, detail=f"max rel dev {d_pn:.2e}"))

    # AXIS 8 -- cross-paper.  This is now a DISCLOSURE, not an agreement check:
    # the companion study's published ladder is a SHIPPED-comb ladder and this one is not, so
    # the two are expected to differ and the number below measures by how much.
    shared = [r for r in one_d if r["label"] in LOBE_S_DOM_SHIPPED]
    # A JOIN, asserted.  A label-format drift would silently shrink the shared
    # set and make the gap look smaller than it is; that is not a cross-check
    # failing, it is a cross-check evaporating.
    ok("cross-paper join covers all five shared rungs",
       len(shared) == len(LOBE_S_DOM_SHIPPED),
       f"{len(shared)}/{len(LOBE_S_DOM_SHIPPED)} rungs joined")
    dev = max(abs(r["S_dom"] - LOBE_S_DOM_SHIPPED[r["label"]]) for r in shared)
    res.append(dict(
        name="cross-paper gap to the companion study's SHIPPED-comb ladder (REPORT only)",
        passed=True,
        detail=f"max |dS| = {dev:.4f}; the companion study has not re-pinned onto the joint "
               f"comb, so this is the size of that debt, not an agreement"))
    return res

def cross_checks(one_d: list, two_d: list) -> list:
    """The cross-check scalars the manuscript PRINTS, as numbers.

    `checks()` returns pass/fail with a human detail string; a string is not a
    bindable number, and a printed number that a claim store cannot re-derive
    is exactly the defect this file exists to avoid.
    """
    lobe = max(abs(r["S_dom"] - LOBE_S_DOM_SHIPPED[r["label"]])
               for r in one_d if r["label"] in LOBE_S_DOM_SHIPPED)
    geo = max(abs(a["S_pn"] - b["S_pn"]) / a["S_pn"]
              for a, b in zip(one_d, two_d))
    span = one_d[-1]["S_pn"] / one_d[0]["S_pn"]
    return [
        dict(label="lobe_max_abs_dS_shipped", value=lobe,
             note="max |S_dom(this paper, JOINT comb) - S_dom(companion study, "
                  "SHIPPED comb)| over the five shared rungs.  Since the "
                  "the companion study is on an earlier mode set; this measures the CROSS-PAPER GAP, "
                  "not agreement: the companion study has not been re-pinned."),
        dict(label="geometry_max_rel_dS", value=geo,
             note="max relative 1D-vs-2D departure of the data-free S; the "
                  "two arms are trained independently.  Dominated by the "
                  "L = 0.3 um 2D rung, whose miss is disclosed."),
        dict(label="S_span_1d", value=span,
             note="S(100 um)/S(0.01 um) -- the dynamic range the ladder spans"),
        dict(label="worst_rel_miss_1d",
             value=max(r["rel_miss"] for r in one_d),
             note="worst |1 - gamma_ratio| over the nine 1D rungs"),
        dict(label="worst_rel_miss_2d",
             value=max(r["rel_miss"] for r in two_d),
             note="worst |1 - gamma_ratio| over the nine 2D rungs"),
    ]

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help=f"write {OUT.relative_to(REPO)}")
    args = ap.parse_args()

    arm_dirs = arms()
    anch = comb_anchors()
    one_d, two_d = ladder("1d", arm_dirs, anch), ladder("2d", arm_dirs, anch)
    sweep = nk_sweep()
    disc = disclosures(one_d, two_d)
    nonexp = nonexponentiality()

    print(f"\n===== joint comb, Nk={NK} ({anch['n_modes']} modes: 60 acoustic "
          f"+ 1 optical reservoir) =====")
    print(f"  sum C      = {anch['C_tot']:.6e} J/(m^3 K) "
          f"({anch['capacity_vs_rho_cp']:.4f} x rho c_p, EMERGENT)")
    print(f"  kappa_bulk = {anch['kappa_bulk']:.3f} W/(m K) "
          f"(experimental band {KAPPA_BAND[0]:g}-{KAPPA_BAND[1]:g})")
    print(f"  D_bulk     = {anch['D_bulk']:.6e} m^2/s "
          f"(DOM q->0 limit agrees to {anch['operator_identity_rel']:.1e})")

    hdr = (f"{'L [um]':>8} {'S_pn':>9} {'S_dom':>9} {'k_eff_pn':>10} "
           f"{'k_eff_dom':>10} {'D_eff_pn':>11} {'D_eff_dom':>11} {'g_ratio':>9}")
    for tag, rows in (("1D", one_d), ("2D", two_d)):
        print(f"\n===== PN {tag}: absolute ladder (Nk={NK}, |A| estimator, "
              f"run window n_decay=7, {EPOCHS_LADDER} epochs, seed "
              f"{SEED_LADDER}) =====")
        print(hdr)
        for r in rows:
            print(f"{r['L_um']:>8g} {r['S_pn']:>9.4f} {r['S_dom']:>9.4f} "
                  f"{r['kappa_eff_pn']:>10.3f} {r['kappa_eff_dom']:>10.3f} "
                  f"{r['D_eff_pn']:>11.4e} {r['D_eff_dom']:>11.4e} "
                  f"{r['gamma_ratio']:>9.4f}")

    print(f"\n{'Nk':>6} {'D_bulk [m^2/s]':>16} {'kappa prod grid':>17} "
          f"{'kappa full BZ':>15} {'sum C [J/m^3/K]':>18}")
    for s in sweep:
        print(f"{s['Nk']:>6} {s['D_bulk']:>16.5e} {s['kappa_bulk']:>17.2f} "
              f"{s['kappa_bulk_fullbz']:>15.2f} {s['C_tot']:>18.5e}")

    print("\n===== disclosures (DIAGNOSTIC arms -- never a ladder rung) =====")
    for r in disc:
        band = r.get("q4_band")
        b = (f"  Q4 band [{band['min']:.4f}, {band['max']:.4f}] "
             f"mean {band['mean']:.4f} n={band['n']}") if band else ""
        print(f"  {r['label']:<20} L={r['L_um']:<6g} {r['geometry']} "
              f"ep={r['epochs']:<7d} s={r['seed']} gr={r['gamma_ratio']:.6f} "
              f"|1-gr|={r['rel_miss']:.4f}{b}")

    print(f"\n===== is L = {nonexp['L_um']:g} um a single exponential? =====")
    print(f"  dispersion root / RTA continuum edge = "
          f"{nonexp['root_over_continuum']:.1f}x  (>1 = inside the continuum)")
    print(f"  local rate / gamma_eff: {nonexp['gloc_early']:.2f} (t="
          f"{nonexp['t_early_over_t_end']:.3f} t_end) -> "
          f"{nonexp['gloc_mid']:.2f} (mid) -> {nonexp['gloc_edge']:.2f} (edge)"
          f"   [a single exponential holds all three at 1.00]")
    print(f"  rebound zero crossing at {nonexp['t_zero_over_t_end']:.3f} t_end;"
          f"  capacity-weighted median xi = {nonexp['xi_median']:.3f}")

    print("\n===== what the SHIPPED-comb ladder said at the cited rungs =====")
    for r in published_comparison():
        if r["label"].startswith("worst") or r["L_um"] in (0.1, 0.3):
            print(f"  {r['label']:<26} L={r['L_um']:<6g} "
                  f"cellD={r['rel_miss_celld']:.4f} "
                  f"arbiter-window(SUPERSEDED)={r['rel_miss_arbiterwindow']:.4f}")

    print("\n===== multi-axis checks =====")
    res = checks(one_d, two_d, anch)
    for c in res:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']:<58} "
              f"{c['detail']}")
    n_fail = sum(not c["passed"] for c in res)
    print(f"\n{len(res) - n_fail}/{len(res)} checks passed")

    if args.write:
        payload = dict(
            passport=dict(
                estimator=GAMMA_ESTIMATOR_RULED,
                denominator_rule=DENOMINATOR_RULE_SAME_WINDOW,
                Nk=NK, T_ref=T_REF, n_decay=7.0, n_ref_samples=12001,
                convention="absolute_joint_comb",
                mode_source=COMB,
                mode_source_detail="k-space measure dw = v dk restored on the "
                                   "three acoustic branches + a lumped Einstein "
                                   "optical reservoir",
                epochs=EPOCHS_LADDER, seed=SEED_LADDER,
                snapshot=SNAPSHOT_LADDER, ladder_source=LADDER_SOURCE,
                budget_rule="UNIFORM across all 18 arms "
                            "(uniform_params_within_experiment_type); the "
                            "pn_joint_weight_2d_{budget,settle}_probe arms are "
                            "diagnostic -- other gauge/snapshot lineage -- and "
                            "are never a ladder rung",
                fourier_rule=f"gamma_F = (<v^2 tau>_C / 3) q^2 on the {COMB} "
                             "comb, ttg_dispersion.fourier_gamma under "
                             "comb_routing.comb_context (grid "
                             f"{GRID or 'production endpoint'}), Nk=20",
                comb=COMB, grid=GRID,
                capacity_J_m3_K=anch["C_tot"],
                capacity_source="the comb's OWN sum C (joint_modes) -- not an "
                                "external constant; rho c_p is a check",
                rho_cp_check_J_m3_K=SI_C_VOLUMETRIC_300K,
                kappa_literature_W_m_K=SI_KAPPA_BULK_300K,
                kappa_band=list(KAPPA_BAND),
            ),
            anchors=[
                dict(label="D_bulk", value=anch["D_bulk"], unit="m^2/s"),
                dict(label="kappa_bulk", value=anch["kappa_bulk"], unit="W/m/K"),
                dict(label="kappa_bulk_converged",
                     value=sweep[-1]["kappa_bulk"], unit="W/m/K"),
                dict(label="kappa_bulk_fullbz_converged",
                     value=sweep[-1]["kappa_bulk_fullbz"], unit="W/m/K"),
                dict(label="C_comb", value=anch["C_tot"], unit="J/m^3/K"),
                dict(label="C_comb_over_rho_cp",
                     value=anch["capacity_vs_rho_cp"], unit="dimensionless"),
                # The capacity check is Nk-specific; disclose its convergence
                dict(label="C_comb_converged_over_rho_cp",
                     value=sweep[-1]["C_tot"] / SI_C_VOLUMETRIC_300K,
                     unit="dimensionless"),
                dict(label="C_comb_fullbz_converged_over_rho_cp",
                     value=sweep[-1]["C_tot_fullbz"] / SI_C_VOLUMETRIC_300K,
                     unit="dimensionless"),
                dict(label="C_rho_cp", value=SI_C_VOLUMETRIC_300K,
                     unit="J/m^3/K"),
            ],
            pn_1d=one_d, pn_2d=two_d, nk_sweep=sweep,
            grid_variants=grid_variants(),
            kappa_vs_nk_unanchored=kappa_vs_nk_unanchored(),
            disclosures=disc, published_comparison=published_comparison(),
            nonexponentiality=[nonexp],
            cross_checks=cross_checks(one_d, two_d),
            checks=res,
        )
        OUT.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"\nwrote {OUT.relative_to(REPO)}")

    raise SystemExit(1 if n_fail else 0)

if __name__ == "__main__":
    main()
