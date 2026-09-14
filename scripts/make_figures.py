"""Publication figures for the spectral-PN paper (Project A)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# their arms EXACTLY the way the repin module does -- one loader, one comb
# guard, one gate -- so it is imported, not duplicated.  Imported BEFORE this
# module's rcParams block: make_repin_figures updates rcParams at import time
# with values identical to the block below except `axes.titlesize`, and the
# later update (ours) must be the one that stands.
import make_repin_figures as repin  # noqa: E402

BASE = REPO / "data" / "runs"
FIG_DIR = REPO / "figures"

# `\includegraphics[width=\textwidth]` and elsarticle[preprint,12pt] has
# \textwidth = 390 pt, so a canvas authored WIDER than that is silently
# demagnified and every font size below is a lie: fig3_traces was authored
# 11.5 in (828 pt) wide and printed at 47.7%, turning its 9 pt ticks into
# 4.3 pt on the page -- under Elsevier's 7 pt artwork floor, and the smallest
# lettering in the paper.  The canvas is therefore authored AT the slot width,
# every font size below is what the reader gets in print, and every LINE and
# MARKER size is the old design value times the old on-page scale, so the ink
# weight the reader sees is unchanged.  MEASURE, don't assume: `pdfinfo` on the
# output must report a width within ~1 pt of figsize[0]*72; anything more is an
# overhang that will demagnify the figure again (this is the rule
# make_mode_spectrum_figure.py:237-239 states and no other generator applied).
# HEIGHT is the other hard constraint: \textheight = 548.5 pt and a float whose
# BOX PLUS CAPTION exceeds it is deferred to the end of the document.
SLOT_IN = 388.53 / 72.0         # 5.3963 in -- the \textwidth slot,
# measured off the page: the clip transform main.pdf applies to a
# width=\textwidth float is 814.139 x 0.47723 = 388.53 pt.
DPI = 300

#: BASE size for every string that can carry mathtext, passed EXPLICITLY to
#: every label/tick/legend/title call so no artist inherits something smaller.
#: mathtext sets a sub/superscript at 0.7x the base, so the smallest glyph this
#: module can emit is 0.7 * MATH_PT.
#:
#: WHY 10.5 AND NOT THE 10 OF make_absolute_ladder_figure.py.  10 pt puts the
#: subscripts at EXACTLY Elsevier's 7.0 pt floor, i.e. at zero headroom, and
#: zero headroom is only safe while every include is width=\textwidth (s =
#: 1.000).  The digest is not: summary.tex sets fig3_traces at 0.78\textwidth,
#: s = 0.967, which took the 7.00 pt subscripts to 6.77 pt ON THE PAGE --
#: measured, not predicted.  10.5 pt bases print at 7.35 pt in the manuscript
#: and 7.11 pt in the digest, so fig3 clears the floor in BOTH documents.
#: It does NOT rescue fig2_regime_map in the digest: summary.tex:153 gives that
#: figure 0.56\textwidth (s = 0.695), where even a 14.4 pt base would be needed
#: to reach 7 pt.  That slot is the defect, not the lettering; see the report.
MATH_PT = 10.5

# rcParams CONVERGED onto make_repin_figures.py:96-116, which is the block
# fig_absolute_ladder inherits (it sets none of its own and gets them as an
# import side effect).  This module keeps its own copy rather than importing
# that one, because make_repin_figures pulls in the campaign-evaluation stack
# (campaign_eval, extract_battery, pn_shape_battery) that these three figures
# do not need.  The VALUES are identical -- if that block moves, move this one.
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": MATH_PT,
    "xtick.labelsize": MATH_PT,
    "ytick.labelsize": MATH_PT,
    "legend.fontsize": MATH_PT,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    # 'standard', not 'tight': the saved width is then exactly
    # figsize[0]*72 and the on-page scale is exactly 1.  fit_check()
    # below is what makes that safe.
    "savefig.bbox": "standard",
    "axes.linewidth": 0.45,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "font.family": "sans-serif",
    # Elsevier permits Arial/Helvetica, Times, Courier, Symbol and SUBSTITUTES
    # anything else ("may lead to missing symbols").  Nimbus Sans IS the URW
    # Helvetica clone, so this stack satisfies the requirement literally;
    # verified present here by `fc-match Helvetica`.  DejaVu -- the matplotlib
    # was emitting Type 3 subsets with uni=no, which `pdftotext` cannot read.
    "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "Arimo",
                        "DejaVu Sans"],
    # 'stix', NOT 'stixsans': stixsans draws sans math from STIXNonUnicode,
    # which carries no usable Unicode map, so pdftotext returns ZERO of a
    # reads figure PDFs with pdftotext.  Rationale in full at
    # make_schematic.py:196-213; copied here rather than re-derived.
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,          # embed TrueType: vector, and searchable text
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.3,
})

# ---------------------------------------------------------------- pinned runs
# Canonical run per (L, formulation): GPU, 20k epochs, Nk=20, seed 42
# (L=1.0 plain adds seeds 0/1 for the robustness claim).  Round-3/4 queue
# logs: spectral_pn_round{3,4}.log; Jul-7 arms: spectral_pn_gpu.log.
# These now feed the APPENDIX figure only; the identifiers are unchanged.
RUNS = {
    (0.01, "duh"): "20260708_231945",
    (0.01, "plain"): "20260709_001714",
    (0.1, "plain"): "20260707_201435",   # lmax16
    (0.1, "duh"): "20260709_010959",
    (0.1, "ce"): "20260709_050612",
    (1.0, "plain"): "20260707_164346",
    (1.0, "plain-s0"): "20260707_173627",
    (1.0, "plain-s1"): "20260707_182908",
    (1.0, "ce"): "20260709_040747",
    (10.0, "plain"): "20260707_192149",
    (10.0, "ce"): "20260709_030913",
    (100.0, "ce"): "20260709_020952",
}

# UGKS-blend composition series (PRIMARY): one data-free uniformly-valid
# Nk=20, L_max=8, float64, seed 42; --blend ugks --snapshot-selection min-loss.
# decades are the subset of the 9-rung ladder that fig2/fig3 plot.
# (--xi-split 1.0, runs 20260713_003144/223728/233329/012715/022235) is
V5_RUNS = {
    100.0: "20260719_030728",
    10.0: "20260719_011313",
    1.0: "20260718_212441",
    0.1: "20260718_134841",
    0.01: "20260718_125320",
}

#: lands and re-pins the ladder; nothing else in this module changes.
#:
#: they predate the field and are LEGACY-comb runs, and the legacy weight FAILS
#: the dimensional axis at kappa = 72.98 W/(m K) against measured bulk silicon
#: 142-156, a factor of two low.  On that comb three of the sign verdicts the
#: figures show are comb artefacts: the arbiter's zero crossings at L = 0.1 and
#: 0.3 um vanish on the joint comb, and the solver's crossing at L = 1.0 um
#: vanishes with them.  The JOINT arms (kappa = 144.2638) are what the paper
#:
#: The reason this is a named switch rather than an edited dict: reading the
#: legacy arms and reporting the result as a property of the METHOD is exactly
#: tests/test_phonon_modes_import_binding.py).
#: READ FROM THE SHARED FILE, never declared here.  data/ladders.yaml is
#: the one place that says which comb the paper is on, and it is read by this
#: ledger.  A private copy here is how the figures and the numbers came within
#: one edit of disagreeing on the same page.
_LADDERS = yaml.safe_load((REPO / "data" / "ladders.yaml").read_text())
LADDER_SOURCE = _LADDERS["ladder_source"]
#: The COMB that ladder was trained on, from the shared map -- never inferred
#: from the ladder's NAME.  corridor60k rides the joint comb, so a name test
#: (`LADDER_SOURCE == "joint"`) would silently route every later joint-comb
LADDER_COMB = _LADDERS["combs"][LADDER_SOURCE]

#: Rung -> repo-relative run directory, for whichever comb is selected.  Keys
#: in the shared file are strings; the figures index by float.
_LADDER_1D = {float(k): v
              for k, v in _LADDERS["ladders"][LADDER_SOURCE]["1d"].items()}

# Supervised reference: round-9 resweep (1D production config of the
# transient paper: dom-hybrid rate-form waypoints + DOM gamma target),
# extract_ground_truth machinery (gamma) + analyze_shape_pilots (RMSE).
L_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
# ledger, not from a stored npz field and not typed here.  The npz field is the
# value the TRAINER wrote -- positive part over the arbiter's own n_decay=12
# sits opposite tab:regime, so plotting one convention beside the other is the
# defect this re-pin exists to remove.
# PN's OWN half of the cell-D ledger.  Split out of the shared
# Regenerating it for a PN re-pin therefore regenerated a frozen paper's numbers
# as a side effect.  Two artefacts, two `--only` invocations, no side effect.
_LEDGER = json.loads(
    (REPO / "data" / "cell_d_pn.json").read_text())

def _cell_d(section: str, key: str, field: str) -> float:
    rows = {str(r["label"]): r for r in _LEDGER[section]}
    return float(rows[key][field])

def _cell_d_run(run: str, section: str, field: str = "after") -> float:
    """Look a re-scored value up by its RUN identifier, never by disk order."""
    for r in _LEDGER[section]:
        if str(r["run"]).endswith(run):
            return float(r[field])
    raise KeyError(f"{run} not in ledger section {section}")

_C13 = "pn_c13_supervised"
SUPERVISED_G = (
    _cell_d(_C13, "C13 L=0.01", "gr_cellD"),
    _cell_d(_C13, "C13 L=0.1", "gr_cellD"),
    # the L=1 point is the three-seed MEAN, as it was before the re-pin
    sum(_cell_d(_C13, f"C13 L=1.0 {s}", "gr_cellD")
        for s in ("s42", "s0", "s1")) / 3.0,
    _cell_d(_C13, "C13 L=10", "gr_cellD"),
    _cell_d(_C13, "C13 L=100", "gr_cellD"))
SUPERVISED_RMSE = (
    _cell_d(_C13, "C13 L=0.01", "rmse"),
    _cell_d(_C13, "C13 L=0.1", "rmse"),
    # the L=1 point is the three-seed MEAN, as for the ratio above
    sum(_cell_d(_C13, f"C13 L=1.0 {s}", "rmse")
        for s in ("s42", "s0", "s1")) / 3.0,
    _cell_d(_C13, "C13 L=10", "rmse"),
    _cell_d(_C13, "C13 L=100", "rmse"))

# Unsupervised MLP control (sampled-residual trainer, waypoints and dt-IC
# off): batch-1/1b closure verdicts (arch campaign, Jul 7 2026) — collapse
# at the transitional pole, freeze at the ballistic pole.
# The data-free MLP control at its two poles, four seeds each, plotted BY RUN
# IDENTIFIER and re-scored under the ruled convention (cell_d_pn.json,
# seeds 0/1/2 plus the seed-42 arm of the same configuration at each pole.
MLP_CONTROL = {
    1.0: tuple(_cell_d_run(r, "pn_mlp_datafree", "gr_cellD") for r in (
        "20260707_085244", "20260707_061448",
        "20260707_073337", "20260703_233149")),
    0.01: tuple(_cell_d_run(r, "pn_mlp_datafree", "gr_cellD") for r in (
        "20260805_141042", "20260707_124826",
        "20260707_101129", "20260707_112951")),
}

# ------------------------------------------------------------------- styles
# House idiom (make_absolute_ladder_figure.py): module-level dicts splatted at
# every use site so no two panels can drift, and the arbiter drawn HEAVY GREY,
# UNMARKED, SEMI-TRANSPARENT and UNDERNEATH so the reference reads as
# background rather than as a rival series.  Grayscale rule: every series
# differs in colour AND marker AND linestyle.
DOM = dict(color="0.35", ls="-", lw=2.4, alpha=0.55, zorder=1,
           label="DOM reference")
#: The composition is the paper's result: the only saturated colour, the only
#: filled marker, always on top.
V5 = dict(color="C1", marker="*", ls="-", lw=1.3, ms=6.0, mfc="C1", mec="k",
          mew=0.35, zorder=6, label="PN composition (UGKS blend)")
V5_LABEL = V5["label"]

# The reduced formulations and the two MLP controls: appendix only.  Labels are
# UNCHANGED from the pre-split figure, because the manuscript's prose names
# them by these strings.
FORM_STYLE = {
    "plain": dict(color="C0", marker="o", ls="-", lw=0.9, ms=3.2, alpha=0.9,
                  label="PN (plain)"),
    "duh": dict(color="C2", marker="^", ls="-.", lw=0.9, ms=3.2, alpha=0.9,
                label="PN + Duhamel"),
    "ce": dict(color="C3", marker="s", ls="--", lw=0.9, ms=3.2, alpha=0.9,
               label="PN + Chapman–Enskog"),
}
SUP = dict(color="0.35", marker="D", ls=":", lw=0.9, ms=3.0, mfc="white",
           mew=0.7, zorder=4, label="MLP + DOM-guided waypoints")
MLP = dict(color="0.45", marker="x", ls="none", ms=4.2, mew=1.0, zorder=3,
           label="MLP, data-free (4 seeds)")
BAND = dict(color="0.85", alpha=0.6, zorder=0, label=r"$\pm$10% band")

# L values each PN formulation covers (order = plot/legend order).
# decade except plain @ L=100 (the unmeasurable CPU probe, tab:regime
# footnote a -- no artefact survives to draw).
FORM_LS = (
    ("plain", (0.01, 0.1, 1.0, 10.0)),
    ("duh", (0.01, 0.1, 1.0, 10.0, 100.0)),
    ("ce", (0.01, 0.1, 1.0, 10.0, 100.0)),
)

def fit_check(fig, stem):
    """Every text artist must lie INSIDE the canvas, which is the print slot."""
    fig.canvas.draw()
    W, H = (fig.get_size_inches() * fig.dpi)
    bad = []
    # Ticks OUTSIDE the view are artists matplotlib keeps and never draws; a
    # naive scan flags them and hides the real defect underneath the noise.
    dead = set()
    for a in fig.axes:
        for axis, lim in ((a.xaxis, a.get_xlim()), (a.yaxis, a.get_ylim())):
            lo, hi = sorted(lim)
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tk.get_loc() <= hi + 1e-9:
                    dead.update((id(tk.label1), id(tk.label2)))
    for t in fig.findobj(plt.Text):
        if not t.get_visible() or not str(t.get_text()).strip():
            continue
        bb = t.get_window_extent(fig.canvas.get_renderer())
        # A label wholly off the canvas is an out-of-view log tick matplotlib
        # keeps an artist for and never draws; the defect is the label that is
        # PARTLY on the page, because that is the one that gets cut in half.
        if bb.x1 < 0 or bb.x0 > W or bb.y1 < 0 or bb.y0 > H:
            continue
        if id(t) in dead:
            continue
        if bb.x0 < -0.5 or bb.x1 > W + 0.5 or bb.y0 < -0.5 or bb.y1 > H + 0.5:
            bad.append(f"{str(t.get_text())[:38]!r} at "
                       f"[{bb.x0:.0f},{bb.x1:.0f}]x[{bb.y0:.0f},{bb.y1:.0f}] "
                       f"vs canvas {W:.0f}x{H:.0f}")
    for lg in fig.legends + [a.get_legend() for a in fig.axes]:
        if lg is None:
            continue
        bb = lg.get_window_extent(fig.canvas.get_renderer())
        if bb.x0 < -0.5 or bb.x1 > W + 0.5:
            bad.append(f"legend at [{bb.x0:.0f},{bb.x1:.0f}] vs width {W:.0f}")
    if bad:
        raise SystemExit(f"{stem}: ink outside the canvas:\n "
                         + "\n ".join(bad))

# ------------------------------------------------------------------ loading
#: JOINT replacements for the ablation runs above, the twelve arms of
#: scripts/queues/p118_joint_repin_queue.sh.  They follow the SAME switch as
#: the ladder: an appendix drawn from legacy ablations beside a headline drawn
#: from joint composition is a mixed-comb figure, and `_arbiter_of()` catches
#: it -- it fired on exactly that state before this table existed.
RUNS_JOINT = {
    (0.01, "duh"): "abl_L0.01_duhamel",
    (0.01, "plain"): "abl_L0.01_plain",
    (0.1, "plain"): "abl_L0.1_plain_lmax16",
    (0.1, "duh"): "abl_L0.1_duhamel",
    (0.1, "ce"): "abl_L0.1_ce",
    (1.0, "plain"): "abl_L1.0_plain_s42",
    (1.0, "plain-s0"): "abl_L1.0_plain_s0",
    (1.0, "plain-s1"): "abl_L1.0_plain_s1",
    (1.0, "ce"): "abl_L1.0_ce",
    (10.0, "plain"): "abl_L10.0_plain",
    (10.0, "ce"): "abl_L10.0_ce",
    (100.0, "ce"): "abl_L100.0_ce",
    # command byte-for-byte, own campaign directory -- a value carrying a
    # slash names its parent under runs/transient (as in the cell-D ledger).
    (0.01, "ce"): "p135_ablation_cells/abl_L0.01_ce",
    (1.0, "duh"): "p135_ablation_cells/abl_L1.0_duhamel",
    (10.0, "duh"): "p135_ablation_cells/abl_L10.0_duhamel",
    (100.0, "duh"): "p135_ablation_cells/abl_L100.0_duhamel",
}

#: Joint-comb path (relative to runs/transient) -> (set, label), from the
#: STATIC `ablations.joint` table of ladders.yaml.  RUNS_JOINT above stays the
#: (L, form) -> arm bridge; the ACTIVE ladder's directory for that arm is then
#: looked up BY LABEL in `ablations[LADDER_SOURCE]` -- the one map the claim
#: with the numbers and cannot leave it on a retired comb.
_LABEL_BY_JOINT_PATH = {rel.removeprefix("data/runs/"): (setname, label)
                        for setname in ("main", "cited")
                        for label, rel in _LADDERS["ablations"]["joint"][setname].items()}

def ablation_dir(L, form):
    sub = RUNS_JOINT[(L, form)]
    joint_rel = sub if "/" in sub else f"p118_joint_repin/{sub}"
    setname, label = _LABEL_BY_JOINT_PATH[joint_rel]
    return REPO / _LADDERS["ablations"][LADDER_SOURCE][setname][label]

def load(L, form):
    # joint-family comb (joint, joint-sumrule) resolves its battery through
    # ladders.yaml `ablations[LADDER_SOURCE]`.
    if LADDER_COMB in ("joint", "joint-sumrule"):
        folder = ablation_dir(L, form)
    else:
        folder = BASE / f"spectral_pn_{L}um" / RUNS[(L, form)]
    npz = list(folder.glob("*_results.npz"))
    if not npz:
        raise FileNotFoundError(folder)
    d = np.load(npz[0])
    got = str(d["mode_source"]) if "mode_source" in d.files else "legacy"
    if got != LADDER_COMB:
        raise AssertionError(f"{folder} carries the {got!r} comb but ladder "
                             f"{LADDER_SOURCE!r} declares {LADDER_COMB!r} ")
    return d

def _cell_d_gamma_ratio(d) -> float:
    """gamma_ratio under the ruled convention, from one run's stored traces.

    modulus estimator on both sides, the denominator being the run's own stored
    12001-sample reference trace (the arbiter's native grid truncated at this
    run's t_end), so a solution identical to the arbiter scores exactly 1.
    """
    from pinn_bte.physics.decay_metrics import (
        GAMMA_ESTIMATOR_RULED, integral_gamma,
    )
    t, A = np.asarray(d["t"]), np.asarray(d["A"])
    t_ref, A_ref = np.asarray(d["t_ref"]), np.asarray(d["A_ref"])
    if t_ref.size != 12001:
        raise SystemExit(
            f"stored reference has {t_ref.size} samples, not the arbiter's "
            f"native 12001 -- the ruled denominator does not apply to it")
    return (integral_gamma(t, A, estimator=GAMMA_ESTIMATOR_RULED)
            / integral_gamma(t_ref, A_ref, estimator=GAMMA_ESTIMATOR_RULED))

def series(form, Ls):
    g = [_cell_d_gamma_ratio(load(L, form)) for L in Ls]
    r = [float(load(L, form)["rmse"]) for L in Ls]
    return g, r

def load_v5(L):
    if L not in _LADDER_1D:
        raise KeyError(f"ladder {LADDER_SOURCE!r}/1d has no rung L={L} "
                       f"(has {sorted(_LADDER_1D)})")
    folder = REPO / _LADDER_1D[L]
    npz = list(folder.glob("*_results.npz"))
    if not npz:
        raise FileNotFoundError(folder)
    d = np.load(npz[0], allow_pickle=True)
    # Fail loudly on a comb the caller did not ask for.  A run that predates
    # the `mode_source` field is legacy BY DEFINITION -- the field was added
    # with the joint comb -- so its ABSENCE is the legacy signature, and an
    # absent field must never read as "fine".
    got = str(d["mode_source"]) if "mode_source" in d.files else "legacy"
    want = LADDER_COMB
    if got != want:
        raise AssertionError(
            f"LADDER_SOURCE={LADDER_SOURCE!r} wants the {want!r} comb but "
            f"{folder} carries {got!r}. Refusing to draw a figure whose comb "
            f"is not the one the caller declared .")
    return d

def series_v5(Ls):
    g = [_cell_d_gamma_ratio(load_v5(L)) for L in Ls]
    r = [float(load_v5(L)["rmse"]) for L in Ls]
    return g, r

def _arbiter_of(dv, others=()):
    """(t/t_end, A_ref/A_ref[0], t_end) from the COMPOSITION run of a period.

    The pre-split figure took both the arbiter curve and the t_end
    normalisation from the first ablation pick.  Every run at a period stores
    the same arbiter, so this asserts rather than assumes: any drift between
    the stored references would silently move the drawn shapes, and a silent
    move is exactly what this check exists to prevent.
    """
    t_end = float(dv["t_end"])
    t_ref, A_ref = np.asarray(dv["t_ref"]), np.asarray(dv["A_ref"])
    for d in others:
        if not (np.array_equal(np.asarray(d["t_ref"]), t_ref)
                and np.array_equal(np.asarray(d["A_ref"]), A_ref)
                and float(d["t_end"]) == t_end):
            raise SystemExit(
                "stored DOM reference differs between runs of the same period "
                "-- the arbiter curve is no longer a property of the period")
    return t_ref / t_end, A_ref / A_ref[0], t_end

# ------------------------------------------------------------------ helpers
def _decades(ax, label=True):
    """The L axis: five labelled decades, minor labels killed.

    The minor formatter keeps labelling 2e-2, 5e-2 ... on top of the majors;
    set_xticklabels only ever touched the majors (the trap
    make_absolute_ladder_figure.py:_decades records).
    """
    ax.set_xscale("log")
    ax.set_xlim(6e-3, 1.7e2)
    ax.set_xticks(L_GRID)
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"], fontsize=MATH_PT)
    ax.xaxis.set_minor_formatter(NullFormatter())
    if label:
        ax.set_xlabel(r"grating period $L$ [$\mu$m]", fontsize=MATH_PT)

def _tag(ax, s, x=0.035, y=0.045, ha="left", va="bottom"):
    """Panel letter in a corner the DATA does not reach -- chosen per panel,
    never a fixed corner (make_absolute_ladder_figure.py:_tag)."""
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va,
            fontweight="bold", fontsize=MATH_PT)

def _emit(fig, stem, outdir, metadata=None):
    fit_check(fig, stem)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{stem}.{ext}",
                    **({"metadata": metadata} if ext == "pdf" and metadata else {}))
    plt.close(fig)
    w, h = fig.get_size_inches() * 72
    print(f"{stem}: canvas {w:.1f} x {h:.1f} pt (slot {SLOT_IN * 72:.2f} pt) "
          f"-> {outdir}")

# ------------------------------------------------------------- fig 2: regime map
def fig2(rows, outdir: Path = FIG_DIR):
    """MAIN: the pinned corridor ladder against the arbiter, both geometries."""
    fig, (ax_g, ax_r) = plt.subplots(1, 2, figsize=(SLOT_IN, 2.95))

    g1L, g1 = repin._geo_series(rows, "1d", "gamma_ratio")
    g2L, g2 = repin._geo_series(rows, "2d", "gamma_ratio")
    r1L, r1 = repin._geo_series(rows, "1d", "rmse")
    r2L, r2 = repin._geo_series(rows, "2d", "rmse")

    # ---- (a) decay-rate ratio against the arbiter -------------------------
    band = ax_g.axhspan(1 - repin.FLOOR_PCT / 100, 1 + repin.FLOOR_PCT / 100,
                        color="0.85", alpha=0.8, zorder=0)
    ax_g.axhline(1.0, color="k", lw=0.5, zorder=1)
    l1d, c1d, m1d, s1d = repin.G1D_STYLE
    l2d, c2d, m2d, s2d = repin.G2D_STYLE
    h1 = ax_g.plot(g1L, g1, color=c1d, marker=m1d, ls=s1d, lw=1.2, ms=3.9,
                   zorder=5)[0]
    h2 = ax_g.plot(g2L, g2, color=c2d, marker=m2d, ls=s2d, lw=1.2, ms=3.9,
                   mfc="white", zorder=5)[0]
    ax_g.set_ylim(0.985, 1.015)
    ax_g.set_yticks([0.985, 0.99, 0.995, 1.0, 1.005, 1.01, 1.015])
    ax_g.set_yticklabels(["0.985", "0.990", "0.995", "1.000",
                          "1.005", "1.010", "1.015"], fontsize=MATH_PT)
    ax_g.set_ylabel(r"$\gamma_{\rm eff}^{\rm model}\,/\,\gamma_{\rm eff}^{\rm DOM}$",
                    fontsize=MATH_PT)
    _decades(ax_g)
    # Bottom-left: the deepest rungs sit at 0.9945, half a band above the
    # 0.985 floor, so the strip under the data is empty.
    _tag(ax_g, "(a)")

    # ---- (b) shape RMSE against the same arbiter --------------------------
    ax_r.plot(r1L, r1, color=c1d, marker=m1d, ls=s1d, lw=1.2, ms=3.9, zorder=5)
    ax_r.plot(r2L, r2, color=c2d, marker=m2d, ls=s2d, lw=1.2, ms=3.9,
              mfc="white", zorder=5)
    ax_r.set_yscale("log")
    ax_r.set_ylim(1e-4, 2e-2)
    # TWO LINES: at MATH_PT this label on one line is 150 pt tall against a
    # ~110 pt axes and fit_check reported it off the canvas.  The alternative
    # was shrinking it below the floor, which is the trade this whole pass
    # exists to refuse.
    ax_r.set_ylabel("shape RMSE vs DOM\n" + r"$A(t)/A_0$", fontsize=MATH_PT)
    _decades(ax_r)
    # The curves peak at the ballistic rung (4.1e-3) and dip at L=1.5
    # (2.7e-4), a decade above the 1e-4 floor: bottom-left is empty.
    _tag(ax_r, "(b)")

    for ax in (ax_g, ax_r):
        ax.tick_params(labelsize=MATH_PT)

    # ONE legend under the row, no frame (house idiom).  Three entries on TWO
    # columns: on one row they measure 405 pt against the 388.5 pt slot
    # (fit_check caught the 16 pt overhang), and shrinking type or handles to
    # squeeze them in is the trade this module refuses.
    fig.legend([h1, h2, band], [l1d, l2d,
                rf"$\pm${repin.FLOOR_PCT}% DOM floor"],
               loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.012), handlelength=1.9,
               columnspacing=1.0, handletextpad=0.45, fontsize=MATH_PT)
    fig.tight_layout(rect=(0, 0.185, 1, 1.0), w_pad=1.2, pad=0.35)
    repin.data_check(fig, "fig2_regime_map")
    repin.legend_check(fig, "fig2_regime_map")
    _emit(fig, "fig2_regime_map", outdir)
    print("  gamma 1d: " + " ".join(f"{v:.4f}" for v in g1))
    print("  gamma 2d: " + " ".join(f"{v:.4f}" for v in g2))
    print("  rmse  1d: " + " ".join(f"{v:.2e}" for v in r1))
    print("  rmse  2d: " + " ".join(f"{v:.2e}" for v in r2))

# ------------------------------------------------------------- fig 3: traces
# Per L: [(formulation, role)] with role 'win' (thick) or 'fail' (thin) —
TRACE_PICKS = {
    0.01: (("duh", "win"), ("plain", "fail")),
    0.1: (("plain", "win"), ("ce", "fail")),
    1.0: (("plain", "win"), ("ce", "fail")),
    10.0: (("ce", "win"), ("plain", "fail")),
    100.0: (("ce", "win"),),
}
# Marker cadence per series, so no two markers land on top of each other and
# promise was false while DOM and `plain` were both unmarked solid lines).
MARK_EVERY = {"v5": (12, 150), "plain": (50, 150), "duh": (87, 150),
              "ce": (125, 150)}

#: Presentation threshold for the signed decade row.  Set deliberately ABOVE
#: the deepest SOLVER excursion (measured on the corridor60k pins: the PN
#: lobe at L = 0.01 um reaches -1.975e-2 of A_0 in both geometries; on the
#: joint-20k pins it was -1.74e-2).  The ARBITER's own lobe is deeper,
#: -2.72e-2, so its minimum renders just below the linear band in the first
#: negative fraction of a decade -- visible, not clipped (axis floor -0.08).
SIGN_LINTHRESH = 2.0e-2

def _signed_decade_axis(ax, linthresh=SIGN_LINTHRESH):
    """Symmetric-log y for a SIGNED amplitude, sub-threshold band kept flat."""
    ax.axhline(0.0, color="0.75", lw=0.6, zorder=0)
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.35)
    ax.set_ylim(-4.0 * linthresh, 1.5)
    ax.set_yticks([1.0, 1e-2, 0.0, -2.0 * linthresh])
    ax.set_yticklabels(["$1$", "$10^{-2}$", "$0$", f"$-{2.0 * linthresh:g}$"])
    ax.yaxis.set_minor_locator(NullLocator())

#: fig3's own DOM stroke (the figC convention): black, on top of nothing,
#: under both PN series.  The heavy grey house stroke is for references drawn
#: UNDER a single rival series; here two near-coincident PN curves ride it and
#: grey-under-two reads as a third data series in grayscale.
DOM3_LW = 1.45

def _dom_separation_pt(ax, d1, d2):
    """Max on-page vertical separation (pt) of the two geometries' arbiters.

    Each geometry's npz stores ITS OWN independently-computed arbiter
    (different t_end / A_ref / collocation).  Whether one black line may stand
    for both is a statement about INK, so it is measured in display points on
    the axes as configured, not assumed from a tolerance in data units --
    symlog makes the data->page map nonlinear.  Call only after the axes
    limits/scales AND the figure layout are final.
    """
    te1, te2 = float(d1["t_end"]), float(d2["t_end"])
    t1 = np.asarray(d1["t_ref"]) / te1
    a1 = np.asarray(d1["A_ref"]) / np.asarray(d1["A_ref"])[0]
    t2 = np.asarray(d2["t_ref"]) / te2
    a2 = np.asarray(d2["A_ref"]) / np.asarray(d2["A_ref"])[0]
    p1 = ax.transData.transform(np.column_stack([t1, a1]))
    p2 = ax.transData.transform(np.column_stack([t1, np.interp(t1, t2, a2)]))
    return (float(np.max(np.abs(p1[:, 1] - p2[:, 1]))) * 72.0 / ax.figure.dpi,
            (t1, a1), (t2, a2))

def fig3(rows, outdir: Path = FIG_DIR):
    """Signed shape on the active ladder, all five periods and both geometries.

    The wide ballistic panel preserves the first sign crossing; all trace
    arrays are unchanged. PDF Subject records the plotted-array fingerprint,
    independently checked against the active NPZ pins by the provenance test.
    """
    # Give the kinetic lobe a wide panel; keep all five original periods.
    fig = plt.figure(figsize=(SLOT_IN, 4.6))
    gs = fig.add_gridspec(2, 6, left=0.12, right=0.98, top=0.95,
                          bottom=0.15, hspace=0.62, wspace=0.95)
    axes = [fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:]),
            fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2:4]),
            fig.add_subplot(gs[1, 4:])]
    l1d, c1d, m1d, s1d = repin.G1D_STYLE
    l2d, c2d, m2d, s2d = repin.G2D_STYLE
    # Axes fully configured BEFORE any artist: _dom_separation_pt reads
    # transData, which is only meaningful once scales, limits and the
    # tight_layout positions are final.
    for j, (ax, L) in enumerate(zip(axes, L_GRID)):
        ax.set_title(f"$L={L:g}\\,\\mu$m", pad=2.5, fontsize=10)
        _signed_decade_axis(ax)
        ax.set_xlim(0, 1)
        ax.set_xticks((0.0, 0.5, 1.0))
        ax.set_xticklabels(("0", "0.5", "1"), fontsize=MATH_PT)
        ax.set_xlabel(r"$t/t_{\rm end}$", labelpad=1.5, fontsize=MATH_PT)
        ax.tick_params(labelsize=MATH_PT)
    axes[0].set_ylabel(r"$A(t)/A_0$", labelpad=1.5, fontsize=MATH_PT)

    handles = {}
    trace_digest = hashlib.sha256()
    for j, (ax, L) in enumerate(zip(axes, L_GRID)):
        b1, b2 = rows[(L, "1d")], rows[(L, "2d")]
        d1, d2 = b1["_npz"], b2["_npz"]
        sep_pt, (t1, a1), (t2, a2) = _dom_separation_pt(ax, d1, d2)
        if sep_pt < DOM3_LW:
            (h,) = ax.plot(t1, a1, "k-", lw=DOM3_LW, zorder=4)
            handles.setdefault("DOM reference", h)
        else:
            (h,) = ax.plot(t1, a1, "k-", lw=DOM3_LW, zorder=4)
            handles.setdefault("DOM reference (1d)", h)
            (h,) = ax.plot(t2, a2, color="0.35", ls=(0, (1, 1)), lw=DOM3_LW,
                           zorder=4)
            handles.setdefault("DOM reference (2d)", h)
        print(f"  L={L:g}: arbiter 1d-vs-2d separation {sep_pt:.3f} pt "
              f"({'one line' if sep_pt < DOM3_LW else 'BOTH drawn'})")
        # Marker cadences staggered so the two geometries' markers never land
        # on the same sample (grayscale rule: colour AND marker AND linestyle).
        (h,) = ax.plot(d1["t"] / float(d1["t_end"]), d1["A"] / d1["A"][0],
                       color=c1d, ls=s1d, lw=0.85, marker=m1d,
                       markevery=(40, 150), ms=3.0, mfc="white", zorder=5)
        handles.setdefault(l1d, h)
        (h,) = ax.plot(d2["t"] / float(d2["t_end"]), d2["A"] / d2["A"][0],
                       color=c2d, ls=s2d, lw=0.85, marker=m2d,
                       markevery=(115, 150), ms=3.0, mfc="white", zorder=6)
        handles.setdefault(l2d, h)
        # Hash exactly the plotted arrays into PDF metadata. Independent
        # tests retain content-based pin checks without ten repeated rates.
        for d in (d1, d2):
            for values in (d["t"] / float(d["t_end"]), d["A"] / d["A"][0]):
                trace_digest.update(np.asarray(values, dtype="<f8").tobytes())
        # Bottom-left: every trace is still near A ~ 1 at early time, so the
        # foot of the panel below the zero line is empty there (the L=0.01
        # negative lobe lives at mid-window, x ~ 0.3-0.7).
        _tag(ax, f"({chr(97 + j)})", x=0.05, y=0.05)
    # A linear late-window inset resolves the reference's second negative
    # lobe. The full traces remain in panel (a); the inset is explicitly x1000.
    inset = axes[0].inset_axes([.54, .58, .42, .32])
    d1, d2 = rows[(.01, "1d")]["_npz"], rows[(.01, "2d")]["_npz"]
    inset_ymax = 0.
    for d, ref, style in ((d1, True, dict(color="k", lw=.9)),
                           (d1, False, dict(color=c1d, ls=s1d, lw=.8)),
                           (d2, False, dict(color=c2d, ls=s2d, lw=.8))):
        xx = d["t_ref" if ref else "t"] / float(d["t_end"])
        yy = d["A_ref" if ref else "A"]
        yy = 1000. * yy / yy[0]
        keep = (xx >= .85) & (xx <= 1.)
        inset.plot(xx[keep], yy[keep], **style)
        inset_ymax = max(inset_ymax, float(yy[keep].max()))
    inset.axhline(0, color=".6", lw=.45)
    inset.set_xlim(.85, 1.)
    inset_top = float(np.ceil(inset_ymax))
    inset.set_ylim(-1., inset_top + .2)
    inset.set_xticks([.85, 1.]); inset.set_xticklabels(["0.85", "1"], fontsize=8)
    inset.set_yticks([-1, 0, inset_top])
    inset.set_yticklabels(["−1", "0", f"{inset_top:g}"], fontsize=8)
    inset.set_title("late window (×1000)", fontsize=8, pad=2.)
    inset.grid(False)
    order = [l1d, l2d, "DOM reference", "DOM reference (1d)",
             "DOM reference (2d)"]
    keys = [k for k in order if k in handles]
    fig.legend([handles[k] for k in keys], keys, loc="lower center",
               ncol=len(keys), frameon=False, bbox_to_anchor=(0.5, -0.012),
               handlelength=1.7, columnspacing=0.9, handletextpad=0.4,
               fontsize=MATH_PT)
    repin.data_check(fig, "fig3_traces")
    repin.legend_check(fig, "fig3_traces")
    _emit(fig, "fig3_traces", outdir,
          metadata={"Subject": "PN signed-trace SHA256: " + trace_digest.hexdigest()})
    print("  source gamma_r (1d/2d): " + " ".join(
        f"{L:g}:{rows[(L, '1d')]['gamma_ratio']:.4f}/"
        f"{rows[(L, '2d')]['gamma_ratio']:.4f}" for L in L_GRID))

# ------------------------------------- appendix: the reduced-formulation ladder
def fig_reduced_full(outdir: Path = FIG_DIR):
    """APPENDIX: every reduced and failed approach, on one page and complete.

    This is the paper's negative-result evidence, not an offcut: the three pure
    PN structures (each correct on its own pole), the data-free MLP control at
    four seeds and the DOM-supervised MLP.  It carries MORE than the main
    figures used to, not less -- the two metrics (a,b) AND the trace at every
    period (c-g), where the pre-split fig3 showed the reduced curves at the
    periods it had room for.

    The composition and the arbiter are drawn in every panel as the yardstick;
    without them the reader cannot see what "fails" means.
    """
    # TWO INDEPENDENT GRIDSPECS WITH EXPLICIT MARGINS, not a nested
    # subgridspec plus tight_layout: matplotlib warns "Axes that are not
    # compatible with tight_layout" on the nested form and then lays it out
    # anyway, which is a silent-wrong layout on a figure whose whole point is
    # legibility.  Every margin below is a measured budget in points on the
    # 388.5 x 360 pt canvas: 47 pt left for the y labels and ticks, ~32 pt
    # under each row for its x label and ticks, 14 pt above the trace row for
    # its titles, and 66 pt at the foot for the legend -- 66 and not the 54 pt
    # first budgeted here, because a 4-row legend at 10.5 pt MEASURES 64-75 pt
    # on the render (labelspacing sets which) and its top line was sitting
    # across the trace row's `t/t_end` labels.  The top row's 59 pt gutter is
    # likewise measured: panel (b)'s two-line y label plus its decade ticks
    # need 48 pt and had 50.
    fig = plt.figure(figsize=(SLOT_IN, 5.30))
    gs_top = fig.add_gridspec(1, 2, left=0.122, right=0.972,
                              top=0.905, bottom=0.615, wspace=0.44)
    gs_bot = fig.add_gridspec(1, 5, left=0.122, right=0.972,
                              top=0.498, bottom=0.254, wspace=0.28)
    ax_g = fig.add_subplot(gs_top[0, 0])
    ax_r = fig.add_subplot(gs_top[0, 1])
    trace_axes = [fig.add_subplot(gs_bot[0, j]) for j in range(5)]

    fig.suptitle("PN: 60k composition / 20k pure controls; frozen coefficients\n"
                 "Historical MLP: 15k, 250–350 K, local τ(T), predecessor mode set",
                 fontsize=9, y=0.99, linespacing=1.2)
    gv5, rv5 = series_v5(L_GRID)
    handles = {}

    def _keep(label, h):
        handles.setdefault(label, h)

    # ---- (a) decay-rate ratio.  LOG y: the controls span 0.147 to 97.5
    ax_g.axhspan(0.9, 1.1, lw=0, **BAND)
    _keep(DOM["label"], ax_g.axhline(1.0, **DOM))
    for L, vals in sorted(MLP_CONTROL.items()):
        h = ax_g.plot([L] * len(vals), vals, **MLP)[0]
        _keep(MLP["label"], h)
    _keep(SUP["label"], ax_g.plot(L_GRID, SUPERVISED_G, **SUP)[0])
    for form_key, Ls in FORM_LS:
        g, _ = series(form_key, Ls)
        _keep(FORM_STYLE[form_key]["label"],
              ax_g.plot(Ls, g, **FORM_STYLE[form_key])[0])
    # plain seed spread at L=1.0, open markers on the plain colour: three
    # seeds of the same formulation, not three formulations.
    for key in ("plain-s0", "plain-s1"):
        ax_g.plot(1.0, _cell_d_gamma_ratio(load(1.0, key)), marker="o",
                  color="C0", ls="none", mfc="none", ms=3.2, alpha=0.9,
                  zorder=5)
    _keep(V5_LABEL, ax_g.plot(L_GRID, gv5, **dict(V5, ms=5.0, lw=1.1))[0])
    ax_g.set_yscale("log")
    # exists to catch a clipped point, never relax it to keep a tight axis.
    ax_g.set_ylim(0.1, 130)
    yt = (0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 100)
    ax_g.set_yticks(yt)
    ax_g.set_yticklabels([f"{v:g}" for v in yt], fontsize=MATH_PT)
    ax_g.minorticks_off()
    ax_g.set_ylabel(r"$\gamma_{\rm eff}^{\rm model}\,/\,\gamma_{\rm eff}^{\rm DOM}$",
                    fontsize=MATH_PT)
    _decades(ax_g)
    # Top-left: nothing reaches gamma > 2 at the ballistic period.
    _tag(ax_g, "(a)", x=0.035, y=0.94, va="top")

    # ---- (b) shape RMSE ---------------------------------------------------
    ax_r.plot(L_GRID, SUPERVISED_RMSE, **SUP)
    for form_key, Ls in FORM_LS:
        _, r = series(form_key, Ls)
        ax_r.plot(Ls, r, **FORM_STYLE[form_key])
    ax_r.plot(L_GRID, rv5, **dict(V5, ms=5.0, lw=1.1))
    ax_r.set_yscale("log")
    ax_r.set_ylim(3e-5, 3.0)
    # TWO LINES: at MATH_PT this label on one line is 150 pt tall against a
    # ~110 pt axes and fit_check reported it off the canvas.  The alternative
    # was shrinking it below the floor, which is the trade this whole pass
    # exists to refuse.
    ax_r.set_ylabel("shape RMSE vs DOM\n" + r"$A(t)/A_0$", fontsize=MATH_PT)
    _decades(ax_r)
    # Bottom-left: the smallest value at the ballistic period is 7.7e-3, two
    # decades above the window floor.
    _tag(ax_r, "(b)")

    # ---- (c)-(g) the traces, log, at every period -------------------------
    # Log rather than linear-plus-log: one row of five holds the same evidence
    # (freeze, drift, late turnover) at twice the panel height, and the density
    # complaint against the pre-split fig3 was two rows of five with four
    # series in each.
    for j, L in enumerate(L_GRID):
        ax = trace_axes[j]
        dv = load_v5(L)
        picks = TRACE_PICKS[L]
        # Assert the arbiter is a property of the PERIOD, not of the run.
        tr, Ar, t_end = _arbiter_of(dv, [load(L, f) for f, _ in picks])
        # Signed, exactly as the main figure -- see _signed_decade_axis.  This
        # row carries MORE curves through the crossing than Figure 3 does, so
        # leaving it on clip(abs(.)) would have made the appendix contradict the
        # figure it exists to support.
        ax.plot(tr, Ar, **DOM)
        for form_key, role in picks:
            d = load(L, form_key)
            # zorder 7: the reduced curve draws ON TOP of the composition.
            # This figure's job is to let the reader check the reduced
            # structures, and at L = 0.01 um the claim is that Duhamel and the
            # composition nearly COINCIDE -- a curve hidden underneath the one
            # it is claimed to coincide with cannot be checked.  Thin lines and
            # open markers keep the composition reading as the yardstick.
            st = dict(FORM_STYLE[form_key], markevery=MARK_EVERY[form_key],
                      ms=2.8, mew=0.5, mfc="none", zorder=7,
                      lw=0.9 if role == "win" else 0.7,
                      alpha=1.0 if role == "win" else 0.8)
            st.pop("label")
            ax.plot(d["t"] / t_end, d["A"] / d["A"][0], **st)
        kw = dict(V5, markevery=MARK_EVERY["v5"], ms=3.8, lw=0.9)
        kw.pop("label")
        ax.plot(dv["t"] / t_end, dv["A"] / dv["A"][0], **kw)
        ax.set_title(f"$L={L:g}\\,\\mu$m", pad=2.5, fontsize=10)
        _signed_decade_axis(ax)
        ax.set_xlim(0, 1)
        ax.set_xticks((0.0, 0.5, 1.0))
        ax.set_xticklabels(("0", "0.5", "1"), fontsize=MATH_PT)
        ax.set_xlabel(r"$t/t_{\rm end}$", labelpad=1.5, fontsize=MATH_PT)
        ax.tick_params(labelsize=MATH_PT)
        if j:
            ax.tick_params(labelleft=False)
        else:
            # SIGNED now, so the modulus bars are gone from the label too -- a
            # stale "|A|" over a signed axis is exactly the kind of mismatch
            # that sends a reader looking for a cusp that is no longer there.
            ax.set_ylabel(r"$A(t)/A_0$", labelpad=1.5, fontsize=MATH_PT)
        # Bottom-left: every trace starts at 1 and decays, so the early-time
        # floor of the window is empty in all five panels.
        _tag(ax, f"({chr(99 + j)})")

    # ONE legend under the whole figure.  Eight entries at MATH_PT do not fit
    # on three columns -- "MLP + DOM-guided waypoints" alone is 165 pt with its
    # handle, and at ncol=2 the first attempt still ran 3 pt off the canvas
    # (fit_check caught it) -- so two columns, four rows, tightened spacing.
    # The method and the arbiter lead; the band is named rather than left as
    # an unexplained grey stripe.
    order = [V5_LABEL, DOM["label"], FORM_STYLE["plain"]["label"],
             FORM_STYLE["duh"]["label"], FORM_STYLE["ce"]["label"],
             SUP["label"], MLP["label"], BAND["label"]]
    handles[BAND["label"]] = plt.Rectangle(
        (0, 0), 1, 1, **{k: v for k, v in BAND.items() if k != "zorder"})
    fig.legend([handles[k] for k in order], order, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.004), handlelength=1.9,
               columnspacing=1.0, handletextpad=0.4, labelspacing=0.38,
               fontsize=MATH_PT)
    _emit(fig, "fig_reduced_traces_appendix", outdir)
    for form_key, Ls in FORM_LS:
        g, r = series(form_key, Ls)
        print(f"  {form_key:6s} gamma " + " ".join(f"{v:.4f}" for v in g)
              + "  rmse " + " ".join(f"{v:.2e}" for v in r))
    print("  supervised " + " ".join(f"{v:.4f}" for v in SUPERVISED_G))
    print("  MLP L=0.01 " + " ".join(f"{v:.4f}" for v in MLP_CONTROL[0.01]))
    print("  MLP L=1.0 " + " ".join(f"{v:.4f}" for v in MLP_CONTROL[1.0]))

def fig_reduced(outdir: Path = FIG_DIR):
    """Main comparison: two metrics and three decisive pure-PN failures.

    All values/curves come from the same loaders as the complete appendix.
    Historical MLP controls remain there, with their distinct physics stated.
    """
    fig = plt.figure(figsize=(SLOT_IN, 5.0))
    top = fig.add_gridspec(1, 2, left=0.12, right=0.98, top=0.91,
                           bottom=0.63, wspace=0.48)
    bottom = fig.add_gridspec(1, 3, left=0.13, right=0.98, top=0.44,
                              bottom=0.20, wspace=0.32)
    ag, ar = (fig.add_subplot(top[0, j]) for j in range(2))
    gv, rv = series_v5(L_GRID)
    handles = {}
    ag.axhspan(0.9, 1.1, **BAND)
    ag.axhline(1, **DOM)
    for form, periods in FORM_LS:
        g, r = series(form, periods)
        handles[form] = ag.plot(periods, g, **FORM_STYLE[form])[0]
        ar.plot(periods, r, **FORM_STYLE[form])
    handles["v5"] = ag.plot(L_GRID, gv, **V5)[0]
    ar.plot(L_GRID, rv, **V5)
    for key in ("plain-s0", "plain-s1"):
        ag.plot(1., _cell_d_gamma_ratio(load(1., key)), marker="o",
                color="C0", ls="none", mfc="none", ms=3.2, zorder=5)
    ag.set_yscale("log"); ag.set_ylim(.1, 130)
    ag.set_yticks([.1, 1, 10, 100]); ag.set_yticklabels(["0.1", "1", "10", "100"])
    ar.set_yscale("log"); ar.set_ylim(3e-5, 3.)
    ag.set_ylabel("rate ratio to DOM")
    ar.set_ylabel("trace RMSE vs DOM")
    for ax, tag in ((ag, "(a)"), (ar, "(b)")):
        _decades(ax); _tag(ax, tag, y=.95, va="top")
    selected = [(0.01, "ce", "CE: missed crossing"),
                (10., "plain", "plain: frozen trace"),
                (100., "duh", "Duhamel: fast decay")]
    for j, (L, form, title) in enumerate(selected):
        ax = fig.add_subplot(bottom[0, j])
        dv, dc = load_v5(L), load(L, form)
        tr, ar_ref, te = _arbiter_of(dv, [dc])
        ax.plot(tr, ar_ref, **DOM)
        ax.plot(dv["t"] / te, dv["A"] / dv["A"][0], **dict(V5, markevery=160, ms=4))
        ax.plot(dc["t"] / te, dc["A"] / dc["A"][0],
                **dict(FORM_STYLE[form], markevery=160, ms=3, zorder=8))
        _signed_decade_axis(ax)
        ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_title(f"{title}\n" + rf"$L={L:g}\,\mu$m", fontsize=9, pad=3)
        ax.set_xlabel(r"$t/t_{\rm end}$", labelpad=1)
        if j == 0: ax.set_ylabel(r"$A(t)/A_0$")
        else: ax.tick_params(labelleft=False)
        _tag(ax, f"({chr(99+j)})")
    labels = ["composition (60k)", "plain (20k)", "Duhamel (20k)", "CE (20k)"]
    fig.legend([handles[k] for k in ("v5", "plain", "duh", "ce")], labels,
                loc="lower center", ncol=2, frameon=False,
                bbox_to_anchor=(.5, -.005), fontsize=9, handlelength=2.)
    fig.suptitle("Pure PN controls on the same frozen material coefficients",
                  fontsize=9, y=.99)
    # The reference line and success band are named in a compact header.
    fig.text(.5, .948, "gray line: DOM reference; shaded band: ±10% in rate",
              fontsize=9, ha="center")
    _emit(fig, "fig_reduced_formulations", outdir)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=FIG_DIR,
                    help="write here instead of figures (dry run)")
    ap.add_argument("--only", choices=("regime", "traces", "reduced", "all"),
                    default="all", help="regenerate only the selected figure")
    args = ap.parse_args()
    outdir = args.outdir
    # ONE resolution for both main figures: 18 arms off ladders.yaml, comb-
    # guarded, battery-gated, cell-D re-scored (make_repin_figures.campaign_rows).
    if args.only in ("regime", "traces", "all"):
        rows = repin.campaign_rows()
        if args.only in ("regime", "all"):
            fig2(rows, outdir)
        if args.only in ("traces", "all"):
            fig3(rows, outdir)
    if args.only in ("reduced", "all"):
        fig_reduced(outdir)
        fig_reduced_full(outdir)

if __name__ == "__main__":
    main()
