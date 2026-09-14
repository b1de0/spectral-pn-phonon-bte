"""Pinned-ladder banded-rates figure for the spectral-PN paper."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/xiblend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign_eval as ce  # noqa: E402
from pn_shape_battery import gate  # noqa: E402

FIG_DIR = REPO / "figures"

def cell_d_gamma_ratio(d) -> float:
    """gamma_ratio under the RULED convention ( cell D)."""
    from pinn_bte.physics.decay_metrics import (
        GAMMA_ESTIMATOR_RULED, integral_gamma,
    )
    t, A = np.asarray(d["t"]), np.asarray(d["A"])
    t_ref, A_ref = np.asarray(d["t_ref"]), np.asarray(d["A_ref"])
    if t_ref.size != 12001:
        raise SystemExit(
            f"stored reference has {t_ref.size} samples, not the arbiter's "
            f"native 12001 -- decision 6 does not apply to this run")
    return (integral_gamma(t, A, estimator=GAMMA_ESTIMATOR_RULED)
            / integral_gamma(t_ref, A_ref, estimator=GAMMA_ESTIMATOR_RULED))
L_DECADES = (0.01, 0.1, 1.0, 10.0, 100.0)

# Style mirrors scripts/make_figures.py (grayscale-safe: every series
# differs in color AND marker AND linestyle).
#
# `\includegraphics[width=\textwidth]` into a 388.53 pt text block, so a canvas
# authored wider than that is demagnified and the font sizes below stop meaning
# anything: figB was authored 7.2 in but its single-row four-column legend
# inflated the tight bbox to 699.2 pt, printing the whole figure at 55.6% --
# 4.45-5.56 pt lettering, against Elsevier's 7 pt floor -- with the legend
# spanning the text width while the three data panels used 55% of it.  The
# canvas is now authored AT the slot, `savefig.bbox` is 'standard' so the saved
# width IS figsize[0]*72, fit_check() reports any ink that runs off it, and
# every line/marker size is the old value times the old on-page scale so the ink
# weight is unchanged.
SLOT_IN = 388.53 / 72.0          # 5.3963 in -- the \textwidth slot
DPI = 300

# and 6 pt for sub/superscripts; matplotlib's mathtext shrinks each script level
# by 0.7.  A 10 pt base therefore lands a subscript at EXACTLY 7.00 -- on the
# floor with ZERO headroom, which is where this module was and why any later
# shrink (a wider legend inflating a bbox, a float scaled down) broke the floor
# on arrival rather than being caught.  10.5 puts them at 7.35: 0.35 pt of
# margin, and it matches the venue evidence (RelaxNet, JCP 490 (2023) 112317,
# sets figure lettering at Helvetica 9.96 pt against a 7.97 pt body -- figure
# type is ~25% LARGER than body text, not smaller).
#
#   * MATH_PT  = base for any string that CAN carry a sub/superscript
#   * PLAIN_PT = base for plain words that never will
#   * nothing below PLAIN_PT, ever; nesting depth >= 2 is refused outright
#     (10.5 * 0.7^2 = 5.1 pt, under the floor at any base this slot affords)
#
# BASE, not what lands on the page.  The only valid check is over the built PDF
# (`mutool draw -F stext`).
MATH_PT = 10.5                   # -> 7.35 pt subscripts on the page
PLAIN_PT = 9.5                   # plain words, no script
plt.rcParams.update({
    "font.size": 9,              # fallback ONLY; every artist is set explicitly
    "axes.titlesize": MATH_PT,
    "axes.labelsize": MATH_PT,
    "xtick.labelsize": MATH_PT,
    "ytick.labelsize": MATH_PT,
    # was 8.5 -> any script in a legend entry would land at 5.95 pt, under the
    # floor, silently, the first time one was added.
    "legend.fontsize": MATH_PT,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "standard",
    "axes.linewidth": 0.45,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    # FONTS.  Elsevier permits Arial/Helvetica, Times, Courier, Symbol.  DejaVu
    # -- matplotlib's default, and what this module used to embed -- is on none
    # of those lists.  Nimbus Sans is the URW Helvetica clone; Liberation Sans
    # and Arimo are the metric-compatible Arial fallbacks.
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "Arimo",
                        "DejaVu Sans"],
    # which has no usable Unicode map, so pdftotext returns EVERY mathtext glyph
    # See make_schematic.py:196-213, where this was measured.  Serif math beside
    # Helvetica words is also the JCP idiom (RelaxNet Fig. 2).
    "mathtext.fontset": "stix",
    # encoding and uni=no: the text is not extractable at all (so the G6 gate
    # rests on nothing), and Type 3 artwork is a standard publisher-rejection
    # trigger.  42 = embed TrueType/CID, Identity-H, uni=yes.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.3,
})

def _dead_tick_ids(fig):
    """Tick labels OUTSIDE the view: artists matplotlib keeps and never draws.

    A naive scan flags them and buries the real defect under the noise -- and
    worse, a guard that trips on a phantom gets loosened until it stops
    tripping on the real thing too.  Shared by every geometric check below so
    they cannot disagree about what is actually on the page.
    """
    dead = set()
    for a in fig.axes:
        for axis, lim in ((a.xaxis, a.get_xlim()), (a.yaxis, a.get_ylim())):
            lo, hi = sorted(lim)
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tk.get_loc() <= hi + 1e-9:
                    dead.update((id(tk.label1), id(tk.label2)))
    return dead

def fit_check(fig, stem):
    """Every text artist must lie INSIDE the canvas, which is the print slot."""
    fig.canvas.draw()
    W, H = (fig.get_size_inches() * fig.dpi)
    r = fig.canvas.get_renderer()
    dead = _dead_tick_ids(fig)
    bad = []
    for t in fig.findobj(plt.Text):
        if not t.get_visible() or not str(t.get_text()).strip():
            continue
        bb = t.get_window_extent(r)
        if bb.x1 < 0 or bb.x0 > W or bb.y1 < 0 or bb.y0 > H:
            continue        # an out-of-view decade tick, never drawn
        if id(t) in dead:
            continue
        if bb.x0 < -0.5 or bb.x1 > W + 0.5 or bb.y0 < -0.5 or bb.y1 > H + 0.5:
            bad.append(f"{str(t.get_text())[:38]!r} at "
                       f"[{bb.x0:.0f},{bb.x1:.0f}]x[{bb.y0:.0f},{bb.y1:.0f}] "
                       f"vs canvas {W:.0f}x{H:.0f}")
    for lg in fig.legends + [a.get_legend() for a in fig.axes]:
        if lg is not None:
            bb = lg.get_window_extent(r)
            if bb.x0 < -0.5 or bb.x1 > W + 0.5:
                bad.append(f"legend [{bb.x0:.0f},{bb.x1:.0f}] vs width {W:.0f}")
    if bad:
        raise SystemExit(f"{stem}: ink outside the canvas:\n "
                         + "\n ".join(bad))

def legend_check(fig, stem, pad=1.0):
    """A figure legend must not sit ON an axes or on any label outside it."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dead = _dead_tick_ids(fig)
    legends = [lg for lg in fig.legends if lg is not None]
    legends += [lg for lg in (a.get_legend() for a in fig.axes)
                if lg is not None]
    bad = []
    for lg in legends:
        lbb = lg.get_window_extent(r).expanded(1.0, 1.0)
        own = {id(t) for t in lg.findobj(plt.Text)} | dead
        for ax in fig.axes:
            abb = ax.patch.get_window_extent(r)
            if (lbb.x0 < abb.x1 - pad and lbb.x1 > abb.x0 + pad
                    and lbb.y0 < abb.y1 - pad and lbb.y1 > abb.y0 + pad):
                bad.append(f"legend overlaps an axes rectangle "
                           f"[{abb.x0:.0f},{abb.x1:.0f}]x"
                           f"[{abb.y0:.0f},{abb.y1:.0f}]")
        for t in fig.findobj(plt.Text):
            if id(t) in own or not t.get_visible():
                continue
            if not str(t.get_text()).strip():
                continue
            tbb = t.get_window_extent(r)
            if (lbb.x0 < tbb.x1 - pad and lbb.x1 > tbb.x0 + pad
                    and lbb.y0 < tbb.y1 - pad and lbb.y1 > tbb.y0 + pad):
                bad.append(f"legend overlaps {str(t.get_text())[:34]!r} at "
                           f"[{tbb.x0:.0f},{tbb.x1:.0f}]x"
                           f"[{tbb.y0:.0f},{tbb.y1:.0f}]")
    if bad:
        raise SystemExit(f"{stem}: legend collides:\n "
                         + "\n ".join(sorted(set(bad))))

def data_check(fig, stem):
    """Every plotted DATA POINT must lie inside its axes' view limits.

    The complement to `fit_check`, which never looks at data artists at all --
    its docstring names "a MARKER clipped by the axes limits" as its first blind
    spot, and this module has already been bitten by exactly that (the extreme
    late-band point drawn half below the bottom spine, patched by `margins`).  A
    hand-set `set_ylim` is the other way in: it is a claim about the data's
    range, and a claim about data belongs in an assertion, not a comment.

    Clipping a datum is not a cosmetic defect -- it silently deletes evidence
    from a figure whose whole job is to show where the ladder departs from 1.
    """
    bad = []
    for ax in fig.axes:
        (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        for ln in ax.get_lines():
            if not ln.get_visible():
                continue
            # Only artists drawn in DATA coordinates are data.  `axhline` and
            # friends carry a blended transform (x in axes fractions, y in
            # data), so their xdata is the literal (0, 1) span of the axes and
            # comparing it against the x view limits is meaningless.
            if ln.get_transform() is not ax.transData:
                continue
            xs, ys = np.asarray(ln.get_xdata(), float), np.asarray(
                ln.get_ydata(), float)
            if xs.size == 0:
                continue        # a proxy handle built for the legend only
            ok = np.isfinite(xs) & np.isfinite(ys)
            if not ok.any():
                continue
            xs, ys = xs[ok], ys[ok]
            # A sample sitting ON the limit up to 1e-9 relative is drawn by
            # matplotlib, not clipped: t[-1]/t_end lands at 1 + O(ulp) on some
            sx = 1e-9 * max(abs(x0), abs(x1), 1.0)
            sy = 1e-9 * max(abs(y0), abs(y1), 1.0)
            out = ((xs < x0 - sx) | (xs > x1 + sx) | (ys < y0 - sy) | (ys > y1 + sy))
            if out.any():
                i = int(np.argmax(out))
                bad.append(f"{stem}: point ({xs[i]:.4g}, {ys[i]:.6g}) of "
                           f"{ln.get_label()!r} is outside the view "
                           f"x[{x0:.4g},{x1:.4g}] y[{y0:.6g},{y1:.6g}] "
                           f"({int(out.sum())} of {xs.size} clipped)")
    if bad:
        raise SystemExit("clipped data:\n " + "\n ".join(bad))

# (label, color, marker, linestyle) — the two ladder geometries get the
# paper's C0/C3.  SHARED with make_figures.py's fig2/fig3, which import these
# tuples so the series convention cannot fork between the main figures and
# this appendix one.
G1D_STYLE = ("PN 1d", "C0", "o", "-")
G2D_STYLE = ("PN 2d", "C3", "s", "--")
FLOOR_PCT = 0.3

def ladder_arms():
    """{(L, geo): RunRef} resolved from data/ladders.yaml ."""
    import yaml
    lad = yaml.safe_load((REPO / "data" / "ladders.yaml").read_text())
    source = lad["ladder_source"]
    comb = lad["combs"][source]
    found = {}
    for geo in ("1d", "2d"):
        for Lk, rel in lad["ladders"][source][geo].items():
            L, folder = float(Lk), REPO / rel
            hits = sorted(folder.glob("*_results.npz"))
            if len(hits) != 1:
                raise SystemExit(f"{folder}: expected exactly one "
                                 f"*_results.npz, got {len(hits)} -- a pin "
                                 f"that does not resolve uniquely pins nothing")
            found[(L, geo)] = ce.RunRef(L, geo, folder.parent.name,
                                        folder.name, hits[0])
    print(f"ladder source: {source!r} ({comb} comb), "
          f"{len(found)} arms from ladders.yaml")
    return source, comb, found

def campaign_rows():
    """{(L, geo): battery-dict} for every ladder arm; gate-enforced."""
    source, comb, found = ladder_arms()
    rows = {}
    for key, ref in sorted(found.items()):
        d = ce.load_npz(ref.path)
        # Refuse a comb the declaration did not ask for (same guard as
        # make_figures.load_v5): a run predating the `mode_source` field is
        # legacy BY DEFINITION, and an absent field must never read as "fine".
        got = str(d.get("mode_source", "legacy"))
        if got != comb:
            raise SystemExit(
                f"{ref.subdir}/{ref.ts}: ladder {source!r} declares the "
                f"{comb!r} comb but this npz carries {got!r} .")
        g = gate(d)
        if not all(g.values()):
            raise SystemExit(f"GATE FAILED on {ref.subdir}/{ref.ts}: {g}")
        b = ce._battery(d)
        # one before anything plots or annotates it.  The battery's own bit-
        # exact gate above still checks the artefact against what wrote it.
        b["gamma_ratio_published"] = b["gamma_ratio"]
        b["gamma_ratio"] = cell_d_gamma_ratio(d)
        gr = b["gamma_ratio"]
        lo, hi = ce.GAMMA_SANITY
        if not (lo <= gr <= hi):
            raise SystemExit(
                f"SANITY GATE: {ref.subdir}/{ref.ts} gamma_ratio={gr:.4f} "
                f"outside [{lo},{hi}] — a diverged/wrong run reached a pinned "
                f"slot; fix CANONICAL_RUNS, do not plot it.")
        b["_ref"], b["_npz"] = ref, d
        rows[key] = b
        print(f"  pinned-in: L={ref.L:g} {ref.geometry} -> "
              f"{ref.subdir}/{ref.ts}  (gamma {gr:.4f})")
    return rows

def _geo_series(rows, geo, key):
    Ls = sorted(L for (L, g) in rows if g == geo)
    return Ls, [rows[(L, geo)][key] for L in Ls]

def _decade_axis(ax):
    ax.set_xscale("log")
    ax.set_xlabel(r"grating period $L$ ($\mu$m)")
    ax.set_xticks(L_DECADES)
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.minorticks_off()

# ------------------------------------------------------------------- FIG-B
def fig_b(rows, outdir):
    # Two legend rows, not one: at 8.5 pt the four entries are 620 pt on one
    # row -- 232 pt wider than the page, which is exactly how the old tight
    # bbox reached 699 pt and demagnified this figure to 55.6%.
    LEG_H = 0.185
    fig, axes = plt.subplots(1, 3, figsize=(SLOT_IN, 2.80), sharey=True)
    bands = (("early", r"early ($0.5 \leq A_{\rm ref} < 0.9$)"),
             ("mid", r"mid ($0.1 \leq A_{\rm ref} < 0.5$)"),
             ("late", r"late ($0.01 \leq A_{\rm ref} < 0.1$)"))
    handles, labels = [], []
    for ax, (bandkey, title) in zip(axes, bands):
        _decade_axis(ax)
        bh = ax.axhspan(1 - FLOOR_PCT / 100, 1 + FLOOR_PCT / 100,
                        color="0.85", alpha=0.8, zorder=0)
        ax.axhline(1.0, color="k", lw=0.5, zorder=1)
        ax.set_title(title, pad=3.0)
        for geo, style in (("1d", G1D_STYLE), ("2d", G2D_STYLE)):
            lbl, c, m, ls = style
            Ls, r = _geo_series(rows, geo, f"rate_{bandkey}")
            if not Ls:
                continue
            h = ax.plot(Ls, r, color=c, ls=ls, lw=1.0, zorder=4)[0]
            if lbl not in labels:
                handles.append(h)
                labels.append(lbl)
            for L, v in zip(Ls, r):
                # looked up -- not in the artwork): hollow marker == late band
                # NOT band-robust there, so quote the terminal axis for that
                # point, never this rate.
                robust = (bandkey != "late"
                          or rows[(L, geo)]["rate_late_robust"])
                ax.plot(L, v, marker=m, color=c, ls="none", ms=3.6,
                        mfc=(c if robust else "white"), mec=c, zorder=5)
    # Ticks pinned, not auto: at the corrected canvas size the locator drops to
    # 0.8/1.0/1.2 and the reader loses the 10%-per-gridline reading the late
    # panel's spread has to be judged against.
    axes[0].set_yticks((0.8, 0.9, 1.0, 1.1, 1.2))
    # explicit ylim the autoscale puts the most extreme sample's marker CENTRE
    # on the limit, so half the glyph falls below the bottom spine -- and in
    # the late band that point is L = 1.5 um in 2D, i.e. the rung the paper
    # discloses as single-seed.  A clipped marker is invisible to `fit_check`
    # (see its docstring), which is why this is set rather than watched for.
    # The axes share y, so one call moves all three.
    axes[0].margins(y=0.10)
    axes[0].set_ylabel(r"$-\,d\ln A/dt$  ours / DOM reference")
    hollow = plt.Line2D([], [], marker="o", color="0.3", ls="none",
                        mfc="white", ms=3.6)
    handles.extend([hollow, axes[0].patches[0]])
    # A LEGEND DECODES A GLYPH; THE CAPTION CARRIES THE INSTRUCTION.  This entry
    # used to read "late band not robust: quote the terminal axis" -- 43
    # characters that are a verbatim duplicate of the caption, which already
    # says both "Late points that are not band-robust are drawn open" and "For
    # points marked open it is the terminal axis, not the late rate, that we
    # quote" (main.tex, fig:repin_bands).  At the corrected 10.5 pt legend base
    # that one string ran the legend 42 pt off the page.  Shortening it costs
    # ZERO information and buys the whole type-size repair.
    labels.extend(["open: not band-robust",
                   rf"$\pm${FLOOR_PCT}% DOM floor"])
    fig.tight_layout(rect=(0, 0, 1, 1 - LEG_H), pad=0.35, w_pad=0.6)
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), framealpha=0.9, columnspacing=1.3,
               handlelength=2.4, borderaxespad=0.15)
    fit_check(fig, "figB_repin_banded_rates")
    data_check(fig, "figB_repin_banded_rates")
    legend_check(fig, "figB_repin_banded_rates")
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"figB_repin_banded_rates.{ext}")
    plt.close(fig)
    print("figB_repin_banded_rates written")

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=FIG_DIR,
                    help="output directory (default: figures; use a "
                    "scratch dir for drafts while the campaign is in flight)")
    args = ap.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = campaign_rows()
    fig_b(rows, args.outdir)
    print(f"figures -> {args.outdir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
