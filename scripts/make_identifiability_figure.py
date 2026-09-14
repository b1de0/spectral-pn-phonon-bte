"""Plot the separately scoped nonlinear DC/q/2q objective diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "data/s3_caveats.json"
ARTEFACT = REPO / "data/p101_identifiability_figure.json"
FIGDIR = REPO / "figures"
STEM = "fig_identifiability"

#: Stored budget-key name. Diagonal/free/POD obey this cap at the fixed
#: pooled weight; the source pools ODE with rho=0, so its forced trajectory is
CAP = "rho<=rho_ref"

#: The four compensating classes the campaign screened, in artefact order.
CLASSES = ("diag", "free", "pod", "ode")

GRID_STYLE = {
    "G_logmix201": dict(color="#1f6f8b", marker="o", label="log-mix grid"),
    "G_hybrid324": dict(color="#b5442a", marker="s", label="hybrid grid"),
}

def load() -> dict:
    return json.loads(SOURCE.read_text())

def weight_axis(doc: dict) -> dict:
    """Panel (a): the DEEPEST measured direction at each macro weight.

    max over the four classes -- the binding direction, because the claim is
    that no weight brings EVERY measured direction under the bar.  min would
    answer a question nobody asked.
    """
    out = {}
    for grid, gv in doc["weight_axis"]["grids"].items():
        scan = gv["caps"][CAP]["scan"]
        rows = []
        for w in sorted(scan, key=float):
            per = scan[w]
            worst = max(CLASSES, key=lambda c: per[c])
            rows.append({"label": w, "weight": float(w),
                         "D_worst": float(per[worst]), "worst_class": worst,
                         "per_class": {c: float(per[c]) for c in CLASSES}})
        out[grid] = rows
    return out

def binding_axis(doc: dict) -> list:
    """THE QUANTITY PANEL (a) IS ABOUT: max over four classes AND two grids."""
    joint = doc["weight_axis"]["joint"][CAP]
    per_grid = weight_axis(doc)
    pooled = {}
    for rows in per_grid.values():
        for r in rows:
            pooled[r["weight"]] = max(pooled.get(r["weight"], 0.0),
                                      r["D_worst"])
    w_derived = float(joint["window_argmin"])
    pooled[w_derived] = max(pooled.get(w_derived, 0.0),
                            max(joint["per_entry_at_min"].values()))
    return [{"weight": w, "D_worst": pooled[w], "derived": w == w_derived}
            for w in sorted(pooled)]

def bandwidth_axis(doc: dict) -> dict:
    """Free-class depth versus bandwidth at the saved fixed pooled weight.

    This is the smallest weight present in this artefact's free ladder, not
    the numerical minimizing weight displayed in panel (a).
    """
    out = {}
    for grid in GRID_STYLE:
        rows = [r for r in doc["free_ladder"]["rows"]
                if r["grid"] == grid and not r["tag"].endswith("_seedflip")]
        w_derived = min(r["weight"] for r in rows)
        rows = sorted((r for r in rows if r["weight"] == w_derived),
                      key=lambda r: r["lam_max"])
        out[grid] = {
            "weight": w_derived,
            "rows": [{"label": f"{r['lam_max']:g}", "lam_max": r["lam_max"],
                      "D": r["D_at_rho_ref"], "D_unbounded": r["D_unbounded"],
                      "n_coef": r["n_coef"]} for r in rows],
        }
    return out

def crossing(rows: list, bar: float) -> tuple:
    """The bracket (lam_below, lam_above) in which D crosses the bar."""
    below = [r for r in rows if r["D"] < bar]
    above = [r for r in rows if r["D"] >= bar]
    return (max(r["lam_max"] for r in below) if below else None,
            min(r["lam_max"] for r in above) if above else None)

def build(doc: dict) -> dict:
    bar = float(doc["weight_axis"]["bar"])
    wa = weight_axis(doc)
    ba = bandwidth_axis(doc)
    joint = doc["weight_axis"]["joint"][CAP]
    lam_phys = {g: v["lam_phys"][0]
                for g, v in doc["projection"]["grids"].items()}

    br = {g: crossing(v["rows"], bar) for g, v in ba.items()}
    return {
        "generated_by": "scripts/p101_identifiability_figure.py --write",
        "plan": "",
        "source": str(SOURCE.relative_to(REPO)),
        "convention": CAP,
        "convention_note": "single-convention by construction; the prose's "
                           "equal-weighting 32.6/33.5 are measured under a "
                           "different cap and are NOT drawn here",
        "bar": [{"label": "bar", "value": bar}],
        "floor": [{"label": "minimax_D", "value": float(joint["window_min"])},
                  {"label": "argmin_weight", "value": float(joint["window_argmin"])},
                  {"label": "margin_over_bar", "value": float(joint["margin_over_bar"])}],
        "lam_phys": [{"label": g, "value": v} for g, v in sorted(lam_phys.items())],
        # Panel (a) prints this as the off-scale note, so it is a bindable
        # section rather than a number the plotting code reaches into.
        "offscale": [{"label": "D_max_over_weights",
                      "value": max(r["D_worst"] for rr in wa.values()
                                   for r in rr)}],
        "crossing": [{"label": g, "lam_below": br[g][0], "lam_above": br[g][1]}
                     for g in sorted(br)],
        # The two grids' own depths AT the joint minimum. Panel (a) draws one
        # pooled curve, so the discretisation-robustness axis that the old
        # two-curve drawing carried has to be reported as a number instead of
        # being dropped: this is that number, and the caption quotes it.
        "grid_spread_at_floor": [
            {"label": g, "value": max(v for k, v in
                                      joint["per_entry_at_min"].items()
                                      if k.startswith(g))}
            for g in sorted(doc["weight_axis"]["grids"])
        ],
        "weight_ladder": wa,
        "binding_ladder": binding_axis(doc),
        "bandwidth_ladder": ba,
    }

def gates(d: dict) -> bool:
    """A TEST THAT CANNOT FAIL IS WORSE THAN NO TEST. Each was confirmed to fire when its target was perturbed (see the report)."""
    ok = True
    bar = d["bar"][0]["value"]
    floor = next(r["value"] for r in d["floor"] if r["label"] == "minimax_D")
    if floor <= bar:
        print(f"  GATE 1 FAIL: minimax D {floor} <= bar {bar}")
        ok = False
    for grid, rows in d["weight_ladder"].items():
        low = [r for r in rows if r["D_worst"] < bar]
        if low:
            print(f"  GATE 2 FAIL {grid}: {len(low)} weight(s) below the bar")
            ok = False
    drawn_min = min(r["D_worst"] for r in d["binding_ladder"])
    if abs(drawn_min - floor) > 1e-9:
        print(f"  GATE 4 FAIL: the drawn binding curve reaches {drawn_min!r}, "
              f"but the annotated floor is {floor!r}. A horizontal reference "
              f"must be attained by the series it labels.")
        ok = False
    for c in d["crossing"]:
        g = c["label"]
        lam = next(r["value"] for r in d["lam_phys"] if r["label"] == g)
        if c["lam_below"] is None or c["lam_above"] is None:
            print(f"  GATE 3 FAIL {g}: ladder does not cross the bar")
            ok = False
        elif not (c["lam_below"] <= lam <= c["lam_above"]):
            print(f"  GATE 3 FAIL {g}: slowest physical rate {lam:.2f} outside "
                  f"the crossing bracket ({c['lam_below']}, {c['lam_above']}]")
            ok = False
    print(f"gates: {'PASS' if ok else 'FAIL'}")
    return ok

#: Minimum white space, IN ON-PAGE POINTS, that a label must keep from any
#: rule or curve it is not itself anchored to.  Below ~1.5 pt a gap reads as a
#: touch once the page is printed rather than viewed at 400%.
MIN_CLEAR_PT = 1.6

def clearance(fig, checks) -> bool:
    """MEASURE THE COLLISIONS A Text-vs-canvas CHECK CANNOT SEE."""
    fig.canvas.draw()
    ppp = 72.0 / fig.dpi                       # display px -> on-page points
    ok = True
    for name, art, others in checks:
        for sub, bb in _ink(art):
            label = f"{name}{sub}"
            for oname, ob in others:
                if isinstance(ob, tuple) and ob and ob[0] == "inside":
                    box = ob[1].get_window_extent()
                    gap = -max(box.x0 - bb.x0, bb.x1 - box.x1,
                               box.y0 - bb.y0, bb.y1 - box.y1) * ppp
                elif hasattr(ob, "get_window_extent") or hasattr(ob, "x0"):
                    other = (ob.get_window_extent()
                             if hasattr(ob, "get_window_extent") else ob)
                    gap = _bbox_gap(bb, other) * ppp
                else:                          # a sampled polyline in display
                    gap = min((_pt_gap(bb, x, y) for x, y in ob),
                              default=1e9) * ppp
                hit = gap < MIN_CLEAR_PT
                flag = "  <-- COLLISION" if hit else ""
                print(f"  clearance {label:30s} vs {oname:18s} "
                      f"{gap:6.2f} pt{flag}")
                ok &= not hit
    return ok

def _ink(art):
    """THE INK, NOT THE BOX.  A legend's window extent is its padded patch, and
    with `frameon=False` that padding is invisible -- a curve crossing it is not
    a collision anyone can see, and reporting it trains the reader of this gate
    to ignore it.  What can collide is the legend's TEXT and its HANDLES, so a
    legend is decomposed into those.  An Annotation is reduced to its text for
    the mirror-image reason: its extent swallows the arrow, and the lam_phys
    arrow is MEANT to cross the bar rule."""
    from matplotlib.legend import Legend
    from matplotlib.text import Text
    if isinstance(art, Legend):
        parts = [(f" text[{i}]", t.get_window_extent())
                 for i, t in enumerate(art.get_texts())]
        parts += [(f" handle[{i}]", h.get_window_extent())
                  for i, h in enumerate(art.legend_handles)
                  if hasattr(h, "get_window_extent")]
        return parts
    if isinstance(art, Text):
        return [("", Text.get_window_extent(art))]
    return [("", art.get_window_extent())]

def _pt_gap(bb, x, y) -> float:
    """Display-unit distance from a point to a bbox; 0 inside."""
    dx = max(bb.x0 - x, x - bb.x1, 0.0)
    dy = max(bb.y0 - y, y - bb.y1, 0.0)
    return (dx * dx + dy * dy) ** 0.5

def _bbox_gap(a, b) -> float:
    """Display-unit separation of two bboxes; 0 if they overlap."""
    dx = max(b.x0 - a.x1, a.x0 - b.x1, 0.0)
    dy = max(b.y0 - a.y1, a.y0 - b.y1, 0.0)
    return (dx * dx + dy * dy) ** 0.5

def _rule_box(ax, yval):
    """The horizontal rule at data-y `yval`, as a 1-px-tall display bbox."""
    from matplotlib.transforms import Bbox
    x0, x1 = ax.get_xlim()
    (px0, py), (px1, _) = ax.transData.transform([(x0, yval), (x1, yval)])
    return Bbox.from_extents(px0, py - 0.5, px1, py + 0.5)

def _polyline(ax, xs, ys, n=600):
    """The drawn series, densely resampled in DISPLAY coordinates, so a label
    can be tested against the curve itself and not merely against its bbox.

    CLIPPED TO THE AXES.  Panel (a)'s series runs three decades above the main
    segment's top; unclipped, the interpolated chord from an off-segment point
    sweeps down through the whole panel and reports a collision with every
    label it passes -- including the legend, which it never actually reaches on
    the page.  Only what is drawn inside the frame can collide with anything.
    """
    import numpy as np
    box = ax.get_window_extent()
    pts = ax.transData.transform(list(zip(xs, ys)))
    out = []
    per = max(2, n // max(1, len(pts) - 1))
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        t = np.linspace(0.0, 1.0, per)
        out += list(zip(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return [(x, y) for x, y in out
            if box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1]

def draw(d: dict) -> None:
    bar = d["bar"][0]["value"]
    floor = next(r["value"] for r in d["floor"] if r["label"] == "minimax_D")
    margin = next(r["value"] for r in d["floor"] if r["label"] == "margin_over_bar")

    # ~0.75") was wrong twice: \textwidth is 388.53 pt, not 6.5 in, and the
    # tight bbox came out 615.8 pt, so the shipped downscale was 0.63 and the
    # 9 pt annotations printed at 5.7 pt -- under Elsevier's 7 pt floor.  Two
    # panels SIDE BY SIDE in 388 pt leave 180 pt each, which no legible
    # two-line title fits in, so the panels are now STACKED: the canvas is
    # authored at the slot width, the scale is 1.000, and every size below is
    # what the reader gets.
    SLOT_IN = 388.53 / 72.0
    plt.rcParams.update({
        # 10 pt bases put the mathtext sub/superscripts (0.7x) AT 7 pt.
        "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.5,
        # was still emitting Type 3 DejaVu subsets with uni=no -- a family
        # Elsevier does not permit and that pdftotext cannot read.  Same stack
        # and same rationale as scripts/make_schematic.py:189-214;
        # 'stix' NOT 'stixsans' (stixsans hides numerals from the G6 gate).
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "Arimo",
                            "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    # HEIGHT IS A HARD CONSTRAINT: \textheight = 546.5 pt and this float's own
    # caption is 213 pt, so the BOX may not exceed ~320 pt.  At 4.55 in LaTeX
    # reported "Float too large for page by 8.3 pt" and deferred BOTH this
    # figure and fig:traces behind it to the end of the document, adding a page.
    # 4.24 in = 305.3 pt.  The break segment wants height and the caption grew
    # to disclose it, so this was tried at 4.42 in and LaTeX answered "Float
    # too large for page by 22.9 pt" -- the ceiling recorded above is real and
    # the caption is part of it.  What paid for the break instead was moving
    # panel (a)'s legend OUT of the frame into the strip: the 11 pt it occupied
    # inside the axes is worth more than the 10 pt the taller canvas bought.
    # 4.30 in still overflowed by 2.2 pt; this leaves ~3.5 pt of slack, which a
    # figure whose caption is 60% of the float needs to keep.
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(SLOT_IN, 4.24))

    # ---- (a) all scanned weights retain a compensating direction --------------------------
    #  THE MAIN SEGMENT IS TIGHT ON PURPOSE: the quantity this panel is ABOUT
    #  is the gap between the valley floor and the bar (a factor 1.54), and on
    #  the full three-and-a-half-decade range of D it is a hairline.
    #
    #  WHAT THIS USED TO DO, AND WHY IT WAS WRONG.  It set ylim to (1.45, 22)
    #  and stopped there.  The two smallest weights carry D = 139.8 and
    #  D = 4387.8, so the curve LEFT THE TOP OF THE AXES and those two points
    #  were simply not on the page -- with no break mark, no arrow, nothing on
    #  the figure to say so.  The only disclosure was a sentence in the
    #  caption, which is not where a reader looks to find out whether a line
    #  that runs off the top of a frame is data or an artefact.  A clip that is
    #  invisible on the figure is a clip that misreports the series.
    #  The range is now BROKEN, not clipped: the off-scale points are drawn, in
    #  their own segment, above a marked break.  Nothing is dropped and the
    #  floor-to-bar gap keeps a segment sized for it.
    # WHERE THE BREAK GOES, anchored to the physics of the panel rather than to
    # an absolute number, so it survives a retrain: the main segment must show
    # the bar, the floor, and every point within MAIN_SPAN of the floor.  The
    # rest is what the break segment is for.
    MAIN_SPAN = 5.5
    ytop = floor * MAIN_SPAN
    # The band under the bar is where the bar's own label goes, so it is sized
    # for a 9 pt label rather than left as the sliver it was.  The band BETWEEN
    # the rules then carries nothing, which is the point: stacking both labels
    # in it is what made the floor rule print tangent to the bar label.
    ybot = bar / 3.2
    rows = d["binding_ladder"]
    wpos = [r["weight"] for r in rows if r["weight"] > 0]
    xlo = min(wpos) / 4.0
    # The w = 0 entry has no place on a log abscissa.  It is parked one third
    # of a decade left of the smallest positive weight AND given its own "0"
    # tick with a break mark on the spine (below), so the reader is told that
    # the leftmost column is a limit point and not 3.3e-5.
    zero_pos = min(wpos) / 3.0
    has_zero = any(r["weight"] <= 0 for r in rows)
    pos = [(r["weight"] if r["weight"] > 0 else zero_pos) for r in rows]
    y = [r["D_worst"] for r in rows]
    # DATA-DRIVEN, so this regenerates cleanly when the numbers move: if a
    # future artefact keeps every point under `ytop` there is no break segment
    # at all and the panel is a plain single axes again.
    hi = [v for v in y if v > ytop]
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(xlo, 1.6)
    axA.set_ylim(ybot, ytop)
    axA.fill_between([xlo, 1.6], ybot, bar, color="#2a9d8f", alpha=0.13,
                     zorder=0)
    axA.plot(pos, y, color="#1d3557", marker="o", lw=1.15, ms=2.9, mfc="white",
             zorder=3,
             # ONE LINE.  Two lines of legend hang 11 pt lower, and with the
             # main segment now ending at 5.5x the floor the curve's right tail
             # rises to meet them -- the gate measures the second line sitting
             # ON the tail. "deepest ... direction" is already the y-label; what
             # the legend has to add is the SET the max is taken over.
             label="max: budgeted classes + ODE; 2 grids")
    # The saved numerical minimum, marked, so the dotted line is ATTAINED by
    # the curve rather than floating under it.  Before this the drawn curve
    # went 1.8% below the line labelled "valley floor" (see `binding_axis`).
    jm = min(range(len(rows)), key=lambda i: y[i])
    axA.plot([pos[jm]], [y[jm]], marker="o", ms=6.0, mfc="none", mew=1.0,
             color="#1d3557", zorder=4)
    axA.axhline(bar, color="#c1121f", lw=1.1, ls="--", zorder=2)
    axA.axhline(floor, color="#444", lw=0.7, ls=":", zorder=2)
    axA.set_xlabel("macro (energy-balance) block weight")
    # Four short lines, not two long ones: rotated, this label's LENGTH is
    # vertical, and two lines of it are 166 pt against a 100 pt panel.
    # y is set below, once it is known whether the label spans one segment or
    # the broken pair.
    axA.set_ylabel("depth $D$ of the\ndeepest measured\ncompensating\ndirection",
                   linespacing=1.15)
    if has_zero:
        # An explicit "0" tick: the leftmost point is the vanishing-weight
        # limit, and unlabelled it reads as a weight of 3e-5.
        decades = [10.0 ** e for e in range(-4, 1)]
        axA.set_xticks([zero_pos] + decades)
        axA.set_xticklabels(["0"] + [f"$10^{{{e}}}$" for e in range(-4, 1)])
        axA.set_xticks([], minor=True)
    if not hi:
        axA.set_title("(a) all scanned weights leave a direction above the bar", fontsize=10,
                      pad=3.0)
    # Only line-ANCHORED labels stay on the panel; the off-scale range and the
    # meaning of the shaded band are stated in the caption instead.  Two
    # earlier drafts put them here and both collided with the curve once the
    # fonts were sized for print.
    from matplotlib.transforms import offset_copy
    up = offset_copy(axA.get_yaxis_transform(), fig=fig, y=1.5, units="points")
    down = offset_copy(axA.get_yaxis_transform(), fig=fig, y=-2.5,
                       units="points")
    tA_floor = axA.text(0.025, floor, f"scanned minimum  $D={floor:.2f}$",
                        color="#222", fontsize=9, ha="left", va="bottom",
                        transform=up)
    # BELOW its own rule, not above it.  Stacked above, this label and the
    # floor label share the one band between the two rules -- 6.4 pt of it once
    # the break segment was carved out -- and the floor rule printed tangent to
    # this label's ascenders.  The shaded band under the bar is empty by
    # construction and is now sized to hold it.  The 2.5 pt offset matters:
    # anchored va="top" exactly AT the rule, the rule prints through the
    # ascenders, which is the same defect one rule lower down.
    tA_bar = axA.text(0.025, bar, f"target: D ≤ {bar:g}",
                      color="#c1121f", fontsize=9, ha="left", va="top",
                      transform=down)
    axA.grid(alpha=0.22, lw=0.32)
    # The legend is attached BELOW, to the break strip when there is one.  Panel
    # (a) is 51 pt tall and already spends ~22 pt of it on the two rule labels;
    # a legend inside the frame has to go top-right, and right-anchored it still
    # reaches back to x = 0.43 where the curve is descending steeply -- the gate
    # measures its handle sitting on the descending limb.  The strip above the
    # break is empty at every x beyond the off-scale points, permanently, and
    # that is where it goes.
    legA = None

    # ---- (b) the compensating class is not a strawman --------------------
    curvesB = {}
    for grid, blk in d["bandwidth_ladder"].items():
        st = GRID_STYLE[grid]
        lam = [r["lam_max"] for r in blk["rows"]]
        dep = [r["D"] for r in blk["rows"]]
        curvesB[grid] = (lam, dep)
        axB.plot(lam, dep, lw=1.15, ms=2.9, mfc="white", **st)
    axB.axhline(bar, color="#c1121f", lw=1.1, ls="--", zorder=2)
    # The value the prose quotes is the LOG-MIX grid's; the hybrid grid's is
    # 0.06% away, indistinguishable at this scale.  Draw and label the quoted
    # one, and carry both in the artefact.
    lam_phys = next(r["value"] for r in d["lam_phys"]
                    if r["label"] == "G_logmix201")
    axB.axvline(lam_phys, color="#5a189a", lw=0.95, ls="-.", zorder=2)
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlim(8, 4e6)
    # HEADROOM AT THE BOTTOM, not decoration: the two labels on this panel both
    # have to live UNDER the rising curve, and on the autoscaled range the only
    # band under it that is wide enough runs straight through the bar rule.
    # Dropping the floor to 0.75 opens a clear strip beneath the bar without
    # touching a single plotted value.
    axB.set_ylim(0.5, 13.0)
    axB.set_xlabel("bandwidth cap $\\lambda_{\\max}$ of the class\n"
                   "(units of the collective rate $\\gamma$)")
    axB.set_ylabel("depth $D$")
    # NOT "exactly where": the ladder brackets the crossing, it does not
    # resolve it, and the physical rate sits inside the bracket.  The panel
    # must not claim a coincidence tighter than the six points support.
    axB.set_title("(b) compensation grows with the trial-class bandwidth", fontsize=10,
                  pad=3.0, linespacing=1.15)
    # THE BAND ABOVE THE BAR IS NOT CLEAR, whatever the note that used to stand
    # here said.  Anchored at y = 0.300 this two-line block ran from x = 0.55 to
    # the right spine, and the curve passes through exactly that rectangle: the
    # clearance gate finds 106 sampled points of the log-mix curve INSIDE the
    # text.  It printed across the rising limb, and no Text-vs-canvas check can
    # see that, which is why it survived several passes that all reported the
    # figure as fitting.  The block moves to the strip BELOW the bar rule --
    # empty at every x, and emptier still on the right, where the curve is four
    # to twelve times the bar -- and the arrow reaches back up to the rule.
    tB_lam = axB.annotate(f"slowest relaxation rate\n"
                 f"$\\lambda={lam_phys:.1f}$",
                 xy=(lam_phys, 0.36), xycoords=axB.get_xaxis_transform(),
                 xytext=(0.985, 0.035), textcoords=axB.transAxes,
                 fontsize=9, color="#5a189a", ha="right", va="bottom",
                 arrowprops=dict(arrowstyle="->", color="#5a189a", lw=1.0,
                                 shrinkA=5, shrinkB=4,
                                 connectionstyle="arc3,rad=0.0"))
    # LEFT end, above the rule -- and WITHOUT the value.  Carrying "D = 1.874"
    # here made the label long enough that its right end reached the curve at
    # the crossing (the gate measures the graze), and the value is redundant
    # three times over: panel (a) labels the identical rule with it, the caption
    # states it, and the two panels share one bar.  The rule needs its name on
    # this panel, not its number.
    tB_bar = axB.text(0.02, bar, "target bar",
                      color="#c1121f", fontsize=9, ha="left", va="bottom",
                      transform=axB.get_yaxis_transform())
    axB.grid(alpha=0.22, lw=0.32)
    legB = axB.legend(fontsize=9, frameon=False, loc="upper left")

    # Scope and the ODE cap exception travel with the artwork. Panel (b)
    # alone uses the capped free class at the fixed pooled weight.
    fig.suptitle("Nonlinear DC + q + 2q diagnostic: local τ(T), L = 100 μm\n"
                 "Slot-budgeted classes + uncapped ODE comparator",
                 fontsize=10, y=0.998, color="#444", linespacing=1.15)
    fig.tight_layout(rect=(0, 0, 1, 0.922), h_pad=1.4)

    # ---- (a) the marked break ------------------------------------------
    # Built AFTER tight_layout and by repositioning, not by a nested gridspec:
    # the layout engine has already reserved the room for panel (a)'s title,
    # ylabel and ticks, and the break segment is carved out of the space it
    # gave the axes.  Every other element on the figure therefore lands where
    # it landed before, which is what makes this diff provable against the
    # shipped PDF outside panel (a).
    if hi:
        FRAC, GAP = 0.22, 0.05             # of panel (a)'s own height
        box = axA.get_position()
        axA.set_position((box.x0, box.y0, box.width,
                          box.height * (1.0 - FRAC - GAP)))
        axA_hi = fig.add_axes((box.x0, box.y0 + box.height * (1.0 - FRAC),
                               box.width, box.height * FRAC))
        axA_hi.set_xscale("log")
        axA_hi.set_yscale("log")
        axA_hi.set_xlim(xlo, 1.6)
        # Padded generously on both sides: the padding is what keeps the two
        # decade labels INTERIOR to a ~17 pt strip instead of jammed against
        # its edges, and it makes the segment read as a range rather than as a
        # box drawn round two points.
        axA_hi.set_ylim(min(hi) / 4.0, max(hi) * 4.0)
        axA_hi.plot(pos, y, color="#1d3557", marker="o", lw=1.15, ms=2.9,
                    mfc="white", zorder=3)
        axA_hi.grid(alpha=0.22, lw=0.32)
        axA_hi.tick_params(labelbottom=False, bottom=False, which="both")
        # ONE LABEL PER VISIBLE DECADE IS TOO MANY.  The strip is ~17 pt tall;
        # three 10 pt decade labels in it overprint each other (they did).
        # Label alternate decades of the covered range, which for this data is
        # 10^2 and 10^4 -- and if a retrain widens the range, this thins
        # further instead of crowding.
        import math
        e0 = math.ceil(math.log10(min(hi) / 4.0))
        e1 = math.floor(math.log10(max(hi) * 4.0))
        exps = list(range(e0, e1 + 1))[::2]
        axA_hi.set_yticks([10.0 ** e for e in exps])
        axA_hi.set_yticklabels([f"$10^{{{e}}}$" for e in exps])
        axA_hi.set_yticks([], minor=True)
        axA.spines["top"].set_visible(False)
        axA_hi.spines["bottom"].set_visible(False)
        # The break itself.  Diagonal ticks on BOTH facing spines, drawn in
        # axes coordinates so they sit on the spine ends whatever the data do.
        brk = dict(marker=[(-1, -0.62), (1, 0.62)], markersize=5.5,
                   linestyle="none", color="#222", mec="#222", mew=0.8,
                   clip_on=False)
        axA_hi.plot([0, 1], [0, 0], transform=axA_hi.transAxes, **brk)
        axA.plot([0, 1], [1, 1], transform=axA.transAxes, **brk)
        axA_hi.set_title("(a) all scanned weights leave a direction above the bar", fontsize=10,
                         pad=3.0)
        legA = axA_hi.legend(*axA.get_legend_handles_labels(), fontsize=9,
                             frameon=False, loc="upper right",
                             borderaxespad=0.15, handlelength=1.6,
                             borderpad=0.1)
        # The ylabel belongs to the PAIR, not to the lower segment: left at
        # 0.5 it centres on the main segment and reads as labelling only it.
        # (Position along the spine only -- the horizontal offset tight_layout
        # computed from the tick-label widths is untouched.)
        lo, hi_box = axA.get_position(), axA_hi.get_position()
        axA.yaxis.label.set_y(
            ((lo.y0 + hi_box.y1) / 2.0 - lo.y0) / lo.height)
    if legA is None:                       # no break strip: fall back inside
        legA = axA.legend(fontsize=9, frameon=False, loc="upper right")
    if has_zero:
        # The abscissa is broken too: "0" is not 3.3e-5, and the gap between
        # that tick and 10^-4 is where the break belongs.
        xb = (zero_pos * 1e-4) ** 0.5
        axA.plot([xb, xb], [0, 0], transform=axA.get_xaxis_transform(),
                 marker=[(-0.62, -1), (0.62, 1)], markersize=5.5,
                 linestyle="none", color="#222", mec="#222", mew=0.8,
                 clip_on=False, zorder=6)

    # ---- the collision gate ---------------------------------------------
    if not clearance(fig, [
        ("(a) valley-floor label", tA_floor,
         [("bar rule", _rule_box(axA, bar)),
          ("binding curve", _polyline(axA, pos, y)),
          ("inside axes", ("inside", axA))]),
        ("(a) falsification label", tA_bar,
         [("floor rule", _rule_box(axA, floor)),
          ("binding curve", _polyline(axA, pos, y)),
          ("inside axes", ("inside", axA))]),
        ("(a) legend", legA,
         [("binding curve", _polyline(axA, pos, y)),
          ("strip curve", _polyline(axA_hi, pos, y) if hi else [])]),
        ("(b) falsification label", tB_bar,
         [("inside axes", ("inside", axB)),
          ("log-mix curve", _polyline(axB, *curvesB["G_logmix201"])),
          ("hybrid curve", _polyline(axB, *curvesB["G_hybrid324"]))]),
        ("(b) lam_phys label", tB_lam,
         [("inside axes", ("inside", axB)),
          ("bar rule", _rule_box(axB, bar)),
          ("log-mix curve", _polyline(axB, *curvesB["G_logmix201"])),
          ("hybrid curve", _polyline(axB, *curvesB["G_hybrid324"]))]),
        ("(b) legend", legB,
         [("log-mix curve", _polyline(axB, *curvesB["G_logmix201"])),
          ("hybrid curve", _polyline(axB, *curvesB["G_hybrid324"]))]),
    ]):
        raise SystemExit("clearance gate FAILED -- figure not written")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        # NOT bbox_inches="tight": the saved width must be figsize[0]*72 so
        # the on-page scale is exactly 1 and the sizes above are the print
        # sizes.  A tight bbox is how this figure grew to 615.8 pt.
        fig.savefig(FIGDIR / f"{STEM}.{ext}", dpi=200)
    plt.close(fig)
    print(f"[DREW] figures/{STEM}.{{pdf,png}}")

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw from the retained figure artefact without rewriting it")
    args = ap.parse_args(argv)

    if args.plot_only:
        d = json.loads(ARTEFACT.read_text())
        if not gates(d):
            return 1
        draw(d)
        return 0
    d = build(load())
    bar = d["bar"][0]["value"]
    print(f"convention {CAP};  bar {bar}")
    for grid, rows in d["weight_ladder"].items():
        print(f"  {grid}: worst-direction D over {len(rows)} weights = "
              f"{min(r['D_worst'] for r in rows):.4g} .. "
              f"{max(r['D_worst'] for r in rows):.4g}")
    for c in d["crossing"]:
        print(f"  {c['label']}: crosses the bar in "
              f"({c['lam_below']:g}, {c['lam_above']:g}]")
    if not gates(d):
        return 1
    if args.write:
        ARTEFACT.write_text(json.dumps(d, indent=1) + "\n")
        print(f"[WROTE] {ARTEFACT.relative_to(REPO)}")
        draw(d)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
