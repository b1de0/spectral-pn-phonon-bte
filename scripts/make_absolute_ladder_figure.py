"""fig_absolute_ladder: PN's ABSOLUTE observable, three panels."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SLOT_PT = 388.53                 # elsarticle[preprint,12pt] \textwidth, in bp
SLOT_IN = SLOT_PT / 72.0         # 5.3963 in = 137.1 mm

#: Elsevier artwork floors, measured off an Elsevier-TYPESET JCP paper
#: (Xiao & Frank, RelaxNet, J. Comput. Phys. 490 (2023) 112317):
#: 7 pt for normal text on the page, 6 pt for sub/superscripts.
FLOOR_TEXT_PT = 7.0
SHRINK = 0.7                     # matplotlib mathtext script shrink per level

#: BASE size for every string that contains mathtext.  mathtext sets a
#: sub/superscript at 0.7x the base, so a 7 pt ON-PAGE floor forces base >= 10
#: for anything carrying one -- and in this figure that is nearly every label:
#: kappa_eff, kappa_bulk, D_eff, N_k, the [W m^-1 K^-1] units and every decade
#: tick.  At the old 7.5-8.5 pt bases those subscripts printed at 5.25-5.95 pt.
#:
#: 10.0 pt is the BARE MINIMUM and gives EXACTLY 7.00 on the page -- ZERO
#: later shrink (a wider legend, a demoted \includegraphics width) broke the
#: floor on arrival with nothing to absorb it.  10.5 puts subscripts at 7.35,
#: i.e. 0.35 pt of margin, and matches the venue evidence: RelaxNet sets its
#: figure lettering ~25% larger than its 7.97 pt body text.
#: The figure is authored at exactly SLOT_IN so the on-page scale is 1.000 and
#: these numbers are what the reader measures with a ruler.
MATH_PT = 10.5
#: Plain words carry no script, so they only have to clear 7 pt themselves --
#: but nothing in this figure goes below 9, and 9.5 keeps the plain labels
#: visually of a piece with the 10.5 pt math.
PLAIN_PT = 9.5

# ---------------------------------------------------------------------------
# PN FIGURE STYLE v2.  Elsevier permits Arial/Helvetica, Times, Courier and
# Symbol; DejaVu (matplotlib's default, and what this figure shipped with) is
# on none of those lists AND embeds as Type 3 with a Custom encoding and
# uni=no, so the artwork was not text-extractable at all -- which quietly puts
# that only held because this figure prints few numerals.  pdf.fonttype=42
# fixes both.
#
# mathtext.fontset is "stix", NOT "stixsans": stixsans draws sans math from
# STIXNonUnicode, which has no usable Unicode map, so pdftotext returns every
# mathtext glyph blank (make_schematic.py:196-213 has the measurement).  Serif
# math beside Helvetica words is also the measured JCP idiom.
DPI = 300
plt.rcParams.update({
    "font.size": 9,                       # fallback ONLY; never relied on
    "axes.titlesize": MATH_PT,
    "axes.labelsize": MATH_PT,
    "xtick.labelsize": MATH_PT,
    "ytick.labelsize": MATH_PT,
    "legend.fontsize": MATH_PT,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "standard",           # NEVER "tight": see verify_page()
    "axes.linewidth": 0.45,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "Arimo",
                        "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.3,
})

ARTEFACT = REPO / "data" / "pn_absolute_ladder.json"
FIG_DIR = REPO / "figures"

# Same families as figA: C0/C3 for the two P_N geometries, and a heavy grey for
# the deterministic arbiter so the reference reads as background, not a rival.
# Grayscale-safe: every series differs in colour AND marker AND linestyle.
PN1D = dict(color="C0", marker="o", ls="-", ms=3.9, lw=1.2,
            label=r"$\mathrm{P}_N$ 1D (data-free)")
PN2D = dict(color="C3", marker="s", ls="--", ms=3.6, lw=1.2,
            label=r"$\mathrm{P}_N$ 2D (data-free)")
DOM = dict(color="0.35", marker="", ls="-", lw=2.4, alpha=0.55,
           label="DOM reference", zorder=1)

def fit_check(fig, stem):
    """Every text artist must lie INSIDE the canvas, which is the print slot."""
    fig.canvas.draw()
    W, H = (fig.get_size_inches() * fig.dpi)
    r = fig.canvas.get_renderer()
    # Ticks OUTSIDE the view are artists matplotlib keeps and never draws; a
    # naive scan flags them and hides the real defect underneath the noise.
    dead = set()
    for a in fig.axes:
        for axis, lim in ((a.xaxis, a.get_xlim()), (a.yaxis, a.get_ylim())):
            lo, hi = sorted(lim)
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tk.get_loc() <= hi + 1e-9:
                    dead.update((id(tk.label1), id(tk.label2)))
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

def _script_depth(s: str) -> int:
    """Deepest mathtext sub/superscript NESTING level in `s`.

    Only `_`/`^` inside a `$...$` span count, and only when they are followed
    by a braced group or a single token: `\\mathrm{eff}` is not a script, but
    `_\\mathrm{eff}` is.  Depth is what sets the on-page size, because
    matplotlib shrinks by 0.7 PER LEVEL (0.49 at depth 2).
    """
    depth = mx = 0
    inmath = False
    stack = []          # brace nesting depths at which a script opened
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":                       # skip an escaped char / macro name
            i += 2
            while i < len(s) and s[i].isalpha():
                i += 1
            continue
        if c == "$":
            inmath = not inmath
            if not inmath:
                depth, stack = 0, []
        elif inmath and c in "_^":
            j = i + 1
            while j < len(s) and s[j] == "\\":      # `_\mathrm{eff}`
                j += 1
                while j < len(s) and s[j].isalpha():
                    j += 1
            if j < len(s) and s[j] == "{":
                stack.append(len(stack))
                depth += 1
                mx = max(mx, depth)
                i = j                        # the '{' is consumed below
            else:
                mx = max(mx, depth + 1)      # single-token script, closes at once
        elif inmath and c == "}" and stack:
            stack.pop()
            depth -= 1
        i += 1
    return mx

def check_type_floor(fig, stem):
    """THE ONLY SOURCE-SIDE CHECK THAT IS NOT A FALSE GREEN."""
    bad = []
    for t in fig.findobj(plt.Text):
        s = str(t.get_text())
        if not t.get_visible() or not s.strip():
            continue
        base = t.get_fontsize()
        d = _script_depth(s)
        if d >= 2:
            bad.append(f"{s[:34]!r}: mathtext nesting depth {d} "
                       f"-> {base * SHRINK ** d:.2f} pt, refused outright")
            continue
        on_page = base * SHRINK ** d
        if on_page < FLOOR_TEXT_PT - 1e-9:
            bad.append(f"{s[:34]!r}: base {base:.2f} pt, script depth {d} "
                       f"-> {on_page:.2f} pt on page (floor "
                       f"{FLOOR_TEXT_PT:.2f})")
    if bad:
        raise SystemExit(f"{stem}: type under the Elsevier floor:\n "
                         + "\n ".join(bad))

def check_data_visible(fig, stem):
    """No plotted POINT may fall outside its axes' view limits.

    fit_check is blind to this by construction -- it iterates Text artists, and
    a marker half a glyph outside the spine is a drawing.  This figure shipped
    with the Nk=800 rung of both curves in panel (c) drawn off the right edge
    for exactly that reason.  Axis anchors (axhline/axhspan) are excluded:
    they are furniture spanning the view on purpose, and matplotlib gives them
    a BLENDED transform (axes-fraction in x) rather than transData -- which is
    how they are told apart here.
    """
    bad = []
    for ax in fig.axes:
        (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
        for ln in ax.get_lines():
            if ln.get_transform() is not ax.transData:
                continue                     # axhline & friends
            xs, ys = np.asarray(ln.get_xdata(), float), \
                np.asarray(ln.get_ydata(), float)
            if xs.size == 0 or xs.size != ys.size:
                continue
            out = ((xs < x0) | (xs > x1) | (ys < y0) | (ys > y1))
            if out.any():
                i = np.flatnonzero(out)
                bad.append(f"{ln.get_label()!r}: {out.sum()} of {xs.size} "
                           f"points outside the view, e.g. "
                           f"({xs[i[0]]:.4g}, {ys[i[0]]:.4g}) vs "
                           f"x[{x0:.4g},{x1:.4g}] y[{y0:.4g},{y1:.4g}]")
    if bad:
        raise SystemExit(f"{stem}: data clipped by the axes limits:\n "
                         + "\n ".join(bad))

def check_text_inside_axes(fig, stem, names, pad_pt=1.0):
    """In-axes annotations must sit inside their OWN axes, not just the canvas.

    `names` is the set of literal strings to police -- the anchor labels and
    the panel tags.  fit_check passes them as long as they are on the page,
    so a label riding over the top spine is invisible to it.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = pad_pt * fig.dpi / 72.0
    bad = []
    for ax in fig.axes:
        ab = ax.get_window_extent(r)
        for t in ax.texts:
            s = str(t.get_text())
            if s.split("\n")[0] not in names:
                continue
            bb = t.get_window_extent(r)
            if (bb.x0 < ab.x0 + pad or bb.x1 > ab.x1 - pad
                    or bb.y0 < ab.y0 + pad or bb.y1 > ab.y1 - pad):
                bad.append(f"{s[:24]!r} at [{bb.x0:.0f},{bb.x1:.0f}]x"
                           f"[{bb.y0:.0f},{bb.y1:.0f}] vs axes "
                           f"[{ab.x0:.0f},{ab.x1:.0f}]x[{ab.y0:.0f},{ab.y1:.0f}]")
    if bad:
        raise SystemExit(f"{stem}: in-axes text outside its axes:\n "
                         + "\n ".join(bad))

def check_legend_clear(fig, stem, gap_pt=2.0):
    """The figure legend must not touch any axis label or tick label.

    Artist-vs-artist overlap is the blind spot fit_check names in its own
    docstring, and it is not hypothetical here: at MATH_PT = 10.5 the
    `L [um]` xlabels of panels (a) and (b) ran straight into the `P_N 1D` and
    `P_N 2D` legend entries -- the `]` sitting on the `P` -- while fit_check
    stayed green because both artists were comfortably on the page.  Only
    reading the 600 dpi render caught it.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    scale = 72.0 / fig.dpi            # window extents are pixels at fig.dpi
    pad = gap_pt / scale
    # Ticks whose LOCATION is outside the view are artists matplotlib keeps and
    # never draws -- and on a log axis it projects them to a position that can
    # land anywhere on the canvas, including right in the legend band.  Same
    # filter fit_check uses, and without it this check is pure noise.
    dead = set()
    for a in fig.axes:
        for axis, lim in ((a.xaxis, a.get_xlim()), (a.yaxis, a.get_ylim())):
            lo, hi = sorted(lim)
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tk.get_loc() <= hi + 1e-9:
                    dead.update((id(tk.label1), id(tk.label2)))
    bad = []
    for lg in fig.legends:
        lb = lg.get_window_extent(r)
        for ax in fig.axes:
            probes = [ax.xaxis.label, ax.yaxis.label]
            probes += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
            for t in probes:
                if not t.get_visible() or not str(t.get_text()).strip():
                    continue
                if id(t) in dead:
                    continue
                tb = t.get_window_extent(r)
                if (tb.x0 < lb.x1 + pad and tb.x1 > lb.x0 - pad
                        and tb.y0 < lb.y1 + pad and tb.y1 > lb.y0 - pad):
                    bad.append(
                        f"legend vs {str(t.get_text())[:20]!r}: "
                        f"legend y[{lb.y0 * scale:.1f},{lb.y1 * scale:.1f}] "
                        f"text y[{tb.y0 * scale:.1f},{tb.y1 * scale:.1f}] pt "
                        f"(need {gap_pt:.1f} pt clear)")
    if bad:
        raise SystemExit(f"{stem}: legend collides with axis lettering:\n "
                         + "\n ".join(bad))

def verify_page(path: Path, fig) -> None:
    """The saved page MUST be figsize*72.

    Under ``savefig.bbox='tight'`` the saved width is the drawn INK, so one
    over-wide legend -- or an invisible artist -- sets the page size and
    \\includegraphics silently demagnifies the whole drawing, taking every
    point size with it (measured elsewhere in this paper: a figure authored at
    7.2 in printed at 55.6%).  This reads the emitted PDF back and asserts the
    page really is the slot.
    """
    if path.suffix != ".pdf":
        return
    want_w, want_h = (fig.get_size_inches() * 72.0)
    head = path.read_bytes()[:4096].decode("latin-1")
    box = re.search(r"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+"
                    r"([\d.]+)\s+([\d.]+)\s*\]", head)
    if not box:
        raise SystemExit(f"{path.name}: no /MediaBox in the first 4 kB")
    x0, y0, x1, y1 = (float(v) for v in box.groups())
    got_w, got_h = x1 - x0, y1 - y0
    if abs(got_w - want_w) > 1.0 or abs(got_h - want_h) > 1.0:
        raise SystemExit(f"{path.name}: page {got_w:.2f} x {got_h:.2f} pt "
                         f"but figsize says {want_w:.2f} x {want_h:.2f} -- "
                         f"savefig.bbox is not 'standard'")
    if abs(got_w - SLOT_PT) > 1.0:
        raise SystemExit(f"{path.name}: page width {got_w:.2f} pt is not the "
                         f"{SLOT_PT:.2f} pt slot; on-page scale != 1.000 and "
                         f"every size in this file stops meaning anything")

def load():
    d = json.loads(ARTEFACT.read_text())
    anchors = {a["label"]: a["value"] for a in d["anchors"]}
    return d, anchors

def _tag(ax, s):
    """Panel letter in the BOTTOM-RIGHT: every panel here rises left-to-right,
    so that corner is the only one guaranteed empty."""
    ax.text(0.955, 0.06, s, transform=ax.transAxes, va="bottom", ha="right",
            fontweight="bold", fontsize=MATH_PT)

def _anchor_label(ax, y, s, fontsize=PLAIN_PT):
    """Anchor-line label at the TOP-LEFT, where no curve reaches.

    `fontsize` is a parameter because only SOME of these carry mathtext: at
    7.5 pt the "bulk" of $\\kappa_\\mathrm{bulk}$ set at 5.25 pt on the page,
    the smallest type in the figure and well under Elsevier's 7 pt floor.
    mathtext sets sub/superscripts at 0.7x, so anything containing one has to
    be at BASE >= MATH_PT; plain words like "Fourier" only have to clear the
    floor themselves, and sit at PLAIN_PT."""
    ax.text(0.035, y, s, transform=ax.get_yaxis_transform(), ha="left",
            va="bottom", fontsize=fontsize, color="0.15")

def _decades(ax):
    """L axis: labels only every SECOND decade -- at a third of the 388.53 pt
    slot, five labelled decades overlap each other into an unreadable smear."""
    ax.set_xscale("log")
    ax.set_xlim(6e-3, 1.7e2)
    ax.set_xticks([1e-2, 1e0, 1e2])
    ax.set_xticklabels([r"$10^{-2}$", r"$1$", r"$10^{2}$"], fontsize=MATH_PT)
    ax.set_xticks([1e-1, 1e1], minor=True)
    # The log MINOR formatter keeps labelling 1e-1 and 1e1 on top of the three
    # majors; set_xticklabels only ever touched the majors.
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$L$ [$\mu$m]", fontsize=MATH_PT)

def build(outdir: Path) -> None:
    d, anchors = load()
    one, two = d["pn_1d"], d["pn_2d"]
    L1 = [r["L_um"] for r in one]
    L2 = [r["L_um"] for r in two]
    D_bulk, kappa_bulk = anchors["D_bulk"], anchors["kappa_bulk"]
    # the joint comb produces a capacity, so kappa_eff no longer borrows one.
    C_comb = anchors["C_comb"]

    # Two independent messages: the dimensional sweep and the material/grid
    # scale. S and D_eff are fixed rescalings, retained in Table 5.
    fig, axes = plt.subplots(1, 2, figsize=(SLOT_IN, 2.80))
    ax_k, ax_nk = axes
    ax_k.plot(L1, [r["kappa_eff_dom"] for r in one], **DOM)
    ax_k.plot(L1, [r["kappa_eff_pn"] for r in one], **PN1D)
    ax_k.plot(L2, [r["kappa_eff_pn"] for r in two], **PN2D)
    ax_k.axhline(kappa_bulk, color="0.15", ls=(0, (4, 2.5)), lw=0.9, zorder=0)
    ax_k.set_yscale("log")
    ax_k.set_ylim(1.2, 5.5e2)
    _anchor_label(ax_k, kappa_bulk * 1.1, r"$\kappa_\mathrm{bulk}$",
                  fontsize=MATH_PT)
    ax_k.set_ylabel(r"$\kappa_\mathrm{eff}$ [W m$^{-1}$ K$^{-1}$]",
                    fontsize=MATH_PT)
    _decades(ax_k)
    _tag(ax_k, "(a)")

    # ---- (c) the absolute-scale passport ---------------------------------
    # The one thing the absolute numbers depend on that the ratio S does not:
    # how converged the Nk=20 production mode set is.  TWO curves since the
    # production k-range kappa_bulk slides just under the band with Nk, and
    # the full-BZ curve shows that the whole of that slide is the deliberate
    # zone-centre/zone-edge truncation rather than the mode count.
    sw = d["nk_sweep"]
    nks = [s["Nk"] for s in sw]
    kbs = [s["kappa_bulk"] for s in sw]
    kfull = [s["kappa_bulk_fullbz"] for s in sw]
    ax_nk.axhspan(142.0, 156.0, color="0.75", alpha=0.45, lw=0, zorder=0)
    ax_nk.axhline(d["passport"]["kappa_literature_W_m_K"], color="0.15",
                  ls=(0, (4, 2.5)), lw=0.9, zorder=1)
    ax_nk.plot(nks, kfull, color="C4", marker="v", ls=":", lw=1.1, ms=3.4,
               zorder=2, label="zone centre restored")
    ax_nk.plot(nks, kbs, color="C2", marker="^", ls="-", lw=1.2, ms=3.6,
               zorder=3, label="production grid")
    prod = kbs[nks.index(20)]
    ax_nk.plot([20], [prod], marker="o", ms=7.5, mfc="none", mec="C0",
               mew=1.3, ls="none", zorder=4)
    # The production point is marked by the ring ALONE and named in the
    # caption.  Every in-axes position for a text label collided with either
    # the left spine, the Nk=10 marker, the "expt." band label or the panel
    # tag -- none of which fit_check can see (artist-vs-artist overlap is
    # outside what it tests).
    ax_nk.set_xscale("log")
    # artefact's sweep runs to Nk=800 -- so the CONVERGED end of both curves
    # (141.70 and 146.90 W/m/K, and `kappa_bulk_converged` is an anchor of the
    # artefact) was drawn outside the view and silently dropped.  A clipped
    # data artist is precisely what fit_check cannot see: it only looks at Text.
    ax_nk.set_xlim(7.5, 1.4e3)
    ax_nk.set_xticks([10, 100, 1000])
    ax_nk.set_xticklabels([r"$10$", r"$10^{2}$", r"$10^{3}$"],
                          fontsize=MATH_PT)
    # Both curves now live inside ~135-155, so the old 78-178 window (sized
    # when the production anchor was 120) would flatten the very drift the
    # panel exists to show.
    # Top is 164, not 162: the two-line "expt." block is anchored at data
    # y=156.6 and grows UPWARD, and against a 162 ceiling its ascender box
    # cleared the top spine and ran 0.5 pt off the top of the PAGE.  Both
    # curves live in 141.7-152.2, so the extra 2 K of headroom costs nothing.
    ax_nk.set_ylim(134, 164)
    ax_nk.set_xlabel(r"$N_k$", fontsize=MATH_PT)
    ax_nk.set_ylabel(r"$\kappa_\mathrm{bulk}$ [W m$^{-1}$K$^{-1}$]",
                     fontsize=MATH_PT)
    # TWO LINES.  At 10 pt mathtext bases the three ylabels and the twin axis
    # take enough width that panel (c) is ~55 pt narrower than it was, and this
    # label on one line ran off the right edge of the canvas (fit_check caught
    # it).  The range is worth keeping ON the figure -- it is what the shaded
    # band means -- so it wraps instead of shrinking below the 7 pt floor.
    _anchor_label(ax_nk, 158.0, "bulk Si, 300 K\n142–156 W/(m K)")
    _tag(ax_nk, "(b)")
    ax_nk.text(900, kfull[-1] + 0.65, "zone centre restored", ha="right", va="bottom",
               fontsize=PLAIN_PT, color="C4")
    ax_nk.text(900, kbs[-1] - 0.75, "production range", ha="right", va="top",
               fontsize=PLAIN_PT, color="C2")
    ax_nk.text(13, 135.4, "ring: production $N_k=20$", fontsize=MATH_PT)

    for ax in (ax_k, ax_nk):
        ax.tick_params(labelsize=MATH_PT)

    # ONE legend under the row.  A per-panel legend at a third of the slot
    # would sit on the data, which fit_check cannot see (its own docstring).
    handles, labels = ax_k.get_legend_handles_labels()
    order = [1, 2, 0]                       # P_N arms first, arbiter last
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.004), handlelength=1.9,
               columnspacing=1.0, handletextpad=0.45, fontsize=MATH_PT)
    # Every panel carries its OWN xlabel.  With only panel (c) labelled,
    # tight_layout sized all three subplots to (c)'s taller requirement and
    # left a band of white under (a) and (b).
    #
    # The reserved bottom strip is 0.175 of 167.8 pt = 29.4 pt, not the old
    # 0.10 of 155.5 pt = 15.6 pt.  The legend occupies ~[6, 25] pt off the
    # bottom edge, so 15.6 pt left the xlabels overlapping it outright: at
    # MATH_PT the `]` of `L [um]` sat on the `P` of `P_N 1D`, and fit_check was
    # green throughout because both artists were comfortably on the page.
    # check_legend_clear() now holds a 2 pt margin mechanically.
    # pad 0.50, not 0.36: at 0.36 panel (a)'s rotated ylabel sat flush on the
    # page's left edge (its STIX italic overhang measured 0.6 pt PAST it) and
    # panel (c)'s 10^3 tick had 1.3 pt to the right edge.
    fig.tight_layout(rect=(0, 0.19, 1, 1.0), w_pad=1.00, pad=0.50)
    fit_check(fig, "fig_absolute_ladder")
    check_type_floor(fig, "fig_absolute_ladder")
    check_data_visible(fig, "fig_absolute_ladder")
    check_legend_clear(fig, "fig_absolute_ladder")
    check_text_inside_axes(fig, "fig_absolute_ladder",
                           {"Fourier", r"$\kappa_\mathrm{bulk}$", "expt.",
                            "(a)", "(b)", "(c)"})

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = outdir / f"fig_absolute_ladder.{ext}"
        fig.savefig(out)
        verify_page(out, fig)
    plt.close(fig)

    w = fig.get_size_inches()[0] * 72
    print(f"fig_absolute_ladder: canvas {w:.1f} pt "
          f"(slot {SLOT_IN * 72:.2f} pt) -> {outdir}")
    print(f"  S:       {one[0]['S_pn']:.4f} .. {one[-1]['S_pn']:.4f}  (1D)")
    print(f"  kappa:   {one[0]['kappa_eff_pn']:.2f} .. "
          f"{one[-1]['kappa_eff_pn']:.2f} W/m/K   anchor {kappa_bulk:.2f}")
    print(f"  D_eff:   {one[0]['D_eff_pn']:.3e} .. {one[-1]['D_eff_pn']:.3e}"
          f" m^2/s   anchor {D_bulk:.3e}")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=FIG_DIR)
    build(ap.parse_args().outdir)

if __name__ == "__main__":
    main()
