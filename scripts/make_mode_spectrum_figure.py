"""presentation of the analytic PN mixing weights and mode spectrum."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p101_mode_spectrum as MS  # noqa: E402

FIG_DIR = REPO / "figures"
STEM = "fig_mode_spectrum"

# --------------------------------------------------------------------------- #
# DISCRETE POINT MASS -- one mode carrying 43.1% of the total heat capacity at
# MFP = 4.4 nm -- and a KDE that smears 43.1% of the weight into a
# 0.13-decade half-spike both misdraws the atom and crushes the acoustic
# humps to ~1/4 of their depth under the shared peak normalisation.  So the
# reservoir is drawn as a per-period STEM (length proportional to its C
# share, at its exact xi_opt = q v_opt tau_opt) and EXCLUDED from the fill,
# whose weights are renormalised over the 60 acoustic modes.  Two annotation
# lines under the rug rows print both shares from the claim store
# (pn.modespec.reservoir.share.C), so the reader is told the weights still
# sum correctly and how.
# --------------------------------------------------------------------------- #
def acoustic_split():
    """(w_ac renormalised over the 60 acoustic modes, reservoir C share).

    The joint comb concatenates the lumped optical reservoir LAST
    (optical_reservoir.joint_modes), so the acoustic block is C[:60];
    `MS.reservoir_share` asserts the ordering two ways (argmin of the MFP
    v*tau, MFP = 4.44 nm) rather than trusting it.
    """
    _, _, C = MS.modes()
    assert C.shape == (61,), C.shape
    return C[:60] / C[:60].sum(), MS.reservoir_share()

def acoustic_density(L_um: float, grid, w_ac, bw: float = 0.13):
    """MS.spectral_density restricted to the 60 acoustic modes, renormalised.

    Same fixed-bandwidth Gaussian kernel in log10(xi) as MS's own drawing
    density; only the mode set (no reservoir) and the weight normalisation
    differ.
    """
    lx = np.log10(MS.xi_of(L_um)[:60])
    d = np.exp(-0.5 * ((grid[:, None] - lx[None, :]) / bw) ** 2) / (
        bw * np.sqrt(2.0 * np.pi))
    return d @ w_ac

def acoustic_support(L_um: float) -> tuple:
    """(xi_min, xi_max) over the 60 acoustic modes only -- what the fill is
    clipped to; the reservoir sits below it as a stem, not as density."""
    xi = MS.xi_of(L_um)[:60]
    return float(xi.min()), float(xi.max())

#: Every numeral this figure prints that is NOT a claim-store value, with the
#: reason it is allowed to be a literal.  Kept as data so the list can be read
#: against the rendered PDF (see --audit).
TICK_AUDIT = {
    "10^-3 .. 10^4": "round decades, xi axis of (a)",
    "0, 0.5, 1": "round fractions, weight axis of (a) and share axis of (b)",
    "10^-2 .. 10^2": "round decades, grating-period axis of (b)",
    "0.1, 1, 10": "the three drawn grating periods, in um -- configuration "
                  "(sweep periods, MS.PANEL_UM)",
    "1, 10": "xi_split and xi_duh, the RETIRED band-routed cuts -- "
             "configuration (spectral_pn.py:480-481 defaults)",
    # measure-corrected acoustic + 1 optical reservoir), not 60.
}

DPI = 300

# Elsevier's artwork spec is 7 pt for normal text on the page and 6 pt for
# sub/superscripts.  matplotlib's mathtext sets a script at 0.7x the BASE, so
# the base -- not the rendered glyph -- is what a source-side size controls,
# and a guard that reads `Text.get_fontsize()` reports the base and is a FALSE
# GREEN.  (That is precisely how this file shipped 573 sub-floor glyphs: every
# nominal size in it was >= 6.3 pt and the smallest glyph on the page was
# 5.16 pt.)  Hence:
#
#     MATH_PT  = 10.5   any string carrying a sub/superscript -> 7.35 pt
#     PLAIN_PT =  9.5   plain words, no script
#
# 10.0 would give exactly 7.00 -- the floor with ZERO headroom, which is the
# state the rest of the paper is in and the reason any later shrink breaks on
# arrival.  10.5 buys 0.35 pt, and matches the venue evidence: a typeset JCP
# figure (Xiao & Frank, JCP 490 (2023) 112317) letters at Helvetica 9.96 pt
# against a 7.97 pt body, i.e. ~25% ABOVE body text.
# NESTING IS REFUSED, not shrunk to fit: depth 2 is 0.49x = 5.15 pt at any base
# this box can afford.  `_assert_type_floor` below is the source-side gate; the
# only authoritative measurement is `mutool draw -F stext` over the built PDF.
MATH_PT = 10.5
PLAIN_PT = 9.5

# WHERE THE HEIGHT FOR IT CAME FROM.  It did not exist as a resize: this box
# already sits against `BOX_MAX_PT`.  It came from deleting the two note blocks
# that used to hang under the axes -- ~455 glyphs at 6.3 pt that were a VERBATIM
# duplicate of the caption (main.tex prints the same span, the same weighted
# interval and the same three cut shares as \PnModeSpecXiSpan,
# \PnModeSpecXiSpanWeighted and \PnModeSpecCut{Quasi,Unit,Ten}), and from moving
# panel (b)'s three-entry legend out of the axes into the strip they vacated.
# Zero information left the float; the same sentences are now set once, at
# caption size, where they are selectable text.
plt.rcParams.update({
    "font.size": PLAIN_PT,
    "axes.titlesize": MATH_PT,
    "axes.labelsize": MATH_PT,
    # size that prints.  It was not: under `savefig.bbox: "tight"` the SAVED
    # width is the drawn ink, so a 6.30 in figsize came out 450.1 pt wide and
    # \includegraphics reduced it into the 390 pt slot at 0.867 -- an 8.2 pt
    # tick label printed at 7.1 pt and the 6.3 pt note blocks at 5.5 pt.  With
    # `standard` the page size IS `figsize`, so the reduction is gone and the
    # same nominal sizes gain 15%.  The price is that anything wider than the
    # canvas is CLIPPED rather than absorbed, which is what `_assert_fits`
    # below now enforces.
    "xtick.labelsize": MATH_PT,
    "ytick.labelsize": MATH_PT,
    "legend.fontsize": MATH_PT,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "standard",
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "font.family": "sans-serif",
    # Elsevier permits Arial/Helvetica, Times, Courier, Symbol.  Nimbus Sans is
    # the URW Helvetica clone and is installed here; DejaVu -- what this file
    # embedded until now, as a Type 3 subset -- is NOT on that list.
    "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "Arimo",
                        "DejaVu Sans"],
    # "stix", NOT "stixsans": stixsans draws sans math out of STIXNonUnicode,
    # which has no usable Unicode map, so pdftotext returns every mathtext
    # pdftotext.  Serif math beside sans words is also the measured JCP idiom.
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Palette: the fig1_schematic families, so the two figures read as one paper.
INK = "#1a1a1a"
MUT = "#5a5a5a"
C_PERIOD = {0.1: "#b4650a", 1.0: "#4f3f96", 10.0: "#12705a"}   # amber/violet/teal
C_K0, C_K1 = "#1a1a1a", "#8a8a8a"
C_CLOSURE = "#a32020"

class Numbers:
    """Printed numerals, derived live from the SAME functions that draw the curves
    (`p101_mode_spectrum`), rendered through declared templates.
    """

    _RENDERS = {"g3": "{v:.3g}", "pct1fig": "{v:.1f}%"}

    def __init__(self) -> None:
        self._values = {
            "pn.modespec.xi.geomean":
                (np.asarray(MS.PANEL_UM, dtype=float),
                 np.asarray([MS.xi_geomean(L) for L in MS.PANEL_UM])),
            "pn.modespec.slaved.share.C":
                (np.asarray(MS.LADDER_UM, dtype=float),
                 np.asarray([MS.slaved_share(L) for L in MS.LADDER_UM])),
            "pn.modespec.slaved.share.closure":
                (np.asarray(MS.LADDER_UM, dtype=float),
                 np.asarray([MS.slaved_share(L, "closure")
                             for L in MS.LADDER_UM])),
            "pn.modespec.reservoir.share.C":
                (np.asarray(["reservoir", "acoustic"]),
                 np.asarray([MS.reservoir_share(),
                             1.0 - MS.reservoir_share()])),
        }

    def values(self, cid: str) -> np.ndarray:
        return self._values[cid][1]

    def index(self, cid: str, axis: str) -> np.ndarray:
        assert axis == "L_um"
        return self._values[cid][0]

    def text(self, cid: str, at: int, render: str) -> str:
        v = float(self._values[cid][1][at])
        if render == "pct1fig":
            return self._RENDERS[render].format(v=100.0 * v)
        return self._RENDERS[render].format(v=v)


def _verify_page(path: Path, fig) -> None:
    r"""Read the emitted PDF back and assert its page is the slot.

    NOT redundant with anything above, and NOT a tautology like comparing the
    canvas to the constant it was built from: `savefig` is free to disagree
    with `figsize`, and under `savefig.bbox="tight"` it always does -- the page
    becomes the DRAWN INK, so \includegraphics[width=\textwidth] then
    demagnifies the whole drawing and every point size in this file prints
    smaller than it says.  That is not hypothetical: this figure once saved
    450.1 pt wide from a 6.30 in canvas and printed at 0.867, which is how
    8.2 pt tick labels reached the reader at 7.1 pt.  The only way to know is
    to read the artefact.
    """
    box = re.search(rb"/MediaBox\s*\[([^\]]*)\]", path.read_bytes())
    if box is None:
        raise AssertionError(f"{path.name}: no /MediaBox to verify")
    x0, y0, x1, y1 = (float(v) for v in box.group(1).split())
    w, h = x1 - x0, y1 - y0
    want_w, want_h = (v * 72.0 for v in fig.get_size_inches())
    if abs(w - want_w) > 1.0 or abs(h - want_h) > 1.0:
        raise AssertionError(
            f"{path.name} page is {w:.2f} x {h:.2f} pt but the canvas is "
            f"{want_w:.2f} x {want_h:.2f} -- savefig resized it (bbox=tight?); "
            "the on-page type size is then NOT what this file sets.")
    if abs(w - SLOT_PT) > 1.0:
        raise AssertionError(
            f"{path.name} page is {w:.2f} pt against a {SLOT_PT} pt slot -- "
            f"\\includegraphics will scale by {SLOT_PT / w:.4f} and every size "
            "here prints at that factor.")

def _save(fig, outdir: Path, stem: str) -> list:
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("pdf", "png"):
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path)
        out.append(path)
    _verify_page(out[0], fig)
    plt.close(fig)
    print(f"  wrote {out[0].name} + {out[1].name}")
    return out

#: The \includegraphics slot, in POSTSCRIPT points: `width=\textwidth` in
#: elsarticle preprint/12pt is \textwidth = 390.0 TeX pt, and a TeX pt is
#: 72/72.27 of a PS pt, so the slot a PDF page is measured in is
#: 390.0 * 72/72.27 = 388.53 bp.  Authoring at 390.0 -- which this file did
#: \includegraphics demagnify by 0.9962, so every size here printed 0.4%
#: smaller than authored for no reason anyone intended.
SLOT_PT = 388.53

#: On-page height budget for the box.  \textheight = 548.5 pt less this
#: figure's caption (~225 pt at 12 pt over 15 lines) less the float
#: separations.  Exceeding it turns into "Float too large for page" in
#: main.log, which is the verdict -- re-measure it there, do not trust this
#: comment after the caption is edited.  MEASURED: 311.0 overran by 2.46 pt,
#: so this is that minus a 5 pt margin.  If the caption grows, this must
#: shrink; the two are one budget.
BOX_MAX_PT = 303.5

def _assert_fits(fig, *artists) -> None:
    """Fail if any of `artists` is wider than the canvas.

    Under `savefig.bbox: "standard"` an over-wide line is silently CLIPPED, and
    under the "tight" it used to have, it silently inflated the canvas and
    shrank the whole figure.  Both failures are invisible in the source, and
    both have shipped here before: a corrected note line came out 392.9 pt
    against a 343.4 pt axes, so the axes frame printed through four lines of
    the figure's own text.  Measured against the drawn extent, per line.
    """
    fig.canvas.draw()
    w_px = fig.get_window_extent().width
    for a in artists:
        bb = a.get_window_extent()
        # is blind to exactly the case that arrives with a CENTRED artist: a
        # figure legend too wide for the canvas overhangs by half its excess on
        # each side, and half an overhang can be small enough to pass a
        # right-edge-only test while the left column is already clipped.
        for over, edge in (((bb.x1 - w_px) / fig.dpi * 72.0, "right"),
                           ((-bb.x0) / fig.dpi * 72.0, "left")):
            if over > 0:
                raise AssertionError(
                    f"{a.get_text()[:60]!r} runs {over:.1f} pt past the {edge} "
                    f"edge of the canvas -- rewrap it; do NOT let it clip.")

#: matplotlib mathtext shrinks one level of sub/superscript to 0.7x the base.
_SHRINK = 0.7
FLOOR_TEXT, FLOOR_SCRIPT = 7.0, 6.0

def _script_depth(s: str) -> int:
    """Deepest sub/superscript nesting inside the math spans of `s`.

    Only `_`/`^` INSIDE `$...$` count; a literal underscore in prose does not
    shrink anything.  Depth is tracked per brace group, because `$x_{a_b}$`
    nests and `$x_a y_b$` does not -- 0.7 vs 0.49 of the base, i.e. 7.35 pt vs
    5.15 pt at MATH_PT, one side of the floor each.
    """
    depth = mx = 0
    stack: list[int] = []
    in_math = pending = False
    for ch in s:
        if ch == "$":
            in_math, pending, depth, stack = not in_math, False, 0, []
            continue
        if not in_math:
            continue
        if ch in "_^":
            pending = True
            depth += 1
            mx = max(mx, depth)
        elif ch == "{":
            stack.append(depth if pending else -1)
            pending = False
        elif ch == "}":
            if stack:
                top = stack.pop()
                if top >= 0:
                    depth = top - 1
        elif pending:
            pending = False
            depth -= 1
    return mx

def _assert_type_floor(fig) -> None:
    """Fail if any string in the figure would print below the Elsevier floor."""
    canvas_pt = fig.get_size_inches()[0] * 72.0
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if not s.strip() or not t.get_visible():
            continue
        d = _script_depth(s)
        if d >= 2:
            raise AssertionError(
                f"{s[:50]!r} nests sub/superscripts {d} deep -> "
                f"{t.get_fontsize() * _SHRINK ** d:.2f} pt on the page; "
                "rewrite it flat, do not shrink to fit.")
        # NOTE the scale term below is 1.000 BY CONSTRUCTION -- the canvas is
        # built from SLOT_PT -- so it is not a check.  What the figure actually
        # prints at is verified over the emitted PDF by `_verify_page`; this
        # gate is only about the sizes set in this file.
        # BOTH parts of a script-bearing string are checked, because they have
        # DIFFERENT floors: the body of "$\xi_m = q v_m \tau_m$" is normal text
        # (7 pt) and only the "m" is a script (6 pt).
        scale = SLOT_PT / canvas_pt
        base = t.get_fontsize() * scale
        for size, floor, what in ((base, FLOOR_TEXT, "body"),
                                  (base * _SHRINK ** d, FLOOR_SCRIPT,
                                   "script")) if d else ((base, FLOOR_TEXT,
                                                          "body"),):
            if size < floor - 1e-9:
                raise AssertionError(
                    f"{s[:50]!r} at base {t.get_fontsize()} pt prints its "
                    f"{what} at {size:.2f} pt, under the {floor} pt floor.")

def _assert_legend_boxes_fit(fig, *legends) -> None:
    """Fail if a legend's BOX -- handles and padding included -- leaves the canvas."""
    fig.canvas.draw()
    w_pt = fig.get_size_inches()[0] * 72.0
    for lg in legends:
        bb = lg.get_window_extent()
        x0, x1 = bb.x0 / fig.dpi * 72.0, bb.x1 / fig.dpi * 72.0
        if x0 < 0 or x1 > w_pt:
            raise AssertionError(
                f"legend box spans {x0:.1f}..{x1:.1f} pt on a {w_pt:.2f} pt "
                "canvas -- shorten the labels or tighten handlelength/"
                "columnspacing; do NOT let it clip.")

def _assert_disjoint(fig, artists) -> None:
    """Fail if any two of `artists` overlap each other."""
    fig.canvas.draw()
    for i, a in enumerate(artists):
        for b in artists[i + 1:]:
            if a.get_window_extent().overlaps(b.get_window_extent()):
                raise AssertionError(
                    f"{a.get_text()!r} overlaps {b.get_text()!r}")

def _assert_clear_of_curves(fig, ax, texts, lines, pad_pt=1.0) -> None:
    """Fail if a label's box lands on a drawn curve.

    THE DEFECT THIS EXISTS FOR IS INVISIBLE TO EVERY OTHER GATE IN THIS FILE.
    `fit_check`-style tests compare text to the CANVAS; `_assert_no_overlap`
    compares text to other TEXT and to the legend.  Neither looks at data, so a
    percentage annotation whose opaque bbox cuts a 1.6 pt hole in the very
    curve it annotates passes both -- which is exactly what shipped: at 220 dpi
    the black heat-capacity curve was visibly severed between L = 3.6 and
    7.9 um by its own "99.3%" label.  Sampled in DISPLAY space, so it is a
    statement about the printed page and not about data coordinates.
    """
    fig.canvas.draw()
    pad = pad_pt * fig.dpi / 72.0
    for ln in lines:
        xy = ax.transData.transform(ln.get_xydata())
        for t in texts:
            bb = t.get_window_extent().expanded(1.0, 1.0).padded(pad)
            hit = ((xy[:, 0] >= bb.x0) & (xy[:, 0] <= bb.x1)
                   & (xy[:, 1] >= bb.y0) & (xy[:, 1] <= bb.y1))
            if hit.any():
                j = int(hit.argmax())
                raise AssertionError(
                    f"{t.get_text()!r} sits on {ln.get_label()!r} at "
                    f"(x={ln.get_xydata()[j][0]:.4g}, "
                    f"y={ln.get_xydata()[j][1]:.4g}) -- move the label off the "
                    "curve; do NOT rely on the white bbox to hide the break.")

def _assert_no_overlap(fig, ax, *artists) -> None:
    """Fail if any of `artists` overlaps the axes legend or the y=0 spine."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    leg = ax.get_legend()
    lb = None if leg is None else leg.get_window_extent().transformed(inv)
    for a in artists:
        bb = a.get_window_extent().transformed(inv)
        if lb is not None and bb.overlaps(lb):
            raise AssertionError(
                f"{a.get_text()!r} at y {bb.y0:.3f}..{bb.y1:.3f} overlaps the "
                f"legend at y {lb.y0:.3f}..{lb.y1:.3f}")
        if bb.y0 < 0.0 < bb.y1:
            raise AssertionError(
                f"{a.get_text()!r} straddles the y=0 axis "
                f"(y {bb.y0:.3f}..{bb.y1:.3f})")

# --------------------------------------------------------------------------- #
def build(N: Numbers, outdir: Path) -> list:
    """Analytic kernels, a separate mode strip, and the low-rate share.

    The material source and KDE bandwidth are unchanged. Acoustic density,
    optical point masses and nonnegative composition weights have separate
    labelled scales; no historical threshold is drawn as an active switch.
    """
    xi_lo, xi_hi = 1e-3, 1e4
    grid = np.logspace(np.log10(xi_lo), np.log10(xi_hi), 900)
    K0, K1 = MS.kernels(grid)
    fig = plt.figure(figsize=(SLOT_PT / 72., BOX_MAX_PT / 72.))
    gs = fig.add_gridspec(3, 1, height_ratios=[1., 1.05, 1.],
                          left=.13, right=.98, top=.945, bottom=.17, hspace=.48)
    ax, sx, bx = (fig.add_subplot(gs[j]) for j in range(3))
    ax.set_xscale("log"); ax.set_xlim(xi_lo, xi_hi); ax.set_ylim(-.03, 1.1)
    ax.plot(grid, K0, color=C_K0, lw=1.7, label=r"$K_0(\xi)$")
    ax.plot(grid, K1, color=C_K1, lw=1.2, ls=(0, (4, 2)), label=r"$K_1(\xi)$")
    ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0", "0.5", "1"])
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("kernel weight")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    ax.text(.015, .20, "(a)", transform=ax.transAxes, fontweight="bold")
    fig.suptitle("Analytic mixing weights and mode distribution", fontsize=10, y=.995)

    # Three rows retain each period's full acoustic support and exact optical
    # location. Density is acoustic-only with the original common peak scale;
    # stem height is the optical fraction of total C, not a density estimate.
    sx.set_xscale("log"); sx.set_xlim(xi_lo, xi_hi); sx.set_ylim(-.12, 2.9)
    lg = np.linspace(np.log10(xi_lo), np.log10(xi_hi), 1400)
    w_ac, res_share = acoustic_split()
    peak = max(acoustic_density(L, lg, w_ac).max() for L in MS.PANEL_UM)
    for i, L in enumerate(MS.PANEL_UM):
        base = 2. - i
        col = C_PERIOD[L]
        xx = 10.**lg
        den = acoustic_density(L, lg, w_ac) / peak
        lo, hi = acoustic_support(L)
        keep = (xx >= lo) & (xx <= hi)
        sx.fill_between(xx, base, base + .55 * den, where=keep,
                         color=col, alpha=.28, linewidth=0)
        sx.plot(xx[keep], base + .55 * den[keep], color=col,
                 ls=("-", "--", "-.")[i], lw=.8)
        xi = MS.xi_of(L)
        sx.plot(xi[:60], np.full(60, base), "|", color=col, ms=3., mew=.5)
        sx.plot([xi[60], xi[60]], [base, base + .55 * res_share],
                 color=col, lw=1.8)
        sx.plot([xi[60]], [base + .55 * res_share], "o", color=col, ms=2.3)
        sx.text(.985, (base + .32 + .12)/3.02, f"{L:g} μm", ha="right", transform=sx.transAxes,
                 va="center", fontsize=PLAIN_PT, color=col)
    sx.set_yticks([])
    sx.set_ylabel("mode spectrum")
    sx.set_xlabel(r"per-mode optical thickness $\xi_m=qv_m\tau_m$", labelpad=1)
    sx.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000])
    sx.text(.99, 1.05, "fill: acoustic density; stem: optical C fraction",
             transform=sx.transAxes, ha="right", va="bottom", fontsize=8.5)
    cid_res = "pn.modespec.reservoir.share.C"
    stored = N.values(cid_res)
    assert abs(res_share-stored[0]) < 1e-12
    assert abs(1-res_share-stored[1]) < 1e-12

    cid_C = "pn.modespec.slaved.share.C"
    cid_cl = "pn.modespec.slaved.share.closure"
    Lr = N.index(cid_C, "L_um")
    fC, fcl = N.values(cid_C), N.values(cid_cl)
    dense = np.logspace(-2.4, 2.4, 400)
    bx.set_xscale("log")
    bx.plot(dense, [MS.slaved_share(L) for L in dense], color=INK, lw=1.6,
             label=r"heat-capacity weight $C_m$")
    bx.plot(dense, [MS.slaved_share(L, "closure") for L in dense],
             color=C_CLOSURE, lw=1.1, ls=(0, (4, 2)),
             label=r"closure weight $C_m/\tau_m$")
    bx.plot(Lr, fC, "o", color=INK, ms=3.4, mfc="white", mew=.9)
    bx.plot(Lr, fcl, "o", color=C_CLOSURE, ms=2.6, mfc="white", mew=.8)
    bx.set_xlim(.72e-2, 1.4e2); bx.set_ylim(-.04, 1.16)
    bx.set_xticks([.01, .1, 1, 10, 100])
    bx.set_yticks([0, .5, 1]); bx.set_yticklabels(["0", "0.5", "1"])
    bx.set_xlabel(r"grating period $L$ [$\mu$m]", labelpad=1)
    bx.set_ylabel("carrier share")
    bx.grid(True, alpha=.22, lw=.4)
    bx.text(.015, .86, "(b)", transform=bx.transAxes, fontweight="bold")
    bx.text(.99, .15, r"$y\to0$ approximation", transform=bx.transAxes,
             ha="right", fontsize=MATH_PT)
    h, labels = bx.get_legend_handles_labels()
    leg = fig.legend(h, labels, loc="lower center", bbox_to_anchor=(.5, .007),
                     ncol=2, frameon=False, handlelength=1.5,
                     columnspacing=.8, handletextpad=.4, fontsize=MATH_PT)
    _assert_fits(fig, *leg.get_texts())
    _assert_legend_boxes_fit(fig, leg, ax.get_legend())
    _assert_type_floor(fig)
    return _save(fig, outdir, STEM)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=FIG_DIR)
    p.add_argument("--audit", action="store_true",
                   help="print the non-claim numerals this figure prints")
    a = p.parse_args()
    if a.audit:
        for k, why in TICK_AUDIT.items():
            print(f"  {k:16s} {why}")
        return
    N = Numbers()
    build(N, a.outdir)

if __name__ == "__main__":
    main()
