"""Fig. 1 for the spectral-PN paper: the ansatz and its data-free loop ."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath             # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# --------------------------------------------------------------------- slot
#: Canvas width in PostScript points (bp).  elsarticle[preprint,12pt] has
#: \textwidth = 390.0 TeX pt = 390.0 * 72/72.27 = 388.54 bp, so authoring here
#: makes the manuscript's on-page scale 1.000 and every size below literal.
#: This is the same constant the other PN generators use as SLOT_IN * 72.
SLOT_BP = 388.53
#: canvas.  The digest is the tighter of the two slots and sets the type floor.
S_DIGEST = 0.72 * 483.697 * (72.0 / 72.27) / SLOT_BP     # 0.893
S_MAIN = 390.0 * (72.0 / 72.27) / SLOT_BP                # 1.000

#: Elsevier, "Sizing of artwork": 7 pt for normal text, no smaller than 6 pt
#: for sub/superscript characters.
FLOOR_TEXT, FLOOR_SCRIPT = 7.0, 6.0
#: mathtext shrinks each script level by this factor (matplotlib SHRINK_FACTOR).
SHRINK = 0.7
#: Minimum BASE for any string carrying a script: 10 x 0.7 x 0.893 = 6.25 pt in
#: the digest, 7.00 pt in the manuscript.  Identical rationale to MATH_PT = 10
#: in make_absolute_ladder_figure.py.
MATH_MIN = 10.0

FS_EQ = 12.5        # the ansatz spine -- the one thing set large
FS_MATH = 10.0      # every other display equation
FS_TAG = 9.0        # brace tags, box notes, wire labels
FS_PANEL = 9.0      # panel letters (bold)
LEAD = 10.5         # baseline-to-baseline for stacked FS_TAG lines

INK = "#000000"
#: THE ENTIRE COLOUR BUDGET: one pale tint, used only as a FILL behind objects
#: the network produces.  Every tinted object is ALSO named "learned" by a
#: brace tag, so `--mono` loses decoration and no information.
TINT = "#dce5ef"
TINT_MONO = "#e2e2e2"

LW_BOX = 0.6        # RelaxNet's measured dominant stroke is 0.59 pt
LW_WIRE = 0.7
LW_BRACE = 0.6

plt.rcParams.update({
    "font.family": "sans-serif",
    # Nimbus Sans IS the URW Helvetica clone, so this satisfies Elsevier's
    # permitted-family list literally; Liberation Sans / Arimo are the Arial
    # metric fallbacks.  DejaVu -- what the predecessor embedded -- is last.
    "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "Arimo",
                        "DejaVu Sans"],
    # stixsans draws its sans math from STIXNonUnicode -- a font with, as the
    # name says, no usable Unicode mapping -- so `pdftotext` returned EVERY
    # with pdftotext, and under stixsans this figure went from 18 machine-
    # readable numerals to ZERO.  No claims value is burned in here, so nothing
    # would have failed today; it would simply have stopped being checkable.
    # Measured: stixsans -> "plain 1199 ≤ , / /"; stix and dejavusans ->
    # "plain 1199 l ≤ 8 am,9 7/γ 2π/L".  dejavusans is rejected separately
    # (non-permitted family, and the predecessor's own defect).
    # "stix" is also what the venue does: RelaxNet's Fig. 2 sets its math in
    # serif italic (M(f), Q=(E-f)/tau) and its WORDS in sans ("weight layer",
    # "ELU").  Serif math + Helvetica text is the measured JCP idiom, and both
    # families are on Elsevier's permitted list.
    # `custom` with Nimbus Sans was tried first and rejected: it falls back to
    # Computer Modern for \sum and \tau, and that \sum collides with the
    # following bracket.
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,          # embed TrueType: vector, searchable text
    "ps.fonttype": 42,
    # "standard", NOT "tight": the canvas is authored AT the slot, and "tight"
    # adds its pad to that width, demagnifying the whole drawing on the page.
    "savefig.bbox": "standard",
})

_TEXTS: list = []        # (artist, name, base_pt, script_depth)
_GUARD: list = []        # (artist, (x0, y0, x1, y1), name)

# ------------------------------------------------------------- mathtext depth
_MATH = re.compile(r"\$(.+?)\$", re.S)

def _depth_of(m: str) -> int:
    """Deepest sub/superscript nesting level in one mathtext expression.

    This is the whole point of the rewritten font gate.  matplotlib sets each
    level at 0.7x, so `e^{-t/\\tau_m}` -- superscript, then subscript inside it
    -- reaches the page at 0.49x of its base.  The predecessor's 8.8 pt display
    equations therefore printed at 4.31 pt while its gate reported OK.
    """
    depth = best = 0
    stack: list[bool] = []
    pending = False
    i, n = 0, len(m)
    while i < n:
        ch = m[i]
        if ch == "\\":                       # command or escaped brace
            if i + 1 < n and m[i + 1] in "{}":
                if pending:
                    best, pending = max(best, depth + 1), False
                i += 2
                continue
            j = i + 1
            while j < n and m[j].isalpha():
                j += 1
            if pending:
                best, pending = max(best, depth + 1), False
            i = max(j, i + 1)
            continue
        if ch in "_^":
            pending, i = True, i + 1
            continue
        if ch == "{":
            if pending:
                depth += 1
                stack.append(True)
                pending = False
                best = max(best, depth)
            else:
                stack.append(False)
            i += 1
            continue
        if ch == "}":
            if stack and stack.pop():
                depth -= 1
            i += 1
            continue
        if pending and not ch.isspace():     # single-token script, e.g. x_m
            best, pending = max(best, depth + 1), False
        i += 1
    return best

def script_depth(s: str) -> int:
    return max((_depth_of(m.group(1)) for m in _MATH.finditer(s)), default=0)

# ---------------------------------------------------------------- primitives
def T(ax, x, y, s, size=FS_TAG, weight="normal", ha="left", va="center",
      color=INK, z=6, guard=None, name="", **kw):
    """Text, registered with both gates.

    The size is RAISED, never lowered, when the string carries a script: a
    9 pt base would put its subscripts at 6.30 pt on the manuscript page and
    5.63 pt in the digest, under Elsevier's 6 pt script floor.  Making that a
    property of the drawing primitive is what stops it being re-lost.
    """
    d = script_depth(s)
    if d and size < MATH_MIN:
        size = MATH_MIN
    if d >= 2:
        raise SystemExit(
            f"nested script (depth {d}) in {s!r}: at base {size} pt that "
            f"reaches the page at {size * SHRINK ** d:.2f} pt. Rewrite it "
            f"(e.g. \\exp(-t/\\tau_m) instead of e^{{-t/\\tau_m}}).")
    a = ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
                color=color, zorder=z, **kw)
    _TEXTS.append((a, name or s[:44], size, d))
    if guard is not None:
        _GUARD.append((a, guard, name or s[:44]))
    return a

def rect(ax, x0, y0, x1, y1, fc="none", ec=INK, lw=LW_BOX, ls="-", z=2):
    """Sharp-cornered rectangle.  Rounded tinted cards are slide furniture and
    appear in none of the published schematics; RelaxNet's boxes are unfilled
    sharp rectangles at 0.59 pt."""
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc,
                           edgecolor=ec, linewidth=lw, linestyle=ls,
                           zorder=z, joinstyle="miter"))
    return (x0, y0, x1, y1)

def wire(ax, p0, p1, lw=LW_WIRE, ls="-", color=INK, head=5.0, z=3):
    """Straight signal wire with a plain solid head."""
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=head, linewidth=lw,
        color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=z,
        joinstyle="miter", capstyle="butt"))

def polyline(ax, pts, lw=LW_WIRE, ls="-", color=INK, z=3):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, lw=lw, ls=ls, color=color, zorder=z,
            solid_joinstyle="miter", solid_capstyle="butt")

def brace(ax, x0, x1, y, depth=5.0, up=False, lw=LW_BRACE, color=INK, z=3):
    """Curly brace spanning [x0, x1] with its tip at `depth` from `y`.

    Returns the tip (x, y).  Four quadratic arcs and two straight runs -- the
    standard construction -- rather than an annotation arrowstyle, so the tip
    lands exactly where the tag is placed.
    """
    s = 1.0 if up else -1.0
    r = min(depth * 0.5, (x1 - x0) * 0.25)
    mid = 0.5 * (x0 + x1)
    ya, yt = y + s * r, y + s * 2 * r
    v = [(x0, y), (x0, ya), (x0 + r, ya), (mid - r, ya), (mid, ya), (mid, yt),
         (mid, ya), (mid + r, ya), (x1 - r, ya), (x1, ya), (x1, y)]
    c = [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
         MPath.CURVE3, MPath.CURVE3, MPath.CURVE3, MPath.CURVE3,
         MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
    ax.add_patch(PathPatch(MPath(v, c), fill=False, lw=lw, edgecolor=color,
                           zorder=z, joinstyle="miter", capstyle="butt"))
    return mid, yt

_RENDER: dict = {}

def _renderer(fig):
    if fig not in _RENDER:
        fig.canvas.draw()
        _RENDER[fig] = fig.canvas.get_renderer()
    return _RENDER[fig]

def measure(fig, ax, artist):
    """Data-coordinate (width, height) of a text artist.  The axes are 1:1 with
    points, so these ARE points."""
    bb = artist.get_window_extent(_renderer(fig))
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return x1 - x0, y1 - y0

def hrow(fig, ax, x, y, parts, size, guard=None, tag=""):
    """Lay `parts` left to right at (x, y), MEASURING each as it goes.

    Returns the (x0, x1) span of every part, so a brace can be drawn under the
    exact horizontal extent of a sub-expression instead of at a guessed offset.
    """
    spans, cur = [], x
    for i, p in enumerate(parts):
        a = T(ax, cur, y, p, size=size, guard=guard, name=f"{tag}{i}")
        w, _ = measure(fig, ax, a)
        spans.append((cur, cur + w))
        cur += w
    return spans

def fit_size(fig, ax, parts, avail, nominal, floor=MATH_MIN):
    """Largest base size (0.25 pt steps) at which `parts` fit in `avail`.

    A long display equation is the one thing in this figure that can force the
    type down, so the trade is made explicitly and reported, never absorbed by
    silently shrinking a font.  Falling below `floor` is a build failure, not a
    smaller figure: shorten the equation instead.
    """
    probe = ax.text(0, -1e4, "".join(parts), fontsize=nominal)
    w = measure(fig, ax, probe)[0]
    probe.remove()
    size = nominal if w <= avail else nominal * avail / w
    size = int(size * 4) / 4.0
    if size < floor:
        raise SystemExit(
            f"display equation needs {w * floor / nominal:.1f} pt of width at "
            f"the {floor} pt floor but only {avail:.1f} pt is available -- "
            f"shorten it, do not shrink it")
    return min(size, nominal)

#: Only draw a leader when the collision resolver moved a tag FARTHER than
#: this.  At the 2 pt threshold the first cut used, a pair of tags that merely
#: brushed each other acquired two stub leaders that dropped between the words
#: of the tag they pointed at -- worse than the off-centring they documented.
LEADER_MIN = 5.0

def spread(centers, halves, lo, hi, gap=10.0):
    """Push overlapping tag columns apart, then clamp them onto the canvas.

    Returns the resolved centres.  Tag blocks are centred under their brace by
    default; a leader is drawn by the caller when the resolver had to move one.
    """
    c = list(centers)
    for _ in range(60):
        moved = False
        for i in range(len(c) - 1):
            need = halves[i] + halves[i + 1] + gap - (c[i + 1] - c[i])
            if need > 0.01:
                c[i] -= need / 2
                c[i + 1] += need / 2
                moved = True
        for i, h in enumerate(halves):
            c[i] = min(max(c[i], lo + h), hi - h)
        if not moved:
            break
    return c

# -------------------------------------------------------------------- gates
def check_bounds(fig, ax, W, H, pad=0.5):
    """Every text artist inside the canvas, and every guarded one inside its box."""
    r = _renderer(fig)
    inv = ax.transData.inverted()
    bad = []
    for artist, name, _, _ in _TEXTS:
        bb = artist.get_window_extent(r)
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        if x0 < -pad or x1 > W + pad or y0 < -pad or y1 > H + pad:
            bad.append(f"  off canvas [{x0:6.1f},{x1:6.1f}]x"
                       f"[{y0:6.1f},{y1:6.1f}]  {name!r}")
    for artist, (bx0, by0, bx1, by1), name in _GUARD:
        bb = artist.get_window_extent(r)
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        d = [bx0 - x0, x1 - bx1, by0 - y0, y1 - by1]
        if max(d) > 0.6:
            side = ["left", "right", "bottom", "top"][d.index(max(d))]
            bad.append(f"  {max(d):5.2f} pt over {side:<6s}  {name!r}")
    print("bounds OK: all %d labels inside the canvas and their boxes"
          % len(_TEXTS) if not bad else
          "BOUNDS: %d of %d labels outside:\n%s"
          % (len(bad), len(_TEXTS), "\n".join(bad)))
    return bad

def check_type_floor():
    """The PREDICTED on-page minimum, in both documents that print this file.

    Replaces a gate that read `artist.get_fontsize()` and therefore reported
    "font floor OK" while main.pdf p.6 carried 4.30 pt and summary.pdf p.2
    carried 3.84 pt.  Sub/superscripts are checked against Elsevier's 6 pt
    script floor, everything else against the 7 pt text floor.
    """
    bad = []
    for label, s in (("main.pdf   (width=\\textwidth)", S_MAIN),
                     ("summary.pdf (0.72\\textwidth)", S_DIGEST)):
        base = min(sz for _, _, sz, _ in _TEXTS) * s
        script = min(sz * SHRINK ** d for _, _, sz, d in _TEXTS)  * s
        ok = base >= FLOOR_TEXT - 1e-9 and script >= FLOOR_SCRIPT - 1e-9
        print(f"  {label}  s={s:.3f}  base min {base:5.2f} pt "
              f"(floor {FLOOR_TEXT})   script min {script:5.2f} pt "
              f"(floor {FLOOR_SCRIPT})   {'OK' if ok else 'FAIL'}")
        if not ok:
            bad.append(label)
            for a, n, sz, d in sorted(_TEXTS, key=lambda r: r[2] * SHRINK ** r[3]):
                if sz * SHRINK ** d * s < FLOOR_SCRIPT - 1e-9 or \
                        sz * s < FLOOR_TEXT - 1e-9:
                    print(f"      {sz * SHRINK ** d * s:5.2f} pt  {n!r}")
    return bad

# ======================================================================
def build(outdir: Path, mono: bool = False) -> list:
    tint = TINT_MONO if mono else TINT
    W, H = SLOT_BP, 248.5
    fig = plt.figure(figsize=(W / 72.0, H / 72.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    LEFT, RIGHT = 1.0, W - 1.0

    # ==================================================================
    # (a) THE ANSATZ.  x and mu are consumed by closed-form factors; the two
    #     coefficients -- functions of t alone -- are all that is left over.
    # ==================================================================
    y_tag_a = H - 8.0
    T(ax, LEFT, y_tag_a, "(a)", size=FS_PANEL, weight="bold", name="(a)")

    y_spine = y_tag_a - 32.0
    # index `l`, set at 0.7x of 12.5 pt hard against an opening paren, reads as
    # a second parenthesis -- under the stixsans fontset tried first,
    # `c_{m,l}(t)` came off the page as `c_{m,(}(t)`.  The serif italic `l` of
    # the "stix" set is far clearer, but the space still earns its place at
    # 8.75 pt.  Only visible at 600 dpi: no gate here can see a glyph shape.
    parts = [r"$f_m(x,\mu,t)\;=\;\sum_l\;[\;$",
             r"$c_{m,l}\,(t)$", r"$\,\cos qx$", r"$\;+\;$",
             r"$s_{m,l}\,(t)$", r"$\,\sin qx\;$", r"$]\;$", r"$P_l\,(\mu)$"]
    fs_eq = fit_size(fig, ax, parts, RIGHT - LEFT - 24.0, FS_EQ, floor=11.0)
    probe = ax.text(0, -1e4, "".join(parts), fontsize=fs_eq)
    w_eq = measure(fig, ax, probe)[0]
    probe.remove()
    x_eq = 0.5 * (W - w_eq)
    sp = hrow(fig, ax, x_eq, y_spine, parts, fs_eq, tag="ansatz")

    # The tint marks the two coefficients; the brace above says the word.
    # ±6.6 pt, not ±5.6: at 12.5 pt the parentheses of `(t)` overrun a band
    # sized to the x-height and the fill clipped them.
    for i in (1, 4):
        ax.add_patch(Rectangle((sp[i][0] - 0.8, y_spine - 6.6),
                               sp[i][1] - sp[i][0] + 1.6, 13.2,
                               facecolor=tint, edgecolor="none", zorder=1))

    # ---- above the line: what the network supplies -------------------
    # +9.0, not +6.2: the first cut put the brace shoulders ON the ascenders
    # of `c_{m,l}(t)` -- invisible to both gates, obvious in the render.
    top = y_spine + 9.0
    t1 = brace(ax, sp[1][0], sp[1][1], top, 4.4, up=True)
    t2 = brace(ax, sp[4][0], sp[4][1], top, 4.4, up=True)
    polyline(ax, [(t1[0], t1[1]), (t2[0], t2[1])], lw=LW_BRACE)
    T(ax, 0.5 * (t1[0] + t2[0]), t1[1] + 7.0,
      r"learned — functions of $t$ alone; exact cosine IC at $t=0$",
      ha="center", name="tag-learned")

    # ---- below the line: what never reaches the network --------------
    bot = y_spine - 8.6
    b1 = brace(ax, sp[2][0], sp[2][1], bot, 4.0)
    b2 = brace(ax, sp[5][0], sp[5][1] - 1.2, bot, 4.0)
    polyline(ax, [(b1[0], b1[1]), (b2[0], b2[1])], lw=LW_BRACE)
    b3 = brace(ax, sp[7][0], sp[7][1], bot, 4.0)

    tags_a = [
        (0.5 * (b1[0] + b2[0]),
         [r"closed form in $x$", r"one harmonic, $q=2\pi/L$",
          "periodic BC exact"]),
        (b3[0],
         [r"closed form in $\mu$", r"Legendre, $l\leq8$",
          "moments algebraic"]),
    ]
    halves = []
    for _, lines in tags_a:
        p = ax.text(0, -1e4, "", fontsize=FS_TAG)
        wmax = 0.0
        for ln in lines:
            p.set_text(ln)
            wmax = max(wmax, measure(fig, ax, p)[0])
        p.remove()
        halves.append(0.5 * wmax)
    xs = spread([c for c, _ in tags_a], halves, LEFT, RIGHT, gap=16.0)
    y_tags = b1[1] - 10.5           # 10.5, not 7.4: the joining rail sat on
    for (c0, lines), c, half in zip(tags_a, xs, halves):   # the first tag line
        if abs(c - c0) > LEADER_MIN:                # resolver moved it: lead
            polyline(ax, [(c0, b1[1]), (c0, b1[1] - 3.4), (c, b1[1] - 3.4),
                          (c, y_tags + 5.6)], lw=0.45)
        for j, ln in enumerate(lines):
            T(ax, c, y_tags - LEAD * j, ln, ha="center", name=f"a-tag{j}")
    y_a_bot = y_tags - LEAD * 2 - 5.0

    # ==================================================================
    # (b) THE LOOP.  One arrow carries `t` into the network and nothing else.
    # ==================================================================
    y_b = y_a_bot - 10.0
    T(ax, LEFT, y_b, "(b)", size=FS_PANEL, weight="bold", name="(b)")

    # The gradient returns on a lane OUTSIDE the drawing, at x = LANE, and all
    # of panel (b) starts at XL.  The first cut ran the return straight down
    # the middle of the residual box and through both note lines, with its
    # label on top of the text -- containment-clean, and unreadable.
    XL = 28.0                       # panel (b) content margin
    LANE = 15.0                     # the gradient return lane, left of XL

    # ---- the network: `t` in, two coefficient generators out ---------
    y_net = y_b - 13.0
    nb = (XL + 20.0, y_net - 9.0, XL + 68.0, y_net + 9.0)
    T(ax, XL, y_net, r"$t$", size=MATH_MIN, name="t-in")
    wire(ax, (XL + 7.5, y_net), (nb[0] - 1.5, y_net))
    rect(ax, *nb, fc=tint)
    T(ax, 0.5 * (nb[0] + nb[2]), y_net, r"$\mathrm{NN}_\theta$",
      size=MATH_MIN, ha="center", guard=nb, name="NN")
    wire(ax, (nb[2] + 1.5, y_net), (nb[2] + 19.0, y_net))
    T(ax, nb[2] + 22.0, y_net, r"$\hat{A}(t),\;r_{m,l}\,(t)$", size=MATH_MIN,
      name="net-out")

    # ---- coefficient assembly: free kernel, learned carrier/correction --
    y_asm = y_net - 32.0
    # Same thin-space rule as the ansatz: `l` hard against `(` is unreadable.
    qs = [r"$a_{m,l}\,(t)\;=\;$",
          r"$a^{\mathrm{free}}_{m,l}\,(t)$", r"$\;+\;$",
          r"$a^{\mathrm{carrier}}_{m,l}\,(t;\hat A)$",
          r"$\;+\;$",
          r"$g_m(t)\,\varepsilon_m C_m\,r_{m,l}\,(t)$"]
    fs_asm = fit_size(fig, ax, qs, RIGHT - XL - 4.0, FS_MATH)
    ea = hrow(fig, ax, XL, y_asm, qs, fs_asm, tag="asm")
    for i in (3, 5):
        ax.add_patch(Rectangle((ea[i][0] - 0.8, y_asm - 5.8),
                               ea[i][1] - ea[i][0] + 1.6, 11.6,
                               facecolor=tint, edgecolor="none", zorder=1))
    # The network's two outputs drop into the assembly, so the arrow starts
    # under the OUTPUT label rather than under the box: the box's own bottom
    # edge is where the gradient comes back, and one wire per direction.
    # One vertical spine at the label's midpoint carries both drops.
    x_drop = nb[2] + 60.0
    wire(ax, (x_drop, y_net - 10.5), (x_drop, y_asm + 8.0))

    y_ab = y_asm - 6.4
    tips, tags_b = [], ["attenuated free streaming",
                        r"learned carrier $\hat{A}(t)$", "learned, all orders"]
    for i in (1, 3, 5):
        tips.append(brace(ax, ea[i][0], ea[i][1], y_ab, 4.0))
    hb = []
    for s in tags_b:
        p = ax.text(0, -1e4, s, fontsize=FS_TAG)
        hb.append(0.5 * measure(fig, ax, p)[0])
        p.remove()
    xb = spread([t[0] for t in tips], hb, XL, RIGHT, gap=10.0)  # clear the lane
    for (tx, ty), c, s in zip(tips, xb, tags_b):
        if abs(c - tx) > LEADER_MIN:
            # -5.0, not -6.4: the leader used to stop 2.5 pt INSIDE the tag it
            # points at, striking through the word.  Both gates passed it.
            polyline(ax, [(tx, ty), (tx, ty - 2.6), (c, ty - 2.6),
                          (c, ty - 5.0)], lw=0.45)
        T(ax, c, ty - 10.0, s, ha="center", name=f"b-tag-{s[:12]}")
    # -18.0: at -15.2 the drop arrow into the residual box started 0.7 pt
    # under the tag above it and read as touching the word.
    y_asm_bot = tips[0][1] - 18.0

    # ---- residual and loss: the only things that reach the gradient ---
    y_rt = y_asm_bot - 13.0
    res_eq = "Galerkin BTE rows + energy-balance row"
    # The box is SIZED FROM its widest line, the way JCP sizes a schematic to
    # its content (RelaxNet's Fig. 1 is half a column wide).  A hard-coded
    # right edge left 70 pt of empty box and squeezed the arbiter.
    probe = ax.text(0, -1e4, res_eq, fontsize=FS_MATH)
    XR = XL + 14.0 + measure(fig, ax, probe)[0]
    probe.remove()
    fs_res = FS_MATH
    y_rb = y_rt - 34.0
    rb = rect(ax, XL, y_rb, XR, y_rt)
    T(ax, XL + 7.0, y_rt - 11.0, res_eq, size=fs_res, guard=rb, name="R")
    y_loss = y_rt - 25.0
    T(ax, XL + 7.0, y_loss,
      "scaled loss; automatic differentiation",
      size=FS_MATH, guard=rb, name="L")
    wire(ax, (x_drop, y_asm_bot + 0.5), (x_drop, y_rt + 1.5))
    # The signal named on the wire, RelaxNet-style: it is the one place the
    # generic coefficient `a` of the assembly meets the `c`, `s` of the ansatz
    # and of the residual, and without it the reader has to infer the link.
    T(ax, x_drop + 3.5, 0.5 * (y_asm_bot + y_rt), r"$a\in\{c,s\}$",
      size=MATH_MIN, va="center", name="parity")

    # ---- the arbiter: outside the loop, and only a window -------------
    ab = rect(ax, RIGHT - 74.0, y_rb, RIGHT, y_rt, ls=(0, (2.6, 1.8)))
    T(ax, 0.5 * (ab[0] + ab[2]), y_rt - 9.5, "DOM reference", ha="center",
      guard=ab, name="arb")
    T(ax, 0.5 * (ab[0] + ab[2]), y_rt - 21.0,
      r"$t_{\mathrm{end}}=7/\gamma_{\mathrm{DOM}}$", ha="center", guard=ab,
      name="arb-t")
    T(ax, 0.5 * (ab[0] + ab[2]), y_rt - 30.0, "window only", ha="center",
      guard=ab, name="arb-w")
    # The reference selects the sampled interval, not a target in the loss.
    T(ax, RIGHT - 38.0, y_rt + 10.0, "time interval", ha="center", name="window")
    wire(ax, (RIGHT - 38.0, y_rt + 1.5), (RIGHT - 38.0, y_rt + 6.0),
         ls=(0, (2.2, 1.6)), head=4.5)

    # ---- the notes that carry the argument ---------------------------
    y_n = y_rb - 9.0
    T(ax, XL, y_n,
      r"Galerkin rows through $l=8$; free-streaming closure at $l=9$", name="note-rows")
    T(ax, XL, y_n - LEAD,
      "time collocation on the reference-set interval; no solution labels or waypoints",
      name="note-macro")

    # ---- the gradient, returned as plain black straight segments ------
    # Out of the loss line, left onto the lane, up the outside, and into the
    # BOTTOM of the network box -- so it never crosses the `t` input wire and
    # never enters the note block.  RelaxNet draws its feedback the same way:
    # plain black straight segments around the outside, no curve, no colour.
    y_up = y_net - 17.0
    polyline(ax, [(XL - 1.0, y_loss), (LANE, y_loss), (LANE, y_up),
                  (0.5 * (nb[0] + nb[2]), y_up)])
    wire(ax, (0.5 * (nb[0] + nb[2]), y_up),
         (0.5 * (nb[0] + nb[2]), nb[1] - 1.5))
    T(ax, LANE + 2.6, 0.5 * (y_loss + y_up), r"$\nabla_\theta\mathcal{L}$",
      size=MATH_MIN, ha="center", va="bottom", rotation=90, name="grad")

    # ==================================================================
    bad = check_bounds(fig, ax, W, H)
    margin = y_n - LEAD - 5.5           # under the last note line
    print(f"canvas {W:.2f} x {H:.2f} pt   ({W * 25.4 / 72:.1f} x "
          f"{H * 25.4 / 72:.1f} mm, aspect {W / H:.2f}:1) "
          f"bottom margin {margin:.1f} pt")
    print(f"equation bases: ansatz {fs_eq} pt, assembly {fs_asm} pt, "
          f"residual {fs_res} pt")
    small = check_type_floor()
    assert 1.0 < margin < 9.0, (
        f"canvas mis-sized: bottom margin {margin:.1f} pt -- set H to "
        f"{H - margin + 4.0:.1f}")

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = outdir / f"fig1_schematic.{ext}"
        fig.savefig(out, dpi=400 if ext == "png" else None)
        print(f"wrote {out}")
    plt.close(fig)
    return bad + small

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=FIG_DIR)
    ap.add_argument("--mono", action="store_true",
                    help="force the single tint to grey (greyscale check)")
    a = ap.parse_args()
    sys.exit(1 if build(a.outdir, a.mono) else 0)

if __name__ == "__main__":
    main()
