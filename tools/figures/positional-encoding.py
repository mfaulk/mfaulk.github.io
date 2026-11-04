"""Generate the sinusoidal positional-encoding figure as SVG.

Three panels, stacked: the encoding table as a heatmap, three dimension pairs
plotted against position, and one (sin, cos) pair on the unit circle, where a
fixed offset k is a rotation by an angle that does not depend on t.

The panels share one table, so the curves and the circle cannot disagree with
the heatmap they are drawn from.
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _panels import FS, FS_SMALL, GREY, MIX, Canvas, r, svg  # noqa: E402

OUT = (pathlib.Path(__file__).resolve().parents[2]
       / "assets" / "images" / "positional-encoding.svg")

D_MODEL = 64
MAX_T = 40


def wavelength(i):
    """Wavelength in positions of the sinusoid held by dimension pair i."""
    return 2 * math.pi * 10000 ** (2 * i / D_MODEL)


def angle(t, i):
    return t / 10000 ** (2 * i / D_MODEL)


def pe(t, j):
    """Entry (t, j) of the encoding table, j indexing d_model."""
    a = angle(t, j // 2)
    return math.sin(a) if j % 2 == 0 else math.cos(a)


TABLE = [[pe(t, j) for j in range(D_MODEL)] for t in range(MAX_T)]

# -- invariants the prose leans on ---------------------------------------
# Values stay in [-1, 1] however long the sequence, so they do not swamp the
# embedding they are added to.
assert all(-1.0 <= v <= 1.0 for row in TABLE for v in row)

# Each (sin, cos) pair is a unit vector, which is what makes panel 3 a circle.
for t in range(MAX_T):
    for i in range(D_MODEL // 2):
        norm = math.hypot(TABLE[t][2 * i], TABLE[t][2 * i + 1])
        assert abs(norm - 1.0) < 1e-12, norm

# Wavelengths are geometric in i: the three curves in panel 2 are a decade apart.
CURVES = (0, 8, 16)
ratios = [wavelength(b) / wavelength(a) for a, b in zip(CURVES, CURVES[1:])]
assert all(abs(x - 10.0) < 1e-9 for x in ratios), ratios
assert abs(wavelength(0) - 2 * math.pi) < 1e-9

# A fixed offset k rotates the pair by k / 10000^(2i/d_model) from any t, which
# is the claim the third panel makes. Checked as the angle between unit vectors.
CIRCLE_PAIR = 8
K = 9
for base in (3, 18):
    u = (TABLE[base][2 * CIRCLE_PAIR], TABLE[base][2 * CIRCLE_PAIR + 1])
    v = (TABLE[base + K][2 * CIRCLE_PAIR], TABLE[base + K][2 * CIRCLE_PAIR + 1])
    between = math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1])))
    assert abs(between - angle(K, CIRCLE_PAIR)) < 1e-12, between

# -- panel 1: the table --------------------------------------------------
CELL = 7
HX0, HY0 = 96, 86
HW, HH = D_MODEL * CELL, MAX_T * CELL

LO = (0xF0, 0xED, 0xF6)   # the -1 end, FILL[MIX]
HI = (0x4A, 0x3C, 0x72)   # the +1 end, a shade under MIX for contrast


def shade(v):
    t = (v + 1.0) / 2.0
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(LO, HI))


c = Canvas()
c.text(HX0, HY0 - 26, "the table PE(t, ·)", anchor="start", fs=FS, weight="600")

for ti, row in enumerate(TABLE):
    for j, v in enumerate(row):
        c.add(f'<rect x="{r(HX0 + j * CELL)}" y="{r(HY0 + ti * CELL)}" '
              f'width="{CELL}" height="{CELL}" fill="{shade(v)}"/>')
c.note(HX0, HY0)
c.note(HX0 + HW, HY0 + HH)
c.add(f'<rect x="{r(HX0)}" y="{r(HY0)}" width="{r(HW)}" height="{r(HH)}" fill="none" '
      f'stroke="{GREY}" stroke-width="1" opacity="0.4"/>')

for j in (0, 16, 32, 48, 63):
    c.text(HX0 + (j + 0.5) * CELL, HY0 + HH + 16, str(j), fs=FS_SMALL, opacity="0.85")
c.text(HX0 + HW / 2, HY0 + HH + 36, "embedding dimension", fs=FS_SMALL, opacity="0.75")

for t in (0, 10, 20, 30):
    c.text(HX0 - 10, HY0 + t * CELL + 5, str(t), anchor="end", fs=FS_SMALL, opacity="0.85")
cy_axis = HY0 + HH / 2
c.add(f'<text x="{r(HX0 - 40)}" y="{r(cy_axis)}" text-anchor="middle" font-size="{FS_SMALL}" '
      f'fill="currentColor" opacity="0.75" '
      f'transform="rotate(-90 {r(HX0 - 40)} {r(cy_axis)})">position t</text>')
c.note(HX0 - 52, cy_axis - 44)

# Colour bar: 48 stacked rects rather than a gradient, so the output stays
# byte-identical between runs without depending on renderer gradient handling.
BAR_X, BAR_Y, BAR_W, BAR_H = HX0 + HW + 26, HY0, 14, 120
STEPS = 48
for s in range(STEPS):
    v = 1.0 - 2.0 * s / (STEPS - 1)
    c.add(f'<rect x="{r(BAR_X)}" y="{r(BAR_Y + s * BAR_H / STEPS)}" width="{BAR_W}" '
          f'height="{r(BAR_H / STEPS + 0.4)}" fill="{shade(v)}"/>')
c.add(f'<rect x="{r(BAR_X)}" y="{r(BAR_Y)}" width="{BAR_W}" height="{r(BAR_H)}" fill="none" '
      f'stroke="{GREY}" stroke-width="1" opacity="0.4"/>')
c.note(BAR_X, BAR_Y)
c.note(BAR_X + BAR_W, BAR_Y + BAR_H)
for lab, yy in (("+1", BAR_Y + 4), ("0", BAR_Y + BAR_H / 2 + 4), ("−1", BAR_Y + BAR_H)):
    c.text(BAR_X + BAR_W + 6, yy, lab, anchor="start", fs=FS_SMALL, opacity="0.85")

# -- panel 2: three dimension pairs --------------------------------------
BX0, BW = 96, 380
BY_MID, BY_AMP = 516, 60
CURVE_COL = {0: "#3E3266", 8: MIX, 16: "#8F82B8"}

c.text(BX0, BY_MID - BY_AMP - 32, "three dimension pairs", anchor="start", fs=FS, weight="600")

c.add(f'<line x1="{r(BX0)}" y1="{r(BY_MID)}" x2="{r(BX0 + BW)}" y2="{r(BY_MID)}" '
      f'stroke="{GREY}" stroke-width="1" opacity="0.35"/>')
for v, lab in ((1.0, "+1"), (0.0, "0"), (-1.0, "−1")):
    c.text(BX0 - 10, BY_MID - v * BY_AMP + 5, lab, anchor="end", fs=FS_SMALL, opacity="0.85")

for i in CURVES:
    pts = []
    steps = 400
    for s in range(steps + 1):
        t = MAX_T * s / steps
        x = BX0 + t / MAX_T * BW
        y = BY_MID - math.sin(angle(t, i)) * BY_AMP
        pts.append(f"{r(x)},{r(y)}")
    c.add(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{CURVE_COL[i]}" '
          f'stroke-width="1.8"/>')
    end_y = BY_MID - math.sin(angle(MAX_T, i)) * BY_AMP
    c.text(BX0 + BW + 8, end_y + 4, f"2i = {i}, λ ≈ {wavelength(i):.0f}",
           CURVE_COL[i], anchor="start", fs=FS_SMALL)
c.note(BX0, BY_MID - BY_AMP)
c.note(BX0 + BW, BY_MID + BY_AMP)

for t in (0, 10, 20, 30, 40):
    c.text(BX0 + t / MAX_T * BW, BY_MID + BY_AMP + 20, str(t), fs=FS_SMALL, opacity="0.85")
c.text(BX0 + BW / 2, BY_MID + BY_AMP + 40, "position t", fs=FS_SMALL, opacity="0.75")

# -- panel 3: one pair on the unit circle --------------------------------
CR = 66
CCX, CCY = BX0 + CR + 34, BY_MID + BY_AMP + 176

c.text(BX0, CCY - CR - 50, f"one (sin, cos) pair, 2i = {CIRCLE_PAIR}", anchor="start", fs=FS,
       weight="600")

c.add(f'<circle cx="{r(CCX)}" cy="{r(CCY)}" r="{r(CR)}" fill="none" stroke="{GREY}" '
      f'stroke-width="1" opacity="0.45"/>')
for d in (f"M {r(CCX - CR - 14)} {r(CCY)} H {r(CCX + CR + 14)}",
          f"M {r(CCX)} {r(CCY - CR - 14)} V {r(CCY + CR + 14)}"):
    c.add(f'<path d="{d}" fill="none" stroke="{GREY}" stroke-width="1" opacity="0.3"/>')
c.note(CCX - CR - 16, CCY - CR - 16)
c.note(CCX + CR + 16, CCY + CR + 16)

HIGHLIGHT = {3: "t", 3 + K: "t + k", 18: "t′", 18 + K: "t′ + k"}


def point(t):
    a = angle(t, CIRCLE_PAIR)
    return CCX + math.sin(a) * CR, CCY - math.cos(a) * CR


for t in range(0, MAX_T, 3):
    px, py = point(t)
    if t in HIGHLIGHT:
        continue
    c.add(f'<circle cx="{r(px)}" cy="{r(py)}" r="2.6" fill="{GREY}" opacity="0.5"/>')

for t, lab in HIGHLIGHT.items():
    px, py = point(t)
    c.add(f'<circle cx="{r(px)}" cy="{r(py)}" r="4.6" fill="{MIX}"/>')
    # Push the label radially outward so it never sits on the circle.
    a = angle(t, CIRCLE_PAIR)
    lx, ly = CCX + math.sin(a) * (CR + 20), CCY - math.cos(a) * (CR + 20)
    anchor = "start" if math.sin(a) > 0.1 else ("end" if math.sin(a) < -0.1 else "middle")
    c.text(lx, ly + 4, lab, MIX, anchor=anchor, fs=FS_SMALL, weight="600")

# Two arcs of equal angle at different places on the circle: same k, same
# rotation, whatever t it starts from.
for base in (3, 18):
    ax, ay = point(base)
    bx, by = point(base + K)
    ar = CR - 12
    sx = CCX + (ax - CCX) * ar / CR
    sy = CCY + (ay - CCY) * ar / CR
    ex = CCX + (bx - CCX) * ar / CR
    ey = CCY + (by - CCY) * ar / CR
    c.add(f'<path d="M {r(sx)} {r(sy)} A {r(ar)} {r(ar)} 0 0 1 {r(ex)} {r(ey)}" fill="none" '
          f'stroke="{MIX}" stroke-width="1.6" marker-end="url(#arrow)" opacity="0.9"/>')
    mid = angle(base + K / 2, CIRCLE_PAIR)
    c.text(CCX + math.sin(mid) * (ar - 20), CCY - math.cos(mid) * (ar - 20) + 4,
           "Δθ", MIX, fs=FS_SMALL, weight="600")

c.text(CCX + CR + 22, CCY + 4, "PE(t, 2i)", anchor="start", fs=FS_SMALL, opacity="0.75")
c.text(CCX - 12, CCY - CR - 22, "PE(t, 2i+1)", anchor="end", fs=FS_SMALL, opacity="0.75")
# Sits below the x-axis label rather than beside it: at the width the post
# renders this at, the two ran into each other.
c.text(CCX + CR + 150, CCY + 52, f"k = {K}", anchor="middle", fs=FS_SMALL, opacity="0.85")
c.text(CCX + CR + 150, CCY + 72, f"Δθ = k / 10000^(2i/d) ≈ {angle(K, CIRCLE_PAIR):.2f}",
       anchor="middle", fs=FS_SMALL, opacity="0.85")

OUT.write_text(svg(
    c,
    "Sinusoidal positional encoding",
    "Sinusoidal positional encoding in three panels. A heatmap of the encoding table shows "
    "banded stripes that are narrow in the low embedding dimensions and wide in the high ones. "
    "Three sinusoids plotted against position have wavelengths of roughly 6, 63 and 628 "
    "positions, a decade apart. One sine-cosine dimension pair is plotted on the unit circle, "
    "where successive positions step around it by a constant angle, so a fixed offset k is the "
    "same rotation wherever it starts.",
))
print(f"wrote {OUT}")
