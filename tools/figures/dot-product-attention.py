"""Generate the dot-product attention figure as SVG.

Shows the whole mechanism for one query: compare the query against each key
to get weights, then take the weighted sum of the values.
"""

import pathlib

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets" / "images" / "dot-product-attention.svg"


WORDS = ["I", "am", "eating", "an", "apple"]
ALPHA = [0.05, 0.08, 0.46, 0.07, 0.34]
assert abs(sum(ALPHA) - 1.0) < 1e-9, sum(ALPHA)

Q_C = "#1F5CA8"   # query
K_C = "#B4531A"   # keys
V_C = "#2F7A4F"   # values
O_C = "#55606b"   # output

CELLH, CELLW, NCELL = 11, 20, 4
GLYPH_H = CELLH * NCELL

LEFT = 76          # gutter for the row labels
COLW = 104
NCOL = len(WORDS)
RIGHT_W = 112      # "=" plus the output glyph

WIDTH = LEFT + NCOL * COLW + RIGHT_W
QY, KY, AY, VY, WY = 18, 116, 182, 208, 276
HEIGHT = WY + 16

# per-cell shading so each glyph reads as a vector of numbers
SHADE = [
    [0.95, 0.45, 0.75, 0.25], [0.35, 0.85, 0.30, 0.60], [0.70, 0.30, 0.90, 0.45],
    [0.25, 0.65, 0.40, 0.85], [0.80, 0.50, 0.25, 0.70], [0.55, 0.75, 0.60, 0.35],
]


def glyph(x, y, colour, shades):
    out = []
    for r in range(NCELL):
        out.append(f'<rect x="{round(x,1)}" y="{y + r * CELLH}" width="{CELLW}" height="{CELLH}" '
                   f'fill="{colour}" fill-opacity="{shades[r]:.2f}"/>')
    out.append(f'<rect x="{round(x,1)}" y="{y}" width="{CELLW}" height="{GLYPH_H}" fill="none" '
               f'stroke="{colour}" stroke-width="1.2"/>')
    return out


colx = [LEFT + i * COLW for i in range(NCOL)]
qx = LEFT + (NCOL * COLW) / 2 - CELLW / 2        # query centred over the row

o = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" '
    f'height="{HEIGHT}" font-family="system-ui, -apple-system, Segoe UI, Helvetica, sans-serif" '
    f'role="img" aria-label="A single query is compared against the key of each word in I am eating '
    f'an apple, producing weights alpha. The output is the sum of the value vectors scaled by those '
    f'weights, dominated by eating and apple.">',
    '<title>Dot-product attention for a single query</title>',
    '<defs><marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
    f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{Q_C}"/></marker></defs>',
]

# query -> each key, dashed
for i in range(NCOL):
    o.append(f'<path d="M {round(qx + CELLW / 2,1)} {QY + GLYPH_H} '
             f'L {round(colx[i] + CELLW / 2,1)} {KY - 6}" stroke="{Q_C}" stroke-width="1.2" '
             f'stroke-dasharray="4 3" fill="none" marker-end="url(#ar)" opacity="0.75"/>')

o += glyph(qx, QY, Q_C, SHADE[0])
o.append(f'<text x="{round(qx + CELLW + 10,1)}" y="{QY + 28}" font-size="14" fill="currentColor" '
         f'opacity="0.85">query</text>')

# row labels
for y, txt in ((KY + 28, "keys"), (VY + 28, "values")):
    o.append(f'<text x="{LEFT - 18}" y="{y}" text-anchor="end" font-size="14" fill="currentColor" '
             f'opacity="0.85">{txt}</text>')

for i, word in enumerate(WORDS):
    x = colx[i]
    o += glyph(x, KY, K_C, SHADE[i + 1])
    o += glyph(x, VY, V_C, SHADE[(i + 2) % len(SHADE)])
    # weight between key and value
    o.append(f'<text x="{round(x + CELLW / 2,1)}" y="{AY}" text-anchor="middle" font-size="14" '
             f'fill="{K_C}">&#945;<tspan font-size="10" dy="3">{i+1}</tspan>'
             f'<tspan font-size="14" dy="-3"> = {ALPHA[i]:.2f}</tspan></text>')
    # "x alpha" next to the value, then a plus between columns
    o.append(f'<text x="{round(x + CELLW + 7,1)}" y="{VY + 28}" font-size="13" fill="currentColor" '
             f'opacity="0.75">&#215;&#945;<tspan font-size="10" dy="3">{i+1}</tspan></text>')
    if i < NCOL - 1:
        o.append(f'<text x="{round(x + COLW - 14,1)}" y="{VY + 28}" font-size="15" '
                 f'fill="currentColor" opacity="0.6">+</text>')
    o.append(f'<text x="{round(x + CELLW / 2,1)}" y="{WY}" text-anchor="middle" font-size="15" '
             f'fill="#1a2a3a">{word}</text>')

# = output
ex = colx[-1] + COLW - 6
o.append(f'<text x="{round(ex,1)}" y="{VY + 28}" font-size="16" fill="currentColor" '
         f'opacity="0.7">=</text>')
ox = ex + 26
o += glyph(ox, VY, O_C, SHADE[3])
o.append(f'<text x="{round(ox + CELLW + 8,1)}" y="{VY + 28}" font-size="14" fill="currentColor" '
         f'opacity="0.85">output</text>')
o.append('</svg>')

open(OUT, "w").write("\n".join(o) + "\n")
print(f"wrote {OUT} ({WIDTH}x{HEIGHT})")
