"""Generate the QKV projection figure as SVG.

Words -> embeddings -> three learned projections -> queries, keys, values.
Colours match the dot-product attention figure: queries blue, keys orange,
values green.
"""

import pathlib

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets" / "images" / "qkv-projections.svg"


WORDS = ["'How'", "'are'", "'you'", "'doing'", "'?'"]
ROWS = [
    ("W", "q", "Q", "#1F5CA8"),
    ("W", "k", "K", "#B4531A"),
    ("W", "v", "V", "#2F7A4F"),
]

X_C = "#5B4B8A"    # the shared embedding matrix
CW, CH = 13, 17    # cell width / height for the T x d matrices
SQ = 13            # square cell for the d x d projections
T, D = len(WORDS), 6

XW, XH = D * CW, T * CH
WW = WH = D * SQ

WORD_X, X_X = 16, 92
W_X, O_X = 268, 436
ROW_CY = [84, 214, 344]
X_CY = ROW_CY[1]

WIDTH = O_X + XW + 92
HEIGHT = ROW_CY[-1] + XH // 2 + 46


def shade(r, c, seed):
    return 0.20 + 0.70 * (((r * 7 + c * 13 + seed * 29) % 17) / 16)


def grid(x, y, cols, rows, cw, ch, colour, seed=0):
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(f'<rect x="{x + c * cw}" y="{y + r * ch}" width="{cw}" height="{ch}" '
                       f'fill="{colour}" fill-opacity="{shade(r, c, seed):.2f}"/>')
    out.append(f'<rect x="{x}" y="{y}" width="{cols * cw}" height="{rows * ch}" fill="none" '
               f'stroke="{colour}" stroke-width="1.3"/>')
    return out


def dim(x, y, a, b):
    return (f'<text x="{x}" y="{y}" text-anchor="middle" font-size="12" fill="currentColor" '
            f'opacity="0.65">{a}<tspan font-size="9" dy="2">model</tspan>'
            f'<tspan font-size="12" dy="-2"> &#215; {b}</tspan>'
            f'<tspan font-size="9" dy="2">model</tspan></text>') if a == "d" else (
           f'<text x="{x}" y="{y}" text-anchor="middle" font-size="12" fill="currentColor" '
           f'opacity="0.65">T &#215; d<tspan font-size="9" dy="2">model</tspan></text>')


o = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" '
    f'height="{HEIGHT}" font-family="system-ui, -apple-system, Segoe UI, Helvetica, sans-serif" '
    f'role="img" aria-label="The words How, are, you, doing and the question mark become rows of an '
    f'embedding matrix X, which feeds three separate learned projection matrices W_q, W_k and W_v, '
    f'producing the queries, keys and values.">',
    '<title>Queries, keys and values are projections of the same embeddings</title>',
    '<defs>',
]
for _, sub, _, col in ROWS:
    o.append(f'<marker id="a{sub}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" '
             f'markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{col}"/></marker>')
o.append(f'<marker id="ax" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
         f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{X_C}"/></marker>')
o.append('</defs>')

# words, one per row of the embedding matrix
xy = X_CY - XH // 2
for i, word in enumerate(WORDS):
    cy = xy + i * CH + CH // 2 + 5
    o.append(f'<text x="{X_X - 14}" y="{cy}" text-anchor="end" font-size="14" '
             f'fill="#1a2a3a">{word}</text>')

o += grid(X_X, xy, D, T, CW, CH, X_C, 2)
o.append(f'<text x="{X_X + XW // 2}" y="{xy - 11}" text-anchor="middle" font-size="15" '
         f'fill="{X_C}" font-weight="600">X</text>')
o.append(f'<text x="{X_X + XW // 2}" y="{xy + XH + 19}" text-anchor="middle" font-size="12" '
         f'fill="currentColor" opacity="0.8">token embeddings</text>')
o.append(dim(X_X + XW // 2, xy + XH + 35, "T", "d"))

for i, (w, sub, out_name, col) in enumerate(ROWS):
    cy = ROW_CY[i]
    wy = cy - WH // 2
    oy = cy - XH // 2
    mid = (X_X + XW + W_X) // 2

    o.append(f'<path d="M {mid} {X_CY} V {cy} H {W_X - 7}" fill="none" stroke="{col}" '
             f'stroke-width="1.4" marker-end="url(#a{sub})"/>')
    o.append(f'<path d="M {W_X + WW + 5} {cy} H {O_X - 7}" fill="none" stroke="{col}" '
             f'stroke-width="1.4" marker-end="url(#a{sub})"/>')

    o += grid(W_X, wy, D, D, SQ, SQ, col, i * 3)
    o.append(f'<text x="{W_X + WW // 2}" y="{wy - 10}" text-anchor="middle" font-size="15" '
             f'fill="{col}" font-weight="600">{w}<tspan font-size="11" dy="3">{sub}</tspan></text>')
    o.append(dim(W_X + WW // 2, wy + WH + 18, "d", "d"))

    o += grid(O_X, oy, D, T, CW, CH, col, i * 5 + 1)
    o.append(f'<text x="{O_X + XW + 12}" y="{cy + 5}" font-size="16" fill="{col}" '
             f'font-weight="600">{out_name}</text>')
    o.append(dim(O_X + XW // 2, oy + XH + 19, "T", "d"))

# shared trunk: one embedding matrix feeding all three branches
o.append(f'<path d="M {X_X + XW + 5} {X_CY} H {(X_X + XW + W_X) // 2}" fill="none" '
         f'stroke="{X_C}" stroke-width="1.6"/>')
o.append('</svg>')

open(OUT, "w").write("\n".join(o) + "\n")
print(f"wrote {OUT} ({WIDTH}x{HEIGHT})")
