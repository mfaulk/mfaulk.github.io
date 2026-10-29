"""Generate a self-attention weight heatmap as SVG."""

import pathlib

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets" / "images" / "self-attention-weights.svg"


TOKENS = ["How", "are", "you", "doing", "?"]

# Row-stochastic: each query's weights over keys sum to 1.
W = [
    [0.55, 0.20, 0.10, 0.10, 0.05],
    [0.22, 0.45, 0.18, 0.10, 0.05],
    [0.08, 0.15, 0.42, 0.28, 0.07],
    [0.06, 0.10, 0.30, 0.44, 0.10],
    [0.10, 0.12, 0.20, 0.33, 0.25],
]
for r in W:
    assert abs(sum(r) - 1.0) < 1e-9, sum(r)

CELL = 62
X0, Y0 = 168, 92
N = len(TOKENS)
WIDTH = X0 + N * CELL + 28
HEIGHT = Y0 + N * CELL + 18

LO = (0xEC, 0xF3, 0xFA)   # near-white blue
HI = (0x1F, 0x5C, 0xA8)   # deep blue
VMAX = 0.55               # scale top, the largest weight present


def colour(w):
    t = min(w / VMAX, 1.0)
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(LO, HI))


out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
    f'width="{WIDTH}" height="{HEIGHT}" font-family="system-ui, -apple-system, Segoe UI, Helvetica, sans-serif" '
    f'role="img" aria-label="Self-attention weight matrix for the tokens How are you doing ?">',
    '<title>Self-attention weights for each pair of positions</title>',
]

# axis captions
out.append(f'<text x="{X0 + N * CELL / 2}" y="30" text-anchor="middle" font-size="15" '
           f'fill="currentColor" opacity="0.75">key — attended to</text>')
out.append(f'<text x="26" y="{Y0 + N * CELL / 2}" text-anchor="middle" font-size="15" '
           f'fill="currentColor" opacity="0.75" '
           f'transform="rotate(-90 26 {Y0 + N * CELL / 2})">query — attending</text>')

# column labels
for j, tok in enumerate(TOKENS):
    out.append(f'<text x="{X0 + j * CELL + CELL / 2}" y="{Y0 - 14}" text-anchor="middle" '
               f'font-size="16" fill="currentColor">{tok}</text>')

# rows
for i, tok in enumerate(TOKENS):
    cy = Y0 + i * CELL + CELL / 2
    out.append(f'<text x="{X0 - 16}" y="{cy + 5}" text-anchor="end" font-size="16" '
               f'fill="currentColor">{tok}</text>')
    for j, w in enumerate(W[i]):
        x, y = X0 + j * CELL, Y0 + i * CELL
        txt = "#ffffff" if w > 0.30 else "#1a2a3a"
        out.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" fill="{colour(w)}" '
                   f'stroke="#ffffff" stroke-width="2"/>')
        out.append(f'<text x="{x + CELL / 2}" y="{y + CELL / 2 + 5}" text-anchor="middle" '
                   f'font-size="14" fill="{txt}">{w:.2f}</text>')

out.append('</svg>')

open(OUT, "w").write("\n".join(out) + "\n")
print(f"wrote {OUT}")
