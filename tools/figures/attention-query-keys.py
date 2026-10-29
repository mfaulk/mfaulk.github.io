"""Generate the query/keys attention figure as SVG.

"crane" is ambiguous (bird or machine); the weights show it resolving
against "lifted" and "beam".
"""

import pathlib

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets" / "images" / "attention-query-keys.svg"


SENTENCE = ["The", "crane", "lifted", "the", "steel", "beam"]
QUERY_IDX = 1
# Attention from the query to each key; sums to 1.
W = [0.03, 0.15, 0.34, 0.03, 0.17, 0.28]
assert abs(sum(W) - 1.0) < 1e-9, sum(W)

FS = 17           # word font size
CHARW = 0.58      # rough advance ratio for the UI font
PADX = 14
GAP = 10
BOXH = 36

BLUE = "#1F5CA8"
FILL = "#ECF3FA"


def textw(s):
    return max(len(s) * FS * CHARW, 18)


widths = [textw(w) + 2 * PADX for w in SENTENCE]
total = sum(widths) + GAP * (len(SENTENCE) - 1)

MARGIN = 34
WIDTH = round(max(total + 2 * MARGIN, 600))
QY, KY = 56, 232                      # query box top, key row top
HEIGHT = KY + BOXH + 18

xs, x = [], MARGIN
for w in widths:
    xs.append(x)
    x += w + GAP

qw = widths[QUERY_IDX]
qx = xs[QUERY_IDX]                    # query box sits above its own word
qcx, qcy = qx + qw / 2, QY + BOXH

out = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
    f'width="{WIDTH}" height="{HEIGHT}" '
    f'font-family="system-ui, -apple-system, Segoe UI, Helvetica, sans-serif" role="img" '
    f'aria-label="The word crane acts as a query and attends to every word of the sentence '
    f'The crane lifted the steel beam. The heaviest edges go to lifted and beam.">',
    '<title>Attention relates a query to every element of the sequence</title>',
]

# edges, drawn first so boxes sit on top
for i, w in enumerate(W):
    kcx = xs[i] + widths[i] / 2
    sw = round(0.8 + w * 16, 2)
    op = round(min(0.22 + w * 1.6, 0.9), 2)
    out.append(f'<line x1="{round(qcx,1)}" y1="{qcy}" x2="{round(kcx,1)}" y2="{KY}" '
               f'stroke="{BLUE}" stroke-width="{sw}" stroke-opacity="{op}" stroke-linecap="round"/>')

# query box
out.append(f'<rect x="{round(qx,1)}" y="{QY}" width="{round(qw,1)}" height="{BOXH}" rx="6" '
           f'fill="{BLUE}"/>')
out.append(f'<text x="{round(qcx,1)}" y="{QY + 24}" text-anchor="middle" font-size="{FS}" '
           f'fill="#ffffff">{SENTENCE[QUERY_IDX]}</text>')

# key boxes
for i, word in enumerate(SENTENCE):
    out.append(f'<rect x="{round(xs[i],1)}" y="{KY}" width="{round(widths[i],1)}" height="{BOXH}" '
               f'rx="6" fill="{FILL}" stroke="{BLUE}" stroke-opacity="0.35"/>')
    out.append(f'<text x="{round(xs[i] + widths[i] / 2,1)}" y="{KY + 24}" text-anchor="middle" '
               f'font-size="{FS}" fill="#1a2a3a">{word}</text>')

# labels naming the parts; all explanation lives in the figure caption
out.append(f'<text x="{round(qx + qw + 16,1)}" y="{QY + 23}" font-size="14" fill="currentColor" '
           f'opacity="0.8">query</text>')
out.append(f'<text x="{round(xs[-1] + widths[-1] + 16,1)}" y="{KY + 24}" font-size="14" '
           f'fill="currentColor" opacity="0.8">keys</text>')
out.append('</svg>')

open(OUT, "w").write("\n".join(out) + "\n")
print(f"wrote {OUT}  ({WIDTH}x{HEIGHT})")
