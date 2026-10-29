"""Generate the RNN encoder-decoder / fixed-length context vector figure as SVG."""

import pathlib

OUT = pathlib.Path(__file__).resolve().parents[2] / "assets" / "images" / "seq2seq-context-vector.svg"


SRC = ["the", "cat", "drank", "the", "milk"]
TGT = ["le", "chat", "a", "bu", "le", "lait"]
CTX = "[0.1, -0.2, 0.8, 1.5, -0.3]"

FS = 16
CHARW = 0.58
PADX = 13
GAP = 26          # room for the arrow between boxes
BOXH = 38

ENC = "#1F5CA8"
DEC = "#B4531A"
ENC_FILL = "#ECF3FA"
DEC_FILL = "#FBF0E8"

LABEL_W = 86      # left gutter for "Encoder" / "Decoder"
MARGIN = 18


def boxw(s):
    return max(len(s) * FS * CHARW, 20) + 2 * PADX


def row(words):
    ws = [boxw(w) for w in words]
    return ws, sum(ws) + GAP * (len(ws) - 1)


SW, STOT = row(SRC)
TW, TTOT = row(TGT)

CONTENT = max(STOT, TTOT, boxw(CTX))
WIDTH = round(LABEL_W + CONTENT + 2 * MARGIN)
X_MID = LABEL_W + MARGIN + CONTENT / 2

ENC_Y, CTX_Y, DEC_Y = 26, 122, 218
CTX_H = 40
HEIGHT = DEC_Y + BOXH + 16


def xs_for(ws, tot):
    x = X_MID - tot / 2
    out = []
    for w in ws:
        out.append(x)
        x += w + GAP
    return out


SX, TX = xs_for(SW, STOT), xs_for(TW, TTOT)

o = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" '
    f'height="{HEIGHT}" font-family="system-ui, -apple-system, Segoe UI, Helvetica, sans-serif" '
    f'role="img" aria-label="An RNN encoder reads the sentence the cat drank the milk into a single '
    f'fixed-length context vector, which a decoder then expands into le chat a bu le lait.">',
    '<title>Encoder-decoder with a fixed-length context vector</title>',
    '<defs>',
    f'<marker id="ae" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
    f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{ENC}"/></marker>',
    f'<marker id="ad" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
    f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{DEC}"/></marker>',
    '</defs>',
]


def draw_row(xs, ws, words, y, stroke, fill, marker):
    for i, word in enumerate(words):
        o.append(f'<rect x="{round(xs[i],1)}" y="{y}" width="{round(ws[i],1)}" height="{BOXH}" '
                 f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        o.append(f'<text x="{round(xs[i]+ws[i]/2,1)}" y="{y+25}" text-anchor="middle" '
                 f'font-size="{FS}" fill="#1a2a3a">{word}</text>')
        if i < len(words) - 1:
            x1 = xs[i] + ws[i] + 5
            x2 = xs[i + 1] - 5
            o.append(f'<line x1="{round(x1,1)}" y1="{y+BOXH/2}" x2="{round(x2,1)}" y2="{y+BOXH/2}" '
                     f'stroke="{stroke}" stroke-width="1.5" marker-end="url(#{marker})"/>')


draw_row(SX, SW, SRC, ENC_Y, ENC, ENC_FILL, "ae")
draw_row(TX, TW, TGT, DEC_Y, DEC, DEC_FILL, "ad")

# row labels
o.append(f'<text x="{MARGIN}" y="{ENC_Y+25}" font-size="16" fill="{ENC}" font-weight="600">Encoder</text>')
o.append(f'<text x="{MARGIN}" y="{DEC_Y+25}" font-size="16" fill="{DEC}" font-weight="600">Decoder</text>')

# context vector
cw = boxw(CTX)
cx = X_MID - cw / 2
o.append(f'<rect x="{round(cx,1)}" y="{CTX_Y}" width="{round(cw,1)}" height="{CTX_H}" rx="19" '
         f'fill="#ffffff" stroke="#55606b" stroke-width="1.5"/>')
o.append(f'<text x="{round(X_MID,1)}" y="{CTX_Y+26}" text-anchor="middle" font-size="15" '
         f'fill="#1a2a3a" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{CTX}</text>')
o.append(f'<text x="{round(cx,1)}" y="{CTX_Y-11}" font-size="14" '
         f'fill="currentColor" opacity="0.8">context vector</text>')

# last encoder box -> context, context -> first decoder box
lx = SX[-1] + SW[-1] / 2
o.append(f'<path d="M {round(lx,1)} {ENC_Y+BOXH} V {CTX_Y+CTX_H/2} H {round(cx+cw+6,1)}" fill="none" '
         f'stroke="{ENC}" stroke-width="1.5" marker-end="url(#ae)"/>')
fx = TX[0] + TW[0] / 2
o.append(f'<path d="M {round(cx-6,1)} {CTX_Y+CTX_H/2} H {round(fx,1)} V {DEC_Y-6}" fill="none" '
         f'stroke="{DEC}" stroke-width="1.5" marker-end="url(#ad)"/>')

o.append('</svg>')

open(OUT, "w").write("\n".join(o) + "\n")
print(f"wrote {OUT} ({WIDTH}x{HEIGHT})")
