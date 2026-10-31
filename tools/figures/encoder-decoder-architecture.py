"""Generate the encoder-decoder transformer architecture figure as SVG.

Draws the model built in the Transformers post, which is pre-norm: each
sublayer is LayerNorm -> sublayer -> residual add, rather than the post-norm
"Add & Norm" of the original paper's Figure 1. The boxes sit on the main
column and the residual bypass is the dashed arc around the outside.

Layout runs bottom-up from y = 0 into negative y, and the viewBox is fitted to
the extent actually used, so nothing has to be measured twice.
"""

import pathlib

OUT = (pathlib.Path(__file__).resolve().parents[2]
       / "assets" / "images" / "encoder-decoder-architecture.svg")

# Palette shared across the figures in this directory.
ATTN = "#1F5CA8"      # self-attention (queries blue)
CROSS = "#B4531A"     # cross-attention (keys orange)
FFN = "#2F7A4F"       # feed-forward (values green)
EMB = "#5B4B8A"       # embeddings / positional (shared input purple)
GREY = "#55606b"      # norms and the output head

FILL = {ATTN: "#ECF3FA", CROSS: "#FBF0E8", FFN: "#EAF3EE",
        EMB: "#F0EDF6", GREY: "#EFF1F3"}

TXT = "#1a2a3a"
FS = 14.5
FS_SMALL = 12.5
LINEH = 17

COLW = 196            # width of a stack's boxes
OUTER = 104           # gutter outside each column: residual arc + side labels
CENTRE = 112          # gutter between the columns for the cross-attention path
MARGIN = 16

BOXH1 = 30            # single-line box
BOXH2 = 46            # two-line box
VGAP = 14             # vertical gap between boxes within a sublayer
SUBGAP = 18           # gap between sublayers
ADD_R = 11
ARC_DX = 24           # how far outside the column the residual arc runs

WIDTH = MARGIN + OUTER + COLW + CENTRE + COLW + OUTER + MARGIN
ENC_CX = MARGIN + OUTER + COLW / 2
DEC_CX = MARGIN + OUTER + COLW + CENTRE + COLW / 2

o = []
_extent = [0.0]       # most negative y reached


def note(y):
    _extent[0] = min(_extent[0], y)


def box(cx, y_bottom, label, colour, w=COLW):
    """Draw a box with its bottom edge at y_bottom. Return its top edge."""
    lines = label if isinstance(label, list) else [label]
    h = BOXH1 if len(lines) == 1 else BOXH2
    y = y_bottom - h
    o.append(f'<rect x="{round(cx - w / 2, 1)}" y="{round(y, 1)}" width="{w}" height="{h}" '
             f'rx="6" fill="{FILL[colour]}" stroke="{colour}" stroke-width="1.5"/>')
    first = y + h / 2 + 5 if len(lines) == 1 else y + h / 2 - 3
    for i, ln in enumerate(lines):
        o.append(f'<text x="{round(cx, 1)}" y="{round(first + i * LINEH, 1)}" '
                 f'text-anchor="middle" font-size="{FS}" fill="{TXT}">{ln}</text>')
    note(y)
    return y


def trunk(cx, y_from, y_to, colour=GREY):
    o.append(f'<line x1="{round(cx, 1)}" y1="{round(y_from, 1)}" x2="{round(cx, 1)}" '
             f'y2="{round(y_to, 1)}" stroke="{colour}" stroke-width="1.5" '
             f'marker-end="url(#arrow)"/>')
    note(y_to)


def add_circle(cx, cy):
    o.append(f'<circle cx="{round(cx, 1)}" cy="{round(cy, 1)}" r="{ADD_R}" fill="#ffffff" '
             f'stroke="{GREY}" stroke-width="1.5"/>')
    o.append(f'<path d="M {round(cx - 5.5, 1)} {round(cy, 1)} h 11 M {round(cx, 1)} '
             f'{round(cy - 5.5, 1)} v 11" stroke="{GREY}" stroke-width="1.5"/>')
    note(cy - ADD_R)


def sublayer(cx, y_in, name, colour, side):
    """LayerNorm -> sublayer -> residual add, bottom-up from y_in.

    Returns (new trunk top, vertical centre of the sublayer box).
    """
    ln_bottom = y_in - VGAP
    trunk(cx, y_in, ln_bottom)
    ln_top = box(cx, ln_bottom, "LayerNorm", GREY)

    sl_bottom = ln_top - VGAP
    trunk(cx, ln_top, sl_bottom)
    sl_top = box(cx, sl_bottom, name, colour)

    add_cy = sl_top - VGAP - ADD_R
    trunk(cx, sl_top, add_cy + ADD_R, colour)
    add_circle(cx, add_cy)

    sign = -1 if side == "left" else 1
    ax = cx + sign * (COLW / 2 + ARC_DX)
    ex = cx + sign * ADD_R
    o.append(f'<path d="M {round(cx, 1)} {round(y_in, 1)} H {round(ax, 1)} '
             f'V {round(add_cy, 1)} H {round(ex, 1)}" fill="none" stroke="{GREY}" '
             f'stroke-width="1.4" stroke-dasharray="4 3" opacity="0.75" '
             f'marker-end="url(#arrow)"/>')
    return add_cy - ADD_R, (sl_top + sl_bottom) / 2


def positional(cx, cy, side):
    """Two-line 'Positional Encoding' label in the outer gutter, arrow into the add."""
    sign = -1 if side == "left" else 1
    tx = cx + sign * (COLW / 2 + ARC_DX + 14)
    anchor = "end" if side == "left" else "start"
    for i, ln in enumerate(["Positional", "Encoding"]):
        o.append(f'<text x="{round(tx, 1)}" y="{round(cy - 4 + i * LINEH, 1)}" '
                 f'text-anchor="{anchor}" font-size="{FS_SMALL}" fill="currentColor" '
                 f'opacity="0.85">{ln}</text>')
    x_from = tx + sign * 8
    o.append(f'<line x1="{round(x_from, 1)}" y1="{round(cy, 1)}" '
             f'x2="{round(cx + sign * (ADD_R + 2), 1)}" y2="{round(cy, 1)}" stroke="{GREY}" '
             f'stroke-width="1.4" opacity="0.75" marker-end="url(#arrow)"/>')


def container(cx, y_top, y_bottom, side, label="N×"):
    o.append(f'<rect x="{round(cx - COLW / 2 - ARC_DX - 10, 1)}" y="{round(y_top, 1)}" '
             f'width="{round(COLW + 2 * (ARC_DX + 10), 1)}" '
             f'height="{round(y_bottom - y_top, 1)}" rx="10" fill="none" stroke="{GREY}" '
             f'stroke-width="1.2" opacity="0.4"/>')
    sign = -1 if side == "left" else 1
    o.append(f'<text x="{round(cx + sign * (COLW / 2 + ARC_DX + 24), 1)}" '
             f'y="{round((y_top + y_bottom) / 2 + 5, 1)}" text-anchor="middle" '
             f'font-size="{FS_SMALL}" fill="currentColor" opacity="0.9" '
             f'font-weight="600">{label}</text>')
    note(y_top)


def caption(cx, y, text):
    o.append(f'<text x="{round(cx, 1)}" y="{round(y, 1)}" text-anchor="middle" '
             f'font-size="{FS}" fill="currentColor">{text}</text>')
    note(y - 12)


# ------------------------------------------------------------------ encoder
caption(ENC_CX, -8, "Inputs")
y = box(ENC_CX, -30, ["Input", "Embedding"], EMB)

pe_cy = y - 20 - ADD_R
trunk(ENC_CX, y, pe_cy + ADD_R)
add_circle(ENC_CX, pe_cy)
positional(ENC_CX, pe_cy, "left")
y = pe_cy - ADD_R

enc_block_bottom = y - SUBGAP + 12
y, _ = sublayer(ENC_CX, y - SUBGAP, ["Multi-Head", "Self-Attention"], ATTN, "left")
y, _ = sublayer(ENC_CX, y - SUBGAP, ["Feed", "Forward"], FFN, "left")
enc_block_top = y - 12
container(ENC_CX, enc_block_top, enc_block_bottom, "left")

trunk(ENC_CX, enc_block_top, enc_block_top - 18)
enc_out_y = box(ENC_CX, enc_block_top - 18, "LayerNorm", GREY)

# ------------------------------------------------------------------ decoder
caption(DEC_CX, -8, "Outputs (shifted right)")
y = box(DEC_CX, -30, ["Output", "Embedding"], EMB)

pe_cy = y - 20 - ADD_R
trunk(DEC_CX, y, pe_cy + ADD_R)
add_circle(DEC_CX, pe_cy)
positional(DEC_CX, pe_cy, "right")
y = pe_cy - ADD_R

dec_block_bottom = y - SUBGAP + 12
y, _ = sublayer(DEC_CX, y - SUBGAP, ["Masked Multi-Head", "Self-Attention"], ATTN, "right")
y, cross_mid = sublayer(DEC_CX, y - SUBGAP, ["Multi-Head", "Cross-Attention"], CROSS, "right")
y, _ = sublayer(DEC_CX, y - SUBGAP, ["Feed", "Forward"], FFN, "right")
dec_block_top = y - 12
container(DEC_CX, dec_block_top, dec_block_bottom, "right")

trunk(DEC_CX, dec_block_top, dec_block_top - 18)
y = box(DEC_CX, dec_block_top - 18, "LayerNorm", GREY)
trunk(DEC_CX, y, y - 20)
y = box(DEC_CX, y - 20, "Linear", GREY)
trunk(DEC_CX, y, y - 20)
y = box(DEC_CX, y - 20, "Softmax", GREY)
trunk(DEC_CX, y, y - 20)
caption(DEC_CX, y - 28, "Output probabilities")

# ------------------------------------- encoder output -> decoder cross-attention
lane = ENC_CX + COLW / 2 + ARC_DX + 10 + (CENTRE - 2 * (ARC_DX + 10)) / 2
o.append(f'<path d="M {round(ENC_CX, 1)} {round(enc_out_y, 1)} V {round(enc_out_y - 22, 1)} '
         f'H {round(lane, 1)} V {round(cross_mid, 1)} '
         f'H {round(DEC_CX - COLW / 2 - 3, 1)}" fill="none" stroke="{CROSS}" '
         f'stroke-width="1.6" marker-end="url(#arrowx)"/>')
note(enc_out_y - 22)
# Two lines, right-aligned clear of the box: the gutter is narrower than
# "keys, values" set on one line.
for i, ln in enumerate(["keys,", "values"]):
    o.append(f'<text x="{round(DEC_CX - COLW / 2 - 8, 1)}" '
             f'y="{round(cross_mid - 24 + i * LINEH, 1)}" text-anchor="end" '
             f'font-size="{FS_SMALL}" fill="{CROSS}">{ln}</text>')

# ------------------------------------------------------------------- assemble
TOP = _extent[0] - MARGIN
HEIGHT = round(MARGIN - TOP)
header = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {round(TOP, 1)} {WIDTH} {HEIGHT}" '
    f'width="{WIDTH}" height="{HEIGHT}" '
    f'font-family="system-ui, -apple-system, Segoe UI, Helvetica, sans-serif" role="img" '
    f'aria-label="A pre-norm encoder-decoder transformer. The encoder stack on the left reads '
    f'the source sequence through a multi-head self-attention sublayer and a feed-forward '
    f'sublayer. Its output supplies the keys and values to a cross-attention sublayer in the '
    f'decoder stack on the right, which also has a masked self-attention sublayer and a '
    f'feed-forward sublayer, followed by a linear projection and a softmax over the '
    f'vocabulary.">',
    '<title>Encoder-decoder transformer, pre-norm</title>',
    '<defs>'
    f'<marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
    f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{GREY}"/></marker>'
    f'<marker id="arrowx" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" '
    f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{CROSS}"/></marker>'
    '</defs>',
]

OUT.write_text("\n".join(header + o + ['</svg>']) + "\n")
print(f"wrote {OUT} ({WIDTH}x{HEIGHT})")
