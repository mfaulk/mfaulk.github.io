"""Generate the encoder-decoder / decoder-only / encoder-only comparison figure.

Three schematic stacks: the full Transformer, a GPT-style decoder-only model,
and a BERT-style encoder-only model. The original this replaces showed three
shrunken copies of the full architecture diagram, which are unreadable at that
size; these are simplified to the sublayers that actually differ, since the
point of the figure is the taxonomy rather than the detail.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _panels import (FS, FS_SMALL, GREY, K_COL, MIX, Q_COL, V_COL,  # noqa: E402
                     Canvas, defs, r)

OUT = (pathlib.Path(__file__).resolve().parents[2]
       / "assets" / "images" / "encoder-decoder-architectures.svg")

COLW = 152
BOXH = 27
GAP = 9           # between sublayer boxes inside a block
PAD = 16
CROSS_GUTTER = 54
GROUP_GAP = 62

c = Canvas()


def small_box(cx, y_bottom, label, colour, w=COLW):
    return c.box(cx, y_bottom, label, colour, w, h=BOXH, fs=FS_SMALL)


def brace(x0, x1, y, depth=8):
    """Underbrace spanning x0..x1, tip pointing down at the midpoint."""
    mid = (x0 + x1) / 2
    c.add(f'<path d="M {r(x0)} {r(y)} q 0 {depth} {depth} {depth} H {r(mid - depth)} '
          f'q {depth} 0 {depth} {depth} q 0 {-depth} {depth} {-depth} H {r(x1 - depth)} '
          f'q {depth} 0 {depth} {-depth}" fill="none" stroke="{GREY}" stroke-width="1.6"/>')
    c.note(x0, y)
    c.note(x1, y + 2 * depth)
    return y + 2 * depth


def stack(cx, y_base, sublayers, in_label, head_label, n_side="left"):
    """One schematic stack, drawn upward. Returns useful anchors."""
    c.text(cx, y_base, in_label, fs=FS_SMALL)
    y = c.box(cx, y_base - 20, "Embedding", MIX, COLW, h=BOXH, fs=FS_SMALL)
    c.vline(cx, y_base - 14, y_base - 20)

    pe_cy = y - 16 - 8
    c.vline(cx, y, pe_cy + 8)
    c.add_node(cx, pe_cy)
    y = pe_cy - 8

    block_bottom = y - 12
    y -= 18
    first = True
    rects = {}
    for label, colour in sublayers:
        if not first:
            c.vline(cx, y + GAP, y)
        else:
            c.vline(cx, block_bottom + 10, y)
        box_bottom = y
        y = small_box(cx, y, label, colour)
        rects[label] = (cx - COLW / 2, y, cx + COLW / 2, box_bottom)
        y -= GAP
        first = False
    y += GAP
    block_top = y - 12
    c.add(f'<rect x="{r(cx - COLW / 2 - 10)}" y="{r(block_top)}" width="{r(COLW + 20)}" '
          f'height="{r(block_bottom - block_top)}" rx="9" fill="none" stroke="{GREY}" '
          f'stroke-width="1.2" opacity="0.45"/>')
    sign = -1 if n_side == "left" else 1
    c.text(cx + sign * (COLW / 2 + 26), (block_top + block_bottom) / 2 + 4, "N×",
           fs=FS_SMALL, weight="600")

    y = block_top
    if head_label:
        c.vline(cx, y, y - 18)
        y = small_box(cx, y - 18, "Linear", GREY)
        c.vline(cx, y, y - GAP)
        y = small_box(cx, y - GAP, "Softmax", GREY)
        c.vline(cx, y, y - 18)
        c.text(cx, y - 26, head_label, fs=FS_SMALL)
        y -= 26
    return {"top": y, "block_top": block_top, "rects": rects}


SELF = ("Self-Attention", Q_COL)
MASKED = ("Masked Self-Attention", Q_COL)
CROSS = ("Cross-Attention", K_COL)
FF = ("Feed Forward", V_COL)

BASE = 0

# ---------------------------------------------------------------- Transformer
enc_cx = 0
dec_cx = COLW + CROSS_GUTTER
enc = stack(enc_cx, BASE, [SELF, FF], "inputs", None, n_side="left")
dec = stack(dec_cx, BASE, [MASKED, CROSS, FF], "outputs (shifted right)",
            "output probabilities", n_side="right")

# The encoder output supplies the decoder's cross-attention keys and values.
# Routed up out of the encoder and along the gutter so it crosses no box.
cx0, cy0, _, cy1 = dec["rects"][CROSS[0]]
lane_x = (enc_cx + COLW / 2 + 10 + dec_cx - COLW / 2 - 10) / 2
c.path(f"M {r(enc_cx)} {r(enc['block_top'])} V {r(enc['block_top'] - 22)} H {r(lane_x)} "
       f"V {r((cy0 + cy1) / 2)} H {r(cx0 - 2)}", colour=K_COL, width=1.5)
c.text(lane_x, enc["block_top"] - 30, "keys, values", K_COL, fs=FS_SMALL)

# -------------------------------------------------------------------- labels
TRANS_X0, TRANS_X1 = enc_cx - COLW / 2, dec_cx + COLW / 2
gpt_cx = TRANS_X1 + GROUP_GAP + COLW / 2 + 26
stack(gpt_cx, BASE, [MASKED, FF], "inputs", "next-token probabilities", n_side="right")

bert_cx = gpt_cx + COLW / 2 + GROUP_GAP + COLW / 2 + 26
stack(bert_cx, BASE, [SELF, FF], "inputs", "masked-token probabilities", n_side="right")

BRACE_Y = BASE + 20
y1 = brace(enc_cx - COLW / 2, enc_cx + COLW / 2, BRACE_Y)
brace(dec_cx - COLW / 2, dec_cx + COLW / 2, BRACE_Y)
brace(gpt_cx - COLW / 2, gpt_cx + COLW / 2, BRACE_Y)
brace(bert_cx - COLW / 2, bert_cx + COLW / 2, BRACE_Y)

for cx, lab in ((enc_cx, "Encoder"), (dec_cx, "Decoder"),
                (gpt_cx, "Decoder-only"), (bert_cx, "Encoder-only")):
    c.text(cx, y1 + 18, lab, fs=FS, weight="600")

TITLE_Y = c.y0 - 14
for x, lab in (((TRANS_X0 + TRANS_X1) / 2, "Transformer"), (gpt_cx, "GPT"), (bert_cx, "BERT")):
    c.text(x, TITLE_Y, lab, fs=17, weight="700")

x0, y0 = c.x0 - PAD, c.y0 - PAD
w, h = c.x1 - c.x0 + 2 * PAD, c.y1 - c.y0 + 2 * PAD
header = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{r(x0)} {r(y0)} {r(w)} {r(h)}" '
    f'width="{r(w)}" height="{r(h)}" '
    f'font-family="system-ui, -apple-system, Segoe UI, Helvetica, sans-serif" role="img" '
    f'aria-label="Three transformer architectures side by side. The Transformer has an '
    f'encoder stack of self-attention and feed-forward sublayers and a decoder stack of '
    f'masked self-attention, cross-attention and feed-forward sublayers. GPT is decoder-only: '
    f'a single stack of masked self-attention and feed-forward sublayers producing next-token '
    f'probabilities. BERT is encoder-only: a single stack of unmasked self-attention and '
    f'feed-forward sublayers producing masked-token probabilities.">')

OUT.write_text("\n".join(
    [header, '<title>Encoder-decoder, decoder-only, and encoder-only architectures</title>',
     defs()] + c.parts + ['</svg>']) + "\n")
print(f"wrote {OUT} ({r(w)}x{r(h)})")
