"""Shared panel builders for the three transformer-architecture figures.

Not a figure generator: it writes nothing and is skipped by the regenerate-all
loop. `scaled-dot-product-attention.py`, `multi-head-attention.py`, and
`transformer-architecture.py` each import it and compose the same panels at
different zoom levels, so the three figures cannot drift apart.

Coordinates run with y increasing downward. Every panel is drawn upward from a
baseline and reports the box it occupied.
"""

# Palette shared across the figures in this directory.
Q_COL = "#1F5CA8"     # queries, blue
K_COL = "#B4531A"     # keys, orange
V_COL = "#2F7A4F"     # values, green
MIX = "#5B4B8A"       # combines q, k and v: purple
GREY = "#55606b"      # plain operations

FILL = {Q_COL: "#ECF3FA", K_COL: "#FBF0E8", V_COL: "#EAF3EE",
        MIX: "#F0EDF6", GREY: "#EFF1F3"}

TXT = "#1a2a3a"
FS = 14
FS_SMALL = 12
LINEH = 16.5

BOXH = 30
BOXH2 = 46
GAP = 20
ADD_R = 10


class Canvas:
    """Accumulates SVG fragments and tracks the extent drawn."""

    def __init__(self):
        self.parts = []
        self.x0 = self.y0 = 1e9
        self.x1 = self.y1 = -1e9

    def add(self, s):
        self.parts.append(s)

    def note(self, x, y):
        self.x0, self.y0 = min(self.x0, x), min(self.y0, y)
        self.x1, self.y1 = max(self.x1, x), max(self.y1, y)

    # -- primitives ------------------------------------------------------
    def box(self, cx, y_bottom, label, colour, w, h=None, fs=FS):
        lines = label if isinstance(label, list) else [label]
        if h is None:
            h = BOXH if len(lines) == 1 else BOXH2
        y = y_bottom - h
        self.add(f'<rect x="{r(cx - w / 2)}" y="{r(y)}" width="{r(w)}" height="{r(h)}" rx="6" '
                 f'fill="{FILL[colour]}" stroke="{colour}" stroke-width="1.5"/>')
        first = y + h / 2 + 5 if len(lines) == 1 else y + h / 2 - 3
        for i, ln in enumerate(lines):
            self.add(f'<text x="{r(cx)}" y="{r(first + i * LINEH)}" text-anchor="middle" '
                     f'font-size="{fs}" fill="{TXT}">{ln}</text>')
        self.note(cx - w / 2, y)
        self.note(cx + w / 2, y_bottom)
        return y

    def vline(self, x, y_from, y_to, colour=GREY, head=True, dash=False):
        d = ' stroke-dasharray="4 3"' if dash else ""
        m = ' marker-end="url(#arrow)"' if head else ""
        self.add(f'<line x1="{r(x)}" y1="{r(y_from)}" x2="{r(x)}" y2="{r(y_to)}" '
                 f'stroke="{colour}" stroke-width="1.5"{d}{m}/>')
        self.note(x, y_to)
        self.note(x, y_from)

    def path(self, d, colour=GREY, head=True, dash=False, width=1.5):
        dd = ' stroke-dasharray="4 3"' if dash else ""
        m = ' marker-end="url(#arrow)"' if head else ""
        self.add(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}"{dd}{m}/>')

    def text(self, x, y, s, colour="currentColor", anchor="middle", fs=FS, weight=None,
             opacity=None):
        w = f' font-weight="{weight}"' if weight else ""
        op = f' opacity="{opacity}"' if opacity else ""
        self.add(f'<text x="{r(x)}" y="{r(y)}" text-anchor="{anchor}" font-size="{fs}" '
                 f'fill="{colour}"{w}{op}>{s}</text>')
        # Approximate the run so anchored text is not clipped by the viewBox.
        tw = len(s) * fs * 0.56
        x0 = {"middle": x - tw / 2, "end": x - tw, "start": x}[anchor]
        self.note(x0, y - fs)
        self.note(x0 + tw, y + 4)

    def dashed_frame(self, x0, y0, x1, y1):
        self.add(f'<rect x="{r(x0)}" y="{r(y0)}" width="{r(x1 - x0)}" height="{r(y1 - y0)}" '
                 f'rx="12" fill="none" stroke="{GREY}" stroke-width="1.2" '
                 f'stroke-dasharray="5 4" opacity="0.5"/>')
        self.note(x0, y0)
        self.note(x1, y1)

    def add_node(self, cx, cy):
        self.add(f'<circle cx="{r(cx)}" cy="{r(cy)}" r="{ADD_R}" fill="#ffffff" stroke="{GREY}" '
                 f'stroke-width="1.5"/>')
        self.add(f'<path d="M {r(cx - 5)} {r(cy)} h 10 M {r(cx)} {r(cy - 5)} v 10" '
                 f'stroke="{GREY}" stroke-width="1.5"/>')
        self.note(cx - ADD_R, cy - ADD_R)
        self.note(cx + ADD_R, cy + ADD_R)


def r(v):
    return round(v + 0.0, 1)


def defs():
    return ('<defs>'
            f'<marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" '
            f'markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{GREY}"/></marker>'
            '</defs>')


def svg(canvas, title, aria, pad=16):
    x0, y0 = canvas.x0 - pad, canvas.y0 - pad
    w, h = canvas.x1 - canvas.x0 + 2 * pad, canvas.y1 - canvas.y0 + 2 * pad
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{r(x0)} {r(y0)} {r(w)} {r(h)}" '
            f'width="{r(w)}" height="{r(h)}" '
            f'font-family="system-ui, -apple-system, Segoe UI, Helvetica, sans-serif" '
            f'role="img" aria-label="{aria}">')
    return "\n".join([head, f'<title>{title}</title>', defs()] + canvas.parts + ['</svg>']) + "\n"


# ---------------------------------------------------------------- panel 1
SDPA_W = 190


def panel_sdpa(c, cx, y_base, frame=True):
    """Scaled dot-product attention, drawn upward from y_base.

    Q and K meet at the first MatMul; V runs up the right and meets the second.
    Returns a dict of useful anchors.
    """
    bw = 104
    qx, kx, vx = cx - 52, cx - 8, cx + 52

    c.text(qx, y_base, "Q", Q_COL, fs=FS, weight="600")
    c.text(kx, y_base, "K", K_COL, fs=FS, weight="600")
    c.text(vx, y_base, "V", V_COL, fs=FS, weight="600")

    mm1_bottom = y_base - 22
    mm1_cx = (qx + kx) / 2
    mm1_top = c.box(mm1_cx, mm1_bottom, "MatMul", GREY, bw)
    c.vline(qx, y_base - 16, mm1_bottom, Q_COL)
    c.vline(kx, y_base - 16, mm1_bottom, K_COL)

    scale_top = c.box(mm1_cx, mm1_top - GREY_GAP, "Scale", GREY, bw)
    c.vline(mm1_cx, mm1_top, mm1_top - GREY_GAP)
    soft_top = c.box(mm1_cx, scale_top - GREY_GAP, "Softmax", GREY, bw)
    c.vline(mm1_cx, scale_top, scale_top - GREY_GAP)

    mm2_bottom = soft_top - GREY_GAP
    mm2_top = c.box(cx, mm2_bottom, "MatMul", GREY, bw + 44)
    c.vline(mm1_cx, soft_top, mm2_bottom)
    # V bypasses the softmax entirely and enters the second MatMul
    c.vline(vx, y_base - 16, mm2_bottom, V_COL)

    c.vline(cx, mm2_top, mm2_top - 24)

    box = None
    if frame:
        x0, x1 = cx - SDPA_W / 2, cx + SDPA_W / 2
        c.dashed_frame(x0, mm2_top - 14, x1, y_base + 10)
        box = (x0, mm2_top - 14, x1, y_base + 10)
    return {"top": mm2_top - 24, "cx": cx, "frame": box}


GREY_GAP = 18


# ---------------------------------------------------------------- panel 2
MHA_W = 330


def panel_mha(c, cx, y_base, frame=True):
    """Multi-head attention: three Linear projections, h parallel heads,
    concatenate, output projection."""
    bw = 86
    kx, vx, qx = cx - 108, cx, cx + 108

    c.text(kx, y_base, "K", K_COL, fs=FS, weight="600")
    c.text(vx, y_base, "V", V_COL, fs=FS, weight="600")
    c.text(qx, y_base, "Q", Q_COL, fs=FS, weight="600")

    lin_bottom = y_base - 22
    lin_top = None
    for x, col in ((kx, K_COL), (vx, V_COL), (qx, Q_COL)):
        lin_top = c.box(x, lin_bottom, "Linear", col, bw)
        c.vline(x, y_base - 16, lin_bottom, col)

    # h stacked heads: two ghost copies offset behind the front one
    sdpa_bottom = lin_top - 26
    sdpa_w = MHA_W - 40
    for off in (10, 5):
        c.add(f'<rect x="{r(cx - sdpa_w / 2 + off)}" y="{r(sdpa_bottom - BOXH - off)}" '
              f'width="{r(sdpa_w)}" height="{BOXH}" rx="6" fill="{FILL[MIX]}" stroke="{MIX}" '
              f'stroke-width="1.2" opacity="0.45"/>')
    sdpa_top = c.box(cx, sdpa_bottom, "Scaled Dot-Product Attention", MIX, sdpa_w)
    for x, col in ((kx, K_COL), (vx, V_COL), (qx, Q_COL)):
        c.vline(x, lin_top, sdpa_bottom, col)

    cat_top = c.box(cx, sdpa_top - 26, "Concatenate", GREY, 150)
    c.vline(cx, sdpa_top - 15, sdpa_top - 26)
    out_top = c.box(cx, cat_top - GREY_GAP, "Linear", GREY, bw)
    c.vline(cx, cat_top, cat_top - GREY_GAP)
    c.vline(cx, out_top, out_top - 24)

    c.text(cx + sdpa_w / 2 - 6, sdpa_top - 20, "h heads", GREY, anchor="end", fs=FS_SMALL,
           opacity="0.9")

    box = None
    if frame:
        x0, x1 = cx - MHA_W / 2, cx + MHA_W / 2
        c.dashed_frame(x0, out_top - 14, x1, y_base + 10)
        box = (x0, out_top - 14, x1, y_base + 10)
    return {"top": out_top - 24, "cx": cx, "frame": box,
            "sdpa_rect": (cx - sdpa_w / 2, sdpa_top, cx + sdpa_w / 2, sdpa_bottom)}


# ---------------------------------------------------------------- panel 3
ARCH_COLW = 176
ARCH_GUTTER = 92


def panel_architecture(c, cx, y_base):
    """The full post-norm encoder-decoder stack, as drawn in the 2017 paper.

    Add & Norm follows each sublayer, which is the paper's arrangement; the
    implementation in the post is pre-norm, and the text explains the swap.
    """
    colw = ARCH_COLW
    enc_cx = cx - (colw + ARCH_GUTTER) / 2
    dec_cx = cx + (colw + ARCH_GUTTER) / 2
    ATTN, FFN, EMB = Q_COL, V_COL, MIX

    TICK = 16          # length of the Q/K/V arrows below an attention box
    FEED = 24          # where the fan runs horizontally, below those arrows
    QKV_IN = 40        # taller input gap for sublayers that show Q/K/V

    def qkv_ticks(bx, y, feed_from=None, skip=()):
        """Three labelled arrows entering an attention box from below.

        `feed_from` fans them out of a single trunk point; entries named in
        `skip` are left for a caller that wires them differently. Labels sit
        beside their arrow rather than on it, which at this scale is the only
        way they stay legible.
        """
        xs = {"K": bx - 34, "V": bx, "Q": bx + 34}
        cols = {"K": K_COL, "V": V_COL, "Q": Q_COL}
        for lab, x in xs.items():
            if lab in skip:
                continue
            c.vline(x, y + TICK, y, cols[lab])
            c.text(x + 7, y + TICK - 3, lab, cols[lab], anchor="start", fs=FS_SMALL,
                   weight="600")
            if feed_from is not None:
                c.path(f"M {r(bx)} {r(feed_from)} V {r(y + FEED)} H {r(x)} V {r(y + TICK)}",
                       colour=cols[lab], head=False, width=1.3)
        return xs

    rects = {}

    def sublayer(bx, y_in, label, colour, side, qkv=False, key=None):
        sl_bottom = y_in - (QKV_IN if qkv else 26)
        sl_top = c.box(bx, sl_bottom, label, colour, colw)
        if key:
            rects[key] = (bx - colw / 2, sl_top, bx + colw / 2, sl_bottom)
        if qkv:
            qkv_ticks(bx, sl_bottom, feed_from=y_in)
        else:
            c.vline(bx, y_in, sl_bottom)
        an_bottom = sl_top - 16
        an_top = c.box(bx, an_bottom, "Add &amp; Norm", GREY, colw)
        c.vline(bx, sl_top, an_bottom)
        sign = -1 if side == "left" else 1
        ax = bx + sign * (colw / 2 + 18)
        c.path(f"M {r(bx)} {r(y_in)} H {r(ax)} V {r(an_bottom - BOXH / 2)} "
               f"H {r(bx + sign * colw / 2)}", dash=True)
        return an_top

    # encoder
    c.text(enc_cx, y_base, "inputs", fs=FS)
    emb_top = c.box(enc_cx, y_base - 22, "Input Embedding", EMB, colw)
    c.vline(enc_cx, y_base - 16, y_base - 22)
    pe_cy = emb_top - 18 - ADD_R
    c.vline(enc_cx, emb_top, pe_cy + ADD_R)
    c.add_node(enc_cx, pe_cy)
    c.text(enc_cx - colw / 2 - 26, pe_cy - 4, "Positional", anchor="end", fs=FS_SMALL,
           opacity="0.85")
    c.text(enc_cx - colw / 2 - 26, pe_cy + 12, "Embedding", anchor="end", fs=FS_SMALL,
           opacity="0.85")
    c.path(f"M {r(enc_cx - colw / 2 - 22)} {r(pe_cy)} H {r(enc_cx - ADD_R - 2)}")

    y = pe_cy - ADD_R
    enc_bottom = y - 10
    y = sublayer(enc_cx, y - 20, ["Multi-Head", "Self-Attention"], ATTN, "left", qkv=True,
                 key="enc_self_attn")
    y = sublayer(enc_cx, y - 22, "Feed Forward", FFN, "left")
    enc_top = y - 10
    c.add(f'<rect x="{r(enc_cx - colw / 2 - 28)}" y="{r(enc_top)}" width="{r(colw + 56)}" '
          f'height="{r(enc_bottom - enc_top)}" rx="10" fill="none" stroke="{GREY}" '
          f'stroke-width="1.2" opacity="0.45"/>')
    c.text(enc_cx - colw / 2 - 44, (enc_top + enc_bottom) / 2, "N×", fs=FS_SMALL, weight="600")
    enc_out = enc_top

    # decoder
    c.text(dec_cx, y_base, "targets", fs=FS)
    emb_top = c.box(dec_cx, y_base - 22, "Output Embedding", EMB, colw)
    c.vline(dec_cx, y_base - 16, y_base - 22)
    pe_cy = emb_top - 18 - ADD_R
    c.vline(dec_cx, emb_top, pe_cy + ADD_R)
    c.add_node(dec_cx, pe_cy)
    c.text(dec_cx + colw / 2 + 26, pe_cy - 4, "Positional", anchor="start", fs=FS_SMALL,
           opacity="0.85")
    c.text(dec_cx + colw / 2 + 26, pe_cy + 12, "Embedding", anchor="start", fs=FS_SMALL,
           opacity="0.85")
    c.path(f"M {r(dec_cx + colw / 2 + 22)} {r(pe_cy)} H {r(dec_cx + ADD_R + 2)}")

    y = pe_cy - ADD_R
    dec_bottom = y - 10
    y = sublayer(dec_cx, y - 20, ["Masked Multi-Head", "Self-Attention"], ATTN, "right", qkv=True)
    cross_in = y - 22
    cross_bottom = cross_in - QKV_IN
    cross_top = c.box(dec_cx, cross_bottom, ["Multi-Head", "Cross-Attention"], K_COL, colw)
    qkv_ticks(dec_cx, cross_bottom, skip=("Q",))
    qkv_ticks(dec_cx, cross_bottom, feed_from=cross_in, skip=("K", "V"))
    an_bottom = cross_top - 16
    y = c.box(dec_cx, an_bottom, "Add &amp; Norm", GREY, colw)
    c.vline(dec_cx, cross_top, an_bottom)
    c.path(f"M {r(dec_cx)} {r(cross_in)} H {r(dec_cx + colw / 2 + 18)} "
           f"V {r(an_bottom - BOXH / 2)} H {r(dec_cx + colw / 2)}", dash=True)
    y = sublayer(dec_cx, y - 22, "Feed Forward", FFN, "right")
    dec_top = y - 10
    c.add(f'<rect x="{r(dec_cx - colw / 2 - 28)}" y="{r(dec_top)}" width="{r(colw + 56)}" '
          f'height="{r(dec_bottom - dec_top)}" rx="10" fill="none" stroke="{GREY}" '
          f'stroke-width="1.2" opacity="0.45"/>')
    c.text(dec_cx + colw / 2 + 44, (dec_top + dec_bottom) / 2, "×N", fs=FS_SMALL, weight="600")

    y = c.box(dec_cx, dec_top - 20, "Linear", GREY, colw)
    c.vline(dec_cx, dec_top, dec_top - 20)
    y2 = c.box(dec_cx, y - GREY_GAP, "Softmax", GREY, colw)
    c.vline(dec_cx, y, y - GREY_GAP)
    c.vline(dec_cx, y2, y2 - 22)
    c.text(dec_cx, y2 - 44, "Output", fs=FS)
    c.text(dec_cx, y2 - 28, "Probabilities", fs=FS)

    # Encoder output feeds the cross-attention keys and values. It is routed up
    # out of the encoder, along the gutter between the two columns, and in at
    # the tick row, so it never crosses a box.
    lane_y = enc_out - 20
    gutter_x = (enc_cx + colw / 2 + 28 + dec_cx - colw / 2 - 28) / 2
    tick_y = cross_bottom + TICK
    c.path(f"M {r(enc_cx)} {r(enc_out)} V {r(lane_y)} H {r(gutter_x)} V {r(tick_y)} "
           f"H {r(dec_cx)}", colour=GREY, head=False, width=1.5)
    c.text(gutter_x + 6, lane_y - 8, "encoder output", GREY, anchor="middle", fs=FS_SMALL,
           opacity="0.9")

    return {"left": enc_cx - colw / 2 - 28, "enc_self_attn": rects["enc_self_attn"]}
