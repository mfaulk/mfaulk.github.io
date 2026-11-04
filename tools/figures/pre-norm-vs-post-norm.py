"""Generate the pre-norm / post-norm comparison figure as SVG.

Two encoder blocks side by side, identical except for where the normalization
sits. The figure's whole claim is geometric: in post-norm a LayerNorm sits on
the residual path between every pair of sublayers, so the path through the block
is drawn in three pieces; in pre-norm the norm moves onto the branch and the
path is one unbroken line from the block's input to its output. The counters at
the bottom assert exactly that, so the drawing cannot drift away from the
argument it is making.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _panels import (ADD_R, BOXH, BOXH2, FILL, FS, FS_SMALL, GREY,  # noqa: E402
                     Q_COL, V_COL, Canvas, r, svg)

OUT = (pathlib.Path(__file__).resolve().parents[2]
       / "assets" / "images" / "pre-norm-vs-post-norm.svg")

BOXW = 150          # sublayer boxes, on the branch
NORMW = 120         # a LayerNorm box when it sits on the residual path
TRUNK_DX = -70      # residual path, offset from the panel centre
BRANCH_DX = 40      # sublayer branch, offset from the panel centre
TRUNK_W = 3         # the residual path is drawn heavier than the branch
PANEL_DX = 336

ATTN, FFN = Q_COL, V_COL

c = Canvas()
tally = {}


def open_node(cx, cy):
    """An add node with no fill, so the residual path shows through it."""
    c.add(f'<circle cx="{r(cx)}" cy="{r(cy)}" r="{ADD_R}" fill="none" stroke="{GREY}" '
          f'stroke-width="1.5"/>')
    c.add(f'<path d="M {r(cx - 5)} {r(cy)} h 10 M {r(cx)} {r(cy - 5)} v 10" '
          f'stroke="{GREY}" stroke-width="1.5"/>')
    c.note(cx - ADD_R, cy - ADD_R)
    c.note(cx + ADD_R, cy + ADD_R)


def trunk(key, x, y_from, y_to, head=False):
    """One contiguous piece of the residual path, drawn through any add nodes
    along it. Counted, because the count is the point of the figure."""
    m = ' marker-end="url(#arrow)"' if head else ""
    c.add(f'<line x1="{r(x)}" y1="{r(y_from)}" x2="{r(x)}" y2="{r(y_to)}" stroke="{GREY}" '
          f'stroke-width="{TRUNK_W}"{m}/>')
    c.note(x, y_from)
    c.note(x, y_to)
    tally[key]["trunk_pieces"] += 1


def branch(d):
    c.path(d, colour=GREY, head=True, width=1.4)


def panel(cx, pre):
    """One block, drawn upward from y = 0. Returns the y of its output."""
    key = "pre" if pre else "post"
    tally[key] = {"trunk_pieces": 0, "norms": 0, "norms_on_trunk": 0,
                  "adds": 0, "sublayers": 0}
    tx, bx = cx + TRUNK_DX, cx + BRANCH_DX

    c.text(tx, 34, "x — from the previous block", anchor="start", fs=FS_SMALL, opacity="0.85")

    def sublayer(y_in, label, colour, two_line):
        """Draw the branch and its add node. Returns the y above the add node
        (pre-norm), or the top of the norm that follows it (post-norm)."""
        h = BOXH2 if two_line else BOXH

        if pre:
            n_bottom = y_in - 26
            n_top = c.box(bx, n_bottom, "LayerNorm", GREY, BOXW, h=BOXH, fs=FS_SMALL)
            tally[key]["norms"] += 1
            branch(f"M {r(tx)} {r(y_in)} H {r(bx)} V {r(n_bottom)}")
            s_bottom = n_top - 14
            s_top = c.box(bx, s_bottom, label, colour, BOXW, h=h)
            tally[key]["sublayers"] += 1
            c.vline(bx, n_top, s_bottom, GREY)
        else:
            s_bottom = y_in - 26
            s_top = c.box(bx, s_bottom, label, colour, BOXW, h=h)
            tally[key]["sublayers"] += 1
            branch(f"M {r(tx)} {r(y_in)} H {r(bx)} V {r(s_bottom)}")

        add_cy = s_top - 26
        branch(f"M {r(bx)} {r(s_top)} V {r(add_cy)} H {r(tx + ADD_R + 3)}")
        c.text((tx + bx) / 2 + 10, add_cy - 9, "dropout", GREY, fs=FS_SMALL, opacity="0.9")
        open_node(tx, add_cy)
        tally[key]["adds"] += 1
        return add_cy

    if pre:
        add1 = sublayer(-18, ["Multi-Head", "Self-Attention"], ATTN, True)
        add2 = sublayer(add1 - ADD_R - 18, "Feed Forward", FFN, False)
        out_y = add2 - ADD_R - 30
        # One line from the block's input to its output: the whole point.
        trunk(key, tx, 0, out_y, head=True)
        # The norm a pre-norm stack still needs, once, above all N blocks.
        c.add(f'<rect x="{r(tx - NORMW / 2)}" y="{r(out_y - 34)}" width="{r(NORMW)}" '
              f'height="30" rx="6" fill="{FILL[GREY]}" stroke="{GREY}" stroke-width="1.5" '
              f'stroke-dasharray="5 4"/>')
        c.text(tx, out_y - 14, "LayerNorm", fs=FS_SMALL)
        c.note(tx - NORMW / 2, out_y - 34)
        c.text(tx + NORMW / 2 + 10, out_y - 15, "once, after the N× stack", GREY,
               anchor="start", fs=FS_SMALL, opacity="0.9")
        c.vline(tx, out_y - 34, out_y - 54, GREY)
        top_y = out_y - 54
    else:
        add1 = sublayer(-18, ["Multi-Head", "Self-Attention"], ATTN, True)
        n1_bottom = add1 - ADD_R - 18
        trunk(key, tx, 0, n1_bottom)                 # input, through the first add
        n1_top = c.box(tx, n1_bottom, "LayerNorm", GREY, NORMW, h=BOXH, fs=FS_SMALL)
        tally[key]["norms"] += 1
        tally[key]["norms_on_trunk"] += 1

        add2 = sublayer(n1_top - 18, "Feed Forward", FFN, False)
        n2_bottom = add2 - ADD_R - 18
        trunk(key, tx, n1_top, n2_bottom)            # through the second add
        n2_top = c.box(tx, n2_bottom, "LayerNorm", GREY, NORMW, h=BOXH, fs=FS_SMALL)
        tally[key]["norms"] += 1
        tally[key]["norms_on_trunk"] += 1

        top_y = n2_top - 30
        trunk(key, tx, n2_top, top_y, head=True)     # out of the block
    c.text(tx, top_y - 14, "to the next block", anchor="start", fs=FS_SMALL, opacity="0.85")

    # Name the heavy line once per panel, alongside its lowest stretch.
    lab_y = -64
    c.add(f'<text x="{r(tx - 16)}" y="{r(lab_y)}" text-anchor="middle" font-size="{FS_SMALL}" '
          f'fill="currentColor" opacity="0.7" '
          f'transform="rotate(-90 {r(tx - 16)} {r(lab_y)})">residual path</text>')
    c.note(tx - 28, lab_y - 46)
    return top_y


tops = {
    "post": panel(0, False),
    "pre": panel(PANEL_DX, True),
}

title_y = min(tops.values()) - 58
for key, cx, title, formula in (
    ("post", 0, "post-norm", "x ← LayerNorm(x + Sublayer(x))"),
    ("pre", PANEL_DX, "pre-norm", "x ← x + Sublayer(LayerNorm(x))"),
):
    c.text(cx - 6, title_y, title, fs=FS, weight="600")
    c.text(cx - 6, title_y + 20, formula, GREY, fs=FS_SMALL)

# -- invariants ----------------------------------------------------------
# Both panels show the same parts, or the comparison is rigged: two sublayers,
# two additions, two norms. Only the norms' position differs.
for key in ("pre", "post"):
    assert tally[key]["sublayers"] == 2, tally
    assert tally[key]["adds"] == 2, tally
    assert tally[key]["norms"] == 2, tally

# The claim: a norm placed on the residual path breaks it. Post-norm puts both
# norms there and so draws the path in one more piece than it has norms on it;
# pre-norm puts neither there and draws it in a single line.
assert tally["post"]["norms_on_trunk"] == 2, tally
assert tally["post"]["trunk_pieces"] == 1 + tally["post"]["norms_on_trunk"], tally
assert tally["pre"]["norms_on_trunk"] == 0, tally
assert tally["pre"]["trunk_pieces"] == 1, tally

OUT.write_text(svg(
    c,
    "Pre-norm and post-norm transformer blocks",
    "Two encoder blocks side by side. In the post-norm block on the left, each sublayer's "
    "output is added back to the residual path and the sum is then passed through a LayerNorm, "
    "so the path is interrupted by a normalization between every pair of sublayers. In the "
    "pre-norm block on the right, the LayerNorm moves onto the branch ahead of each sublayer "
    "and the residual path runs unbroken from the block's input to its output, with a single "
    "LayerNorm after the whole stack. Dropout is applied to each sublayer's output before the "
    "addition in both arrangements, never to the residual path itself.",
))
print(f"wrote {OUT}")
