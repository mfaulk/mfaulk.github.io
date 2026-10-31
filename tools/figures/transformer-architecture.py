"""Generate the three-panel transformer architecture figure as SVG.

Left to right: scaled dot-product attention, multi-head attention built out of
h of those, and the full encoder-decoder stack built out of those. Dashed
wedges connect each panel to the block it magnifies.

The stack is drawn post-norm, with Add & Norm after each sublayer, because
that is the arrangement in *Attention Is All You Need*. The implementation
later in the post is pre-norm, and the text explains the difference.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _panels import (GREY, Canvas, defs, panel_architecture, panel_mha,  # noqa: E402
                     panel_sdpa, r)

OUT = (pathlib.Path(__file__).resolve().parents[2]
       / "assets" / "images" / "transformer-architecture.svg")

GAP = 74          # horizontal space for each zoom wedge
PAD = 16


def measure(fn):
    """Draw a panel into its own canvas so its extent can be measured."""
    c = Canvas()
    anchors = fn(c)
    return c, anchors


def shifted(rect, dx):
    x0, y0, x1, y1 = rect
    return (x0 + dx, y0, x1 + dx, y1)


sdpa_c, sdpa_a = measure(lambda c: panel_sdpa(c, 0, 0))
mha_c, mha_a = measure(lambda c: panel_mha(c, 0, 0))
arch_c, arch_a = measure(lambda c: panel_architecture(c, 0, 0))

# Lay the three out left to right, bottoms aligned on the shared baseline.
dx_sdpa = -sdpa_c.x0
dx_mha = dx_sdpa + sdpa_c.x1 + GAP - mha_c.x0
dx_arch = dx_mha + mha_c.x1 + GAP - arch_c.x0

out = Canvas()
for canvas, dx in ((sdpa_c, dx_sdpa), (mha_c, dx_mha), (arch_c, dx_arch)):
    out.add(f'<g transform="translate({r(dx)},0)">')
    out.parts.extend(canvas.parts)
    out.add('</g>')
    out.note(canvas.x0 + dx, canvas.y0)
    out.note(canvas.x1 + dx, canvas.y1)

# Dashed wedges, each joining a panel's frame to the block it magnifies.
sdpa_frame = shifted(sdpa_a["frame"], dx_sdpa)
mha_sdpa_box = shifted(mha_a["sdpa_rect"], dx_mha)
mha_frame = shifted(mha_a["frame"], dx_mha)
arch_attn_box = shifted(arch_a["enc_self_attn"], dx_arch)

for src, dst in ((sdpa_frame, mha_sdpa_box), (mha_frame, arch_attn_box)):
    for sy, dy in ((src[1], dst[1]), (src[3], dst[3])):
        out.path(f"M {r(src[2])} {r(sy)} L {r(dst[0])} {r(dy)}", colour=GREY, head=False,
                 dash=True, width=1.2)

x0, y0 = out.x0 - PAD, out.y0 - PAD
w, h = out.x1 - out.x0 + 2 * PAD, out.y1 - out.y0 + 2 * PAD
header = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{r(x0)} {r(y0)} {r(w)} {r(h)}" '
    f'width="{r(w)}" height="{r(h)}" '
    f'font-family="system-ui, -apple-system, Segoe UI, Helvetica, sans-serif" role="img" '
    f'aria-label="The transformer architecture at three zoom levels. On the left, scaled '
    f'dot-product attention: queries and keys meet in a matrix multiply, which is scaled, '
    f'passed through a softmax, and multiplied by the values. In the middle, multi-head '
    f'attention: keys, values and queries are each projected by a linear layer and fed to h '
    f'parallel scaled dot-product attention heads, whose outputs are concatenated and '
    f'projected again. On the right, the full encoder-decoder stack: an encoder of N blocks, '
    f'each a multi-head self-attention sublayer and a feed-forward sublayer with Add and Norm '
    f'after each, and a decoder of N blocks that adds a masked self-attention sublayer and a '
    f'cross-attention sublayer taking its keys and values from the encoder output, ending in '
    f'a linear projection and a softmax.">')

OUT.write_text("\n".join(
    [header, '<title>The transformer architecture at three zoom levels</title>', defs()]
    + out.parts + ['</svg>']) + "\n")
print(f"wrote {OUT} ({r(w)}x{r(h)})")
