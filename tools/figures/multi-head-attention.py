"""Generate the multi-head attention detail figure as SVG.

Two panels: the scaled dot-product attention detail on the left, zoomed out of
the "Scaled Dot-Product Attention" block inside multi-head attention on the
right.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _panels import GREY, Canvas, MHA_W, SDPA_W, panel_mha, panel_sdpa, r, svg  # noqa: E402

OUT = (pathlib.Path(__file__).resolve().parents[2]
       / "assets" / "images" / "multi-head-attention.svg")

ZOOM_GAP = 86

c = Canvas()
left = panel_sdpa(c, 0, 0)
right = panel_mha(c, SDPA_W / 2 + ZOOM_GAP + MHA_W / 2, 0)

# Dashed wedge: the left panel is a magnification of the block on the right.
lx0, ly0, lx1, ly1 = left["frame"]
sx0, sy0, _, sy1 = right["sdpa_rect"]
for ly, sy in ((ly0, sy0), (ly1, sy1)):
    c.path(f"M {r(lx1)} {r(ly)} L {r(sx0)} {r(sy)}", colour=GREY, head=False, dash=True,
           width=1.2)

OUT.write_text(svg(
    c,
    "Multi-head attention",
    "Multi-head attention. Keys, values and queries are each projected by a separate linear "
    "layer, the projections feed h parallel scaled dot-product attention heads, and the heads' "
    "outputs are concatenated and projected once more. Dashed lines magnify one of those heads "
    "into the scaled dot-product attention detail on the left.",
))
print(f"wrote {OUT}")
