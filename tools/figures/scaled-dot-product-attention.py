"""Generate the scaled dot-product attention detail figure as SVG."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _panels import Canvas, panel_sdpa, svg  # noqa: E402

OUT = (pathlib.Path(__file__).resolve().parents[2]
       / "assets" / "images" / "scaled-dot-product-attention.svg")

c = Canvas()
panel_sdpa(c, 0, 0)

OUT.write_text(svg(
    c,
    "Scaled dot-product attention",
    "Scaled dot-product attention. Queries and keys enter a matrix multiply, whose result is "
    "scaled by one over the square root of the key dimension, passed through a softmax, and "
    "then multiplied by the values. The values bypass the softmax and enter only the second "
    "matrix multiply.",
))
print(f"wrote {OUT}")
