# Figure generators

Scripts that generate the hand-drawn SVG figures in `assets/images/`. Each
script writes exactly one figure, and is named after the file it produces.

| Script | Output | Used in |
| --- | --- | --- |
| `seq2seq-context-vector.py` | `seq2seq-context-vector.svg` | Transformers — recurrence |
| `attention-query-keys.py` | `attention-query-keys.svg` | Transformers — attention |
| `dot-product-attention.py` | `dot-product-attention.svg` | Transformers — attention |
| `self-attention-weights.py` | `self-attention-weights.svg` | Transformers — self-attention |
| `qkv-projections.py` | `qkv-projections.svg` | Transformers — keys, values, queries |
| `encoder-decoder-architecture.py` | `encoder-decoder-architecture.svg` | Transformers to Language Models — where we left off |
| `scaled-dot-product-attention.py` | `scaled-dot-product-attention.svg` | Transformers — scaled dot product attention |
| `multi-head-attention.py` | `multi-head-attention.svg` | Transformers — multi-head attention |
| `transformer-architecture.py` | `transformer-architecture.svg` | Transformers — opening diagram |

`_panels.py` is the exception to one-script-one-figure: it generates nothing by
itself. The three figures above are the same panels at three zoom levels, so
the panel builders live there and the figures compose them. Files beginning
with an underscore are skipped by the regenerate-all loop below.

## Running them

Standard library only, no dependencies. Each resolves its output path from its
own location, so the working directory does not matter:

    python3 tools/figures/self-attention-weights.py

Regenerating with no source change rewrites the identical bytes, so a clean
`git status` after running them all is the check that the committed SVGs match
their scripts.

To regenerate everything:

    for f in tools/figures/[!_]*.py; do python3 "$f"; done

## Conventions these figures follow

- **Colours are consistent across figures**: queries `#1F5CA8` (blue), keys
  `#B4531A` (orange), values `#2F7A4F` (green), shared input `#5B4B8A`
  (purple), outputs `#55606b` (grey). The architecture figures extend this:
  anything combining q, k and v is purple, and plain operations are grey.
- **The image carries labels, the caption carries prose.** A figure names its
  parts — `Encoder`, `query`, `keys`, axis labels, dimensions — and everything
  explanatory lives in the `figcaption` in the post, where it stays selectable,
  translatable, and reachable by screen readers.
- **Text uses `fill="currentColor"`** wherever it is annotation rather than part
  of a coloured element, so it follows the page's text colour.
- **Weights sum to 1** where a figure depicts a softmax. The generators assert
  this, so a figure cannot drift into showing a distribution that is not one.
- **Every figure sets `role="img"`, a `<title>`, and an `aria-label`** that
  describes what it shows.

## A note on checking changes

These are laid out by arithmetic, not by eye, so collisions and clipping do not
show up by reading the source. Render the output before committing — every
defect found while writing these (labels overlapping the row above, captions
clipped past the viewBox, a shared line drawn once per branch so only the last
colour showed, shading falling into a visible checkerboard) was invisible in the
markup and obvious on screen.
