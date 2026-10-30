# Toy decoder-only language model

Companion code for the first half of the post *From Transformers to Language
Models*: the decoder-only model built by deleting parts of the previous post's
encoder-decoder, trained on the same sequence-reversal task so the two are
directly comparable.

`causal_lm.py` is self-contained and needs only `torch` — no `transformers`, no
model download. It carries over `MultiHeadAttention`, `PositionalEncoding`,
`causal_mask`, and `padding_mask` unchanged from the previous post, and adds
`CausalBlock` (the previous post's `DecoderBlock` with the cross-attention
sublayer removed) and `CausalLM`.

The task is the reversal task reformatted as one stream:

    BOS x1 x2 x3 SEP x3 x2 x1 EOS

## Running it

    ~/.venvs/olmo-sft/bin/python tools/toy-lm/causal_lm.py                  # loss on whole stream
    ~/.venvs/olmo-sft/bin/python tools/toy-lm/causal_lm.py --loss-on answer # loss on answer span only

About a minute each on a GPU. It falls back to CPU automatically, where it takes
a few minutes.

`--loss-on` is response-token masking at a scale where the ablation costs a
minute: `stream` scores every position, `answer` scores only the span after
`SEP`.

## Results

Both at `--seed 0`, 6000 steps, 256 held-out sequences:

| | Encoder-decoder (previous post) | `--loss-on stream` | `--loss-on answer` |
| --- | --- | --- | --- |
| Parameters | 930,828 | 398,336 | 398,336 |
| Final loss | 0.0059 | 1.0486 | 0.0323 |
| Exact match | 256/256 | 256/256 | 256/256 |

Two things to note about that table.

The decoder-only model matches the encoder-decoder's accuracy at 43% of the
parameters, because deleting the encoder removes two of the four transformer
blocks, deleting cross-attention removes a sublayer from each of the two that
remain, and tying the single embedding table to the output head removes the rest.

The loss figures are not comparable across columns even though the accuracy is.
With the loss over the whole stream, half the positions are input symbols drawn
uniformly from eight options, contributing an irreducible `ln(8) = 2.079` nats
each, so the achievable floor is about `ln(8)/2 = 1.0397` — which is essentially
where it lands. Restricting the loss to the answer span removes that constant
and the same model reports 0.03. The encoder-decoder's 0.0059 is a third
convention again, since its loss only ever covered the target sequence.
