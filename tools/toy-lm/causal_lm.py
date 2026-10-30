#!/usr/bin/env python3
"""A decoder-only language model built from the previous post's encoder-decoder parts.

This is the code in the post, runnable. The point of the exercise is the
comparison: the previous post solved sequence reversal with an encoder-decoder
transformer at 256/256 exact match, and this solves the same task with the
encoder and cross-attention deleted and the source moved into the prefix.

    BOS x1 x2 x3 SEP x3 x2 x1 EOS

`--loss-on` selects whether the loss covers the whole stream or only the answer
span after SEP, which is response-token masking at a scale where it can be
ablated in a minute rather than an hour.

No dependencies beyond torch.
"""

from __future__ import annotations

import argparse
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS, SEP = 0, 1, 2, 3
FIRST_SYMBOL = 4
VOCAB_SIZE = 12  # PAD, BOS, EOS, SEP + 8 content symbols
MAX_SYMBOLS = 8


# --------------------------------------------------------------------------
# Components carried over unchanged from the encoder-decoder post
# --------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads = d_model, num_heads
        self.d_head = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_p)
        for lin in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_uniform_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(
        self,
        x_q: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.w_q(x_q))
        k = self._split_heads(self.w_k(x_q))
        v = self._split_heads(self.w_v(x_q))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        return self.w_o(self._combine_heads(weights @ v))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout_p: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def causal_mask(T: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """(1, 1, T, T) boolean, True where attention is allowed."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device)).view(1, 1, T, T)


def padding_mask(lengths: torch.Tensor, T_k: int) -> torch.Tensor:
    """(B, 1, 1, T_k) boolean, True on real tokens."""
    idx = torch.arange(T_k, device=lengths.device).view(1, T_k)
    return (idx < lengths.view(-1, 1)).view(-1, 1, 1, T_k)


# --------------------------------------------------------------------------
# The delta: three sublayers become two
# --------------------------------------------------------------------------


class CausalBlock(nn.Module):
    """A decoder block with the cross-attention sublayer removed.

    Structurally identical to the previous post's EncoderBlock. The only thing
    that makes it a decoder is the mask it is called with.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout_p: float = 0.0, activation: str = "gelu", bias: bool = True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_p=dropout_p, bias=bias)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=bias),
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(x_q=self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class CausalLM(nn.Module):
    """Decoder-only autoregressive language model."""

    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 4,
                 d_ff: int = 512, num_layers: int = 2, dropout_p: float = 0.1,
                 max_len: int = 512, tie_weights: bool = True):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout_p=dropout_p)
        self.blocks = nn.ModuleList([
            CausalBlock(d_model, num_heads, d_ff, dropout_p=dropout_p)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        nn.init.normal_(self.embed.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[PAD].zero_()
        if tie_weights:
            self.lm_head.weight = self.embed.weight

    def forward(self, idx: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = idx.size(1)
        if mask is None:
            mask = causal_mask(T, device=idx.device)
        x = self.pos(self.embed(idx) * math.sqrt(self.d_model))
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def generate_greedy(self, prefix: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Append max_new_tokens greedily. No KV cache: this is the slow version."""
        out = prefix
        for _ in range(max_new_tokens):
            logits = self(out)[:, -1, :]
            out = torch.cat([out, logits.argmax(dim=-1, keepdim=True)], dim=1)
        return out


# --------------------------------------------------------------------------
# The task, as a single stream
# --------------------------------------------------------------------------


def make_batch(batch_size: int, generator: torch.Generator, device) -> tuple:
    """Build `BOS x.. SEP reversed(x).. EOS`, padded, plus the answer-span mask."""
    n_symbols = VOCAB_SIZE - FIRST_SYMBOL
    lengths = torch.randint(1, MAX_SYMBOLS + 1, (batch_size,), generator=generator)
    T = int(2 * MAX_SYMBOLS + 3)

    tokens = torch.full((batch_size, T), PAD, dtype=torch.long)
    is_answer = torch.zeros((batch_size, T), dtype=torch.bool)
    stream_lengths = torch.empty(batch_size, dtype=torch.long)

    for i, L in enumerate(lengths.tolist()):
        syms = (torch.randint(0, n_symbols, (L,), generator=generator) + FIRST_SYMBOL)
        stream = torch.cat([
            torch.tensor([BOS]), syms, torch.tensor([SEP]),
            syms.flip(0), torch.tensor([EOS]),
        ])
        tokens[i, : stream.numel()] = stream
        # The answer span is the reversed symbols plus EOS: everything after SEP.
        is_answer[i, L + 2 : 2 * L + 3] = True
        stream_lengths[i] = stream.numel()

    return (tokens.to(device), is_answer.to(device),
            stream_lengths.to(device), lengths.to(device))


def loss_for(model, tokens, is_answer, stream_lengths, loss_on: str) -> torch.Tensor:
    T = tokens.size(1)
    mask = causal_mask(T, device=tokens.device) & padding_mask(stream_lengths, T)
    logits = model(tokens[:, :-1], mask=mask[:, :, :-1, :-1])

    targets = tokens[:, 1:].clone()
    if loss_on == "answer":
        # Score only positions whose *target* lies in the answer span.
        targets = targets.masked_fill(~is_answer[:, 1:], PAD)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD
    )


@torch.no_grad()
def exact_match(model, n: int, generator: torch.Generator, device) -> tuple[int, int]:
    """Feed `BOS x.. SEP`, generate L tokens, require an exact reversal."""
    model.eval()
    tokens, _, _, lengths = make_batch(n, generator, device)
    correct = 0
    for L in lengths.unique().tolist():
        sel = (lengths == L).nonzero(as_tuple=True)[0]
        prefix = tokens[sel, : L + 2]          # BOS + L symbols + SEP
        want = tokens[sel, L + 2 : 2 * L + 2]  # the reversed symbols
        got = model.generate_greedy(prefix, L)[:, L + 2 :]
        correct += int((got == want).all(dim=1).sum())
    model.train()
    return correct, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss-on", choices=["stream", "answer"], default="stream")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    eval_gen = torch.Generator().manual_seed(12345)  # held out from training

    model = CausalLM(VOCAB_SIZE, d_model=128, num_heads=4, d_ff=512, num_layers=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device         {device}")
    print(f"loss computed  {'whole stream' if args.loss_on == 'stream' else 'answer span only'}")
    print(f"parameters     {n_params:,}   (encoder-decoder post: 930,828)")

    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (128 ** -0.5) * min((max(s, 1)) ** -0.5, max(s, 1) * args.warmup ** -1.5),
    )

    for step in range(1, args.steps + 1):
        tokens, is_answer, stream_lengths, _ = make_batch(args.batch_size, gen, device)
        loss = loss_for(model, tokens, is_answer, stream_lengths, args.loss_on)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 1000 == 0 or step == 1:
            c, n = exact_match(model, 256, torch.Generator().manual_seed(999), device)
            print(f"step {step:5d}  loss {loss.item():.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  exact-match {c}/{n}")

    c, n = exact_match(model, 256, eval_gen, device)
    print(f"\nFINAL exact-match on {n} held-out sequences: {c}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
