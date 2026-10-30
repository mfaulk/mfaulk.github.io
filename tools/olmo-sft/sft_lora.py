#!/usr/bin/env python3
"""Instruction-tune OLMo 2 1B with LoRA on a small slice of the Tulu 3 mixture.

A hand-rolled training loop rather than `trl`'s SFTTrainer, because the thing
worth seeing here is the label masking, and a Trainer hides it.

The collator is the part to read. Prompt and role tokens get label -100 so the
loss covers only the assistant response, and the span is located by rendering
the prompt and the full conversation through the *same* chat template and taking
the difference, rather than by string-matching role markers — which would be
wrong here anyway, since OLMo 2's role markers are ordinary multi-token text.

    python sft_lora.py --examples 4000 --out out/lora
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

BASE = "allenai/OLMo-2-0425-1B"
# The base checkpoint has no chat template; borrow the one the official SFT
# checkpoint uses so training and evaluation cannot disagree about the format.
TEMPLATE_SOURCE = "allenai/OLMo-2-0425-1B-SFT"
MIXTURE = "allenai/tulu-3-sft-olmo-2-mixture-0225"

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",      # attention
    "gate_proj", "up_proj", "down_proj",          # the gated feed-forward network
]


def build_example(tok, messages, max_len: int):
    """Render one single-turn conversation; mask everything but the response.

    Returns None if the example does not fit or the token spans do not line up.
    """
    prompt_text = tok.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
    full_text = tok.apply_chat_template(messages, tokenize=False)

    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]

    # The generation prompt must be a token-level prefix of the full rendering.
    # BPE does not guarantee this in general, so check rather than assume.
    if full_ids[: len(prompt_ids)] != prompt_ids:
        return None
    if len(full_ids) > max_len or len(full_ids) <= len(prompt_ids):
        return None

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    assert len(labels) == len(full_ids)
    return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id: int):
    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(attn),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=4000)
    ap.add_argument("--val-examples", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/lora")
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(TEMPLATE_SOURCE)

    # ---- data -----------------------------------------------------------
    print("loading mixture ...", flush=True)
    ds = load_dataset(MIXTURE, split="train")
    # Single-turn only. Multi-turn masking needs per-turn span arithmetic, and
    # everything measured in this post is single-turn, so this is a stated
    # scope limit rather than a shortcut.
    ds = ds.filter(
        lambda ex: len(ex["messages"]) == 2
        and ex["messages"][0]["role"] == "user"
        and ex["messages"][1]["role"] == "assistant",
        num_proc=8,
    )
    print(f"single-turn examples available: {len(ds):,}")

    want = args.examples + args.val_examples
    ds = ds.shuffle(seed=args.seed).select(range(min(want * 3, len(ds))))

    built, skipped = [], 0
    for ex in ds:
        item = build_example(tok, ex["messages"], args.max_len)
        if item is None:
            skipped += 1
            continue
        built.append(item)
        if len(built) >= want:
            break

    train_set, val_set = built[: args.examples], built[args.examples:]
    resp_tokens = sum(sum(1 for t in b["labels"] if t != -100) for b in train_set)
    all_tokens = sum(len(b["input_ids"]) for b in train_set)
    print(f"train {len(train_set):,}  val {len(val_set):,}  skipped {skipped:,} "
          f"(too long or span mismatch)")
    print(f"tokens: {all_tokens:,} total, {resp_tokens:,} scored "
          f"({resp_tokens / all_tokens:.1%} of the loss)")

    # Show one masked example, so the mask is visible rather than asserted.
    demo = train_set[0]
    shown = "".join(
        tok.decode([i]) if l != -100 else "·"
        for i, l in zip(demo["input_ids"], demo["labels"])
    )
    print(f"\nmasked example (prompt positions shown as ·):\n{shown[:600]}\n")

    # ---- model ----------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda")
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    model.config.use_cache = False

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    steps_per_epoch = math.ceil(len(train_set) / (args.batch_size * args.accum))
    total_steps = steps_per_epoch * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total_steps), total_steps)
    print(f"optimizer steps: {total_steps} "
          f"(effective batch {args.batch_size * args.accum})")

    def evaluate() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_set), args.batch_size):
                ids, labels, attn = collate(val_set[i: i + args.batch_size], tok.pad_token_id)
                out = model(input_ids=ids.cuda(), attention_mask=attn.cuda(), labels=labels.cuda())
                tot += out.loss.detach().item()
                n += 1
        model.train()
        return tot / max(n, 1)

    # ---- train ----------------------------------------------------------
    print(f"\nheld-out loss before training: {evaluate():.4f}")
    model.train()
    history = []
    t0 = time.time()
    micro = 0
    opt.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        order = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(epoch))
        for start in range(0, len(train_set), args.batch_size):
            batch = [train_set[j] for j in order[start: start + args.batch_size].tolist()]
            ids, labels, attn = collate(batch, tok.pad_token_id)
            out = model(input_ids=ids.cuda(), attention_mask=attn.cuda(), labels=labels.cuda())
            (out.loss / args.accum).backward()
            micro += 1

            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step = micro // args.accum
                if step % 25 == 0 or step == 1:
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    cur = out.loss.detach().item()
                    print(f"step {step:4d}/{total_steps}  loss {cur:.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  peak {peak:.2f} GiB  "
                          f"{time.time() - t0:.0f}s", flush=True)
                    history.append({"step": step, "loss": cur})

    val = evaluate()
    mins = (time.time() - t0) / 60
    print(f"\nheld-out loss after training:  {val:.4f}")
    print(f"wall clock: {mins:.1f} min   peak VRAM: "
          f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump({"args": vars(args), "val_loss": val,
                   "minutes": mins, "history": history}, f, indent=1)
    print(f"adapter saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
