#!/usr/bin/env python3
"""Probe what a checkpoint does with continuations, few-shot prompts, and instructions.

Runs the same prompts and scoring against any checkpoint so that base, official
SFT, and our own fine-tune are measured identically. The scoring is mechanical
throughout — no judge model — so the numbers are reproducible.

    python base_probes.py --model allenai/OLMo-2-0425-1B
    python base_probes.py --model allenai/OLMo-2-0425-1B-SFT --chat
    python base_probes.py --model allenai/OLMo-2-0425-1B --adapter out/lora --chat

`--chat` renders each prompt through the tokenizer's chat template, which is the
only honest way to prompt a post-trained model. The base checkpoint has no
template, so it must be run without it.
"""

import argparse
import json
import re

import torch

# One sampling configuration, fixed here and reused for every generation in the
# post, because "how you sample" is otherwise a confound in every comparison.
TEMPERATURE = 0.7
TOP_P = 0.9
SEEDS = list(range(8))

CODE_PROMPT = '''def running_mean(xs: list[float], window: int) -> list[float]:
    """Return the moving average of xs over the given window size."""
'''

FEWSHOT_TRANSLATE = """English: the book is on the table
French: le livre est sur la table

English: she drinks coffee every morning
French: elle boit du cafe chaque matin

English: my brother lives in a small village
French: mon frere habite dans un petit village

English: the children are playing in the garden
French:"""

FEWSHOT_JSON = """Text: name: Ada Lovelace, born: 1815
JSON: {"name": "Ada Lovelace", "born": 1815}

Text: name: Alan Turing, born: 1912
JSON: {"name": "Alan Turing", "born": 1912}

Text: name: Grace Hopper, born: 1906
JSON: {"name": "Grace Hopper", "born": 1906}

Text: name: Katherine Johnson, born: 1918
JSON:"""

# Experiment G: does generation terminate on its own?
STOPPING_PROMPTS = [
    "Write a haiku about recursion.",
    "Explain what a hash table is.",
    "List three uses for a paperclip.",
    "What is the capital of France?",
    "Summarize the plot of Hamlet in one sentence.",
    "Give me a recipe for pancakes.",
]

# Experiment H: constraints a checker can verify without judgement.
FORMAT_TASKS = [
    ("one_word", "What is the capital of France? Answer in exactly one word."),
    ("one_word", "What color is a ripe banana? Answer in exactly one word."),
    ("one_word", "How many legs does a spider have? Answer in exactly one word."),
    ("json", 'Give me the name and year of birth of Ada Lovelace as JSON with keys "name" and "born". Output only JSON.'),
    ("json", 'Give me the capital and population of France as JSON with keys "capital" and "population". Output only JSON.'),
    ("bullets", "List exactly three primary colors, as three bullet points starting with '- ' and nothing else."),
    ("bullets", "List exactly three planets, as three bullet points starting with '- ' and nothing else."),
    ("no_commas", "Describe the ocean in one sentence without using any commas."),
]


def check_one_word(text: str) -> bool:
    return len(text.strip().split()) == 1


def check_json(text: str) -> bool:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return False
    try:
        return isinstance(json.loads(match.group(0)), dict)
    except json.JSONDecodeError:
        return False


def check_bullets(text: str) -> bool:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return len(lines) == 3 and all(ln.strip().startswith("- ") for ln in lines)


def check_no_commas(text: str) -> bool:
    return "," not in text.strip()


CHECKERS = {
    "one_word": check_one_word,
    "json": check_json,
    "bullets": check_bullets,
    "no_commas": check_no_commas,
}


class Runner:
    def __init__(self, model, tok, use_chat: bool):
        self.model, self.tok, self.use_chat = model, tok, use_chat

    def _render(self, prompt: str) -> str:
        if not self.use_chat:
            return prompt
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt: str, max_new_tokens: int, seed=None) -> tuple[str, bool]:
        """Return (continuation, stopped_on_its_own)."""
        text = self._render(prompt)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=not self.use_chat)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        n_in = enc["input_ids"].shape[1]

        if seed is None:
            gen_kwargs = dict(do_sample=False)
        else:
            torch.manual_seed(seed)
            gen_kwargs = dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
            **gen_kwargs,
        )[0]

        new = out[n_in:]
        stopped = bool((new == self.tok.eos_token_id).any())
        return self.tok.decode(new, skip_special_tokens=True), stopped

    def perplexity(self, text: str) -> float:
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            loss = self.model(**enc, labels=enc["input_ids"]).loss
        return float(torch.exp(loss))


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    ap.add_argument("--adapter", default=None, help="optional PEFT adapter directory")
    ap.add_argument("--chat", action="store_true", help="render prompts via chat template")
    ap.add_argument("--tokenizer", default=None, help="defaults to --model")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    r = Runner(model, tok, args.chat)
    print(f"model      {args.model}")
    print(f"adapter    {args.adapter}")
    print(f"chat       {args.chat} (template present: {tok.chat_template is not None})")
    print(f"sampling   greedy for single answers; T={TEMPERATURE} top_p={TOP_P} for seeded runs")

    rule("B - code completion (greedy, 120 tokens)")
    body, stopped = r.generate(CODE_PROMPT, 120)
    print(body)
    print(f"\n[stopped on its own: {stopped}]")
    # Did it wander into a second definition it was not asked for?
    extra = len(re.findall(r"^\s*def ", body, re.MULTILINE))
    print(f"[unrequested extra 'def' blocks: {extra}]")

    rule("C - few-shot: English to French (greedy, 20 tokens)")
    out, _ = r.generate(FEWSHOT_TRANSLATE, 20)
    print(repr(out))

    rule("C - few-shot: text to JSON (greedy, 40 tokens)")
    out, _ = r.generate(FEWSHOT_JSON, 40)
    print(repr(out))
    print(f"[valid JSON object in output: {check_json(out)}]")

    rule("D - perplexity")
    samples = {
        "prose written after the model's cutoff": (
            "The reordered normalization is not a return to the original post-norm "
            "formulation, which normalized after the residual addition and broke the "
            "identity path. The residual stream here is still untouched."
        ),
        "the same words, shuffled": (
            "identity norm original formulation the residual which post stream "
            "normalized here addition still after not broke untouched the a is "
            "reordered return path and normalization to the"
        ),
        "repeated token": "the the the the the the the the the the the the the the",
    }
    for name, text in samples.items():
        print(f"  {r.perplexity(text):9.2f}   {name}")

    rule(f"E - bare instruction, {len(SEEDS)} seeds (T={TEMPERATURE}, 60 tokens)")
    bare = "Write a haiku about recursion."
    for seed in SEEDS:
        out, stopped = r.generate(bare, 60, seed=seed)
        flat = out.replace("\n", " / ").strip()
        print(f"  seed {seed} stop={str(stopped):<5} {flat[:150]}")

    rule("F - the same instruction, scaffolded (greedy, 60 tokens)")
    scaffolds = {
        "Q/A framing": f"Q: {bare}\nA:",
        "document framing": f"The following is a haiku about recursion.\n\n",
        "one-shot": (
            "Q: Write a haiku about the ocean.\n"
            "A: Waves fold into waves,\nthe shore forgets each arrival,\nsalt remains behind.\n\n"
            f"Q: {bare}\nA:"
        ),
    }
    for name, prompt in scaffolds.items():
        out, stopped = r.generate(prompt, 60)
        print(f"\n-- {name} (stop={stopped}) --")
        print(out.strip()[:400])

    rule(f"G - stopping rate ({len(STOPPING_PROMPTS)} prompts x {len(SEEDS)} seeds, 100 tokens)")
    stops = 0
    total = 0
    for prompt in STOPPING_PROMPTS:
        hits = 0
        for seed in SEEDS:
            _, stopped = r.generate(prompt, 100, seed=seed)
            hits += stopped
            total += 1
        stops += hits
        print(f"  {hits}/{len(SEEDS)}  {prompt}")
    print(f"\n  STOPPING RATE: {stops}/{total} = {stops / total:.1%}")

    rule(f"H - format compliance ({len(FORMAT_TASKS)} tasks x {len(SEEDS)} seeds, 80 tokens)")
    passed = 0
    total = 0
    by_kind: dict[str, list[int]] = {}
    for kind, prompt in FORMAT_TASKS:
        hits = 0
        for seed in SEEDS:
            out, _ = r.generate(prompt, 80, seed=seed)
            hits += CHECKERS[kind](out)
            total += 1
        passed += hits
        by_kind.setdefault(kind, []).append(hits)
        print(f"  {hits}/{len(SEEDS)}  [{kind}] {prompt[:64]}")
    print()
    for kind, hitlist in by_kind.items():
        n = len(hitlist) * len(SEEDS)
        print(f"  {kind:<10} {sum(hitlist)}/{n} = {sum(hitlist) / n:.1%}")
    print(f"\n  FORMAT COMPLIANCE: {passed}/{total} = {passed / total:.1%}")

    rule("one example of each format task's raw output (greedy)")
    for kind, prompt in FORMAT_TASKS[:4]:
        out, _ = r.generate(prompt, 80)
        print(f"\n-- [{kind}] {prompt[:70]}")
        print(f"   {out.strip()[:300]!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
