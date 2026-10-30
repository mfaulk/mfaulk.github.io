#!/usr/bin/env python3
"""Print the tokenizer and chat-template facts the post asserts, from the real files.

The post makes several claims that are easy to get wrong by transcribing from a
model card: that the base checkpoint has no chat template, and the exact
whitespace and `<|endoftext|>` placement of the rendered template. Everything
here is read off the published tokenizer rather than written by hand.

Needs network on first run; afterwards it is served from the HF cache. No GPU.
"""

BASE = "allenai/OLMo-2-0425-1B"
SFT = "allenai/OLMo-2-0425-1B-SFT"


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    from transformers import AutoTokenizer

    base = AutoTokenizer.from_pretrained(BASE)
    sft = AutoTokenizer.from_pretrained(SFT)

    rule(f"{BASE} — the base checkpoint")
    print(f"class            {type(base).__name__}")
    print(f"vocab_size       {base.vocab_size}")
    print(f"bos_token        {base.bos_token!r} (id {base.bos_token_id})")
    print(f"eos_token        {base.eos_token!r} (id {base.eos_token_id})")
    print(f"pad_token        {base.pad_token!r} (id {base.pad_token_id})")
    print(f"chat_template    {base.chat_template!r}")

    rule("Tokens are not words")
    text = "Tokens are not words."
    ids = base(text)["input_ids"]
    print(f"{text!r}")
    print(f"  ids    {ids}")
    print(f"  pieces {[base.decode([i]) for i in ids]}")
    for variant in ("the", " the"):
        print(f"{variant!r:>8} -> {base(variant)['input_ids']}")

    rule(f"{SFT} — the chat template")
    print(f"chat_template is None: {sft.chat_template is None}")

    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]

    training = sft.apply_chat_template(messages, tokenize=False)
    print("\n-- a training example (add_generation_prompt=False) --")
    print(repr(training))

    inference = sft.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
    print("\n-- an inference prompt (add_generation_prompt=True) --")
    print(repr(inference))

    rule("Role tokens")
    for tok in ("<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>", "<|pad|>"):
        print(f"{tok:<16} -> {sft.convert_tokens_to_ids(tok)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
