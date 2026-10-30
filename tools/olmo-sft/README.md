# OLMo 2 SFT experiments

Companion code for the post *From Transformers to Language Models*. These
scripts produce the numbers and generations quoted in the post, so that every
result in it can be regenerated rather than taken on trust.

Unlike `tools/figures/`, these need third-party packages and a GPU.

## Environment

The scripts expect a virtualenv outside the repo, so nothing here needs to be
gitignored:

    python3 -m venv ~/.venvs/olmo-sft
    ~/.venvs/olmo-sft/bin/python -m pip install \
        torch transformers accelerate datasets peft trl safetensors sentencepiece

Pinned versions actually used for the post are recorded in `requirements.txt`.

## Scripts

| Script | What it does | GPU |
| --- | --- | --- |
| `check_env.py` | Verifies the GPU is usable and prints driver/runtime/package versions | required |
| `verify_tokenizer.py` | Prints the tokenizer and chat-template facts the post asserts, read off the published tokenizers | no |
| `base_probes.py` | Experiments A–H: continuation, few-shot, perplexity, stopping rate, format compliance | required |

Run them with the venv's interpreter:

    ~/.venvs/olmo-sft/bin/python tools/olmo-sft/check_env.py
    ~/.venvs/olmo-sft/bin/python tools/olmo-sft/verify_tokenizer.py
    ~/.venvs/olmo-sft/bin/python tools/olmo-sft/base_probes.py --model allenai/OLMo-2-0425-1B

`base_probes.py` takes `--model`, `--adapter`, and `--chat`, so the identical
prompts and scoring run against the base checkpoint, the official SFT
checkpoint, and our own fine-tune:

    # base: no chat template exists, so prompts go in raw
    base_probes.py --model allenai/OLMo-2-0425-1B
    # official SFT: must be prompted through its template
    base_probes.py --model allenai/OLMo-2-0425-1B-SFT --chat
    # ours: base weights + adapter, borrowing the SFT tokenizer's template
    base_probes.py --model allenai/OLMo-2-0425-1B --adapter out/lora \
                   --tokenizer allenai/OLMo-2-0425-1B-SFT --chat

Sampling is fixed in the script (greedy for single answers; `T=0.7`,
`top_p=0.9`, seeds 0–7 for distributions) rather than passed in, so the three
runs cannot accidentally differ in how they sample.

### Results so far

Recorded here so a rerun can be checked against them:

| Measure | Base | Ours (4k LoRA) | Official SFT (866k) |
| --- | --- | --- | --- |
| Stopping rate (G) | 13/48 = 27.1% | 36/48 = 75.0% | 40/48 = 83.3% |
| Format compliance (H) | 9/64 = 14.1% | 36/64 = 56.2% | 62/64 = 96.9% |
| — one word | 0/24 | 16/24 | 24/24 |
| — valid JSON | 4/16 | 14/16 | 14/16 |
| — three bullets | 0/16 | 0/16 | 16/16 |
| — no commas | 5/8 | 6/8 | 8/8 |
| Perplexity, 169 tokens of ordinary prose | 9.19 | 9.28 | 11.48 |
| Perplexity, same words shuffled | 392.45 | 393.19 | 565.95 |
| Perplexity, one token repeated 60x | 4.37 | 3.46 | 26313.75 |

Raw captures are in `results-base.txt`, `results-sft.txt`, and `results-ours.txt`.

Our run: `sft_lora.py --examples 4000`, rank 16, all seven projections, lr 1e-4,
effective batch 16, 1 epoch, 250 optimizer steps. Held-out loss 1.4419 to
1.2452. 21.9 min and 11.83 GiB peak on an RTX A4000.

Two results worth knowing before rerunning:

- Our adapter scores 0/16 on three-bullet lists, the same as the untouched base
  model, against the official checkpoint's 16/16. A uniform random 4,000-example
  sample drew only 158 examples from the precise-instruction-following subset
  (`personahub_ifdata_manual_seed_v3_29980`), which has 29,980 in the full mixture.
- The same uniform sample drew 598 examples (15%) from the three refusal-shaped
  subsets (`wildguardmixtrain`, `wildjailbreak`, `coconot`), and the resulting
  model refuses harmless requests at roughly two of eight seeds. Weighting the
  sample rather than shuffling it is the obvious next experiment.

Caveat on those two captures: their experiment D sections predate a fix. The
original perplexity samples were ~35 tokens, short enough that the first few
positions dominate the average and the number is noise (it read 213 where the
current 169-token passage reads 9.19). `PERPLEXITY_SAMPLES` was replaced and the
table above holds the corrected figures; the D section inside the saved captures
does not.

Two measurement caveats worth carrying forward:

- The stopping rate is bounded by `max_new_tokens=100`. "Give me a recipe for
  pancakes" scores 2/8 even post-SFT because a recipe genuinely needs more than
  100 tokens, so it is truncated rather than failing to stop.
- `no_commas` is the only constraint satisfiable by accident, which is why the
  base model scores 62.5% on it and near zero on the rest.

## Hardware this was developed on

NVIDIA RTX A4000, 16 GiB. The post's claim is that everything in it runs on one
16 GiB GPU, so that is the machine to keep it honest against.

## If `nvidia-smi` fails but training works

`nvidia-smi` reporting

    Failed to initialize NVML: Driver/library version mismatch

does **not** necessarily mean CUDA is broken. NVML and the CUDA driver API are
separate interfaces with separate version checks, and NVML's is the stricter of
the two. After a driver update leaves a stale kernel module resident — compare

    cat /proc/driver/nvidia/version   # version loaded in RAM
    modinfo -F version nvidia         # version on disk

— `nvidia-smi` refuses outright while `torch.cuda` continues to work across a
same-branch mismatch (observed here with 615.71.09 resident against 615.78.08 on
disk). So run `check_env.py` before concluding anything: it exercises the path
that matters by allocating and multiplying on the device.

A reboot restores `nvidia-smi`, which is worth having for watching VRAM during a
training run, but it is not a prerequisite for training. The modules cannot be
reloaded without a reboot when the card is also driving the display.

Note that `check_env.py` reports VRAM via torch rather than `nvidia-smi`, so it
keeps working in exactly this situation.

### How small is our training slice?

Three defensible answers, which is why the post states more than one:

| Measure | Ours | Official mixture | Fraction |
| --- | --- | --- | --- |
| Examples | 4,000 | 866,138 | 0.46% |
| Tokens | 1,555,866 | ~699M (estimated) | 0.22% |
| Trained parameters | 12,058,624 (LoRA) | 1,484,916,736 (full) | 0.81% |

The token estimate comes from tokenizing a random 3,000-example sample through
the chat template: mean 807.4 tokens per example. Our examples average 389,
roughly half, because `--max-len 1024` discards the longest candidates. So
counting examples overstates our data share by about 2x relative to counting
tokens.
