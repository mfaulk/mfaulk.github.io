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

Run them with the venv's interpreter:

    ~/.venvs/olmo-sft/bin/python tools/olmo-sft/check_env.py
    ~/.venvs/olmo-sft/bin/python tools/olmo-sft/verify_tokenizer.py

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
