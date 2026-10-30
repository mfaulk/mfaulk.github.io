#!/usr/bin/env python3
"""Smoke test for the GPU and the training stack.

Run this first after any driver update or reboot. A stale NVIDIA kernel module
reports `torch.cuda.is_available() == False` with no further explanation, so
this prints the driver and runtime versions alongside the result to make that
specific failure obvious.
"""

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL: torch is not installed in this interpreter")
        print(f"       interpreter: {sys.executable}")
        return 1

    print(f"python              {sys.version.split()[0]}")
    print(f"torch               {torch.__version__}")
    print(f"torch CUDA runtime  {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("\nFAIL: torch.cuda.is_available() is False")
        print("      Most often a stale kernel module after a driver update.")
        print("      Compare these two and reboot if they differ:")
        print("        cat /proc/driver/nvidia/version   # module loaded in RAM")
        print("        modinfo -F version nvidia         # module on disk")
        return 1

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    gib = 1024**3

    print(f"device              {props.name}")
    print(f"compute capability  {props.major}.{props.minor}")
    print(f"VRAM                {total / gib:.1f} GiB total, {free / gib:.1f} GiB free")
    print(f"bf16 supported      {torch.cuda.is_bf16_supported()}")

    # Prove the device actually executes, not just that it enumerates.
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    assert torch.isfinite(x @ x).all()
    print("bf16 matmul         ok")

    for name in ("transformers", "datasets", "peft", "trl", "accelerate"):
        try:
            print(f"{name:<19} {__import__(name).__version__}")
        except ImportError:
            print(f"{name:<19} NOT INSTALLED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
