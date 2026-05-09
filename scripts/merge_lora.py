"""Merge LoRA adapter weights into the base model and save as a full model.

After merging, the resulting model can be loaded via vLLM/transformers
without any LoRA-specific code paths.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/merge_lora.py \\
        --base Qwen/Qwen3-4B-Thinking-2507 \\
        --adapter lora_weights/runj_distill_v1 \\
        --output lora_weights/runj_distill_v1_merged
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] base    = {args.base}")
    print(f"[merge] adapter = {args.adapter}")
    print(f"[merge] output  = {args.output}")

    print("[merge] loading base in BF16...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("[merge] loading adapter...")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("[merge] merging weights...")
    merged = model.merge_and_unload()

    print(f"[merge] saving merged model to {args.output}")
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    print("[merge] done.")


if __name__ == "__main__":
    main()
