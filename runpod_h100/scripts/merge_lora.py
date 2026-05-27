"""Merge LoRA adapter into BF16 base model.

Run after train_lora_bf16.py to produce a standalone model usable by vLLM / for GRPO base.

Usage:
    python scripts/merge_lora.py
    python scripts/merge_lora.py --adapter checkpoints/lora_sft_h100/final --out checkpoints/lora_sft_merged
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=str(REPO / "checkpoints" / "lora_sft_h100" / "final"))
    p.add_argument("--out", default=str(REPO / "checkpoints" / "lora_sft_merged"))
    p.add_argument("--base", default=MODEL_ID)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading base BF16: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("Merging adapter weights into base...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {out}")
    merged.save_pretrained(str(out), safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.save_pretrained(str(out))

    import subprocess
    subprocess.run(["du", "-sh", str(out)])
    print("Done.")


if __name__ == "__main__":
    main()
