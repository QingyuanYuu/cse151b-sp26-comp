"""Merge the best GRPO LoRA adapter (picked by val_225 sweep) into the SFT-merged BF16 base.

Reads results/grpo_ckpt_sweep_summary.json -> best.adapter_path
Merges with checkpoints/lora_sft_merged -> outputs checkpoints/grpo_v6_merged/

Usage:
    python scripts/merge_best_grpo.py
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

REPO = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default=str(REPO / "results" / "grpo_ckpt_sweep_summary.json"),
                   help="Output of eval_all_ckpts.py")
    p.add_argument("--base", default=str(REPO / "checkpoints" / "lora_sft_merged"),
                   help="SFT-merged BF16 base (the model that GRPO trained on top of)")
    p.add_argument("--out", default=str(REPO / "checkpoints" / "grpo_v6_merged"),
                   help="Final merged BF16 model directory")
    p.add_argument("--adapter", default=None,
                   help="Override: explicit adapter path (skip summary lookup)")
    args = p.parse_args()

    if args.adapter:
        adapter_path = args.adapter
        print(f"[merge_best] using explicit adapter: {adapter_path}")
    else:
        with open(args.summary) as f:
            s = json.load(f)
        best = s["best"]
        adapter_path = best["adapter_path"]
        print(f"[merge_best] best ckpt: label={best['label']}, val_acc={best['overall_acc']*100:.2f}%")
        print(f"            adapter: {adapter_path}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[merge_best] loading SFT-merged base BF16: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    print(f"[merge_best] applying LoRA adapter")
    model = PeftModel.from_pretrained(base, adapter_path)

    print(f"[merge_best] merging adapter weights into base...")
    merged = model.merge_and_unload()

    print(f"[merge_best] saving merged model to: {out}")
    merged.save_pretrained(str(out), safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tok.save_pretrained(str(out))

    import subprocess
    subprocess.run(["du", "-sh", str(out)])
    print("[merge_best] Done.")


if __name__ == "__main__":
    main()
