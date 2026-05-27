"""BF16 LoRA SFT training on H100 80GB.

Differences from QLoRA (4090) version:
  - BF16 base (no 4-bit quantization)
  - per_device_batch_size=4 (vs 1 on 4090)
  - max_seq=8192 (vs 5120 on 4090)
  - 5 epochs, r=64, alpha=128
  - completion_only_loss=True (Run F prompt template applied via data prep)

Usage:
    python scripts/train_lora_bf16.py

Output:
    checkpoints/lora_sft_h100/final/  (adapter)
"""
import argparse, json, os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

REPO = Path(__file__).resolve().parent.parent
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
SFT_FILE = REPO / "data" / "h100_lora_sft.jsonl"
OUT_DIR = REPO / "checkpoints" / "lora_sft_h100"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-steps", type=int, default=-1, help="-1 for full training")
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--per-device-bsz", type=int, default=2,
                   help="2 is safe on H100 80GB with seq=8192. Try 4 if you have memory headroom")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Effective bsz = per_device_bsz × grad_accum (target 8)")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Enable gradient checkpointing (safer but slower). Default off on H100.")
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading BF16 base model (no quantization) on H100")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    print(f"Wrapping with LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print(f"Loading SFT dataset: {SFT_FILE}")
    rows = [json.loads(l) for l in open(SFT_FILE)]
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    print(f"  Rows: {len(ds)}")

    cfg = SFTConfig(
        output_dir=str(OUT_DIR),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_bsz,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.grad_checkpoint,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        max_length=args.max_seq,
        bf16=True,
        optim="adamw_torch",  # Full BF16, no paged 8-bit needed
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        save_strategy="steps",
        report_to=[],
        seed=args.seed,
        dataset_text_field=None,
        packing=False,
        completion_only_loss=True,  # ⭐ Critical
    )

    print(f"Effective batch size: {args.per_device_bsz * args.grad_accum}")
    print(f"Optim steps per epoch: ~{len(ds) // (args.per_device_bsz * args.grad_accum)}")

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final adapter → {OUT_DIR}/final")
    trainer.model.save_pretrained(str(OUT_DIR / "final"))
    tok.save_pretrained(str(OUT_DIR / "final"))
    print("Done.")


if __name__ == "__main__":
    main()
