"""Train LoRA on Qwen3-4B-Thinking using SFT distillation pool.

Input: data/sft_train.jsonl from cse151b-prepare-sft (K=32 winning responses
on public minus val_ids, ~700-800 rows).

Output: lora_weights/<run_name>/ (PEFT adapter dir; merge with base for
inference via adapter_path).

Config defaults are tuned for 4B BF16 on 96GB Blackwell:
- LoRA r=32, alpha=64
- batch_size 4 (effective via grad accum)
- 3 epochs, lr 2e-4 cosine warmup
- BF16 throughout (no quantization — base model is BF16 from HF)
- Mask system+user tokens; only train on response tokens

Wallclock estimate: ~12-18h on Blackwell for ~800 rows × 3 epochs.

Usage:
    PYTHONPATH=src python scripts/train_lora.py \\
        --train data/sft_train.jsonl \\
        --output lora_weights/runj_distill_v1 \\
        [--r 32] [--alpha 64] [--epochs 3] [--lr 2e-4]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# Required imports — must be in train extra: peft datasets trl
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

# Avoid HF tokenizer warning noise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"


def _load_sft(path: pathlib.Path) -> list[dict]:
    """Load SFT jsonl. Each row: {id, system_prompt, user_prompt, target_response, ...}."""
    rows = [json.loads(line) for line in open(path)]
    print(f"[lora] loaded {len(rows)} SFT rows from {path}")
    return rows


def _format_chat(row: dict, tokenizer) -> str:
    """Format one row as Qwen chat-template string with system + user + assistant."""
    messages = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["user_prompt"]},
        {"role": "assistant", "content": row["target_response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="SFT jsonl from cse151b-prepare-sft")
    p.add_argument("--output", required=True, help="LoRA adapter output dir")
    p.add_argument("--r", type=int, default=32)
    p.add_argument("--alpha", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4, help="effective bsz = batch_size × grad_accum")
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--save-every", type=int, default=200, help="save checkpoint every N steps")
    args = p.parse_args()

    print(
        f"[lora] config: r={args.r} alpha={args.alpha} epochs={args.epochs} "
        f"lr={args.lr} bsz={args.batch_size}×{args.grad_accum}"
    )

    # Lazy imports (heavy)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    rows = _load_sft(pathlib.Path(args.train))
    dataset = Dataset.from_list([{"text": _format_chat(r, tokenizer)} for r in rows])
    print(f"[lora] formatted {len(dataset)} examples")

    # Base model in BF16 — 96GB Blackwell can hold it
    print(f"[lora] loading {MODEL_NAME} in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required for grad ckpt
    model.gradient_checkpointing_enable()

    # LoRA config — target Qwen attention + mlp
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # SFT config
    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        max_seq_length=args.max_seq_len,
        logging_steps=10,
        save_steps=args.save_every,
        save_total_limit=3,
        dataset_text_field="text",
        report_to="none",  # set to "wandb" if you want
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        tokenizer=tokenizer,
    )

    print(f"[lora] starting training, output → {args.output}")
    trainer.train()

    print(f"[lora] saving final adapter to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("[lora] done.")


if __name__ == "__main__":
    main()
