"""GRPO training on Qwen3-4B-Thinking using Judger.auto_judge as binary reward.

Continues from LoRA v1 merged model. Adds new LoRA adapter on top, trains
via Group Relative Policy Optimization with the course Judger giving
binary 1.0/0.0 rewards.

Pipeline:
1. Build train dataset from data/public.jsonl (excluding val_ids).
   Each row: {prompt: <Run F system + user>, answer: gold, options: ...}
2. For each prompt, sample K=4 completions during training.
3. Score each completion via Judger.auto_judge → reward 1.0 if correct, 0.0 if wrong.
4. GRPO group-normalizes rewards within the K samples → advantage.
5. Update policy LoRA to favor higher-reward completions.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/train_grpo.py \\
        --base lora_weights/runj_distill_v1_merged \\
        --output lora_weights/runj_grpo_v1 \\
        [--num-generations 4] [--epochs 1] [--lr 5e-6] [--max-prompts 600]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _build_dataset(public_path: pathlib.Path, val_path: pathlib.Path | None, max_prompts: int):
    """Load public.jsonl, exclude val_ids, format as GRPO-ready dataset.

    Each example: {prompt, answer, options, id}. The 'prompt' is the
    Run F system+user formatted as a chat-template string.
    """
    from cse151b_comp.prompts import build_prompt_runf

    rows = [json.loads(line) for line in open(public_path)]
    val_ids: set = set()
    if val_path is not None and val_path.exists():
        val_ids = set(json.load(open(val_path))["val_ids"])

    train = []
    for r in rows:
        if r["id"] in val_ids:
            continue
        # Skip rows without answer (private set has none, but public.jsonl has all)
        if not r.get("answer"):
            continue
        system, user = build_prompt_runf(r["question"], r.get("options"))
        train.append(
            {
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "answer": json.dumps(r["answer"]),  # serialize for safety
                "options": json.dumps(r.get("options") or []),
                "id": r["id"],
            }
        )
        if max_prompts is not None and len(train) >= max_prompts:
            break

    return train


def _make_reward_fn():
    """Build a reward function that judges each completion via course Judger."""
    from cse151b_comp.evaluate import score_response
    from judger import Judger

    judger = Judger(strict_extract=False)

    def reward_correct(completions, answer, options, **kwargs):
        """Return 1.0 for correct completion, 0.0 otherwise.

        completions: list of generated assistant strings
        answer: list of gold answers (one per completion in batch order)
        options: list of options (or empty lists for free-form)
        """
        rewards = []
        for c, ans_str, opt_str in zip(completions, answer, options):
            try:
                ans = json.loads(ans_str)
                opt = json.loads(opt_str) if opt_str else None
                # If completion is a list of message dicts (chat format), extract content
                if isinstance(c, list) and c and isinstance(c[0], dict):
                    text = c[-1].get("content", "")
                else:
                    text = c
                ok = score_response(text, ans, opt, judger)
                rewards.append(1.0 if ok else 0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    return reward_correct


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base", default="lora_weights/runj_distill_v1_merged", help="Base model path (LoRA v1 merged or pure base)"
    )
    p.add_argument("--output", required=True, help="GRPO LoRA adapter output dir")
    p.add_argument("--public", default="data/public.jsonl")
    p.add_argument("--val", default="data/val_indices.json")
    p.add_argument("--max-prompts", type=int, default=900, help="Subset size (default 900 = nearly all train)")
    p.add_argument("--epochs", type=int, default=4, help="More epochs = longer + more refinement")
    p.add_argument("--num-generations", type=int, default=8, help="K samples per prompt (group size)")
    p.add_argument("--lr", type=float, default=3e-6, help="Conservative for stability over long training")
    p.add_argument("--beta", type=float, default=0.04, help="KL penalty vs reference model (drift control)")
    p.add_argument("--use-vllm", action="store_true", default=True, help="Use vLLM colocate for fast sampling")
    p.add_argument(
        "--max-completion-length",
        type=int,
        default=6144,
        help="Max generation length. Longer = slower per step but allows longer reasoning.",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--r", type=int, default=16, help="LoRA rank (smaller than SFT for stability)")
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8, help="bsz × grad_accum must be divisible by num_generations")
    args = p.parse_args()

    print(f"[grpo] base = {args.base}")
    print(f"[grpo] output = {args.output}")
    print(f"[grpo] config: K={args.num_generations} epochs={args.epochs} " f"lr={args.lr} beta={args.beta} r={args.r}")

    # Lazy heavy imports
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    # Tokenizer (same for base and LoRA-merged)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build training data
    print(f"[grpo] loading prompts from {args.public}...")
    train_rows = _build_dataset(
        pathlib.Path(args.public),
        pathlib.Path(args.val) if args.val else None,
        args.max_prompts,
    )
    print(f"[grpo] train prompts: {len(train_rows)}")
    train_ds = Dataset.from_list(train_rows)

    # Reward function
    reward_fn = _make_reward_fn()

    # LoRA on top of (potentially already-LoRA-merged) base
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
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

    # GRPO config
    grpo_config_kwargs = dict(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        # GRPO-specific
        num_generations=args.num_generations,
        beta=args.beta,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        # Standard
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=args.epochs,
        report_to="none",
        seed=42,
    )
    # vLLM colocate for fast sampling (~5-15s per group vs 30-60s with HF)
    if args.use_vllm:
        grpo_config_kwargs.update(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.4,  # leave room for trainer's policy + ref
            vllm_max_model_length=8192,  # prompt + completion
        )
    grpo_config = GRPOConfig(**grpo_config_kwargs)

    # GRPOTrainer
    print("[grpo] initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=args.base,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"[grpo] starting training, output → {args.output}")
    trainer.train()

    print(f"[grpo] saving final adapter to {args.output}")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("[grpo] done.")


if __name__ == "__main__":
    main()
