"""GRPO training on H100 using grpo_train_extended_v6.jsonl (305 prompts).

Dataset composition:
  - 196 public edge-filtered (real gold from public.jsonl)
  - 80 verified private uncertain (hand-verified + Judger-friendly format)
  - 25 hybrid+solved dual-verified (two independent pipelines agree)
  - 4 lora_solved hand-verified (single-pipeline labels, hand-verified by sympy)

Uses our merged LoRA SFT model as base (or the v3.5 merged from local).

Usage:
    python scripts/train_grpo.py --base checkpoints/lora_sft_merged
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def _load_dataset(
    jsonl_path: pathlib.Path,
    max_prompts: int | None = None,
    hard_pool_path: pathlib.Path | None = None,
    hard_dup: int = 0,
) -> list[dict]:
    """Load grpo_train_extended_v6.jsonl format.

    Each row already has: id, source, prompt (chat messages), answer (json), options (json).

    If `hard_pool_path` is provided AND `hard_dup > 0`, prompts whose pid (parsed from
    "public_<pid>" id field) is in the hard set are duplicated `hard_dup` extra times.
    This emulates "higher K for hard prompts" since TRL's num_generations is global.

    Examples:
      hard_dup=1 → hard prompts visited 2× per epoch → effective K_hard = 2 × global K
      hard_dup=0 → no duplication, all prompts visited once
    """
    rows = [json.loads(line) for line in open(jsonl_path)]
    if max_prompts:
        rows = rows[:max_prompts]
    print(f"[grpo] loaded {len(rows)} training prompts from {jsonl_path}")
    src_counts = {}
    for r in rows:
        src = r.get("source", "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1
    for s, n in src_counts.items():
        print(f"  {s:<40s} {n}")

    if hard_pool_path and hard_dup > 0:
        with open(hard_pool_path) as f:
            hard_ids = set(json.load(f)["hard_ids"])
        n_dup = 0
        extra = []
        for r in rows:
            # Hard IDs live in the PUBLIC namespace only. Private prompts have a
            # separate integer ID space — must not match them.
            rid = r.get("id", "")
            if not rid.startswith("public_"):
                continue
            try:
                pid = int(rid[len("public_"):])
            except ValueError:
                continue
            if pid in hard_ids:
                for _ in range(hard_dup):
                    extra.append(r)
                n_dup += 1
        rows = rows + extra
        print(f"[grpo] hard-pool boost: {n_dup} hard prompts duplicated {hard_dup}× → "
              f"+{len(extra)} extra rows; total now {len(rows)}")
        print(f"[grpo] effective K on hard = {hard_dup + 1} × global_K")
    return rows


def _make_reward_fn():
    """Length-aware binary reward via course Judger."""
    from cse151b_comp.evaluate import score_response
    from judger import Judger
    import signal
    judger_local = Judger(strict_extract=False)

    def _judge_with_timeout(text, ans, opt, timeout=15):
        class _TimeoutErr(Exception): pass
        def _handler(signum, frame): raise _TimeoutErr()
        old = signal.signal(signal.SIGALRM, _handler)
        try:
            signal.alarm(timeout)
            ok = score_response(text, ans, opt, judger_local)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
            return 1.0 if ok else 0.0
        except _TimeoutErr:
            signal.signal(signal.SIGALRM, old)
            return 0.0
        except Exception:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
            return 0.0

    def reward_correct(completions, answer, options, **kwargs):
        rewards = []
        for c, ans_str, opt_str in zip(completions, answer, options):
            try:
                ans = json.loads(ans_str)
                opt = json.loads(opt_str) if opt_str else None
                text = c[-1].get("content", "") if isinstance(c, list) and c and isinstance(c[0], dict) else c
                base = _judge_with_timeout(text, ans, opt, timeout=15)
                # Length-aware anti-collapse — only on free-response (MCQ exempt).
                # MCQ correct answers like "C" or "C, A" have intrinsically short
                # reasoning; penalizing them encourages padding/hallucinated steps.
                is_mcq = bool(opt)
                if base > 0.5 and not is_mcq and len(text) < 2000:
                    base = 0.5
                rewards.append(base)
            except Exception:
                rewards.append(0.0)
        return rewards

    return reward_correct


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base merged model path")
    p.add_argument("--output", default=str(REPO / "checkpoints" / "grpo_v6"))
    p.add_argument("--data", default=str(REPO / "data" / "grpo_train_extended_v6.jsonl"))
    p.add_argument("--num-generations", type=int, default=4,
                   help="K rollouts per prompt. With --hard-dup 1 (default), hard subset is "
                        "duplicated → hard K_eff = 8, rest K = 4. To go uniform K=6, set this 6 "
                        "and --hard-dup 0.")
    p.add_argument("--hard-pool",
                   default=str(REPO / "data" / "hard_prompt_ids.json"),
                   help="JSON with {'hard_ids': [...]} marking sparse-reward prompts to duplicate.")
    p.add_argument("--hard-dup", type=int, default=1,
                   help="Extra copies of each hard prompt in training data. 1 means visited 2× per "
                        "epoch (effective K = 2 × global). Set 0 to disable.")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--max-completion-length", type=int, default=10240,
                   help="Max generated tokens per rollout. 6144 was too tight: SFT data has "
                        "P85≈8K and P90≈12K response lengths, so 6144 was truncating most hard-prompt "
                        "rollouts before they could write \\boxed{}. 10240 covers P85 reasoning, "
                        "+~2h training time. vllm_max_model_len auto-derives from this + max_prompt.")
    p.add_argument("--max-prompt-length", type=int, default=2048,
                   help="Run F system + question fit easily (median ~400 tokens). Don't increase.")
    p.add_argument("--per-device-bsz", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Must be divisible by num_generations. 8 with K=8 → 1 fwd accumulation per update; "
                        "use 16 with K=8 if you want 2 unique prompts per gradient step.")
    p.add_argument("--beta", type=float, default=0.04,
                   help="KL penalty toward reference policy. 0.04 lightly anchors a Thinking model "
                        "to its SFT distribution, preventing collapse of long reasoning chains.")
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--epsilon-high", type=float, default=0.4)
    p.add_argument("--warmup-ratio", type=float, default=0.05,
                   help="Linear warmup over first 5% of steps. Avoids early-step gradient spikes.")
    p.add_argument("--save-steps", type=int, default=20,
                   help="Save checkpoint every N steps. With ~114 total steps, 20 → 5-6 checkpoints.")
    p.add_argument("--save-total-limit", type=int, default=10,
                   help="Keep up to 10 checkpoints so you can pick the best by val accuracy post-train.")
    p.add_argument("--importance-sampling-level", default="sequence",
                   help="'sequence' is stabler than 'token' under vLLM colocate (Blackwell GRPO v2 used this)")
    p.add_argument("--scale-rewards", default="none",
                   help="dr_grpo recommends 'none' to avoid group-std normalization canceling signal")
    args = p.parse_args()

    from transformers import AutoTokenizer
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    print(f"Loading tokenizer from: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading training data: {args.data}")
    hp = pathlib.Path(args.hard_pool) if args.hard_pool else None
    rows = _load_dataset(
        pathlib.Path(args.data),
        args.max_prompts,
        hard_pool_path=hp if hp and hp.exists() else None,
        hard_dup=args.hard_dup,
    )

    # Build HF dataset
    from datasets import Dataset
    train_ds = Dataset.from_list([
        {
            "prompt": r["prompt"],
            "answer": r["answer"],
            "options": r["options"],
        }
        for r in rows
    ])
    print(f"[grpo] dataset built: {len(train_ds)} rows")

    print(f"Setting up GRPO config")
    # LoRA on top of base
    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    grpo_cfg = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_bsz,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.0,
        bf16=True,
        optim="adamw_torch",
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        report_to=["tensorboard"],
        logging_dir=str(pathlib.Path(args.output) / "logs"),
        seed=42,
        # GRPO-specific
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        beta=args.beta,
        loss_type="dr_grpo",
        importance_sampling_level=args.importance_sampling_level,  # sequence-level IS
        scale_rewards=args.scale_rewards,                          # 'none' for dr_grpo
        # vLLM for fast generation
        use_vllm=True,
        vllm_mode="colocate",
    )

    reward_fn = _make_reward_fn()

    print(f"Initializing GRPOTrainer with base={args.base}")
    trainer = GRPOTrainer(
        model=args.base,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=train_ds,
        peft_config=lora_cfg,
    )

    print("Starting GRPO training...")
    trainer.train()

    final = pathlib.Path(args.output) / "final"
    final.mkdir(parents=True, exist_ok=True)
    print(f"Saving final adapter → {final}")
    trainer.model.save_pretrained(str(final))
    tok.save_pretrained(str(final))
    print("Done.")


if __name__ == "__main__":
    main()
