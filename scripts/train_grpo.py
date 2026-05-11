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


def _build_dataset(
    public_path: pathlib.Path,
    val_path: pathlib.Path | None,
    max_prompts: int,
    pool_path: pathlib.Path | None = None,
):
    """Load public.jsonl, exclude val_ids, format as GRPO-ready dataset.

    If pool_path is provided, only keep prompts whose id is in pool_path's
    "prompt_ids" — used for edge-filtered training (avoids zero-std groups).
    """
    from cse151b_comp.prompts import build_prompt_runf

    rows = [json.loads(line) for line in open(public_path)]
    val_ids: set = set()
    if val_path is not None and val_path.exists():
        val_ids = set(json.load(open(val_path))["val_ids"])

    pool_ids: set | None = None
    if pool_path is not None and pool_path.exists():
        pool_ids = set(json.load(open(pool_path))["prompt_ids"])
        print(f"[grpo] using edge-filtered pool: {len(pool_ids)} ids from {pool_path}")

    train = []
    for r in rows:
        if r["id"] in val_ids:
            continue
        if pool_ids is not None and r["id"] not in pool_ids:
            continue
        if not r.get("answer"):
            continue
        system, user = build_prompt_runf(r["question"], r.get("options"))
        train.append(
            {
                "prompt": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "answer": json.dumps(r["answer"]),
                "options": json.dumps(r.get("options") or []),
                "id": r["id"],
            }
        )
        if max_prompts is not None and len(train) >= max_prompts:
            break

    return train


def _make_reward_fn():
    """Build a reward function that judges each completion via course Judger.

    Per-completion timeout (15s) prevents sympy infinite loops from hanging
    training. Observed: course Judger's internal SIGALRM timeout doesn't
    always fire (multi-thread sympy bypasses signal). Use multiprocessing
    Process.join(timeout) for hard kill.
    """

    from cse151b_comp.evaluate import score_response
    from judger import Judger

    # Worker process function — must be top-level for spawn
    def _judge_worker(text, ans, opt, q):
        try:
            from cse151b_comp.evaluate import score_response as _sr
            from judger import Judger as _J

            j = _J(strict_extract=False)
            ok = _sr(text, ans, opt, j)
            q.put(1.0 if ok else 0.0)
        except Exception:
            q.put(0.0)

    # Fallback: in-process judge (fast path, no multiprocessing overhead)
    judger_local = Judger(strict_extract=False)

    def _judge_with_timeout(text, ans, opt, timeout=15):
        """Try in-process; if it hangs, fall through to subprocess kill."""
        # In-process attempt with signal-based timeout (SIGALRM)
        import signal

        class _TimeoutErr(Exception):
            pass

        def _handler(signum, frame):
            raise _TimeoutErr()

        old = signal.signal(signal.SIGALRM, _handler)
        try:
            signal.alarm(timeout)
            ok = score_response(text, ans, opt, judger_local)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
            return 1.0 if ok else 0.0
        except _TimeoutErr:
            signal.signal(signal.SIGALRM, old)
            return 0.0  # treat as wrong on timeout
        except Exception:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
            return 0.0

    def reward_correct(completions, answer, options, **kwargs):
        """Length-aware binary reward.

        - Correct + len >= 2000 chars (~500 tokens reasoning): 1.0
        - Correct + len < 2000 chars: 0.5  (anti-collapse: previous run
          collapsed mean_terminated_length 2680→1345 tokens, -10pp on MC.
          Penalize short correct answers to keep reasoning depth.)
        - Wrong: 0.0
        """
        rewards = []
        for c, ans_str, opt_str in zip(completions, answer, options):
            try:
                ans = json.loads(ans_str)
                opt = json.loads(opt_str) if opt_str else None
                if isinstance(c, list) and c and isinstance(c[0], dict):
                    text = c[-1].get("content", "")
                else:
                    text = c
                base = _judge_with_timeout(text, ans, opt, timeout=15)
                if base > 0.5 and len(text) < 2000:
                    base = 0.5
                rewards.append(base)
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
    p.add_argument(
        "--pool",
        default=None,
        help="Optional JSON with {prompt_ids:[...]} to filter to edge-of-difficulty prompts. "
        "Without this, GRPO collapses with frac_reward_zero_std → 1.0 (no gradient).",
    )
    p.add_argument("--epochs", type=int, default=4, help="More epochs = longer + more refinement")
    p.add_argument("--num-generations", type=int, default=8, help="K samples per prompt (group size)")
    p.add_argument("--lr", type=float, default=1e-5, help="Higher than first run (3e-6 was too small)")
    p.add_argument("--beta", type=float, default=0.0, help="KL penalty vs reference (0 = let policy drift freely)")
    p.add_argument("--use-vllm", action="store_true", default=True, help="Use vLLM colocate for fast sampling")
    p.add_argument(
        "--max-completion-length",
        type=int,
        default=6144,
        help="Max generation length. Longer = slower per step but allows longer reasoning.",
    )
    p.add_argument("--temperature", type=float, default=1.0, help="Higher = more variance in K samples (was 0.7)")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--r", type=int, default=16, help="LoRA rank (smaller than SFT for stability)")
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8, help="bsz × grad_accum must be divisible by num_generations")
    p.add_argument("--lr-scheduler", default="constant", help="constant avoids cosine decay to ~0")
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--loss-type", default="dr_grpo", help="dr_grpo = length-normalized; prevents length collapse")
    p.add_argument("--importance-sampling-level", default="sequence", help="sequence-level IS = stabler than token")
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--epsilon-high", type=float, default=0.4, help="Asymmetric clipping (DAPO-style)")
    p.add_argument("--top-entropy-quantile", type=float, default=1.0, help="<1 to focus loss on uncertain tokens")
    p.add_argument(
        "--disable-vllm-is-correction",
        action="store_true",
        help="Disable vLLM IS correction (default sequence_mask mode kills gradient on long sequences).",
    )
    args = p.parse_args()

    print(f"[grpo] base = {args.base}")
    print(f"[grpo] output = {args.output}")
    print(
        f"[grpo] config: K={args.num_generations} epochs={args.epochs} "
        f"lr={args.lr} sched={args.lr_scheduler} beta={args.beta} temp={args.temperature} "
        f"loss={args.loss_type} is_level={args.importance_sampling_level} eps={args.epsilon}/{args.epsilon_high}"
    )

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
        pathlib.Path(args.pool) if args.pool else None,
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

    # GRPO config — fixes after first run failed (see HANDOFF):
    #   - constant lr (was cosine→2e-11), higher base lr 1e-5 (was 3e-6)
    #   - temperature 1.0 (was 0.7) → more variance per group → non-zero advantage
    #   - dr_grpo loss (was dapo) → length-normalized, prevents 4070→2574 collapse
    #   - sequence-level importance sampling → stabler than token-level under vLLM colocate
    #   - asymmetric clipping eps_high=0.4 → allow larger updates on positive advantage
    #   - beta=0.0 (was 0.04) → no KL drag, drift freely from LoRA-v1 base
    grpo_config_kwargs = dict(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        # GRPO-specific
        num_generations=args.num_generations,
        beta=args.beta,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        loss_type=args.loss_type,
        importance_sampling_level=args.importance_sampling_level,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        top_entropy_quantile=args.top_entropy_quantile,
        scale_rewards="none",  # Dr.GRPO recommends no group-std normalization (was 'group')
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
        if args.disable_vllm_is_correction:
            # Default sequence_mask correction multiplies per-token logp diffs
            # across the full completion. With ~5000-token completions and 0.015
            # nat/tok diff, the cumulative ratio collapses to ~exp(-75) ≈ 0.
            grpo_config_kwargs["vllm_importance_sampling_correction"] = False
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
