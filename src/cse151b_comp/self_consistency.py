"""Self-consistency inference: K samples per question, voted per type.

Generates K samples per question in a single vLLM call (using the engine's
``n=K`` parameter, which shares the prompt prefix and only diverges KV
cache for the generation), then votes across samples with a strategy
appropriate to the question type.

Phase 2 entry point per :file:`reports/public_private_gap_analysis.md`.
The base prompt defaults to the Phase 0 starter prompts (highest single-
shot leaderboard score, 0.575), since the Phase 1 rule additions did not
transfer to the private set. Pass ``--prompt current`` to use the current
:mod:`cse151b_comp.prompts` instead.

CLI::

    PYTHONPATH=src .venv/bin/python -m cse151b_comp.self_consistency \\
        --input data/public.jsonl \\
        --output results/sc_k8.jsonl \\
        --k 8 \\
        --temperature 0.7

    # quick experiment on a 100-question subset
    PYTHONPATH=src .venv/bin/python -m cse151b_comp.self_consistency \\
        --input data/public.jsonl --output results/sc_k4_quick.jsonl \\
        --k 4 --limit 100

Output JSONL schema (per question)::

    {
      "id": 42,
      "question_type": "free_single",
      "all_responses": [response_1, ..., response_K],
      "all_extracted": [...K canonical-form strings or tuples...],
      "vote_counts": {"<canonical>": count, ...},
      "winning_answer": "<canonical winner>",
      "winning_response": "<the response shown to grader>",
      "answer": [...gold (if input had it)...],
      "correct": true|false,
      "solvable_but_missed": true|false  (only if gold was provided)
    }
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

from cse151b_comp.extract import (
    extract_all_final_boxed,
    extract_letter,
    normalize_answer,
)
from cse151b_comp.prompts import detect_question_type
from cse151b_comp.voting import (
    _extract_tuple,
    solvable_but_missed,
    vote_free_multi,
    vote_free_single,
    vote_mcq,
)


# ─── Prompt selection ──────────────────────────────────────────────────────

_STARTER_SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a "
    "single \\boxed{}, e.g. \\boxed{3, 7}."
)

_STARTER_SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best "
    "answer. Output ONLY the letter of your chosen option inside \\boxed{}, "
    "e.g. \\boxed{C}."
)


def _build_starter_prompt(question: str, options: list[str] | None) -> tuple[str, str]:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return _STARTER_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return _STARTER_SYSTEM_PROMPT_MATH, question


def _select_prompt_builder(name: str):
    """Return ``(question, options) -> (system, user)`` for a named variant."""
    if name == "phase0":
        return _build_starter_prompt
    if name == "current":
        from cse151b_comp.prompts import build_prompt
        return build_prompt
    if name == "runb":
        from cse151b_comp.prompts import build_prompt_runb
        return build_prompt_runb
    if name == "runc":
        from cse151b_comp.prompts import build_prompt_runc
        return build_prompt_runc
    raise ValueError(f"Unknown --prompt {name!r}; choices: phase0, current, runb, runc")


# ─── Question type ─────────────────────────────────────────────────────────


def question_type(item: dict) -> str:
    """Question-type label based on the question text alone.

    Mirrors :func:`cse151b_comp.prompts.detect_question_type` so that the
    voting strategy matches the system prompt routing. The earlier
    gold-based heuristic silently collapsed every free-form question into
    ``free_single`` on the private set (no gold present), causing the
    free_multi voting branch to never fire — see
    ``reports/public_private_gap_analysis.md`` and the v6_sc_k8 0.448
    leaderboard regression.
    """
    return detect_question_type(item.get("question", ""), item.get("options"))


# ─── Per-sample extraction (shared with voting helpers) ────────────────────


def _per_sample_extract(qtype: str, response: str):
    """Return the canonical extracted answer from a single response."""
    if qtype == "mc":
        return extract_letter(response)
    if qtype == "free_single":
        boxes = extract_all_final_boxed(response)
        return normalize_answer(boxes[-1]) if boxes else ""
    # free_multi
    return _extract_tuple(response) or ()


# ─── Gold normalization for solvable-but-missed and correctness ────────────


def _normalize_gold(qtype: str, gold) -> str | tuple[str, ...]:
    if qtype == "mc":
        return str(gold).strip().upper()
    if qtype == "free_single":
        if isinstance(gold, list):
            return normalize_answer(gold[0]) if gold else ""
        return normalize_answer(gold)
    # free_multi
    if isinstance(gold, list):
        return tuple(normalize_answer(g) for g in gold)
    return (normalize_answer(gold),)


# ─── Main ─────────────────────────────────────────────────────────────────


def _setup_env(gpu_id: str = "0") -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM self-consistency inference (n=K + voting).")
    p.add_argument("--input", required=True, help="Input JSONL with id+question(+options)(+answer).")
    p.add_argument("--output", required=True, help="Output JSONL with vote results.")
    p.add_argument("--k", type=int, default=8, help="Samples per question.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=12288, dest="max_tokens")
    p.add_argument("--prompt", default="phase0",
                   choices=["phase0", "current", "runb", "runc"],
                   help="Which prompt set to use. phase0 = starter (v5_sanity 0.583), "
                        "current = v6 per-type (0.448, retired), "
                        "runb = Phase 0 + anti-pattern + symbolic preference (0.600), "
                        "runc = Run B + end-with-box + text/bool examples.")
    p.add_argument("--per-type-budget", action="store_true",
                   help="Use cse151b_comp.budget.allocate_max_tokens per question instead "
                        "of the flat --max-tokens value. Overrides --max-tokens.")
    p.add_argument("--limit", type=int, default=None, help="Only run on first N rows (debug).")
    p.add_argument("--val", default=None, help="Optional val_indices.json to filter --input.")
    p.add_argument("--gpu-mem-util", type=float, default=0.70)
    p.add_argument("--max-model-len", type=int, default=20480)
    p.add_argument("--max-num-seqs", type=int, default=16,
                   help="vLLM max concurrent sequences. Lower for K=8 to keep KV cache feasible.")
    p.add_argument("--seed", type=int, default=42, help="Base seed; per-sample seed varies internally.")
    args = p.parse_args()

    _setup_env()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in open(args.input)]
    if args.val:
        val_ids = set(json.loads(pathlib.Path(args.val).read_text())["val_ids"])
        rows = [r for r in rows if r["id"] in val_ids]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[sc] Loaded {len(rows)} rows. K={args.k}, T={args.temperature}, top_p={args.top_p}.")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model="Qwen/Qwen3-4B-Thinking-2507",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=True,        # K samples share prompt prefix
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_model_len,
        seed=args.seed,
    )
    build_prompt = _select_prompt_builder(args.prompt)
    prompts = []
    per_prompt_max_tokens: list[int] = []
    if args.per_type_budget:
        from cse151b_comp.budget import allocate_max_tokens
    for item in rows:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        ))
        if args.per_type_budget:
            per_prompt_max_tokens.append(
                allocate_max_tokens(item["question"], item.get("options"))
            )

    if args.per_type_budget:
        sampling_params = [
            SamplingParams(
                n=args.k,
                max_tokens=mt,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0,
                seed=args.seed,
            )
            for mt in per_prompt_max_tokens
        ]
        budget_lo = min(per_prompt_max_tokens)
        budget_hi = max(per_prompt_max_tokens)
        print(f"[sc] Per-type budget enabled: max_tokens range [{budget_lo}, {budget_hi}]")
    else:
        sampling_params = SamplingParams(
            n=args.k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            seed=args.seed,
        )

    print(f"[sc] Generating {len(prompts)} × n={args.k} = {len(prompts)*args.k} samples...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    print(f"[sc] Generation done in {(time.time() - t0) / 60:.1f} min")

    # Process results.
    out_rows = []
    n_solvable_but_missed = 0
    n_correct = 0
    n_with_gold = 0
    type_counts = {"mc": 0, "free_single": 0, "free_multi": 0}

    for item, out in zip(rows, outputs):
        qtype = question_type(item)
        type_counts[qtype] += 1
        responses = [c.text.strip() for c in out.outputs]

        if qtype == "mc":
            winning, vote_counts, win_idx = vote_mcq(responses)
            extracted = [extract_letter(r) for r in responses]
        elif qtype == "free_single":
            winning, vote_counts, win_idx = vote_free_single(responses)
            extracted = []
            for r in responses:
                boxes = extract_all_final_boxed(r)
                extracted.append(normalize_answer(boxes[-1]) if boxes else "")
        else:  # free_multi
            winning, vote_counts, win_idx = vote_free_multi(responses)
            extracted = [_extract_tuple(r) or () for r in responses]

        winning_response = responses[win_idx] if responses else ""

        record: dict = {
            "id": item["id"],
            "question_type": qtype,
            "all_responses": responses,
            "all_extracted": [
                list(e) if isinstance(e, tuple) else e for e in extracted
            ],
            "vote_counts": vote_counts,
            "winning_answer": list(winning) if isinstance(winning, tuple) else winning,
            "winning_response": winning_response,
            "K": args.k,
        }

        # Optional: gold-based correctness + solvable_but_missed.
        gold = item.get("answer")
        if gold is not None:
            gold_norm = _normalize_gold(qtype, gold)
            record["answer"] = gold
            record["correct"] = (winning == gold_norm)
            record["solvable_but_missed"] = solvable_but_missed(
                extracted, winning, gold_norm
            )
            n_with_gold += 1
            if record["correct"]:
                n_correct += 1
            if record["solvable_but_missed"]:
                n_solvable_but_missed += 1

        out_rows.append(record)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"[sc] Wrote {len(out_rows)} rows → {out_path}")
    print(f"[sc] Question-type counts: {type_counts}")
    if n_with_gold:
        print(f"[sc] Accuracy: {n_correct}/{n_with_gold} = {n_correct/n_with_gold*100:.2f}%")
        print(f"[sc] Solvable-but-missed: {n_solvable_but_missed}/{n_with_gold} "
              f"= {n_solvable_but_missed/n_with_gold*100:.2f}%  "
              f"(upper bound on gain from better voting)")


if __name__ == "__main__":
    main()
