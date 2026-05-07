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

from cse151b_comp.evaluate import score_response
from cse151b_comp.extract import (
    extract_all_final_boxed,
    extract_letter,
    normalize_answer,
)
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
    # Run B/C/D/E variants live in prompts.py and are self-contained — each
    # is a (question, options) -> (system, user) function with no extra state.
    if name in ("runb", "runc", "rund", "rune", "runf", "rung", "runh", "runi"):
        import cse151b_comp.prompts as _p

        return getattr(_p, f"build_prompt_{name}")
    # Run J = data-driven topic routing (see src/cse151b_comp/runj.py).
    # Sub-variants for ablation: runj_olympiad / runj_trig / runj_geom /
    # runj_stats_hyp / runj_stats_reg / runj_stats_desc / runj_stats
    # (combined) / runj_calc / runj_prob / runj_number_alg. Each enables
    # one branch (or one group), others fall through to Run F generic.
    # `runj` is the full 9-branch design; `runj_final` is dynamically
    # composed from ablation results.
    if name.startswith("runj_v2"):
        import cse151b_comp.runj_v2 as _rj2

        attr = "build_prompt_" + name
        if hasattr(_rj2, attr):
            return getattr(_rj2, attr)
        raise ValueError(
            f"Unknown Run J v2 variant {name!r}. Available: runj_v2, "
            f"runj_v2_olympiad, runj_v2_trig, runj_v2_geom, runj_v2_stats_hyp, "
            f"runj_v2_stats_reg, runj_v2_stats_desc, runj_v2_calc, runj_v2_prob, "
            f"runj_v2_number_alg"
        )
    if name.startswith("runj"):
        import cse151b_comp.runj as _rj

        attr = "build_prompt_" + name  # build_prompt_runj or _runj_trig etc.
        if hasattr(_rj, attr):
            return getattr(_rj, attr)
        raise ValueError(
            f"Unknown Run J variant {name!r}. Available: runj, runj_final, "
            f"runj_olympiad, runj_trig, runj_geom, runj_stats_hyp, "
            f"runj_stats_reg, runj_stats_desc, runj_stats, runj_calc, "
            f"runj_prob, runj_number_alg"
        )
    raise ValueError(
        f"Unknown --prompt {name!r}; choices: phase0, current, runb, runc, rund, "
        f"rune, runf, rung, runh, runi, runj plus runj_* ablation variants"
    )


# ─── Question type ─────────────────────────────────────────────────────────


def question_type(item: dict) -> str:
    """Decide which voting strategy to use for this question.

    **Must mirror prompts.detect_question_type** (which decides which system
    prompt to use). If the two diverge, samples are generated under one
    routing assumption and voted under another — the bug we hit on private
    submissions where every free-form question got routed to ``free_single``
    voting because gold was missing, even though the question text was
    multi-part. The fix: route off the question text + options, never gold.
    """
    from cse151b_comp.prompts import detect_question_type

    qtype = detect_question_type(item.get("question", ""), item.get("options"))
    # detect_question_type returns "mc" / "free_single" / "free_multi" already.
    return qtype


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
    p.add_argument(
        "--prompt",
        default="phase0",
        choices=[
            "phase0",
            "current",
            "runb",
            "runc",
            "rund",
            "rune",
            "runf",
            "rung",
            "runh",
            "runi",
            "runj",
            "runj_final",
            "runj_olympiad",
            "runj_trig",
            "runj_geom",
            "runj_stats_hyp",
            "runj_stats_reg",
            "runj_stats_desc",
            "runj_stats",
            "runj_calc",
            "runj_prob",
            "runj_number_alg",
            "runj_v2",
            "runj_v2_final",
            "runj_v2_olympiad",
            "runj_v2_trig",
            "runj_v2_geom",
            "runj_v2_stats_hyp",
            "runj_v2_stats_reg",
            "runj_v2_stats_desc",
            "runj_v2_calc",
            "runj_v2_prob",
            "runj_v2_number_alg",
        ],
        help="Which prompt set to use. phase0 = starter (proven 0.575 leaderboard); "
        "current = v6 per-type routing; runb/c/d/e are jason/dev's incremental "
        "improvements (Run B = leaderboard 0.60). Run E is the aggressive "
        "ceiling probe — validate on val first.",
    )
    p.add_argument("--limit", type=int, default=None, help="Only run on first N rows (debug).")
    p.add_argument("--val", default=None, help="Optional val_indices.json to filter --input.")
    p.add_argument("--gpu-mem-util", type=float, default=0.70)
    p.add_argument("--max-model-len", type=int, default=20480)
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=16,
        help="vLLM max concurrent sequences. Lower for K=8 to keep KV cache feasible.",
    )
    p.add_argument("--seed", type=int, default=42, help="Base seed; per-sample seed varies internally.")
    p.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="If > 0, process prompts in chunks of this size, writing output JSONL "
        "incrementally after each chunk. Survives mid-run kills. 0 = single batch.",
    )
    p.add_argument(
        "--resume", action="store_true", help="If set and --output already exists, skip ids already present and append."
    )
    p.add_argument(
        "--allocate-tokens",
        action="store_true",
        help="Use cse151b_comp.budget.allocate_max_tokens to assign per-question "
        "max_tokens (raises budget for MCQ-with-many-options and multi-part free-form). "
        "Floor 12288, ceiling 20480. When set, --max-tokens is ignored.",
    )
    p.add_argument(
        "--max-tokens-floor",
        type=int,
        default=12288,
        help="Floor for --allocate-tokens. Default matches the previous uniform max_tokens.",
    )
    p.add_argument(
        "--max-tokens-ceiling",
        type=int,
        default=20480,
        help="Ceiling for --allocate-tokens. Leaves ~4k headroom under max_model_len.",
    )
    args = p.parse_args()

    _setup_env()
    # Singleton Judger for the per-row `correct` field. score_response()
    # dispatches Judger.auto_judge for free-form types (handles SymPy +
    # LaTeX equivalence) and exact-letter match for MCQ. See bug fix
    # rationale in the comment near the `correct` assignment below.
    import sys as _sys

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    _repo_root = pathlib.Path(__file__).resolve().parents[2]
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    from judger import Judger as _Judger  # noqa: E402  (sys.path patch)

    _judger = _Judger(strict_extract=False)

    rows = [json.loads(line) for line in open(args.input)]
    if args.val:
        val_ids = set(json.loads(pathlib.Path(args.val).read_text())["val_ids"])
        rows = [r for r in rows if r["id"] in val_ids]
    if args.limit:
        rows = rows[: args.limit]

    # Resume: skip rows whose id is already in the output JSONL.
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set = set()
    if args.resume and out_path.exists():
        for line in open(out_path):
            try:
                completed_ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
        if completed_ids:
            before = len(rows)
            rows = [r for r in rows if r["id"] not in completed_ids]
            print(f"[sc] Resume: {len(completed_ids)} already done, {len(rows)} remaining (was {before}).")

    print(f"[sc] Loaded {len(rows)} rows. K={args.k}, T={args.temperature}, top_p={args.top_p}.")
    if args.chunk_size > 0:
        print(f"[sc] Incremental mode: chunk_size={args.chunk_size} (output written per chunk).")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model="Qwen/Qwen3-4B-Thinking-2507",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=True,  # K samples share prompt prefix
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_model_len,
        seed=args.seed,
    )

    def _build_sp(max_tokens: int) -> SamplingParams:
        return SamplingParams(
            n=args.k,
            max_tokens=max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            seed=args.seed,
        )

    sampling_params = _build_sp(args.max_tokens)

    # If --allocate-tokens, build a per-prompt SamplingParams list using
    # cse151b_comp.budget. vLLM's llm.generate(prompts, sampling_params=list)
    # accepts a list whose length matches prompts, applying each per-prompt.
    per_prompt_sp: list[SamplingParams] | None = None
    if args.allocate_tokens:
        # Auto-enable v2 budget (floor 16k, MCQ cap 22k, multi cap 30k) for
        # Run F-final and Run G — both designed against v2. Other prompts use
        # v1 (floor 12k, MCQ cap 18k, multi cap 22k). Mirrors jason/dev's
        # `auto_v2_prompts` list in inference.py (commit ee43f97).
        # Run F final, Run G, Run I, and Run J family all use v2 budget.
        if args.prompt in ("rung", "runf", "runi") or args.prompt.startswith("runj"):
            from cse151b_comp.budget import allocate_max_tokens_v2 as _allocator

            budget_label = "v2"
        else:
            from cse151b_comp.budget import allocate_max_tokens as _allocator

            budget_label = "v1"

        per_prompt_sp = []
        budget_hist: dict[int, int] = {}
        for item in rows:
            mt = _allocator(item.get("question", ""), item.get("options"))
            per_prompt_sp.append(_build_sp(mt))
            budget_hist[mt] = budget_hist.get(mt, 0) + 1
        # Show the budget distribution so we can spot-check calibration.
        hist_str = ", ".join(f"{k}:{v}" for k, v in sorted(budget_hist.items()))
        print(f"[sc] Per-question max_tokens histogram ({budget_label}): {hist_str}")
        # Note: --max-tokens-floor / --max-tokens-ceiling CLI args are kept
        # for backward compat but ignored; the budget module now hardcodes
        # the floor/ceiling per version (v1 vs v2) for reproducibility.

    build_prompt = _select_prompt_builder(args.prompt)
    prompts = []
    for item in rows:
        system, user = build_prompt(item["question"], item.get("options"))
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    if not rows:
        print("[sc] Nothing to do (all ids in --output already).")
        return

    # Chunk size 0 = "single batch" semantics, identical to old behaviour.
    chunk = args.chunk_size if args.chunk_size > 0 else len(rows)
    file_mode = "a" if (args.resume and completed_ids) else "w"

    print(
        f"[sc] Generating {len(prompts)} × n={args.k} = {len(prompts)*args.k} samples"
        f" in {(len(prompts) + chunk - 1) // chunk} chunk(s)..."
    )
    t_total_start = time.time()
    n_written = 0

    with open(out_path, file_mode) as out_f:
        for chunk_start in range(0, len(rows), chunk):
            chunk_rows = rows[chunk_start : chunk_start + chunk]
            chunk_prompts = prompts[chunk_start : chunk_start + chunk]
            chunk_idx = chunk_start // chunk + 1
            n_chunks = (len(rows) + chunk - 1) // chunk
            print(f"[sc] Chunk {chunk_idx}/{n_chunks}: " f"generating {len(chunk_rows)} prompts × n={args.k}...")
            t0 = time.time()
            if per_prompt_sp is not None:
                chunk_sp = per_prompt_sp[chunk_start : chunk_start + chunk]
                outputs = llm.generate(chunk_prompts, sampling_params=chunk_sp)
            else:
                outputs = llm.generate(chunk_prompts, sampling_params=sampling_params)
            elapsed = (time.time() - t0) / 60
            print(f"[sc] Chunk {chunk_idx} generation done in {elapsed:.1f} min.")

            for item, out in zip(chunk_rows, outputs):
                qtype = question_type(item)
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
                    "all_extracted": [list(e) if isinstance(e, tuple) else e for e in extracted],
                    "vote_counts": vote_counts,
                    "winning_answer": list(winning) if isinstance(winning, tuple) else winning,
                    "winning_response": winning_response,
                    "K": args.k,
                }

                gold = item.get("answer")
                if gold is not None:
                    record["answer"] = gold
                    # Use the full course Judger.auto_judge (via score_response)
                    # for `correct`. The previous string-equality on canonical
                    # forms (winning == _normalize_gold(...)) silently under-
                    # counted symbolic answers — Run F K=1 public was off by
                    # 9.41pp (53.73% reported vs 63.14% true). See
                    # reports/runF_public_rejudge_finding.md (jason/dev 4f250a9).
                    options = item.get("options")
                    record["correct"] = score_response(winning_response, gold, options, _judger)
                    # solvable_but_missed retains the canonical-form comparison
                    # since it's about voting headroom, not leaderboard prediction.
                    gold_norm = _normalize_gold(qtype, gold)
                    record["solvable_but_missed"] = solvable_but_missed(extracted, winning, gold_norm)

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            out_f.flush()
            os.fsync(out_f.fileno())  # durable per-chunk: survives kill/crash
            print(
                f"[sc] Chunk {chunk_idx}: wrote {len(chunk_rows)} rows "
                f"(total now {n_written + len(completed_ids)} of "
                f"{len(rows) + len(completed_ids)})."
            )

    print(f"\n[sc] Done in {(time.time() - t_total_start) / 60:.1f} min total.")
    print(f"[sc] Wrote {n_written} new rows → {out_path}")

    # Final summary: re-read the whole file so resume runs report cumulative stats.
    type_counts = {"mc": 0, "free_single": 0, "free_multi": 0}
    n_correct = 0
    n_solvable_but_missed = 0
    n_with_gold = 0
    for line in open(out_path):
        rec = json.loads(line)
        type_counts[rec.get("question_type", "free_single")] += 1
        if "correct" in rec:
            n_with_gold += 1
            if rec["correct"]:
                n_correct += 1
            if rec.get("solvable_but_missed"):
                n_solvable_but_missed += 1

    print(f"[sc] Question-type counts: {type_counts}")
    if n_with_gold:
        print(f"[sc] Accuracy: {n_correct}/{n_with_gold} = {n_correct/n_with_gold*100:.2f}%")
        print(
            f"[sc] Solvable-but-missed: {n_solvable_but_missed}/{n_with_gold} "
            f"= {n_solvable_but_missed/n_with_gold*100:.2f}%  "
            f"(upper bound on gain from better voting)"
        )


if __name__ == "__main__":
    main()
