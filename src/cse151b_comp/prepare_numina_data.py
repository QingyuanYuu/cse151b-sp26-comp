"""Build a NuminaMath-derived SFT training file, decontaminated against the
competition's public + private question sets.

Designed to run on a non-GPU-bound machine (e.g. 4090 24GB) in parallel with
the K=32 self-distillation SC running on the 96GB box. Final output JSONL has
the same schema as ``data/sft_train.jsonl`` produced by ``prepare_sft_data``,
so the two can be concatenated for LoRA training without extra glue.

Pipeline:
1. ``datasets.load_dataset("AI-MO/NuminaMath-CoT")`` (or another --dataset name).
2. Quality filter: keep only rows whose ``solution`` ends in a ``\\boxed{...}``
   and whose ``problem`` is non-empty / non-trivial.
3. Decontamination: TF-IDF cosine similarity of every numina problem against
   every public + private question; drop numina rows with max-sim ≥ threshold.
4. Reformat each survivor's ``solution`` into Qwen3-Thinking style:
   ``<think>{everything before the last \\boxed{} block}</think>\\n\\nFinal
   answer: \\boxed{...}``. This matches the target shape used by self-distill
   SFT rows.
5. Cap to ``--max-keep`` rows (default 10k) and write JSONL.

Key design choices and their reasons:

- **TF-IDF, not embedding-based dedup.** Math problems share a lot of structural
  vocabulary (numbers, variable names, "Find the ...", "Compute the ..."), and
  TF-IDF over (1, 2)-grams catches near-duplicates with no GPU and no model
  download. Threshold default 0.85 is conservative; raise to be stricter.
- **Wrap the entire pre-boxed solution in ``<think>``.** Qwen3-Thinking-2507
  emits ``<think>``/`</think>` markers via its chat template; SFT targets
  must match. NuminaMath solutions are CoT but unmarked; this wrapping is the
  minimum-surgery transform that makes them shape-compatible.
- **No multi-part handling.** NuminaMath is overwhelmingly single-answer
  competition style. The few multi-part problems (detected via repeated
  ``[ANS]``) are simply dropped — they account for < 1 % of the dataset and
  the schema mismatch isn't worth the parsing complexity.
- **Skip ``NuminaMath-TIR``.** That variant interleaves Python tool calls;
  the competition rule explicitly forbids inference-time tools, so training
  on tool-augmented traces would teach the model the wrong distribution.

CLI::

    # On the 4090 box (after git pull + uv sync --extra numina):
    cse151b-prepare-numina \\
        --dataset AI-MO/NuminaMath-CoT \\
        --output data/numina_sft.jsonl \\
        --public data/public.jsonl \\
        --private data/private.jsonl \\
        --tfidf-threshold 0.85 \\
        --max-keep 10000

    # Then scp data/numina_sft.jsonl to the Blackwell box; concatenate
    # with data/sft_train.jsonl from prepare_sft_data for LoRA training.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable

from cse151b_comp.extract import extract_all_final_boxed
from cse151b_comp.prompts import SYSTEM_PROMPT_FREE_SINGLE

# ─── Step 2: quality filter ─────────────────────────────────────────────────


def is_valid_row(problem: str, solution: str) -> bool:
    """A row is usable only if the solution ends in a `\\boxed{...}` and the
    problem text is non-trivial."""
    if not problem or len(problem.strip()) < 10:
        return False
    if "[ANS]" in problem and problem.count("[ANS]") >= 2:
        # Multi-part placeholder — drop (rare in NuminaMath, schema mismatch).
        return False
    boxes = extract_all_final_boxed(solution)
    return bool(boxes)


def filter_quality(rows: Iterable[dict]) -> list[dict]:
    """Keep only rows with extractable boxed answers and non-trivial problems."""
    out = []
    for r in rows:
        problem = r.get("problem") or r.get("question") or ""
        solution = r.get("solution") or r.get("answer") or ""
        if is_valid_row(problem, solution):
            out.append({"problem": problem, "solution": solution, "source": r.get("source", "numina")})
    return out


# ─── Step 3: decontamination via TF-IDF ─────────────────────────────────────


def decontaminate(
    numina_rows: list[dict],
    contam_texts: list[str],
    tfidf_threshold: float = 0.85,
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 50_000,
) -> tuple[list[dict], list[float]]:
    """Drop numina rows whose problem has TF-IDF cosine similarity >= threshold
    to any contamination text. Returns (kept_rows, max_sim_per_kept_row)."""
    if not numina_rows or not contam_texts:
        return numina_rows, [0.0] * len(numina_rows)

    # sklearn import deferred until we actually need it — trivial / no-op
    # inputs should not require the extra dep.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    n_numina = len(numina_rows)
    all_texts = [r["problem"] for r in numina_rows] + contam_texts

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=True,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(all_texts)

    numina_X = X[:n_numina]
    contam_X = X[n_numina:]

    # Compute max similarity per numina row in chunks to bound memory
    # (numina_X can be 100k × 50k sparse; contam_X is small).
    chunk = 5_000
    max_sims = []
    for start in range(0, n_numina, chunk):
        block = numina_X[start : start + chunk]
        sims = cosine_similarity(block, contam_X)
        max_sims.extend(sims.max(axis=1).tolist())

    kept_rows = []
    kept_sims = []
    for r, s in zip(numina_rows, max_sims):
        if s < tfidf_threshold:
            kept_rows.append(r)
            kept_sims.append(s)
    return kept_rows, kept_sims


# ─── Step 4: reformat solution into <think> + boxed answer ──────────────────


def split_at_last_boxed(solution: str) -> tuple[str, str] | None:
    """Return ``(prefix_before_last_box, last_box_segment)`` or ``None`` if
    no `\\boxed{...}` is found.

    ``last_box_segment`` retains the original `\\boxed{...}` markup verbatim;
    we don't try to canonicalize it, because the judger handles equivalence
    on the answer side.
    """
    # Find the last "\boxed{" by scanning right-to-left for matching braces.
    marker = "\\boxed{"
    last_start = solution.rfind(marker)
    if last_start < 0:
        return None
    # Walk forward to find the matching closing brace.
    depth = 0
    i = last_start + len(marker) - 1  # at the '{' of "\boxed{"
    while i < len(solution):
        c = solution[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                return solution[:last_start].rstrip(), solution[last_start:end]
        i += 1
    return None  # unbalanced; treat as malformed


def format_sft_row(row: dict, idx: int) -> dict | None:
    """Build one SFT pair from a numina row. Returns ``None`` if the solution
    can't be split (defensive — quality filter should have caught this)."""
    split = split_at_last_boxed(row["solution"])
    if split is None:
        return None
    reasoning_prefix, last_box = split

    # Wrap reasoning in <think>...</think>; final line is the boxed answer.
    target_response = "<think>\n" + reasoning_prefix.strip() + "\n</think>\n\nFinal answer: " + last_box

    user_prompt = row["problem"].strip() + "\n\nSolve and put the final answer in \\boxed{}."

    return {
        "id": f"numina:{idx}",
        "question_type": "free_single",
        "system_prompt": SYSTEM_PROMPT_FREE_SINGLE,
        "user_prompt": user_prompt,
        "target_response": target_response,
        "source": row.get("source", "numina"),
    }


# ─── Glue: load contamination texts ────────────────────────────────────────


def load_contam(public_path: pathlib.Path | None, private_path: pathlib.Path | None) -> list[str]:
    texts: list[str] = []
    for p in (public_path, private_path):
        if p is None or not p.exists():
            continue
        for line in open(p):
            row = json.loads(line)
            q = row.get("question") or row.get("problem")
            if q:
                texts.append(q)
    return texts


# ─── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="NuminaMath → SFT pairs, decontaminated against competition data.")
    p.add_argument(
        "--dataset", default="AI-MO/NuminaMath-CoT", help="HF dataset id. Avoid 'NuminaMath-TIR' (uses tools)."
    )
    p.add_argument("--split", default="train")
    p.add_argument("--output", required=True, help="Output SFT JSONL.")
    p.add_argument("--public", default="data/public.jsonl", help="Competition public.jsonl for decontamination.")
    p.add_argument(
        "--private", default="data/private.jsonl", help="Competition private.jsonl for decontamination (optional)."
    )
    p.add_argument(
        "--tfidf-threshold",
        type=float,
        default=0.85,
        help="Drop numina rows with cosine sim >= this to any contam question.",
    )
    p.add_argument(
        "--max-keep",
        type=int,
        default=10_000,
        help="Cap output to this many rows (random subsample if more pass filters).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit-load", type=int, default=None, help="Optional cap on rows loaded from HF dataset (for fast iteration)."
    )
    args = p.parse_args()

    print(f"[numina] Loading {args.dataset} split={args.split}...")
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit_load:
        ds = ds.select(range(min(args.limit_load, len(ds))))
    print(f"[numina] Loaded {len(ds)} raw rows.")

    raw_rows = [
        {"problem": r.get("problem", ""), "solution": r.get("solution", ""), "source": r.get("source", "numina")}
        for r in ds
    ]

    print("[numina] Quality filtering (must have \\boxed{}, non-trivial problem)...")
    rows = filter_quality(raw_rows)
    print(f"[numina] After quality filter: {len(rows)} ({len(rows) / max(len(raw_rows), 1) * 100:.1f}%)")

    contam = load_contam(
        pathlib.Path(args.public) if args.public else None,
        pathlib.Path(args.private) if args.private else None,
    )
    print(f"[numina] Loaded {len(contam)} contamination texts (public+private).")

    if contam:
        print(f"[numina] Decontamination at threshold {args.tfidf_threshold}...")
        rows, sims = decontaminate(rows, contam, tfidf_threshold=args.tfidf_threshold)
        print(f"[numina] After decontamination: {len(rows)} kept.")
        if sims:
            sorted_sims = sorted(sims, reverse=True)
            top = sorted_sims[: min(5, len(sorted_sims))]
            print(f"[numina]   Top 5 retained sims (just below threshold): {[f'{s:.3f}' for s in top]}")
    else:
        print("[numina] No contam texts found — skipping dedup. WARNING: this risks data leakage.")

    if len(rows) > args.max_keep:
        import random

        rng = random.Random(args.seed)
        rows = rng.sample(rows, args.max_keep)
        print(f"[numina] Subsampled to --max-keep={args.max_keep}.")

    print("[numina] Formatting SFT pairs...")
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_format_failed = 0
    with open(out_path, "w") as f:
        for idx, r in enumerate(rows):
            sft_row = format_sft_row(r, idx)
            if sft_row is None:
                n_format_failed += 1
                continue
            f.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[numina] Wrote {n_written} rows → {out_path}")
    if n_format_failed:
        print(f"[numina] Format-failed rows (post-filter): {n_format_failed}")


if __name__ == "__main__":
    main()
