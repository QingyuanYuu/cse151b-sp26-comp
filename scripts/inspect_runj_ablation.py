"""Deep per-branch ablation review: baseline vs variant side-by-side.

For each Run J branch, dumps:
- Per-question win/loss (correct/incorrect under baseline AND variant)
- Reasoning length distribution (mean / p50 / p95) for both
- Multi-box rate (a Run F regression we should monitor)
- A handful of *wins* (variant got it, baseline didn't) — what helped
- A handful of *losses* (baseline got it, variant didn't) — what broke
- Question-text snippets for both, so we can spot routing mistakes

This is the file I (Claude) read when reviewing ablation results before
deciding which branches to KEEP / DROP for final-J. It's deliberately
verbose; pipe through `less` or write to a file for offline review.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/inspect_runj_ablation.py
    PYTHONPATH=src .venv/bin/python scripts/inspect_runj_ablation.py --branch olympiad
    PYTHONPATH=src .venv/bin/python scripts/inspect_runj_ablation.py > reports/runj_ablation_review.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics

TOPICS = [
    "olympiad",
    "trig",
    "geometry",
    "stats_hyp_test",
    "stats_regression",
    "stats_descriptive",
    "calculus",
    "prob_combi",
    "discrete_math",
]

_BOXED_RE = re.compile(r"\\boxed\{")


def _load(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path)]


def _len_chars(row: dict) -> int:
    """Best-effort response length. Some rows have winning_response, others
    have responses[0]. Fall back to '' if missing."""
    return len(row.get("winning_response") or (row.get("responses") or [""])[0] or "")


def _boxed_count(text: str) -> int:
    return len(_BOXED_RE.findall(text or ""))


def _summarize_branch(topic: str) -> None:
    b_path = pathlib.Path(f"results/runj_ablation_{topic}_baseline.jsonl")
    v_path = pathlib.Path(f"results/runj_ablation_{topic}_variant.jsonl")
    b_rows = _load(b_path)
    v_rows = _load(v_path)
    if not b_rows or not v_rows:
        print(f"\n## {topic}\n  ⚠ missing data: baseline={len(b_rows)}, variant={len(v_rows)}\n")
        return

    by_id_b = {r["id"]: r for r in b_rows}
    by_id_v = {r["id"]: r for r in v_rows}
    common_ids = sorted(set(by_id_b) & set(by_id_v))

    n = len(common_ids)
    b_correct = sum(1 for qid in common_ids if by_id_b[qid].get("correct"))
    v_correct = sum(1 for qid in common_ids if by_id_v[qid].get("correct"))
    b_pct = 100 * b_correct / n if n else 0
    v_pct = 100 * v_correct / n if n else 0
    delta = v_pct - b_pct

    # Length stats
    b_lens = [_len_chars(by_id_b[qid]) for qid in common_ids]
    v_lens = [_len_chars(by_id_v[qid]) for qid in common_ids]

    def _stats(xs: list[int]) -> str:
        if not xs:
            return "n/a"
        xs = sorted(xs)
        return f"mean={statistics.mean(xs):.0f} p50={xs[len(xs) // 2]} p95={xs[int(0.95 * len(xs))]}"

    # Multi-box rate (Run F regression flag)
    b_multibox = sum(1 for qid in common_ids if _boxed_count(by_id_b[qid].get("winning_response") or "") > 1)
    v_multibox = sum(1 for qid in common_ids if _boxed_count(by_id_v[qid].get("winning_response") or "") > 1)

    # Wins (variant correct, baseline wrong) and losses (variant wrong, baseline right)
    wins = [qid for qid in common_ids if by_id_v[qid].get("correct") and not by_id_b[qid].get("correct")]
    losses = [qid for qid in common_ids if by_id_b[qid].get("correct") and not by_id_v[qid].get("correct")]

    print(f"\n## {topic}")
    print(
        f"\n  n={n}    baseline={b_correct}/{n} ({b_pct:.1f}%)    "
        f"variant={v_correct}/{n} ({v_pct:.1f}%)    Δ={delta:+.1f}pp"
    )
    print(f"  wins(v_only)={len(wins)}  losses(b_only)={len(losses)}  flips_total={len(wins) + len(losses)}")
    print(f"  baseline len: {_stats(b_lens)}")
    print(f"  variant  len: {_stats(v_lens)}")
    print(f"  multi-box (>1 \\boxed{{}}): baseline={b_multibox}, variant={v_multibox}")

    if wins:
        print(f"\n  Wins ({min(3, len(wins))} samples — variant fixed):")
        for qid in wins[: min(3, len(wins))]:
            r = by_id_v[qid]
            q = r["question"][:200].replace("\n", " ")
            ans = str(r.get("answer", ""))[:80]
            ext = str(r.get("extracted") or r.get("voted_answer") or "")[:80]
            print(f"    [{qid}] {q}")
            print(f"      gold: {ans!r}")
            print(f"      variant_extracted: {ext!r}")

    if losses:
        print(f"\n  Losses ({min(3, len(losses))} samples — variant broke):")
        for qid in losses[: min(3, len(losses))]:
            r_b = by_id_b[qid]
            r_v = by_id_v[qid]
            q = r_b["question"][:200].replace("\n", " ")
            ans = str(r_b.get("answer", ""))[:80]
            ext_b = str(r_b.get("extracted") or r_b.get("voted_answer") or "")[:80]
            ext_v = str(r_v.get("extracted") or r_v.get("voted_answer") or "")[:80]
            print(f"    [{qid}] {q}")
            print(f"      gold: {ans!r}")
            print(f"      baseline_extracted: {ext_b!r}  ✓")
            print(f"      variant_extracted:  {ext_v!r}  ✗")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--branch", default=None, help="Inspect only this branch (default: all 9)")
    args = p.parse_args()

    branches = [args.branch] if args.branch else TOPICS
    print("# Run J ablation — deep review")
    print("\nFor each branch: paired baseline (Run F) vs variant (Run J branch enabled).")
    print("Wins = variant fixed; Losses = variant broke. Multi-box >1 is a regression flag.")
    for t in branches:
        _summarize_branch(t)


if __name__ == "__main__":
    main()
