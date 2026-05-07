"""Summarize Run J branch ablation results.

Reads results/runj_ablation_<topic>_{baseline,variant}.jsonl pairs and
prints a side-by-side accuracy table to identify which branches help
their target topic.
"""

from __future__ import annotations

import json
import pathlib

TOPICS = ["trig", "geometry", "logic_proof", "stats_hyp_test", "stats_regression", "probability", "num_theory"]


def acc(path: pathlib.Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    rows = [json.loads(line) for line in open(path)]
    n = len(rows)
    n_correct = sum(1 for r in rows if r.get("correct"))
    return n_correct, n


def main() -> None:
    print(f"{'Topic':22} {'baseline':>15} {'variant':>15} {'Δ':>10} {'verdict':>10}")
    print("-" * 75)
    keep: list[str] = []
    drop: list[str] = []
    for topic in TOPICS:
        b_path = pathlib.Path(f"results/runj_ablation_{topic}_baseline.jsonl")
        v_path = pathlib.Path(f"results/runj_ablation_{topic}_variant.jsonl")
        b_c, b_n = acc(b_path)
        v_c, v_n = acc(v_path)
        if b_n == 0 or v_n == 0:
            print(f"{topic:22} (missing data)")
            continue
        b_pct = 100 * b_c / b_n
        v_pct = 100 * v_c / v_n
        delta = v_pct - b_pct
        # Heuristic: keep branch if Δ ≥ +2pp on its topic subset.
        # K=1 noise on n=30 is roughly ±10pp std error; we want ≥ 1σ
        # signal, but also save if branch helps even a little.
        verdict = "KEEP" if delta >= 2.0 else ("HURTS" if delta <= -2.0 else "noise")
        if verdict == "KEEP":
            keep.append(topic)
        elif verdict == "HURTS":
            drop.append(topic)
        line = (
            f"{topic:22} {b_c:>3}/{b_n:<3} {b_pct:>5.1f}% "
            f"{v_c:>3}/{v_n:<3} {v_pct:>5.1f}% {delta:>+8.1f}pp {verdict:>10}"
        )
        print(line)
    print()
    print("=== Recommended Run J branch composition ===")
    if keep:
        print(f"  KEEP branches:  {', '.join(keep)}")
    if drop:
        print(f"  DROP branches:  {', '.join(drop)}  (regressed on val)")
    if not keep and not drop:
        print("  All branches in noise band; default to keep all (low risk).")


if __name__ == "__main__":
    main()
