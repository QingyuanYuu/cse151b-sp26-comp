"""Summarize Run J v2 ablation: compare v2 variant vs v1 variant vs Run F baseline.

Reads three sets of files per branch:
  - baseline:    results/runj_ablation_<topic>_baseline.jsonl  (Run F)
  - v2 variant:  results/runj_ablation_<topic>_variant.jsonl   (after v2 ablation)
  - v1 variant:  results/v1_archive/runj_ablation_<topic>_variant.v1.jsonl

For each topic, prints baseline%, v1%, v2%, Δv1, Δv2, and verdict.
"""

from __future__ import annotations

import json
import math
import pathlib

TOPICS = [
    "olympiad",
    "trig",
    "geometry",
    "stats_hyp_test",
    "stats_regression",
    "stats_descriptive",
    "calculus",
    "prob_combi",
    "number_alg",
]


def acc(path: pathlib.Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    rows = [json.loads(line) for line in open(path)]
    return sum(1 for r in rows if r.get("correct")), len(rows)


def std_err_pp(n: int) -> float:
    if n <= 0:
        return 0.0
    return 100.0 * math.sqrt(0.25 / n)


def main() -> None:
    print(f"{'Topic':22} {'Baseline':>11} {'v1':>11} {'v2':>11} {'Δv1':>8} {'Δv2':>8} {'noise':>8}  v2 verdict")
    print("-" * 112)
    for t in TOPICS:
        b_path = pathlib.Path(f"results/runj_ablation_{t}_baseline.jsonl")
        v1_path = pathlib.Path(f"results/v1_archive/runj_ablation_{t}_variant.v1.jsonl")
        v2_path = pathlib.Path(f"results/runj_ablation_{t}_variant.jsonl")

        b_c, b_n = acc(b_path)
        v1_c, v1_n = acc(v1_path)
        v2_c, v2_n = acc(v2_path)

        if b_n == 0 or v2_n == 0:
            print(f"{t:22}  ⚠ missing data: baseline n={b_n}, v2 n={v2_n}")
            continue

        b_pct = 100 * b_c / b_n
        v1_pct = 100 * v1_c / v1_n if v1_n else 0
        v2_pct = 100 * v2_c / v2_n
        d1 = v1_pct - b_pct
        d2 = v2_pct - b_pct
        noise = std_err_pp(b_n)

        # Verdict for v2
        if v2_n < 25:
            verdict = f"weak n={v2_n}"
        elif d2 >= max(2.0, noise):
            verdict = f"KEEP (Δ={d2:+.1f}pp)"
        elif d2 <= -max(2.0, noise):
            verdict = f"DROP (Δ={d2:+.1f}pp)"
        else:
            verdict = f"noise (Δ={d2:+.1f}pp)"

        # Direction-of-fix indicator
        fix_arrow = "✓" if d2 > d1 else ("✗" if d2 < d1 else "=")

        print(
            f"{t:22} "
            f" {b_c:>3}/{b_n:<3} ({b_pct:>4.1f}%) "
            f" {v1_c:>3}/{v1_n:<3} ({v1_pct:>4.1f}%) "
            f" {v2_c:>3}/{v2_n:<3} ({v2_pct:>4.1f}%) "
            f" {d1:>+5.1f}pp "
            f" {d2:>+5.1f}pp "
            f" ±{noise:>4.1f}pp "
            f" {fix_arrow} {verdict}"
        )


if __name__ == "__main__":
    main()
