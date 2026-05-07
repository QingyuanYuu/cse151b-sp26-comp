"""Summarize Run J v3 ablation: compare v3 vs v2 vs v1 vs Run F baseline.

Reads four sets of files per branch:
  - baseline:    results/runj_ablation_<topic>_baseline.jsonl  (Run F)
  - v3 variant:  results/runj_ablation_<topic>_variant.jsonl   (after v3 ablation)
  - v2 variant:  results/v2_archive/runj_ablation_<topic>_variant.v2.jsonl
  - v1 variant:  results/v1_archive/runj_ablation_<topic>_variant.v1.jsonl
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


def acc(path: pathlib.Path, restrict_ids: set | None = None) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    rows = [json.loads(line) for line in open(path)]
    if restrict_ids is not None:
        rows = [r for r in rows if r["id"] in restrict_ids]
    return sum(1 for r in rows if r.get("correct")), len(rows)


def std_err_pp(n: int) -> float:
    if n <= 0:
        return 0.0
    return 100.0 * math.sqrt(0.25 / n)


def main() -> None:
    print(f"{'Topic':22} {'Baseline':>11} {'v1':>11} {'v2':>11} {'v3':>11} {'Δv3 vs B':>9} {'noise':>7}  v3 verdict")
    print("-" * 116)

    total_b = total_v1 = total_v2 = total_v3 = 0
    total_n_b = total_n_v1 = total_n_v2 = total_n_v3 = 0

    for t in TOPICS:
        b_path = pathlib.Path(f"results/runj_ablation_{t}_baseline.jsonl")
        v1_path = pathlib.Path(f"results/v1_archive/runj_ablation_{t}_variant.v1.jsonl")
        v2_path = pathlib.Path(f"results/v2_archive/runj_ablation_{t}_variant.v2.jsonl")
        v3_path = pathlib.Path(f"results/runj_ablation_{t}_variant.jsonl")

        # Use baseline ID set as canonical. Old v1/v2 may have fewer IDs.
        b_c, b_n = acc(b_path)
        if b_n == 0:
            print(f"{t:22}  ⚠ no baseline data")
            continue
        baseline_ids = {json.loads(line)["id"] for line in open(b_path)}

        v1_c, v1_n = acc(v1_path, restrict_ids=baseline_ids)
        v2_c, v2_n = acc(v2_path, restrict_ids=baseline_ids)
        v3_c, v3_n = acc(v3_path, restrict_ids=baseline_ids)

        total_b += b_c
        total_n_b += b_n
        total_v1 += v1_c
        total_n_v1 += v1_n
        total_v2 += v2_c
        total_n_v2 += v2_n
        total_v3 += v3_c
        total_n_v3 += v3_n

        # Reference percentages on the BASELINE id set
        b_pct = 100 * b_c / b_n
        v1_pct = 100 * v1_c / v1_n if v1_n else 0
        v2_pct = 100 * v2_c / v2_n if v2_n else 0
        v3_pct = 100 * v3_c / v3_n if v3_n else 0
        d3 = v3_pct - b_pct
        noise = std_err_pp(b_n)

        if v3_n < b_n - 5:
            verdict = f"v3 INCOMPLETE n={v3_n}/{b_n}"
        elif v3_n < 25:
            verdict = f"weak n={v3_n}"
        elif d3 >= max(2.0, noise):
            verdict = f"KEEP (Δ={d3:+.1f}pp)"
        elif d3 <= -max(2.0, noise):
            verdict = f"DROP (Δ={d3:+.1f}pp)"
        else:
            verdict = f"noise (Δ={d3:+.1f}pp)"

        print(
            f"{t:22} "
            f" {b_c:>3}/{b_n:<3} ({b_pct:>4.1f}%) "
            f" {v1_c:>3}/{v1_n:<3} ({v1_pct:>4.1f}%) "
            f" {v2_c:>3}/{v2_n:<3} ({v2_pct:>4.1f}%) "
            f" {v3_c:>3}/{v3_n:<3} ({v3_pct:>4.1f}%) "
            f" {d3:>+5.1f}pp "
            f" ±{noise:>3.1f}pp "
            f" {verdict}"
        )

    print("-" * 116)
    if total_n_b:
        agg_b = 100 * total_b / total_n_b
        agg_v1 = 100 * total_v1 / total_n_v1 if total_n_v1 else 0
        agg_v2 = 100 * total_v2 / total_n_v2 if total_n_v2 else 0
        agg_v3 = 100 * total_v3 / total_n_v3 if total_n_v3 else 0
        d3 = agg_v3 - agg_b
        print(
            f"{'AGGREGATE':22}  {total_b}/{total_n_b} ({agg_b:.1f}%)  "
            f"v1 ({agg_v1:.1f}%)  v2 ({agg_v2:.1f}%)  "
            f"v3 ({agg_v3:.1f}%)  Δv3={d3:+.1f}pp"
        )


if __name__ == "__main__":
    main()
