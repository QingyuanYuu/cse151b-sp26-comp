"""Read ablation results, decide which branches to keep, write final-J config.

Output: `data/runj_final_branches.txt` — one keep branch per line.
The `build_prompt_runj_final` builder in runj.py reads this file at
construction time so we can swap final-J composition without code changes.

Logic:
- KEEP if Δ ≥ max(2pp, std_err)         strong signal
- DROP if Δ ≤ -max(2pp, std_err)         clearly hurts
- WEAK (n < 25): keep by default         we couldn't measure; cost is low
- NOISE in band: keep by default         err on side of including

(Drops are rare unless a branch actively regresses.)
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
    "discrete_math",
]


def acc(path: pathlib.Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    rows = [json.loads(line) for line in open(path)]
    n = len(rows)
    n_correct = sum(1 for r in rows if r.get("correct"))
    return n_correct, n


def std_err_pp(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return 100.0 * math.sqrt(max(p * (1 - p), 1e-9) / n)


def main() -> None:
    keep: list[str] = []
    drop: list[str] = []
    print("=== Final J branch composition ===\n")
    for topic in TOPICS:
        b_path = pathlib.Path(f"results/runj_ablation_{topic}_baseline.jsonl")
        v_path = pathlib.Path(f"results/runj_ablation_{topic}_variant.jsonl")
        b_c, b_n = acc(b_path)
        v_c, v_n = acc(v_path)
        if b_n == 0 or v_n == 0:
            print(f"{topic:20} ⚠ missing data, defaulting to keep")
            keep.append(topic)
            continue
        b_pct = 100 * b_c / b_n
        v_pct = 100 * v_c / v_n
        delta = v_pct - b_pct
        noise = std_err_pp(0.5, max(b_n, 1))
        thresh = max(2.0, noise)

        if v_n < 25:
            verdict = f"weak n={v_n}, keep by default"
            keep.append(topic)
        elif delta <= -thresh:
            verdict = f"DROP (Δ={delta:+.1f}pp)"
            drop.append(topic)
        elif delta >= thresh:
            verdict = f"KEEP (Δ={delta:+.1f}pp signal)"
            keep.append(topic)
        else:
            verdict = f"noise (Δ={delta:+.1f}pp), keep"
            keep.append(topic)

        print(f"{topic:20} {b_pct:>5.1f}% → {v_pct:>5.1f}% Δ{delta:+5.1f}pp ±{noise:.1f}pp  {verdict}")

    out = pathlib.Path("data/runj_final_branches.txt")
    out.write_text("\n".join(keep) + "\n")
    print()
    print(f"=== Written to {out} ===")
    print(f"Final J branches ({len(keep)}/9): {', '.join(keep)}")
    if drop:
        print(f"DROPPED ({len(drop)}): {', '.join(drop)}")


if __name__ == "__main__":
    main()
