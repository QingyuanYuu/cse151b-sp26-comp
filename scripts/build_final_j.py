"""Read ablation results, decide which branches to keep, write final-J config.

Output: `data/runj_final_branches.txt` — one keep branch per line.
The `build_prompt_runj_final` builder in runj.py reads this file at
construction time so we can swap final-J composition without code changes.

TWO modes:

1. Auto (default): apply threshold rules to ablation Δ
   - KEEP if Δ ≥ max(2pp, std_err)         strong signal
   - DROP if Δ ≤ -max(2pp, std_err)         clearly hurts
   - WEAK (n < 25): keep by default         we couldn't measure; cost is low
   - NOISE in band: keep by default         err on side of including

2. Manual override: `--branches olympiad,trig,...`
   Takes a comma-separated list and writes EXACTLY that. Use this after
   reviewing `scripts/inspect_runj_ablation.py` output and applying
   judgment that the auto threshold can't capture (consistent improvement
   pattern vs. one outlier, multi-box regression, response-length blowup,
   topic routing mistakes, etc.).
"""

from __future__ import annotations

import argparse
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
    p = argparse.ArgumentParser()
    p.add_argument(
        "--branches",
        default=None,
        help="Comma-separated explicit list of branches to enable. Skips "
        "auto-threshold logic. Use after manual review of "
        "scripts/inspect_runj_ablation.py output.",
    )
    args = p.parse_args()
    out = pathlib.Path("data/runj_final_branches.txt")

    if args.branches:
        valid = [b.strip() for b in args.branches.split(",") if b.strip()]
        unknown = [b for b in valid if b not in TOPICS]
        if unknown:
            print(f"ERROR: unknown branch names: {unknown}")
            print(f"Valid: {TOPICS}")
            raise SystemExit(1)
        out.write_text("\n".join(valid) + "\n")
        print("=== Manual override applied ===")
        print(f"Final J branches ({len(valid)}/9): {', '.join(valid)}")
        print(f"Written to {out}")
        return

    keep: list[str] = []
    drop: list[str] = []
    print("=== Final J branch composition (auto threshold) ===\n")
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

    out.write_text("\n".join(keep) + "\n")
    print()
    print(f"=== Written to {out} ===")
    print(f"Final J branches ({len(keep)}/9): {', '.join(keep)}")
    if drop:
        print(f"DROPPED ({len(drop)}): {', '.join(drop)}")
    print()
    print("To override after review of scripts/inspect_runj_ablation.py output:")
    print("  PYTHONPATH=src .venv/bin/python scripts/build_final_j.py " "--branches olympiad,trig,...")


if __name__ == "__main__":
    main()
