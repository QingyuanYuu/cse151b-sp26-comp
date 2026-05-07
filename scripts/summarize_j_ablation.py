"""Summarize Run J branch ablation results.

Reads results/runj_ablation_<topic>_{baseline,variant}.jsonl pairs and
prints a side-by-side accuracy table to identify which branches help
their target topic.
"""

from __future__ import annotations

import json
import math
import pathlib

# Topics tested in the ablation harness (matches run_j_ablation.sh PAIRS)
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
    n = len(rows)
    n_correct = sum(1 for r in rows if r.get("correct"))
    return n_correct, n


def std_err_pp(p: float, n: int) -> float:
    """Std error in percentage points for a Bernoulli accuracy estimate."""
    if n <= 0:
        return 0.0
    return 100.0 * math.sqrt(max(p * (1 - p), 1e-9) / n)


def main() -> None:
    header = f"{'Topic':18} {'baseline':>11} {'variant':>11} " f"{'Δ':>8} {'noise':>8} {'verdict':>10}"
    print(header)
    print("-" * len(header))
    keep: list[str] = []
    drop: list[str] = []
    weak: list[str] = []
    for topic in TOPICS:
        b_path = pathlib.Path(f"results/runj_ablation_{topic}_baseline.jsonl")
        v_path = pathlib.Path(f"results/runj_ablation_{topic}_variant.jsonl")
        b_c, b_n = acc(b_path)
        v_c, v_n = acc(v_path)
        if b_n == 0 or v_n == 0:
            print(f"{topic:18} (missing data)")
            continue
        b_pct = 100 * b_c / b_n
        v_pct = 100 * v_c / v_n
        delta = v_pct - b_pct
        # Paired comparison std error: roughly 2× single-sample std error
        # (assumes worst-case independence; in practice less due to
        # paired correlation). We report the single-sample std error
        # at p ≈ 0.5 as a noise floor reference.
        noise = std_err_pp(0.5, max(b_n, 1))
        # Verdict thresholds:
        # - KEEP if Δ ≥ +max(2pp, noise)  (signal beats noise)
        # - DROP if Δ ≤ -max(2pp, noise)
        # - WEAK if subset too small (n < 25) — verdict unreliable
        thresh = max(2.0, noise)
        if v_n < 25:
            verdict = "weak"
            weak.append(topic)
        elif delta >= thresh:
            verdict = "KEEP"
            keep.append(topic)
        elif delta <= -thresh:
            verdict = "DROP"
            drop.append(topic)
        else:
            verdict = "noise"
        line = (
            f"{topic:18} {b_c:>3}/{b_n:<3} {b_pct:>5.1f}% "
            f"{v_c:>3}/{v_n:<3} {v_pct:>5.1f}% "
            f"{delta:>+6.1f}pp ±{noise:>4.1f}pp {verdict:>10}"
        )
        print(line)
    print()
    print("=== Recommended Run J branch composition ===")
    if keep:
        print(f"  KEEP (signal):   {', '.join(keep)}")
    if drop:
        print(f"  DROP (regress):  {', '.join(drop)}")
    if weak:
        print(f"  WEAK (n<25):     {', '.join(weak)}  — keep by default (low risk)")
    print()
    print("Final Run J = generic + KEEP branches + WEAK branches")
    print(
        "  (DROP branches removed; WEAK kept to avoid throwing out " "potential signal that we just couldn't measure)"
    )


if __name__ == "__main__":
    main()
