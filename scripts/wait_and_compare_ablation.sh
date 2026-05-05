#!/bin/bash
# Wait for all 3 ablation outputs, then auto-generate comparison report.
set -eu
cd "$(dirname "$0")/.."

V3A=results/ablation/v3a_val.jsonl
V3B=results/ablation/v3b_val.jsonl
V3C=results/ablation/v3c_val.jsonl
REPORT=reports/ablation_summary.md

while [ ! -f "$V3A" ] || [ ! -f "$V3B" ] || [ ! -f "$V3C" ]; do
    sleep 60
done

echo "[$(date)] All 3 ablation outputs ready, generating report..."

PYTHONPATH=src .venv/bin/python <<'PY'
import json, pathlib

REPORT = "reports/ablation_summary.md"
CONFIGS = [
    ("phase0",  "results/baseline_v0_val.jsonl",     "Phase 0 starter"),
    ("phase1",  "results/baseline_v1_val.jsonl",     "Phase 1 (full)"),
    ("v3a",     "results/ablation/v3a_val.jsonl",    "v3a = P0 + anti-(C) only"),
    ("v3b",     "results/ablation/v3b_val.jsonl",    "v3b = P1 - anti-rounding"),
    ("v3c",     "results/ablation/v3c_val.jsonl",    "v3c = P1 - token-rescue"),
]

def acc(rows):
    n = len(rows)
    correct = sum(r.get("correct", False) for r in rows)
    return correct, n, correct / n if n else 0

results = {}
for key, path, label in CONFIGS:
    if not pathlib.Path(path).exists():
        results[key] = (label, None)
        continue
    rows = [json.loads(l) for l in open(path)]
    mcq = [r for r in rows if r.get("is_mcq") or r.get("options")]
    free = [r for r in rows if not (r.get("is_mcq") or r.get("options"))]
    free_multi = [r for r in free if isinstance(r.get("gold", r.get("answer")), list) and len(r.get("gold", r.get("answer"))) > 1]
    free_single = [r for r in free if r not in free_multi]
    results[key] = (label, {
        "overall": acc(rows),
        "mcq": acc(mcq),
        "free_single": acc(free_single),
        "free_multi": acc(free_multi),
    })

lines = ["# Phase 1 Ablation Summary", "",
         "All variants evaluated on the same stratified 20% public val (n=225, seed=42).",
         "Phase 0 leaderboard score: **0.575**, Phase 1: **0.494** (-0.081, regression).",
         "",
         "| variant | description | overall | MCQ | free_single | free_multi |",
         "|---|---|---|---|---|---|"]

for key, (label, stats) in results.items():
    if stats is None:
        lines.append(f"| `{key}` | {label} | _missing_ | | | |")
        continue
    def cell(t):
        c, n, a = t
        return f"{c}/{n} ({a*100:.2f}%)" if n else "n/a"
    lines.append(f"| `{key}` | {label} | "
                 f"{cell(stats['overall'])} | "
                 f"{cell(stats['mcq'])} | "
                 f"{cell(stats['free_single'])} | "
                 f"{cell(stats['free_multi'])} |")

lines.append("")
lines.append("## Interpretation guide")
lines.append("")
lines.append("- If **v3a** ≈ **phase1** on overall: the MCQ anti-paren rule alone explains most of Phase 1's val gain.")
lines.append("- If **v3b** > **phase1** on overall: anti-rounding is the public→private regression cause; drop it.")
lines.append("- If **v3c** > **phase1** on overall: token-budget rescue is the public→private regression cause; drop it.")
lines.append("- If both **v3b** and **v3c** > **phase1**: combine both fixes for the final prompt.")

pathlib.Path("reports").mkdir(exist_ok=True)
pathlib.Path(REPORT).write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nWrote {REPORT}")
PY

echo "[$(date)] Report ready at $REPORT"
