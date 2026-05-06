#!/bin/bash
# Run C: single-shot inference with Run C prompt + per-type token budget.
#
# Usage:
#   scripts/run_c.sh                # full private-set run (~3-3.5 hours)
#   scripts/run_c.sh --val          # val dry-run on 225 public questions (~30-40 min)
#   scripts/run_c.sh --resume       # skip inference if jsonl already exists, just rebuild CSV
#
# Outputs:
#   results/private_runC_singleshot.jsonl   — raw responses (or val_runC.jsonl in --val mode)
#   results/submission_runC.csv             — Kaggle CSV
#   reports/runC_summary.md                 — sanity stats + top-line numbers

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="private"
RESUME=0
for arg in "$@"; do
    case "$arg" in
        --val) MODE="val" ;;
        --resume) RESUME=1 ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

if [ "$MODE" = "val" ]; then
    INPUT="data/public.jsonl"
    JSONL="results/val_runC_singleshot.jsonl"
    CSV=""    # no submission CSV in val mode
    LIMIT_FLAG="--limit 225"
    JUDGE_FLAG=""    # public has gold → let judger score
else
    INPUT="data/private.jsonl"
    JSONL="results/private_runC_singleshot.jsonl"
    CSV="results/submission_runC.csv"
    LIMIT_FLAG=""
    JUDGE_FLAG="--no-judge"
fi

echo "[$(date)] === Run C ($MODE mode) ==="
echo "  input  : $INPUT"
echo "  output : $JSONL"
[ -n "$CSV" ] && echo "  csv    : $CSV"
echo "  resume : $RESUME"
echo

# ─── Step 1: free GPU (kill any straggler vLLM EngineCore) ────────────────
echo "[$(date)] Reaping any straggler vLLM processes..."
for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null || true); do
    echo "  killing pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
done
sleep 3
echo "[$(date)] GPU state:"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo

# ─── Step 2: inference ────────────────────────────────────────────────────
if [ "$RESUME" = "1" ] && [ -f "$JSONL" ]; then
    echo "[$(date)] --resume + $JSONL exists; skipping inference."
else
    echo "[$(date)] Starting Run C inference..."
    PYTHONPATH=src .venv/bin/python -m cse151b_comp.inference \
        --data "$INPUT" \
        --out "$JSONL" \
        --prompt runc \
        --per-type-budget \
        --temperature 0.6 \
        --top-p 0.95 \
        --top-k 20 \
        --max-model-len 24576 \
        --gpu-mem-util 0.70 \
        $LIMIT_FLAG \
        $JUDGE_FLAG
    echo "[$(date)] Inference done. Output: $(stat -c %s "$JSONL") bytes."
fi
echo

# ─── Step 3: build Kaggle CSV (private mode only) ─────────────────────────
if [ -n "$CSV" ]; then
    echo "[$(date)] Building Kaggle CSV..."
    PYTHONPATH=src .venv/bin/python -m cse151b_comp.submission \
        --results "$JSONL" \
        --out "$CSV"
    echo
fi

# ─── Step 4: sanity checks + summary ──────────────────────────────────────
echo "[$(date)] Running sanity checks..."
PYTHONPATH=src .venv/bin/python - <<PY
import csv, json, pathlib, re
from collections import Counter

JSONL = "$JSONL"
CSV = "$CSV"
MODE = "$MODE"

rows = [json.loads(l) for l in open(JSONL)]
print(f"=== {MODE} mode: {len(rows)} responses ===")

# Boxed-rate
boxed = sum(1 for r in rows if "\\\\boxed{" in r.get("response", ""))
print(f"  has \\\\boxed{{}} : {boxed}/{len(rows)} = {boxed/len(rows)*100:.1f}%")

# Truncation-prone v6 patterns we explicitly forbid in Run C
quad_pat = re.compile(r"\\\\boxed\{[^{}]*\}\s*\\\\quad\s*\\\\boxed")
n_quad = sum(1 for r in rows if quad_pat.search(r.get("response", "")))
qquad_pat = re.compile(r"\\\\boxed\{[^{}]*\}\s*\\\\qquad\s*\\\\boxed")
n_qquad = sum(1 for r in rows if qquad_pat.search(r.get("response", "")))
print(f"  \\\\quad-truncation pattern : {n_quad} (must be 0 or near-0)")
print(f"  \\\\qquad-truncation pattern: {n_qquad}")

# Question-type breakdown by extracted boxed count
def n_boxed(text):
    return text.count("\\\\boxed{")
print(f"  boxed-count distribution: {Counter(n_boxed(r['response']) for r in rows).most_common(10)}")

# Accuracy if val mode (rows have 'correct' field from evaluate_rows)
if MODE == "val":
    has_correct = [r for r in rows if "correct" in r]
    if has_correct:
        n_correct = sum(1 for r in has_correct if r["correct"])
        print(f"  val accuracy: {n_correct}/{len(has_correct)} = {n_correct/len(has_correct)*100:.2f}%")
        print(f"    Phase 0 val baseline: 56.44%")
        print(f"    Gate: ≥ 55.4% to proceed to leaderboard")

# Write a summary report
report_lines = [
    f"# Run C Summary ({MODE} mode)",
    "",
    f"Source: \`{JSONL}\` ({len(rows)} responses).",
    "",
    "## Sanity",
    "",
    f"- has \\\\boxed{{}}: {boxed}/{len(rows)} = {boxed/len(rows)*100:.1f}%",
    f"- \\\\quad-truncation pattern (v6 dead pattern): {n_quad} (target: 0)",
    f"- \\\\qquad-truncation pattern: {n_qquad}",
    "",
    "## Boxed-count distribution",
    "",
    f"\`{Counter(n_boxed(r['response']) for r in rows).most_common(10)}\`",
]
if MODE == "val":
    has_correct = [r for r in rows if "correct" in r]
    if has_correct:
        n_correct = sum(1 for r in has_correct if r["correct"])
        report_lines += [
            "",
            "## Val accuracy",
            "",
            f"- {n_correct}/{len(has_correct)} = {n_correct/len(has_correct)*100:.2f}%",
            f"- Phase 0 val baseline: 56.44%",
            f"- Gate to proceed: ≥ 55.4%",
        ]
elif CSV:
    report_lines += [
        "",
        "## Action",
        "",
        f"Submit \`{CSV}\` to Kaggle.",
    ]

pathlib.Path("reports").mkdir(exist_ok=True)
out = pathlib.Path("reports/runC_summary.md")
out.write_text("\n".join(report_lines))
print(f"\nWrote {out}")
PY

echo
echo "[$(date)] === Run C done ==="
if [ -n "$CSV" ]; then
    echo "Submit: $CSV"
fi
