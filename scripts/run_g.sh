#!/bin/bash
# Run G: Run F prompt + v2 budget (16k floor, 20k MCQ cap, 24k multi cap).
#
# Hypothesis: Run C's private regression came partly from free_single budget
# being compressed to 12k (vs Run B's flat 16k, which scored 0.600). v2
# budget restores the Run B floor and lifts ceilings +2k. Run F's prompt
# (cleanest of the post-B variants — no Yes/No bool rule, no Tuesday
# example, just sqrt(75) demo) provides the format-rule scaffolding without
# the Run C side effects.
#
# Usage:
#   scripts/run_g.sh              # full private (943 q, ~4-5h on RTX 4090)
#   scripts/run_g.sh --val        # 225-val dry-run (~30-40 min)
#   scripts/run_g.sh --resume     # rebuild CSV from existing jsonl

set -uo pipefail
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
    JSONL="results/val_runG_singleshot.jsonl"
    CSV=""
    LIMIT_FLAG="--limit 225"
    JUDGE_FLAG=""
else
    INPUT="data/private.jsonl"
    JSONL="results/private_runG_singleshot.jsonl"
    CSV="results/submission_runG.csv"
    LIMIT_FLAG=""
    JUDGE_FLAG="--no-judge"
fi

echo "[$(date)] === Run G ($MODE mode) ==="
echo "  prompt : runf (Run F prompt)"
echo "  budget : v2 (16k floor, 20k MCQ cap, 24k multi cap)"
echo "  input  : $INPUT"
echo "  output : $JSONL"
[ -n "$CSV" ] && echo "  csv    : $CSV"
echo

# Reap stale processes
echo "[$(date)] Reaping stale vLLM processes..."
for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null || true); do
    echo "  killing pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
done
sleep 3
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo

# Inference
if [ "$RESUME" = "1" ] && [ -f "$JSONL" ]; then
    echo "[$(date)] --resume + $JSONL exists; skipping inference."
else
    echo "[$(date)] Starting Run G inference..."
    PYTHONPATH=src .venv/bin/python -m cse151b_comp.inference \
        --data "$INPUT" \
        --out "$JSONL" \
        --prompt rung \
        --temperature 0.6 --top-p 0.95 --top-k 20 \
        --max-model-len 32768 \
        --max-num-seqs 6 \
        --gpu-mem-util 0.70 \
        $LIMIT_FLAG \
        $JUDGE_FLAG
    echo "[$(date)] Inference done. Output: $(stat -c %s "$JSONL") bytes."
fi
echo

# Build CSV (private only)
if [ -n "$CSV" ]; then
    echo "[$(date)] Building Kaggle CSV..."
    PYTHONPATH=src .venv/bin/python -m cse151b_comp.submission \
        --results "$JSONL" \
        --out "$CSV"
fi
echo

# Sanity check
echo "[$(date)] Sanity checks..."
PYTHONPATH=src .venv/bin/python - <<PY
import csv, json, pathlib, re
from collections import Counter

JSONL = "$JSONL"
CSV = "$CSV"
MODE = "$MODE"

rows = [json.loads(l) for l in open(JSONL)]
print(f"=== {MODE} mode: {len(rows)} responses ===")

boxed = sum(1 for r in rows if "\\\\boxed{" in r.get("response", ""))
print(f"  has \\\\boxed{{}} : {boxed}/{len(rows)} = {boxed/len(rows)*100:.1f}%")

quad_pat = re.compile(r"\\\\boxed\{[^{}]*\}\s*\\\\quad\s*\\\\boxed")
n_quad = sum(1 for r in rows if quad_pat.search(r.get("response", "")))
print(f"  \\\\quad-truncation : {n_quad} (target: 0)")

if MODE == "val":
    has_correct = [r for r in rows if "correct" in r]
    if has_correct:
        n_correct = sum(1 for r in has_correct if r["correct"])
        print(f"  val accuracy: {n_correct}/{len(has_correct)} = {n_correct/len(has_correct)*100:.2f}%")
PY

echo
echo "[$(date)] === Run G done ==="
[ -n "$CSV" ] && echo "Submit: $CSV"
