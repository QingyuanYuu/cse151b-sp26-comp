#!/usr/bin/env bash
#
# Run J v3 ablation on EXPANDED eval subsets (9 branches, cap raised
# to all available). Total 313 questions (was 271; +42 from geometry
# 50→83 and stats_hyp 50→59).
#
# Runs:
# - 9 baseline (Run F) — uses --resume to pick up new IDs only
# - 9 v3 variant (one branch enabled per variant)
# Total: ~95 min K=1 wallclock.
#
# Usage: scripts/run_j_ablation_v3.sh

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

mkdir -p logs results

LOG="logs/runj_ablation_v3.log"
PID_FILE="logs/runj_ablation_v3.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running (PID $(cat "$PID_FILE"))."
    exit 1
fi
rm -f "$PID_FILE"

# Archive v2 variant outputs aside before v3 overwrites them
if ls results/runj_ablation_*_variant.jsonl 2>/dev/null > /dev/null; then
    mkdir -p results/v2_archive
    for f in results/runj_ablation_*_variant.jsonl; do
        mv "$f" "results/v2_archive/$(basename $f .jsonl).v2.jsonl"
    done
fi

setsid bash -c '
set -e

run_one() {
    local topic="$1" prompt="$2" out_suffix="$3"
    local input="data/eval_subsets/eval_${topic}.jsonl"
    local out="results/runj_ablation_${topic}_${out_suffix}.jsonl"
    if [ ! -f "$input" ]; then
        echo "  SKIP: $input not found"
        return 0
    fi
    local n_input=$(wc -l < "$input")
    echo ""
    echo "=== $(date "+%H:%M:%S") topic=$topic prompt=$prompt input=$input ($n_input qs) ==="
    uv run --no-sync cse151b-sc \
        --input "$input" \
        --output "$out" \
        --resume \
        --chunk-size 100 \
        --k 1 \
        --temperature 0.6 --top-p 0.95 \
        --prompt "$prompt" \
        --gpu-mem-util 0.92 \
        --max-model-len 32768 \
        --max-num-seqs 64 \
        --allocate-tokens 2>&1 | grep -E "^\[sc\]|Wrote|Accuracy"
}

declare -a PAIRS=(
    "olympiad:runj_v3_olympiad"
    "trig:runj_v3_trig"
    "geometry:runj_v3_geom"
    "stats_hyp_test:runj_v3_stats_hyp"
    "stats_regression:runj_v3_stats_reg"
    "stats_descriptive:runj_v3_stats_desc"
    "calculus:runj_v3_calc"
    "prob_combi:runj_v3_prob"
    "number_alg:runj_v3_number_alg"
)

# Baseline (Run F) — --resume picks up only new IDs from expanded subsets
echo ""
echo "=== Baseline pass (Run F, --resume on expanded subsets) ==="
for pair in "${PAIRS[@]}"; do
    topic="${pair%%:*}"
    run_one "$topic" "runf" "baseline"
done

# Variants (Run J v3, one branch each)
echo ""
echo "=== Variant pass (Run J v3 branch prompts) ==="
for pair in "${PAIRS[@]}"; do
    topic="${pair%%:*}"
    prompt="${pair##*:}"
    run_one "$topic" "$prompt" "variant"
done

echo ""
echo "=== $(date "+%H:%M:%S") V3 ABLATION DONE ==="
echo ""
echo "=== Re-judging all outputs ==="
for f in results/runj_ablation_*.jsonl; do
    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py "$f" 2>&1 | grep -E "Re-judged|New correct"
done

echo ""
echo "=== V3 vs V2 vs V1 vs Baseline summary ==="
PYTHONPATH=src .venv/bin/python scripts/summarize_v3_ablation.py

echo ""
echo "=== Deep review ==="
PYTHONPATH=src .venv/bin/python scripts/inspect_runj_ablation.py \
    > reports/runj_ablation_v3_review.md 2>&1
echo "Wrote deep review to reports/runj_ablation_v3_review.md"

echo ""
echo "=== V3 ablation done at $(date "+%H:%M:%S") ==="
rm -f '"$PID_FILE"'
' < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started Run J v3 ablation (9 baseline + 9 variant on expanded 313-q subsets).
  PID:    $PID
  Log:    $LOG

Will run up to 18 inferences (~95 min total).
- baseline: --resume picks up only NEW questions vs old subsets
  (geometry +33, stats_hyp +9 = 42 new baselines)
- variants: full 9 branches × all available questions

Then auto re-judge + summary + deep review.

v2 variant outputs archived to results/v2_archive/.

Monitor: tail -f $LOG
EOF
