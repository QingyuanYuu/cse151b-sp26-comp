#!/usr/bin/env bash
#
# Run J v2 ablation — re-test the 9 branches with revised prompts.
# v1 prompts had Q:/A: examples + verbatim word rules → 18 losses on
# 271 questions. v2 strips examples + adds multi-part counting + format-
# matching rules. Compare against the SAME v1 baselines (Run F) which
# are already on disk in results/runj_ablation_<topic>_baseline.jsonl.
#
# Total: 9 inferences (variants only), ~45 min K=1 wallclock.
#
# Usage: scripts/run_j_ablation_v2.sh

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

mkdir -p logs results

LOG="logs/runj_ablation_v2.log"
PID_FILE="logs/runj_ablation_v2.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running (PID $(cat "$PID_FILE"))."
    exit 1
fi
rm -f "$PID_FILE"

setsid bash -c '
set -e

run_one() {
    local topic="$1" prompt="$2"
    local input="data/eval_subsets/eval_${topic}.jsonl"
    local out="results/runj_ablation_${topic}_variant.jsonl"
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

# Topic → v2 ablation prompt name mapping
declare -a PAIRS=(
    "olympiad:runj_v2_olympiad"
    "trig:runj_v2_trig"
    "geometry:runj_v2_geom"
    "stats_hyp_test:runj_v2_stats_hyp"
    "stats_regression:runj_v2_stats_reg"
    "stats_descriptive:runj_v2_stats_desc"
    "calculus:runj_v2_calc"
    "prob_combi:runj_v2_prob"
    "number_alg:runj_v2_number_alg"
)

# Run each v2 branch variant on its target topic subset
for pair in "${PAIRS[@]}"; do
    topic="${pair%%:*}"
    prompt="${pair##*:}"
    run_one "$topic" "$prompt"
done

echo ""
echo "=== $(date "+%H:%M:%S") V2 VARIANTS DONE ==="
echo ""
echo "=== Re-judging v2 variant outputs ==="
for f in results/runj_ablation_*_variant.jsonl; do
    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py "$f" 2>&1 | grep -E "Re-judged|New correct"
done

echo ""
echo "=== V2 vs V1 vs Baseline summary ==="
PYTHONPATH=src .venv/bin/python scripts/summarize_v2_ablation.py

echo ""
echo "=== Deep review ==="
PYTHONPATH=src .venv/bin/python scripts/inspect_runj_ablation.py \
    > reports/runj_ablation_v2_review.md 2>&1
echo "Wrote deep review to reports/runj_ablation_v2_review.md"

echo ""
echo "=== V2 ablation done at $(date "+%H:%M:%S") ==="
rm -f '"$PID_FILE"'
' < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started Run J v2 ablation (variants only — baseline already on disk).
  PID:    $PID
  Log:    $LOG

Will run 9 inferences (~5 min/each = ~45 min total).
Then auto re-judge + summary + deep review.

Monitor: tail -f $LOG
EOF
