#!/usr/bin/env bash
#
# Run J branch ablation (UPDATED 8-branch design):
# Test each Run J branch on its target topic eval subset.
#
# Branches tested (and their eval subset sizes from data/eval_subsets/):
#   trig              43
#   geometry          50
#   stats_hyp_test    50
#   stats_regression  28
#   stats_descriptive 50
#   calculus           7  ← weak signal (free-form calc is tiny on this corpus)
#   prob_combi        24
#   discrete_math      6  ← weak signal
#
# Plus baseline Run F on each subset for paired comparison.
# Total: 16 inferences, ~1h K=1 wallclock.
#
# After SC finishes: auto re-judges with Judger.auto_judge (Run J ablation
# was launched after the bug fix in self_consistency.py, so `correct`
# should already be right; rejudge is a sanity belt-and-braces step).
#
# Usage: scripts/run_j_ablation.sh

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

mkdir -p logs results

LOG="logs/runj_ablation.log"
PID_FILE="logs/runj_ablation.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running (PID $(cat "$PID_FILE"))."
    exit 1
fi
rm -f "$PID_FILE"

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

# Topic → ablation prompt name mapping
# (each ablation enables ONE branch; rest fall through to Run F generic)
declare -a PAIRS=(
    "trig:runj_trig"
    "geometry:runj_geom"
    "stats_hyp_test:runj_stats_hyp"
    "stats_regression:runj_stats_reg"
    "stats_descriptive:runj_stats_desc"
    "calculus:runj_calc"
    "prob_combi:runj_prob"
    "discrete_math:runj_discrete"
)

# Run baseline (Run F) on each topic subset (control)
for pair in "${PAIRS[@]}"; do
    topic="${pair%%:*}"
    run_one "$topic" "runf" "baseline"
done

# Run each branch variant on its target topic subset
for pair in "${PAIRS[@]}"; do
    topic="${pair%%:*}"
    prompt="${pair##*:}"
    run_one "$topic" "$prompt" "variant"
done

echo ""
echo "=== $(date "+%H:%M:%S") ALL VARIANTS DONE ==="
echo ""
echo "=== Re-judging ablation outputs (belt-and-braces) ==="
for f in results/runj_ablation_*.jsonl; do
    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py "$f" 2>&1 | grep -E "Re-judged|New correct"
done

echo ""
echo "=== Summary ==="
PYTHONPATH=src .venv/bin/python scripts/summarize_j_ablation.py

echo ""
echo "=== Building final-J composition from ablation results ==="
PYTHONPATH=src .venv/bin/python scripts/build_final_j.py

echo ""
echo "=== Launching final-J on full public 1126 (auto-chain) ==="
scripts/run_final_j_public.sh

echo ""
echo "=== Ablation chain done at $(date "+%H:%M:%S") ==="
echo "Final-J public run is now running detached. Monitor: logs/runj_final_public.log"
rm -f '"$PID_FILE"'
' < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started Run J ablation (UPDATED 8-branch design).
  PID:    $PID
  Log:    $LOG

Will run 16 inferences:
  8 baselines (Run F on each topic subset)
  8 variants  (each Run J branch on its target subset)

Then auto re-judge with Judger.auto_judge and run summarize_j_ablation.py.

Total wallclock: ~1.5h on Blackwell K=1.
Monitor: tail -f $LOG
EOF
