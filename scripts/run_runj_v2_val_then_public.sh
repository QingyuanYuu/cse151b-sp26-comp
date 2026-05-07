#!/usr/bin/env bash
#
# Run J v2 final (5 winning branches) on val_225 then full public 1126.
# K=1, sequential, ~85 min total.
#
# Composition: olympiad, stats_hyp_test, stats_descriptive, prob_combi, number_alg
# (the 5 branches with positive direction in v2 ablation).
# Other free questions and MCQ fall through to Run F generic.
#
# Usage: scripts/run_runj_v2_val_then_public.sh

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

LOG="logs/runj_v2_final_chain.log"
PID_FILE="logs/runj_v2_final_chain.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running (PID $(cat "$PID_FILE"))."
    exit 1
fi
rm -f "$PID_FILE"

echo "Composition (data/runj_v2_final_branches.txt):"
sed 's/^/  /' data/runj_v2_final_branches.txt

setsid bash -c '
    set -e
    SC_RUN() {
        local label="$1" input="$2" output="$3" extra="$4"
        echo
        echo "=== $(date "+%H:%M:%S") $label ==="
        eval uv run --no-sync cse151b-sc \
            --input "$input" \
            --output "$output" \
            --resume \
            --chunk-size 200 \
            --k 1 \
            --temperature 0.6 --top-p 0.95 \
            --prompt runj_v2_final \
            --gpu-mem-util 0.92 \
            --max-model-len 32768 \
            --max-num-seqs 64 \
            --allocate-tokens \
            $extra
        echo "=== $(date "+%H:%M:%S") $label DONE ==="
    }

    SC_RUN "Step 1/2: val_225" "data/public.jsonl" "results/runj_v2_final_val.jsonl" "--val data/val_indices.json"
    SC_RUN "Step 2/2: public 1126" "data/public.jsonl" "results/runj_v2_final_public.jsonl" ""

    echo
    echo "=== Re-judging both outputs ==="
    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py results/runj_v2_final_val.jsonl 2>&1 | tail -5
    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py results/runj_v2_final_public.jsonl 2>&1 | tail -5

    echo
    echo "=== Chain done at $(date "+%H:%M:%S") ==="
    rm -f '"$PID_FILE"'
' < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started Run J v2 final chain (val_225 → public 1126).
  PID: $PID
  Log: $LOG
  val output:    results/runj_v2_final_val.jsonl    (~14-16 min)
  public output: results/runj_v2_final_public.jsonl (~70 min)

Total: ~85 min.

Started: $(date '+%Y-%m-%d %H:%M %Z')
Monitor: tail -f $LOG
EOF
