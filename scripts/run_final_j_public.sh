#!/usr/bin/env bash
#
# Run final-J prompt (composition determined by build_final_j.py from
# ablation results) on full public.jsonl (1126 questions). Single-shot
# K=1 with auto v2 budget.
#
# Wallclock: ~80-90 min on Blackwell.
#
# Prerequisite: data/runj_final_branches.txt must exist (written by
# scripts/build_final_j.py after ablation). If absent, build_prompt_runj_final
# silently falls through to all 8 branches (= equivalent to runj).
#
# Usage:
#     scripts/run_final_j_public.sh

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

INPUT="data/public.jsonl"
OUTPUT="results/runj_final_public.jsonl"
LOG="logs/runj_final_public.log"
PID_FILE="logs/runj_final_public.pid"

mkdir -p logs results

if [ ! -f "$INPUT" ]; then
    echo "ERROR: $INPUT not found"
    exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running (PID $(cat "$PID_FILE"))."
    exit 1
fi
rm -f "$PID_FILE"

if [ -f data/runj_final_branches.txt ]; then
    echo "Final-J composition (from ablation):"
    sed 's/^/  /' data/runj_final_branches.txt
else
    echo "WARNING: data/runj_final_branches.txt missing; using full 8-branch fallback."
fi

setsid bash -c "
    set -e
    uv run --no-sync cse151b-sc \
        --input '$INPUT' \
        --output '$OUTPUT' \
        --resume \
        --chunk-size 200 \
        --k 1 \
        --temperature 0.6 --top-p 0.95 \
        --prompt runj_final \
        --gpu-mem-util 0.92 \
        --max-model-len 32768 \
        --max-num-seqs 64 \
        --allocate-tokens

    rc=\$?

    # Re-judge to ensure correct field uses Judger.auto_judge
    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py '$OUTPUT' 2>&1 | tail -10

    rm -f '$PID_FILE'
    exit \$rc
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started final-J on public 1126.
  PID:    $PID
  Log:    $LOG
  Output: $OUTPUT
  Stop:   kill -TERM -$PID

Started: $(date '+%Y-%m-%d %H:%M %Z')
Expected: ~80-90 min on Blackwell.

Monitor: tail -f $LOG
EOF
