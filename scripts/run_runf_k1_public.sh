#!/usr/bin/env bash
#
# Single-shot (K=1) inference on full data/public.jsonl (1126 questions)
# with Run F (final) prompt + v2 budget. Useful as a fresh public-set
# baseline measurement — comparable to reports/baseline_public_v1.md
# (Phase 1 + 12k flat budget, 60.12% on full public).
#
# Wallclock: ~70-90 min on Blackwell (1126 questions × 1 sample, v2
# budget stretches some to 30k tokens).

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

INPUT="data/public.jsonl"
OUTPUT="results/runf_k1_public.jsonl"
LOG="logs/runf_k1_public.log"
PID_FILE="logs/runf_k1_public.pid"

mkdir -p logs results

if [ ! -f "$INPUT" ]; then
    echo "ERROR: $INPUT not found"
    exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running with PID $(cat "$PID_FILE")."
    exit 1
fi
rm -f "$PID_FILE"

setsid bash -c "
    set -e
    uv run --no-sync cse151b-sc \
        --input '$INPUT' \
        --output '$OUTPUT' \
        --resume \
        --chunk-size 200 \
        --k 1 \
        --temperature 0.6 --top-p 0.95 \
        --prompt runf \
        --gpu-mem-util 0.92 \
        --max-model-len 32768 \
        --max-num-seqs 64 \
        --allocate-tokens

    rc=\$?
    rm -f '$PID_FILE'
    exit \$rc
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started K=1 + Run F + v2 on $INPUT.
  PID:    $PID
  Log:    $LOG
  Raw:    $OUTPUT (chunked, resumable)
  Stop:   kill -TERM -$PID

Started: $(date '+%Y-%m-%d %H:%M %Z')
Expected: ~70-90 min.

Monitor: tail -f $LOG
EOF
