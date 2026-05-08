#!/usr/bin/env bash
#
# K=32 self-consistency on data/public.jsonl with Run F (final) prompt.
#
# Used as the K-curve validation: K=8 → K=32 marginal gain on public 1126.
# If K=32 vs K=8 shows clear lift, justifies running K=32 on private later.
#
# Wallclock estimate: ~33-38h on Blackwell (1126 × K=32 = 36032 samples,
# vLLM prefix-share helps but K=32 memory pressure caps batching).

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

INPUT="data/public.jsonl"
OUTPUT="results/sc_runf_k32_public.jsonl"
LOG="logs/sc_runf_k32_public.log"
PID_FILE="logs/sc_runf_k32_public.pid"

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

setsid bash -c "
    set -e
    uv run --no-sync cse151b-sc \
        --input '$INPUT' \
        --output '$OUTPUT' \
        --resume \
        --chunk-size 30 \
        --k 32 \
        --temperature 0.7 --top-p 0.95 \
        --prompt runf \
        --gpu-mem-util 0.92 \
        --max-model-len 32768 \
        --max-num-seqs 16 \
        --allocate-tokens

    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py '$OUTPUT' 2>&1 | tail -8

    rc=\$?
    rm -f '$PID_FILE'
    exit \$rc
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started K=32 SC + Run F on public 1126.
  PID:    $PID
  Log:    $LOG
  Output: $OUTPUT
  Stop:   kill -TERM -$PID

Started: $(date '+%Y-%m-%d %H:%M %Z')
Expected: ~33-38h on Blackwell.

Monitor: tail -f $LOG
EOF
