#!/usr/bin/env bash
#
# K=16 self-consistency on data/public.jsonl with Run F prompt.
# Used as SFT distillation pool (excludes val_ids in prepare-sft step).
#
# Wallclock: ~17-20h on Blackwell (1126 × K=16 = 18016 samples).
#
# Tradeoff vs K=32: K=16 catches ~95% of pass@K=32's hard examples,
# saves ~17h. Marginal SFT pool quality difference ~+0.5pp expected.

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

INPUT="data/public.jsonl"
OUTPUT="results/sc_runf_k16_public.jsonl"
LOG="logs/sc_runf_k16_public.log"
PID_FILE="logs/sc_runf_k16_public.pid"

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
        --chunk-size 40 \
        --k 16 \
        --temperature 0.7 --top-p 0.95 \
        --prompt runf \
        --gpu-mem-util 0.92 \
        --max-model-len 32768 \
        --max-num-seqs 20 \
        --allocate-tokens

    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py '$OUTPUT' 2>&1 | tail -8

    rc=\$?
    rm -f '$PID_FILE'
    exit \$rc
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started K=16 SC + Run F on public 1126.
  PID:    $PID
  Log:    $LOG
  Output: $OUTPUT

Started: $(date '+%Y-%m-%d %H:%M %Z')
Expected: ~17-20h on Blackwell.

Monitor: tail -f $LOG
EOF
