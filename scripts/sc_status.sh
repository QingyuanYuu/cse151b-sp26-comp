#!/usr/bin/env bash
#
# Compact status dashboard for an in-progress run_sc_k32.sh.
# Reads the output JSONL (counts completed prompts) and the log (latest tqdm).

set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT="results/sc_v6_k32_public_train.jsonl"
LOG="logs/sc_v6_k32.log"
PID_FILE="logs/sc_v6_k32.pid"
TOTAL=901  # 1126 - 225 val_ids

echo "== SC K=32 status =="
date

# Process
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    PID=$(cat "$PID_FILE")
    ETIME=$(ps -o etime= -p "$PID" | xargs)
    echo "Process: PID $PID alive (uptime $ETIME)"
else
    echo "Process: NOT running"
fi

# Output file
if [ -f "$OUTPUT" ]; then
    DONE=$(wc -l < "$OUTPUT")
    echo "Output:  $DONE / $TOTAL completed ($(awk -v d=$DONE -v t=$TOTAL 'BEGIN{printf "%.1f%%", 100*d/t}'))"
    SIZE=$(du -h "$OUTPUT" | cut -f1)
    echo "         file size $SIZE"
else
    echo "Output:  no file yet"
fi

# GPU
echo "GPU:     $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"

# Latest tqdm line from log
if [ -f "$LOG" ]; then
    LATEST=$(tr '\r' '\n' < "$LOG" | grep -E "Processed prompts:" | tail -1)
    if [ -n "$LATEST" ]; then
        echo "tqdm:    $LATEST"
    fi
    LATEST_SC=$(grep -E "^\[sc\]" "$LOG" | tail -3)
    if [ -n "$LATEST_SC" ]; then
        echo "Recent [sc] events:"
        echo "$LATEST_SC" | sed 's/^/    /'
    fi
fi
