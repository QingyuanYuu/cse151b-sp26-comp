#!/usr/bin/env bash
#
# Run K=32 self-consistency on data/public_train.jsonl, fully detached from
# any parent shell / harness so it survives session reapers.
#
# Features:
# - setsid: process becomes its own session leader; SIGHUP to parent doesn't
#   propagate. Survives Claude Code bg-task timeouts and ssh disconnects.
# - Incremental checkpointing via cse151b-sc --chunk-size 50 --resume:
#   output JSONL is flushed + fsync'd every 50 prompts. Re-run this script
#   to resume from where it left off.
# - PID file at logs/sc_v6_k32.pid for monitoring/killing from another shell.
#
# Usage:
#     scripts/run_sc_k32.sh           # start (or resume) K=32 SC
#     tail -f logs/sc_v6_k32.log      # watch progress
#     kill -TERM "$(cat logs/sc_v6_k32.pid)"   # graceful stop
#
# Cancel-safe: a graceful kill mid-chunk loses only the current chunk's
# in-progress samples; previously-completed chunks are durable on disk.

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

OUTPUT="results/sc_v6_k32_public_train.jsonl"
LOG="logs/sc_v6_k32.log"
PID_FILE="logs/sc_v6_k32.pid"

mkdir -p logs results

# Auto-regenerate data/public_train.jsonl if missing (it's gitignored;
# derived from data/public.jsonl minus data/val_indices.json's val_ids).
if [ ! -f data/public_train.jsonl ]; then
    echo "Building data/public_train.jsonl (gitignored, derived)..."
    uv run --no-sync python -c "
import json, pathlib
val_ids = set(json.loads(pathlib.Path('data/val_indices.json').read_text())['val_ids'])
rows = [json.loads(l) for l in open('data/public.jsonl')]
train = [r for r in rows if r['id'] not in val_ids]
with open('data/public_train.jsonl', 'w') as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + chr(10))
print(f'Wrote {len(train)} train rows ({len(rows)} - {len(val_ids)} val).')
"
fi

# Refuse duplicate launch.
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running with PID $(cat "$PID_FILE")."
    echo "  Log:  $LOG"
    echo "  Kill: kill -TERM $(cat "$PID_FILE")"
    exit 1
fi
rm -f "$PID_FILE"

# Note: --resume is safe even if $OUTPUT doesn't exist yet (cse151b-sc handles
# both fresh-start and continue-from-checkpoint paths).
setsid bash -c "
    uv run --no-sync cse151b-sc \
        --input data/public_train.jsonl \
        --output '$OUTPUT' \
        --resume \
        --chunk-size 50 \
        --k 32 \
        --temperature 0.7 --top-p 0.95 \
        --prompt current \
        --gpu-mem-util 0.92 \
        --max-model-len 24576 \
        --max-num-seqs 128 \
        --max-tokens 16384
    rc=\$?
    rm -f '$PID_FILE'
    exit \$rc
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started SC K=32, fully detached.
  PID:    $PID
  Log:    $LOG
  Output: $OUTPUT (chunked, resumable)
  Stop:   kill -TERM $PID

To monitor: tail -f $LOG
To check progress (compact): scripts/sc_status.sh
EOF
