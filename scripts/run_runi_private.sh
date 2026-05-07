#!/usr/bin/env bash
#
# Single-shot K=1 Run I on data/private.jsonl with auto v2 budget.
# Auto-emits Kaggle CSV.
#
# Why: Run I val_225 = 64.89% (best across all runs, +2.2pp vs Run F).
# The original Run I private chain was killed at 200/943 mid-run after a
# misdiagnosed regression (which turned out to be a judger bug, not a
# real Run I regression). Re-running for a second independent submission.
#
# Wallclock: ~70-80 min on Blackwell.
#
# Usage:
#     scripts/run_runi_private.sh

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

INPUT="data/private.jsonl"
OUTPUT="results/runi_k1_private.jsonl"
CSV="submissions/runi_k1_private.csv"
LOG="logs/runi_k1_private.log"
PID_FILE="logs/runi_k1_private.pid"

mkdir -p logs results submissions

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
        --chunk-size 200 \
        --k 1 \
        --temperature 0.6 --top-p 0.95 \
        --prompt runi \
        --gpu-mem-util 0.92 \
        --max-model-len 32768 \
        --max-num-seqs 64 \
        --allocate-tokens

    echo
    echo '=== inference complete; converting to CSV ==='
    uv run --no-sync python <<'PY'
import csv
import json
import pathlib

src = pathlib.Path('$OUTPUT')
dst = pathlib.Path('$CSV')
dst.parent.mkdir(parents=True, exist_ok=True)

rows = [json.loads(l) for l in open(src)]
seen = set()
with open(dst, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'response'])
    for r in rows:
        rid = r['id']
        if rid in seen:
            continue
        seen.add(rid)
        resp = r.get('winning_response', '')
        w.writerow([rid, resp])

print(f'Wrote {len(seen)} rows -> {dst}')
boxed = sum(1 for r in rows if '\\\\boxed{' in r.get('winning_response', ''))
print(f'Rows with \\\\boxed{{}}: {boxed} / {len(rows)}')
PY

    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py '$OUTPUT' 2>&1 | tail -8

    rc=\$?
    rm -f '$PID_FILE'
    exit \$rc
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started Run I K=1 on private 943.
  PID:    $PID
  Log:    $LOG
  Raw:    $OUTPUT
  CSV:    $CSV (Kaggle submission format)
  Stop:   kill -TERM -$PID

Started: $(date '+%Y-%m-%d %H:%M %Z')
Expected: ~70-80 min on Blackwell.

Monitor: tail -f $LOG
EOF
