#!/usr/bin/env bash
#
# Single-shot (K=1) inference on data/private.jsonl with Run F (final)
# prompt + v2 budget. Tests the prompt improvement WITHOUT SC variance
# reduction — fast (~30 min) but noisier than K=4 or K=8.
#
# Reference points:
# - Run B single-shot (no v2 budget):                       0.600
# - K=8 SC + Phase 0 + budget v1 (already submitted):       0.611
# - This run (K=1 + Run F prompt + v2 budget) expected:     0.605-0.615
#
# What this isolates: pure Run F prompt + v2 budget gain vs Run B
# baseline. Subtracting from K=8 SC + Phase 0's 0.611 also tells us
# whether the prompt change alone is enough or if SC contributes most.

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

INPUT="data/private.jsonl"
OUTPUT="results/runf_k1_private.jsonl"
CSV="submissions/runf_k1_private.csv"
LOG="logs/runf_k1_private.log"
PID_FILE="logs/runf_k1_private.pid"

mkdir -p logs results submissions

if [ ! -f "$INPUT" ]; then
    echo "ERROR: $INPUT not found"
    exit 1
fi
ROW_COUNT=$(wc -l < "$INPUT")
echo "Found $INPUT with $ROW_COUNT rows."

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
        # K=1 — winning_response is the only response.
        resp = r.get('winning_response', '')
        w.writerow([rid, resp])

print(f'Wrote {len(seen)} rows -> {dst}')
boxed = sum(1 for r in rows if '\\\\boxed{' in r.get('winning_response', ''))
print(f'Rows with \\\\boxed{{}}: {boxed} / {len(rows)}')
PY

    rc=\$?
    rm -f '$PID_FILE'
    exit \$rc
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started K=1 + Run F + v2 budget on $INPUT.
  PID:    $PID
  Log:    $LOG
  Raw:    $OUTPUT (chunked, resumable)
  CSV:    $CSV (auto-generated)
  Stop:   kill -TERM -$PID

Started: $(date '+%Y-%m-%d %H:%M %Z')
Expected: ~30-40 min.

Monitor: tail -f $LOG
EOF
