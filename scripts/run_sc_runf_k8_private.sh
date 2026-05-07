#!/usr/bin/env bash
#
# K=8 self-consistency on data/private.jsonl with **Run F (final) prompt + v2
# budget**. Designed for the user's last leaderboard quota of the day.
#
# Run F final = Run F prompt (sqrt75 example, end-with-box, MCQ elim) +
# v2 budget (floor 16k, MCQ cap 22k, multi cap 30k). Auto-enabled when
# --prompt runf is set under the updated self_consistency.py dispatch.
#
# Expected leaderboard (per jason/dev ee43f97 commit message):
#   - Run B single-shot baseline:                    0.600
#   - K=8 SC + Phase 0 + budget v1 (already submitted): 0.611
#   - This run (K=8 SC + Run F + v2 budget):         0.615 - 0.625 expected
#
# Wallclock: ~7-9h on Blackwell (943 × K=8 = 7544 samples; v2 budget
# stretches some questions to 30k tokens, dragging batch tail).
#
# Usage:
#     scripts/run_sc_runf_k8_private.sh
#     tail -f logs/sc_runf_k8_private.log
#     scripts/sc_status.sh   (manually adjust paths)

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

INPUT="data/private.jsonl"
OUTPUT="results/sc_runf_k8_private.jsonl"
CSV="submissions/sc_runf_k8_private.csv"
LOG="logs/sc_runf_k8_private.log"
PID_FILE="logs/sc_runf_k8_private.pid"

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
        --chunk-size 50 \
        --k 8 \
        --temperature 0.7 --top-p 0.95 \
        --prompt runf \
        --gpu-mem-util 0.92 \
        --max-model-len 32768 \
        --max-num-seqs 24 \
        --allocate-tokens

    echo
    echo '=== SC complete; converting to CSV ==='
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

    rc=\$?
    rm -f '$PID_FILE'
    exit \$rc
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started K=8 SC + Run F + v2 budget on $INPUT.
  PID:    $PID
  Log:    $LOG
  Raw:    $OUTPUT (chunked, resumable)
  CSV:    $CSV (auto-generated)
  Stop:   kill -TERM -$PID

Started: $(date '+%Y-%m-%d %H:%M %Z')
Expected: ~7-9h. Finishes ~02:00-04:00 PDT.

Monitor:
    tail -f $LOG
    wc -l $OUTPUT
EOF
