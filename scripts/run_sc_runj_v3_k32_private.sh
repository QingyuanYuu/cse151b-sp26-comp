#!/usr/bin/env bash
#
# K=32 self-consistency on data/private.jsonl with Run J v3 final prompt.
#
# Why v3: ablation +2.6pp aggregate vs baseline; val_225 +2.67pp vs Run F.
# Why K=32: SC majority vote over 32 samples — proven path, +1-3pp on
# top of K=1. Combined with v3 prompt expected ~0.66-0.68 leaderboard.
#
# Wallclock estimate: ~28-32h on Blackwell (943 × K=32 = 30176 samples,
# vLLM prefix-share helps but memory pressure caps batching).
#
# Output: submissions/sc_runj_v3_k32_private.csv

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

INPUT="data/private.jsonl"
OUTPUT="results/sc_runj_v3_k32_private.jsonl"
CSV="submissions/sc_runj_v3_k32_private.csv"
LOG="logs/sc_runj_v3_k32_private.log"
PID_FILE="logs/sc_runj_v3_k32_private.pid"

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

echo "Composition: data/runj_v3_final_branches.txt"
sed 's/^/  /' data/runj_v3_final_branches.txt

setsid bash -c "
    set -e
    uv run --no-sync cse151b-sc \
        --input '$INPUT' \
        --output '$OUTPUT' \
        --resume \
        --chunk-size 30 \
        --k 32 \
        --temperature 0.7 --top-p 0.95 \
        --prompt runj_v3_final \
        --gpu-mem-util 0.92 \
        --max-model-len 32768 \
        --max-num-seqs 16 \
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
Started K=32 SC + Run J v3 final on $INPUT.
  PID:    $PID
  Log:    $LOG
  Raw:    $OUTPUT (chunked, resumable)
  CSV:    $CSV
  Stop:   kill -TERM -$PID

Started: $(date '+%Y-%m-%d %H:%M %Z')
Expected: ~28-32h. Done ~tomorrow 20:00-24:00 PDT.

Monitor: tail -f $LOG
EOF
