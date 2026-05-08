#!/usr/bin/env bash
#
# K=1 single-shot Run J v3 final on private 943. Auto-emits Kaggle CSV.
#
# v3 final on public 1126 = 723/1126 = 64.21% vs Run F 63.14% = +1.07pp.
# Expected leaderboard improvement on private: ~0.005-0.015 over Run F K=1's 0.632.
#
# Wallclock: ~70-80 min on Blackwell.

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

INPUT="data/private.jsonl"
OUTPUT="results/runj_v3_final_private.jsonl"
CSV="submissions/runj_v3_final_private.csv"
LOG="logs/runj_v3_final_private.log"
PID_FILE="logs/runj_v3_final_private.pid"

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

echo "Composition (data/runj_v3_final_branches.txt):"
sed 's/^/  /' data/runj_v3_final_branches.txt

setsid bash -c "
    set -e
    uv run --no-sync cse151b-sc \
        --input '$INPUT' \
        --output '$OUTPUT' \
        --resume \
        --chunk-size 200 \
        --k 1 \
        --temperature 0.6 --top-p 0.95 \
        --prompt runj_v3_final \
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
Started Run J v3 final K=1 on private 943.
  PID:    $PID
  Log:    $LOG
  Raw:    $OUTPUT
  CSV:    $CSV (Kaggle submission format)

Started: $(date '+%Y-%m-%d %H:%M %Z')
Expected: ~70-80 min on Blackwell.
EOF
