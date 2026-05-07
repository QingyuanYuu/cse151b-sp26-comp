#!/usr/bin/env bash
#
# Run final-J prompt on full data/private.jsonl (943 questions) and
# auto-emit a submission CSV. K=1 + auto v2 budget.
#
# Wallclock: ~70-80 min on Blackwell.
#
# Prerequisite: data/runj_final_branches.txt must exist (written by
# scripts/build_final_j.py after ablation review). If absent,
# build_prompt_runj_final silently falls through to all 9 branches.
#
# Usage:
#     scripts/run_final_j_private.sh

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

INPUT="data/private.jsonl"
OUTPUT="results/runj_final_private.jsonl"
CSV="submissions/runj_final_private.csv"
LOG="logs/runj_final_private.log"
PID_FILE="logs/runj_final_private.pid"

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

if [ -f data/runj_final_branches.txt ]; then
    echo "Final-J composition (from ablation):"
    sed 's/^/  /' data/runj_final_branches.txt
else
    echo "WARNING: data/runj_final_branches.txt missing; using full 9-branch fallback."
fi

setsid bash -c "
    set -e
    uv run --no-sync cse151b-sc \
        --input '$INPUT' \
        --output '$OUTPUT' \
        --resume \
        --chunk-size 200 \
        --k 1 \
        --temperature 0.6 --top-p 0.95 \
        --prompt runj_final \
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

    # Belt-and-braces rejudge with Judger.auto_judge (same as public).
    # Private has no answer field so this is a no-op now, but keeps the
    # pipeline pattern uniform and ready if private answers ever publish.
    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py '$OUTPUT' 2>&1 | tail -8

    rc=\$?
    rm -f '$PID_FILE'
    exit \$rc
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started final-J on private 943.
  PID:    $PID
  Log:    $LOG
  Raw:    $OUTPUT
  CSV:    $CSV (Kaggle submission format)
  Stop:   kill -TERM -$PID

Started: $(date '+%Y-%m-%d %H:%M %Z')
Expected: ~70-80 min on Blackwell.

Monitor: tail -f $LOG
EOF
