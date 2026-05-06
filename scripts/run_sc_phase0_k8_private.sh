#!/usr/bin/env bash
#
# Phase 0 + K=8 SC on data/private.jsonl (943 questions) → JSONL → CSV.
# Designed for overnight runs: launches detached via setsid, chunk-resumable.
#
# Estimated wallclock: 7–9h on Blackwell (4.2× the val_225 run).
# Output: results/sc_phase0_k8_private.jsonl (raw SC pool with all_responses)
#         submissions/sc_phase0_k8_private.csv (Kaggle-ready id,response)
#
# CSV conversion runs automatically once SC finishes. If you wake up and only
# the JSONL exists (CSV step crashed), re-run scripts/sc_to_csv.sh manually.
#
# Usage:
#     scripts/run_sc_phase0_k8_private.sh
#     tail -f logs/sc_phase0_k8_private.log
#     kill -TERM "$(cat logs/sc_phase0_k8_private.pid)"

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

INPUT="data/private.jsonl"
OUTPUT="results/sc_phase0_k8_private.jsonl"
CSV="submissions/sc_phase0_k8_private.csv"
LOG="logs/sc_phase0_k8_private.log"
PID_FILE="logs/sc_phase0_k8_private.pid"

mkdir -p logs results submissions

# Sanity: private.jsonl must be present (gitignored, scp'd from elsewhere).
if [ ! -f "$INPUT" ]; then
    echo "ERROR: $INPUT does not exist on this machine."
    echo "Get it from the 4090 box first:"
    echo "    scp <4090>:/path/to/$INPUT $INPUT"
    exit 1
fi

ROW_COUNT=$(wc -l < "$INPUT")
echo "Found $INPUT with $ROW_COUNT rows."
if [ "$ROW_COUNT" -lt 900 ]; then
    echo "WARNING: $INPUT has only $ROW_COUNT rows; expected 943. Check that"
    echo "you copied the full Kaggle private set, not a sample."
fi

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
        --prompt phase0 \
        --gpu-mem-util 0.92 \
        --max-model-len 26624 \
        --max-num-seqs 48 \
        --allocate-tokens \
        --max-tokens-floor 12288 \
        --max-tokens-ceiling 20480

    echo
    echo '=== SC complete; converting to CSV ==='
    # cse151b-submit expects a JSONL of {id, response, ...} per row. The SC
    # output uses 'winning_response'; convert via a tiny in-line transform
    # rather than relying on cse151b-submit's exact arg surface.
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

# Quick sanity check: row count + boxed presence
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
Started Phase 0 + K=8 SC on $INPUT.
  PID:    $PID
  Log:    $LOG
  Raw:    $OUTPUT (chunked, resumable)
  CSV:    $CSV (auto-generated when SC finishes)
  Stop:   kill -TERM -$PID

Expected wallclock: 7–9h. Started: $(date '+%Y-%m-%d %H:%M %Z')

To monitor in another shell:
    tail -f $LOG
    wc -l $OUTPUT
EOF
