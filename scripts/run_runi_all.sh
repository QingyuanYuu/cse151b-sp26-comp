#!/usr/bin/env bash
#
# Sequential Run I (final): val_225 → full public 1126 → private 943.
# Single-shot K=1 with auto v2 budget. Detached via setsid; CSV
# auto-generated for private.
#
# Total wallclock estimate (sequential, K=1, Blackwell):
#   - val_225  ~14-16 min
#   - public_1126  ~75-90 min
#   - private_943  ~70-85 min
#   - Total: ~3 hours

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

LOG="logs/runi_chain.log"
PID_FILE="logs/runi_chain.pid"
mkdir -p logs results submissions

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running with PID $(cat "$PID_FILE")."
    exit 1
fi
rm -f "$PID_FILE"

setsid bash -c '
set -e

run_step() {
    local label="$1" input="$2" out_jsonl="$3" extra="$4"
    echo
    echo "=== $(date "+%H:%M:%S") $label ==="
    eval uv run --no-sync cse151b-sc \
        --input "$input" \
        --output "$out_jsonl" \
        --resume \
        --chunk-size 200 \
        --k 1 \
        --temperature 0.6 --top-p 0.95 \
        --prompt runi \
        --gpu-mem-util 0.92 \
        --max-model-len 32768 \
        --max-num-seqs 64 \
        --allocate-tokens \
        $extra
    echo "=== $(date "+%H:%M:%S") $label DONE ==="
}

run_step "Step 1/3: Run I val_225" "data/public.jsonl" "results/runi_val.jsonl" "--val data/val_indices.json"
run_step "Step 2/3: Run I public 1126" "data/public.jsonl" "results/runi_public.jsonl" ""
run_step "Step 3/3: Run I private 943" "data/private.jsonl" "results/runi_private.jsonl" ""

echo
echo "=== $(date "+%H:%M:%S") Generating private CSV ==="
uv run --no-sync python <<PY
import csv, json, pathlib
src = pathlib.Path("results/runi_private.jsonl")
dst = pathlib.Path("submissions/runi_k1_private.csv")
dst.parent.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(src)]
seen = set()
with open(dst, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "response"])
    for r in rows:
        rid = r["id"]
        if rid in seen: continue
        seen.add(rid)
        resp = r.get("winning_response", "") or (r.get("all_responses", [""])[0] if r.get("all_responses") else "")
        w.writerow([rid, resp])
print(f"Wrote {len(seen)} rows to {dst}")
boxed = sum(1 for r in rows if "\\\\boxed{" in (r.get("winning_response", "") or ""))
print(f"Boxed: {boxed}/{len(rows)} = {100*boxed/len(rows):.1f}%")
PY

echo
echo "=== $(date "+%H:%M:%S") ALL DONE ==="
rm -f '"$PID_FILE"'
' < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started Run I chain: val → public → private.
  PID:    $PID
  Log:    $LOG

Output files (will appear sequentially):
  results/runi_val.jsonl
  results/runi_public.jsonl
  results/runi_private.jsonl
  submissions/runi_k1_private.csv

Started: $(date '+%Y-%m-%d %H:%M %Z')
Expected: ~3 hours total. Finishes ~01:00-01:30 PDT.

Monitor: tail -f $LOG
EOF
