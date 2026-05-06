#!/usr/bin/env bash
#
# Diagnostic baseline: Phase 0 starter prompts + K=8 self-consistency on
# val_225. The point is to verify that the SC mechanism actually delivers
# variance reduction on this dataset / model BEFORE we invest in any
# self-distillation pool.
#
# Why this config:
# - --prompt phase0: starter prompts (single-shot leaderboard 0.575). These
#   are the only prompts proven on private. v6 just got falsified by a
#   leaderboard submission — 13.5pp regression, see HANDOFF.md §9.
# - --k 8: matches the gap_analysis.md recommended K range (5–8). Enough
#   for variance reduction without burning hours.
# - --val data/val_indices.json: only the 225 stratified val ids. ~1–2h
#   on Blackwell vs ~6–8h on the full 901 train rows. Diagnostic, not
#   self-distillation.
# - --max-num-seqs 64: K=8 with ~225 prompts → 1800 sequences total. 64
#   concurrent decode streams = ~8 questions in flight, plenty for KV cache.
# - --chunk-size 50: every 50 prompts (i.e., every 8 chunks for val), flush
#   to disk + fsync. Resume-safe.
#
# Usage:
#     scripts/run_sc_phase0_k8_val.sh           # start (or resume)
#     scripts/sc_status.sh                       # ⚠️ that script is hardcoded
#                                                #    to the v6 paths; just
#                                                #    use tail -f for now.
#     tail -f logs/sc_phase0_k8_val.log
#     kill -TERM "$(cat logs/sc_phase0_k8_val.pid)"

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

OUTPUT="results/sc_phase0_k8_val.jsonl"
LOG="logs/sc_phase0_k8_val.log"
PID_FILE="logs/sc_phase0_k8_val.pid"

mkdir -p logs results

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running with PID $(cat "$PID_FILE")."
    exit 1
fi
rm -f "$PID_FILE"

setsid bash -c "
    uv run --no-sync cse151b-sc \
        --input data/public.jsonl \
        --val data/val_indices.json \
        --output '$OUTPUT' \
        --resume \
        --chunk-size 50 \
        --k 8 \
        --temperature 0.7 --top-p 0.95 \
        --prompt phase0 \
        --gpu-mem-util 0.92 \
        --max-model-len 24576 \
        --max-num-seqs 64 \
        --max-tokens 16384
    rc=\$?
    rm -f '$PID_FILE'
    exit \$rc
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started Phase 0 + K=8 SC on val_225.
  PID:    $PID
  Log:    $LOG
  Output: $OUTPUT (chunked, resumable)
  Stop:   kill -TERM $PID

Expected wallclock: 1–2h on Blackwell.
After completion:
  uv run --no-sync python -c "
import json
rows = [json.loads(l) for l in open('$OUTPUT')]
correct = sum(r['correct'] for r in rows if 'correct' in r)
total = sum(1 for r in rows if 'correct' in r)
solv = sum(r.get('solvable_but_missed', False) for r in rows)
print(f'val acc K=8 SC = {correct}/{total} = {100*correct/total:.2f}%')
print(f'solvable_but_missed: {solv}/{total}')"

  # Compare to baseline_v0_val.json (Phase 0 single shot on same val_225):
  python -c "
import json
b = json.load(open('reports/baseline_v0_val.json'))
print(f'phase 0 single-shot val: {b}')"

The key question: does Phase 0 + K=8 SC beat Phase 0 single-shot val (56.44%)
by a meaningful margin (≥ 3pp)? If yes, SC mechanism works on this dataset
and we can scale K up. If no, the regression on private was structural in
v6, not a "K too small" issue.
EOF
