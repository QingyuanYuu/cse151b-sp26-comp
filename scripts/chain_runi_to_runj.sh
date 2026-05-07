#!/usr/bin/env bash
#
# After the Run I chain finishes, automatically:
# 1. Run Run J ablation harness (~1.5h)
# 2. Build final J from ablation results (instant)
# 3. Run final J on public 1126 (~80-90min)
#
# Total: ~3h after Run I chain finishes.
#
# Usage:
#     scripts/chain_runi_to_runj.sh

set -euo pipefail

cd "$(dirname "$0")/.."

RUNI_PID_FILE="logs/runi_chain.pid"
LOG="logs/chain_runi_to_runj.log"

if [ ! -f "$RUNI_PID_FILE" ]; then
    echo "WARNING: $RUNI_PID_FILE not found — Run I chain not running?"
    echo "         Will skip the wait and start Run J ablation immediately."
    RUNI_PID=0
else
    RUNI_PID=$(cat "$RUNI_PID_FILE")
fi

setsid bash -c "
    if [ '$RUNI_PID' != '0' ]; then
        echo \"[\$(date)] Waiting for Run I chain PID $RUNI_PID to finish...\"
        while kill -0 $RUNI_PID 2>/dev/null; do
            sleep 60
        done
        echo \"[\$(date)] Run I chain done.\"
    fi

    sleep 10  # extra grace period for vLLM to release GPU
    echo \"[\$(date)] Starting Run J ablation...\"
    scripts/run_j_ablation.sh

    # ablation script is itself detached, so we need to wait for ITS PID file to clear.
    sleep 30
    while [ -f logs/runj_ablation.pid ] && kill -0 \$(cat logs/runj_ablation.pid) 2>/dev/null; do
        sleep 60
    done

    # build_final_j and run_final_j_public are inside ablation script's tail,
    # which fires *before* the ablation pid clears. So once we get here,
    # final-J public is also running detached.

    echo \"[\$(date)] Chain dispatched. Final-J public now running detached.\"
    echo \"      Monitor: tail -f logs/runj_final_public.log\"
" < /dev/null > "$LOG" 2>&1 &

CHAIN_PID=$!

cat <<EOF
Chain watcher started (waiting for Run I chain to finish, then runs J ablation + final-J public).
  PID: $CHAIN_PID  (detached)
  Log: $LOG

Sequence:
  1. Wait for Run I chain (PID $RUNI_PID) to exit
  2. Run scripts/run_j_ablation.sh           (~1.5h)
  3. (auto) Run scripts/build_final_j.py     (instant)
  4. (auto) Run scripts/run_final_j_public.sh (~80-90min)

Monitors:
  Chain         : tail -f $LOG
  Run I chain   : tail -f logs/runi_chain.log
  Ablation      : tail -f logs/runj_ablation.log    (will appear after #1)
  Final-J public: tail -f logs/runj_final_public.log (will appear after #3)
EOF
