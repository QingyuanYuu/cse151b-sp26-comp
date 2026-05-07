#!/usr/bin/env bash
#
# After the Run I chain finishes, automatically:
# 1. Run Run J ablation harness (~1.5h, 18 inferences)
# 2. Auto-summarize + write deep per-branch review to reports/runj_ablation_review.md
# 3. STOP — wait for human/Claude review of the ablation results
#
# Final-J composition + public run are NOT auto-fired. After the chain
# stops, do (manually):
#     # Read the review:
#     less reports/runj_ablation_review.md
#     # Decide KEEP/DROP per branch, then build final-J:
#     PYTHONPATH=src .venv/bin/python scripts/build_final_j.py \
#         --branches olympiad,trig,geometry,...
#     # Fire public 1126:
#     scripts/run_final_j_public.sh
#
# This split exists so we can apply judgment beyond a Δ threshold —
# response length blowup, multi-box rate regression, topic routing
# mistakes, etc. — before committing the final composition.
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

    echo \"[\$(date)] Ablation chain done. Pipeline PAUSED for review.\"
    echo \"      Read review:  less reports/runj_ablation_review.md\"
    echo \"      Then decide branches and run:\"
    echo \"        PYTHONPATH=src .venv/bin/python scripts/build_final_j.py \\\\\"
    echo \"            --branches <comma-list>\"
    echo \"        scripts/run_final_j_public.sh\"
" < /dev/null > "$LOG" 2>&1 &

CHAIN_PID=$!

cat <<EOF
Chain watcher started (waiting for Run I chain to finish, then runs J ablation + final-J public).
  PID: $CHAIN_PID  (detached)
  Log: $LOG

Sequence:
  1. Wait for Run I chain (PID $RUNI_PID) to exit
  2. Run scripts/run_j_ablation.sh                    (~1.5h, 18 inferences)
  3. Auto-summarize + write reports/runj_ablation_review.md
  4. STOP — manual review of ablation results.

After the pause, you (or Claude) decide branch composition:
  PYTHONPATH=src .venv/bin/python scripts/build_final_j.py --branches <list>
  scripts/run_final_j_public.sh

Monitors:
  Chain         : tail -f $LOG
  Run I chain   : tail -f logs/runi_chain.log
  Ablation      : tail -f logs/runj_ablation.log    (will appear after #1)
EOF
