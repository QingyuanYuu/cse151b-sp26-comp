#!/usr/bin/env bash
#
# Wait for the Run J private launcher (PID in logs/runj_final_private.pid)
# to finish, then fire scripts/run_runi_private.sh — chains a second
# private inference for a backup submission.
#
# Usage:
#     scripts/queue_runi_after_runj.sh

set -euo pipefail

cd "$(dirname "$0")/.."

LOG="logs/queue_runi_after_runj.log"

setsid bash -c '
    while [ -f logs/runj_final_private.pid ] && kill -0 $(cat logs/runj_final_private.pid 2>/dev/null) 2>/dev/null; do
        sleep 60
    done
    echo "[$(date)] Run J private done. Sleeping 15s for vLLM cleanup..."
    sleep 15
    echo "[$(date)] Firing Run I private..."
    scripts/run_runi_private.sh
    echo "[$(date)] Queue done — Run I private now running detached."
' < /dev/null > "$LOG" 2>&1 &

PID=$!

cat <<EOF
Watcher queued. After Run J private finishes:
  → scripts/run_runi_private.sh fires automatically
  → submissions/runi_k1_private.csv generated

  Watcher PID: $PID
  Log:         $LOG

Monitor: tail -f $LOG
EOF
