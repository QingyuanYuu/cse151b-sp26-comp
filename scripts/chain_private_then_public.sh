#!/usr/bin/env bash
#
# Wait for the running K=1 Run F private process to finish, then
# automatically launch K=1 Run F public. Detached — survives shell exit.
#
# Usage:
#     scripts/chain_private_then_public.sh

set -euo pipefail

cd "$(dirname "$0")/.."

PRIVATE_PID_FILE="logs/runf_k1_private.pid"
LOG="logs/chain_private_then_public.log"

if [ ! -f "$PRIVATE_PID_FILE" ]; then
    echo "ERROR: $PRIVATE_PID_FILE not found — is the private run actually running?"
    exit 1
fi

PRIVATE_PID=$(cat "$PRIVATE_PID_FILE")

setsid bash -c "
    echo \"[\$(date)] Waiting for private PID $PRIVATE_PID to finish...\"
    while kill -0 $PRIVATE_PID 2>/dev/null; do
        sleep 30
    done
    echo \"[\$(date)] Private done. Launching public...\"
    sleep 5  # give vLLM EngineCore process time to fully release GPU
    scripts/run_runf_k1_public.sh
    echo \"[\$(date)] Public launched.\"
" < /dev/null > "$LOG" 2>&1 &

CHAIN_PID=$!

cat <<EOF
Chain watcher started.
  PID: $CHAIN_PID  (PPID 1, detached)
  Log: $LOG
  Will fire scripts/run_runf_k1_public.sh when PID $PRIVATE_PID exits.

Monitor private:  tail -f logs/runf_k1_private.log
Monitor chain:    tail -f $LOG
Monitor public (after private done):  tail -f logs/runf_k1_public.log
EOF
