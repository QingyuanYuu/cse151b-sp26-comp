#!/usr/bin/env bash
#
# Sequential chain:
#   1. K=8 SC private (Run F prompt) → submissions/sc_runf_k8_private.csv
#   2. K=32 SC public (Run F prompt) → results/sc_runf_k32_public.jsonl
#
# Usage: scripts/chain_k8_priv_then_k32_pub.sh

set -euo pipefail

cd "$(dirname "$0")/.."

LOG="logs/chain_k8priv_k32pub.log"

setsid bash -c '
    echo "[$(date)] Starting K=8 SC private (Run F)..."
    scripts/run_sc_runf_k8_private.sh

    sleep 30
    while [ -f logs/sc_runf_k8_private.pid ] && kill -0 $(cat logs/sc_runf_k8_private.pid 2>/dev/null) 2>/dev/null; do
        sleep 60
    done
    echo "[$(date)] K=8 private done; sleeping 15s for vLLM cleanup..."
    sleep 15

    echo "[$(date)] Starting K=32 SC public (Run F)..."
    scripts/run_sc_runf_k32_public.sh

    sleep 30
    while [ -f logs/sc_runf_k32_public.pid ] && kill -0 $(cat logs/sc_runf_k32_public.pid 2>/dev/null) 2>/dev/null; do
        sleep 120
    done
    echo "[$(date)] K=32 public done; chain complete."
' < /dev/null > "$LOG" 2>&1 &

PID=$!

cat <<EOF
Sequential chain started.
  Watcher PID: $PID
  Log:         $LOG

Sequence:
  1. K=8 SC private  (~9h)   → submissions/sc_runf_k8_private.csv
  2. K=32 SC public  (~33-38h) → results/sc_runf_k32_public.jsonl

Total wallclock: ~42-47h ≈ ~2 days

Monitors:
  Chain          : tail -f $LOG
  K=8 private    : tail -f logs/sc_runf_k8_private.log
  K=32 public    : tail -f logs/sc_runf_k32_public.log  (after K=8 done)
EOF
