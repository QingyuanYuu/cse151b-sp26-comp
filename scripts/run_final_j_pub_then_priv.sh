#!/usr/bin/env bash
#
# After data/runj_final_branches.txt is in place, fire final-J on
# public 1126 then automatically follow with private 943. They share
# the GPU sequentially (vLLM engine reload between runs). Total ~3h.
#
# Usage:
#     scripts/run_final_j_pub_then_priv.sh

set -euo pipefail

cd "$(dirname "$0")/.."

LOG="logs/runj_final_pub_then_priv.log"

if [ ! -f data/runj_final_branches.txt ]; then
    echo "ERROR: data/runj_final_branches.txt missing."
    echo "       Run build_final_j.py first (--branches <list> override or auto)."
    exit 1
fi

setsid bash -c "
    echo \"[\$(date)] Starting final-J on PUBLIC 1126...\"
    scripts/run_final_j_public.sh

    # run_final_j_public.sh launches detached + writes its own pid file.
    # Wait for the public run to clear before launching private.
    sleep 30
    while [ -f logs/runj_final_public.pid ] && kill -0 \$(cat logs/runj_final_public.pid) 2>/dev/null; do
        sleep 60
    done
    echo \"[\$(date)] Public done; waiting 10s for vLLM cleanup...\"
    sleep 10

    echo \"[\$(date)] Starting final-J on PRIVATE 943...\"
    scripts/run_final_j_private.sh
    sleep 30
    while [ -f logs/runj_final_private.pid ] && kill -0 \$(cat logs/runj_final_private.pid) 2>/dev/null; do
        sleep 60
    done

    echo \"[\$(date)] Both runs complete.\"
    echo \"  Public  → results/runj_final_public.jsonl\"
    echo \"  Private → results/runj_final_private.jsonl + submissions/runj_final_private.csv\"
" < /dev/null > "$LOG" 2>&1 &

PID=$!

cat <<EOF
Sequential public → private chain started.
  PID: $PID  (detached watcher)
  Log: $LOG

Sequence:
  1. final-J on public 1126 (~80-90min)  → results/runj_final_public.jsonl
  2. final-J on private 943 (~70-80min)  → submissions/runj_final_private.csv

Composition:
$(sed 's/^/  /' data/runj_final_branches.txt)

Monitor public : tail -f logs/runj_final_public.log
Monitor private: tail -f logs/runj_final_private.log
Monitor chain  : tail -f $LOG
EOF
