#!/bin/bash
# Chain runner: val → gate check → private. Survives parent shell death.
#
# Stages:
#   1. Run val on 225 stratified public questions (~30-40 min).
#   2. Parse val accuracy from log.
#   3. If val accuracy ≥ 55.4 % (Phase 0 baseline 56.44 % minus 1 pp tolerance),
#      proceed to full private run (~3-3.5 hours).
#   4. Otherwise: stop, write a clear failure summary, do NOT touch private.
#
# All stdout/stderr goes to logs/runD_chain.log. Each stage also writes its
# own log: logs/runD_val.log and logs/runD_private.log.
#
# Launch with:
#   setsid nohup scripts/run_d_chain.sh > logs/runD_chain.log 2>&1 &
#
# This detaches from the calling session; killing the parent shell (or losing
# the Claude Code process) does not affect the chain.

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs reports

VAL_LOG="logs/runD_val.log"
PRIV_LOG="logs/runD_private.log"
CHAIN_STATUS="logs/runD_chain_status.txt"

# Phase 0 val baseline 56.44 %; tolerate 1pp noise band.
GATE_PCT="63.0"

echo "[$(date)] === Run D chain started, pid=$$ ==="
echo "[$(date)] PWD=$(pwd)"
echo "[$(date)] Val log : $VAL_LOG"
echo "[$(date)] Priv log: $PRIV_LOG"
echo "[$(date)] Gate    : ≥${GATE_PCT}%"
echo

# ─── Stage 1: val ─────────────────────────────────────────────────────────
echo "[$(date)] Stage 1/2: launching val (225 questions, ~30-40 min)..."
scripts/run_d.sh --val > "$VAL_LOG" 2>&1
VAL_EXIT=$?
echo "[$(date)] Val exit code: $VAL_EXIT"

if [ "$VAL_EXIT" -ne 0 ]; then
    msg="VAL_FAILED exit=$VAL_EXIT — see $VAL_LOG"
    echo "[$(date)] $msg"
    echo "$msg" > "$CHAIN_STATUS"
    exit 10
fi

# ─── Stage 1.5: gate check ────────────────────────────────────────────────
echo "[$(date)] Stage 1.5: parsing val accuracy and checking gate..."

VAL_ACC=$(.venv/bin/python - "$VAL_LOG" <<'PY'
"""Find the val accuracy line in the run_d.sh log and print the percentage."""
import re, sys
log = open(sys.argv[1]).read()
# Looks like: "  val accuracy: 127/225 = 56.44%"
m = re.search(r"val accuracy:\s*\d+\s*/\s*\d+\s*=\s*([\d.]+)\s*%", log)
print(m.group(1) if m else "")
PY
)

if [ -z "$VAL_ACC" ]; then
    msg="GATE_PARSE_FAILED (no val accuracy in $VAL_LOG)"
    echo "[$(date)] $msg"
    echo "$msg" > "$CHAIN_STATUS"
    exit 11
fi

echo "[$(date)] Val accuracy parsed: ${VAL_ACC}%"

PASS=$(.venv/bin/python -c "import sys; print(1 if float('$VAL_ACC') >= float('$GATE_PCT') else 0)")
if [ "$PASS" != "1" ]; then
    msg="GATE_FAILED ${VAL_ACC}% < ${GATE_PCT}%; private run skipped"
    echo "[$(date)] $msg"
    echo "$msg" > "$CHAIN_STATUS"
    {
        echo "# Run D chain — gate failed"
        echo
        echo "Val accuracy: **${VAL_ACC}%** (gate: ≥${GATE_PCT}%)"
        echo
        echo "Phase 0 val baseline: 56.44%"
        echo
        echo "Run D prompt change regressed val. Private run **was not** triggered."
        echo "Inspect \`$VAL_LOG\` to diagnose, then either:"
        echo "- revert the prompt change, or"
        echo "- relax the symbolic rule further, or"
        echo "- fall back to Run A (Phase 0 prompt + K=8 SC)."
    } > reports/runD_chain_summary.md
    exit 12
fi

echo "[$(date)] Gate passed (${VAL_ACC}% ≥ ${GATE_PCT}%)."

# ─── Stage 2: private ─────────────────────────────────────────────────────
echo "[$(date)] Stage 2/2: launching private run (943 questions, ~3-3.5 hours)..."
scripts/run_d.sh > "$PRIV_LOG" 2>&1
PRIV_EXIT=$?
echo "[$(date)] Private exit code: $PRIV_EXIT"

if [ "$PRIV_EXIT" -ne 0 ]; then
    msg="PRIVATE_FAILED exit=$PRIV_EXIT — see $PRIV_LOG"
    echo "[$(date)] $msg"
    echo "$msg" > "$CHAIN_STATUS"
    exit 20
fi

# ─── Done ─────────────────────────────────────────────────────────────────
msg="DONE val=${VAL_ACC}% csv=results/submission_runD.csv"
echo "[$(date)] === Run D chain complete: $msg ==="
echo "$msg" > "$CHAIN_STATUS"

{
    echo "# Run D chain — complete"
    echo
    echo "- Val accuracy: **${VAL_ACC}%** (gate: ≥${GATE_PCT}%)"
    echo "- Private CSV : \`results/submission_runD.csv\`"
    echo "- Val log     : \`$VAL_LOG\`"
    echo "- Private log : \`$PRIV_LOG\`"
    echo
    echo "Submit \`results/submission_runD.csv\` to Kaggle."
} > reports/runD_chain_summary.md
