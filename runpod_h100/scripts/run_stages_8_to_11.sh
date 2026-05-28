#!/usr/bin/env bash
# Resume pipeline from Stage 8 (after Stages 6 + 7 + 7.5 finished).
# Private K=4 SC (downgrade from K=8 for speed), then Kaggle CSV + handoff + git push.
set -euo pipefail

REPO=/workspace/cse151b-grpo/cse151b-sp26-comp/runpod_h100
PY=/workspace/cse151b-grpo/.venv/bin/python
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

cd "$REPO"

# Stale lock from killed run
rm -f "$REPO/results/private_sc_k8.jsonl"

echo "==[ $(date '+%F %T') ]== Stage 8: private K=4 SC inference (Run F + v2 budget)"
bash scripts/run_private_sc.sh \
    "$REPO/checkpoints/grpo_v6_merged" \
    "$REPO/results/private_sc_k4.jsonl" \
    2>&1 | tee "$LOG_DIR/private_sc.log"

echo "==[ $(date '+%F %T') ]== Stage 9: build Kaggle submission CSV"
PYTHONPATH=src $PY -m cse151b_comp.submission \
    --results "$REPO/results/private_sc_k4.jsonl" \
    --out "$REPO/results/private_submission.csv" \
    2>&1 | tee "$LOG_DIR/submission.log" || \
    echo "[pipeline] submission build failed; JSONL still saved."

echo "==[ $(date '+%F %T') ]== Stage 10: build handoff/"
$PY scripts/build_handoff.py 2>&1 | tee "$LOG_DIR/build_handoff.log" || \
    echo "[pipeline] handoff build failed; continuing."

echo "==[ $(date '+%F %T') ]== Stage 11: git push handoff/ + STATUS.md"
bash scripts/git_push_handoff.sh 2>&1 | tee "$LOG_DIR/git_push.log" || \
    echo "[pipeline] git push failed; handoff/ still on disk."

echo "==[ $(date '+%F %T') ]== Stages 8-11 complete."
