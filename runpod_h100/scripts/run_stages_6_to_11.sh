#!/usr/bin/env bash
# Resume pipeline from Stage 6 (after Stage 5 sweep finished).
# Best ckpt is step-606 @ 66.22% (per results/grpo_ckpt_sweep_summary.json).
set -euo pipefail

REPO=/workspace/cse151b-grpo/cse151b-sp26-comp/runpod_h100
PY=/workspace/cse151b-grpo/.venv/bin/python
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

cd "$REPO"

echo "==[ $(date '+%F %T') ]== Stage 6: merge best GRPO adapter → BF16"
$PY scripts/merge_best_grpo.py 2>&1 | tee "$LOG_DIR/merge_best.log"

echo "==[ $(date '+%F %T') ]== Stage 7: upload to HuggingFace Hub"
bash scripts/upload_to_hf.sh "$REPO/checkpoints/grpo_v6_merged" 2>&1 | tee "$LOG_DIR/hf_upload.log" || \
    echo "[pipeline] HF upload failed; continuing to inference."

echo "==[ $(date '+%F %T') ]== Stage 7.5: public K=1 inference (1126 questions, scored)"
bash scripts/run_public_k1.sh \
    "$REPO/checkpoints/grpo_v6_merged" \
    "$REPO/results/public_k1.jsonl" \
    2>&1 | tee "$LOG_DIR/public_k1.log" || \
    echo "[pipeline] public K=1 failed; continuing to private."

echo "==[ $(date '+%F %T') ]== Stage 8: private K=8 SC inference (Run F + v2 budget)"
bash scripts/run_private_sc.sh \
    "$REPO/checkpoints/grpo_v6_merged" \
    "$REPO/results/private_sc_k8.jsonl" \
    2>&1 | tee "$LOG_DIR/private_sc.log"

echo "==[ $(date '+%F %T') ]== Stage 9: build Kaggle submission CSV"
PYTHONPATH=src $PY -m cse151b_comp.submission \
    --input "$REPO/results/private_sc_k8.jsonl" \
    --output "$REPO/results/private_submission.csv" \
    2>&1 | tee "$LOG_DIR/submission.log" || \
    echo "[pipeline] submission build failed; private JSONL still saved."

echo "==[ $(date '+%F %T') ]== Stage 10: build handoff/"
$PY scripts/build_handoff.py 2>&1 | tee "$LOG_DIR/build_handoff.log" || \
    echo "[pipeline] handoff build failed; continuing."

echo "==[ $(date '+%F %T') ]== Stage 11: git push handoff/ + STATUS.md"
bash scripts/git_push_handoff.sh 2>&1 | tee "$LOG_DIR/git_push.log" || \
    echo "[pipeline] git push failed; handoff/ still on disk."

echo "==[ $(date '+%F %T') ]== Stages 6-11 complete."
