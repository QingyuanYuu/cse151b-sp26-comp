#!/usr/bin/env bash
# GRPO + ckpt sweep only (val_225 + SFT already done).
set -euo pipefail

# Reduce memory fragmentation — system suggested after step 3 OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/workspace/cse151b-grpo/cse151b-sp26-comp/runpod_h100
PY=/workspace/cse151b-grpo/.venv/bin/python
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR" "$REPO/results"

cd "$REPO"

# Pass --resume as $1 to continue from latest checkpoint (after OOM/crash).
# Default (no arg): wipe and start fresh.
RESUME_FLAG=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume"
    echo "==[ $(date '+%F %T') ]== RESUME mode — keeping existing checkpoints"
elif [[ -d "$REPO/checkpoints/grpo_v6" ]]; then
    echo "==[ $(date '+%F %T') ]== removing partial GRPO output (pass --resume to keep)"
    rm -rf "$REPO/checkpoints/grpo_v6"
fi

echo "==[ $(date '+%F %T') ]== Stage 4: GRPO training (3 epoch, K=4 + hard-dup=1, vllm_util=0.35) $RESUME_FLAG"
$PY scripts/train_grpo.py \
    --base checkpoints/lora_sft_merged \
    --output checkpoints/grpo_v6 \
    $RESUME_FLAG \
    2>&1 | tee -a "$LOG_DIR/grpo_full.log"

echo "==[ $(date '+%F %T') ]== Stage 5: best-ckpt-by-val sweep"
$PY scripts/eval_all_ckpts.py \
    --base checkpoints/lora_sft_merged \
    --grpo-dir checkpoints/grpo_v6 \
    --out results/grpo_ckpt_sweep.jsonl \
    --summary results/grpo_ckpt_sweep_summary.json \
    2>&1 | tee "$LOG_DIR/eval_all_ckpts.log"

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

echo "==[ $(date '+%F %T') ]== Stage 10: build handoff/ (small artifacts for git)"
$PY scripts/build_handoff.py 2>&1 | tee "$LOG_DIR/build_handoff.log" || \
    echo "[pipeline] handoff build failed; continuing."

echo "==[ $(date '+%F %T') ]== Stage 11: git push handoff/ + STATUS.md"
bash scripts/git_push_handoff.sh 2>&1 | tee "$LOG_DIR/git_push.log" || \
    echo "[pipeline] git push failed; handoff/ still on disk."

echo "==[ $(date '+%F %T') ]== Full pipeline complete."
echo "  - GRPO adapter:        $REPO/checkpoints/grpo_v6/final/"
echo "  - Ckpt sweep:          $REPO/results/grpo_ckpt_sweep_summary.json"
echo "  - Final merged model:  $REPO/checkpoints/grpo_v6_merged/"
echo "  - HF Hub:              https://huggingface.co/JaasonYuu/jason-cse151b-model"
echo "  - Private K=8 SC:      $REPO/results/private_sc_k8.jsonl"
echo "  - Kaggle CSV:          $REPO/results/private_submission.csv"
echo "  - TensorBoard:         tensorboard --logdir $REPO/checkpoints/grpo_v6/logs"
