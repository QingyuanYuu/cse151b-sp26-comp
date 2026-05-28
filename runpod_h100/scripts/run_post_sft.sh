#!/usr/bin/env bash
# Stage 3 + 4: val_225 eval (Run F prompt) → GRPO training.
# Run this AFTER run_full_pipeline.sh finishes (SFT + Merge).
set -euo pipefail

REPO=/workspace/cse151b-grpo/cse151b-sp26-comp/runpod_h100
PY=/workspace/cse151b-grpo/.venv/bin/python
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR" "$REPO/results"

cd "$REPO"

if [[ ! -f "$REPO/checkpoints/lora_sft_merged/model.safetensors" ]]; then
    echo "ERROR: merged model not found at $REPO/checkpoints/lora_sft_merged/" >&2
    echo "Run scripts/run_full_pipeline.sh first." >&2
    exit 1
fi

echo "==[ $(date '+%F %T') ]== Stage 3: val_225 eval (Run F prompt + v2 budget)"
$PY scripts/eval_val225.py \
    --model checkpoints/lora_sft_merged \
    --out results/val225_sft.jsonl \
    2>&1 | tee "$LOG_DIR/eval_val225.log"

echo "==[ $(date '+%F %T') ]== Stage 4: GRPO training (3 epoch, K=4 + hard-dup=1)"
$PY scripts/train_grpo.py \
    --base checkpoints/lora_sft_merged \
    --output checkpoints/grpo_v6 \
    2>&1 | tee "$LOG_DIR/grpo_full.log"

echo "==[ $(date '+%F %T') ]== Stage 5: best-ckpt-by-val sweep (all GRPO checkpoints)"
$PY scripts/eval_all_ckpts.py \
    --base checkpoints/lora_sft_merged \
    --grpo-dir checkpoints/grpo_v6 \
    --out results/grpo_ckpt_sweep.jsonl \
    --summary results/grpo_ckpt_sweep_summary.json \
    2>&1 | tee "$LOG_DIR/eval_all_ckpts.log"

echo "==[ $(date '+%F %T') ]== Post-SFT pipeline complete."
echo "  - val_225 (SFT):   $REPO/results/val225_sft.jsonl"
echo "  - GRPO adapter:    $REPO/checkpoints/grpo_v6/final/"
echo "  - Ckpt sweep:      $REPO/results/grpo_ckpt_sweep_summary.json"
echo "  - TensorBoard:     tensorboard --logdir $REPO/checkpoints/grpo_v6/logs"
