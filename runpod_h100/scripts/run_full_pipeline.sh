#!/usr/bin/env bash
# Full SFT + Merge pipeline for H100 80GB.
# GRPO step is intentionally separate — kick off after reviewing SFT loss curve.
set -euo pipefail

REPO=/workspace/cse151b-grpo/cse151b-sp26-comp/runpod_h100
PY=/workspace/cse151b-grpo/.venv/bin/python
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

cd "$REPO"

echo "==[ $(date '+%F %T') ]== Stage 1: BF16 LoRA SFT (5 epochs)"
$PY scripts/train_lora_bf16.py \
    --max-seq 16384 \
    --per-device-bsz 1 \
    --grad-accum 8 \
    --grad-checkpoint \
    --epochs 5 \
    --logging-steps 5 \
    --save-steps 100 \
    2>&1 | tee "$LOG_DIR/sft_full.log"

echo "==[ $(date '+%F %T') ]== Stage 2: Merge LoRA into BF16 base"
$PY scripts/merge_lora.py 2>&1 | tee "$LOG_DIR/merge_full.log"

echo "==[ $(date '+%F %T') ]== Pipeline complete. Run GRPO manually after review:"
echo "   python scripts/train_grpo.py --base checkpoints/lora_sft_merged --output checkpoints/grpo_v6"
