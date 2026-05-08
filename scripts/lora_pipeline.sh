#!/usr/bin/env bash
#
# LoRA pipeline: K=32 SC pool → SFT data → train LoRA → save adapter
#
# Triggered after K=32 SC public completes.
# Total wallclock: ~12-18h (mostly LoRA training).
#
# Usage: scripts/lora_pipeline.sh

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

POOL="results/sc_runf_k32_public.jsonl"
SFT_OUT="data/sft_train_v1.jsonl"
LORA_OUT="lora_weights/runj_distill_v1"
LOG="logs/lora_pipeline.log"

if [ ! -f "$POOL" ]; then
    echo "ERROR: $POOL not found — did K=32 SC public finish?"
    exit 1
fi

# Step 1: Extract SFT pairs (excludes val_ids)
echo "=== Step 1: Prepare SFT data ==="
uv run --no-sync cse151b-prepare-sft \
    --pool "$POOL" \
    --source data/public.jsonl \
    --val data/val_indices.json \
    --output "$SFT_OUT"

echo
n=$(wc -l < "$SFT_OUT")
echo "SFT data: $n rows → $SFT_OUT"

# Step 2: Train LoRA (detached, log to file)
echo
echo "=== Step 2: LoRA training (detached) ==="
mkdir -p "$LORA_OUT"
setsid bash -c "
    PYTHONPATH=src python scripts/train_lora.py \
        --train '$SFT_OUT' \
        --output '$LORA_OUT' \
        --r 32 \
        --alpha 64 \
        --epochs 3 \
        --lr 2e-4 \
        --batch-size 4 \
        --grad-accum 4
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "LoRA training started (PID $PID)."
echo "  Log:    $LOG"
echo "  Output: $LORA_OUT"
echo
echo "Monitor: tail -f $LOG"
