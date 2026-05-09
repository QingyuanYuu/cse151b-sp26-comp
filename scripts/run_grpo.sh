#!/usr/bin/env bash
#
# Run GRPO training on top of LoRA v1 (or v2 if available).
# Reward = Judger.auto_judge binary correct/wrong.
#
# Wallclock estimate: ~6-10h on 600 prompts × K=4 × 1 epoch.
#
# Usage:
#     scripts/run_grpo.sh
#     scripts/run_grpo.sh --base lora_weights/runj_distill_v2_merged
#     scripts/run_grpo.sh --max-prompts 100  # smoke test (~1h)

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

# Default base = LoRA v1 merged
BASE="${BASE:-lora_weights/runj_distill_v1_merged}"
OUTPUT="${OUTPUT:-lora_weights/runj_grpo_v1}"
LOG="logs/grpo_train.log"
PID_FILE="logs/grpo_train.pid"

# Override base/output via flags
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --base) BASE="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [ ! -d "$BASE" ]; then
    echo "ERROR: base model not found: $BASE"
    echo "       Expected LoRA-merged dir."
    exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running (PID $(cat "$PID_FILE"))."
    exit 1
fi
rm -f "$PID_FILE"

mkdir -p logs "$OUTPUT"

setsid bash -c "
    cd '$(pwd)'
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PYTHONPATH=src .venv/bin/python scripts/train_grpo.py \
        --base '$BASE' \
        --output '$OUTPUT' \
        --max-prompts 900 \
        --epochs 4 \
        --num-generations 8 \
        --lr 3e-6 \
        --beta 0.04 \
        --r 16 \
        --alpha 32 \
        --batch-size 1 \
        --grad-accum 4 \
        --max-completion-length 6144 \
        --use-vllm \
        $EXTRA_ARGS

    rc=\$?
    rm -f '$PID_FILE'
    exit \$rc
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
GRPO training started.
  PID:    $PID
  Base:   $BASE
  Output: $OUTPUT
  Log:    $LOG

Config:
  900 prompts × K=8 generations × 4 epochs (= 28800 sample-passes)
  LoRA r=16 alpha=32
  lr=3e-6 (conservative for stability)
  beta=0.04 (KL penalty)
  max_completion=6144 tokens
  use_vllm=True (colocate, fast sampling)

Wallclock estimate: ~14-16h on Blackwell with vLLM colocate.
Started: $(date '+%Y-%m-%d %H:%M %Z')

Monitor:
  tail -f $LOG
  watch -n 30 'grep -E "loss|reward|step|epoch" $LOG | tail -10'

For smoke test (~1h, 100 prompts):
  scripts/run_grpo.sh --max-prompts 100
EOF
