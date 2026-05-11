#!/usr/bin/env bash
# GRPO v2 — applies all fixes from v1 post-mortem:
#   * Edge-filtered prompt pool (196 prompts, base K=4 1-3/4 + sample of 0/4)
#   * lr=1e-5 constant (was 3e-6 cosine→0)
#   * temp=1.0 (was 0.7) → variance in K=8 groups
#   * dr_grpo loss + scale_rewards=none (length-normalized)
#   * sequence-level importance sampling (stable under vLLM colocate)
#   * epsilon=0.3, epsilon_high=0.4 (asymmetric DAPO clip)
#   * beta=0.0 (no KL drag)
#   * length-aware reward: correct + <2000 chars → 0.5 (anti-collapse)
# Wallclock target: ~7-9h on Blackwell.

set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

BASE="${BASE:-lora_weights/runj_distill_v1_merged}"
OUTPUT="${OUTPUT:-lora_weights/runj_grpo_v2}"
POOL="${POOL:-data/grpo_pool_v2.json}"
LOG="logs/grpo_v2_train.log"
PID_FILE="logs/grpo_v2_train.pid"

EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --base) BASE="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --pool) POOL="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

if [ ! -d "$BASE" ]; then echo "ERROR: base not found: $BASE"; exit 1; fi
if [ ! -f "$POOL" ]; then echo "ERROR: pool not found: $POOL"; exit 1; fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running (PID $(cat "$PID_FILE"))."; exit 1
fi
rm -f "$PID_FILE"
mkdir -p logs "$OUTPUT"

setsid bash -c "
    cd '$(pwd)'
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    PYTHONPATH=src .venv/bin/python scripts/train_grpo.py \
        --base '$BASE' \
        --output '$OUTPUT' \
        --pool '$POOL' \
        --max-prompts 200 \
        --epochs 3 \
        --num-generations 8 \
        --lr 1e-5 \
        --lr-scheduler constant \
        --warmup-ratio 0.0 \
        --beta 0.0 \
        --temperature 1.0 \
        --top-p 0.95 \
        --loss-type dr_grpo \
        --importance-sampling-level token \
        --epsilon 0.3 \
        --epsilon-high 0.4 \
        --r 16 --alpha 32 \
        --batch-size 1 --grad-accum 8 \
        --max-completion-length 6144 \
        --use-vllm \
        --disable-vllm-is-correction \
        $EXTRA_ARGS

    rc=\$?
    if [ \$rc -eq 0 ]; then
        echo
        echo \"[\$(date)] GRPO v2 done. Merging...\"
        PYTHONPATH=src .venv/bin/python scripts/merge_lora.py \
            --base '$BASE' \
            --adapter '$OUTPUT' \
            --output '${OUTPUT}_merged'
        echo \"[\$(date)] Merge complete.\"
    fi
    rm -f '$PID_FILE'
    exit \$rc
" < /dev/null > "$LOG" 2>&1 &

echo "$!" > "$PID_FILE"
echo "GRPO v2 launched. PID $(cat $PID_FILE)"
echo "Log:    $LOG"
echo "Output: $OUTPUT (then ${OUTPUT}_merged)"
echo "ETA:    ~7-9h (196 × K=8 × 3 epochs ≈ 588 steps × ~50s)"
echo "Started: $(date '+%Y-%m-%d %H:%M %Z')"
