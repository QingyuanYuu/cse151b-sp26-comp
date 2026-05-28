#!/usr/bin/env bash
# Private K=8 self-consistency inference with the final merged GRPO model.
# Uses Run F prompt + v2 budget. Output: results/private_sc_k8.jsonl
set -euo pipefail

REPO=/workspace/cse151b-grpo/cse151b-sp26-comp/runpod_h100
PY=/workspace/cse151b-grpo/.venv/bin/python
MODEL="${1:-$REPO/checkpoints/grpo_v6_merged}"
OUT="${2:-$REPO/results/private_sc_k8.jsonl}"

mkdir -p "$REPO/results"
cd "$REPO"

echo "[private-sc] model:  $MODEL"
echo "[private-sc] data:   $REPO/data/private.jsonl  (943 questions)"
echo "[private-sc] K=8 self-consistency, Run F prompt, v2 budget"
echo "[private-sc] output: $OUT"

PYTHONPATH=src $PY -m cse151b_comp.self_consistency \
    --input "$REPO/data/private.jsonl" \
    --output "$OUT" \
    --model "$MODEL" \
    --k 4 \
    --bf16 \
    --prompt runf \
    --allocate-tokens \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-model-len 32768 \
    --max-tokens-floor 16000 \
    --max-tokens-ceiling 30000 \
    --max-num-seqs 16 \
    --gpu-mem-util 0.85 \
    --chunk-size 50 \
    --resume

echo "[private-sc] done. Convert to Kaggle CSV with:"
echo "  $PY -m cse151b_comp.submission --input $OUT --output $REPO/results/private_submission.csv"
