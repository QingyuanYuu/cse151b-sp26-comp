#!/usr/bin/env bash
# Public set K=1 inference (full 1126 questions, Run F prompt + v2 budget).
# Faster than K=8 — used for full-public scoring before private K=8 SC.
set -euo pipefail

REPO=/workspace/cse151b-grpo/cse151b-sp26-comp/runpod_h100
PY=/workspace/cse151b-grpo/.venv/bin/python
MODEL="${1:-$REPO/checkpoints/grpo_v6_merged}"
OUT="${2:-$REPO/results/public_k1.jsonl}"

mkdir -p "$REPO/results"
cd "$REPO"

echo "[public-k1] model:  $MODEL"
echo "[public-k1] data:   $REPO/data/public.jsonl  (1126 questions, gold-labelled)"
echo "[public-k1] K=1 single-shot, Run F prompt, v2 budget"
echo "[public-k1] output: $OUT"

PYTHONPATH=src $PY -m cse151b_comp.self_consistency \
    --input "$REPO/data/public.jsonl" \
    --output "$OUT" \
    --model "$MODEL" \
    --k 1 \
    --bf16 \
    --prompt runf \
    --allocate-tokens \
    --temperature 0.6 \
    --top-p 0.95 \
    --max-model-len 32768 \
    --max-num-seqs 32 \
    --gpu-mem-util 0.85 \
    --chunk-size 200 \
    --resume

echo "[public-k1] done. Quick accuracy breakdown:"
PYTHONPATH=src $PY -c "
import json
val_ids = set(json.load(open('$REPO/data/val_indices.json'))['val_ids'])
rows = [json.loads(l) for l in open('$OUT')]
n = len(rows)
ncorrect = sum(1 for r in rows if r.get('correct'))
val_rows = [r for r in rows if r['id'] in val_ids]
train_rows = [r for r in rows if r['id'] not in val_ids]
def acc(rs): return (sum(1 for r in rs if r.get('correct')), len(rs))
oa, on = acc(rows); va, vn = acc(val_rows); ta, tn = acc(train_rows)
print(f'  overall   {oa}/{on} = {100*oa/on:.2f}%')
print(f'  train(901) {ta}/{tn} = {100*ta/tn:.2f}% (model saw these in SFT/GRPO)')
print(f'  val(225)   {va}/{vn} = {100*va/vn:.2f}% (true holdout — compare to SFT 64.44%)')
"
