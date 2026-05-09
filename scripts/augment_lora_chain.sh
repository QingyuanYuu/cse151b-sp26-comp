#!/usr/bin/env bash
#
# After v4 fix-format finishes on hopeless 254, automatically:
# 1. Re-build SFT pool with all 4 sources (K=4, K=4 hard1, K=8 hard2, v4 fix)
#    → newly-fixed questions get added to training set
# 2. Retrain LoRA on augmented pool
# 3. Merge new adapter
# 4. (Optional) re-evaluate on Run F wrong subset
#
# Usage: scripts/augment_lora_chain.sh

set -euo pipefail

cd "$(dirname "$0")/.."

LOG="logs/augment_lora.log"

setsid bash -c '
    set -e
    unset VIRTUAL_ENV CONDA_PREFIX
    export PATH="$HOME/.local/bin:$PATH"

    # ─── Wait for v4 fix-format to finish ────────────────────────────
    echo "[$(date)] Waiting for v4 fix-format on hopeless to finish..."
    while [ -f logs/runk_fix_hopeless.pid ] && ps -p $(cat logs/runk_fix_hopeless.pid 2>/dev/null) > /dev/null 2>&1; do
        sleep 60
    done
    # Also wait for any still-active SC process to finish
    while pgrep -f "EngineCore" > /dev/null 2>&1; do
        sleep 30
    done
    sleep 15  # vLLM cleanup

    if [ ! -f results/runk_fix_hopeless.jsonl ]; then
        echo "[$(date)] ERROR: v4 fix output not found"
        exit 1
    fi
    n_v4=$(wc -l < results/runk_fix_hopeless.jsonl)
    echo "[$(date)] v4 fix done. Output: $n_v4 questions."

    # ─── Step 1: Build augmented SFT pool (4 sources) ──────────────────
    echo "[$(date)] Building augmented SFT pool from 4 SC sources..."
    PYTHONPATH=src .venv/bin/python scripts/prepare_sft_staged.py \
        --pools \
            results/sc_k4_public.jsonl \
            results/sc_k4_extra1_public.jsonl \
            results/sc_k8_extra2_public.jsonl \
            results/runk_fix_hopeless.jsonl \
        --source data/public.jsonl \
        --val data/val_indices.json \
        --output data/sft_train_v2.jsonl

    n_sft=$(wc -l < data/sft_train_v2.jsonl)
    n_orig=$(wc -l < data/sft_train_staged.jsonl)
    n_added=$((n_sft - n_orig))
    echo "[$(date)] SFT v2 pool: $n_sft pairs ($n_added new from v4 fix)"

    # ─── Step 2: Retrain LoRA on augmented pool ─────────────────────────
    echo "[$(date)] LoRA v2 training (5 epochs, eval split, best ckpt)..."
    rm -rf lora_weights/runj_distill_v2/*  # fresh
    mkdir -p lora_weights/runj_distill_v2

    PYTHONPATH=src .venv/bin/python scripts/train_lora.py \
        --train data/sft_train_v2.jsonl \
        --output lora_weights/runj_distill_v2 \
        --epochs 5 \
        --eval-frac 0.1 \
        --r 32 \
        --alpha 64 \
        --batch-size 1 \
        --grad-accum 16 \
        --max-seq-len 6144

    # ─── Step 3: Merge ──────────────────────────────────────────────────
    echo "[$(date)] Merging LoRA v2 adapter..."
    PYTHONPATH=src .venv/bin/python scripts/merge_lora.py \
        --base Qwen/Qwen3-4B-Thinking-2507 \
        --adapter lora_weights/runj_distill_v2 \
        --output lora_weights/runj_distill_v2_merged

    # ─── Step 4: Quick eval on val_225 ──────────────────────────────────
    echo "[$(date)] Evaluating LoRA v2 on val_225..."
    uv run --no-sync cse151b-sc \
        --input data/public.jsonl \
        --output results/lora_v2_val.jsonl \
        --resume \
        --val data/val_indices.json \
        --chunk-size 200 \
        --k 1 \
        --temperature 0.6 --top-p 0.95 \
        --prompt runf \
        --model lora_weights/runj_distill_v2_merged \
        --bf16 \
        --gpu-mem-util 0.85 \
        --max-model-len 32768 \
        --max-num-seqs 32 \
        --allocate-tokens

    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py results/lora_v2_val.jsonl 2>&1 | tail -5

    # ─── Step 5: Compare LoRA v1 vs v2 vs Run F on val ──────────────────
    PYTHONPATH=src .venv/bin/python -c "
import json
vd = json.load(open('"'"'data/val_indices.json'"'"'))
val_ids = set(vd['"'"'val_ids'"'"'])

runf = {r['"'"'id'"'"']: r for r in (json.loads(l) for l in open('"'"'results/runf_k1_public.jsonl'"'"'))}
v1 = {r['"'"'id'"'"']: r for r in (json.loads(l) for l in open('"'"'results/lora_v1_public.jsonl'"'"'))}
v2 = {r['"'"'id'"'"']: r for r in (json.loads(l) for l in open('"'"'results/lora_v2_val.jsonl'"'"'))}

f_c = sum(1 for q in val_ids if runf.get(q,{}).get('"'"'correct'"'"'))
v1_c = sum(1 for q in val_ids if q in v1 and v1[q].get('"'"'correct'"'"'))
v2_c = sum(1 for q in val_ids if q in v2 and v2[q].get('"'"'correct'"'"'))

print(f'"'"'=== LoRA v2 vs v1 vs Run F on val_225 ==='"'"')
print(f'"'"'  Run F K=1:    {f_c}/225 = {100*f_c/225:.2f}%'"'"')
print(f'"'"'  LoRA v1 K=1:  {v1_c}/225 = {100*v1_c/225:.2f}%'"'"')
print(f'"'"'  LoRA v2 K=1:  {v2_c}/225 = {100*v2_c/225:.2f}%   Δ_v2-v1 = {100*(v2_c-v1_c)/225:+.2f}pp'"'"')
"

    echo "[$(date)] Augmentation pipeline done."
    echo "  LoRA v2 adapter:  lora_weights/runj_distill_v2/"
    echo "  LoRA v2 merged:   lora_weights/runj_distill_v2_merged/"
    echo "  val results:      results/lora_v2_val.jsonl"
' < /dev/null > "$LOG" 2>&1 &

PID=$!

cat <<EOF
Augment LoRA chain watcher started.
  PID: $PID
  Log: $LOG

Will wait for v4 fix-format, then:
  1. Build augmented SFT pool (4 sources, includes newly-fixed from v4)
  2. Retrain LoRA on augmented pool (~30-45min)
  3. Merge LoRA v2 adapter
  4. Eval on val_225 (~16min)
  5. Compare LoRA v2 vs v1 vs Run F

Total wallclock from v4 done: ~70-90min.

Monitor: tail -f $LOG
EOF
