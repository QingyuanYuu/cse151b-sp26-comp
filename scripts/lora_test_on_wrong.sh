#!/usr/bin/env bash
#
# Auto-chain: wait for LoRA training to finish → merge adapter → build
# Run F wrong subset → run K=1 inference with merged LoRA model →
# compare to Run F baseline (count flips).
#
# Total wallclock after LoRA done: ~5min merge + ~30-40min inference.
#
# Usage: scripts/lora_test_on_wrong.sh

set -euo pipefail

cd "$(dirname "$0")/.."

LOG="logs/lora_test_on_wrong.log"

setsid bash -c '
    set -e
    unset VIRTUAL_ENV CONDA_PREFIX
    export PATH="$HOME/.local/bin:$PATH"

    # ─── Wait for LoRA to finish ────────────────────────────────────────
    echo "[$(date)] Waiting for LoRA training to finish..."
    while ps -p $(cat logs/lora_train.pid 2>/dev/null) > /dev/null 2>&1 || pgrep -af "train_lora.py" > /dev/null 2>&1; do
        sleep 60
    done
    sleep 10

    if [ ! -d lora_weights/runj_distill_v1 ] || [ -z "$(ls -A lora_weights/runj_distill_v1 2>/dev/null)" ]; then
        echo "[$(date)] ERROR: LoRA adapter not found in lora_weights/runj_distill_v1/"
        exit 1
    fi

    echo "[$(date)] LoRA training done."

    # ─── Step 1: Merge adapter into base ────────────────────────────────
    if [ ! -d lora_weights/runj_distill_v1_merged ] || [ -z "$(ls lora_weights/runj_distill_v1_merged 2>/dev/null)" ]; then
        echo "[$(date)] Merging LoRA adapter into base model..."
        PYTHONPATH=src .venv/bin/python scripts/merge_lora.py \
            --base Qwen/Qwen3-4B-Thinking-2507 \
            --adapter lora_weights/runj_distill_v1 \
            --output lora_weights/runj_distill_v1_merged
    else
        echo "[$(date)] Merged model already exists, skipping merge."
    fi

    # ─── Step 2: Build Run F wrong subset ────────────────────────────────
    echo "[$(date)] Building Run F wrong subset..."
    PYTHONPATH=src .venv/bin/python scripts/build_runf_wrong_subset.py

    # ─── Step 3: Run K=1 inference with merged LoRA model ────────────────
    echo "[$(date)] Running LoRA K=1 inference on Run F wrong subset..."
    uv run --no-sync cse151b-sc \
        --input data/runf_wrong_subset.jsonl \
        --output results/lora_v1_on_runf_wrong.jsonl \
        --resume \
        --chunk-size 100 \
        --k 1 \
        --temperature 0.6 --top-p 0.95 \
        --prompt runf \
        --model lora_weights/runj_distill_v1_merged \
        --bf16 \
        --gpu-mem-util 0.85 \
        --max-model-len 32768 \
        --max-num-seqs 32 \
        --allocate-tokens

    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py results/lora_v1_on_runf_wrong.jsonl 2>&1 | tail -5

    # ─── Step 4: Compute flips ───────────────────────────────────────────
    echo
    echo "[$(date)] Comparing LoRA vs Run F on wrong subset..."
    PYTHONPATH=src .venv/bin/python << "PY"
import json
runf = {r["id"]: r for r in (json.loads(l) for l in open("results/runf_k1_public.jsonl"))}
lora = {r["id"]: r for r in (json.loads(l) for l in open("results/lora_v1_on_runf_wrong.jsonl"))}

# All these are Run F wrong by construction
total = len(lora)
fixed = sum(1 for q in lora if lora[q].get("correct"))
print(f"=== LoRA on Run F wrong subset ({total} questions) ===")
print(f"  Run F was wrong on all {total}")
print(f"  LoRA correct: {fixed}/{total} = {100*fixed/total:.1f}%")
print(f"  → Run F + LoRA combined would now be {711 + fixed}/1126 = {100*(711+fixed)/1126:.2f}%")
print(f"  → Lift: +{100*fixed/1126:.2f}pp on full public")
PY

    echo "[$(date)] Test done."
' < /dev/null > "$LOG" 2>&1 &

PID=$!

cat <<EOF
Auto-test watcher started.
  PID: $PID
  Log: $LOG

Will wait for LoRA, then:
  1. Merge LoRA adapter into base model        (~5min)
  2. Build Run F wrong subset (415 questions)  (~5s)
  3. Run K=1 inference on those 415 questions  (~30-40min)
  4. Compare → count how many LoRA can fix

Total wallclock after LoRA done: ~40-50min.

Monitor: tail -f $LOG
EOF
