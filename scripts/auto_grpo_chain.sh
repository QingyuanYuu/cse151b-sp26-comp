#!/usr/bin/env bash
#
# Auto-fire GRPO after LoRA v2 chain completes.
#
# Steps:
# 1. Wait for LoRA v2 merged + val eval done (signaled by results/lora_v2_val.jsonl)
# 2. Compare val_225: LoRA v2 vs LoRA v1 vs Run F
# 3. Pick better base (v2 if it beats v1 on val, else v1)
# 4. Fire GRPO with chosen base, runs ~14-16h
#
# Usage: scripts/auto_grpo_chain.sh

set -euo pipefail

cd "$(dirname "$0")/.."

LOG="logs/auto_grpo.log"

setsid bash -c '
    set -e
    unset VIRTUAL_ENV CONDA_PREFIX
    export PATH="$HOME/.local/bin:$PATH"

    # ─── Wait for LoRA v2 chain (watches output files, no pgrep) ─────────
    echo "[$(date)] Waiting for LoRA v2 val eval to finish..."
    while [ ! -f results/lora_v2_val.jsonl ] || [ ! -f lora_weights/runj_distill_v2_merged/config.json ]; do
        sleep 60
    done

    # Wait extra 30s for Step 5 (compare script) to finish writing
    sleep 30

    # Also wait if val jsonl is < 225 rows (still being written)
    while [ "$(wc -l < results/lora_v2_val.jsonl 2>/dev/null || echo 0)" -lt 220 ]; do
        sleep 30
    done

    # And wait for any in-progress vLLM to finish
    while pgrep -f "EngineCore" > /dev/null 2>&1; do
        # Use file-based check + grep that avoids self-match
        if pgrep -af "EngineCore" 2>/dev/null | grep -v "auto_grpo" | grep -q "EngineCore"; then
            sleep 30
        else
            break
        fi
    done
    sleep 15

    # ─── Pick base: v2 if it beats v1 on val_225, else v1 ───────────────
    echo "[$(date)] Comparing v2 vs v1 on val..."
    PICK=$(PYTHONPATH=src .venv/bin/python -c "
import json
vd = json.load(open('"'"'data/val_indices.json'"'"'))
val_ids = set(vd['"'"'val_ids'"'"'])
runf = {r['"'"'id'"'"']: r for r in (json.loads(l) for l in open('"'"'results/runf_k1_public.jsonl'"'"'))}
v1 = {r['"'"'id'"'"']: r for r in (json.loads(l) for l in open('"'"'results/lora_v1_public.jsonl'"'"'))}

import pathlib
v2_path = pathlib.Path('"'"'results/lora_v2_val.jsonl'"'"')
if not v2_path.exists():
    print('"'"'v1'"'"')  # fallback
else:
    v2 = {r['"'"'id'"'"']: r for r in (json.loads(l) for l in open(v2_path))}
    f_c = sum(1 for q in val_ids if runf.get(q,{}).get('"'"'correct'"'"'))
    v1_c = sum(1 for q in val_ids if v1.get(q,{}).get('"'"'correct'"'"'))
    v2_c = sum(1 for q in val_ids if v2.get(q,{}).get('"'"'correct'"'"'))
    import sys
    print(f'"'"'val_225: F {f_c}/225 ({100*f_c/225:.2f}%)  v1 {v1_c}/225 ({100*v1_c/225:.2f}%)  v2 {v2_c}/225 ({100*v2_c/225:.2f}%)'"'"', file=sys.stderr)
    # Pick v2 only if it strictly beats v1
    if v2_c > v1_c:
        print('"'"'v2'"'"')
    else:
        print('"'"'v1'"'"')
")

    if [ "$PICK" = "v2" ]; then
        BASE="lora_weights/runj_distill_v2_merged"
        OUTPUT="lora_weights/runj_grpo_v2_base"
        echo "[$(date)] LoRA v2 wins val. GRPO will train on v2 base."
    else
        BASE="lora_weights/runj_distill_v1_merged"
        OUTPUT="lora_weights/runj_grpo_v1_base"
        echo "[$(date)] LoRA v1 ≥ v2 on val. GRPO will train on v1 base (more stable)."
    fi
    echo "[$(date)] BASE=$BASE  OUTPUT=$OUTPUT"

    # ─── Fire GRPO ──────────────────────────────────────────────────────
    echo "[$(date)] Starting GRPO (~14-16h)..."
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    PYTHONPATH=src .venv/bin/python scripts/train_grpo.py \
        --base "$BASE" \
        --output "$OUTPUT" \
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
        --use-vllm

    echo "[$(date)] GRPO done. Output: $OUTPUT"

    # ─── Auto-merge GRPO adapter ───────────────────────────────────────
    echo "[$(date)] Merging GRPO adapter..."
    PYTHONPATH=src .venv/bin/python scripts/merge_lora.py \
        --base "$BASE" \
        --adapter "$OUTPUT" \
        --output "${OUTPUT}_merged"

    echo "[$(date)] Auto-chain complete."
    echo "  GRPO adapter: $OUTPUT"
    echo "  GRPO merged:  ${OUTPUT}_merged"
    echo "  Next: run inference (e.g. K=1 or K=8 SC) with merged GRPO model"
' < /dev/null > "$LOG" 2>&1 &

PID=$!

cat <<EOF
Auto GRPO chain watcher started.
  PID: $PID
  Log: $LOG

Will wait for LoRA v2 val eval done (file-based check, no pgrep self-match),
compare v2 vs v1, pick the winning base, then fire GRPO.

GRPO config:
  900 prompts × K=8 × 4 epochs
  vLLM colocate (fast sampling)
  ~14-16h wallclock

Then auto-merge the GRPO adapter.

ETA full chain: ~07:00-10:00 Sun morning.

Monitor: tail -f $LOG
EOF
