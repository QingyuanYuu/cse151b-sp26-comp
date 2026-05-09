#!/usr/bin/env bash
#
# Full pipeline: K=8 SC private (already running) → staged K-sampling
# (K=4/4/8) → SFT pool merge → LoRA train 4 epochs.
#
# Triggered to start automatically after K=8 private finishes.
# Total wallclock from now: ~9h K=8 + ~6-7h staged + ~1.5h LoRA ≈ ~17h.
#
# Usage: scripts/staged_pipeline.sh

set -euo pipefail

cd "$(dirname "$0")/.."

LOG="logs/staged_pipeline.log"

setsid bash -c '
    set -e
    unset VIRTUAL_ENV CONDA_PREFIX
    export PATH="$HOME/.local/bin:$PATH"

    # ─── Step 0: Wait for K=8 private to finish ────────────────────────
    echo "[$(date)] Waiting for K=8 SC private to finish..."
    while [ -f logs/sc_runf_k8_private.pid ] && kill -0 $(cat logs/sc_runf_k8_private.pid 2>/dev/null) 2>/dev/null; do
        sleep 60
    done
    sleep 15  # vLLM cleanup
    echo "[$(date)] K=8 private done."

    # ─── Step 1: K=4 on full public ────────────────────────────────────
    echo "[$(date)] Stage 1: K=4 on public 1126..."
    uv run --no-sync cse151b-sc \
        --input data/public.jsonl \
        --output results/sc_k4_public.jsonl \
        --resume \
        --chunk-size 100 \
        --k 4 \
        --temperature 0.7 --top-p 0.95 \
        --prompt runf \
        --gpu-mem-util 0.92 \
        --max-model-len 32768 \
        --max-num-seqs 32 \
        --allocate-tokens

    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py results/sc_k4_public.jsonl 2>&1 | tail -4

    # ─── Step 2: Build hard1 subset ─────────────────────────────────
    echo "[$(date)] Building hard1 subset (questions still pass=0)..."
    PYTHONPATH=src .venv/bin/python scripts/build_subset.py \
        --pools results/sc_k4_public.jsonl \
        --source data/public.jsonl \
        --output data/subset_hard1.jsonl \
        --exclude-val

    n_hard1=$(wc -l < data/subset_hard1.jsonl)
    echo "[$(date)] hard1: $n_hard1 questions"

    if [ "$n_hard1" -gt 0 ]; then
        # ─── Step 3: K=4 more on hard1 ────────────────────────────
        echo "[$(date)] Stage 2: K=4 on hard1 ($n_hard1 questions)..."
        uv run --no-sync cse151b-sc \
            --input data/subset_hard1.jsonl \
            --output results/sc_k4_extra1_public.jsonl \
            --resume \
            --chunk-size 100 \
            --k 4 \
            --temperature 0.7 --top-p 0.95 \
            --prompt runf \
            --gpu-mem-util 0.92 \
            --max-model-len 32768 \
            --max-num-seqs 32 \
            --allocate-tokens

        PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py results/sc_k4_extra1_public.jsonl 2>&1 | tail -4

        # ─── Step 4: Build hard2 subset ───────────────────────────
        echo "[$(date)] Building hard2 subset..."
        PYTHONPATH=src .venv/bin/python scripts/build_subset.py \
            --pools results/sc_k4_public.jsonl results/sc_k4_extra1_public.jsonl \
            --source data/public.jsonl \
            --output data/subset_hard2.jsonl \
            --exclude-val

        n_hard2=$(wc -l < data/subset_hard2.jsonl)
        echo "[$(date)] hard2: $n_hard2 questions"

        if [ "$n_hard2" -gt 0 ]; then
            # ─── Step 5: K=8 more on hard2 ────────────────────────
            echo "[$(date)] Stage 3: K=8 on hard2 ($n_hard2 questions)..."
            uv run --no-sync cse151b-sc \
                --input data/subset_hard2.jsonl \
                --output results/sc_k8_extra2_public.jsonl \
                --resume \
                --chunk-size 50 \
                --k 8 \
                --temperature 0.7 --top-p 0.95 \
                --prompt runf \
                --gpu-mem-util 0.92 \
                --max-model-len 32768 \
                --max-num-seqs 24 \
                --allocate-tokens

            PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py results/sc_k8_extra2_public.jsonl 2>&1 | tail -4
        fi
    fi

    # ─── Step 6: Merge pools → SFT pool ────────────────────────────
    echo "[$(date)] Merging staged pools into SFT pool..."
    pools=(results/sc_k4_public.jsonl)
    [ -f results/sc_k4_extra1_public.jsonl ] && pools+=(results/sc_k4_extra1_public.jsonl)
    [ -f results/sc_k8_extra2_public.jsonl ] && pools+=(results/sc_k8_extra2_public.jsonl)

    PYTHONPATH=src .venv/bin/python scripts/prepare_sft_staged.py \
        --pools "${pools[@]}" \
        --source data/public.jsonl \
        --val data/val_indices.json \
        --output data/sft_train_staged.jsonl

    n_sft=$(wc -l < data/sft_train_staged.jsonl)
    echo "[$(date)] SFT pool: $n_sft pairs"

    # ─── Step 7: LoRA training (up to 5 epochs, pick best by eval loss) ─────
    echo "[$(date)] LoRA training up to 5 epochs (auto-pick best checkpoint)..."
    mkdir -p lora_weights/runj_distill_v1

    PYTHONPATH=src .venv/bin/python scripts/train_lora.py \
        --train data/sft_train_staged.jsonl \
        --output lora_weights/runj_distill_v1 \
        --epochs 5 \
        --eval-frac 0.1 \
        --r 32 \
        --alpha 64

    echo "[$(date)] Pipeline complete."
    echo "  LoRA adapter: lora_weights/runj_distill_v1/"
    echo "  Next step: load LoRA + run inference (K=1 or K=8 SC)"
' < /dev/null > "$LOG" 2>&1 &

PID=$!

cat <<EOF
Staged pipeline watcher started.
  Watcher PID: $PID
  Log:         $LOG

Pipeline (will fire after K=8 SC private finishes):
  Stage 1: K=4 on public 1126                  (~5h)
  Stage 2: K=4 on hard1 subset (~170 q)         (~0.8h)
  Stage 3: K=8 on hard2 subset (~70 q)          (~0.5h)
  Step 6:  Merge pools → SFT data               (~5min)
  Step 7:  LoRA train up to 5 epochs + eval     (~2-2.5h)
           (10% eval split; auto-pick best by eval_loss)

Total from K=8 finish: ~7.5-8.5h
Total from now: ~17h ≈ tomorrow morning ~12:00 PDT

Monitor: tail -f $LOG
EOF
