#!/usr/bin/env bash
#
# Staged rejection sampling for SFT pool:
#   Stage 1: K=4 on all 1126
#   Stage 2: K=4 on questions still pass@4=0 (~170 expected)
#   Stage 3: K=8 on questions still pass@8=0 (~70 expected)
#
# Each later stage adds samples to questions that need more.
# Final pool: each question has 4 / 8 / 16 samples depending on difficulty.
#
# Total ~6-8h vs full K=16 ~17h.
#
# Triggered after K=8 SC private completes.
# Usage: scripts/staged_sc_chain.sh

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

LOG="logs/staged_sc.log"

# Wait until K=8 private finishes
echo "Watching for K=8 private completion..."
setsid bash -c '
    set -e

    # Wait for K=8 private chain to complete
    while [ -f logs/sc_runf_k8_private.pid ] && kill -0 $(cat logs/sc_runf_k8_private.pid 2>/dev/null) 2>/dev/null; do
        sleep 60
    done
    sleep 15  # vLLM cleanup

    echo "[$(date)] Stage 1: K=4 on full public 1126..."
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

    echo "[$(date)] Stage 1 done. Building hard subset..."
    PYTHONPATH=src .venv/bin/python scripts/build_subset.py \
        --pools results/sc_k4_public.jsonl \
        --source data/public.jsonl \
        --output data/subset_hard1.jsonl \
        --exclude-val

    n_hard1=$(wc -l < data/subset_hard1.jsonl)
    if [ "$n_hard1" = "0" ]; then
        echo "[$(date)] No hard questions left after Stage 1. Skipping Stage 2 + 3."
        exit 0
    fi

    echo "[$(date)] Stage 2: K=4 more on $n_hard1 hard1 questions..."
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

    echo "[$(date)] Stage 2 done. Building still-hard subset..."
    PYTHONPATH=src .venv/bin/python scripts/build_subset.py \
        --pools results/sc_k4_public.jsonl results/sc_k4_extra1_public.jsonl \
        --source data/public.jsonl \
        --output data/subset_hard2.jsonl \
        --exclude-val

    n_hard2=$(wc -l < data/subset_hard2.jsonl)
    if [ "$n_hard2" = "0" ]; then
        echo "[$(date)] No hard questions left after Stage 2. Skipping Stage 3."
        exit 0
    fi

    echo "[$(date)] Stage 3: K=8 more on $n_hard2 hard2 questions..."
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

    echo "[$(date)] All stages done. SFT pool ready: 3 files to merge."
    echo "  results/sc_k4_public.jsonl (K=4 on all 1126)"
    echo "  results/sc_k4_extra1_public.jsonl (K=4 extra on hard1)"
    echo "  results/sc_k8_extra2_public.jsonl (K=8 extra on hard2)"
' < /dev/null > "$LOG" 2>&1 &

PID=$!

cat <<EOF
Staged SC chain watcher started.
  Watcher PID: $PID
  Log:         $LOG

Pipeline (after K=8 private finishes):
  1. K=4 on all 1126                   (~5h)
  2. K=4 on hard subset (~170 q)       (~0.8h)
  3. K=8 on still-hard subset (~70 q)  (~0.5h)
  Total: ~6-7h

Output pools:
  results/sc_k4_public.jsonl
  results/sc_k4_extra1_public.jsonl
  results/sc_k8_extra2_public.jsonl

Monitor: tail -f $LOG
EOF
