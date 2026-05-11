#!/usr/bin/env bash
# After GRPO v2 finishes + auto-merges, run:
#   1. val K=1 sanity check
#   2. public K=1 full inference (1126 questions)
#   3. private K=8 SC (943 questions) → CSV submission
#   4. git commit + push everything

set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="lora_weights/runj_grpo_v2_merged"
LOG_DIR="logs"
RES_DIR="results"
mkdir -p "$LOG_DIR" "$RES_DIR"

# Wait for merged model to appear (auto-merge step from run_grpo_v2.sh)
echo "[chain] Waiting for $MODEL to exist..."
until [ -f "$MODEL/model.safetensors.index.json" ]; do
    sleep 30
done
echo "[chain] Merged model present: $MODEL"

run_sc() {
    local input=$1 output=$2 label=$3 k=$4 limit=$5
    local log="$LOG_DIR/grpo_v2_${label}.log"
    echo
    echo "[chain] === ${label} K=${k} starting $(date '+%H:%M:%S') ==="
    local extra=""
    [ -n "$limit" ] && extra="--val $limit"
    PYTHONPATH=src .venv/bin/python -m cse151b_comp.self_consistency \
        --input "$input" \
        --output "$output" \
        --prompt runf \
        --k "$k" \
        --model "$MODEL" \
        --bf16 \
        --gpu-mem-util 0.85 \
        --max-model-len 16384 \
        --max-num-seqs 64 \
        --chunk-size 25 \
        --resume \
        $extra > "$log" 2>&1
    echo "[chain] ${label} done $(date '+%H:%M:%S')"
    tail -5 "$log" | tr '\r' '\n' | grep -E "Accuracy|Done in" | tail -3
}

# 1. val K=1 sanity
run_sc data/public.jsonl  "$RES_DIR/grpo_v2_val_k1.jsonl"  val_k1  1  data/val_indices.json

# 2. public K=1 (full train)
run_sc data/public.jsonl  "$RES_DIR/grpo_v2_public_k1.jsonl"  public_k1  1  ""

# 3. private K=8 SC (final submission target)
run_sc data/private.jsonl "$RES_DIR/grpo_v2_private_k8.jsonl"  private_k8  8  ""

# 4. Build CSV from private K=8 (id, winning_response)
echo
echo "[chain] === build CSV $(date '+%H:%M:%S') ==="
mkdir -p submissions
PYTHONPATH=src .venv/bin/python << 'PYEOF'
import csv, json, pathlib
src = pathlib.Path('results/grpo_v2_private_k8.jsonl')
dst = pathlib.Path('submissions/grpo_v2_private_k8.csv')
rows = [json.loads(l) for l in open(src)]
seen = set()
with open(dst, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['id','response'])
    for r in rows:
        if r['id'] in seen: continue
        seen.add(r['id'])
        w.writerow([r['id'], r.get('winning_response','')])
boxed = sum(1 for r in rows if '\\boxed{' in r.get('winning_response',''))
print(f"Wrote {len(seen)} rows → {dst}")
print(f"Rows with \\boxed{{}}: {boxed}/{len(rows)}")
PYEOF

# 5. Compute val accuracy for quick comparison
echo
echo "[chain] === val accuracy comparison $(date '+%H:%M:%S') ==="
PYTHONPATH=src .venv/bin/python << 'PYEOF'
import json
val_ids = set(json.load(open('data/val_indices.json'))['val_ids'])
def acc(p, label):
    rows = [json.loads(l) for l in open(p)]
    rows = [r for r in rows if r['id'] in val_ids]
    correct = sum(1 for r in rows if r.get('correct'))
    return label, correct, len(rows)
for path, label in [
    ('results/runF_val.jsonl', 'Run F K=1'),
    ('results/lora_v2_val.jsonl', 'LoRA v2 K=1'),
    ('results/grpo_v1_val_runf_k1.jsonl', 'GRPO v1 K=1'),
    ('results/grpo_v2_val_k1.jsonl', 'GRPO v2 K=1 ★'),
]:
    try:
        l, c, n = acc(path, label)
        print(f"  {l:<22s} {c}/{n} = {c/n*100:.2f}%")
    except FileNotFoundError:
        print(f"  {label:<22s} (not found)")
PYEOF

# 6. Public accuracy for full 1126
echo
echo "[chain] === public accuracy $(date '+%H:%M:%S') ==="
PYTHONPATH=src .venv/bin/python << 'PYEOF'
import json
rows = [json.loads(l) for l in open('results/grpo_v2_public_k1.jsonl')]
correct = sum(1 for r in rows if r.get('correct'))
print(f"GRPO v2 public K=1: {correct}/{len(rows)} = {correct/len(rows)*100:.2f}%")
PYEOF

echo
echo "[chain] === DONE $(date '+%H:%M:%S') ==="
echo "[chain] All inference + CSV ready. Next: review + git commit+push (manual step)."
