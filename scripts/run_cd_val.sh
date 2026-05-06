#!/usr/bin/env bash
#
# Re-validate Run C and Run D on val_225 (stratified) — the jason/dev
# numbers (Run C 65.33 %, Run D 63.56 %) were measured on the FIRST 225
# rows of public.jsonl, which is not stratified by question type. This
# script gets fresh, comparable numbers under the same setup as Run E
# (56.89 %) and Run F (58.67 %).
#
# Two runs sequential, ~14 min each → ~28 min total.

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

mkdir -p logs results

run_one() {
    local prompt="$1"
    local out="results/${prompt}_val.jsonl"
    local log="logs/${prompt}_val.log"
    echo
    echo "=== $(date '+%H:%M:%S') Starting $prompt val_225 ==="
    uv run --no-sync cse151b-sc \
        --input data/public.jsonl \
        --val data/val_indices.json \
        --output "$out" \
        --k 1 \
        --temperature 0.6 --top-p 0.95 \
        --prompt "$prompt" \
        --gpu-mem-util 0.92 \
        --max-model-len 26624 \
        --max-num-seqs 128 \
        --allocate-tokens \
        --max-tokens-floor 12288 \
        --max-tokens-ceiling 20480 2>&1 | tee "$log" | grep -E "^\[sc\]|histogram|Wrote|Accuracy|Solvable|Done|Error|Traceback"
    echo "=== $(date '+%H:%M:%S') $prompt done; output: $out ==="
}

run_one runc
run_one rund

echo
echo "=== All done at $(date '+%H:%M:%S') ==="
echo
echo "Quick comparison:"
for p in runc rund; do
    f="results/${p}_val.jsonl"
    if [ -f "$f" ]; then
        n=$(wc -l < "$f")
        c=$(uv run --no-sync python -c "
import json
rows = [json.loads(l) for l in open('$f')]
correct = sum(1 for r in rows if r.get('correct'))
print(f'{correct}/{len(rows)} = {100*correct/len(rows):.2f}%')
")
        echo "  $p: $c"
    fi
done
