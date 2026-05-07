#!/usr/bin/env bash
#
# Run J branch ablation: test each Run J variant on its target topic
# eval subset (~25-50 questions per topic, sampled from public_train).
#
# Each variant enables ONE branch only and routes everything else to
# Run F generic. Comparing variant accuracy to Run F baseline on the
# same topic subset isolates that branch's contribution.
#
# Output: results/runj_ablation_<topic>.jsonl (per variant)
# Summary: prints accuracy table at end.
#
# Wallclock estimate (K=1, single-shot, Blackwell, all sequential):
#   8 variants × ~30 questions × ~14 sec/question = ~1 hour total
#
# Usage:
#     scripts/run_j_ablation.sh

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

mkdir -p logs results

LOG="logs/runj_ablation.log"
PID_FILE="logs/runj_ablation.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Already running (PID $(cat "$PID_FILE"))."
    exit 1
fi
rm -f "$PID_FILE"

# Map: topic → (eval input file, prompt name to test)
# We test:
# 1. Run F baseline on EACH topic subset (control)
# 2. Each Run J branch variant on its target subset
declare -A TOPICS_TO_PROMPTS=(
    ["trig"]="runj_trig"
    ["geometry"]="runj_geom"
    ["logic_proof"]="runj_logic"
    ["stats_hyp_test"]="runj_stats"
    ["stats_regression"]="runj_stats"
    ["probability"]="runj_prob"
    ["num_theory"]="runj_num"
)

setsid bash -c "
set -e

run_one() {
    local topic=\"\$1\" prompt=\"\$2\" out_suffix=\"\$3\"
    local input=\"data/eval_subsets/eval_\${topic}.jsonl\"
    local out=\"results/runj_ablation_\${topic}_\${out_suffix}.jsonl\"
    if [ ! -f \"\$input\" ]; then
        echo \"  SKIP: \$input not found\"
        return 0
    fi
    local n_input=\$(wc -l < \"\$input\")
    echo \"\"
    echo \"=== \$(date '+%H:%M:%S') topic=\$topic prompt=\$prompt input=\$input (\$n_input qs) ===\"
    uv run --no-sync cse151b-sc \\
        --input \"\$input\" \\
        --output \"\$out\" \\
        --resume \\
        --chunk-size 100 \\
        --k 1 \\
        --temperature 0.6 --top-p 0.95 \\
        --prompt \"\$prompt\" \\
        --gpu-mem-util 0.92 \\
        --max-model-len 32768 \\
        --max-num-seqs 64 \\
        --allocate-tokens 2>&1 | grep -E '^\\\[sc\\\]|Wrote|Accuracy'
}

# Run baseline (Run F) on each topic subset (control)
for topic in trig geometry logic_proof stats_hyp_test stats_regression probability num_theory; do
    run_one \"\$topic\" \"runf\" \"baseline\"
done

# Run each branch variant on its target topic subset
for topic_prompt in \\
    'trig:runj_trig' \\
    'geometry:runj_geom' \\
    'logic_proof:runj_logic' \\
    'stats_hyp_test:runj_stats' \\
    'stats_regression:runj_stats' \\
    'probability:runj_prob' \\
    'num_theory:runj_num'; do
    topic=\${topic_prompt%%:*}
    prompt=\${topic_prompt##*:}
    run_one \"\$topic\" \"\$prompt\" \"variant\"
done

echo \"\"
echo \"=== \$(date '+%H:%M:%S') ALL VARIANTS DONE ===\"
echo \"\"
echo \"=== Re-judging all ablation outputs ===\"
for f in results/runj_ablation_*.jsonl; do
    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py \"\$f\" 2>&1 | grep -E 'Re-judged|New correct'
done
rm -f '$PID_FILE'
" < /dev/null > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

cat <<EOF
Started Run J ablation harness.
  PID: $PID
  Log: $LOG

Will run 14 inferences:
  7 baselines (Run F on each topic subset)
  7 variants  (each Run J branch on its target subset)

Then auto re-judge all outputs with Judger.auto_judge.

Monitor: tail -f $LOG
Summary script (run after completion): scripts/summarize_j_ablation.py
EOF
