#!/usr/bin/env bash
# Post-pipeline fixup:
#  - Build private CSV with correct CLI (--results not --input)
#  - Build public CSV (new! not in original pipeline)
#  - Rename both to grpov3_{public,private}.csv
#  - Rebuild handoff/ to include both CSVs
#  - Re-push handoff/ to git
set -euo pipefail

REPO=/workspace/cse151b-grpo/cse151b-sp26-comp/runpod_h100
PY=/workspace/cse151b-grpo/.venv/bin/python
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"

cd "$REPO"

echo "==[ $(date '+%F %T') ]== Fixup: build grpov3 CSVs"

# Private (K=4 SC)
PYTHONPATH=src $PY -m cse151b_comp.submission \
    --results "$REPO/results/private_sc_k4.jsonl" \
    --out     "$REPO/results/grpov3_private.csv" \
    2>&1 | tee "$LOG_DIR/grpov3_private_csv.log"

# Public (K=1)
PYTHONPATH=src $PY -m cse151b_comp.submission \
    --results "$REPO/results/public_k1.jsonl" \
    --out     "$REPO/results/grpov3_public.csv" \
    2>&1 | tee "$LOG_DIR/grpov3_public_csv.log"

# Clean up the old wrong-named ones if they exist
rm -f "$REPO/results/private_submission.csv"

echo "==[ $(date '+%F %T') ]== Fixup: rebuild handoff/"
$PY scripts/build_handoff.py 2>&1 | tee "$LOG_DIR/build_handoff_fixup.log"

echo "==[ $(date '+%F %T') ]== Fixup: git push"
bash scripts/git_push_handoff.sh 2>&1 | tee "$LOG_DIR/git_push_fixup.log"

echo "==[ $(date '+%F %T') ]== Fixup complete."
echo "  - Public CSV:  $REPO/results/grpov3_public.csv"
echo "  - Private CSV: $REPO/results/grpov3_private.csv"
