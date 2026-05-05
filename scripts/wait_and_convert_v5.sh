#!/bin/bash
# Wait for results/private_v5_sanity.jsonl and convert to CSV.
set -eu
cd "$(dirname "$0")/.."

JSONL="results/private_v5_sanity.jsonl"
CSV="results/submission_v5_sanity.csv"

while [ ! -f "$JSONL" ]; do
    sleep 60
done

echo "[$(date)] JSONL detected, converting to CSV..."
PYTHONPATH=src .venv/bin/python -m cse151b_comp.submission \
    --results "$JSONL" \
    --out    "$CSV"
echo "[$(date)] Done."
ls -la "$CSV"
