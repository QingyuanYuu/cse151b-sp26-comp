#!/bin/bash
# Wait for results/private_v4.jsonl to appear, then convert to CSV.
set -eu
cd "$(dirname "$0")/.."

JSONL="results/private_v4.jsonl"
CSV="results/submission_v4.csv"

while [ ! -f "$JSONL" ]; do
    sleep 60
done

echo "[$(date)] JSONL detected, converting to CSV..."
PYTHONPATH=src .venv/bin/python -m cse151b_comp.submission \
    --results "$JSONL" \
    --out    "$CSV"

echo "[$(date)] Done."
ls -la "$CSV"
