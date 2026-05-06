#!/usr/bin/env bash
#
# Standalone JSONL → submission CSV converter.
# Use this if the auto-conversion in run_sc_phase0_k8_private.sh didn't run
# (e.g. SC was killed mid-flight, or you want to re-export an existing JSONL).
#
# Usage:
#     scripts/sc_to_csv.sh <input.jsonl> <output.csv>
#
# Example:
#     scripts/sc_to_csv.sh results/sc_phase0_k8_private.jsonl \
#                          submissions/sc_phase0_k8_private.csv

set -euo pipefail

cd "$(dirname "$0")/.."
unset VIRTUAL_ENV CONDA_PREFIX
export PATH="$HOME/.local/bin:$PATH"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input.jsonl> <output.csv>"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

if [ ! -f "$INPUT" ]; then
    echo "ERROR: $INPUT not found"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

uv run --no-sync python <<PY
import csv
import json
import pathlib

src = pathlib.Path('$INPUT')
dst = pathlib.Path('$OUTPUT')

rows = [json.loads(l) for l in open(src)]
seen = set()
n_no_box = 0
with open(dst, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'response'])
    for r in rows:
        rid = r['id']
        if rid in seen:
            continue
        seen.add(rid)
        resp = r.get('winning_response', '')
        if '\\\\boxed{' not in resp:
            n_no_box += 1
        w.writerow([rid, resp])

print(f'Wrote {len(seen)} unique rows -> {dst}')
print(f'Rows missing \\\\boxed{{}}: {n_no_box} ({100*n_no_box/max(len(seen),1):.1f}%)')
print()
print('Sanity check (first 3 ids + first 100 chars of response):')
import csv as _csv
with open(dst) as f:
    reader = _csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 3:
            break
        print(f'  id={row["id"]}: {row["response"][:100]!r}...')
PY
