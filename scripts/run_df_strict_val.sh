#!/bin/bash
# Re-run Run D and Run F on the stratified val_indices.json (same as
# day1-distill-pool branch's setup) for an apples-to-apples comparison.
#
# Our prior Run D val (63.56%) used --limit 225 (first 225 rows of public).
# The day1-distill-pool team's Run E and Run F val numbers used the
# stratified subset from data/val_indices.json (different 225 questions).
# Only 50 questions overlap, so the absolute numbers are not comparable.
#
# This script re-runs both Run D and Run F using --val val_indices.json
# (via self_consistency with k=1 = single-shot semantics) so we get
# Run D and Run F evaluated on the EXACT same 225 questions.

set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs reports

LOG=logs/df_strict_val.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] === Run D + Run F strict val ==="
echo

# Reap stale vLLM
for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null || true); do
    echo "  killing stale EngineCore pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
done
sleep 3
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

# ─── Run D on val_indices.json ──────────────────────────────────────────
echo
echo "[$(date)] Stage 1/2: Run D on stratified val (k=1)..."
PYTHONPATH=src .venv/bin/python -m cse151b_comp.self_consistency \
    --input data/public.jsonl \
    --val data/val_indices.json \
    --output results/val_runD_strict.jsonl \
    --prompt rund \
    --per-type-budget \
    --k 1 \
    --temperature 0.6 \
    --top-p 0.95 --top-k 20 \
    --max-model-len 24576 \
    --max-num-seqs 8 \
    --gpu-mem-util 0.70

# ─── Run F on val_indices.json ──────────────────────────────────────────
echo
echo "[$(date)] Stage 2/2: Run F on stratified val (k=1)..."
PYTHONPATH=src .venv/bin/python -m cse151b_comp.self_consistency \
    --input data/public.jsonl \
    --val data/val_indices.json \
    --output results/val_runF_strict.jsonl \
    --prompt runf \
    --per-type-budget \
    --k 1 \
    --temperature 0.6 \
    --top-p 0.95 --top-k 20 \
    --max-model-len 24576 \
    --max-num-seqs 8 \
    --gpu-mem-util 0.70

# ─── Comparison summary ────────────────────────────────────────────────
echo
echo "[$(date)] Building apples-apples comparison..."
PYTHONPATH=src .venv/bin/python <<'PY'
import json, pathlib
from collections import Counter

def load(path):
    return {r['id']: r for r in [json.loads(l) for l in open(path)]}

D = load('results/val_runD_strict.jsonl')
F = load('results/val_runF_strict.jsonl')

# Pull the day1-distill-pool reference numbers.
ref_E = 56.89
ref_F = 58.67

def acc(rows):
    n = sum(1 for r in rows.values() if r.get('correct'))
    return n, len(rows), n / len(rows) * 100

nD, tD, pD = acc(D)
nF, tF, pF = acc(F)

print()
print("=" * 70)
print(f"{'config':<30} {'val_acc':<15} {'note':<25}")
print("=" * 70)
print(f"{'Run D (this run, val_indices)':<30} {nD}/{tD} = {pD:.2f}%   our local rerun")
print(f"{'Run F (this run, val_indices)':<30} {nF}/{tF} = {pF:.2f}%   our local rerun")
print(f"{'Run E (day1-distill ref)':<30} -/225 = {ref_E:.2f}%      from runE_val_summary.md")
print(f"{'Run F (day1-distill ref)':<30} -/225 = {ref_F:.2f}%      from runF_val_summary.md")
print("=" * 70)

print()
print(f"Δ Run F − Run D (our setup, same val): {pF - pD:+.2f}pp")
print(f"Δ Run F (ours) vs Run F (day1): {pF - ref_F:+.2f}pp (machine + sampling noise)")
print()

# Per-type breakdown
def qtype(item):
    if item.get('options'):
        return 'mc'
    q = item.get('question', '')
    if q.count('[ANS]') >= 2:
        return 'free_multi'
    parts = sum(1 for m in ['(a)','(b)','(c)','(d)','(e)'] if m in q.lower())
    return 'free_multi' if parts >= 2 else 'free_single'

print(f"{'type':<14} {'n':<5} {'Run D':<14} {'Run F':<14} {'Δ':<8}")
for t in ['mc', 'free_single', 'free_multi']:
    ids = [i for i in D if qtype(D[i]) == t and i in F]
    cD = sum(1 for i in ids if D[i].get('correct')) if ids else 0
    cF = sum(1 for i in ids if F[i].get('correct')) if ids else 0
    if ids:
        print(f"{t:<14} {len(ids):<5} {cD}/{len(ids)} = {cD/len(ids)*100:.1f}%   "
              f"{cF}/{len(ids)} = {cF/len(ids)*100:.1f}%   "
              f"{(cF-cD)/len(ids)*100:+.1f}pp")

# Did Run F fix the 4 specific Run D bugs we identified earlier?
print()
print("=== Did Run F fix the 4 specific Run D bug cases? ===")
target_ids = [5, 30, 135, 192]
for tid in target_ids:
    if tid in D and tid in F:
        cD = D[tid].get('correct', False)
        cF = F[tid].get('correct', False)
        flag = ("✓ FIXED" if cF and not cD
                else "= both correct" if cF and cD
                else "✗ REGRESSED" if cD and not cF
                else "- both wrong")
        print(f"  id={tid}: D={cD} F={cF}  {flag}")
    else:
        in_d = "in D" if tid in D else "not in D"
        in_f = "in F" if tid in F else "not in F"
        print(f"  id={tid}: {in_d}, {in_f} (val_indices subset)")

# Summary report
report = pathlib.Path('reports/df_strict_val_summary.md')
report.write_text(f"""# Run D vs Run F — apples-to-apples on val_indices.json

Both runs use single-shot K=1 with --val data/val_indices.json + per-type budget.

| Config | Val acc | Source |
|---|---|---|
| Run D (this) | {nD}/{tD} = {pD:.2f}% | local rerun |
| Run F (this) | {nF}/{tF} = {pF:.2f}% | local rerun |
| Run E (ref)  | {ref_E:.2f}% | day1-distill-pool/reports/runE_val_summary.md |
| Run F (ref)  | {ref_F:.2f}% | day1-distill-pool/reports/runF_val_summary.md |

Δ Run F − Run D: **{pF - pD:+.2f}pp** (our setup, same val_indices).
Δ Run F (ours) − Run F (day1): {pF - ref_F:+.2f}pp (machine + sampling noise).
""")
print(f"\nWrote {report}")
PY

echo
echo "[$(date)] === Done ==="
