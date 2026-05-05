#!/bin/bash
# Chain: wait for v5 sanity → wait briefly for GPU to free → run v6 + K=8 SC.
# Then convert SC output to CSV. All steps fully detached from any kernel.
#
# Outputs (the ones that matter for submission):
#   results/private_v6_sc_k8.jsonl      — raw SC results (per-question voting trace)
#   results/submission_v6_sc_k8.csv     — Kaggle CSV
#   reports/sc_k8_summary.md            — quick analysis

set -u
cd "$(dirname "$0")/.."

V5_JSONL="results/private_v5_sanity.jsonl"
V6_SC_JSONL="results/private_v6_sc_k8.jsonl"
V6_SC_CSV="results/submission_v6_sc_k8.csv"

# ─── Step 1: wait for v5 sanity to land ──────────────────────────────────
echo "[$(date)] Waiting for $V5_JSONL ..."
while [ ! -f "$V5_JSONL" ]; do
    sleep 60
done
echo "[$(date)] v5 sanity done. Size: $(stat -c %s "$V5_JSONL") bytes."

# ─── Step 2: free GPU (kill any straggler vLLM EngineCore) ────────────────
echo "[$(date)] Reaping any straggler EngineCore processes..."
for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null); do
    echo "  killing pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
done
for pid in $(pgrep -f "cse151b_comp.inference_starter_baseline" 2>/dev/null); do
    echo "  killing inference pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
done
sleep 5
echo "[$(date)] GPU state:"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

# ─── Step 3: run v6 + self-consistency K=8 on private ─────────────────────
echo "[$(date)] Starting self-consistency K=8 with v6 prompts..."
PYTHONPATH=src .venv/bin/python -m cse151b_comp.self_consistency \
    --input data/private.jsonl \
    --output "$V6_SC_JSONL" \
    --k 8 \
    --temperature 0.7 \
    --top-p 0.95 \
    --top-k 20 \
    --max-tokens 12288 \
    --prompt current \
    --max-model-len 20480 \
    --max-num-seqs 16 \
    --gpu-mem-util 0.70

echo "[$(date)] Self-consistency done. Output: $(stat -c %s "$V6_SC_JSONL") bytes."

# ─── Step 4: convert SC JSONL → Kaggle CSV ────────────────────────────────
echo "[$(date)] Building Kaggle CSV from SC results (using winning_response)..."
PYTHONPATH=src .venv/bin/python <<'PY'
import csv, json, pathlib

src = pathlib.Path("results/private_v6_sc_k8.jsonl")
dst = pathlib.Path("results/submission_v6_sc_k8.csv")

rows = [json.loads(line) for line in open(src)]
with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, quoting=csv.QUOTE_ALL)
    w.writerow(["id", "response"])
    for r in rows:
        w.writerow([r["id"], r["winning_response"]])

# Sanity
with open(dst, encoding="utf-8") as f:
    out = list(csv.DictReader(f))
boxed = sum(1 for r in out if r"\boxed{" in r["response"])
empty = sum(1 for r in out if not r["response"])
unique = len(set(r["id"] for r in out))
print(f"Wrote {len(out)} rows to {dst}")
print(f"  Unique ids       : {unique}")
print(f"  Empty responses  : {empty}")
print(f"  Has boxed marker : {boxed}/{len(out)} = {boxed/len(out)*100:.1f}%")
PY

# ─── Step 5: tiny summary report ─────────────────────────────────────────
echo "[$(date)] Writing summary..."
PYTHONPATH=src .venv/bin/python <<'PY'
import json, pathlib
from collections import Counter

rows = [json.loads(l) for l in open("results/private_v6_sc_k8.jsonl")]
type_counts = Counter(r["question_type"] for r in rows)

# Vote concentration: how decisive was each vote?
def concentration(r):
    vc = r.get("vote_counts", {})
    if not vc:
        return 0.0
    top = max(vc.values())
    total = sum(vc.values())
    return top / total if total else 0.0

conc = [concentration(r) for r in rows]
unanimous = sum(1 for c in conc if c == 1.0)
near_split = sum(1 for c in conc if c <= 0.5)

lines = [
    "# Self-Consistency K=8 (v6 prompts) — Submission Summary",
    "",
    f"Source: `results/private_v6_sc_k8.jsonl` ({len(rows)} questions).",
    "",
    "## Routing distribution",
    "",
    f"- mc          : {type_counts.get('mc', 0)}",
    f"- free_single : {type_counts.get('free_single', 0)}",
    f"- free_multi  : {type_counts.get('free_multi', 0)}",
    "",
    "## Vote concentration (winning fraction)",
    "",
    f"- Unanimous (8/8): {unanimous} ({unanimous/len(rows)*100:.1f}%)",
    f"- Decisive (≥6/8): {sum(1 for c in conc if c >= 0.75)} ({sum(1 for c in conc if c >= 0.75)/len(rows)*100:.1f}%)",
    f"- Plurality (4-5/8): {sum(1 for c in conc if 0.5 < c < 0.75)} ({sum(1 for c in conc if 0.5 < c < 0.75)/len(rows)*100:.1f}%)",
    f"- Near-tie (≤4/8): {near_split} ({near_split/len(rows)*100:.1f}%)",
    "",
    "## Action",
    "",
    f"Submit `results/submission_v6_sc_k8.csv` to Kaggle.",
]

pathlib.Path("reports").mkdir(exist_ok=True)
out = pathlib.Path("reports/sc_k8_summary.md")
out.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nWrote {out}")
PY

echo "[$(date)] All done. Submit: results/submission_v6_sc_k8.csv"
