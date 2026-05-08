"""Build SFT training pool from staged SC outputs (multi-pool merger).

Merges responses across stages (K=4 + K=4 extra + K=8 extra), picks best
correct sample as SFT target, excludes val_ids.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/prepare_sft_staged.py \\
        --pools results/sc_k4_public.jsonl results/sc_k4_extra1_public.jsonl results/sc_k8_extra2_public.jsonl \\
        --source data/public.jsonl \\
        --val data/val_indices.json \\
        --output data/sft_train_staged.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cse151b_comp.evaluate import score_response  # noqa: E402
from cse151b_comp.prompts import build_prompt_runf  # noqa: E402
from judger import Judger  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pools", nargs="+", required=True, help="One or more SC output jsonls (with all_responses)")
    p.add_argument("--source", required=True)
    p.add_argument("--val", default=None, help="val_indices.json (excludes val_ids)")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    # Index source
    source = {}
    for line in open(args.source):
        r = json.loads(line)
        source[r["id"]] = r
    print(f"[sft] source: {len(source)} questions")

    # Exclude val_ids
    val_ids: set = set()
    if args.val:
        vd = json.load(open(args.val))
        val_ids = set(vd["val_ids"])
        print(f"[sft] excluding {len(val_ids)} val_ids")

    # Merge pools: per-id list of all samples
    samples_by_id: dict[int, list[str]] = {}
    for pool in args.pools:
        for line in open(pool):
            r = json.loads(line)
            qid = r["id"]
            if qid in val_ids:
                continue
            if qid not in samples_by_id:
                samples_by_id[qid] = []
            samples_by_id[qid].extend(r.get("all_responses", []))
        print(f"[sft] merged from {pool}, total questions so far: {len(samples_by_id)}")

    # Per question: pick best correct sample
    judger = Judger(strict_extract=False)
    out_rows = []
    n_kept = 0
    n_no_correct = 0
    n_total_samples = 0

    for qid, samples in samples_by_id.items():
        if qid not in source:
            continue
        src = source[qid]
        gold = src.get("answer")
        options = src.get("options")
        n_total_samples += len(samples)

        # Find correct samples
        correct_samples = []
        for s in samples:
            try:
                ok = score_response(s, gold, options, judger)
            except Exception:
                ok = False
            if ok:
                correct_samples.append(s)

        if not correct_samples:
            n_no_correct += 1
            continue

        # Pick the longest correct sample (richer reasoning)
        target = max(correct_samples, key=len)

        # Build prompt for SFT input (Run F prompt — same as used for SC sampling)
        system_prompt, user_prompt = build_prompt_runf(src["question"], src.get("options"))

        out_rows.append(
            {
                "id": qid,
                "question_type": "mc" if src.get("options") else "free",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "target_response": target,
                "n_correct": len(correct_samples),
                "n_samples": len(samples),
            }
        )
        n_kept += 1

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_K = n_total_samples / max(len(samples_by_id), 1)
    print()
    print(f"[sft] questions seen     : {len(samples_by_id)}")
    print(f"[sft] avg samples/question: {avg_K:.1f}")
    print(f"[sft] kept (≥1 correct)   : {n_kept}")
    print(f"[sft] dropped (all wrong) : {n_no_correct}")
    print(f"[sft] wrote → {out_path}")


if __name__ == "__main__":
    main()
