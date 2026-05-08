"""Build a subset jsonl of questions that need more SC samples.

Reads one or more SC output files (each row has 'all_responses', 'correct',
'id'). Identifies questions where pass@K = 0 (no correct sample yet).
Writes a subset jsonl with those question IDs + original question/options/answer
fields, ready to feed into another cse151b-sc run.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_subset.py \\
        --pools results/sc_k4_public.jsonl [results/sc_k4_extra1_public.jsonl ...] \\
        --source data/public.jsonl \\
        --output data/subset_hard.jsonl \\
        [--threshold 0]   # min number of correct samples to "resolve"
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
from judger import Judger  # noqa: E402


def _judge_samples(samples: list[str], gold, options, judger: Judger) -> int:
    """Return count of correct samples among the K responses."""
    n = 0
    for resp in samples:
        try:
            ok = score_response(resp, gold, options, judger)
        except Exception:
            ok = False
        if ok:
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pools", nargs="+", required=True, help="One or more SC output jsonls to merge")
    p.add_argument("--source", required=True, help="Original public.jsonl with question/options/answer")
    p.add_argument("--output", required=True, help="Output subset jsonl (questions still pass=0)")
    p.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Resolve when n_correct > threshold (default 0 = need at least 1 correct)",
    )
    p.add_argument("--exclude-val", action="store_true", help="Exclude val_indices ids")
    args = p.parse_args()

    # Index source
    source = {}
    for line in open(args.source):
        r = json.loads(line)
        source[r["id"]] = r
    print(f"[subset] source: {len(source)} questions")

    # Optionally exclude val_ids
    excluded_ids = set()
    if args.exclude_val:
        vd = json.load(open("data/val_indices.json"))
        excluded_ids = set(vd["val_ids"])
        print(f"[subset] excluding {len(excluded_ids)} val_ids")

    # Merge pools: per-id list of all samples seen
    samples_by_id: dict[int, list[str]] = {}
    for pool in args.pools:
        n_in_pool = 0
        for line in open(pool):
            r = json.loads(line)
            qid = r["id"]
            if qid in excluded_ids:
                continue
            if qid not in samples_by_id:
                samples_by_id[qid] = []
            samples_by_id[qid].extend(r.get("all_responses", []))
            n_in_pool += 1
        print(f"[subset] loaded {n_in_pool} rows from {pool}")

    # Judge each question's accumulated samples
    judger = Judger(strict_extract=False)
    n_resolved = 0
    n_unresolved = 0
    out_rows = []
    for qid, samples in samples_by_id.items():
        if qid not in source:
            continue
        src = source[qid]
        n_correct = _judge_samples(samples, src.get("answer"), src.get("options"), judger)
        if n_correct > args.threshold:
            n_resolved += 1
        else:
            n_unresolved += 1
            out_rows.append(src)

    print(f"[subset] resolved (n_correct > {args.threshold}): {n_resolved}")
    print(f"[subset] unresolved (need more samples): {n_unresolved}")

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[subset] wrote {len(out_rows)} unresolved questions → {out_path}")


if __name__ == "__main__":
    main()
