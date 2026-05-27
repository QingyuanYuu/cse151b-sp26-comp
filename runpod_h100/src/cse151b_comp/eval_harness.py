"""Stratified validation split + comparison utilities.

Phase 2c module. The most useful function for now is :func:`make_split`,
which produces a stratified 20% validation set from ``data/public.jsonl`` and
saves the indices to ``data/val_indices.json``. Once that file exists, every
phase loads the same val set, so comparisons across runs are valid.

Compare-mode CLI:

    cse151b-eval-compare \\
        --baseline results/baseline_public_v1.jsonl \\
        --candidate results/sc_k8.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from collections import defaultdict


def question_type(row: dict) -> str:
    if row.get("options") or row.get("is_mcq"):
        return "mcq"
    gold = row.get("gold", row.get("answer"))
    if isinstance(gold, list) and len(gold) > 1:
        return "free_multi"
    return "free_single"


def make_split(
    data_path: pathlib.Path,
    out_path: pathlib.Path,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> dict:
    """Stratified val/train split by question type.

    Reads JSONL, assigns each row to val or train such that each
    question_type's val proportion ≈ ``val_fraction``. Writes
    ``{"val_ids": [...], "train_ids": [...], "seed": 42, "val_fraction": ...}``
    to ``out_path``.
    """
    rng = random.Random(seed)
    rows = [json.loads(line) for line in open(data_path)]

    by_type: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_type[question_type(r)].append(r["id"])

    val_ids: list[int] = []
    train_ids: list[int] = []
    for qtype, ids in by_type.items():
        ids_sorted = sorted(ids)  # deterministic ordering
        rng.shuffle(ids_sorted)
        n_val = int(round(len(ids_sorted) * val_fraction))
        val_ids.extend(ids_sorted[:n_val])
        train_ids.extend(ids_sorted[n_val:])

    val_ids.sort()
    train_ids.sort()

    payload = {
        "data_path": str(data_path),
        "seed": seed,
        "val_fraction": val_fraction,
        "n_total": len(rows),
        "n_val": len(val_ids),
        "n_train": len(train_ids),
        "by_type": {k: len(v) for k, v in by_type.items()},
        "val_ids": val_ids,
        "train_ids": train_ids,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return payload


def load_val_ids(path: pathlib.Path) -> set[int]:
    return set(json.loads(path.read_text())["val_ids"])


def load_train_ids(path: pathlib.Path) -> set[int]:
    return set(json.loads(path.read_text())["train_ids"])


# ─── Compare two scored result files ────────────────────────────────────────


def compare(baseline_rows: list[dict], candidate_rows: list[dict]) -> dict:
    """Return a confusion-style breakdown between two scored runs."""
    by_id_a = {r["id"]: r for r in baseline_rows if "correct" in r}
    by_id_b = {r["id"]: r for r in candidate_rows if "correct" in r}
    common = sorted(set(by_id_a) & set(by_id_b))

    both_right = both_wrong = only_a = only_b = 0
    only_a_ids: list[int] = []
    only_b_ids: list[int] = []
    for i in common:
        a = by_id_a[i]["correct"]
        b = by_id_b[i]["correct"]
        if a and b:
            both_right += 1
        elif a and not b:
            only_a += 1
            only_a_ids.append(i)
        elif b and not a:
            only_b += 1
            only_b_ids.append(i)
        else:
            both_wrong += 1
    return {
        "n_common": len(common),
        "both_right": both_right,
        "only_baseline_right (regression)": only_a,
        "only_candidate_right (gain)": only_b,
        "both_wrong": both_wrong,
        "regression_ids": only_a_ids[:20],
        "gain_ids": only_b_ids[:20],
    }


# ─── CLI entry points ──────────────────────────────────────────────────────


def make_split_main() -> None:
    p = argparse.ArgumentParser(description="Build stratified val/train split.")
    p.add_argument("--data", default="data/public.jsonl")
    p.add_argument("--out", default="data/val_indices.json")
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    info = make_split(
        pathlib.Path(args.data),
        pathlib.Path(args.out),
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"Split: n_total={info['n_total']} n_val={info['n_val']} n_train={info['n_train']}")
    print(f"By type: {info['by_type']}")
    print(f"Wrote {args.out}")


def compare_main() -> None:
    p = argparse.ArgumentParser(description="Compare two scored result JSONLs.")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    args = p.parse_args()
    a = [json.loads(line) for line in open(args.baseline)]
    b = [json.loads(line) for line in open(args.candidate)]
    diff = compare(a, b)
    print(json.dumps(diff, indent=2))


if __name__ == "__main__":
    make_split_main()
