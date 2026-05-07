"""Re-judge an existing SC JSONL using the course Judger (via
``cse151b_comp.evaluate.score_response``), overwriting the buggy
``correct`` field.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/rejudge_jsonl.py <jsonl> [--source data/public.jsonl]

If --source is omitted, the script tries to recover (question, options,
answer) from data/public.jsonl by row id (the natural use case for any
of the val_*/public_* outputs). For private outputs (no gold), this is
a no-op and the script just prints "no gold, nothing to re-judge".

Background:
    self_consistency.py prior to commit <fix> used a string-equality
    comparison on canonical-normalized forms for the per-row ``correct``
    field. That comparator did not normalise LaTeX fractions to decimals
    before comparing, so any prompt that prefers \\frac / \\sqrt
    (Run F final, Run G, Run I) was systematically under-counted by
    ~9pp on the full public set. See reports/runF_public_rejudge_finding.md
    (jason/dev 4f250a9) for the full diagnosis.

After running this script, the JSONL's ``correct`` field reflects the
real Judger.auto_judge verdict — same path as the Kaggle leaderboard.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cse151b_comp.evaluate import score_response  # noqa: E402
from judger import Judger  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", help="SC output JSONL to re-judge in place.")
    p.add_argument(
        "--source",
        default="data/public.jsonl",
        help="JSONL with question/options/answer keyed by id. Default data/public.jsonl.",
    )
    p.add_argument("--in-place", action="store_true", help="Overwrite the input file (default: True).")
    p.add_argument("--out", help="Write to this path instead of in-place.")
    args = p.parse_args()

    jsonl_path = pathlib.Path(args.jsonl)
    src_path = pathlib.Path(args.source)
    if not src_path.exists():
        print(f"ERROR: source {src_path} not found")
        sys.exit(1)

    src = {r["id"]: r for r in (json.loads(line) for line in open(src_path))}

    rows = [json.loads(line) for line in open(jsonl_path)]
    judger = Judger(strict_extract=False)

    n_total = 0
    n_with_gold = 0
    n_correct_old = 0
    n_correct_new = 0
    n_changed = 0
    type_total = Counter()
    type_correct_old = Counter()
    type_correct_new = Counter()

    for r in rows:
        n_total += 1
        qid = r["id"]
        item = src.get(qid)
        gold = (item or {}).get("answer") if item else r.get("answer")
        options = (item or {}).get("options") if item else None
        qtype = r.get("question_type", "unknown")
        type_total[qtype] += 1

        if gold is None:
            # Private set — no gold available, can't re-judge.
            continue
        n_with_gold += 1

        winning_response = r.get("winning_response", "") or (
            r.get("all_responses", [""])[0] if r.get("all_responses") else ""
        )

        old_correct = bool(r.get("correct", False))
        new_correct = bool(score_response(winning_response, gold, options, judger))

        if old_correct:
            n_correct_old += 1
            type_correct_old[qtype] += 1
        if new_correct:
            n_correct_new += 1
            type_correct_new[qtype] += 1
        if old_correct != new_correct:
            n_changed += 1

        r["correct"] = new_correct
        # Stash the old value for audit-ability; remove if cluttering.
        r["_correct_old_buggy"] = old_correct
        # Also fill answer if it wasn't there.
        if "answer" not in r:
            r["answer"] = gold

    out_path = pathlib.Path(args.out) if args.out else jsonl_path
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary.
    print(f"Re-judged {jsonl_path} → {out_path}")
    print(f"  rows total: {n_total}")
    print(f"  with gold : {n_with_gold}")
    if n_with_gold:
        print(f"  Old correct: {n_correct_old}/{n_with_gold} = {100*n_correct_old/n_with_gold:.2f}%")
        print(f"  New correct: {n_correct_new}/{n_with_gold} = {100*n_correct_new/n_with_gold:.2f}%")
        print(f"  Changed   : {n_changed} ({100*n_changed/n_with_gold:.1f}% of rows had wrong `correct`)")
        print()
        print("  By type (new acc vs old acc):")
        for qt in sorted(type_total):
            t = type_total[qt]
            if t == 0:
                continue
            old = type_correct_old[qt]
            new = type_correct_new[qt]
            print(
                f"    {qt:12} {old}/{t} = {100*old/t:.1f}%  →  {new}/{t} = {100*new/t:.1f}%  Δ {100*(new-old)/t:+.2f}pp"
            )


if __name__ == "__main__":
    main()
