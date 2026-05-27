"""3-way submission ensemble via per-type voting.

Takes N submission CSVs (each ``id,response``), looks up question type
from the source JSONL, votes per-type, and writes an ensembled CSV
preserving the *full* original response of the winning submission so
the grader's ``extract_all_boxed`` pipeline still sees real reasoning.

Tiebreak: when there's no plurality, pick the submission with highest
known leaderboard score (passed via ``--rank``).

Usage::

    cse151b-ensemble \\
        --data data/private.jsonl \\
        --csv results/sc_runf_k8_private.csv \\
        --csv submissions/runf_k1_private.csv \\
        --csv submissions/runj_v3_final_private.csv \\
        --rank sc_runf_k8_private.csv,runf_k1_private.csv,runj_v3_final_private.csv \\
        --out results/ensemble_3way_private.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import Counter

from cse151b_comp.extract import (
    extract_all_final_boxed,
    extract_letter,
    normalize_answer,
)
from cse151b_comp.prompts import detect_question_type


def _load_csv(path: pathlib.Path) -> dict[str, str]:
    with open(path, newline="") as fp:
        return {r["id"]: r["response"] for r in csv.DictReader(fp)}


def _load_data(path: pathlib.Path) -> dict[str, dict]:
    return {str(json.loads(l)["id"]): json.loads(l) for l in open(path)}


def _extract_tuple(response: str) -> tuple[str, ...]:
    boxes = extract_all_final_boxed(response)
    if not boxes:
        return ()
    if len(boxes) == 1 and "," in boxes[0]:
        parts = [p.strip() for p in boxes[0].split(",")]
    else:
        parts = boxes
    return tuple(normalize_answer(p) for p in parts)


def _vote_letter(responses: list[str], rank: list[int]) -> int:
    """Return index into ``responses`` of the winning submission."""
    letters = [extract_letter(r) for r in responses]
    counts = Counter(l for l in letters if l)
    if not counts:
        return rank[0]
    top = max(counts.values())
    winners = {l for l, c in counts.items() if c == top}
    # Among submissions whose letter is in winners, pick highest-rank.
    for i in rank:
        if letters[i] in winners:
            return i
    return rank[0]


def _vote_free_single(responses: list[str], rank: list[int]) -> int:
    extracted = []
    for r in responses:
        boxes = extract_all_final_boxed(r)
        extracted.append(normalize_answer(boxes[-1]) if boxes else "")
    counts = Counter(e for e in extracted if e)
    if not counts:
        return rank[0]
    top = max(counts.values())
    winners = {e for e, c in counts.items() if c == top}
    for i in rank:
        if extracted[i] in winners:
            return i
    return rank[0]


def _vote_free_multi(responses: list[str], rank: list[int]) -> int:
    tuples = [_extract_tuple(r) for r in responses]
    counts = Counter(t for t in tuples if t)
    if not counts:
        return rank[0]
    top = max(counts.values())
    if top > 1:
        winners = {t for t, c in counts.items() if c == top}
        for i in rank:
            if tuples[i] in winners:
                return i
    # All 3 different — take highest-rank submission.
    return rank[0]


def main() -> None:
    p = argparse.ArgumentParser(description="3-way submission ensemble.")
    p.add_argument("--data", required=True, help="Source JSONL (private/public).")
    p.add_argument("--csv", action="append", required=True,
                   help="Submission CSV (repeatable, in any order).")
    p.add_argument("--rank", required=True,
                   help="Comma-separated CSV basenames in priority order "
                        "(highest leaderboard first; tiebreak prefers earlier).")
    p.add_argument("--out", required=True, help="Output ensembled CSV.")
    args = p.parse_args()

    data = _load_data(pathlib.Path(args.data))
    csv_paths = [pathlib.Path(c) for c in args.csv]
    submissions = [_load_csv(cp) for cp in csv_paths]
    name_to_idx = {cp.name: i for i, cp in enumerate(csv_paths)}
    rank_names = [n.strip() for n in args.rank.split(",")]
    rank = [name_to_idx[n] for n in rank_names if n in name_to_idx]
    if len(rank) != len(csv_paths):
        missing = [n for n in rank_names if n not in name_to_idx]
        raise SystemExit(f"--rank names don't match --csv basenames: missing {missing}")

    # Stats
    n_total = 0
    n_agree_all = 0
    n_split_2_1 = 0
    n_disagree_all = 0
    by_type = Counter()
    winner_source = Counter()

    rows_out: list[dict[str, str]] = []
    for qid, item in data.items():
        responses = [s.get(qid, "") for s in submissions]
        if not all(responses):
            # Fall back to highest-ranked non-empty submission.
            for i in rank:
                if responses[i]:
                    rows_out.append({"id": qid, "response": responses[i]})
                    winner_source[csv_paths[i].name] += 1
                    break
            continue

        qtype = detect_question_type(item.get("question", ""), item.get("options"))
        by_type[qtype] += 1

        if qtype == "mc":
            keys = [extract_letter(r) for r in responses]
            voter = _vote_letter
        elif qtype == "free_multi":
            keys = [_extract_tuple(r) for r in responses]
            voter = _vote_free_multi
        else:
            keys = []
            for r in responses:
                boxes = extract_all_final_boxed(r)
                keys.append(normalize_answer(boxes[-1]) if boxes else "")
            voter = _vote_free_single

        # Agreement bucket
        valid_keys = [k for k in keys if k]
        unique = len(set(valid_keys))
        if unique == 1 and len(valid_keys) == len(responses):
            n_agree_all += 1
        elif unique == 2:
            n_split_2_1 += 1
        else:
            n_disagree_all += 1

        n_total += 1
        winning_idx = voter(responses, rank)
        winner_source[csv_paths[winning_idx].name] += 1
        rows_out.append({"id": qid, "response": responses[winning_idx]})

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["id", "response"], quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"Wrote {len(rows_out)} rows to {out_path}")
    print()
    print("Question type breakdown:")
    for t in ("mc", "free_single", "free_multi"):
        print(f"  {t:12s}: {by_type[t]}")
    print()
    print(f"Agreement breakdown (n={n_total}):")
    print(f"  all 3 agree   : {n_agree_all} ({100*n_agree_all/n_total:.1f}%)")
    print(f"  2-1 split     : {n_split_2_1} ({100*n_split_2_1/n_total:.1f}%)")
    print(f"  3 disagree    : {n_disagree_all} ({100*n_disagree_all/n_total:.1f}%)")
    print()
    print("Winner source (which submission's response was kept):")
    for name, c in winner_source.most_common():
        print(f"  {name:40s}: {c} ({100*c/sum(winner_source.values()):.1f}%)")


if __name__ == "__main__":
    main()
