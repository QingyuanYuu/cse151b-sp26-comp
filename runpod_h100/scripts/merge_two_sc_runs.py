"""Merge two K=4 SC runs into an effective K=8 voting result.

Input: round1.jsonl + round2.jsonl (each is `cse151b_comp.self_consistency` output)
Output: merged.jsonl with combined all_extracted/all_responses + re-voted winner.

Usage:
    python scripts/merge_two_sc_runs.py \\
        --round1 results/private_sc_k4.jsonl \\
        --round2 results/private_sc_k4_round2.jsonl \\
        --output results/private_sc_k8_merged.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def _index_by_id(path: pathlib.Path) -> dict:
    rows: dict = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--round1", required=True)
    p.add_argument("--round2", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    from cse151b_comp.voting import vote_for_row

    r1 = _index_by_id(pathlib.Path(args.round1))
    r2 = _index_by_id(pathlib.Path(args.round2))
    common_ids = sorted(set(r1) & set(r2))
    print(f"[merge] round1 rows: {len(r1)}, round2 rows: {len(r2)}, common: {len(common_ids)}")
    if not common_ids:
        sys.exit("[merge] no common ids — aborting")

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_out = 0
    n_kept_r1_winner = 0
    n_changed_winner = 0
    with open(out_path, "w") as fout:
        for qid in common_ids:
            a = r1[qid]
            b = r2[qid]
            merged_responses = list(a.get("all_responses", [])) + list(b.get("all_responses", []))
            merged_extracted = list(a.get("all_extracted", [])) + list(b.get("all_extracted", []))
            qtype = a.get("question_type") or b.get("question_type")
            options = a.get("options") or b.get("options")
            answer  = a.get("answer", b.get("answer"))

            # Re-vote across K=4+K=4 using the same voting logic the SC pipeline uses.
            # Construct a synthetic row in the format vote_for_row expects.
            synth = {
                "id": qid,
                "question_type": qtype,
                "all_responses": merged_responses,
                "all_extracted": merged_extracted,
                "K": len(merged_responses),
                "options": options,
                "answer": answer,
            }
            try:
                vote = vote_for_row(synth)
                winning = vote.get("winning_answer")
                winning_response = vote.get("winning_response")
                vote_counts = vote.get("vote_counts")
            except Exception:
                # Fallback: plurality on extracted strings
                counts = Counter([e for e in merged_extracted if e is not None])
                winning, _ = counts.most_common(1)[0] if counts else ("", 0)
                winning_response = next((r for r, e in zip(merged_responses, merged_extracted) if e == winning), merged_responses[0] if merged_responses else "")
                vote_counts = dict(counts)

            if winning == a.get("winning_answer"):
                n_kept_r1_winner += 1
            else:
                n_changed_winner += 1

            out = {
                "id": qid,
                "question_type": qtype,
                "all_responses": merged_responses,
                "all_extracted": merged_extracted,
                "vote_counts": vote_counts,
                "winning_answer": winning,
                "winning_response": winning_response,
                "K": len(merged_responses),
                "options": options,
            }
            if answer is not None:
                out["answer"] = answer
            fout.write(json.dumps(out) + "\n")
            n_out += 1

    print(f"[merge] wrote {n_out} rows → {out_path}")
    print(f"  kept round-1 winner: {n_kept_r1_winner}")
    print(f"  changed winner:      {n_changed_winner}")


if __name__ == "__main__":
    main()
