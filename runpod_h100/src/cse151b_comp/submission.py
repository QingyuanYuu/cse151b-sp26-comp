"""Convert a results JSONL into a Kaggle-format CSV submission.

The CSV must have ``id, response`` columns (no ``is_mcq``, no ``gold``).
Strings are wrapped with :data:`csv.QUOTE_ALL` so commas, newlines, and
backslashes inside the response trace are preserved verbatim.

Run via:
    cse151b-submit --results results/baseline_public_v1.jsonl \\
                   --out results/submission_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib


def write_csv(rows: list[dict], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for r in rows:
            # K=1 inference writes "response"; self-consistency writes
            # "winning_response" (the vote-winning trace). Accept either.
            resp = r.get("response") or r.get("winning_response", "")
            writer.writerow([r["id"], resp])


def sanity_check(out_path: pathlib.Path) -> dict:
    with open(out_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    boxed_marker = r"\boxed{"
    return {
        "total_rows": len(rows),
        "unique_ids": len(set(r["id"] for r in rows)),
        "empty_responses": sum(1 for r in rows if not r["response"]),
        "has_boxed": sum(1 for r in rows if boxed_marker in r["response"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Build a Kaggle CSV from results JSONL.")
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.results)]
    out_path = pathlib.Path(args.out)
    write_csv(rows, out_path)

    stats = sanity_check(out_path)
    print(f"Saved {stats['total_rows']} rows to {out_path}")
    print(f"  Unique ids       : {stats['unique_ids']}")
    print(f"  Empty responses  : {stats['empty_responses']}")
    print(f"  Has boxed marker : {stats['has_boxed']}")


if __name__ == "__main__":
    main()
