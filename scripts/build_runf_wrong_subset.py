"""Build a subset jsonl of Run F wrong questions on public 1126.

Reads results/runf_k1_public.jsonl, filters questions where correct=False,
pulls full question/options/answer fields from data/public.jsonl,
writes data/runf_wrong_subset.jsonl.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_runf_wrong_subset.py
"""

from __future__ import annotations

import json
import pathlib

RUNF = "results/runf_k1_public.jsonl"
PUBLIC = "data/public.jsonl"
OUTPUT = "data/runf_wrong_subset.jsonl"


def main() -> None:
    runf = {json.loads(line)["id"]: json.loads(line) for line in open(RUNF)}
    public = {json.loads(line)["id"]: json.loads(line) for line in open(PUBLIC)}

    wrong_ids = sorted(qid for qid, r in runf.items() if not r.get("correct"))
    print(f"Run F wrong: {len(wrong_ids)}/{len(runf)} = {100 * len(wrong_ids) / len(runf):.1f}%")

    out_path = pathlib.Path(OUTPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for qid in wrong_ids:
            if qid in public:
                f.write(json.dumps(public[qid], ensure_ascii=False) + "\n")

    print(f"Wrote {len(wrong_ids)} questions → {out_path}")


if __name__ == "__main__":
    main()
