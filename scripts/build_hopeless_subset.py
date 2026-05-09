"""Build subset jsonl of 254 'hopeless' questions (K=16 effective, 0 correct).

Reads SFT pool to get the IDs that DID get a correct sample, then takes
the complement among public train (excluding val_ids).

Output: data/hopeless_subset.jsonl
"""

from __future__ import annotations

import json
import pathlib


def main() -> None:
    vd = json.load(open("data/val_indices.json"))
    val_ids = set(vd["val_ids"])
    public = {json.loads(line)["id"]: json.loads(line) for line in open("data/public.jsonl")}

    sft_ids = {json.loads(line)["id"] for line in open("data/sft_train_staged.jsonl")}

    # Hopeless = train (not in val) AND not in SFT pool (= no K-sample correct)
    hopeless = [qid for qid in public if qid not in val_ids and qid not in sft_ids]
    print(f"Public train (excl val): {len([qid for qid in public if qid not in val_ids])}")
    print(f"In SFT pool (≥1 correct): {len(sft_ids & set(public))}")
    print(f"Hopeless (0 correct): {len(hopeless)}")

    out_path = pathlib.Path("data/hopeless_subset.jsonl")
    with open(out_path, "w") as f:
        for qid in sorted(hopeless):
            f.write(json.dumps(public[qid], ensure_ascii=False) + "\n")

    print(f"Wrote {len(hopeless)} questions → {out_path}")


if __name__ == "__main__":
    main()
