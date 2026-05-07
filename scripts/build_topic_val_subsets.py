"""Generate topic-specific val_indices subsets from val_225.

For each topic detected by `cse151b_comp.topics.detect_topic`, write a
val_indices_<topic>.json file that the SC CLI can consume via --val.
This lets us test Run J ablations on the questions a branch is meant
to help, instead of averaging out the signal across full val_225.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_topic_val_subsets.py
"""

from __future__ import annotations

import json
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from cse151b_comp.topics import detect_topic  # noqa: E402


def main() -> None:
    public = {r["id"]: r for r in (json.loads(line) for line in open("data/public.jsonl"))}
    val = json.loads(pathlib.Path("data/val_indices.json").read_text())
    val_ids = set(val["val_ids"])

    # Bucket val ids by topic. MCQ stays in MCQ pool (no per-topic split for MCQ).
    by_topic: dict[str, list[int]] = {}
    n_mcq = 0
    for vid in sorted(val_ids):
        item = public[vid]
        if item.get("options"):
            n_mcq += 1
            by_topic.setdefault("mcq", []).append(vid)
            continue
        topic = detect_topic(item["question"])
        by_topic.setdefault(topic, []).append(vid)

    out_dir = pathlib.Path("data/val_subsets")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== val_225 topic distribution ===\n")
    print(f"{'Topic':22} {'count':>6}")
    print("-" * 32)
    for topic, ids in sorted(by_topic.items(), key=lambda kv: -len(kv[1])):
        print(f"{topic:22} {len(ids):>6}")
        # Same shape as val_indices.json — only val_ids field is needed
        # by self_consistency.py's --val loader.
        out = {"val_ids": ids, "topic": topic, "n": len(ids)}
        path = out_dir / f"val_{topic}.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"   → {path}")
    print()
    print(f"Total: {sum(len(v) for v in by_topic.values())} (val_225)")


if __name__ == "__main__":
    main()
