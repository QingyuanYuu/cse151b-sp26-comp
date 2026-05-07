"""Build topic-balanced eval subsets from public.jsonl (excluding val_ids).

val_225 is stratified by question TYPE (mc/single/multi) but topic
distribution is essentially random — trig is 41% of private but only 5
of 225 in val. That's not enough for per-topic ablation signal.

This script samples N questions per topic from public_train (= public
minus val_ids) so each Run J branch ablation can be measured on a
statistically meaningful topic subset (~50 questions vs val's 5-19).

Output: data/eval_subsets/eval_<topic>.jsonl  (one row per question;
        same schema as public.jsonl, ready to feed `cse151b-sc --input`)

Usage:
    PYTHONPATH=src .venv/bin/python scripts/build_topic_eval_subsets.py \
        [--per-topic 50] [--seed 151]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from cse151b_comp.topics import detect_topic  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--per-topic",
        type=int,
        default=50,
        help="Max questions to sample per topic. Topics with " "fewer available questions get all of them.",
    )
    p.add_argument("--seed", type=int, default=151)
    args = p.parse_args()

    public = [json.loads(line) for line in open("data/public.jsonl")]
    val = json.loads(pathlib.Path("data/val_indices.json").read_text())
    val_ids = set(val["val_ids"])

    # Pool = public minus val (free-form only; MCQ tested separately)
    pool_by_topic: dict[str, list[dict]] = {}
    for r in public:
        if r["id"] in val_ids:
            continue
        if r.get("options"):
            pool_by_topic.setdefault("mcq", []).append(r)
            continue
        topic = detect_topic(r["question"])
        pool_by_topic.setdefault(topic, []).append(r)

    rng = random.Random(args.seed)
    out_dir = pathlib.Path("data/eval_subsets")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Topic-balanced eval subsets (per-topic ≤ {args.per_topic}) ===\n")
    print(f"{'Topic':22} {'available':>10} {'sampled':>9}")
    print("-" * 45)
    summary: list[tuple[str, int, int]] = []
    for topic, rows in sorted(pool_by_topic.items(), key=lambda kv: -len(kv[1])):
        n_avail = len(rows)
        n_take = min(args.per_topic, n_avail)
        sampled = rng.sample(rows, n_take)
        path = out_dir / f"eval_{topic}.jsonl"
        with open(path, "w") as f:
            for r in sampled:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{topic:22} {n_avail:>10} {n_take:>9}")
        summary.append((topic, n_avail, n_take))

    print()
    total_sampled = sum(n for _, _, n in summary)
    print(f"Total sampled: {total_sampled}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
