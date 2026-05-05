"""Filter a self-consistency K-sample pool into LoRA SFT training pairs.

Pipeline (Day 1 of PLAN_6DAY):
1. Read ``results/sc_v6_k32_public.jsonl`` (output of ``cse151b-sc`` with
   ``--prompt current --k 32`` over ``data/public.jsonl``).
2. Read ``data/public.jsonl`` to recover ``question`` / ``options`` /
   ``answer`` per id (the SC output keeps ``answer`` if the input had
   one, but not ``question``).
3. Read ``data/val_indices.json`` to exclude the 225 val ids.
4. For each remaining question, judge every one of the K responses
   against gold using the course-provided ``Judger``. Select a single
   "winning" response to use as the SFT target, preferring:

   - the vote-winning response if it's correct, otherwise
   - the longest correct response (more reasoning trace = better target),
     otherwise
   - skip the question (no usable target).

5. Format each surviving row as ``{id, system_prompt, user_prompt,
   target_response, question_type, n_correct, K}`` and write to
   ``data/sft_train.jsonl``.

The target's ``response`` is left as-is — the full ``<think>...</think>``
+ trailing ``\\boxed{...}``. We deliberately do NOT use the bare gold
because the model would learn to skip the boxed wrapper.

CLI::

    cse151b-prepare-sft \\
        --pool results/sc_v6_k32_public.jsonl \\
        --source data/public.jsonl \\
        --val data/val_indices.json \\
        --output data/sft_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

from cse151b_comp.evaluate import score_response
from cse151b_comp.prompts import build_prompt

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from judger import Judger  # noqa: E402


def _index_source(path: pathlib.Path) -> dict[int, dict]:
    """Map ``id`` → row from a public-style JSONL with question/options/answer."""
    out: dict[int, dict] = {}
    for line in open(path):
        row = json.loads(line)
        out[row["id"]] = row
    return out


def _load_val_ids(path: pathlib.Path | None) -> set[int]:
    if path is None:
        return set()
    return set(json.loads(path.read_text())["val_ids"])


def select_winning_response(
    pool_row: dict,
    source_row: dict,
    judger: Judger,
) -> tuple[str, int, int] | None:
    """Pick the SFT target response from ``pool_row['all_responses']``.

    Returns ``(target_response, winning_index, n_correct)``, or
    ``None`` if no response is correct (skip this question).

    Preference order:
      1. Vote-winner response if it's correct.
      2. Longest correct response among the K.
    """
    responses: list[str] = pool_row["all_responses"]
    gold = source_row["answer"]
    options = source_row.get("options")

    correct_mask = [score_response(r, gold, options, judger) for r in responses]
    n_correct = sum(correct_mask)
    if n_correct == 0:
        return None

    # Look at the SC vote-winner (already chosen by voting.py at SC time).
    # SC stores the winning response under ``winning_response`` and the
    # index that produced it isn't recorded; we just check that exact
    # string against the responses list.
    winner_text = pool_row.get("winning_response", "")
    if winner_text:
        try:
            winner_idx = responses.index(winner_text)
            if correct_mask[winner_idx]:
                return winner_text, winner_idx, n_correct
        except ValueError:
            pass

    # Fall back to longest correct response.
    correct_indices = [i for i, c in enumerate(correct_mask) if c]
    longest_idx = max(correct_indices, key=lambda i: len(responses[i]))
    return responses[longest_idx], longest_idx, n_correct


def build_sft_row(
    pool_row: dict,
    source_row: dict,
    target_response: str,
    n_correct: int,
) -> dict:
    """Assemble one SFT training example."""
    system, user = build_prompt(source_row["question"], source_row.get("options"))
    return {
        "id": pool_row["id"],
        "question_type": pool_row.get("question_type", "unknown"),
        "system_prompt": system,
        "user_prompt": user,
        "target_response": target_response,
        "n_correct": n_correct,
        "K": pool_row.get("K", len(pool_row.get("all_responses", []))),
    }


def prepare(
    pool_path: pathlib.Path,
    source_path: pathlib.Path,
    val_path: pathlib.Path | None,
    out_path: pathlib.Path,
) -> dict:
    """Run the full pipeline. Return a stats dict for the caller to print."""
    source = _index_source(source_path)
    val_ids = _load_val_ids(val_path)
    judger = Judger(strict_extract=False)

    n_pool = 0
    n_skipped_val = 0
    n_no_correct = 0
    type_kept = Counter()
    type_total = Counter()
    pass_at_k = Counter()  # bucket → count of questions with that #correct

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for line in open(pool_path):
            pool_row = json.loads(line)
            n_pool += 1
            qid = pool_row["id"]
            qtype = pool_row.get("question_type", "unknown")
            type_total[qtype] += 1

            if qid in val_ids:
                n_skipped_val += 1
                continue

            source_row = source.get(qid)
            if source_row is None:
                # Pool has an id not in source — shouldn't happen but be safe.
                continue

            picked = select_winning_response(pool_row, source_row, judger)
            if picked is None:
                n_no_correct += 1
                pass_at_k["0"] += 1
                continue

            target_response, _, n_correct = picked
            sft_row = build_sft_row(pool_row, source_row, target_response, n_correct)
            f.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
            type_kept[qtype] += 1

            # Bucket pass@K density for the report.
            k = pool_row.get("K", len(pool_row.get("all_responses", [])))
            frac = n_correct / max(k, 1)
            if frac >= 0.75:
                pass_at_k["≥75%"] += 1
            elif frac >= 0.50:
                pass_at_k["50-75%"] += 1
            elif frac >= 0.25:
                pass_at_k["25-50%"] += 1
            else:
                pass_at_k["1-25%"] += 1

    n_kept = sum(type_kept.values())
    return {
        "n_pool": n_pool,
        "n_skipped_val": n_skipped_val,
        "n_no_correct": n_no_correct,
        "n_kept": n_kept,
        "type_total": dict(type_total),
        "type_kept": dict(type_kept),
        "pass_at_k_buckets": dict(pass_at_k),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Filter SC K-sample pool → SFT training set.")
    p.add_argument("--pool", required=True, help="SC output JSONL with all_responses.")
    p.add_argument("--source", required=True, help="Original public.jsonl with question/options/answer.")
    p.add_argument("--val", default=None, help="val_indices.json to exclude val_ids (optional).")
    p.add_argument("--output", required=True, help="Output SFT JSONL.")
    args = p.parse_args()

    stats = prepare(
        pool_path=pathlib.Path(args.pool),
        source_path=pathlib.Path(args.source),
        val_path=pathlib.Path(args.val) if args.val else None,
        out_path=pathlib.Path(args.output),
    )

    print(f"[prepare-sft] pool rows : {stats['n_pool']}")
    print(f"[prepare-sft] skipped val: {stats['n_skipped_val']}")
    print(f"[prepare-sft] all-K-wrong : {stats['n_no_correct']}")
    print(f"[prepare-sft] kept       : {stats['n_kept']}")
    print(f"[prepare-sft] by type    : kept={stats['type_kept']}  total={stats['type_total']}")
    print(f"[prepare-sft] pass@K      : {stats['pass_at_k_buckets']}")
    print(f"[prepare-sft] wrote      : {args.output}")


if __name__ == "__main__":
    main()
