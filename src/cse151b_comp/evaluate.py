"""Re-judge a results JSONL using the course-provided ``Judger``.

Reads a JSONL where each row has at least ``id``, ``response``, and (for
public-set rows) ``answer``/``options``. Adds/overwrites ``correct`` and
``is_mcq`` fields, prints accuracy, and optionally writes the rejudged file
to a new path.

Run via:
    cse151b-evaluate --results results/baseline_public_v1.jsonl

Or just import :func:`score_response` for one-off use.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Iterable

from cse151b_comp.extract import extract_letter

# The judger and utils modules live at the repo root, not in our package.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from judger import Judger  # noqa: E402  (after sys.path patch)


def score_response(
    response: str,
    gold: Any,
    options: list[str] | None,
    judger: Judger,
) -> bool:
    """Return True iff ``response`` is judged correct against ``gold``.

    - MCQ: compare extracted letter to gold letter.
    - Free-form: delegate to ``judger.auto_judge``.
    """
    is_mcq = bool(options)
    if is_mcq:
        return extract_letter(response) == str(gold).strip().upper()
    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return bool(
            judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        )
    except Exception:
        return False


def evaluate_rows(rows: Iterable[dict], judger: Judger | None = None) -> list[dict]:
    """Score every row in-place, return rows with ``correct`` and ``is_mcq``.

    Skips rows that have no ``answer`` field (private/test set).
    """
    judger = judger or Judger(strict_extract=False)
    out: list[dict] = []
    for r in rows:
        if "answer" not in r and "gold" not in r:
            # Private set — can't score, just pass through.
            out.append(r)
            continue
        gold = r.get("gold", r.get("answer"))
        options = r.get("options")
        is_mcq = bool(options)
        correct = score_response(r["response"], gold, options, judger)
        out.append({**r, "is_mcq": is_mcq, "gold": gold, "correct": correct})
    return out


def summarize(rows: list[dict]) -> dict:
    """Return overall, MCQ, and free-form accuracy + format-failure rate."""
    scored = [r for r in rows if "correct" in r]
    if not scored:
        return {"n": 0}
    mcq = [r for r in scored if r["is_mcq"]]
    free = [r for r in scored if not r["is_mcq"]]
    return {
        "n": len(scored),
        "overall_acc": sum(r["correct"] for r in scored) / len(scored),
        "mcq_n": len(mcq),
        "mcq_acc": (sum(r["correct"] for r in mcq) / len(mcq)) if mcq else 0.0,
        "free_n": len(free),
        "free_acc": (sum(r["correct"] for r in free) / len(free)) if free else 0.0,
        "no_box_rate": sum(
            1 for r in scored if "\\boxed{" not in r["response"]
        )
        / len(scored),
    }


def _print_summary(s: dict) -> None:
    if s.get("n", 0) == 0:
        print("No scored rows (private/test set?).")
        return
    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    pct = lambda x: f"{x*100:.2f}%"  # noqa: E731
    print(f"  Overall    : {s['overall_acc']:.4f}  ({pct(s['overall_acc'])})  n={s['n']}")
    print(f"  MCQ        : {s['mcq_acc']:.4f}  ({pct(s['mcq_acc'])})  n={s['mcq_n']}")
    print(f"  Free-form  : {s['free_acc']:.4f}  ({pct(s['free_acc'])})  n={s['free_n']}")
    print(f"  No-box rate: {pct(s['no_box_rate'])}  (responses without \\boxed{{}})")
    print("=" * 50)


def main() -> None:
    p = argparse.ArgumentParser(description="Re-judge a results JSONL.")
    p.add_argument("--results", required=True, help="Input JSONL with response + answer/gold.")
    p.add_argument(
        "--out",
        default=None,
        help="Optional: write rejudged JSONL here. If unset, only prints summary.",
    )
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.results)]
    judger = Judger(strict_extract=False)
    scored = evaluate_rows(rows, judger)
    _print_summary(summarize(scored))

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in scored:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote rejudged file to {out_path}")


if __name__ == "__main__":
    main()
