"""Failure-mode breakdown for milestone-report-grade error analysis.

Given a scored JSONL (after :mod:`evaluate`), produces:

- accuracy by question type (mcq / free_single / free_multi)
- accuracy by question-length bucket
- accuracy by topic (regex-based heuristic)
- failure modes: no-box / wrong-shape / numeric-but-wrong / max-tokens-truncated
- top-K confident-but-wrong responses (high boxing rate but graded wrong)

Run via:
    cse151b-analyze --results results/baseline_public_v1.jsonl --report reports/baseline.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter
from dataclasses import dataclass, field

from cse151b_comp.extract import extract_all_final_boxed, extract_letter


# ─── Topic tagger (regex heuristic, not authoritative) ──────────────────────

_TOPIC_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("calculus", re.compile(r"\\int|\\frac\{d|integral|derivative|\\lim", re.IGNORECASE)),
    ("linalg", re.compile(r"matrix|matrices|eigen|determinant|\\det|vector\s+space", re.IGNORECASE)),
    ("probability", re.compile(r"\\Pr|\bP\(|probab|expected\s+value|combin|permut", re.IGNORECASE)),
    ("ode", re.compile(r"differential\s+equation|\\frac\{d.*?\}\{dt\}|y'|y\s*'\s*=", re.IGNORECASE)),
    ("complex", re.compile(r"\\Re|\\Im|complex\s+number|residue|analytic\s+function", re.IGNORECASE)),
    ("series", re.compile(r"\\sum|series|sequence|converge", re.IGNORECASE)),
    ("geometry", re.compile(r"triangle|circle|polygon|angle|\\sin|\\cos|\\tan", re.IGNORECASE)),
]


def tag_topic(question: str) -> str:
    for name, rx in _TOPIC_PATTERNS:
        if rx.search(question):
            return name
    return "other"


# ─── Length buckets ─────────────────────────────────────────────────────────


def length_bucket(question: str) -> str:
    n = len(question)
    if n < 150:
        return "short(<150)"
    if n < 500:
        return "medium(150-500)"
    if n < 1500:
        return "long(500-1500)"
    return "vlong(>=1500)"


# ─── Question type ──────────────────────────────────────────────────────────


def question_type(row: dict) -> str:
    if row.get("options") or row.get("is_mcq"):
        return "mcq"
    gold = row.get("gold", row.get("answer"))
    if isinstance(gold, list) and len(gold) > 1:
        return "free_multi"
    return "free_single"


# ─── Failure-mode classification ────────────────────────────────────────────


@dataclass
class FailureBreakdown:
    no_box: int = 0  # response has zero \boxed{}
    wrong_shape: int = 0  # multi-part question, wrong number of boxed
    numeric_but_wrong: int = 0  # boxed found, judged wrong
    truncated: int = 0  # response > truncation threshold (likely max_tokens)
    other: int = 0
    by_topic_wrong: Counter = field(default_factory=Counter)
    by_length_wrong: Counter = field(default_factory=Counter)
    by_type_wrong: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "no_box": self.no_box,
            "wrong_shape": self.wrong_shape,
            "numeric_but_wrong": self.numeric_but_wrong,
            "truncated": self.truncated,
            "other": self.other,
            "by_topic_wrong": dict(self.by_topic_wrong),
            "by_length_wrong": dict(self.by_length_wrong),
            "by_type_wrong": dict(self.by_type_wrong),
        }


def classify_failure(row: dict, truncation_chars: int = 30000) -> str:
    """Classify a *wrong* row's failure mode. Caller filters for correct=False."""
    response = row.get("response", "")
    if "\\boxed{" not in response:
        # Distinguish max-tokens truncation vs reasoning failure.
        if len(response) >= truncation_chars:
            return "truncated"
        return "no_box"
    qtype = question_type(row)
    boxes = extract_all_final_boxed(response)
    if qtype == "free_multi":
        gold = row.get("gold", row.get("answer"))
        # gold is a list of K values, response should yield K elements
        # whether by K boxed or one comma-separated boxed
        if len(boxes) == 1:
            parts = [p.strip() for p in boxes[0].split(",")]
            if len(parts) != len(gold):
                return "wrong_shape"
        elif len(boxes) != len(gold):
            return "wrong_shape"
    return "numeric_but_wrong"


# ─── Aggregation ────────────────────────────────────────────────────────────


def analyze(rows: list[dict]) -> dict:
    """Return aggregate stats + failure breakdown."""
    by_type = Counter()
    by_topic = Counter()
    by_length = Counter()
    correct_by_type = Counter()
    correct_by_topic = Counter()
    correct_by_length = Counter()

    fb = FailureBreakdown()
    no_box_lengths = []
    confident_wrong: list[dict] = []  # boxed but graded wrong

    for r in rows:
        if "correct" not in r:
            continue
        qtype = question_type(r)
        topic = tag_topic(r.get("question", ""))
        length = length_bucket(r.get("question", ""))

        by_type[qtype] += 1
        by_topic[topic] += 1
        by_length[length] += 1

        if r["correct"]:
            correct_by_type[qtype] += 1
            correct_by_topic[topic] += 1
            correct_by_length[length] += 1
        else:
            fb.by_type_wrong[qtype] += 1
            fb.by_topic_wrong[topic] += 1
            fb.by_length_wrong[length] += 1
            mode = classify_failure(r)
            setattr(fb, mode, getattr(fb, mode) + 1)
            if mode == "numeric_but_wrong":
                confident_wrong.append(r)

        if "\\boxed{" not in r.get("response", ""):
            no_box_lengths.append(len(r.get("response", "")))

    def acc_table(seen: Counter, correct: Counter) -> dict:
        return {
            k: {
                "n": seen[k],
                "correct": correct.get(k, 0),
                "acc": (correct.get(k, 0) / seen[k]) if seen[k] else 0.0,
            }
            for k in seen
        }

    return {
        "n_total": sum(by_type.values()),
        "n_correct": sum(correct_by_type.values()),
        "overall_acc": (sum(correct_by_type.values()) / sum(by_type.values()))
        if by_type
        else 0.0,
        "by_type": acc_table(by_type, correct_by_type),
        "by_topic": acc_table(by_topic, correct_by_topic),
        "by_length": acc_table(by_length, correct_by_length),
        "failure_breakdown": fb.to_dict(),
        "no_box_response_lengths": {
            "n": len(no_box_lengths),
            "median": sorted(no_box_lengths)[len(no_box_lengths) // 2] if no_box_lengths else 0,
            "max": max(no_box_lengths) if no_box_lengths else 0,
        },
        "top_confident_wrong": [
            {
                "id": r.get("id"),
                "question_type": question_type(r),
                "gold": r.get("gold", r.get("answer")),
                "boxed": extract_all_final_boxed(r["response"])[-1:] or [None],
                "topic": tag_topic(r.get("question", "")),
            }
            for r in confident_wrong[:10]
        ],
    }


# ─── Report rendering ───────────────────────────────────────────────────────


def render_markdown(stats: dict, src_path: str = "") -> str:
    lines = ["# Error Analysis Report", ""]
    if src_path:
        lines.append(f"_Source: `{src_path}`_")
        lines.append("")
    lines.append(f"**Overall accuracy**: {stats['overall_acc']*100:.2f}% "
                 f"({stats['n_correct']}/{stats['n_total']})")
    lines.append("")

    def table(title: str, dict_: dict) -> list[str]:
        out = [f"## {title}", "", "| key | n | correct | acc |", "|---|---|---|---|"]
        for k, v in sorted(dict_.items(), key=lambda kv: -kv[1]["n"]):
            out.append(f"| {k} | {v['n']} | {v['correct']} | {v['acc']*100:.2f}% |")
        out.append("")
        return out

    lines += table("By question type", stats["by_type"])
    lines += table("By topic", stats["by_topic"])
    lines += table("By question length", stats["by_length"])

    fb = stats["failure_breakdown"]
    lines.append("## Failure modes")
    lines.append("")
    lines.append("| mode | count |")
    lines.append("|---|---|")
    for k in ["no_box", "truncated", "wrong_shape", "numeric_but_wrong", "other"]:
        lines.append(f"| {k} | {fb.get(k, 0)} |")
    lines.append("")

    nbl = stats["no_box_response_lengths"]
    lines.append(
        f"**No-box response lengths**: n={nbl['n']}, median={nbl['median']}, max={nbl['max']}"
    )
    lines.append("")

    if stats["top_confident_wrong"]:
        lines.append("## Top confident-but-wrong (boxed something, graded wrong)")
        lines.append("")
        for r in stats["top_confident_wrong"]:
            lines.append(
                f"- id={r['id']} type={r['question_type']} topic={r['topic']} "
                f"boxed={r['boxed']!r} gold={r['gold']!r}"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Failure-mode analysis for a results JSONL.")
    p.add_argument("--results", required=True)
    p.add_argument("--report", default=None, help="Write markdown report here.")
    p.add_argument("--json", default=None, help="Write raw stats JSON here.")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.results)]
    stats = analyze(rows)
    md = render_markdown(stats, src_path=args.results)
    print(md)

    if args.report:
        out = pathlib.Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        print(f"\nWrote markdown to {out}")
    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2))
        print(f"Wrote stats JSON to {out}")


if __name__ == "__main__":
    main()
