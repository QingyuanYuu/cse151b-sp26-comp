"""Tests for eval_harness.py — split + compare."""
from __future__ import annotations

import json
import pathlib

import pytest

from cse151b_comp.eval_harness import compare, make_split, question_type


def test_question_type_mcq() -> None:
    assert question_type({"options": ["a", "b"], "answer": "A"}) == "mcq"


def test_question_type_freeform_single() -> None:
    assert question_type({"answer": ["42"]}) == "free_single"


def test_question_type_freeform_multi() -> None:
    assert question_type({"answer": ["41", "35", "16"]}) == "free_multi"


def test_make_split_writes_file_and_is_stratified(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data.jsonl"
    rows = []
    for i in range(50):
        rows.append({"id": i, "question": f"q{i}", "answer": "A", "options": ["a", "b"]})
    for i in range(50, 100):
        rows.append({"id": i, "question": f"q{i}", "answer": [str(i)]})
    data.write_text("\n".join(json.dumps(r) for r in rows))

    out = tmp_path / "val_indices.json"
    info = make_split(data, out, val_fraction=0.2, seed=42)
    assert info["n_total"] == 100
    assert info["n_val"] == 20  # 10 mcq + 10 free_single
    assert info["n_train"] == 80
    assert sorted(info["val_ids"] + info["train_ids"]) == list(range(100))

    # Stratification: roughly 20% of each type in val
    saved = json.loads(out.read_text())
    val_ids = set(saved["val_ids"])
    n_mcq_val = sum(1 for i in val_ids if i < 50)
    n_free_val = sum(1 for i in val_ids if i >= 50)
    assert n_mcq_val == 10
    assert n_free_val == 10


def test_make_split_seed_reproducible(tmp_path: pathlib.Path) -> None:
    data = tmp_path / "data.jsonl"
    rows = [{"id": i, "question": f"q{i}", "answer": "A", "options": ["a", "b"]} for i in range(20)]
    data.write_text("\n".join(json.dumps(r) for r in rows))

    o1 = tmp_path / "v1.json"
    o2 = tmp_path / "v2.json"
    make_split(data, o1, val_fraction=0.2, seed=42)
    make_split(data, o2, val_fraction=0.2, seed=42)
    assert o1.read_text() == o2.read_text()


def test_compare_basic() -> None:
    a = [
        {"id": 1, "correct": True},
        {"id": 2, "correct": True},
        {"id": 3, "correct": False},
        {"id": 4, "correct": False},
    ]
    b = [
        {"id": 1, "correct": True},   # both right
        {"id": 2, "correct": False},  # regression
        {"id": 3, "correct": True},   # gain
        {"id": 4, "correct": False},  # both wrong
    ]
    diff = compare(a, b)
    assert diff["both_right"] == 1
    assert diff["only_baseline_right (regression)"] == 1
    assert diff["only_candidate_right (gain)"] == 1
    assert diff["both_wrong"] == 1
    assert 2 in diff["regression_ids"]
    assert 3 in diff["gain_ids"]
