"""Tests for prepare_sft_data.py.

Covers selection logic against a stub Judger and the wrapper end-to-end
on a tiny in-memory pool.
"""

from __future__ import annotations

import json

from cse151b_comp.prepare_sft_data import (
    build_sft_row,
    prepare,
    select_winning_response,
)


class _StubJudger:
    """Score by string equality of the response's last `\\boxed{...}` to gold.

    This bypasses the real Judger so the test runs fast and is deterministic.
    The real `Judger` is exercised by the merged test_evaluate.py suite.
    """

    def __init__(self, correct_responses: set[str]):
        self._correct = correct_responses

    def auto_judge(self, pred: str, gold, options) -> bool:
        return pred in self._correct


def _patch_score(monkeypatch, correct_responses: set[str]):
    """Force ``score_response`` to return True only for known-correct responses."""

    def fake_score(response, gold, options, judger):
        return response in correct_responses

    import cse151b_comp.prepare_sft_data as mod

    monkeypatch.setattr(mod, "score_response", fake_score)


# ─── select_winning_response ────────────────────────────────────────────────


def test_all_wrong_returns_none(monkeypatch):
    _patch_score(monkeypatch, correct_responses=set())
    pool_row = {
        "all_responses": ["wrong A", "wrong B", "wrong C"],
        "winning_response": "wrong A",
    }
    source_row = {"question": "q", "answer": "42", "options": None}
    assert select_winning_response(pool_row, source_row, judger=None) is None


def test_vote_winner_correct_is_picked(monkeypatch):
    _patch_score(monkeypatch, correct_responses={"vote winner"})
    pool_row = {
        "all_responses": ["wrong", "vote winner", "wrong"],
        "winning_response": "vote winner",
    }
    source_row = {"question": "q", "answer": "42", "options": None}
    target, idx, n = select_winning_response(pool_row, source_row, judger=None)
    assert target == "vote winner"
    assert idx == 1
    assert n == 1


def test_vote_winner_wrong_picks_longest_correct(monkeypatch):
    _patch_score(monkeypatch, correct_responses={"short ok", "longer correct response"})
    pool_row = {
        "all_responses": ["short ok", "wrong but voted", "longer correct response"],
        "winning_response": "wrong but voted",
    }
    source_row = {"question": "q", "answer": "42", "options": None}
    target, idx, n = select_winning_response(pool_row, source_row, judger=None)
    assert target == "longer correct response"
    assert idx == 2
    assert n == 2


def test_winning_text_not_in_responses_falls_back(monkeypatch):
    """If winning_response string isn't in all_responses (rare, but possible),
    we should still pick a correct one.
    """
    _patch_score(monkeypatch, correct_responses={"correct one"})
    pool_row = {
        "all_responses": ["wrong", "correct one"],
        "winning_response": "some-other-string-not-in-list",
    }
    source_row = {"question": "q", "answer": "42", "options": None}
    target, idx, n = select_winning_response(pool_row, source_row, judger=None)
    assert target == "correct one"
    assert idx == 1
    assert n == 1


# ─── build_sft_row ──────────────────────────────────────────────────────────


def test_build_sft_row_includes_required_keys():
    pool_row = {"id": 7, "question_type": "free_single", "K": 32}
    source_row = {"question": "Compute 1+1.", "options": None, "answer": 2}
    row = build_sft_row(
        pool_row=pool_row,
        source_row=source_row,
        target_response="<think>obvious</think>\n\\boxed{2}",
        n_correct=12,
    )
    assert row["id"] == 7
    assert row["question_type"] == "free_single"
    assert row["target_response"].endswith("\\boxed{2}")
    assert row["n_correct"] == 12
    assert row["K"] == 32
    assert "1+1" in row["user_prompt"]
    assert "expert mathematician" in row["system_prompt"].lower()


def test_build_sft_row_routes_mcq_prompt():
    pool_row = {"id": 8, "question_type": "mc", "K": 32}
    source_row = {
        "question": "What is 2+2?",
        "options": ["3", "4", "5"],
        "answer": "B",
    }
    row = build_sft_row(
        pool_row=pool_row,
        source_row=source_row,
        target_response="\\boxed{B}",
        n_correct=30,
    )
    # MCQ prompt should mention the boxed letter form
    assert (
        "\\boxed{X}" in row["system_prompt"]
        or "\\boxed{C}" in row["system_prompt"]
        or "letter" in row["system_prompt"].lower()
    )
    # User prompt should include the options block
    assert "A. 3" in row["user_prompt"]
    assert "B. 4" in row["user_prompt"]


# ─── prepare end-to-end ─────────────────────────────────────────────────────


def test_prepare_excludes_val_and_keeps_correct(monkeypatch, tmp_path):
    correct_set = {"good response 1", "good response 3"}
    _patch_score(monkeypatch, correct_responses=correct_set)

    source_path = tmp_path / "public.jsonl"
    source_path.write_text(
        json.dumps({"id": 1, "question": "q1", "answer": "42", "options": None})
        + "\n"
        + json.dumps({"id": 2, "question": "q2", "answer": "B", "options": ["A", "B", "C"]})
        + "\n"
        + json.dumps({"id": 3, "question": "q3", "answer": [1, 2], "options": None})
        + "\n"
    )

    pool_path = tmp_path / "pool.jsonl"
    pool_path.write_text(
        json.dumps(
            {
                "id": 1,
                "question_type": "free_single",
                "K": 4,
                "all_responses": ["bad 1", "good response 1", "bad 2", "bad 3"],
                "winning_response": "good response 1",
            }
        )
        + "\n"
        + json.dumps(
            {  # id=2 is in val → skipped
                "id": 2,
                "question_type": "mc",
                "K": 4,
                "all_responses": ["bad", "bad", "bad", "bad"],
                "winning_response": "bad",
            }
        )
        + "\n"
        + json.dumps(
            {  # id=3 has no correct samples → skipped
                "id": 3,
                "question_type": "free_multi",
                "K": 4,
                "all_responses": ["good response 3", "bad", "bad", "bad"],
                "winning_response": "good response 3",
            }
        )
        + "\n"
    )

    val_path = tmp_path / "val.json"
    val_path.write_text(json.dumps({"val_ids": [2]}))

    out_path = tmp_path / "sft.jsonl"
    stats = prepare(
        pool_path=pool_path,
        source_path=source_path,
        val_path=val_path,
        out_path=out_path,
    )

    assert stats["n_pool"] == 3
    assert stats["n_skipped_val"] == 1
    assert stats["n_kept"] == 2  # ids 1 and 3 both have a correct sample

    rows = [json.loads(line) for line in open(out_path)]
    assert len(rows) == 2
    ids = {r["id"] for r in rows}
    assert ids == {1, 3}


def test_prepare_runs_without_val_filter(monkeypatch, tmp_path):
    _patch_score(monkeypatch, correct_responses={"ok"})
    source_path = tmp_path / "public.jsonl"
    source_path.write_text(json.dumps({"id": 1, "question": "q", "answer": "1", "options": None}) + "\n")
    pool_path = tmp_path / "pool.jsonl"
    pool_path.write_text(
        json.dumps(
            {
                "id": 1,
                "question_type": "free_single",
                "K": 2,
                "all_responses": ["ok", "ok"],
                "winning_response": "ok",
            }
        )
        + "\n"
    )
    out_path = tmp_path / "sft.jsonl"
    stats = prepare(pool_path, source_path, val_path=None, out_path=out_path)
    assert stats["n_kept"] == 1
    assert stats["n_skipped_val"] == 0
