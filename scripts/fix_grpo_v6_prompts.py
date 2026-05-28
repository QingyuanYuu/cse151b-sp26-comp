"""Migrate grpo_train_extended_v6.jsonl to use Run F system + user prompts.

CRITICAL FIX: SFT was trained with build_prompt_runf (RUNF_SYSTEM_PROMPT_MCQ for
MCQ, RUNF_SYSTEM_PROMPT_FREE for free-response). The previous GRPO builder used
a generic simplified system prompt that doesn't match SFT.

Effect of mismatch: at GRPO rollout time, the model sees an unfamiliar system
prompt and regresses toward base behavior — losing all format rules learned in
SFT (single \\boxed{} for multi-part, symbolic preservation, comma-separated
sub-answers, MCQ letter-only output, etc.).

This is the same bug class that hit v2/v3.5 evaluation before (10-20pp drop).

Fix: re-construct each row's prompt[0] (system) and prompt[1] (user) using the
same build_prompt_runf as SFT data. The answer/options/source fields are
unchanged.

Input:  data/grpo_train_extended_v6.jsonl  (old, with generic prompt)
Output: data/grpo_train_extended_v6.jsonl  (overwritten with Run F prompts)
        Also copied to runpod_h100/data/
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from cse151b_comp.prompts import build_prompt_runf

GRPO_FILE = REPO / "data" / "grpo_train_extended_v6.jsonl"
RUNPOD_COPY = REPO / "runpod_h100" / "data" / "grpo_train_extended_v6.jsonl"


def main():
    rows = [json.loads(l) for l in open(GRPO_FILE)]
    print(f"Loaded {len(rows)} rows from {GRPO_FILE}")

    # Track what we change
    n_mcq = 0
    n_free = 0
    old_sys = rows[0]["prompt"][0]["content"][:80]
    print(f"OLD system prompt (preview): {old_sys}...")

    for r in rows:
        opts = json.loads(r["options"])
        # The current user message is the raw question (no options appended).
        # We need it for re-building.
        original_question = r["prompt"][1]["content"]
        # If MCQ, the previous question text might have or not have options inline.
        # Use the bare question from prompts.py logic — build_prompt_runf will append
        # options properly. To get a clean question without prior appended options,
        # we strip any "Options:" suffix that an earlier build may have added.
        bare_q = original_question.split("\n\nOptions:\n")[0].strip()

        system, user = build_prompt_runf(bare_q, opts if opts else None)
        r["prompt"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if opts:
            n_mcq += 1
        else:
            n_free += 1

    new_sys_mcq = next((r["prompt"][0]["content"] for r in rows if json.loads(r["options"])), "(none)")
    new_sys_free = next((r["prompt"][0]["content"] for r in rows if not json.loads(r["options"])), "(none)")

    with open(GRPO_FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Also copy to runpod_h100/
    RUNPOD_COPY.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNPOD_COPY, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nMCQ rows: {n_mcq}  |  Free-response rows: {n_free}")
    print(f"\nNEW system prompt for MCQ (preview):\n  {new_sys_mcq[:200]}...")
    print(f"\nNEW system prompt for free-response (preview):\n  {new_sys_free[:200]}...")
    print(f"\nWrote: {GRPO_FILE}")
    print(f"Wrote: {RUNPOD_COPY}")


if __name__ == "__main__":
    main()
