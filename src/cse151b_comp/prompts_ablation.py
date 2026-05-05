"""Prompt variants for the Phase~1 regression ablation.

Phase 0 leaderboard score: 0.575 (starter prompts, MAX_TOKENS=12288).
Phase 1 leaderboard score: 0.494 (-0.081, regression). Local val showed +6.23pp.

Hypothesis: at least one of the Phase 1 rules helps public val but hurts
private leaderboard. We isolate the suspects:

  - v3a = Phase 0 + ONLY the MCQ anti-paren rule.
          Tests whether MCQ-format gain is independent of the other rules.
  - v3b = Phase 1 minus the anti-rounding rule.
          Tests whether forcing 6 sig-figs is what poisons private.
  - v3c = Phase 1 minus the token-budget self-rescue rule.
          Tests whether early-bailout is what poisons private.

All four variants share the same MCQ/free-form switch logic in
:func:`build_prompt_variant`.
"""

from __future__ import annotations

# ─── Common rule fragments ──────────────────────────────────────────────────

_TOKEN_BUDGET_RULE = (
    "If you are running out of reasoning space, IMMEDIATELY output your "
    "best-guess answer inside \\boxed{...} before stopping. Never end "
    "without a final boxed answer."
)

_NUMERIC_PRECISION_RULE = (
    "Do not round numerical answers. Report at least 6 significant figures or "
    "the exact symbolic form. Example: 143.224229 not 143; 2.32625 not 2.33."
)

_MCQ_ANTI_PAREN_RULE = (
    "Output ONLY the letter inside \\boxed{}. Example: \\boxed{C}. "
    "Do NOT write \\boxed{(C)}, \\boxed{C.}, \\boxed{C)}, or "
    "\\boxed{C: ...}. Letter only, no punctuation, no parentheses."
)

_FREEFORM_BASIC_RULES = (
    "Use plain numbers (write 0.5, not '1/2 of pi'), no units, "
    "no 'x = ', no trailing punctuation."
)

_MULTIPART_RULE = (
    "If the problem has K sub-answers (e.g. parts (a) (b) (c), or multiple "
    "[ANS] placeholders), use ONE of these two styles for the FINAL line:\n"
    "  (1) PREFERRED — single boxed, comma-separated:\n"
    "      \\boxed{41, 35, 16}\n"
    "  (2) Multiple boxed blocks separated ONLY by whitespace/commas:\n"
    "      \\boxed{41} \\boxed{35} \\boxed{16}\n"
    "      DO NOT put labels like '(a)', '(b)', words, or sentences "
    "BETWEEN boxed blocks — that BREAKS the parser.\n"
    "Do NOT mix the two styles in one response."
)

_ANS_PLACEHOLDER_RULE = (
    "If the question contains [ANS] placeholders, replace each with your "
    "boxed answer in order. The very last line of your response must be a "
    "\\boxed{...}."
)


# ─── Variant builders ───────────────────────────────────────────────────────

def _wrap_freeform(extra_rules: list[str]) -> str:
    base = "You are an expert mathematician. Solve the problem step-by-step.\n\n"
    body = ""
    if _MULTIPART_RULE in extra_rules:
        body += _MULTIPART_RULE + "\n\n"
        body += _ANS_PLACEHOLDER_RULE + "\n\n"
    elif _ANS_PLACEHOLDER_RULE in extra_rules:
        body += _ANS_PLACEHOLDER_RULE + "\n\n"
    body += "Output rules for the FINAL answer:\n"
    body += "- " + _FREEFORM_BASIC_RULES + "\n"
    for rule in extra_rules:
        if rule in (_MULTIPART_RULE, _ANS_PLACEHOLDER_RULE):
            continue
        body += "- " + rule + "\n"
    return base + body.rstrip()


def _wrap_mcq(extra_mcq_rule: str | None) -> str:
    base = (
        "You are an expert mathematician. "
        "Read the problem and the answer choices below, then select the "
        "single best answer.\n\n"
    )
    if extra_mcq_rule:
        return base + "Output rules for the FINAL answer:\n- " + extra_mcq_rule
    # Phase 0 baseline MCQ: just the simple instruction.
    return base + (
        "Output ONLY the letter of your chosen option inside "
        "\\boxed{}, e.g. \\boxed{C}."
    )


# Variant definitions. Each is a (system_math, system_mcq) pair.
_VARIANTS: dict[str, tuple[str, str]] = {

    # Phase 0 baseline (starter prompts).
    "phase0": (
        ("You are an expert mathematician. Solve the problem step-by-step. "
         "Put your final answer inside \\boxed{}. "
         "If the problem has multiple sub-answers, separate them by commas inside a "
         "single \\boxed{}, e.g. \\boxed{3, 7}."),
        ("You are an expert mathematician. "
         "Read the problem and the answer choices below, then select the single best "
         "answer. Output ONLY the letter of your chosen option inside \\boxed{}, "
         "e.g. \\boxed{C}."),
    ),

    # v3a: Phase 0 free-form + Phase 1 MCQ anti-paren only.
    "v3a": (
        ("You are an expert mathematician. Solve the problem step-by-step. "
         "Put your final answer inside \\boxed{}. "
         "If the problem has multiple sub-answers, separate them by commas inside a "
         "single \\boxed{}, e.g. \\boxed{3, 7}."),
        _wrap_mcq(_MCQ_ANTI_PAREN_RULE),
    ),

    # v3b: Phase 1 minus anti-rounding rule.
    "v3b": (
        _wrap_freeform([_MULTIPART_RULE, _TOKEN_BUDGET_RULE]),
        _wrap_mcq(_MCQ_ANTI_PAREN_RULE + "\n- " + _TOKEN_BUDGET_RULE),
    ),

    # v3c: Phase 1 minus token-budget self-rescue rule.
    "v3c": (
        _wrap_freeform([_MULTIPART_RULE, _NUMERIC_PRECISION_RULE]),
        _wrap_mcq(_MCQ_ANTI_PAREN_RULE),
    ),
}


def build_prompt_variant(
    variant: str, question: str, options: list[str] | None
) -> tuple[str, str]:
    """Return (system, user) for the given variant."""
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Choices: {sorted(_VARIANTS)}")
    system_math, system_mcq = _VARIANTS[variant]
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return system_mcq, f"{question}\n\nOptions:\n{opts_text}"
    return system_math, question


def list_variants() -> list[str]:
    return sorted(_VARIANTS)
