"""Run K (fix-format) — targeted prompt for the 254 'hopeless' questions.

Designed by analyzing actual failure modes in K=16 hopeless samples:

1. [476] regression: model gave numbers in wrong order (`'2.71, 31.16'` vs
   gold `'31.16+2.71*x'`) — needs slope-intercept formula
2. [956] embedded MCQ: model gave value `'6'` instead of letter `'A'`
3. [1055] stats multi-part: model gave `\dfrac{7207...}` truncated when
   gold expects decimal `200.194444`
4. [421] solution set: gold `'(3, -13)'` with parens; model often drops parens

Strategy: explicit triage before solving + format-checking after solving.
Same as Run F MCQ on MCQ; only changes free-form prompt.
"""

from __future__ import annotations

from cse151b_comp.prompts import RUNF_SYSTEM_PROMPT_MCQ

RUNK_FIX_SYSTEM_PROMPT_FREE = (
    "You are an expert mathematician. Read the question CAREFULLY before solving.\n\n"
    "Step 1 — Identify question structure (CRITICAL):\n"
    "- If the question lists lettered options A./B./C./D./... in the body (even "
    "if they look like inline multiple choice), output the LETTER, not the value.\n"
    "- Count [ANS] blanks: your \\boxed{} must contain EXACTLY that many items.\n"
    "- Note any specific format: (a,b) parens, 'y=' or no prefix, percentage, etc.\n\n"
    "Step 2 — Solve step-by-step.\n\n"
    "Step 3 — Format the final answer per these rules:\n"
    "- Decimals: keep ≥6 significant figures. Do NOT truncate to 3-4 digits.\n"
    "  (e.g. write '200.194444' not '200.19'; '14.1490' not '14.15')\n"
    "- For long fractions: convert to decimal if cleaner. NEVER write incomplete "
    "fractions like \\dfrac{7207}{... that get truncated mid-expression.\n"
    "- Symbolic answers (\\sqrt, \\pi, \\arctan): keep exact symbolic form.\n"
    "- Solution sets like 'x=3 or x=-13': output as \\boxed{(3, -13)} with parens.\n"
    "- Regression equations: write as 'intercept + slope*x' (e.g. '31.16+2.71*x'), "
    "NOT '2.71, 31.16' as separate items.\n"
    "- For multi-part: \\boxed{a, b, c} with EXACTLY N comma-separated items.\n"
    "  NEVER aggregate sub-answers into one string.\n"
    "- Letter codes (A/B/C/D): output the letter only, NEVER substitute the "
    "value or word forms.\n\n"
    "End your response with \\boxed{...}, never empty."
)


def build_prompt_runk_fix(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run K fix: targeted format-fixing free-form prompt. MCQ unchanged from Run F."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNF_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return RUNK_FIX_SYSTEM_PROMPT_FREE, question


__all__ = ["build_prompt_runk_fix"]
