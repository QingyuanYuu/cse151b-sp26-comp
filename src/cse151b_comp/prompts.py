"""System and user prompt templates — v6 (per-type routing).

History:

- Phase 0 (starter): leaderboard 0.575, val 56.44%.
- Phase 1 (full rules + anti-rounding + token-budget rescue): 0.494
  (regression -8.1pp), val 62.67%.
- v4 (Phase 1 minus the two ablation-flagged rules): 0.462 (regression
  -11.3pp). v4 still had "Use plain numbers" + "no 'x = '" which we
  diagnosed as the cause of:
    - True/False → 1/0  (id 44, 319, 705, 785, 623; 7 questions)
    - Letters A/B/C → 1/2/3 (id 60, 101, 119, +26 more; 29 questions)
    - Equation prefix stripping (id 40, 526, 906; ~3 questions)
  These bugs accounted for ~33 % of the v1 → v4 regression.
- **v6 (this file)**: per-type routing, drops the bug-causing rules,
  adds explicit symbolic-form examples to fight the v2 anti-rounding
  symptom (where the model converted symbolic gold to decimal).

Three system prompts now, routed by ``detect_question_type``:

- ``SYSTEM_PROMPT_MCQ``  — for option-bearing questions.
- ``SYSTEM_PROMPT_FREE_SINGLE`` — for free-form with one answer.
- ``SYSTEM_PROMPT_FREE_MULTI`` — for free-form with multiple sub-answers
  (detected by ``[ANS]`` count >= 2 or ``(a)/(b)`` markers).

Why per-type:

- 32 % of private questions are free_single (single answer). They do
  not need to see the multi-part formatting block, saving ~100 tokens
  per such question and reducing prompt-induced reasoning drift.
- free_multi gets a single-style mandate (multiple consecutive boxed)
  rather than a two-style menu, removing model indecision.
- Each type's examples target its own failure modes only.

What's intentionally NOT here:

- No "Use plain numbers" rule (caused True/False → 1/0).
- No "no 'x = '" rule (caused Bug C — equation prefix stripping).
- No "Do not round / 6 sig figs" (Phase 1 anti-rounding, public-overfit).
- No "if running out of room, output best guess" (Phase 1 token-rescue).
- ``free_single`` examples include a ``\\boxed{D = 800 - 50d}`` form
  to show that equation-form answers are valid; this is the
  conservative counterpoint to the v4 "no 'x = '" rule.
"""

from __future__ import annotations

import re


# ─── Per-type rule fragments ───────────────────────────────────────────────

_MCQ_RULE = (
    "Output the answer letter only, in the form: \\boxed{X} where X is one "
    "of A, B, C, D, E, ... . "
    "Do NOT include parentheses, periods, words, \\text{}, or \\textbf{}. "
    "Do NOT output the option content; output the LETTER only."
)

_FREE_SINGLE_RULE = (
    "Put your final answer in \\boxed{...} as a clean mathematical "
    "expression. Do NOT include $ delimiters or full sentences inside the "
    "box.\n"
    "Acceptable answer forms:\n"
    "- integer / decimal: \\boxed{-512},  \\boxed{0.625}\n"
    "- fraction: \\boxed{\\frac{1}{2}}\n"
    "- symbolic: \\boxed{2\\pi},  \\boxed{\\sqrt{101}}\n"
    "- expression / equation (when the question asks for one): "
    "\\boxed{D = 800 - 50d},  \\boxed{y = 5x^4}\n"
    "- text answer (when the question asks for one): \\boxed{Yes},  "
    "\\boxed{Tuesday}\n"
    "Use the form that most directly matches what the question asks for."
)

_FREE_MULTI_RULE = (
    "This problem has multiple sub-answers (e.g. parts (a), (b), (c) or "
    "multiple [ANS] placeholders). Output EACH sub-answer in its own "
    "\\boxed{} block, in the same order as the question.\n"
    "Example: \\boxed{41} \\boxed{35} \\boxed{16}\n"
    "DO NOT put labels like '(a)' between or inside the boxes — that breaks "
    "the parser. DO NOT combine multiple values inside a single box."
)


# ─── Three system prompts ──────────────────────────────────────────────────

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Read the problem and choices, "
    "then select the single best answer.\n\n"
    + _MCQ_RULE
)

SYSTEM_PROMPT_FREE_SINGLE = (
    "You are an expert mathematician. Solve the problem step-by-step.\n\n"
    + _FREE_SINGLE_RULE
)

SYSTEM_PROMPT_FREE_MULTI = (
    "You are an expert mathematician. Solve each sub-question step-by-step.\n\n"
    + _FREE_MULTI_RULE
)

# ─── Backward-compat alias ─────────────────────────────────────────────────
# The notebook + some scripts import ``SYSTEM_PROMPT_MATH``. Keep it as an
# alias for ``SYSTEM_PROMPT_FREE_MULTI`` (the most rule-rich variant), since
# anything that doesn't know question-type routing will see whichever of the
# three is most defensive.
SYSTEM_PROMPT_MATH = SYSTEM_PROMPT_FREE_MULTI


# ─── Question-type detection ───────────────────────────────────────────────

_MULTI_PART_LETTER_RX = re.compile(r"\([a-e]\)")


def detect_question_type(question: str, options: list[str] | None) -> str:
    """Return one of ``"mc"``, ``"free_multi"``, ``"free_single"``.

    Routing rules:

    - ``options`` truthy → ``"mc"``.
    - ``[ANS]`` placeholder appears 2+ times → ``"free_multi"``.
    - ``(a)``…``(e)`` labels appear 2+ distinct → ``"free_multi"``.
    - else → ``"free_single"``.

    The ``(a)``-marker fallback risks false positives on math expressions
    like ``compute (a+b)^2``, but those typically don't repeat with
    different letters, so the ``len(set(...)) >= 2`` guard catches the
    real case.
    """
    if options:
        return "mc"
    if question.count("[ANS]") >= 2:
        return "free_multi"
    distinct_markers = set(m.lower() for m in _MULTI_PART_LETTER_RX.findall(question))
    if len(distinct_markers) >= 2:
        return "free_multi"
    return "free_single"


# ─── Run B: Phase 0 base + targeted anti-pattern rules ────────────────────
#
# Phase 0 starter is the only prompt that has shipped >= 0.575 on the private
# leaderboard. Run B keeps Phase 0's two-prompt structure (MCQ + free-form)
# and adds three rules that target failure modes verified on private:
#
# 1. ``\quad`` between boxes truncates judger contiguity (48 v6 cases).
# 2. Multiple ``\boxed{}`` blocks invite formatting drift; force single box.
# 3. Symbolic gold (e.g. ``-7\sqrt{149}/149``) gets rounded to decimal under
#    the v2 anti-rounding rule. Reverse: keep symbolic forms symbolic.
#
# Length budget: each system prompt < 150 tokens to avoid the v6 reasoning
# drift observed at 349-token Phase 1 prompts (median response +47 %).

RUNB_SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Read the problem and the answer "
    "choices, then select the single best answer.\n\n"
    "Output ONLY the letter inside \\boxed{}, e.g. \\boxed{C}. "
    "Do NOT write \\boxed{(C)}, \\boxed{C.}, or \\boxed{C)}. "
    "Do NOT include the option content or any \\text{} / \\textbf{} macros. "
    "Output exactly one \\boxed{...} at the end of your response."
)

RUNB_SYSTEM_PROMPT_FREE = (
    "You are an expert mathematician. Solve step-by-step. Put your final "
    "answer in \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated, like \\boxed{3, 7, 12}. Do NOT use multiple "
    "\\boxed{} blocks. Do NOT use \\quad, \\qquad, or section headers "
    "near the final answer.\n\n"
    "If the exact answer is irrational (involves \\sqrt, \\pi, e^x, \\ln, "
    "or unsimplified fractions \\frac{p}{q}), keep it symbolic — write "
    "\\boxed{2\\pi}, \\boxed{\\frac{1}{2}}, "
    "\\boxed{-\\frac{7\\sqrt{149}}{149}} — do not convert to decimal "
    "unless the question asks for one."
)


def build_prompt_runb(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run B prompt builder: Phase 0 structure + targeted rules.

    Two prompts (MCQ vs free-form), not three. Multi-part free-form gets
    the same system prompt as single free-form because the Run B mandate
    is single-box-comma-separated regardless of K — no per-K branching.
    """
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNB_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return RUNB_SYSTEM_PROMPT_FREE, question


# ─── Run C: Run B + structural end-with-box rule + text/bool examples ─────
#
# Run B (leaderboard 0.600) left two failure modes on private:
#
# 1. 113/943 = 12 % no-``\\boxed{}`` rate, 75 % of which had response > 30k
#    chars — i.e. Qwen ran out of token budget mid-thinking and never
#    reached a final answer. Adding budget alone doesn't fully fix this
#    because some traces are simply too verbose. We need a structural
#    rule that cues the model to *always* emit a final ``\\boxed{}``.
# 2. Run B's free-form prompt only gave mathematical examples, so
#    free_single's boxed-rate dipped from v5's 81.8 % to 80.5 %. The
#    likely cause is the model treating the symbolic-preference rule as
#    "answer must be mathematical", suppressing valid text/bool answers.
#
# Run C addresses both:
#
# 1. **End-with-box rule** (both prompts). Phrased as an unconditional
#    structural ending, *not* as a "if running out, output guess"
#    fallback. v2 Phase 1 tried the latter and the model produced
#    literal ``\\boxed{...}`` placeholder text. Run C frames the box as
#    the natural end of the response, not as a panic button.
# 2. **Text/bool examples** in the free-form prompt: ``\\boxed{Yes}``,
#    ``\\boxed{Tuesday}``, ``\\boxed{True}``. This counter-balances the
#    symbolic-preference rule for non-numeric answers.
#
# Length budget: Run B was 87 / 137 tokens (MCQ / free). Run C target
# under 110 / 175 tokens. Both still well under the 349-token Phase 1
# regression zone.

RUNC_SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Read the problem and the answer "
    "choices, then select the single best answer.\n\n"
    "Output ONLY the letter inside \\boxed{}, e.g. \\boxed{C}. "
    "Do NOT write \\boxed{(C)}, \\boxed{C.}, or \\boxed{C)}. "
    "Do NOT include the option content or any \\text{} / \\textbf{} macros.\n\n"
    "Your response must end with exactly one \\boxed{X} containing your "
    "chosen letter."
)

RUNC_SYSTEM_PROMPT_FREE = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated, like \\boxed{3, 7, 12}. Do NOT use multiple "
    "\\boxed{} blocks. Do NOT use \\quad, \\qquad, or section headers "
    "near the final answer.\n\n"
    "If the exact answer is irrational (involves \\sqrt, \\pi, e^x, \\ln, "
    "or unsimplified fractions \\frac{p}{q}), keep it symbolic — write "
    "\\boxed{2\\pi}, \\boxed{\\frac{1}{2}}, "
    "\\boxed{-\\frac{7\\sqrt{149}}{149}} — do not convert to decimal "
    "unless the question asks for one. For text or boolean answers, use "
    "the natural form: \\boxed{Yes}, \\boxed{Tuesday}, \\boxed{True}."
)


def build_prompt_runc(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run C prompt builder: Run B + end-with-box + text/bool examples."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNC_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return RUNC_SYSTEM_PROMPT_FREE, question


# ─── Run D: Run C + few-shot worked examples ──────────────────────────────
#
# Run C (val 65.33 %, private predicted 0.605-0.625) hit the practical
# ceiling of *rule-based* prompt engineering. The next gain has to come
# from *demonstration*, not instruction. Few-shot CoT examples are a
# documented +3-7 pp lift on math benchmarks (GSM8K, MATH) for models
# of this size class, and we have not used a single example so far.
#
# Run D adds:
#
# - MCQ: 1 worked example showing letter-only output discipline.
# - Free-form: 3 worked examples, one per critical failure mode:
#   * symbolic answer  (counter the decimal-conversion regression of v2)
#   * multi-part comma format  (counter the 17 free_multi no-box cases)
#   * bool/text answer (counter free_single's residual no-box rate)
#
# Each example uses an explicit ``Q: ... A: ...`` frame so the model
# cannot accidentally lift the boxed value as its final answer the way
# Run C's id=5 ANOVA case copied ``\\boxed{2\\pi}, \\boxed{1/2},
# \\boxed{-7\\sqrt{149}/149}`` from the inline rule examples. Echoing
# the Q→A frame would require fabricating the question too — much less
# likely.
#
# Length budget: MCQ ~130 tokens, free-form ~270 tokens. Both stay
# under the 349-token Phase 1 regression zone but enter the upper end
# of the empirical "sweet spot". Risk-mitigated by val gate at 63 %
# (Run C achieved 65.33 %, allow ~2.3 pp noise).

RUND_SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Read the problem and the answer "
    "choices, then select the single best answer.\n\n"
    "Output ONLY the letter inside \\boxed{}, e.g. \\boxed{C}. "
    "Do NOT write \\boxed{(C)}, \\boxed{C.}, or \\boxed{C)}. "
    "Do NOT include the option content or any \\text{} / \\textbf{} macros. "
    "Your response must end with exactly one \\boxed{X} containing your "
    "chosen letter.\n\n"
    "Example:\n"
    "Q: Which integer is closest to 17/3? Options: A. 4  B. 5  C. 6  D. 7\n"
    "A: 17/3 ≈ 5.667. The closest integer is 6. \\boxed{C}"
)

RUND_SYSTEM_PROMPT_FREE = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated, like \\boxed{3, 7, 12}. Do NOT use multiple "
    "\\boxed{} blocks. Do NOT use \\quad, \\qquad, or section headers "
    "near the final answer.\n\n"
    "If the exact answer is irrational (involves \\sqrt, \\pi, e^x, \\ln, "
    "or unsimplified fractions), keep it symbolic — do not convert to "
    "decimal unless the question asks for one. For text/boolean answers, "
    "use natural form: Yes / Tuesday / True.\n\n"
    "Examples (study the format):\n\n"
    "Q: Compute the area of a circle with radius 3.\n"
    "A: Area = \\pi r^2 = 9\\pi. \\boxed{9\\pi}\n\n"
    "Q: For y = 4x - 7, find the slope and y-intercept.\n"
    "A: This is slope-intercept form. slope = 4, intercept = -7. "
    "\\boxed{4, -7}\n\n"
    "Q: If today is Sunday, what day of the week will it be 9 days from now?\n"
    "A: 9 = 7 + 2 days. Sunday + 2 days = Tuesday. \\boxed{Tuesday}"
)


def build_prompt_rund(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run D prompt builder: Run C + 1-3 few-shot worked examples."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUND_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return RUND_SYSTEM_PROMPT_FREE, question


# ─── Build prompt ─────────────────────────────────────────────────────────


def build_prompt(question: str, options: list[str] | None) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` routed by question type.

    Backward-compatible 2-tuple so existing callers in
    :mod:`cse151b_comp.inference` and :mod:`cse151b_comp.self_consistency`
    continue to work without change. Use :func:`detect_question_type`
    directly if the caller needs the type label.
    """
    qtype = detect_question_type(question, options)

    if qtype == "mc":
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user = (
            f"{question}\n\nOptions:\n{opts_text}\n\n"
            "Select the best answer and output the letter only inside \\boxed{}."
        )
        return SYSTEM_PROMPT_MCQ, user

    if qtype == "free_multi":
        user = (
            f"{question}\n\n"
            "Solve each sub-question and put each answer in its own \\boxed{} "
            "block, in order. The final line should look like: "
            "\\boxed{ans1} \\boxed{ans2} \\boxed{ans3}"
        )
        return SYSTEM_PROMPT_FREE_MULTI, user

    # free_single
    user = f"{question}\n\nSolve and put the final answer in \\boxed{{}}."
    return SYSTEM_PROMPT_FREE_SINGLE, user
