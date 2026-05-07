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
    "then select the single best answer.\n\n" + _MCQ_RULE
)

SYSTEM_PROMPT_FREE_SINGLE = "You are an expert mathematician. Solve the problem step-by-step.\n\n" + _FREE_SINGLE_RULE

SYSTEM_PROMPT_FREE_MULTI = (
    "You are an expert mathematician. Solve each sub-question step-by-step.\n\n" + _FREE_MULTI_RULE
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


# ─── Run E: aggressive — topic routing + 5-shot + MCQ elimination + concise ─
#
# Run E is the "ceiling probe" — pushes every untouched prompt-engineering
# axis simultaneously. Risk is higher than Run D (more variables, more
# total tokens), so it should always be validated on public val before
# Kaggle submission. If val regresses ≥ 2 pp vs Run D, fall back to Run D.
#
# Four axes Run D did not touch:
#
# 1. **Topic routing** (PLAN.md Phase 3a). Detect statistics / calculus /
#    linear-algebra / probability keywords and append a 1-line topic-
#    specific tip. The base free-form prompt stays type-agnostic; the
#    suffix only adds where the topic is detected.
# 2. **5-shot worked examples** (vs Run D's 3-shot). Adds a regression
#    R^2 example (statistics) and a calculus derivative example to
#    cover the two highest-occurrence private topics.
# 3. **MCQ elimination strategy** instruction: "first eliminate clearly
#    wrong options, then commit". Targets the 11 % residual no-box rate
#    on 10-option MCQ where the model exhausts its budget enumerating
#    every option in detail.
# 4. **Conciseness hint** in free-form: "Do not restate intermediate
#    computations multiple times." Direct counter-measure against the
#    truncation pattern (75 % of Run B no-box failures had response >
#    30k chars from over-elaborate reasoning).
#
# Length budget: MCQ 150 tokens, free base 270 tokens, free + topic
# suffix up to 320 tokens. Approaches Phase 1's 349-token regression
# zone — accept the risk because the failure mode is recoverable
# (fall back to Run D) and this is the only way to verify the prompt-
# engineering ceiling without LoRA / SC investment.

_RUNE_TOPIC_SUFFIX_STATS = (
    "\n\nStatistics tip: keep SS, MS, F, R² in exact form during "
    "derivation; round only the final answer if explicitly requested."
)

_RUNE_TOPIC_SUFFIX_CALCULUS = (
    "\n\nCalculus tip: keep limits, derivatives, integrals in exact "
    "symbolic form (e.g. \\sin(x)/x, \\frac{d}{dx}, \\int)."
)

_RUNE_TOPIC_SUFFIX_LINALG = (
    "\n\nLinear algebra tip: present matrix / vector answers as ordered "
    "tuples inside one \\boxed{}; do not unfold step-by-step row reductions."
)

_RUNE_TOPIC_SUFFIX_PROBABILITY = (
    "\n\nProbability tip: keep fractions exact (e.g. \\boxed{1/4}) " "unless decimals are explicitly requested."
)


def _detect_topic_suffix(question: str) -> str:
    """Detect math sub-domain from question keywords; return suffix or empty."""
    q = question.lower()
    stats_kw = (
        "regression",
        "anova",
        "p-value",
        "p value",
        "t-test",
        "f-statistic",
        "f-test",
        "chi-square",
        "variance",
        "standard deviation",
        "confidence interval",
        "ss_total",
        "ss_residual",
        "ss_treatment",
        "null hypothesis",
        "alternative hypothesis",
        "mean square",
    )
    calc_kw = (
        "derivative",
        "differentiate",
        "integral",
        "integrate",
        "limit",
        "antiderivative",
        "lim_{",
        "\\int",
        "\\frac{d}",
    )
    linalg_kw = (
        "matrix",
        "matrices",
        "eigenvalue",
        "eigenvector",
        "determinant",
        "row reduc",
        "rank ",
        "null space",
        "column space",
        "linear combination",
    )
    prob_kw = (
        "probability",
        "p(",
        "expected value",
        "bayes",
        "bernoulli",
        "binomial distribution",
        "poisson",
        "uniformly at random",
    )
    if any(kw in q for kw in stats_kw):
        return _RUNE_TOPIC_SUFFIX_STATS
    if any(kw in q for kw in calc_kw):
        return _RUNE_TOPIC_SUFFIX_CALCULUS
    if any(kw in q for kw in linalg_kw):
        return _RUNE_TOPIC_SUFFIX_LINALG
    if any(kw in q for kw in prob_kw):
        return _RUNE_TOPIC_SUFFIX_PROBABILITY
    return ""


RUNE_SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Read the problem and the answer "
    "choices, then select the single best answer.\n\n"
    "Output ONLY the letter inside \\boxed{}, e.g. \\boxed{C}. "
    "Do NOT write \\boxed{(C)}, \\boxed{C.}, or \\boxed{C)}. "
    "Do NOT include the option content or any \\text{} / \\textbf{} macros. "
    "Your response must end with exactly one \\boxed{X} containing your "
    "chosen letter.\n\n"
    "Strategy for many options (8+): first eliminate clearly-wrong "
    "choices in one or two lines, then commit. Do not derive every "
    "option in detail — partial elimination is enough.\n\n"
    "Example:\n"
    "Q: Which integer is closest to 17/3? Options: A. 4  B. 5  C. 6  D. 7\n"
    "A: 17/3 ≈ 5.667. Closest integer is 6. \\boxed{C}"
)

RUNE_SYSTEM_PROMPT_FREE_BASE = (
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
    "Be concise: identify what is asked, derive efficiently, then box. "
    "Do not restate intermediate computations multiple times.\n\n"
    "Examples (study the format):\n\n"
    "Q: Compute the area of a circle with radius 3.\n"
    "A: A = \\pi r^2 = 9\\pi. \\boxed{9\\pi}\n\n"
    "Q: For y = 4x - 7, find the slope and y-intercept.\n"
    "A: Slope-intercept form. slope=4, intercept=-7. \\boxed{4, -7}\n\n"
    "Q: If today is Sunday, what day will it be 9 days from now?\n"
    "A: 9 = 7 + 2 days. Sunday + 2 = Tuesday. \\boxed{Tuesday}\n\n"
    "Q: A regression yields SS_total=100, SS_residual=20. Compute R^2.\n"
    "A: R^2 = 1 - SS_res/SS_tot = 1 - 20/100 = 0.8. \\boxed{0.8}\n\n"
    "Q: Differentiate f(x) = x^3 + 2x.\n"
    "A: f'(x) = 3x^2 + 2. \\boxed{3x^2 + 2}"
)


def build_prompt_rune(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run E prompt builder: Run D + topic routing + 2 extra examples + MCQ elimination."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNE_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    suffix = _detect_topic_suffix(question)
    return RUNE_SYSTEM_PROMPT_FREE_BASE + suffix, question


# ─── Run F: surgical fixes from Run D val + Run E learnings ───────────────
#
# Run D val regressed -1.78 pp vs Run C (65.33 % → 63.56 %). Detailed
# inspection of the 10 questions where Run C was right and Run D wrong
# isolated four prompt-induced bugs:
#
# - id=30: inline "Yes / Tuesday / True" rule caused the model to
#   substitute Yes/No for A/B in multi-part questions whose sub-answers
#   are letters from MCQ-style sub-options.
# - id=5: any decimal-looking worked example ("R^2 = 0.8" in Run E,
#   weekday in Run D) cues premature rounding (model wrote 62.78 for
#   gold 62.7777...).
# - id=135: the 4, -7 multi-part example over-promotes comma format —
#   model split a single answer "5\sqrt{11}" into "5, \sqrt{(}11)".
# - id=192: irrational-symbolic rule was over-applied to 10^6 vs 1E+06.
#
# Run E ceiling probe (val 56.89 %, -8.44 pp vs Run D) confirmed:
#
# - Long prompt (320 t) + 5-shot + topic suffix made the model **ignore**
#   the format rules: 40 % multi-box, 18.7 % \quad — worse than v6.
# - The single Run E change that survived was MCQ elimination clause:
#   MCQ accuracy 73.3 % was on par with Run D. Cherry-pick this.
#
# Run F design: keep Run D's working core, surgically remove the three
# Run D footguns, add Run E's MCQ elimination clause, do NOT add
# topic routing or 5-shot or "be concise". Target free <= 230 tokens.

# NOTE: Run F is now a "config bundle" not just a prompt. The CLI auto-
# enables v2 budget when --prompt runf is selected (floor 16k, MCQ cap
# 22k, multi cap 30k). This is the "Run F final" config:
#
#     Run F (final) = Run F prompt + v2 budget
#
# Day1-distill-pool's Run F val 58.67% used v1 budget (12k floor); the
# final config inherits the prompt but adds the v2 budget fix that
# addresses Run C's free_single starvation. Use --budget-v1 to force
# the older v1 budget for reproducing the day1 number.

RUNF_SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Read the problem and the answer "
    "choices, then select the single best answer.\n\n"
    "Output ONLY the letter inside \\boxed{}, e.g. \\boxed{C}. "
    "Do NOT write \\boxed{(C)}, \\boxed{C.}, or \\boxed{C)}. "
    "Do NOT include the option content or any \\text{} / \\textbf{} macros. "
    "Your response must end with exactly one \\boxed{X} containing your "
    "chosen letter.\n\n"
    "For 8+ option questions: eliminate clearly-wrong choices first, then "
    "commit. Do not derive every option in detail.\n\n"
    "Example:\n"
    "Q: Which integer is closest to 17/3? Options: A. 4  B. 5  C. 6  D. 7\n"
    "A: 17/3 ≈ 5.667. Closest is 6. \\boxed{C}"
)

RUNF_SYSTEM_PROMPT_FREE = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated, like \\boxed{3, 7, 12}. Do NOT use multiple "
    "\\boxed{} blocks. Do NOT use \\quad, \\qquad, or section headers "
    "near the final answer.\n\n"
    "If the exact answer involves \\sqrt, \\pi, e^x, \\ln, or unsimplified "
    "fractions, keep it symbolic — do not convert to decimal unless the "
    "question asks for one.\n\n"
    "Examples (study the format):\n\n"
    "Q: Compute the area of a circle with radius 3.\n"
    "A: A = \\pi r^2 = 9\\pi. \\boxed{9\\pi}\n\n"
    "Q: For y = 4x - 7, find the slope and y-intercept.\n"
    "A: Slope-intercept form. slope=4, intercept=-7. \\boxed{4, -7}\n\n"
    "Q: Simplify \\sqrt{75}.\n"
    "A: 75 = 25 \\times 3, so \\sqrt{75} = 5\\sqrt{3}. \\boxed{5\\sqrt{3}}"
)


def build_prompt_runf(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run F prompt builder: Run D core + bug fixes + MCQ elimination from E."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNF_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return RUNF_SYSTEM_PROMPT_FREE, question


# ─── Run G: Run F + mixed multi-part example + v2 budget ──────────────────
#
# Once the "long prompt causes drift" assumption is relaxed (B/C analysis
# attributed Run C's regression mostly to specific bad rules — Yes/Tuesday/
# True inline + 12k free_single budget — not raw token count), Run G adds
# one more worked example targeting Run C's id=30 failure mode:
#
# - Multi-part questions whose sub-answers include letters or words like
#   "reject"/"accept" (statistics test conclusions). Run C's inline rule
#   "use natural form: Yes / Tuesday / True" caused these to be answered
#   as Yes/No instead of the question's own option text.
#
# Solution: a Q→A example showing a t-test multi-part with mixed letter +
# numeric sub-answer, output as ``\boxed{reject, 2.45}``. Demonstrates:
# (1) using the question's own language (reject) for letter-style sub-
# answers, (2) mixed types in single comma-separated boxed.
#
# MCQ prompt is identical to Run F. Free-form gains ~150 chars (~37
# tokens) from the new example. Budget bumped to v2's higher tier
# (MCQ 22k cap, multi 30k cap) to maximize headroom for hard problems.

RUNG_SYSTEM_PROMPT_MCQ = RUNF_SYSTEM_PROMPT_MCQ  # No change for MCQ.

RUNG_SYSTEM_PROMPT_FREE = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated, like \\boxed{3, 7, 12}. Do NOT use multiple "
    "\\boxed{} blocks. Do NOT use \\quad, \\qquad, or section headers "
    "near the final answer.\n\n"
    "If the exact answer is irrational (involves \\sqrt, \\pi, e^x, \\ln, "
    "or unsimplified fractions), keep it symbolic — do not convert to "
    "decimal unless the question asks for one.\n\n"
    "Examples (study the format):\n\n"
    "Q: Compute the area of a circle with radius 3.\n"
    "A: A = \\pi r^2 = 9\\pi. \\boxed{9\\pi}\n\n"
    "Q: For y = 4x - 7, find the slope and y-intercept.\n"
    "A: Slope-intercept form. slope=4, intercept=-7. \\boxed{4, -7}\n\n"
    "Q: Simplify \\sqrt{75}.\n"
    "A: 75 = 25 \\times 3, so \\sqrt{75} = 5\\sqrt{3}. \\boxed{5\\sqrt{3}}\n\n"
    "Q: Test H0: μ=10 vs Ha: μ≠10. t-stat=2.45, critical value 1.96. "
    "(a) Reject H0? (b) Report the t-stat.\n"
    "A: 2.45 > 1.96, so reject. (a) reject. (b) 2.45. "
    "\\boxed{reject, 2.45}"
)


def build_prompt_rung(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run G prompt builder: Run F + mixed multi-part example for sub-letter answers."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNG_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return RUNG_SYSTEM_PROMPT_FREE, question


# ─── Run H: Run B + 2 cautious additions only ─────────────────────────────
#
# After validating that prompt iteration B → C → D → E → F → G is in K=1
# noise on stratified val (1.78 pp spread) and that K=8 SC + Phase 0
# already wins on private (0.611 vs Run B 0.600), Run H is the
# minimum-risk extension of Run B as the K=8 SC base prompt.
#
# Only TWO additions, each independently validated as non-harmful:
#
# 1. **End-with-box rule for free-form**: rephrase "Put your final
#    answer in \\boxed{}" → "End your response with your final answer
#    inside \\boxed{}". Run C/D/F/G all carry this rule; no run was
#    attributable regression to it. May rescue 5-10 of Run B's 113
#    private no-box failures.
#
# 2. **MCQ 8+ option elimination clause**: Run E (the run that crashed
#    overall) was the only place this clause was tested in isolation,
#    and MCQ accuracy on val held at 73.3% — the elimination clause
#    did NOT contribute to E's collapse. Run F kept it; Run F MCQ
#    74.7%. Targeted at 10-opt MCQ (267/300 of private MCQ) where Run
#    B has 11 % no-box rate from Qwen enumerating every option.
#
# What's intentionally NOT in Run H:
#
# - No worked examples (C/D/E/F/G all tried; net wash on val,
#   id=5/30/135 echo bugs introduced).
# - No Yes/Tuesday/True inline rule (Run C's id=30 cause).
# - No topic routing (Run E crashed).
# - No per-type budget (Run B's flat 16 k on private = 0.600; varying
#   the floor introduced 12 k starvation in Run C).
# - Length kept under 175 tokens (Run B's 137 + ~30 for elim clause).
#
# Hypothesis: Run H ≈ Run B on K=1 single-shot val (within noise), but
# slightly stronger when paired with K=8 SC because end-with-box +
# elimination both reduce no-box rate, and SC's voting then converges
# on the boxed answers.

RUNH_SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Read the problem and the answer "
    "choices, then select the single best answer.\n\n"
    "Output ONLY the letter inside \\boxed{}, e.g. \\boxed{C}. "
    "Do NOT write \\boxed{(C)}, \\boxed{C.}, or \\boxed{C)}. "
    "Do NOT include the option content or any \\text{} / \\textbf{} macros. "
    "Output exactly one \\boxed{...} at the end of your response.\n\n"
    "For 8+ option questions: eliminate clearly-wrong choices first, then "
    "commit. Do not derive every option in detail."
)

RUNH_SYSTEM_PROMPT_FREE = (
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
    "unless the question asks for one."
)


def build_prompt_runh(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run H prompt builder: Run B + end-with-box (free) + 8+ option elim (MCQ)."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNH_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return RUNH_SYSTEM_PROMPT_FREE, question


# ─── Run I: topic-routed free-form, building on Run F final ───────────────
#
# Run F final scored 0.632 on private (vs Run B 0.600, K=8 SC + Phase 0
# 0.611). The +3.2 pp gain over Run B from prompt + v2 budget proves
# prompt engineering is FAR from saturated — directly contradicting the
# earlier "diminishing returns" framing.
#
# Run I escalates: 5-way topic routing on free-form (MCQ unchanged
# from Run F final, since MCQ already at 96.3 % box rate).
#
# Lessons from Run E (which crashed val to 56.89%):
#   1. Run E used append style (base prompt + topic tip suffix). Run I
#      uses REPLACE style — each topic gets its own complete prompt.
#   2. Run E added 5-shot + topic tip + 'be concise' simultaneously.
#      Run I changes only the topic axis; each branch keeps F final's
#      anti-pattern core unchanged.
#   3. Run E used liberal keyword matching. Run I uses STRICT keywords
#      (high precision, low recall) — when ambiguous, fall through to
#      generic.
#
# Five branches:
#   - GENERIC (default): Run F final's prompt verbatim
#   - STATISTICS: hypothesis-test, regression, ANOVA examples
#   - CALCULUS: integral / derivative example, symbolic emphasis
#   - LINALG: matrix / eigenvalue example, ordered-tuple emphasis
#   - PROBABILITY: exact-fraction emphasis, expected-value example
#
# Routing priority (most specific first):
#   stats → linalg → calc → prob → generic

# ─── Topic detection (strict keywords) ────────────────────────────────────

_RUNI_STATS_KW = (
    # Hypothesis testing
    "null hypothesis",
    "alternative hypothesis",
    "p-value",
    "p value",
    "reject h0",
    "reject the null",
    "fail to reject",
    "test the claim",
    "significance level",
    "test statistic",
    # ANOVA / regression
    "anova",
    "regression",
    "least squares",
    "slope and intercept",
    "f-test",
    "f test",
    "t-test",
    "t test",
    "chi-square",
    "ss_total",
    "ss_residual",
    "treatment effect",
    "main effect",
    "experiment is conducted",
    "experiment was conducted",
    # Estimators / parameters
    "sample mean",
    "sample variance",
    "sample standard deviation",
    "population mean",
    "population variance",
    "standard error",
    "confidence interval",
    "margin of error",
    "r^2",
    "r-squared",
    "coefficient of determination",
    # Misc statistical
    "compare the means",
    "compare the mean",
    "estimate the percentage",
    "estimate the proportion",
)
_RUNI_CALC_KW = (
    "derivative",
    "differentiate",
    "integral",
    "integrate",
    "antiderivative",
    "limit of",
    "lim_{",
    "lim(",
    "\\int",
    "\\frac{d}",
    "find dy/dx",
    "rate of change",
    "tangent line",
    "concav",
    "inflection",
)
_RUNI_LINALG_KW = (
    "matrix",
    "matrices",
    "eigenvalue",
    "eigenvector",
    "determinant",
    "row reduc",
    "rank ",
    "null space",
    "column space",
    "linearly independ",
    "linear transformation",
    "\\begin{pmatrix}",
    "\\begin{bmatrix}",
)
_RUNI_PROB_KW = (
    "probability",
    "probabilities",
    "p(",
    "expected value",
    "bernoulli",
    "binomial distribution",
    "poisson",
    "uniformly at random",
    "random variable",
    "bayes",
)


def _detect_runi_topic(question: str) -> str:
    """Return one of: 'stats', 'calc', 'linalg', 'prob', 'generic'.

    Priority order (most-specific keywords first). False positives go
    to 'generic'.
    """
    q = question.lower()
    if any(kw in q for kw in _RUNI_STATS_KW):
        return "stats"
    if any(kw in q for kw in _RUNI_LINALG_KW):
        return "linalg"
    if any(kw in q for kw in _RUNI_CALC_KW):
        return "calc"
    if any(kw in q for kw in _RUNI_PROB_KW):
        return "prob"
    return "generic"


# ─── Five free-form prompt branches ───────────────────────────────────────
# Common scaffold: Phase 0 base + Run B anti-pattern + end-with-box +
# topic-specific guidance + 1-2 worked example per branch.

RUNI_SYSTEM_PROMPT_MCQ = RUNF_SYSTEM_PROMPT_MCQ  # MCQ already at 96.3% box, no change

RUNI_SYSTEM_PROMPT_FREE_GENERIC = RUNF_SYSTEM_PROMPT_FREE  # Run F final default

RUNI_SYSTEM_PROMPT_FREE_STATS = (
    "You are an expert statistician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated, like \\boxed{3, 7, 12}. Do NOT use multiple "
    "\\boxed{} blocks. Do NOT use \\quad, \\qquad, or section headers "
    "near the final answer.\n\n"
    "Statistics-specific guidance:\n"
    "- For hypothesis-test conclusions, use the question's own option "
    'text (e.g., "reject" or "fail to reject"), NOT yes/no/true/false.\n'
    "- Keep p-values, F-stats, t-stats, R^2 as exact fractions or "
    "high-precision decimals (do not pre-round).\n"
    "- For regression coefficients, report exact values from the "
    "least-squares formula.\n\n"
    "Examples:\n"
    "Q: Test H0: μ=10 vs Ha: μ≠10. t-stat=2.45, critical 1.96. "
    "(a) Reject H0? (b) Report t-stat.\n"
    "A: 2.45 > 1.96, so reject. (a) reject. (b) 2.45. "
    "\\boxed{reject, 2.45}\n\n"
    "Q: A regression yields SS_total=100, SS_residual=20. Compute R^2.\n"
    "A: R^2 = 1 - SS_res/SS_tot = 1 - 20/100 = 0.8. \\boxed{0.8}"
)

RUNI_SYSTEM_PROMPT_FREE_CALCULUS = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated, like \\boxed{3, 7, 12}. Do NOT use multiple "
    "\\boxed{} blocks. Do NOT use \\quad, \\qquad, or section headers "
    "near the final answer.\n\n"
    "Calculus-specific guidance:\n"
    "- Keep derivatives and integrals in exact symbolic form (e.g., "
    "\\boxed{2x + 3}, \\boxed{\\frac{x^3}{3} + C}).\n"
    "- For limits, evaluate to the exact value or symbolic form, do "
    "NOT round.\n"
    "- Use \\frac for fractions, \\sqrt for roots, \\pi for π.\n\n"
    "Examples:\n"
    "Q: Differentiate f(x) = x^3 + 2x.\n"
    "A: f'(x) = 3x^2 + 2. \\boxed{3x^2 + 2}\n\n"
    "Q: Compute \\int 2x \\, dx.\n"
    "A: \\int 2x \\, dx = x^2 + C. \\boxed{x^2 + C}"
)

RUNI_SYSTEM_PROMPT_FREE_LINALG = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated, like \\boxed{3, 7, 12}. Do NOT use multiple "
    "\\boxed{} blocks. Do NOT use \\quad, \\qquad, or section headers "
    "near the final answer.\n\n"
    "Linear algebra guidance:\n"
    "- For eigenvalues, list as comma-separated values: \\boxed{1, 2, 3}.\n"
    "- For matrix answers, use \\begin{pmatrix} ... \\end{pmatrix} "
    "inside one \\boxed{}.\n"
    "- For determinants, output the exact numeric or symbolic value.\n\n"
    "Examples:\n"
    "Q: Find the determinant of [[2,1],[3,4]].\n"
    "A: det = 2*4 - 1*3 = 5. \\boxed{5}\n\n"
    "Q: Find eigenvalues of [[2,0],[0,3]].\n"
    "A: Diagonal entries are eigenvalues. \\boxed{2, 3}"
)

RUNI_SYSTEM_PROMPT_FREE_PROB = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated, like \\boxed{3, 7, 12}. Do NOT use multiple "
    "\\boxed{} blocks. Do NOT use \\quad, \\qquad, or section headers "
    "near the final answer.\n\n"
    "Probability guidance:\n"
    "- Express probabilities as exact fractions like \\boxed{1/4} or "
    "\\boxed{\\frac{1}{4}}, NOT rounded decimals (unless the question "
    "asks for a decimal).\n"
    "- Expected values: keep as exact fractions or symbolic.\n"
    "- Conditional probabilities P(A|B): use Bayes' theorem with exact "
    "values.\n\n"
    "Examples:\n"
    "Q: A fair die is rolled. What is P(rolling a 4)?\n"
    "A: There are 6 equally likely outcomes. P = 1/6. \\boxed{1/6}\n\n"
    "Q: Compute the expected value of a fair 6-sided die roll.\n"
    "A: E[X] = (1+2+3+4+5+6)/6 = 21/6 = 7/2. \\boxed{7/2}"
)


def build_prompt_runi(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run I prompt builder: 5-way topic-routed free-form on top of Run F final."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNI_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    topic = _detect_runi_topic(question)
    if topic == "stats":
        return RUNI_SYSTEM_PROMPT_FREE_STATS, question
    if topic == "calc":
        return RUNI_SYSTEM_PROMPT_FREE_CALCULUS, question
    if topic == "linalg":
        return RUNI_SYSTEM_PROMPT_FREE_LINALG, question
    if topic == "prob":
        return RUNI_SYSTEM_PROMPT_FREE_PROB, question
    return RUNI_SYSTEM_PROMPT_FREE_GENERIC, question


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
