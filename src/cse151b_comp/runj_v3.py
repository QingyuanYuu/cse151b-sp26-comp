"""Run J v3 — fix the two remaining v2 failure modes.

V2 ablation showed v2 fixed 5 branches (olympiad / stats_hyp / stats_desc /
prob_combi / number_alg) but still hurt 3 (trig -7.5pp, geometry -18pp,
stats_reg -7.4pp). Root causes from reports/runj_ablation_v2_review.md:

  Loss mode A (5/8 v2 losses): "decimal data → ≥4 sig figs" rule too weak.
    Judger needs ≥7 sig figs OR symbolic. v2 gave 7.951 vs gold 7.95110...
    (off by 1e-4, judger rejected). Run F's no-rule baseline let model
    keep symbolic — symbolic evaluates exactly to gold, judger accepts.

  Loss mode B (1/8 v2 losses): multi-part rule misfired on composite
    expressions. Trig [1061] gold = single 'A·sin(B·t+φ)' expression.
    v2 forced decomposition into 3 items (A, B, φ separately).

  Loss mode C (2/8 v2 losses): real reasoning errors — not prompt-fixable.

V3 changes from v2:

1. **Drop the "≥4 sig figs" rule entirely**. Replace with "DO NOT round —
   prefer symbolic, OR full-precision decimal (≥7 sig figs) only if
   required". This restores Run F's flexible behavior while keeping v2's
   other fixes.

2. **Refine multi-part rule**: clarify that a single composite expression
   (A·sin(B·t+φ), \\sqrt{a²+b²}, etc.) counts as ONE item — only count
   separate [ANS] blanks or labeled (a)/(b)/(c) sub-parts.

3. Keep all v2 fixes that worked: no examples, no verbatim word rules,
   "match question's option codes", "follow specified syntax literally".
"""

from __future__ import annotations

from cse151b_comp.prompts import RUNF_SYSTEM_PROMPT_FREE, RUNF_SYSTEM_PROMPT_MCQ
from cse151b_comp.topics import detect_topic

# ─── Common preamble (shared by all v3 branches) ──────────────────────────

_PREAMBLE = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "Multi-part rule: If the question has N separate [ANS] blanks or N "
    "labeled sub-parts ((a)/(b)/(c)/(1)/(2)...), your single \\boxed{} "
    "must contain EXACTLY N comma-separated items, like \\boxed{a, b, c}. "
    "Do NOT aggregate sub-answers into one string item. IMPORTANT: a "
    "single composite expression (e.g. A\\sin(Bt+\\phi), \\sqrt{a^2+b^2}) "
    "counts as ONE item even if structurally complex — only count separate "
    "[ANS] blanks. Do NOT use multiple \\boxed{} blocks. Do NOT use \\quad "
    "or section headers near the final answer.\n\n"
    "Format-matching rule: Match the question's expected style:\n"
    "- DO NOT round numerical answers. Either keep symbolic (\\sqrt, \\pi, "
    "fractions, \\arctan) when natural — symbolic is preferred over decimal "
    "— OR give full precision (≥7 significant figures) if a decimal is "
    "required.\n"
    "- Letter options (A/B/C/D) → output the letter; do NOT substitute "
    "Yes/No or word forms.\n"
    '- Specified syntax ("(a,b)" with parens, "use sqrt") → follow '
    "literally."
)


# ─── Branch-specific tips (same as v2 — those didn't cause regressions) ──


_TRIG_TIPS = (
    "\n\nTrigonometry tips:\n"
    "- Inverse functions: \\arcsin returns values in [-\\pi/2, \\pi/2]; "
    "\\arccos in [0, \\pi]. Adjust if the question restricts a different range.\n"
    "- Use identities (\\sin^2 + \\cos^2 = 1, double-angle, sum/difference) "
    "before computing.\n"
    "- For periodic equations on [0, 2\\pi), enumerate ALL solutions in order."
)

_GEOMETRY_TIPS = (
    "\n\nGeometry tips:\n"
    "- Identify the figure (triangle/circle/polygon) and applicable formulas.\n"
    "- Standard formulas: A_circle = \\pi r^2, A_triangle = (1/2)bh, "
    "Pythagorean theorem.\n"
    "- Arc / sector: arclength = r\\theta (radians), area = (1/2)r^2\\theta.\n"
    "- Coordinate answers: include parentheses if the question shows them, "
    "e.g. (a, b)."
)

_STATS_HYP_TIPS = (
    "\n\nHypothesis-testing tips:\n"
    "- State H0/Ha, compute the test statistic, compare to critical value "
    "or p-value.\n"
    "- Decision: match the question's expected response. If the question "
    "lists option codes (A/B/C/D), output the matching letter — NOT "
    "'reject'/'fail to reject' as words.\n"
    "- Multi-part order: follow the question's stated order."
)

_STATS_REG_TIPS = (
    "\n\nRegression / model-fit tips:\n"
    "- R^2 = 1 - SS_res/SS_tot; keep exact when inputs are clean integers.\n"
    "- Slope and intercept: comma-separated in the question's order.\n"
    "- Multiple regression: compute beta coefficients via least-squares "
    "normal equations (X^T X)^{-1} X^T y; double-check the system before "
    "reporting.\n"
    "- Predicted values: hat-y = beta_0 + beta_1 x_1 + beta_2 x_2 + ..."
)

_STATS_DESC_TIPS = (
    "\n\nDescriptive-statistics tips:\n"
    "- Mean: \\bar{x} = \\sum x_i / n. Sort first for median; if even n, "
    "average the middle two.\n"
    "- Sample SD uses n-1 (Bessel's correction); population SD uses n.\n"
    "- Quartiles: Q1/Q3 are 25th/75th percentile of sorted data.\n"
    "- Multi-part order usually (mean, median, sd), unless the question "
    "specifies a different order — follow the question."
)

_CALCULUS_TIPS = (
    "\n\nCalculus tips:\n"
    "- Keep derivatives, integrals, limits in exact symbolic form.\n"
    "- Apply chain / product / quotient rules; for indefinite integrals add +C.\n"
    "- For convergence questions: match the question's expected response — "
    "if it asks T/F, output 'True'/'False'; if it asks a letter, output the letter."
)

_PROB_COMBI_TIPS = (
    "\n\nProbability / combinatorics tips:\n"
    "- Keep fractions exact when natural (1/4, not 0.25).\n"
    "- Expected value: E[X] = \\sum x_i P(x_i).\n"
    "- Counting: C(n,k) = n!/(k!(n-k)!); P(n,k) = n!/(n-k)!\n"
    "- Conditional: P(A|B) = P(A \\cap B) / P(B); independence: "
    "P(A \\cap B) = P(A) P(B)."
)

_OLYMPIAD_TIPS = (
    "\n\nOlympiad / proof tips:\n"
    "- For 'find all' problems: present complete characterization "
    "(all solutions, no extras, no missing).\n"
    "- For 'prove that' problems: end with a clear conclusion; the boxed "
    "answer is typically a set, value, or closed-form expression.\n"
    "- For routine equation answers (e.g. slope-intercept), use the simplest "
    "form requested. If the question asks for an expression, do NOT prepend "
    "the dependent variable: \\boxed{0.25x+15.7}, not \\boxed{y=0.25x+15.7}."
)

_NUMBER_ALG_TIPS = (
    "\n\nNumber theory / sequences / linear algebra tips:\n"
    "- Remainders: N mod k is the integer in [0, k-1]; show the division step.\n"
    "- gcd / lcm: brief prime factorization.\n"
    "- Arithmetic seq: a_n = a_1 + (n-1)d. Geometric: a_n = a_1 r^{n-1}.\n"
    "- Linear algebra: matrix det, eigenvalues, characteristic polynomial "
    "det(A - \\lambda I) = 0.\n"
    "- Multi-step expansions (Cantor, base conversion, etc.): output "
    "exactly the number of digits / coefficients the question requests, "
    "no fewer and no extras."
)


# ─── Compose branches (preamble + tips) ──────────────────────────────────


_BRANCH_TRIG = _PREAMBLE + _TRIG_TIPS
_BRANCH_GEOMETRY = _PREAMBLE + _GEOMETRY_TIPS
_BRANCH_STATS_HYP = _PREAMBLE + _STATS_HYP_TIPS
_BRANCH_STATS_REG = _PREAMBLE + _STATS_REG_TIPS
_BRANCH_STATS_DESC = _PREAMBLE + _STATS_DESC_TIPS
_BRANCH_CALCULUS = _PREAMBLE + _CALCULUS_TIPS
_BRANCH_PROB_COMBI = _PREAMBLE + _PROB_COMBI_TIPS
_BRANCH_OLYMPIAD = _PREAMBLE + _OLYMPIAD_TIPS
_BRANCH_NUMBER_ALG = _PREAMBLE + _NUMBER_ALG_TIPS


_BRANCH_PROMPTS = {
    "olympiad": _BRANCH_OLYMPIAD,
    "trig": _BRANCH_TRIG,
    "geometry": _BRANCH_GEOMETRY,
    "stats_hyp_test": _BRANCH_STATS_HYP,
    "stats_regression": _BRANCH_STATS_REG,
    "stats_descriptive": _BRANCH_STATS_DESC,
    "calculus": _BRANCH_CALCULUS,
    "prob_combi": _BRANCH_PROB_COMBI,
    "number_alg": _BRANCH_NUMBER_ALG,
}


# ─── Builders ────────────────────────────────────────────────────────────


def build_prompt_runj_v3(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run J v3: revised branch prompts (drop ≥4 sig figs rule, refine multi-part)."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNF_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    topic = detect_topic(question)
    system = _BRANCH_PROMPTS.get(topic, RUNF_SYSTEM_PROMPT_FREE)
    return system, question


def _load_v3_final_branches() -> tuple[str, ...]:
    """Read enabled branches from data/runj_v3_final_branches.txt.

    Default if file missing: the 6 winners from v3 ablation
    (olympiad, stats_hyp_test, stats_descriptive, prob_combi, calculus, number_alg).
    """
    import pathlib

    path = pathlib.Path("data/runj_v3_final_branches.txt")
    if not path.exists():
        return ("olympiad", "stats_hyp_test", "stats_descriptive", "prob_combi", "calculus", "number_alg")
    enabled = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return tuple(b for b in enabled if b in _BRANCH_PROMPTS)


def build_prompt_runj_v3_final(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run J v3 final: only enabled branches; rest fall through to Run F."""
    enabled = _load_v3_final_branches()
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNF_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    topic = detect_topic(question)
    if topic in enabled and topic in _BRANCH_PROMPTS:
        return _BRANCH_PROMPTS[topic], question
    return RUNF_SYSTEM_PROMPT_FREE, question


def _make_ablation(enabled_topics: tuple[str, ...]):
    def builder(question: str, options: list[str] | None) -> tuple[str, str]:
        if options:
            labels = [chr(65 + i) for i in range(len(options))]
            opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
            return RUNF_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
        topic = detect_topic(question)
        if topic in enabled_topics:
            return _BRANCH_PROMPTS[topic], question
        return RUNF_SYSTEM_PROMPT_FREE, question

    return builder


build_prompt_runj_v3_olympiad = _make_ablation(("olympiad",))
build_prompt_runj_v3_trig = _make_ablation(("trig",))
build_prompt_runj_v3_geom = _make_ablation(("geometry",))
build_prompt_runj_v3_stats_hyp = _make_ablation(("stats_hyp_test",))
build_prompt_runj_v3_stats_reg = _make_ablation(("stats_regression",))
build_prompt_runj_v3_stats_desc = _make_ablation(("stats_descriptive",))
build_prompt_runj_v3_calc = _make_ablation(("calculus",))
build_prompt_runj_v3_prob = _make_ablation(("prob_combi",))
build_prompt_runj_v3_number_alg = _make_ablation(("number_alg",))


__all__ = [
    "build_prompt_runj_v3",
    "build_prompt_runj_v3_final",
    "build_prompt_runj_v3_olympiad",
    "build_prompt_runj_v3_trig",
    "build_prompt_runj_v3_geom",
    "build_prompt_runj_v3_stats_hyp",
    "build_prompt_runj_v3_stats_reg",
    "build_prompt_runj_v3_stats_desc",
    "build_prompt_runj_v3_calc",
    "build_prompt_runj_v3_prob",
    "build_prompt_runj_v3_number_alg",
]
