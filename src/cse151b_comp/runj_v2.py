"""Run J v2 — branch prompts revised based on Run J v1 ablation losses.

Lessons applied (from 18 v1 losses, see reports/runj_ablation_review.md):

1. **Drop Q:/A: in-prompt examples** — they bleed style (decimal vs frac,
   numbers vs letters). Branches now use semantic tips only.
2. **Drop verbatim word requirements** ("output 'reject'", "keep symbolic")
   — they conflict with question-specific format. Replaced with
   "match the question's option/data style".
3. **Add multi-part counting rule** — biggest fixable category (5/18
   losses). Explicit: "exactly N items, NO aggregation."
4. **Add format-matching rule** — fixes precision loss, paren drops,
   letter→Yes/No substitution.

Branches still cover the same 9 topics. Generic still falls through to
Run F's prompt unchanged.
"""

from __future__ import annotations

from cse151b_comp.prompts import RUNF_SYSTEM_PROMPT_FREE, RUNF_SYSTEM_PROMPT_MCQ
from cse151b_comp.topics import detect_topic

# ─── Common preamble (shared by all v2 branches) ──────────────────────────
#
# Length budget: ~130 tokens preamble + ~70 per-branch tips ≤ 200 tokens
# total per branch (well under 250-token reasoning-drift threshold).

_PREAMBLE = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "Multi-part rule: If the question has N [ANS] blanks or N sub-parts "
    "(a)/(b)/(c)..., your single \\boxed{} must contain EXACTLY N "
    "comma-separated items, like \\boxed{a, b, c}. Do NOT aggregate "
    "sub-answers into one string item (e.g. NOT \\boxed{'3,7,12 — slope'} "
    "as 1 item; output 3 items: \\boxed{3, 7, 12}). Do NOT use multiple "
    "\\boxed{} blocks. Do NOT use \\quad or section headers.\n\n"
    "Format-matching rule: Match the question's answer style:\n"
    "- Decimal data in question (e.g. r=5.1) → decimal answer with "
    "≥4 significant figures.\n"
    "- Symbolic data (\\sqrt, \\pi, fractions) → keep symbolic; do not "
    "approximate to decimals unless the question requests one.\n"
    "- Letter options (A/B/C/D) → output the letter; do NOT substitute "
    "Yes/No/word forms.\n"
    '- Specified syntax ("(a,b)" with parens, "use sqrt") → follow '
    "literally."
)


# ─── Branch-specific semantic tips (NO examples, NO verbatim rules) ─────


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
    "- Keep p-values, F-stats, t-stats to 4 significant figures.\n"
    "- Multi-part order: follow the question's stated order."
)

_STATS_REG_TIPS = (
    "\n\nRegression / model-fit tips:\n"
    "- R^2 = 1 - SS_res/SS_tot; keep exact when inputs are clean integers.\n"
    "- Slope and intercept: comma-separated in the question's order.\n"
    "- Multiple regression: compute beta coefficients via least-squares "
    "normal equations (X^T X)^{-1} X^T y.\n"
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
    "if it asks T/F, output 'True'/'False'; if it asks a letter, output the letter.\n"
    "- For numerical evaluation (e.g., f(2.5)), preserve precision from "
    "the question's data."
)

_PROB_COMBI_TIPS = (
    "\n\nProbability / combinatorics tips:\n"
    "- Keep fractions exact when natural (1/4, not 0.25). Use decimals "
    "only if the question gives decimal data or asks for them.\n"
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


def build_prompt_runj_v2(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run J v2: revised branch prompts. MCQ unchanged from Run F."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNF_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    topic = detect_topic(question)
    system = _BRANCH_PROMPTS.get(topic, RUNF_SYSTEM_PROMPT_FREE)
    return system, question


# ─── Final v2: only the branches that won in v2 ablation ─────────────────


def _load_v2_final_branches() -> tuple[str, ...]:
    """Read enabled branches from data/runj_v2_final_branches.txt.

    Default if file missing: the 5 winners from v2 ablation
    (olympiad, stats_hyp_test, stats_descriptive, prob_combi, number_alg).
    """
    import pathlib

    path = pathlib.Path("data/runj_v2_final_branches.txt")
    if not path.exists():
        return ("olympiad", "stats_hyp_test", "stats_descriptive", "prob_combi", "number_alg")
    enabled = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return tuple(b for b in enabled if b in _BRANCH_PROMPTS)


def build_prompt_runj_v2_final(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run J v2 final: only enabled branches; rest fall through to Run F."""
    enabled = _load_v2_final_branches()
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


build_prompt_runj_v2_olympiad = _make_ablation(("olympiad",))
build_prompt_runj_v2_trig = _make_ablation(("trig",))
build_prompt_runj_v2_geom = _make_ablation(("geometry",))
build_prompt_runj_v2_stats_hyp = _make_ablation(("stats_hyp_test",))
build_prompt_runj_v2_stats_reg = _make_ablation(("stats_regression",))
build_prompt_runj_v2_stats_desc = _make_ablation(("stats_descriptive",))
build_prompt_runj_v2_calc = _make_ablation(("calculus",))
build_prompt_runj_v2_prob = _make_ablation(("prob_combi",))
build_prompt_runj_v2_number_alg = _make_ablation(("number_alg",))


__all__ = [
    "build_prompt_runj_v2",
    "build_prompt_runj_v2_final",
    "build_prompt_runj_v2_olympiad",
    "build_prompt_runj_v2_trig",
    "build_prompt_runj_v2_geom",
    "build_prompt_runj_v2_stats_hyp",
    "build_prompt_runj_v2_stats_reg",
    "build_prompt_runj_v2_stats_desc",
    "build_prompt_runj_v2_calc",
    "build_prompt_runj_v2_prob",
    "build_prompt_runj_v2_number_alg",
]
