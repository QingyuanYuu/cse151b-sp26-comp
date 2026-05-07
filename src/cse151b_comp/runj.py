"""Run J — data-driven topic-routed prompts on top of Run F final.

Designed using empirical topic distribution
(reports/empirical_topic_distribution.md), not Run I's intuition. Drops
unused branches (LINALG, CALCULUS sub-routes) and adds the three
biggest unclaimed sub-domains: TRIG (358 hits), LOGIC_PROOF (176),
GEOMETRY (121).

Architecture:

- 7 specialized branches + 1 GENERIC (= Run F final) fallback
- Each branch is a complete REPLACE-style prompt (no append-suffix)
- Length kept under ~250 tokens per branch (Phase 1 reasoning-drift
  threshold is 349 tokens)
- MCQ prompt unchanged from Run F final (Run J adds nothing for MCQ;
  the wins are in free-form subdomain routing)

Per-branch ablation (test on topic-filtered val_225):

    J_base    = Run F final                                  control
    J_trig    = Run F + TRIG only
    J_geom    = Run F + GEOMETRY only
    J_logic   = Run F + LOGIC_PROOF only
    J_stats   = Run F + STATS_HYPOTHESIS + STATS_REGRESSION
    J_prob    = Run F + PROBABILITY only
    J_num     = Run F + NUM_THEORY only
    J_full    = Run F + all 7 branches                       final

The ablation harness in `scripts/run_j_ablation.sh` picks per-branch
val subsets, runs each variant, and the winning branches assemble
into J_full's deployment.
"""

from __future__ import annotations

from cse151b_comp.prompts import RUNF_SYSTEM_PROMPT_FREE, RUNF_SYSTEM_PROMPT_MCQ
from cse151b_comp.topics import detect_topic

# ─── Branch system prompts (compact; ≤ 250 tokens each) ────────────────

_BRANCH_TRIG = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated, like \\boxed{3, 7, 12}. Do NOT use multiple "
    "\\boxed{} blocks. Do NOT use \\quad or section headers.\n\n"
    "Trigonometry tips:\n"
    "- Keep exact values: \\sin(\\pi/6) = 1/2, \\cos(\\pi/4) = \\sqrt{2}/2.\n"
    "- Use identities (\\sin^2 + \\cos^2 = 1, double-angle, etc.) to "
    "simplify; do not approximate to decimals unless asked.\n"
    "- For periodic functions, state the principal value first; if the "
    "question asks for all solutions in [0, 2\\pi), enumerate them.\n\n"
    "Example:\n"
    "Q: Find all x in [0, 2\\pi) with \\sin(x) = \\sqrt{3}/2.\n"
    "A: \\sin(x) = \\sqrt{3}/2 → x = \\pi/3 or 2\\pi/3. "
    "\\boxed{\\pi/3, 2\\pi/3}"
)

_BRANCH_GEOMETRY = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated, like \\boxed{3, 7, 12}. Do NOT use multiple "
    "\\boxed{} blocks. Do NOT use \\quad or section headers.\n\n"
    "Geometry tips:\n"
    "- Identify the figure first (triangle, circle, polygon).\n"
    "- Apply standard formulas: A_circle = \\pi r^2, A_triangle = "
    "(1/2)bh, Pythagorean theorem.\n"
    "- Keep \\pi and \\sqrt symbolic unless a decimal is requested.\n"
    "- For angle problems, state which angle you computed.\n\n"
    "Example:\n"
    "Q: A right triangle has legs 3 and 4. Find the hypotenuse.\n"
    "A: c = \\sqrt{3^2 + 4^2} = \\sqrt{25} = 5. \\boxed{5}"
)

_BRANCH_LOGIC_PROOF = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated. Do NOT use multiple \\boxed{} blocks.\n\n"
    "Proof / true-false / set-statement tips:\n"
    "- For 'prove' questions: state the conclusion first, then the "
    "reasoning, then the boxed final answer.\n"
    "- For 'true or false' questions: output exactly the word True or "
    "False inside \\boxed{}, e.g. \\boxed{True}.\n"
    "- For 'show that A = B' questions: end with \\boxed{A = B} or "
    "\\boxed{Q.E.D.} as appropriate to the question's wording.\n\n"
    "Example:\n"
    "Q: True or false: every prime greater than 2 is odd.\n"
    "A: A prime > 2 cannot be divisible by 2, so it must be odd. "
    "\\boxed{True}"
)

_BRANCH_STATS_HYP = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated. Do NOT use multiple \\boxed{} blocks.\n\n"
    "Hypothesis-testing tips:\n"
    "- State H0 and Ha, compute the test statistic, then compare to "
    "the critical value or p-value.\n"
    "- For decision: output 'reject' or 'fail to reject' verbatim "
    "(not 'accept' — never accept H0).\n"
    "- For p-values: keep 4 significant figures (e.g. 0.0234 not 0.02).\n"
    "- Order multi-part: (test stat, p-value or decision, conclusion).\n\n"
    "Example:\n"
    "Q: t-stat = 2.45, critical value = 1.96. (a) Reject? (b) Test stat?\n"
    "A: 2.45 > 1.96, so reject H0. \\boxed{reject, 2.45}"
)

_BRANCH_STATS_REG = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated. Do NOT use multiple \\boxed{} blocks.\n\n"
    "Regression / model-fit tips:\n"
    "- Keep R^2 = 1 - SS_res/SS_tot in exact fractional form when "
    "the inputs are clean integers (e.g. 1 - 20/100 = 4/5 = 0.8).\n"
    "- For slope and intercept: report both as comma-separated, in "
    "the order the question asked.\n"
    "- For predicted values: compute via slope·x + intercept.\n\n"
    "Example:\n"
    "Q: SS_total=100, SS_residual=20. Compute R^2.\n"
    "A: R^2 = 1 - 20/100 = 0.8. \\boxed{0.8}"
)

_BRANCH_PROBABILITY = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated. Do NOT use multiple \\boxed{} blocks.\n\n"
    "Probability tips:\n"
    "- Keep fractions exact (e.g. \\boxed{1/4}) unless decimals are "
    "explicitly asked. Don't approximate 1/3 to 0.333.\n"
    "- For 'expected value', compute \\sum x_i p_i.\n"
    "- For Bayes', state the formula: P(A|B) = P(B|A)P(A)/P(B).\n\n"
    "Example:\n"
    "Q: A fair coin is flipped 3 times. P(exactly 2 heads)?\n"
    "A: C(3,2)·(1/2)^3 = 3/8. \\boxed{3/8}"
)

_BRANCH_NUM_THEORY = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated. Do NOT use multiple \\boxed{} blocks.\n\n"
    "Number-theory tips:\n"
    "- For 'find the remainder of N divided by k', compute N mod k as "
    "an integer in [0, k-1].\n"
    "- For 'gcd(a,b)' or 'lcm(a,b)', show factorization briefly.\n"
    "- For 'divisible by': output yes/no inside \\boxed{} or, if asked "
    "'how many', output the count.\n\n"
    "Example:\n"
    "Q: Find the remainder of 1000 divided by 7.\n"
    "A: 1000 = 7·142 + 6. \\boxed{6}"
)


# ─── Run J builder ─────────────────────────────────────────────────────


_BRANCH_PROMPTS = {
    "trig": _BRANCH_TRIG,
    "geometry": _BRANCH_GEOMETRY,
    "logic_proof": _BRANCH_LOGIC_PROOF,
    "stats_hyp_test": _BRANCH_STATS_HYP,
    "stats_regression": _BRANCH_STATS_REG,
    "probability": _BRANCH_PROBABILITY,
    "num_theory": _BRANCH_NUM_THEORY,
    # 'generic' falls through to RUNF_SYSTEM_PROMPT_FREE
}


def build_prompt_runj(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run J: data-driven topic-routed free-form, MCQ unchanged from Run F.

    Routing decision is on the question text alone (no gold needed).
    Returns the same (system, user) tuple shape as other build_prompt_*.
    """
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNF_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    topic = detect_topic(question)
    system = _BRANCH_PROMPTS.get(topic, RUNF_SYSTEM_PROMPT_FREE)
    return system, question


# ─── Ablation builders ─────────────────────────────────────────────────
# Each ablation enables ONE branch only, falls through to Run F generic
# for everything else. Used by scripts/run_j_ablation.sh.


def _make_ablation(enabled_topics: tuple[str, ...]):
    """Return a build_prompt fn that only routes to the specified topics."""

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


build_prompt_runj_trig = _make_ablation(("trig",))
build_prompt_runj_geom = _make_ablation(("geometry",))
build_prompt_runj_logic = _make_ablation(("logic_proof",))
build_prompt_runj_stats = _make_ablation(("stats_hyp_test", "stats_regression"))
build_prompt_runj_prob = _make_ablation(("probability",))
build_prompt_runj_num = _make_ablation(("num_theory",))


__all__ = [
    "build_prompt_runj",
    "build_prompt_runj_trig",
    "build_prompt_runj_geom",
    "build_prompt_runj_logic",
    "build_prompt_runj_stats",
    "build_prompt_runj_prob",
    "build_prompt_runj_num",
]
