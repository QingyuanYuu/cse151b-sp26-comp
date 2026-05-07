"""Run J — data-driven topic-routed prompts on top of Run F final.

Designed using the empirical PRIVATE corpus scan in
`reports/empirical_topic_distribution.md`. Run J's 9 branches map
1:1 to `cse151b_comp.topics.detect_topic` outputs (excluding generic).

Branches (priority-routed — olympiad first as proof style trumps topic):

    BRANCH              notes
    ─────────────────────────────────────────────────────────
    olympiad           proof / construction / 'find all' (~45 free open)
    stats_hyp_test     hypothesis tests
    stats_regression   R^2, residuals, slope/intercept
    stats_descriptive  mean/median/sd, percentile
    calculus           derivative/integral/limit/series merged
    prob_combi         probability + combinatorics merged
    geometry           triangles, circles, area
    trig               sin/cos and trigonometry
    discrete_math      num_theory + sequences (short-answer style)
    (default GENERIC)  Run F final fallback

Architecture:

- 9 specialized branches + 1 GENERIC (= Run F final) fallback
- Each branch is a complete REPLACE-style prompt (no append-suffix)
- Length kept under ~250 tokens per branch (Phase 1 reasoning-drift
  threshold is 349 tokens)
- MCQ prompt unchanged from Run F final (Run J adds nothing for MCQ;
  the wins are in free-form subdomain routing)

Vs earlier draft of Run J:
- DROPPED logic_proof (only 6 free-form private; mostly false-positive)
- ADDED stats_descriptive (40 free-form, 81% multi-part — must train
  strict K-boxed format)
- ADDED calculus (catch-all for derivative/integral/limit/series; only
  ~5 free-form private but ~79 MCQ — though Run J only routes free)
- MERGED probability + combinatorics → prob_combi (sample-size for
  ablation power: probability alone was 24 in eval pool, too few)
- MERGED num_theory + sequences → discrete_math (same reason)
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
    "- Use identities (\\sin^2 + \\cos^2 = 1, double-angle, etc.); do "
    "not approximate to decimals unless asked.\n"
    "- For periodic equations on [0, 2\\pi), enumerate ALL solutions.\n\n"
    "Example:\n"
    "Q: Find all x in [0, 2\\pi) with \\sin(x) = \\sqrt{3}/2.\n"
    "A: x = \\pi/3 or 2\\pi/3. \\boxed{\\pi/3, 2\\pi/3}"
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
    "- Keep \\pi and \\sqrt symbolic unless a decimal is requested.\n\n"
    "Example:\n"
    "Q: A right triangle has legs 3 and 4. Find the hypotenuse.\n"
    "A: c = \\sqrt{3^2 + 4^2} = \\sqrt{25} = 5. \\boxed{5}"
)

_BRANCH_STATS_HYP = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated. Do NOT use multiple \\boxed{} blocks.\n\n"
    "Hypothesis-testing tips:\n"
    "- State H0 and Ha, compute the test statistic, then compare to "
    "critical value or p-value.\n"
    "- For decision: output 'reject' or 'fail to reject' verbatim "
    "(NEVER 'accept' H0).\n"
    "- For p-values: keep 4 significant figures (e.g. 0.0234 not 0.02).\n"
    "- Order multi-part: (test stat, p-value, decision) as the question asks.\n\n"
    "Example:\n"
    "Q: t-stat = 2.45, critical value = 1.96. (a) Reject? (b) Test stat?\n"
    "A: 2.45 > 1.96, reject H0. \\boxed{reject, 2.45}"
)

_BRANCH_STATS_REG = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated. Do NOT use multiple \\boxed{} blocks.\n\n"
    "Regression / model-fit tips:\n"
    "- Keep R^2 = 1 - SS_res/SS_tot in exact form when inputs are clean "
    "integers (e.g. 1 - 20/100 = 4/5 = 0.8).\n"
    "- For slope and intercept: report comma-separated in question order.\n"
    "- For predicted values: \\hat{y} = slope·x + intercept.\n\n"
    "Example:\n"
    "Q: SS_total=100, SS_residual=20. Compute R^2.\n"
    "A: R^2 = 1 - 20/100 = 0.8. \\boxed{0.8}"
)

_BRANCH_STATS_DESC = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated. Do NOT use multiple \\boxed{} blocks.\n\n"
    "Descriptive-statistics tips:\n"
    "- Mean: \\bar{x} = \\sum x_i / n. Sort data first for median.\n"
    "- Sample standard deviation uses n-1 (Bessel's correction); "
    "population SD uses n.\n"
    "- Round to the precision the question requests; if not stated, "
    "keep 4-5 significant figures.\n"
    "- Multi-part questions usually ask (mean, median, sd) in that order.\n\n"
    "Example:\n"
    "Q: Find mean and SD of {2, 4, 4, 6}.\n"
    "A: mean = 16/4 = 4; SD = \\sqrt{8/3} ≈ 1.633. \\boxed{4, 1.633}"
)

_BRANCH_CALCULUS = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated. Do NOT use multiple \\boxed{} blocks.\n\n"
    "Calculus tips:\n"
    "- Keep derivatives, integrals, limits in exact symbolic form "
    "(e.g. \\boxed{2x}, \\boxed{\\frac{x^3}{3}}).\n"
    "- Use chain / product / quotient rules where applicable.\n"
    "- For convergence: state 'convergent' / 'divergent' verbatim "
    "inside the box.\n\n"
    "Example:\n"
    "Q: Differentiate f(x) = x^3 + 2x.\n"
    "A: f'(x) = 3x^2 + 2. \\boxed{3x^2 + 2}"
)

_BRANCH_PROB_COMBI = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated. Do NOT use multiple \\boxed{} blocks.\n\n"
    "Probability / combinatorics tips:\n"
    "- Keep fractions exact (e.g. \\boxed{1/4}). Don't approximate "
    "1/3 to 0.333 unless decimals are asked.\n"
    "- For 'expected value', compute \\sum x_i p_i.\n"
    "- For 'how many ways', use C(n,k) = n!/(k!(n-k)!) or n!/k! as "
    "appropriate. Result is an integer.\n\n"
    "Example:\n"
    "Q: A fair coin flipped 3 times. P(exactly 2 heads)?\n"
    "A: C(3,2)·(1/2)^3 = 3/8. \\boxed{3/8}"
)

_BRANCH_OLYMPIAD = (
    "You are an expert mathematician solving an olympiad / proof-style "
    "problem. Construct a complete, rigorous argument step-by-step.\n\n"
    "End your response with the final answer inside \\boxed{}. The answer "
    "may be a value, an explicit set ({1, 2, 3} or {n: n is prime}), or "
    "a closed-form expression. Use ONE \\boxed{}; do NOT use multiple "
    "\\boxed{} blocks.\n\n"
    "Olympiad strategy:\n"
    "- Identify what is being asked: prove an identity, find ALL solutions, "
    "construct an example, or determine an extremal value.\n"
    "- For 'find all' problems: prove sufficiency (these work) AND "
    "necessity (no others work). Casework on small values is fine, but "
    "justify why larger values fail.\n"
    "- For 'prove that' problems: end the body with QED and put the "
    "key conclusion (or 'proved') inside \\boxed{}.\n"
    "- Use induction, pigeonhole, or invariants when natural; cite the "
    "principle by name.\n\n"
    "Example:\n"
    "Q: Find all positive integers n such that n^2 + 1 is divisible by 5.\n"
    "A: n^2 mod 5 ∈ {0,1,4}, so n^2+1 mod 5 ∈ {1,2,0}. Divisible iff "
    "n^2 ≡ 4 (mod 5), iff n ≡ ±2 (mod 5). \\boxed{n \\equiv 2, 3 \\pmod 5}"
)

_BRANCH_DISCRETE = (
    "You are an expert mathematician. Solve step-by-step. End your "
    "response with your final answer inside \\boxed{}.\n\n"
    "For multiple sub-answers: use ONE \\boxed{} with values "
    "comma-separated. Do NOT use multiple \\boxed{} blocks.\n\n"
    "Number-theory / sequence tips:\n"
    "- For remainders: N mod k is the integer in [0, k-1].\n"
    "- For gcd / lcm: show prime factorization briefly.\n"
    "- For arithmetic sequences: a_n = a_1 + (n-1)d.\n"
    "- For geometric sequences: a_n = a_1 · r^{n-1}.\n\n"
    "Example:\n"
    "Q: Find the remainder of 1000 divided by 7.\n"
    "A: 1000 = 7·142 + 6. \\boxed{6}"
)


# ─── Run J builder ─────────────────────────────────────────────────────


_BRANCH_PROMPTS = {
    "olympiad": _BRANCH_OLYMPIAD,
    "trig": _BRANCH_TRIG,
    "geometry": _BRANCH_GEOMETRY,
    "stats_hyp_test": _BRANCH_STATS_HYP,
    "stats_regression": _BRANCH_STATS_REG,
    "stats_descriptive": _BRANCH_STATS_DESC,
    "calculus": _BRANCH_CALCULUS,
    "prob_combi": _BRANCH_PROB_COMBI,
    "discrete_math": _BRANCH_DISCRETE,
    # 'generic' falls through to RUNF_SYSTEM_PROMPT_FREE
}


def build_prompt_runj(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run J: data-driven topic-routed free-form, MCQ unchanged from Run F."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNF_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    topic = detect_topic(question)
    system = _BRANCH_PROMPTS.get(topic, RUNF_SYSTEM_PROMPT_FREE)
    return system, question


# ─── Ablation builders (one branch enabled, rest fall through) ─────────


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


# Single-topic ablations (one branch only)
build_prompt_runj_olympiad = _make_ablation(("olympiad",))
build_prompt_runj_trig = _make_ablation(("trig",))
build_prompt_runj_geom = _make_ablation(("geometry",))
build_prompt_runj_stats_hyp = _make_ablation(("stats_hyp_test",))
build_prompt_runj_stats_reg = _make_ablation(("stats_regression",))
build_prompt_runj_stats_desc = _make_ablation(("stats_descriptive",))
build_prompt_runj_calc = _make_ablation(("calculus",))
build_prompt_runj_prob = _make_ablation(("prob_combi",))
build_prompt_runj_discrete = _make_ablation(("discrete_math",))

# Group ablations
build_prompt_runj_stats = _make_ablation(("stats_hyp_test", "stats_regression", "stats_descriptive"))


__all__ = [
    "build_prompt_runj",
    "build_prompt_runj_olympiad",
    "build_prompt_runj_trig",
    "build_prompt_runj_geom",
    "build_prompt_runj_stats_hyp",
    "build_prompt_runj_stats_reg",
    "build_prompt_runj_stats_desc",
    "build_prompt_runj_calc",
    "build_prompt_runj_prob",
    "build_prompt_runj_discrete",
    "build_prompt_runj_stats",
    "build_prompt_runj_final",
]


# ─── Final Run J: dynamic from ablation results ────────────────────────


def _load_final_j_branches() -> tuple[str, ...]:
    """Read which branches to enable from data/runj_final_branches.txt.

    The file is written by `scripts/build_final_j.py` after ablation.
    If missing, default to all 8 branches (full Run J = build_prompt_runj).
    """
    import pathlib

    path = pathlib.Path("data/runj_final_branches.txt")
    if not path.exists():
        return tuple(_BRANCH_PROMPTS.keys())
    enabled = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    # Filter to known branch names (skip typos / orphans).
    return tuple(b for b in enabled if b in _BRANCH_PROMPTS)


def build_prompt_runj_final(question: str, options: list[str] | None) -> tuple[str, str]:
    """Run J with branches determined by ablation results.

    Reads `data/runj_final_branches.txt` at every call (cheap; small file).
    If the file is absent, falls through to the full 8-branch design.
    """
    enabled = _load_final_j_branches()

    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return RUNF_SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"

    topic = detect_topic(question)
    if topic in enabled and topic in _BRANCH_PROMPTS:
        return _BRANCH_PROMPTS[topic], question
    return RUNF_SYSTEM_PROMPT_FREE, question
