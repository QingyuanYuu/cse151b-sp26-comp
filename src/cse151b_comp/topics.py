"""Empirical topic detection for routing free-form questions.

Keywords + priorities are derived from
`reports/empirical_topic_distribution.md`'s analysis of the actual
public + private corpus, not from intuition. The analysis identified
TRIG / LOGIC_PROOF / GEOMETRY as the three biggest sub-domains Run I
missed, and showed LINALG is a 30-question niche where keyword
matching is unreliable — so LINALG has no branch in Run J.

Detection priority (most-specific first):

    stats_hyp_test
    stats_regression
    probability
    geometry
    num_theory
    logic_proof
    trig
    (default) generic    ← Run F final

Stats-descriptive merges into generic (the stats_regression branch's
prompt format already handles descriptive multi-part questions).

Returns one of: "trig" | "geometry" | "logic_proof" | "stats_hyp_test"
| "stats_regression" | "probability" | "num_theory" | "generic"
"""

from __future__ import annotations

import re

# ─── Topic keyword banks (priority-ordered detection) ──────────────────

# Tier-2 specific stats branches: detected first so they don't fall through
# to the more general probability or generic.
_STATS_HYP_KW = (
    "hypothesis",
    "null hypothesis",
    "alternative hypothesis",
    "p-value",
    "p value",
    "t-test",
    "t-statistic",
    "f-test",
    "f-statistic",
    "chi-square",
    "chi square",
    "reject",
    "significance level",
    "type i error",
    "type ii error",
    "alpha level",
)
_STATS_REG_KW = (
    "regression",
    "least squares",
    "r-squared",
    "r squared",
    "r^2",
    "coefficient of determination",
    "residual",
    "slope",
    "intercept",
    "predicted",
    "fitted",
)

# Probability & combinatorics — distinct enough from stats
_PROB_KW = (
    "probability",
    "p(",
    "expected value",
    "bayes",
    "binomial distribution",
    "poisson",
    "geometric distribution",
    "uniform distribution",
    "normal distribution",
    "random variable",
    "stochastic",
    "markov",
    "conditional probability",
    "independent events",
    "mutually exclusive",
)

# Num theory — small but distinct
_NUM_KW = (
    "prime",
    "divisible",
    "divisibility",
    "modulo",
    "remainder",
    "factor of",
    "gcd",
    "lcm",
    "congruent modulo",
    "diophantine",
    "perfect square",
    "consecutive integer",
)

# Geometry
_GEOM_KW = (
    "triangle",
    "circle",
    "rectangle",
    "polygon",
    "perimeter",
    "area of",
    "volume of",
    "circumference",
    "diameter",
    "radius",
    "angle",
    "degrees",
    "parallel",
    "perpendicular",
    "pythagorean",
    "similar triangle",
    "congruent",
    "vertices",
    "altitude",
    "median of",
    "centroid",
)

# Logic / proof
_LOGIC_KW = (
    "prove",
    "disprove",
    "show that",
    "true or false",
    "if and only if",
    "iff",
    "implies",
    "contrapositive",
    "converse",
    "let ",
    "biconditional",
    "negation",
)

# Trig — biggest single bucket
_TRIG_KW = (
    "sin(",
    "cos(",
    "tan(",
    "sec(",
    "csc(",
    "cot(",
    "\\sin",
    "\\cos",
    "\\tan",
    "\\sec",
    "\\csc",
    "\\cot",
    "trigonometric",
    "trigonometry",
    "radian",
    "amplitude",
    "period of",
    "phase shift",
    "law of sines",
    "law of cosines",
)
# Stricter: word-boundary trig functions (avoid "since" / "consider" /
# "construct" / "increase" false positives but catch "sin x", "cos θ").
_TRIG_RE = re.compile(
    r"\b(sin|cos|tan|sec|csc|cot|sine|cosine|tangent|cotangent|secant|cosecant)\b",
    re.IGNORECASE,
)


def detect_topic(question: str) -> str:
    """Route a free-form question to one of 8 buckets (priority order).

    MCQ questions should be routed elsewhere — this function assumes
    free-form context. For MCQ, use the MCQ system prompt unchanged.

    Returns one of: trig / geometry / logic_proof / stats_hyp_test /
    stats_regression / probability / num_theory / generic.
    """
    q = question.lower()

    # Most specific stats branches first
    if any(kw in q for kw in _STATS_HYP_KW):
        return "stats_hyp_test"
    if any(kw in q for kw in _STATS_REG_KW):
        return "stats_regression"
    if any(kw in q for kw in _PROB_KW):
        return "probability"
    if any(kw in q for kw in _GEOM_KW):
        return "geometry"
    if any(kw in q for kw in _NUM_KW):
        return "num_theory"
    if any(kw in q for kw in _LOGIC_KW):
        return "logic_proof"
    # Trig last among Tier-1 because its keywords (sin/cos) can
    # appear in stats / geometry questions; we don't want to route
    # a stats t-test question to trig just because someone mentions
    # cosine in a sample calc.
    if any(kw in q for kw in _TRIG_KW) or _TRIG_RE.search(q):
        return "trig"

    return "generic"


__all__ = ["detect_topic"]
