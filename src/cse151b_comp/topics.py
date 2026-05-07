"""Empirical topic detection for routing free-form questions to Run J branches.

Designed using the comprehensive PRIVATE corpus scan (multi-signal regex,
priority-routed, sample-verified) — see `reports/empirical_topic_distribution.md`.

Returns one of 10 BRANCH names (not fine-grained topics):

    olympiad          — proof / construction style (highest priority — trumps topic)
    stats_hyp_test    — hypothesis tests (t/F/chi-square, p-value, reject)
    stats_regression  — regression (R^2, residuals, slope/intercept)
    stats_descriptive — mean/median/sd, percentile, quartile
    calculus          — derivative / integral / limit / series merged
    prob_combi        — probability + combinatorics merged (small topics combined for ablation power)
    geometry          — triangles, circles, area, angles
    trig              — sin/cos/tan and trigonometry
    discrete_math     — number theory + sequences (small topics combined)
    generic           — fallback (Run F final prompt)

Dropped vs earlier Run I:
- LOGIC_PROOF (only 6 free-form private; mostly false-positive earlier)
- LINALG (19 MCQ but 0 free-form; Run J's branches target free-form)
- ALGEBRA_POLY (8 free-form private; merge with generic)
- DIFF_EQ, COMPLEX, OPTIMIZATION (each < 10 questions, drop)

Priority order: olympiad first (proof/construction trumps topic — an
olympiad number-theory question needs proof-style reasoning, not a
short-answer DISCRETE_MATH prompt). Then most-specific stats, then
calc, then prob/combi, then geometry/trig, then discrete, then generic.
Stats sub-routes are mutually exclusive.
"""

from __future__ import annotations

import re

# ─── Regex patterns per topic (word-boundary aware to avoid false +) ──

# Olympiad / proof — must come FIRST (proof style overrides any topic).
#
# Built from a manual scan of private.jsonl free-form questions WITHOUT
# `[ANS]` blanks (50 questions, mostly olympiad-style). Strong signals:
#
# 1. Explicit proof verbs: "prove that", "show that", "demonstrate that"
# 2. Contest references: IMO, USAMO, Putnam, AIME, AMC
# 3. Math letter-sets: \mathbb{Z}, \mathbb{N}, \mathbb{R}, \mathbb{Q}, \mathbb{C}
#    (textbook word problems almost never use these)
# 4. Function-definition opener: "Let f :" or "f : A \to B"
# 5. Quantifier setup: "for all integers", "for every prime",
#    "such that for all", "does there exist", "infinitely many"
# 6. Find-all enumeration: "find all positive integers / triples / functions"
# 7. Game setup: "Alice and Bob", "two players", "play a game"
#
# Tuned to OVER-route into olympiad rather than under-route — false
# positive (textbook → proof prompt) is mild; false negative (olympiad
# → short-answer prompt) is severe.
_OLYMPIAD_RE = re.compile(
    # Explicit proof verbs
    r"\bprove\s+(that|the|by|or)\b|\bshow\s+that\b|\bdemonstrate\s+that\b|"
    # Contest references
    r"\b(IMO|USAMO|USAJMO|Putnam|AMC|AIME|olympiad)\b|"
    # \mathbb{Z}, \mathbb{N}, \mathbb{R}, \mathbb{Q}, \mathbb{C} — strong olympiad signal
    r"\\mathbb\s*\{[ZNRQC]\}|"
    # Function-definition: requires explicit \to or \rightarrow (avoids the
    # noisy "any letter : N" trap which matched stats notation like "n: N")
    r"[a-zA-Z]\s*:\s*[\$\\\w\s]{0,15}\\(to|rightarrow)\b|"
    # find/determine all <object>
    r"\b(determine|find)\s+all\s+("
    r"positive\s+integers?|natural\s+numbers?|primes?|integers?|"
    r"real\s+numbers?|triples?|pairs?|polynomials?|functions?|"
    r"values?\s+of|solutions?|(n|m|k|x|y)\s+(such|with|that)"
    r")\b|"
    # Quantifier setups
    r"\bsuch\s+that\s+for\s+(all|every|any)\b|"
    r"\bfor\s+(every|all|any)\s+(positive\s+integer|prime|integer|real)\b|"
    r"\bdoes\s+there\s+exist\b|\binfinitely\s+many\b|"
    # Game-theoretic / olympiad combinatorics setup
    r"\b(Alice\s+and\s+Bob|two\s+players?\s+play|play\s+a\s+game)\b|"
    # Constructive language
    r"\bconstruct\s+(a|an|the)\s+\w+\s+(such|that|with)\b|"
    # "Determine the maximum/minimum POSSIBLE / VALUE OF / REAL NUMBER" —
    # the qualifier disambiguates olympiad ("determine the max possible
    # value of x") from textbook ("find the minimum of the data set").
    r"\b(determine|compute|find)\s+the\s+(maximum|minimum|largest|smallest|"
    r"least|greatest)\s+(possible|positive\s+integer|real\s+number|"
    r"value\s+of\s+(?!the\s+(mean|median|sample|sum)))|"
    # "Let $X$ be a ..." opener with abstract objects
    r"\bLet\s+\$?[A-Z][_a-z\d]*\$?\s+be\s+(a|an|the|several)\s+"
    r"(set|sequence|polygon|polynomial|polynomi|graph|"
    r"region|function|positive\s+integer|prime|"
    r"finite|simple|convex|family|collection|tree)|"
    r"\bLet\s+\$?[a-z]\$?\s+(and\s+\$?[a-z]\$?\s+)?be\s+"
    r"(a|an|the)?\s*positive\s+integers?\b",
    re.IGNORECASE,
)

_STATS_HYP_RE = re.compile(
    r"\bhypothes(is|es)\b|\bnull\s+hypothes|\balternative\s+hypothes|"
    r"\bp[\s-]?value\b|\bt[\s-]?test\b|\bf[\s-]?test\b|"
    r"\bchi[\s-]?square\b|\breject(ing|ed)?\s+(h_?0|the\s+null)|"
    r"\bsignifican(ce|t)\s+level|\btype\s+i+\s+error\b|\balpha\s+level\b|"
    r"\btest\s+statistic\b|\bcritical\s+value\b",
    re.IGNORECASE,
)

_STATS_REG_RE = re.compile(
    r"\bregression\b|\bleast\s+squares\b|\br[\s-]?squared?\b|\br\^2\b|"
    r"\bcoefficient\s+of\s+determination\b|\bresidual(s)?\b|"
    r"\b(slope|intercept)\s+of\s+the\s+(line|regression)|"
    r"\bpredicted\s+value\b|\bfitted\s+(value|line)\b|"
    r"\b(beta|β)[\s_]*(0|1|hat)\b",
    re.IGNORECASE,
)

_STATS_DESC_RE = re.compile(
    r"\b(sample|population)\s+mean\b|\bmean\s+of\s+(the\s+)?(sample|data)|"
    r"\bmedian\s+of\b|\bstandard\s+deviation\b|\bvariance\s+of\b|"
    r"\binterquartile\s+range\b|\b(first|third|upper|lower)\s+quartile\b|"
    r"\bpercentile\b|\bbox(\s|-)?plot\b|\bhistogram\b|"
    r"\bfrequency\s+(distribution|table)\b|\bz[\s-]?score\b",
    re.IGNORECASE,
)

# Calculus: any derivative / integral / limit / series → CALCULUS branch
_CALC_RE = re.compile(
    r"\bderivative\b|\bdifferenti(ate|ation)\b|\brate\s+of\s+change\b|"
    r"\btangent\s+line\b|\bf'\(|\bdy/dx\b|\\frac\{d|\bchain\s+rule\b|"
    r"\bproduct\s+rule\b|\bquotient\s+rule\b|"
    r"\bintegral\b|\bintegrat(e|ion|ing)\b|\bantiderivative\b|"
    r"\barea\s+under\b|\\int\b|\briemann\s+sum\b|"
    r"\b(definite|indefinite)\s+integral\b|"
    r"\blimit\b|\blim_\{|\\lim\b|\bapproaches\b|\bl'?hopital|"
    r"\bindeterminate\b|"
    r"\binfinite\s+sum\b|\bgeometric\s+series\b|\b(taylor|maclaurin|power)\s+series\b|"
    r"\b(ratio|root|integral|comparison)\s+test\b|"
    r"\b(convergent|divergent)\s+series\b",
    re.IGNORECASE,
)

# Probability + combinatorics merged
_PROB_COMBI_RE = re.compile(
    r"\bprobabil|\bP\s*\(|\bexpected\s+value\b|\bbayes|"
    r"\bbinomial\s+(distribution|coefficient|probability)\b|\bpoisson\b|"
    r"\b(geometric|uniform|normal|exponential)\s+distribution\b|"
    r"\brandom\s+variable\b|\bstochastic\b|\bmarkov\b|"
    r"\b(conditional|joint|marginal)\s+probabil|"
    r"\bindependent\s+events\b|\bmutually\s+exclusive\b|"
    r"\bpermutation(s)?\b|\bcombination(s)?\b|\bn\s+choose\s+k\b|"
    r"\bfactorial\b|\bnumber\s+of\s+ways\b|\bhow\s+many\s+ways\b|"
    r"\barrangement(s)?\b",
    re.IGNORECASE,
)

# Geometry — high-precision keywords
_GEOM_RE = re.compile(
    r"\btriangle(s)?\b|\bcircle(s)?\b|\brectangle(s)?\b|\bsquare(s)?\b|"
    r"\bpolygon(s)?\b|\b(hexagon|pentagon|octagon)\b|"
    r"\bperimeter\b|\barea\s+of\b|\bvolume\s+of\b|\bcircumference\b|"
    r"\bdiameter\b|\bradius\b|"
    r"\bangle(s)?\b|\b(parallel|perpendicular)\b|"
    r"\bpythagorean\b|\bhypotenuse\b|"
    r"\bvertic(es|al)\b|\baltitude\b|\bcentroid\b|"
    r"\b(sphere|cylinder|cone|cube)\b",
    re.IGNORECASE,
)

# Trig — must use word boundary to avoid 'since'/'consist' false positives
_TRIG_RE = re.compile(
    r"\\sin\b|\\cos\b|\\tan\b|\\sec\b|\\csc\b|\\cot\b|"
    r"\b(sin|cos|tan|sec|csc|cot)\s*\(|"
    r"\b(sin|cos|tan)\s*[xyzθαβ\\]|"
    r"\b(sine|cosine|tangent|secant|cosecant|cotangent)\b|"
    r"\btrigonom|\bradian(s)?\b|\bunit\s+circle\b|"
    r"\blaw\s+of\s+(sines|cosines|tangents)\b",
    re.IGNORECASE,
)

# Discrete math: num theory + sequences (small topics merged)
_DISCRETE_RE = re.compile(
    r"\bprime\s+(number|factor|factorization)\b|\bprime\b|"
    r"\bdivisib(le|ility)\b|\bmodulo\b|\bmod\s+\d|\bremainder\b|"
    r"\bgcd\b|\blcm\b|\bcongruent\s+modulo\b|\bdiophantine\b|"
    r"\bperfect\s+square\b|\bconsecutive\s+integer|"
    r"\b(arithmetic|geometric)\s+sequence\b|"
    r"\bcommon\s+(ratio|difference)\b|"
    r"\brecurrence\b|\bfibonacci\b|"
    r"\bnth\s+term\b|\b(a_n|a_\{n\})|\brecursive\s+formula\b",
    re.IGNORECASE,
)


# ─── Detection (priority-routed, mutually exclusive) ───────────────────


def detect_topic(question: str) -> str:
    """Return the Run J branch name for a free-form question.

    Priority order matters: most-specific stats first, then calculus
    (its keywords are unambiguous), then prob/combi (specific math
    objects), then geometry / trig, then discrete (catch-all for
    number / sequence questions), then generic.

    Returns one of: olympiad, stats_hyp_test, stats_regression,
    stats_descriptive, calculus, prob_combi, geometry, trig,
    discrete_math, generic.
    """
    q = question  # patterns are IGNORECASE

    if _OLYMPIAD_RE.search(q):
        return "olympiad"
    if _STATS_HYP_RE.search(q):
        return "stats_hyp_test"
    if _STATS_REG_RE.search(q):
        return "stats_regression"
    if _STATS_DESC_RE.search(q):
        return "stats_descriptive"
    if _CALC_RE.search(q):
        return "calculus"
    if _PROB_COMBI_RE.search(q):
        return "prob_combi"
    if _GEOM_RE.search(q):
        return "geometry"
    if _TRIG_RE.search(q):
        return "trig"
    if _DISCRETE_RE.search(q):
        return "discrete_math"
    return "generic"


# Helper for downstream stats / debugging
ALL_BRANCHES = (
    "olympiad",
    "stats_hyp_test",
    "stats_regression",
    "stats_descriptive",
    "calculus",
    "prob_combi",
    "geometry",
    "trig",
    "discrete_math",
    "generic",
)


__all__ = ["detect_topic", "ALL_BRANCHES"]
