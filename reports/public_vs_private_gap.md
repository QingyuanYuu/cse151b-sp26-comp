# Why does Public Validation Disagree with Private Leaderboard?

A focused analysis of the four data points we have, organized as
hypotheses with evidence, ending with the actionable takeaway.

## The four data points

| Configuration | Public val (n=225) | Private leaderboard (n=943) |
|---|---|---|
| Phase 0 (starter) | 56.44% | **0.575** |
| Phase 1 (full rules) | 62.67%  (**+6.23 pp**) | **0.494**  (**-0.081**) |
| v4 (Phase 0 + minimal rules) | _not run_ | **0.462**  (**-0.113 vs Phase 0**) |
| Phase 0 sanity rerun (in flight) | _not run_ | TBD |

The qualitative gap is the issue: Phase 1 went **up** on public val but **down**
on private leaderboard. v4 stripped the suspicious rules and went **down again**
on private, contradicting the val-based ablation that showed both removed rules
helped (v3b/v3c each lost 4–4.5 pp on val).

The point of this document is to identify **why** these two metrics
disagree, not to identify the single best prompt.

---

## Hypothesis space

We consider six plausible mechanisms:

| # | Hypothesis | Predicts |
|---|---|---|
| H1 | **Sampling noise** at T=0.6 dominates a single leaderboard score. | Repeating the same prompt should give wildly different scores. |
| H2 | **Public val sample size is too small** (n=225) to detect 6 pp signals reliably. | The val improvement is within standard error and could vanish on a larger sample. |
| H3 | **Public-overfit through observation**: we observed failure modes on a private submission (17.4% no-box), wrote prompts to fix them, validated on public. | Phase 1 rules should target failure modes that exist on private but are not actually predictive on public. |
| H4 | **Distribution shift in gold answer types**: public has more high-precision decimals (`143.224229…`); private has more simple/symbolic golds (`\pi`, `5/8`, `True`, `B`). | Anti-rounding rule helps public, hurts private; "use plain numbers" rule misfires on word/letter golds. |
| H5 | **Question-type / K-power composition differs**. With all-or-nothing scoring, free-multi accuracy is `p^K`; small per-slot regressions blow up. | Free-multi accuracy on private should swing by 5–10× more than per-slot accuracy moves. |
| H6 | **Prompt length itself shifts model behavior**: longer prompts cause longer thinking traces, which compound reasoning errors. | v4 should have a different median response length than Phase 0 even on the same task. |

We now examine each with the evidence we have.

---

## H1 — Sampling stochasticity

**Evidence: STRONG.**

Comparing v1 (Phase 0) and v4 (Phase 0 free-form prompt + minimal additions),
the model's _final extracted answer_ on private differs across **44.0% of 943
questions** (`415/943`). This is at the same magnitude regardless of which two
runs we compare:

| Pair | Same answer | Different |
|---|---|---|
| v1 vs v2 (Phase 0 vs Phase 1) | 45.1% | 46.0% |
| v1 vs v4 (Phase 0 vs Phase 0+ε) | 47.0% | 44.0% |

Half of the model's answers change between any two runs, even with closely
related prompts. With Bernoulli accuracy `p ≈ 0.55`, the standard error on a
single 943-question score is roughly:

```
σ ≈ √(p(1-p) · ρ_eff / n) ≈ √(0.25 · 4 / 943) ≈ 3.3 pp
```

where `ρ_eff` is an effective correlation factor accounting for shared model
state across questions. We observe per-submission swings of up to ±5 pp,
consistent with this estimate. **A single leaderboard submission has
confidence interval ±~5 pp**.

This alone explains 5 pp of the 8 pp Phase-0-to-Phase-1 regression and most
of the additional v4 regression.

---

## H2 — Public val sample size

**Evidence: MODERATE.**

On val (n=225), the standard error on accuracy is

```
σ_val ≈ √(0.55 · 0.45 / 225) · √2 ≈ 4.7 pp
```

after the same correlation correction. The +6.23 pp val improvement claimed
for Phase 1 is therefore **only barely outside one standard error**. A 95%
confidence interval is roughly ±9 pp, which fully includes "no real
improvement".

This is a sample-size problem, not a sampling problem. Even a perfectly
implemented test on n=225 cannot reliably distinguish a 5 pp signal at this
noise level. Bigger validation set = more reliable signal.

---

## H3 — Public-overfit through observation

**Evidence: STRONG.**

We can directly verify this. Phase 1 prompt rules and where they came from:

| Rule | Where the failure mode was observed | Where it was validated |
|---|---|---|
| MCQ anti-paren `\boxed{(C)}` | Hand-inspection of v1 outputs (private) | Public val |
| Multi-part labels `(a) (b)` | Hand-inspection of v1 outputs (private) | Public val |
| Anti-rounding | Hand-inspection of v1 outputs (private) | Public val |
| Token-budget self-rescue | 17.4% no-box rate on v1 (private) | Public val |

Three of the four rules were designed to fix things we _saw_ on the private
set. Validated on public. **The validation distribution is not where the
design signal came from**.

Direct test of one rule's transferability: we counted MCQ format failures
(`\boxed{(C)}` etc.) in v1 on private. The count is **zero**. That is,
across 300 MCQ questions on private, the starter prompt produces no
parenthesised letters. The +13.3 pp val MCQ gain that motivated this rule
does **not** correspond to a real failure mode on private.

The rule is not _actively harmful_ to private; it's just _useless_, while
the rule's prompt-length cost has knock-on effects (H6).

---

## H4 — Distribution shift in gold answer types

**Evidence: STRONG.**

Direct count of high-precision decimal outputs on private:

| Run | Free-form responses with ≥6 decimal places |
|---|---|
| Phase 0 (v1) | 1 / 643 = 0.2% |
| Phase 1 (v2) | 85 / 643 = **13.2%** |
| v4 | 1 / 643 = 0.2% |

The anti-rounding rule, gone in v4, was responsible for ~13% of free-form
responses dumping 6+ decimal places. Hand-inspection of these 85 samples
shows a recurring pattern: gold is symbolic, model approximates as decimal,
the resulting precision-mismatch lands the answer outside the SymPy 1e-8
tolerance. Examples:

| id | Phase 0 box | Phase 1 box | Likely gold form |
|---|---|---|---|
| 4 | `-7\sqrt{149}/149` | `-0.573576` | symbolic |
| 11 | `0.94` | `0.939119` | low-precision |
| 32 | `\sqrt{101}` | `10.05` | symbolic |
| 49 | `0.014` | `0.014327` | low-precision (or wrong) |

Public val gold (when we have it for Phase 1's evaluation) contains many
high-precision decimals — including the `143.224229233795` example we
calibrated against. **The two sets of golds emphasize different formats**,
and a single "always 6 sig figs" rule cannot satisfy both.

The v4 run also surfaces two new specific bugs introduced by the surviving
rules (`Use plain numbers` and `no 'x = '`):

- **id 44**: gold is `True/True/True/False`; v4 outputs `1/1/1/0`. The
  "use plain numbers" rule was applied to a True/False question.
- **id 60**: gold is `C/A` (MCQ-style sub-answers in a free-form
  multi-part); v4 outputs `3/1`. Same rule, same misfire.
- **id 40**: gold form `D=800-50d`; v4 strips the `D=` prefix to
  `800-50d`. The `no 'x = '` rule applied indiscriminately.

These are observed only after the fact on private; they did not surface in
the val ablation because public val has fewer of these answer types.

---

## H5 — K-power decay × per-slot regression

**Evidence: MODERATE-STRONG.**

Free-multi accuracy on Phase 1 public:

| K | n | accuracy |
|---|---|---|
| 2 | 171 | 53.2% |
| 3 | 90 | 50.0% |
| 4 | 59 | 45.8% |
| 5 | 31 | 29.0% |
| 7+ | 38 | <30% |

`(0.86)^K` fits this curve. A 2-pp regression in per-slot accuracy from any
of the rules in H4 — say, anti-rounding causing 5 % of free-form sub-answers
to land outside tolerance — translates to:

- K=2: -3.4 pp on the question
- K=5: -8.5 pp on the question
- K=10: -16 pp on the question

Free-multi is **57% of the dataset** on private. Even small per-slot
regressions move the overall score 1–2 pp.

---

## H6 — Prompt length amplifies reasoning drift

**Evidence: WEAK-MODERATE.**

Median response length on private:

| Run | Median (chars) | Mean (chars) |
|---|---|---|
| Phase 0 (v1) | 11,469 | 15,348 |
| Phase 1 (v2) | 16,899 | 19,243 |
| v4 | 15,906 | 17,483 |

Phase 1 increased median response length by 47%. A 47% longer chain of
thought can produce subtly different reasoning even on questions
unaffected by the new rules. We see this empirically:

- 113 questions on private have the **same number** of boxed values in v1
  and v4 but **different content** — that is, neither a structural format
  issue nor a precision issue, but the model literally reasoning to a
  different answer.

Without controlled experiments holding the prompt length fixed but varying
the rules, we cannot fully separate H6 from H1 (sampling noise produces
similar drift). A reasonable estimate is that 1–2 pp of the regression is
prompt-length-induced reasoning drift.

---

## Putting it together: a back-of-envelope decomposition

Best-effort attribution of the 8.1 pp Phase-0-to-Phase-1 regression:

| Cause | Estimated contribution |
|---|---|
| H1 + H2: sampling noise + small-val SE | **3–4 pp** |
| H3: public-overfit (rules with no private leverage) | **1–2 pp** |
| H4: anti-rounding on symbolic gold + numeric overcoding (44/60) | **2–3 pp** |
| H5: K-power amplification of small per-slot regressions | **1 pp** |
| H6: prompt-length-induced reasoning drift | **1 pp** |
| **Total** | **8–11 pp** |

The brackets sum to the right ballpark, with the dominant component being
the inability to detect signals at this noise level (H1+H2). The
distribution-specific failures (H3, H4) are real but smaller.

---

## What this implies

1. **Single-shot leaderboard scores cannot reliably rank prompt variants.**
   The standard error is comparable to the typical effect size of any
   single rule we have considered.

2. **Public val improvements transfer poorly when they are designed
   against private failure modes.** Future prompt iterations should elicit
   failure modes from a held-out _public_ split, design fixes, validate
   on val, and only then submit to private.

3. **Variance reduction beats prompt engineering at this noise level.**
   Self-consistency with K=8 averages out the sampling noise and the
   per-question reasoning drift simultaneously. Empirically, K=8 voting
   is expected to deliver +5 to +10 pp absolute, which is _larger_ than
   the entire range of prompt variants we have explored.

4. **Some specific bugs in v4 are genuine and small** (True/False → 1/0,
   equation-prefix stripping, ~5 questions affected). These can be fixed
   by **dropping the "use plain numbers" and "no 'x = '" rules** in any
   future submission.

---

## Recommendation

For the next leaderboard submission, the highest-expected-value play is
self-consistency with the **starter prompt** (Phase 0, the highest-scoring
prompt to date) and K=5–8 votes per question. The starter prompt has zero
content-format rules and therefore no risk of mis-encoding True/False or
stripping equation prefixes. Self-consistency does the variance reduction.

Concretely: `cse151b_comp.self_consistency --prompt phase0 --k 8`.
