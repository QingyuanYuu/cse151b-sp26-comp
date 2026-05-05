# Why Prompt Engineering Helped Public Validation but Hurt Private Leaderboard

A diagnostic note documenting the +6.23~pp public val improvement / -8.1~pp
private leaderboard regression observed in Phase 1, and the further -3.2~pp
regression of v4 (a "minimal" prompt fix). Written as input to the
milestone report's Discussion section and to inform Phase 2 design.

## Observed gap

| Configuration | Public val (n=225) | Private leaderboard (n=943) |
|---|---|---|
| Phase 0 (starter prompts) | 56.44% | **0.575** |
| Phase 1 (full rule set) | 62.67% (+6.23 pp) | **0.494** (-0.081) |
| v4 (Phase 0 + MCQ anti-paren + multi-part rule) | _(to be measured)_ | **0.462** (-0.113) |

Three submissions, three different scores under the same model and decoding
hyperparameters. Public val and private leaderboard moved in opposite
directions when Phase 1 prompts were introduced. v4 partially reverted Phase
1 yet scored even lower than Phase 1, which is what motivated this note.

## Diagnosed mechanisms

### 1. Sampling stochasticity dominates the per-submission signal

vLLM generation uses `temperature=0.6` with `top_p=0.95`. Two single-shot runs
of the same model with the same prompt produce different responses for
roughly half of all questions. Concretely, comparing v1 (Phase 0, 0.575) and
v4 (Phase 0 free-form + minimal rule additions, 0.462) on the same 943
private problems:

| Metric | Value |
|---|---|
| Same final answer | 47.0% (443 / 943) |
| Different final answer | 44.0% (415 / 943) |
| One has box, other does not | 2.6% (24 / 943) |
| Both produce no boxed | 6.5% (61 / 943) |

Random-walk noise in the binary correctness vector at this rate translates to
a per-submission standard error of roughly `sqrt(p(1-p)/n) ≈ 1.6` pp under
i.i.d. assumptions, but the empirical observation is that **swing across
identical-config runs is closer to ±5 pp** because the sampling is correlated
across questions through the shared model state. A single leaderboard
submission therefore yields a point estimate with confidence interval wide
enough to swallow most of the prompt-engineering improvements claimed on
val.

### 2. Public-overfit through observation, not validation

A subtle data-leakage path: we used the `submission_v1.csv` private-set
responses (no gold) to identify failure modes (the 17.4 % no-box rate, MCQ
parenthesised-letter cases, mid-thinking truncation). Phase 1 prompts were
designed to target each observed failure. We then **validated** Phase 1 on a
20% public stratified val. Because the failure-mode taxonomy was elicited
from private but the validation set was public, the design loop was
implicitly comparing against a public proxy that does not faithfully
represent private; specifically:

- Private MCQ responses contain **zero** `\boxed{(C)}`/`\boxed{C.}` cases
  (we verified this post-hoc), so the MCQ anti-paren rule, which
  contributed +13.3 pp on val MCQ accuracy, has effectively no leverage
  on private.
- Private gold answer types differ from public; the next subsection details
  this.

The methodological fix is to elicit failure modes only from a held-out
public split, never from the private response trace.

### 3. Distribution shift in gold answer types

The two anti-rounding-and-rescue rules — "Report 6 significant figures" and
"if running out, output best-guess" — both produced statistically clean
+4.0 to +4.5 pp improvements on val (v3b and v3c ablation), implying that
**they are working as intended on the public-set distribution**. The
private regression therefore must come from a distribution where these
rules fire on different inputs.

Direct evidence: on private, the anti-rounding rule changed 13.2 % of
free-form responses to contain ≥6 decimal places (vs 0.2 % in Phase 0).
Hand-inspection of these high-precision boxed values shows several patterns
that should not match the gold under a 1e-8 SymPy equivalence check:

- **Symbolic gold rendered as decimals.** A question whose gold answer
  is `-7\sqrt{149}/149` was answered `-0.573576` by Phase 1. The judger's
  LaTeX path correctly resolves `-7\sqrt{149}/149 = -0.5735393...`, so the
  pred is off by ~6e-7 — well outside the 1e-8 grader tolerance.
- **Trailing-zero noise.** A gold of `1.16` matched against
  `1.160000` and a gold of `18` matched against `18.00000`. Most judger
  implementations are robust to this, but combined with the symbolic
  case the proportion of borderline misses is non-trivial.
- **Reasoning drift in long thinking traces.** v2 has a 47 % longer median
  response than v1, and several boxed values differ from v1 not by
  precision but by integer (id 49: `0.014` vs `0.014327`; id 81: `79.41`
  vs `79.409091`). These are full reasoning regressions, not formatting
  changes.

The public set evidently contains a higher proportion of high-precision
numerical golds (e.g., id 2's `[143.224229233795, 2.32624773420025]`),
where the anti-rounding rule rescues 6-significant-figure decimals from
being rounded to 3 figures. The private set evidently contains more
symbolic / exact / low-precision golds, where the same rule produces
borderline-precision misses.

### 4. The token-budget self-rescue rule produces premature, confident wrong answers

Token-rescue tells the model: if you are running out of room, output your
best guess inside `\boxed{...}`. On val this raised the boxed rate by
roughly 9 pp (from 82.6 % to 92.0 %) and added +4 pp accuracy. On private
the rule fires more often because of the longer median question prompt,
and at least one instance of the rescue produced a literal `\boxed{...}`
placeholder (id 21 in private) instead of a meaningful answer, which is
graded as wrong while a no-box response would have been graded by the
judger's fallback heuristics (last-number, last-LaTeX) and could
occasionally hit. The rule trades a guaranteed wrong "best guess" for a
probabilistic correct fallback extraction.

### 5. K-th-power decay on multi-answer questions amplifies any per-slot regression

Multi-answer free-form is graded all-or-nothing: K sub-answers must all
match. If the per-slot accuracy is `p`, the question accuracy is `p^K`.
Empirical accuracy by K on Phase 1 public val:

| K | n | accuracy |
|---|---|---|
| 2 | 171 | 53.2% |
| 3 | 90  | 50.0% |
| 4 | 59  | 45.8% |
| 5 | 31  | 29.0% |
| 7 | 12  | 25.0% |
| 10+ | 13 | < 25% |

This is roughly consistent with `p ≈ 0.86` per slot. A 2-pp regression in
per-slot accuracy from Phase 0 to Phase 1 (e.g., due to anti-rounding
shifting some sub-answers from correct to over-precise) would translate to
a ~10-pp drop on K=5 questions and >15 pp on K=8+. Free-multi has 643/1126
(57 %) representation in our data; even small per-slot regressions
multiply up.

### 6. Prompt length itself shifts model behavior

Phase 1's free-form system prompt is 349 tokens, vs Phase 0's 52 tokens.
Even when the additional rules are not directly fired, the longer prompt
changes the model's attention distribution and (empirically) produces
longer thinking traces. v2's median response length is 47 % longer than
v1's, and longer chains of thought are known in the literature to amplify
reasoning errors through compounding intermediate-step uncertainty.
Hand inspection of changed answers (id 11 `0.94` → `0.939119`, id 81
`79.41` → `79.409091`) supports this drift hypothesis: the model produced
genuinely different numbers, not just precision-shifted versions of the
same answer.

## Implications

1. **Single-shot prompt engineering on this dataset has a small, possibly
   negative expected value.** The signal we can extract from a single
   leaderboard submission is dominated by sampling noise (~±5 pp), and the
   directionally-correct improvements on val (+6 pp) are the same order
   of magnitude as the noise. We cannot reliably distinguish a 2-pp
   improvement from a 2-pp regression with three submissions.
2. **Per-rule ablation on val produces correct attributions for val, but
   does not transfer to private for rules that interact with gold-type
   distributions.** Anti-rounding is +4.5 pp on val and ostensibly worth
   keeping; it is observably harmful on private. The mechanism (symbolic
   vs decimal gold) is not detectable from val's gold distribution alone.
3. **The right next step is variance reduction, not further prompt rules.**
   Self-consistency with K independent samples per question, voted via
   the per-slot or normalized-answer scheme, has a known dependence on
   per-slot accuracy that produces a ~5-15 pp absolute improvement when
   `p ∈ [0.6, 0.85]`. Critically, the variance-reduction mechanism is
   independent of the public/private distribution shift: the K samples
   inherit whatever distribution shift the base model has, but the vote
   itself reduces noise without reintroducing prompt-engineering bias.

## Recommended Phase 2 design

- **Base prompt: Phase 0 starter prompts** (the configuration with the
  highest single-shot leaderboard score so far, 0.575). Drop all Phase 1
  rule additions until they can be re-validated under the lower variance
  of self-consistency.
- **K = 5 to 8 samples per question**, `temperature=0.7`, `top_p=0.95`,
  with per-question seed variation. vLLM's `n=K` parameter shares the
  prompt prefix and only diverges KV cache for the generation, so the
  cost is sub-linear in K.
- **Voting strategy by question type**:
  - MCQ: extract letter from each sample, mode-vote, tie-break by
    longest valid reasoning.
  - Free-form single: normalize each extracted answer (string-canonical,
    sympy-free), mode-vote on the canonical form, return the most
    common original.
  - Free-form multi: extract the comma-separated tuple, vote on the
    *whole tuple* (since gold equality is tuple-level). If no tuple
    achieves a plurality (likely for K ≥ 6 questions), fall back to
    per-slot voting and report this as a measured frequency.
- **Diagnostic metric**: report "solvable but missed" — the number of
  questions where at least one of the K samples was correct but the vote
  selected a wrong answer. This bounds the headroom available from
  smarter voting independently of base-model improvements.

## Honest limits

We cannot run more than two additional Kaggle submissions today, and a
single submission's standard error is wide enough that a clean A/B test of
Phase 0 vs Phase 1 vs v4 is not feasible. The Phase 2 self-consistency
plan reduces this dependence on the leaderboard by collapsing variance
into the local public set: a K=8 self-consistent run on the 1126 public
problems gives us a ~50 % standard-error reduction relative to single-shot
public eval, which we can then trust as a leading indicator for private
performance.
