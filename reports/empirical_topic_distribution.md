# Empirical Topic Distribution — Public + Private Corpus

Direct keyword-based topic analysis on the 2069-question competition corpus
(public 1126 + private 943). Used to drive Run J design (data-driven topic
routing) instead of Run I's intuition-based 5-way split.

Date: 2026-05-06. Source: `data/public.jsonl` + `data/private.jsonl`.

## Headline finding (CORRECTED — first version had broad-keyword false positives)

The original analysis used substring keywords (`"sin"`, `"cos"`) that
matched false positives like "consist", "since", "construct". Real
counts using word-boundary detection (`\bsin\b`) on private free-form:

```
Topic              real hits   share   first-pass(buggy)  vs Run I
───────────────────────────────────────────────────────────────────
generic            367         57%     —                  default
geometry            74         12%     121 (13%)          ❌ missed
logic_proof         64         10%     176 (19%)          ❌ missed
stats_hyp_test      39          6%     45                 ✅ partial
stats_regression    33          5%     33                 ✅ partial
probability         26          4%     37                 ✅ partial
trig                25          4%     358 (38%) ← buggy  ❌ missed
num_theory          15          2%     69                 ❌ missed
```

The "trig is 41%" first-pass story was wrong. **Trig is actually
modest (~4%)**. The real big-block missed sub-domains are
**GEOMETRY (12%)** and **LOGIC_PROOF (10%)** — both still 0% covered
by Run I.

Run I's LINALG branch is genuinely wasted (corpus has < 30 questions
total; LINALG content is mostly in MCQ where Run J doesn't change
routing). CALCULUS branch captures only 4-8 of ~30 calc questions
even with strict keywords.

## Private structure

```
943 total = 300 MCQ (32%) + 643 free-form (68%)

MCQ option-count: 89% are 10-option (267/300)

Free-form parts (sub-answers per Q):
  1 part:  305 (47%)
  2 parts: 141 (22%)
  3 parts:  77 (12%)
  4 parts:  58  (9%)
  5+ parts: 62 (10%)
```

## Private free-form topic distribution (n=643)

| Topic | count | % | multi-part % |
|---|---|---|---|
| **trig** | **265** | **41%** | 60% |
| **word_problem** | **208** | **32%** | 51% |
| **algebra_general** | **132** | **21%** | 47% |
| **logic_proof** | **109** | **17%** | 55% |
| **geometry** | **84** | **13%** | 44% |
| stats_descriptive | 62 | 10% | **81%** |
| stats_hyp_test | 45 | 7% | **80%** |
| set_theory | 43 | 7% | 63% |
| probability | 37 | 6% | 57% |
| algebra_poly | 35 | 5% | 54% |
| stats_regression | 33 | 5% | 61% |
| optimization | 22 | 3% | 32% |
| num_theory | 20 | 3% | 20% |
| stats_anova | 10 | 2% | 80% |
| sequences | 10 | 2% | 10% |
| complex | 8 | 1% | 50% |
| calculus_limit | 8 | 1% | 25% |
| combinatorics | 7 | 1% | 14% |
| calculus_diff | 4 | 1% | 75% |
| calculus_series | 4 | 1% | 75% |
| linalg | 2 | 0% | 50% |
| diff_eq | 1 | 0% | 100% |
| **\<untagged generic\>** | **117** | **18%** | — |

(Topics multi-tag: a question with both "trig" and "word_problem"
keywords appears in both rows.)

## Private MCQ topic distribution (n=300)

| Topic | count | % |
|---|---|---|
| word_problem | 135 | 45% |
| **trig** | **93** | **31%** |
| logic_proof | 67 | 22% |
| **num_theory** | **49** | **16%** |
| **sequences** | **45** | **15%** |
| **calculus_series** | **44** | **15%** |
| calculus_int | 37 | 12% |
| geometry | 37 | 12% |
| algebra_general | 33 | 11% |
| algebra_poly | 28 | 9% |
| linalg | 24 | 8% |
| set_theory | 22 | 7% |
| probability | 20 | 7% |
| optimization | 17 | 6% |
| combinatorics | 15 | 5% |
| \<untagged generic\> | 26 | 9% |

**MCQ vs free-form distribution differs**: MCQ has more sequences
(15% vs 2% in free), num_theory (16% vs 3%), calculus_series (15%
vs 1%). These topics are predominantly tested as MCQ in this course.
linalg too (8% MCQ vs 0.3% free).

## Stats sub-domain note: 80% multi-part

`stats_descriptive` (81%), `stats_hyp_test` (80%), `stats_anova` (80%)
are dominated by multi-part questions. These are exactly where the
"K boxed comma-separated" format compliance matters most. LoRA SFT
data balancing must over-sample these topics or LoRA misses the format
training signal.

## Topic priorities for Run J (CORRECTED ROI ranking)

By real private hits (priority-routed, mutually exclusive):

### Tier 1 (best ROI — biggest unclaimed buckets)
1. **GEOMETRY** (74 private free-form, 12%) — 0 Run I coverage
2. **LOGIC_PROOF** (64, 10%) — 0 Run I coverage
3. **STATS_HYP_TEST** (39, 6%) — Run I has stats but generic-merged

### Tier 2 (modest gain)
4. **STATS_REGRESSION** (33, 5%)
5. **PROBABILITY** (26, 4%) — Run I already covers
6. **TRIG** (25, 4%) — small but easy add (clean keyword detection)

### Tier 3 (small but include for completeness)
7. **NUM_THEORY** (15, 2%) — also 49 in private MCQ if we route MCQ

### Drop entirely
- **LINALG** branch (corpus has < 30 questions; mostly MCQ)
- CALCULUS sub-branches (no critical mass)
- DIFF_EQ (1 question)

### Realistic gain estimate

If each branch saves 5-10% of its target topic on private:
- geom: 4-7 questions
- logic_proof: 3-6
- stats (combined): 4-7
- prob: 1-3
- trig: 1-3
- num_theory: 0-1
- **Total: 13-27 saved questions = +1.4 to +2.9 pp on private**

Modest but real. Comparable to Run B → Run F's prompt-iteration gains
(+3.2 pp) but harder to extract since most easy fixes are already in
Run F.

## Run J design (next section)

8 branches:
- TRIG (Tier 1)
- GEOMETRY (Tier 1)
- LOGIC_PROOF (Tier 1)
- STATS_HYPOTHESIS (Tier 2; t/F/chi-square specific)
- STATS_REGRESSION (Tier 2; R², residuals)
- PROBABILITY (preserve from Run I)
- NUM_THEORY (Tier 2; MCQ bonus)
- GENERIC = Run F final (default fallback)

Each branch is REPLACE-style (full prompt), not Run E's append-style.
Strict keyword routing with generic fallback. Stats descriptive merges
into STATS_REGRESSION's branch (similar form), or kept generic.

## Ablation strategy

To verify each branch contributes signal (not just adds prompt
complexity), run per-topic val ablation:

```
J_base    = Run F final (control)
J_trig    = Run F + TRIG branch only
J_geom    = Run F + GEOMETRY branch only
J_logic   = Run F + LOGIC_PROOF only
J_stats   = Run F + 2 STATS sub-branches
J_prob    = Run F + PROBABILITY only
J_num     = Run F + NUM_THEORY only
J_full    = Run F + all 7 branches (final candidate)
```

Each variant is tested on a **topic-filtered val subset** (val_225 ∩
target_topic) — typically 20-90 questions. This isolates the branch's
effect from full-set noise. Final J keeps only branches with measurable
gain on their topic subset.

## LoRA SFT data balancing implication

Independent of Run J, the SFT training data should reflect this
distribution. After K=32 SC produces ~700 (input, target) pairs, check
that topic distribution roughly matches private. If under-represented:

- Trig under-rep → can't fix without resampling (the model already
  failed pass@32 on those)
- Stats under-rep → over-sample remaining stats correct cases (oversample
  with replacement OK for small datasets like this)

`prepare_sft_data.py` should optionally output topic distribution stats
so we can spot-check before training.

## Files

- This report: `reports/empirical_topic_distribution.md`
- Source data: `data/public.jsonl`, `data/private.jsonl`
- Topic keyword bank: see `src/cse151b_comp/topics.py` (next commit)
- Reproduction: see code in commit message attaching this report.
