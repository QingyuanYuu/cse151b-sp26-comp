# Milestone Report — Results Summary

Generated 2026-05-03 from `results/baseline_v0_val.jsonl` and
`results/baseline_public_v1.jsonl`.

## Headline numbers

| Metric | Phase 0 (starter) | Phase 1 (our prompts) | Δ |
|---|---|---|---|
| **Validation accuracy (n=225)** | **56.44%** (127/225) | **62.67%** (141/225) | **+6.23pp** |
| **Full-set accuracy (n=1126)** | not run (would take 3-5h) | **60.12%** (677/1126) | — |
| Format-failure rate (no `\boxed{}`) | — | 3.6% (40/1126) | — vs 17.4% baseline_v1 on private |

**Compare on the same 225 val questions** (`eval_harness.compare`):
- Both right: 119
- Phase 0 only (regression): **8**
- Phase 1 only (gain): **22**
- Both wrong: 76
- **Net gain: +14 questions** (= +6.22pp ≈ matches +6.23pp above)

## Table 1 — Accuracy by question type (val, n=225)

| Type | n | Phase 0 | Phase 1 | Δ |
|---|---|---|---|---|
| MCQ | 75 | 60.00% | 73.33% | **+13.33pp** |
| Free-form single | 67 | 62.69% | 65.67% | +2.99pp |
| Free-form multi | 83 | 48.19% | 50.60% | +2.41pp |
| **Overall** | 225 | 56.44% | 62.67% | **+6.23pp** |

Most of the gain comes from **MCQ**. Free-multi remains the weakest (gold has K
sub-answers all of which must match — strict).

## Table 2 — Accuracy by question length (val, n=225)

| Bucket | n | Phase 0 | Phase 1 | Δ |
|---|---|---|---|---|
| short (<150) | 59 | 67.80% | 74.58% | +6.78pp |
| medium (150–500) | 119 | 60.50% | 65.55% | +5.04pp |
| long (500–1500) | 39 | 33.33% | 43.59% | **+10.26pp** |
| vlong (≥1500) | 8 | 25.00% | 25.00% | 0pp |

Long questions improve dramatically (the new "token-budget self-rescue" rule
helped). Very long questions (>=1500 chars, n=8) still bottleneck — likely
need higher MAX_TOKENS or self-consistency.

## Table 3 — Accuracy by topic (val, n=225)

| Topic | n | Phase 0 | Phase 1 | Δ |
|---|---|---|---|---|
| calculus | 10 | 70.0% | 80.0% | +10pp |
| ode | 1 | 100% | 100% | 0pp |
| series | 13 | 30.77% | 53.85% | **+23.08pp** |
| geometry | 21 | 52.38% | 57.14% | +4.76pp |
| linalg | 6 | 50.0% | 50.0% | 0pp |
| probability | 16 | 50.0% | 50.0% | 0pp |
| complex | 1 | 0% | 0% | 0pp |
| other | 157 | 59.24% | 64.97% | +5.73pp |

Caveat: topic tagger is regex-based, n is small per topic. Treat as directional
not authoritative.

## Failure mode breakdown — Phase 1 on full public (n=1126, 449 wrong)

| Mode | Count | % of wrong | Notes |
|---|---|---|---|
| Wrong answer (reasoning error) | **340** | **75.7%** | The real ceiling — prompts don't fix this |
| Wrong shape (multi-part mismatch) | 70 | 15.6% | E.g. gold has 5 entries, model emits 4 |
| Truncated (hit MAX_TOKENS) | 35 | 7.8% | Down from ~17% in baseline_v1 (12288 tokens) |
| No `\boxed{}` (truly missing) | 4 | 0.9% | Almost eliminated |

## Top "confident-but-wrong" examples (illustrative)

| id | type | model output | gold | comment |
|---|---|---|---|---|
| 12 | free_multi | `380, 315, **14**, 310` | `380, 315, **13**, 310` | off-by-one in one sub-answer |
| 22 | free_multi | `0.16 \lceil p/16 \rceil` | `0.16*p/16` | over-formatted with LaTeX |
| 23 | free_multi | `2,1,2,1` | `B,A,B,A` | converted MCQ-style answers to integers |
| 21 | free_multi | `70t^4(1-t)^4` | 5-poly tuple | gave only the last sub-answer |

## Figures (saved to `templates/milestone-report/figures/`)

- `fig_acc_by_type.pdf` — Phase 0 vs Phase 1 by question type
- `fig_acc_by_length.pdf` — accuracy collapses on long questions
- `fig_acc_by_topic.pdf` — uneven gains across math topics
- `fig_failure_modes.pdf` — 75% of remaining errors are real reasoning mistakes
- `fig_phase_compare.pdf` — confusion-matrix-style 2×2 of outcomes

## Suggested LaTeX usage

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.7\linewidth]{figures/fig_acc_by_type.pdf}
  \caption{Accuracy by question type on the validation set (n=225).
  Phase~1 prompt rewrites improve overall accuracy by 6.23~pp, with the
  largest gain on multiple-choice (+13.3~pp).}
  \label{fig:acc_by_type}
\end{figure}
```

Cite these in the **Experiments → Results** section.

## What's still missing for the milestone report

- [ ] Kaggle leaderboard score for `submission_v1.csv` (Phase 0 starter prompts on private 943) — go to Kaggle, copy
- [ ] Decision: do we submit `submission_v2_phase1.csv` to Kaggle now? (would burn one of the 5/day) — if yes, do it before claiming the +6pp result
- [ ] Pipeline diagram (TikZ or PowerPoint export) — not generatable from data
- [ ] Team responsibilities, timeline — non-data sections
