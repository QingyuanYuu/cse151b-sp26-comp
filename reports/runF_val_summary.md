# Run F Val Probe — Improvement vs Run E, Still Below Run D

Single-shot K=1 with budget allocator, run on `data/public.jsonl[val_indices]`
(225 stratified val questions). Source: `results/runF_val.jsonl`.

Date: 2026-05-06. Branch: `day1-distill-pool` (cherry-picked Run F prompt
from `origin/jason/dev` commit `431baa3`).

## Headline

| Configuration | Val Acc | Δ vs Run D | Pass Gate (≥64.0%)? |
|---|---|---|---|
| **Run F** (this run) | **132/225 = 58.67 %** | **−4.89 pp** | ❌ −5.33 pp |
| Run E | 56.89 % | −6.67 pp | ❌ |
| Run D (per `runD_summary.md`) | 63.56 % | 0 | — (set the gate) |
| Phase 0 (`baseline_v0_val`) | 56.44 % | −7.12 pp | ❌ |

Run F missed its own commit-message gate (≥ 64.0 %, 1 pp above Run D)
by 5.33 pp. Per the standard fall-back rule ("regress ≥ 2 pp vs Run D
→ revert"), Run F should not promote to public 1126 / private leaderboard.

## By question type (cross-comparison vs Run E)

| Type | n | Run E | Run F | Δ |
|---|---|---|---|---|
| MCQ | 75 | 73.3 % | 74.7 % | +1.4 pp |
| free_single | 66 | 54.5 % | 56.1 % | +1.6 pp |
| free_multi | 84 | 44.0 % | 46.4 % | +2.4 pp |

Run F's "surgical fix" thesis (drop Run D's 4 bug examples + add MCQ
elimination from E + drop topic suffix / 5-shot / be-concise) shows a
**uniform +1.4 to +2.4 pp lift over Run E** in every bucket — direction
is right, but the magnitude is small relative to the Run D ↔ Run E gap.

## Format-rule compliance — still largely ignored

| Forbidden pattern | Run E | Run F |
|---|---|---|
| ≥ 2 `\boxed{...}` blocks | 40.0 % | **44.0 %** |
| `\quad` / `\qquad` in response | 18.7 % | 19.6 % |
| Response > 30 000 chars | 11.5 % | (similar) |
| no-box rate | 4.4 % | 5.3 % |

Run F's MULTI-BOX rate is actually **slightly worse** than Run E's, even
though Run F's prompt is shorter and has the SAME "ONE \\boxed{}
comma-separated" mandate. Hypothesis: this rule is a **model-behavior
ceiling, not a prompt issue** — Qwen3-4B-Thinking has a strong inductive
bias toward emitting one boxed per logical unit, and no amount of
prompt instruction overrides it.

The implication for production: don't bet on prompt rules to fix the
multi-box pattern. Either accept it (judger can recover via
`split_by_comma` if format is consistent) or solve at the LoRA level
(SFT data forces single-box format → model learns to comply).

## Probable causes of Run F < Run D gap

Two competing hypotheses:

1. **Run F genuinely regressed**: Run D had some prompt feature that
   Run F removed — possibly the Tuesday weekday example, which Run F
   replaced with `\sqrt{75}` → `5\sqrt{3}`. Even though Tuesday was
   diagnosed as a bug source (id=5 rounding), it may also have been
   teaching the model "single-value boxed answers are OK" pattern.
2. **Sampling noise**: Run D's reported 63.56 % is a single K=1 point
   estimate; gap-analysis std error is ~3 pp on val_225, so the
   "true" Run D mean is plausibly anywhere in [60.5, 66.5]. Run F at
   58.67 % is just below the lower edge.

To distinguish, **re-run Run D in our exact setup** (`day1-distill-pool`
branch with the routing fix + budget allocator + same val_indices) and
compare both numbers. 13 min of compute, locks the baseline.

## Recommendation

- **Do NOT promote Run F to public 1126 / leaderboard.**
- Either re-test Run D for a fresh baseline (option A above), or
  accept that Run D is the production prompt and proceed to Day 2:
  K=8 SC + LoRA on Run D self-distillation pool.
- Bug-fix philosophy from Run F → Run G is plausible if a future
  ablation isolates which Run D footgun matters most. Not a priority
  this week.

## Files

- `results/runF_val.jsonl` — 225 raw rows. Schema identical to other
  SC outputs; K=1 means `winning_response == all_responses[0]`.
- `/tmp/runF_val.log` — full vLLM trace (not committed).

## Reproduce

```bash
uv run --no-sync cse151b-sc \
    --input data/public.jsonl \
    --val data/val_indices.json \
    --output results/runF_val.jsonl \
    --k 1 --temperature 0.6 --top-p 0.95 \
    --prompt runf \
    --gpu-mem-util 0.92 \
    --max-model-len 26624 --max-num-seqs 128 \
    --allocate-tokens --max-tokens-floor 12288 --max-tokens-ceiling 20480
```

Wallclock: 13.9 min on Blackwell 96GB (model warm).

## Same comparison setup as Run E summary

`reports/runE_val_summary.md` has the same structure for Run E. Run F
is the bug-fix successor; this report should be read alongside that one
for the full prompt-iteration trajectory: Phase 0 → Run B → Run C →
Run D → Run E (probe) → Run F.
