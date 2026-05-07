# Run B Val Probe — Reveals Val ↔ Leaderboard Decoupling

Single-shot K=1 with v1 budget allocator on `data/public.jsonl[val_indices]`
(225 stratified val questions). Source: `results/runB_val.jsonl`.

Date: 2026-05-06. Branch: `day1-distill-pool`. Same setup as
runs C/D/E/F (max-model-len 26624, max-num-seqs 128).

## Headline — Run B is val-LOWEST but leaderboard-HIGHEST

| Run | Val Acc | Leaderboard | Val Δ vs Phase 0 | Lb Δ vs Phase 0 |
|---|---|---|---|---|
| Phase 0 | 56.44 % | 0.575 | 0 | 0 |
| **Run B** | **56.00 %** | **0.600** | **−0.44 pp** | **+2.5 pp** ✅ |
| Run C | 57.78 % | (not yet measured) | +1.34 pp | ? |
| Run D | 57.33 % | (not yet measured) | +0.89 pp | ? |
| Run E | 56.89 % | not submitted | +0.45 pp | — |
| Run F | 58.67 % | not submitted | +2.23 pp | — |
| Run G | 57.78 % | not submitted | +1.34 pp | — |

**Run B has the lowest stratified val accuracy of all seven prompts but
the highest leaderboard score.** Val ranking does not predict leaderboard
ranking on this dataset.

## What this overturns

Earlier reports (`runC_runD_val_summary.md`, `runE_val_summary.md`,
`runF_val_summary.md`, `runG_val_summary.md`) recommended choosing the
val-best prompt (Run F at 58.67 %) for downstream K=8 SC + LoRA work.
**Run B's val (56.00 %) → leaderboard (0.60) data point invalidates that
recommendation**:

1. The val→leaderboard mapping is not monotonic. Run B (val 56.00 %)
   beats Phase 0 (val 56.44 %) on leaderboard despite losing on val.
2. The 1.78 pp val spread among C/D/E/F/G is essentially random noise
   relative to leaderboard — none of them have been leaderboard-tested
   except indirectly.
3. The "Run F is best" intuition was an artefact of stratified val being
   a poor proxy for private — exactly the failure mode `gap_analysis.md`
   diagnosed for Phase 1 (val +6 pp / leaderboard −8 pp).

## Why Run B might dominate on leaderboard despite mediocre val

Three plausible mechanisms (not mutually exclusive):

1. **Shortest prompt wins on private's symbolic / low-precision golds.**
   Run B is 135 tokens (Phase 0: 52 tokens; Run F: 196; Run G: 240).
   Among the prompts that beat Phase 0 on leaderboard (so far only Run B),
   Run B is the shortest non-trivial change. Less prompt → less reasoning
   drift → output stays closer to the model's natural distribution.
   `gap_analysis.md` already identified prompt-length-induced drift as
   a Phase 1 → leaderboard regression cause.
2. **Run B's MCQ anti-paren rule has 0 cost on private** (no `\boxed{(C)}`
   cases per Phase 0 hand-inspection) but recovers ~10 MCQs on val. This
   is a free win that all subsequent C/D/E/F/G inherited but those runs
   added other rules that hurt private gold distribution.
3. **Free-form rules in C/D/E/F/G that boost val regress private**:
   anti-rounding (Run B does NOT have this), example-based formatting,
   topic-routed advice. These all interact with private's higher
   symbolic-gold proportion in ways val cannot detect.

## Per-type breakdown (Run B vs the others)

| Run | MCQ | F-single | F-multi |
|---|---|---|---|
| Phase 0 | 60.0 % | 62.7 % | 48.2 % |
| **Run B** | **70.7 %** | 56.1 % | 42.9 % |
| Run F | 74.7 % | 56.1 % | 46.4 % |

Run B's MCQ gain (+10.7 pp vs Phase 0) is real and replicates across
C/D/E/F (all in the +12 to +15 pp range). Free-single takes a 6.6 pp hit
across all variants. Free-multi takes 1.8 to 5.3 pp hit.

The MCQ improvement *is* the source of Run B's leaderboard gain, since
the free-form regression on val cancels it out on val but apparently
does not cancel it on private (different gold distribution).

## Strategic correction (overrides earlier recommendations)

For Day 2+ K=8 SC + LoRA work:

- **Use Run B as the production prompt** (leaderboard-validated 0.600).
  Not Run F, not Run G. Earlier reports' recommendations to use Run F
  are superseded.
- **Self-distillation pool**: K=32 SC on `data/public_train.jsonl` with
  `--prompt runb`, not `--prompt runf`.
- **LoRA SFT**: targets generated under Run B's system+user prompts.
  Inference time also uses Run B.
- **Don't iterate prompts further on val**: the C/D/E/F/G ablation
  yielded no signal that transfers to leaderboard. New prompts must be
  validated on leaderboard directly to get a real signal, but submission
  budget is limited.

## Caveats

1. Run B's val 56.00 % vs Run F's 58.67 % is 2.67 pp on n=225 — close to
   the K=1 sampling noise std error (~3 pp). It's plausible that Run F
   would score similar or better on private — but absent direct
   measurement, Run B is the only proven choice.
2. K=8 SC may flatten the per-prompt differences. If K=8 SC + Run F
   beats K=8 SC + Run B on leaderboard, Run F becomes the choice.
   That's a worthwhile experiment for Day 2.

## Files

- `results/runB_val.jsonl` (6.6 MB) — 225 raw rows.

## Reproduce

```bash
uv run --no-sync cse151b-sc \
    --input data/public.jsonl \
    --val data/val_indices.json \
    --output results/runB_val.jsonl \
    --k 1 --temperature 0.6 --top-p 0.95 \
    --prompt runb \
    --gpu-mem-util 0.92 \
    --max-model-len 26624 --max-num-seqs 128 \
    --allocate-tokens
```

Wallclock: 16.7 min on Blackwell 96 GB.
