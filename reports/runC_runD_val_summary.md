# Run C and Run D Re-validated on Stratified val_225

The jason/dev reported numbers (Run C 65.33 %, Run D 63.56 %) were
measured on the FIRST 225 rows of `data/public.jsonl`, which is **not**
stratified by question type. This re-runs both prompts on the same
stratified `val_indices.json` we used for Run E and Run F, so all five
prompts (C/D/E/F + Phase 0 baseline) can be compared like-for-like.

Date: 2026-05-06. Branch: `day1-distill-pool`. Same setup as Run E/F:
single-shot K=1, T=0.6, top_p=0.95, budget allocator
[12288, 20480], `max_model_len=26624`.

## Headline (stratified val_225)

| Run | Overall | MCQ (75) | F-single (66) | F-multi (84) | no-box | multi-box | \quad/\qquad |
|---|---|---|---|---|---|---|---|
| Phase 0 | 56.44 % | 60.0 % | 62.7 % | 48.2 % | 17.4 % | — | — |
| **Run C** | **57.78 %** | 74.7 % | 56.1 % | 44.0 % | 5.3 % | 99/225 | 47/225 |
| **Run D** | **57.33 %** | 72.0 % | 54.5 % | 46.4 % | 4.4 % | 103/225 | 53/225 |
| Run E | 56.89 % | 73.3 % | 54.5 % | 44.0 % | 4.4 % | 90/225 | 42/225 |
| **Run F** | **58.67 %** | 74.7 % | 56.1 % | 46.4 % | 5.3 % | 99/225 | 44/225 |

## "First 225" vs stratified val_225 gap

| Run | jason/dev (first 225) | stratified | Δ |
|---|---|---|---|
| Run C | 65.33 % | 57.78 % | **−7.55 pp** |
| Run D | 63.56 % | 57.33 % | **−6.23 pp** |

The "first 225" metric over-estimated by 6–8 pp. The first 225 rows of
`public.jsonl` skew toward easier / higher-MCQ-ratio questions. Per-type
distributions on the first 225 vs stratified val:

- First 225 had a higher MCQ proportion, where C/D/E/F do well (+12–15 pp
  vs Phase 0 baseline).
- Stratified val_225 has more free_multi (84/225 = 37.3 %), where C/D/E/F
  underperform Phase 0 by 2–4 pp.

**The two effects ALMOST cancel on stratified val** — overall gain over
Phase 0 across all four prompts is only +0.45 to +2.23 pp.

## Key signal we can trust

1. **MCQ is the genuinely improved bucket**: 60.0 % (Phase 0) → 72–75 %
   (Run C/D/E/F), a +12–15 pp lift. This is the largest signal in the
   table and is reproducible across all four prompt variants — it
   reflects Run B's anti-`\boxed{(C)}` / anti-period rule actually
   working. Worth keeping.
2. **Free-single regressed 6–8 pp** vs Phase 0 (62.7 % → 54.5–56.1 %).
   The longer prompts trigger reasoning drift on questions where
   Phase 0's terse formulation was sufficient.
3. **Free-multi roughly matched Phase 0** (48.2 % → 44.0–46.4 %), losing
   only 2–4 pp despite the strict "single \\boxed{} comma-separated"
   mandate that the model ignores 40 %+ of the time.

The MCQ gain (~+5 pp on overall when scaled by MCQ's 75/225 weight)
nearly cancels the free-single regression (~−4.5 pp scaled by 66/225),
which is why the overall accuracy moves only ±1 pp across runs.

## Format-rule compliance is not improving across iterations

| Run | multi-box rate | \quad rate |
|---|---|---|
| Run C | 44.0 % | 20.9 % |
| Run D | 45.8 % | 23.6 % |
| Run E | 40.0 % | 18.7 % |
| Run F | 44.0 % | 19.6 % |

All four are within ~5 pp of each other on multi-box rate. The "ONE
\\boxed{} comma-separated" mandate is not being followed by the model
regardless of prompt length, examples, or wording. → **Model-behavior
ceiling, not prompt-fixable.** Solve at the LoRA SFT level (force
single-box format in target_response).

## Strategic implication

Prompt iteration B → C → D → E → F has hit diminishing returns:

- Public val_225 spread is 1.78 pp (56.89 % – 58.67 %), within K=1
  sampling noise (~±3 pp std error).
- Best prompt by stratified val (Run F at 58.67 %) is statistically
  indistinguishable from Run C (57.78 %) and Run D (57.33 %).
- Run B is the only prompt with leaderboard-validated improvement
  (0.60 vs Phase 0 0.575).

**Recommendation: stop iterating prompts on val_225.** The next units
of compute should go to:

1. **K=8 SC + Run F** (or Run B): stratified val has not yet seen
   the SC variance reduction these prompts deserve. If K=8 SC on top
   of Run F val_225 hits ≥ 62 %, that's a real, statistically clean
   improvement vs the 58.67 % single-shot.
2. **LoRA SFT on Run F K=32 self-distill pool**: format compliance
   (multi-box / quad) can be solved at SFT level, not prompt level.
   Train target_response in the format judger expects, model learns to
   comply regardless of prompt.

For private leaderboard:
- Existing K=8 SC + Phase 0 + budget submission (already in
  `submissions/sc_phase0_k8_private.csv`) is the leading bet today.
- If submitted and beats 0.60, prompt iteration is fully closed.
- If not, the next experiment is **K=8 SC + Run F** on private (one
  fresh run, ~8 h, then submit).

## Files

- `results/runc_val.jsonl` (5.7 MB) — 225 raw rows.
- `results/rund_val.jsonl` (5.8 MB) — 225 raw rows.
- `/tmp/cd_val.log` — full vLLM trace (not committed).

## Reproduce

```bash
scripts/run_cd_val.sh
```

Wallclock: ~28 min on Blackwell 96GB (Run C 14.5 min + Run D 14.3 min).
