# Run H Val Probe — Run B + 2 Cautious Additions

Single-shot K=1 with v1 budget allocator on `data/public.jsonl[val_indices]`
(225 stratified val questions). Source: `results/runH_val.jsonl`.

Date: 2026-05-06. Branch: `day1-distill-pool`. Cherry-picked Run H prompt
from `origin/jason/dev` commit `e26e7d3`.

## Headline (full 8-row val table)

| Run | Val Acc | Lb | MCQ | F-single | F-multi | no-box | multi-box |
|---|---|---|---|---|---|---|---|
| Phase 0 | 56.44 % | 0.575 | 60.0 % | 62.7 % | 48.2 % | ~17 % | — |
| Run B | 56.00 % | **0.600** | 70.7 % | 56.1 % | 42.9 % | 12/225 | 100/225 |
| Run C | 57.78 % | ? | 74.7 % | 56.1 % | 44.0 % | 12/225 | 99/225 |
| Run D | 57.33 % | ? | 72.0 % | 54.5 % | 46.4 % | 10/225 | 103/225 |
| Run E | 56.89 % | — | 73.3 % | 54.5 % | 44.0 % | 10/225 | 90/225 |
| Run F | 58.67 % | — | 74.7 % | 56.1 % | 46.4 % | 12/225 | 99/225 |
| Run G | 57.78 % | — | 72.0 % | 56.1 % | 46.4 % | 8/225 | 108/225 |
| **Run H** | **58.22 %** | — | 74.7 % | 56.1 % | 45.2 % | **9/225** | 100/225 |

Run H sits at **58.22 % (131/225)** — second-best val behind Run F's
58.67 %, +2.22 pp above its Run B base. The two additions delivered
exactly what they promised:

- **End-with-box (free-form)**: no-box rate 12 → **9** (saved ~3 questions).
- **MCQ 8+ option elimination**: MCQ 70.7 % → **74.7 %** (saved ~3
  questions, matches Run F's MCQ).

## Why Run H matters as a K=8 SC base candidate

Run H is the **minimum-risk extension of Run B** (the only leaderboard-
validated prompt at 0.600). It adds two surgical changes whose
non-harmfulness was independently established by earlier ablations:

- End-with-box rule appears in C/D/F/G — none of them lost free-form
  performance attributable to it.
- MCQ elimination clause was the single Run E feature that survived
  (MCQ 73.3 % when other axes crashed; Run F kept it at 74.7 %).

Each addition reduces a known no-box failure mode that K=8 SC
amplifies via voting:

- More boxed samples per question → SC voter has more valid candidates
  to count.
- 10-opt MCQ truncation rate was 11 % on Run B; cutting it boosts SC's
  per-MCQ vote integrity.

By contrast, what Run H **does not** include is everything diagnosed as
private-set-harmful:

- ❌ Yes/Tuesday/True inline rule (Run C id=30 cause).
- ❌ Worked examples (id=5/30/135 echo bugs).
- ❌ Topic routing (Run E collapse).
- ❌ Anti-rounding rule (Phase 1 / v2 / private regression).
- ❌ Per-type budget on free_single (Run C 12k starvation).

Length: MCQ 116 t (Run B 87 t, +30 t for elim clause). Free 143 t (Run B
137 t, +6 t for end-with-box rephrase). Both under the 175-token safety
band that gap_analysis identified as the reasoning-drift threshold.

## Important caveat: val ≠ leaderboard on this dataset

`runB_val_summary.md` documented that Run B's val (56.00 %) is the
LOWEST of any prompt yet still produces the HIGHEST leaderboard (0.600).
Run H's +2.22 pp val gain over Run B is in the same K=1 noise band that
all C/D/E/F/G occupy without delivering leaderboard signal. Run H may
or may not beat Run B on private — only a leaderboard submission would
tell.

The argument for Run H is structural, not empirical:

- Specifically targets the no-box rate that K=8 SC voting most benefits
  from reducing.
- Preserves everything Run B did right, adds only changes whose
  non-harmfulness is established.

## Recommendation

For Day 2+ K=8 SC + LoRA work:

1. **Run F (final) + v2 budget** is what jason/dev's commit ee43f97
   recommends and what's running on private now (PID 85169 as of this
   writing). Expected leaderboard 0.615–0.625.
2. **Run H is the conservative alternative** if Run F regresses below
   K=8 SC + Phase 0's already-validated 0.611. The SC pool generated
   under Run H would also be a reasonable LoRA SFT data source (or
   Run B's, given Run B's leaderboard).
3. Don't spend more leaderboard quota on prompt iteration — stratified
   val adds no signal here. Spend the next quota on something that
   moves the variance frontier (LoRA, larger K, etc.).

## Files

- `results/runH_val.jsonl` (6.3 MB) — 225 raw rows.

## Reproduce

```bash
uv run --no-sync cse151b-sc \
    --input data/public.jsonl \
    --val data/val_indices.json \
    --output results/runH_val.jsonl \
    --k 1 --temperature 0.6 --top-p 0.95 \
    --prompt runh \
    --gpu-mem-util 0.92 \
    --max-model-len 26624 --max-num-seqs 128 \
    --allocate-tokens
```

Wallclock: 15.6 min on Blackwell 96GB.
