# Run G Val Probe — No-box -1.7pp, Overall Flat vs Run F

Single-shot K=1 with **v2 budget** (floor 16k, MCQ cap 22k, multi cap 30k),
on `data/public.jsonl[val_indices]` (225 stratified val questions).
Source: `results/runG_val.jsonl`.

Date: 2026-05-06. Branch: `day1-distill-pool` (cherry-picked Run G prompt
+ v2 budget from `origin/jason/dev` commit `8071d19`).

## Headline (vs all prior prompt iterations on stratified val_225)

| Run | Overall | MCQ | F-single | F-multi | no-box | multi-box | mean response len |
|---|---|---|---|---|---|---|---|
| Phase 0 | 56.44 % | 60.0 % | 62.7 % | 48.2 % | — | — | — |
| Run C | 57.78 % | 74.7 % | 56.1 % | 44.0 % | 12/225 | 99/225 | 13 531 |
| Run D | 57.33 % | 72.0 % | 54.5 % | 46.4 % | 10/225 | 103/225 | 13 594 |
| Run E | 56.89 % | 73.3 % | 54.5 % | 44.0 % | 10/225 | 90/225 | 12 849 |
| Run F | 58.67 % | 74.7 % | 56.1 % | 46.4 % | 12/225 | 99/225 | 13 229 |
| **Run G** | **57.78 %** | 72.0 % | 56.1 % | 46.4 % | **8/225** | **108/225** | **14 420** |

Five-prompt total spread: **56.89 % — 58.67 % (1.78 pp)**, all within
single-shot K=1 sampling noise (~±3 pp).

## Run G's two design changes vs Run F

1. **+1 worked example: t-test "reject, 2.45"** to fix Run C id=30 sub-letter
   answer bug (model was substituting Yes/No for the question's own option
   text in multi-part stat questions).
2. **v2 budget**: floor 12 k → 16 k, MCQ cap 18 k → 22 k, multi cap 22 k →
   30 k. Targeted Run F's 12/225 no-box rate.

## What changed empirically

**Wins**
- no-box rate: 12/225 (Run F) → **8/225 (Run G)**, −1.78 pp. The v2
  budget rescued ~4 questions that previously truncated.
- free_multi: 46.4 % (same as Run F) — the new "reject" example did
  not regress free_multi as feared.

**Losses**
- MCQ: 74.7 % → **72.0 %** (−2.7 pp). The new prompt is identical to
  Run F's on MCQ (`RUNG_SYSTEM_PROMPT_MCQ = RUNF_SYSTEM_PROMPT_MCQ`),
  so this is either: (a) sampling noise, or (b) v2 budget gave 10-opt
  MCQ more room to ramble into wrong answers (+2 k → 20 k cap).
- multi-box rate: 99/225 (44 %) → **108/225 (48 %)** — *highest of any
  run*. More budget = more room to emit multiple boxed blocks.
- mean response length: 13 229 → 14 420 (+9 %). Longer thinking,
  no accuracy translation.
- Wallclock: 13.9 min (Run F) → **22.2 min (Run G), +60 %**. v2's
  multi-part 30 k cap makes the longest-tail question dominate batch
  time.

**Net**: 4 questions rescued via no-box → 4 wrong rescued, but ~6
questions lost via MCQ regression (75 × 2.7 % ≈ 2 — actually the MCQ
loss explains roughly 2 questions, sample noise the rest). Approximately
washes.

## Why the "reject" example didn't help free_multi

The example targets exactly the failure mode it was designed to fix
(letter-style sub-answers in stat questions). On val_225, however:
- Stat-test questions with letter-style sub-answers are sparse (≤ 5).
- The 4 questions rescued by v2 budget are almost certainly different
  from the questions that the "reject" example would help.
- 1.78 pp val spread = noise floor, so a 2-3 question targeted fix is
  invisible above noise.

The example may still help on the private set if it has more such
questions, but val cannot detect it.

## Confirms the strategic recommendation

The C/D/E/F/G ablation arc on stratified val converges on the same
conclusion documented in `reports/runC_runD_val_summary.md`:

> **Prompt iteration B → C → D → E → F → G has hit diminishing returns;
> stratified val spread is in K=1 sampling noise.**

Don't run Run H. Move to:

1. **K=8 SC + Run F** (or Run G — they tie on val) on private →
   expected +3-5 pp variance reduction.
2. **LoRA SFT** on the K=32 Run F/G self-distill pool, with
   `target_response` forced to single-box-comma-separated format
   (LoRA can fix the 44-48 % multi-box rate that prompt cannot).

## Files

- `results/runG_val.jsonl` (6.5 MB) — 225 raw rows.
- `/tmp/runG_val.log` — full vLLM trace (not committed).

## Reproduce

```bash
uv run --no-sync cse151b-sc \
    --input data/public.jsonl \
    --val data/val_indices.json \
    --output results/runG_val.jsonl \
    --k 1 --temperature 0.6 --top-p 0.95 \
    --prompt rung \
    --gpu-mem-util 0.92 \
    --max-model-len 32768 --max-num-seqs 32 \
    --allocate-tokens
```

Wallclock: 22.2 min on Blackwell 96GB (max-num-seqs=32, KV concurrency
~18× per the engine startup log).
