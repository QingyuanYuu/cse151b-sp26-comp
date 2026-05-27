# Run E Val Probe — Regression vs Run D, Format Rules Ignored

Single-shot K=1 with budget allocator, run on `data/public.jsonl[val_indices]`
(225 stratified val questions). Source: `results/runE_val.jsonl`.

Date: 2026-05-06. Branch: `day1-distill-pool` (cherry-picked Run E prompt
from `origin/jason/dev` commit `d153658`).

## Headline

| Configuration | Val Acc | Δ vs Run D |
|---|---|---|
| **Run E** (this run) | **128/225 = 56.89 %** | **−8.44 pp** ❌ |
| Run D (per `d153658` commit) | 65.33 % | 0 |
| Phase 0 (`baseline_v0_val`) | 56.44 % | −8.89 pp |
| Phase 1 (`milestone_results`) | 62.67 % | −2.66 pp |

Run E barely beats Phase 0 (+0.45 pp) and undoes Run D's accumulated +9 pp
gain. Per Run E's own design rule ("if val regresses ≥ 2 pp vs Run D, fall
back to Run D"), this prompt should not promote to private leaderboard.

## By question type

| Type | n | Run E acc | No-box rate |
|---|---|---|---|
| MCQ | 75 | 55/75 = 73.3 % | 9.3 % (7 / 75) |
| free_single | 66 | 36/66 = 54.5 % | 1.5 % (1 / 66) |
| free_multi | 84 | 37/84 = 44.0 % | 2.4 % (2 / 84) |

**MCQ holds up; both free-form types collapse**, with free_multi worst hit.
Topic suffix only fires on free questions, which is the same population that
regressed — strong correlation but not yet causation (need ablation).

## Format rule compliance — model largely ignored Run E's mandates

Run E's prompt explicitly forbids `\quad`/`\qquad`, multiple separate
`\boxed{}` blocks, and "section headers near the final answer". Empirically:

| Forbidden pattern | Frequency in Run E output |
|---|---|
| ≥ 2 `\boxed{...}` blocks | **90 / 225 = 40.0 %** |
| `\quad` or `\qquad` in response | **42 / 225 = 18.7 %** |
| Response > 30 000 chars (rambling) | 26 / 225 = 11.5 % |

The "ONE single boxed comma-separated" rule is violated in 40 % of responses
— **higher than v6's rate**. The model reads the rule, then ignores it.
This is the worst of both worlds: the rule consumes prompt tokens (cognitive
cost / reasoning drift) without delivering format compliance.

## Probable failure mechanism

Four axes pushed simultaneously (per `d153658` commit):

1. Topic routing (stats / calculus / linalg / probability suffixes).
2. 5-shot worked examples (vs Run D's 3-shot).
3. MCQ elimination strategy clause.
4. "Be concise" hint.

System prompt length pushes 320+ tokens with topic suffix, approaching
Phase 1's 349-token regression zone. The combined cognitive overhead
appears to:

- Make the model gloss the format rules (since 5 worked examples + topic
  tip + concise hint compete for attention with the format mandates).
- Trigger reasoning drift on free-form (median response 8.4k chars, mean
  12.8k — long but not enough to box correctly).
- Leave MCQ relatively unaffected because the elimination strategy is
  short and concrete, and MCQ has a constrained output space (just a
  letter) so format drift has nowhere to hide.

free_multi is hardest hit because:

- Its "single boxed comma-separated" rule is the one most violated
  (40 % multi-box rate).
- All-or-nothing scoring: K-th-power decay turns even a small per-slot
  error into a large question-level miss.

## Recommendation

**Do NOT promote Run E to public 1126 or to leaderboard.** Save the 30
minutes + 1 submission slot.

For Day 2+ work (K=8 SC + LoRA self-distillation pool):

- **Use Run D** as the production prompt. It is the val-best documented
  prompt at 65.33 %, prompt length stays within safe bounds, and its
  format rules are simpler and presumably better-respected than Run E's.
- If a future ablation isolates which one of Run E's four axes contributed
  most negatively, those individual axes could be cherry-picked (e.g.,
  the 5-shot examples might be net-positive in isolation). Not a priority
  this week.

## Files

- `results/runE_val.jsonl` — 225 raw rows with `winning_response`,
  `all_extracted`, `correct`, `solvable_but_missed`. K=1 so
  `winning_response == all_responses[0]`.
- `/tmp/runE_val.log` — full vLLM + tqdm trace (not committed).

## Reproduce

```bash
# After cherry-picking prompts.py from origin/jason/dev:
uv run --no-sync cse151b-sc \
    --input data/public.jsonl \
    --val data/val_indices.json \
    --output results/runE_val.jsonl \
    --k 1 --temperature 0.6 --top-p 0.95 \
    --prompt rune \
    --gpu-mem-util 0.92 \
    --max-model-len 26624 --max-num-seqs 128 \
    --allocate-tokens --max-tokens-floor 12288 --max-tokens-ceiling 20480
```

Wallclock: 13.2 min on Blackwell 96GB (model warm).
