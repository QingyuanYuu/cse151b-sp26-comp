# Run F K=1 + v2 budget on full public — Day1 jsonl `correct` field is buggy

The `results/runf_k1_public.jsonl` published in
`day1-distill-pool` commit `519c92b` reports **53.73 %** total accuracy
on full public (1126 questions). Re-evaluating the same JSONL with our
local `cse151b_comp.evaluate.evaluate_rows` (which dispatches the
course-provided `judger.Judger.judge`) gives **63.14 %**, an upward
correction of **+9.41 pp**.

## Real numbers (re-judged with local judger)

| Metric | Day1 reported | **Local re-judge** | Δ |
|---|---|---|---|
| Total | 605 / 1126 = 53.73 % | **711 / 1126 = 63.14 %** | **+9.41 pp** |
| MCQ | 284 / 375 = 75.7 % | 284 / 375 = 75.7 % | 0 (no diff) |
| free_single | 154 / 336 = 45.8 % | **204 / 336 = 60.7 %** | **+14.9 pp** |
| free_multi | 167 / 415 = 40.2 % | **223 / 415 = 53.7 %** | **+13.5 pp** |
| Mismatched rows | — | 110 (all Day1=False → Local=True) | — |

MCQ accuracy is unaffected because the judger handles single-letter
answers identically. The bug is concentrated in free-form types where
symbolic vs decimal equivalence matters.

## Cross-validation: F is consistently strong

| Eval set | Phase 0 / Run B baseline | Run F K=1 + v2 budget | Δ |
|---|---|---|---|
| Private (Kaggle leaderboard, official judger) | Run B 0.600 | **0.632** | **+3.2 pp** |
| Full public (re-judged locally) | Phase 0 60.12 % | **63.14 %** | **+3.0 pp** |
| Stratified val_225 (re-judged) | Phase 0 56.44 % | 58.67 % | +2.2 pp |

Public-rejudge and private-leaderboard agree at +3 pp. The Day1
"public 53.73 %" number was an artefact of the buggy `correct` field,
not a real regression.

## Sample mismatches (Day1=False, Local=True)

Three concrete cases from `data/public.jsonl`:

### id=0: 325 \* (1+325) = 105 950

| | Day1 `correct` | extracted | re-judge |
|---|---|---|---|
| Phase 0 | True | `105950` | True |
| Run F | **False** | `105950` | True |

Identical extracted answer, identical numeric value, opposite Day1
labels. Pure evaluation bug.

### id=7: 13/9 = 1.444…

| | Day1 `correct` | extracted | re-judge |
|---|---|---|---|
| Phase 0 | True | `1.444444` | True |
| Run F | **False** | `\frac{13}{9}` | True |

`\frac{13}{9}` and `1.444444` differ by < 1 e-6 and are equivalent under
the course judger's `judge_single_numerical_value` / `judge_expression`
dispatch with sympy parsing. Day1 marked them differently.

### id=32: 21275/3 = 7091.666…

| | Day1 `correct` | extracted | re-judge |
|---|---|---|---|
| Phase 0 | True | `7091.666666` | True |
| Run F | **False** | `\frac{21275}{3}` | True |

Same pattern as id=7. Pure fractional vs decimal display, mathematically
equivalent.

## Diagnosed root cause (hypothesis)

`day1-distill-pool` may dispatch `correct` through a path that does not
go through the full `judger.Judger.judge`/`auto_judge` machinery, but
instead does a string-or-numeric comparison that fails when the model
emits `\frac{a}{b}` against a decimal gold. The fact that 110 / 1126
mismatches are all in free-form types and all in the same direction
(Day1 says False, our re-judge says True) is consistent with a
narrower comparator that doesn't normalise LaTeX fractions to decimals.

## Implications for ongoing work

1. **Run F (final) is genuinely strong**: +3 pp lift on both private
   leaderboard (verified) and full public (verified after re-judge).
   The K=8 SC + Run F + v2 budget run on private should be expected to
   land 0.640 – 0.655 (Run F K=1 = 0.632, +0.5 to +1.5 pp from SC
   variance reduction). The earlier interpretation of "Run F regresses
   on full public" was an artifact, not a real signal.

2. **Don't trust Day1's `correct` field for prompts that emit
   symbolic / fractional answers**: Run B / Phase 0 emit cleaner
   decimals so they're less affected; Run F / Run G / Run I (which
   prefer `\frac` and `\sqrt`) will be systematically under-counted by
   Day1's evaluator. Re-judge locally before drawing conclusions.

3. **Run I evaluation must re-judge locally**: Run I's stats branch
   asks for exact fractions for p-values / R^2; Day1's eval would
   penalise these as decimal-mismatch even when the math is right.
   When the Run I results land, run them through local
   `evaluate_rows` first, not the per-row `correct` field.

4. **Consider syncing one judger across both repos**: This bug
   suggests Day1 is using a different evaluator than the one we ship
   in `cse151b_comp.evaluate`. Aligning to a single source of truth
   would prevent future mis-readings of cross-machine results.

## Reproduce

```bash
# Pull Day1's F public jsonl
git fetch origin day1-distill-pool
git show origin/day1-distill-pool:results/runf_k1_public.jsonl > /tmp/runf_public.jsonl

# Re-judge locally
PYTHONPATH=src .venv/bin/python <<'PY'
import json, copy
from cse151b_comp.evaluate import evaluate_rows

F = [json.loads(l) for l in open('/tmp/runf_public.jsonl')]
public = {r['id']: r for r in [json.loads(l) for l in open('data/public.jsonl')]}

# Augment with question / options / answer (eval_rows expects them)
for r in F:
    if 'response' not in r:
        r['response'] = r.get('winning_response', '') or (r.get('all_responses') or [''])[0]
    item = public[r['id']]
    r['question'] = item['question']
    r['options'] = item.get('options')
    r['answer'] = item.get('answer')
    r['type_sequence'] = item.get('type_sequence')

re_eval = evaluate_rows(copy.deepcopy(F))
n_local = sum(1 for r in re_eval if r.get('correct'))
n_day1 = sum(1 for r in F if r.get('correct'))
print(f"Day1: {n_day1}/{len(F)} = {n_day1/len(F)*100:.2f}%")
print(f"Local re-judge: {n_local}/{len(F)} = {n_local/len(F)*100:.2f}%")
PY
```

Wallclock: ~30 s for 1126 rows on the 4090 box (no GPU needed for
re-judge; pure CPU sympy work).
