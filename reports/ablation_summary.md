# Phase 1 Ablation Summary

All variants evaluated on the same stratified 20% public val (n=225, seed=42).
Phase 0 leaderboard score: **0.575**, Phase 1: **0.494** (-0.081, regression).

| variant | description | overall | MCQ | free_single | free_multi |
|---|---|---|---|---|---|
| `phase0` | Phase 0 starter | 127/225 (56.44%) | 45/75 (60.00%) | 42/67 (62.69%) | 40/83 (48.19%) |
| `phase1` | Phase 1 (full) | 141/225 (62.67%) | 55/75 (73.33%) | 44/67 (65.67%) | 42/83 (50.60%) |
| `v3a` | v3a = P0 + anti-(C) only | 137/225 (60.89%) | 55/75 (73.33%) | 43/67 (64.18%) | 39/83 (46.99%) |
| `v3b` | v3b = P1 - anti-rounding | 131/225 (58.22%) | 53/75 (70.67%) | 41/67 (61.19%) | 37/83 (44.58%) |
| `v3c` | v3c = P1 - token-rescue | 132/225 (58.67%) | 49/75 (65.33%) | 41/67 (61.19%) | 42/83 (50.60%) |

## Interpretation guide

- If **v3a** ≈ **phase1** on overall: the MCQ anti-paren rule alone explains most of Phase 1's val gain.
- If **v3b** > **phase1** on overall: anti-rounding is the public→private regression cause; drop it.
- If **v3c** > **phase1** on overall: token-budget rescue is the public→private regression cause; drop it.
- If both **v3b** and **v3c** > **phase1**: combine both fixes for the final prompt.