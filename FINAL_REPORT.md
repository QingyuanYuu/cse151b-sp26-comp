# CSE 151B Competition — Final Submission Report

**Team members**: Qingyuan Yu, Chengzhi Zhang, Po-Chen Lin
**Base model**: `Qwen/Qwen3-4B-Thinking-2507`
**Final model (HuggingFace Hub)**: `JaasonYuu/jason-cse151b-model`
  — fully merged BF16 weights (Qwen3-4B-Thinking-2507 + SFT LoRA + GRPO LoRA, all merged)
**Standalone LoRA adapters (optional, for reproducing intermediate states)**:
  `JaasonYuu/jason-cse151b-sft-lora` (r=64) and `JaasonYuu/jason-cse151b-grpo-lora` (r=32)
**Public code repository**: see Gradescope submission link
**Kaggle private leaderboard**: _to be added after final upload_

---

## TL;DR

We built a five-phase pipeline:
1. **Prompt engineering** (Runs B → K) — iterated on system+user prompts;
   Run F selected for production after +6.23 pp Phase 1 lift and Run D/E/F
   stratified-val ablation.
2. **Supervised fine-tuning (SFT)** on Qwen3-4B-Thinking-2507 with K=32
   self-distillation targets, using the Run F prompt format throughout.
   **val_225: 64.44%.**
3. **GRPO v1** — first RL attempt. Silently failed: gradient norm collapsed to
   ~1e-18, val_225 regressed by −2.22 pp. Training aborted; **no usable
   checkpoint**.
4. **GRPO v2** — second RL attempt, applying the three v1 bugfixes
   (importance sampling, reward saturation, length collapse) on top of the
   same LoRA-v1 merged 4B base, using the **196-prompt edge-filtered
   training pool**. Training-loop bugs eliminated; **val_225: 64.00%** (flat
   vs the LoRA-v1 baseline of 64.44%; the val improvement only materialized
   in v3).
5. **GRPO v3** — final RL training on the Phase 2 SFT-merged 4B model with all
   v1/v2 fixes inherited, plus hard-prompt boost, KL anchoring, and corrected
   train/inference prompt alignment. **Best val_225: 66.22%**, +1.78 pp over
   Phase 2 SFT.

Total compute on the final pipeline (Phase 2 + Phase 5): ~24.5 h H100 PCIe at
~$50. Phase 1 prompt engineering and Phase 3–4 ran on separate machines and
budgets earlier.

---

## Phase 1 — Prompt Engineering (Runs B → F)

Before any fine-tuning, we iterated on the system + user prompt itself. Each
Run was evaluated on a stratified 225-prompt validation split (`val_225`),
using K=1 greedy decoding to isolate prompt effects from sampling noise.

### Iteration log

| Run | Change | val_225 (K=1) | Outcome |
|---|---|---|---|
| Phase 0 / Baseline | Naive system prompt, no formatting rules | ~47% (no clean boxed → low extraction) | Multi-`\boxed{}`, garbage chars, low MCQ extraction |
| Phase 1 (Run A) | Initial structured prompt, +6.23 pp on val | ~53–54% | First major lift; reported in Milestone 1 |
| **Run B** | Audit fixes — budget floor + tighter symbolic rule | stable base ~54% | Used as the baseline for B-vs-D probes |
| **Run C** | End-with-box rule + text/bool examples + 10-opt MCQ + 2 k budget | ~55% | Established final-box-only convention |
| **Run D** | Few-shot worked examples (1 MCQ + 3 free-form) | ~58% | Best on val at this point |
| Run E | Ceiling probe: topic routing + 5-shot + MCQ elimination | regression vs D | Format rules ignored under heavy load |
| **Run F** | Surgical fixes for 4 Run D bugs + Run E's MCQ-elimination idea | **58.67%** | **Selected for production**; public K=1 = 53.73% |
| Run G | Run F prompt + v2 budget (16 k floor / 20 k MCQ / 24 k multi cap) | flat vs F | Adopted v2 budget logic for production |
| Run H | Run B + 2 cautious additions only — K=8 SC base candidate | worse than F | Rejected |
| Run I | 5-way topic-routed free-form prompts on top of Run F | mixed | Used only as a fallback ensemble component |
| Run J (v1 → v3) | 7 → 8 → 9-branch topic routing (olympiad split last) | mixed | Used only on a topic-stratified subset |
| Run K | Fix-format prompt for "hopeless" subset | format-only | Used as a post-processing tool, not a primary prompt |

Quantitative anchors:
- Run F val_225 K=1: **58.67%**
- Run F public-split K=1: **53.73%**
- Run F K=8 SC private: produced the production CSV that became the SFT
  distillation seed

### Final prompt template (Run F)

Run F uses two distinct system prompts depending on question type:

- **MCQ** (`RUNF_SYSTEM_PROMPT_MCQ`): forces letter-only output in a single
  `\boxed{X}`, with explicit elimination heuristics for 8+ option questions
  and one inline worked example.
- **Free-response** (`RUNF_SYSTEM_PROMPT_FREE`): mandates a single
  `\boxed{}` (comma-separated for multi-part), bans `\quad`/`\qquad` near the
  final answer, requires symbolic preservation for `\sqrt`/`\pi`/`e^x`/`\ln`
  unless decimals are requested, and includes three worked examples covering
  area-of-circle, slope-intercept, and `\sqrt{75}` simplification.

The selection is `build_prompt_runf(question, options)` in
`src/cse151b_comp/prompts.py`. This builder is **used identically in SFT
data construction, GRPO data construction, and final inference** — keeping the
prompt distribution constant across phases turned out to be one of the most
important reproducibility decisions (see Phase 5 bugfix #4).

### Validation-leaderboard decoupling

A separate finding during this phase (commit `d4383a5`): val_225 score
movements did **not** always track the Kaggle leaderboard. We anchored
production decisions on K=8 self-consistency probes of held-out prompts
rather than single-decode val numbers, after Run B's val-vs-LB decoupling
was discovered.

---

## Phase 2 — Supervised Fine-Tuning (LoRA SFT on Qwen3-4B-Thinking)

### Training data

`runpod_h100/data/h100_lora_sft.jsonl` — **737 (prompt, response) pairs**, all
formatted via Run F's chat template.

Data sources and filtering:
- **612 K=32 self-distillation samples** from the public training pool, filtered
  to `n_correct ≥ 2` (the K=32 SC pool gave each prompt 32 sampled rollouts;
  we kept the consensus answer's reasoning chain only if at least 2 of 32 hit
  the gold answer). Split by quality tier:
  - `public_n4`: 530 (highest quality, 4 of 4 SC samples correct in K=4 probe)
  - `public_n3`: 48
  - `public_n2`: 34
- **125 additional auxiliary problems** drawn from past public math competition
  archives (AMC / AIME / USAMO problem sets matched to the competition's
  difficulty distribution). Reasoning traces for these were generated by an
  independent stronger teacher model (Qwen3-30B-A3B-Thinking with K=8
  self-consistency) and hand-verified for the highest-confidence subset
  (25 of 125).

Token length distribution after Run F templating:
- Median: 3,646 tokens
- P90: 11,763 tokens
- Max: 16,355 tokens
- Truncated after smart head-tail truncation (preserving `\boxed{}` answer):
  **15 of 737 (2.0%)**, down from 21.6% at max_seq=8,192.

### Architecture and training config

| Component | Setting |
|---|---|
| Base | Qwen3-4B-Thinking-2507 (BF16, no quantization) |
| Adapter | LoRA r=64, α=128, dropout 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Attention | flash_attention_2 (with SDPA auto-fallback) |
| max_seq | **16,384** (data + training caps aligned) |
| Loss | Completion-only (mask system/user tokens) |
| Epochs | 5 |
| LR | 2e-4 cosine, 5% warmup |
| Per-device batch | 1 |
| Gradient accumulation | 8 (effective batch = 8) |
| Gradient checkpointing | True |
| Optimizer | AdamW |

### Phase 2 results

| Metric | Value |
|---|---|
| Training loss start → end | 0.46 → 0.121 |
| Token-level accuracy at end of training | 96% |
| Wall clock on H100 PCIe 80GB | ~3 h |
| Merged checkpoint size | ~7.6 GB |
| **val_225 (Run F prompt + v2 budget, K=1 greedy)** | **64.44%** |
| ↳ MCQ | 74.67% |
| ↳ Free-response single | 62.69% |
| ↳ Free-response multi-part | 56.63% |

The merged SFT model (`checkpoints/lora_sft_merged/`) became the
base for Phase 5 GRPO v3.

### Why upgrade SFT from 8,192 → 16,384 max_seq

The original SFT data prep used `MAX_SEQ=8192`, truncating 159 of 737 examples
(21.6%) at the cap and inserting a `[... reasoning truncated ...]` marker mid-
chain. Even though the head-tail strategy preserved the final `\boxed{}`,
the model was effectively training on stitched-up reasoning chains. We
rebuilt the SFT data with `MAX_SEQ=16384`, dropping truncation to 2.0% and
recovering the full reasoning for 144 of 159 previously truncated examples.

---

## Phase 3 — GRPO v1 (First RL Attempt, Cancelled)

GRPO v1 was the first attempt to use RL on top of the LoRA-SFT model. It ran
in two stages — a small smoke test that looked plausible, then a full-scale
run that catastrophically degenerated — and ultimately did not produce a
usable checkpoint. The diagnostic data below directly drove Phase 4's design.

### Stage A: smoke test (small, looked OK)

The initial launch (`scripts/run_grpo.sh`, commit `556f2ba` on 2026-05-09)
was conservative: 600 prompts, K=4 rollouts, 1 epoch, max_completion 4,096.
We checked that the loop ran end-to-end, that reward extraction worked, and
that loss was non-zero. Wall clock estimate was 6–10 h.

| Stage A config | Value |
|---|---|
| Base | LoRA-SFT merged (Qwen3-4B-Thinking + LoRA v1 adapter) |
| Training prompts | 600 (subset of public split, excluding val_ids) |
| Rollouts per prompt (K) | 4 |
| Epochs | 1 |
| LoRA r / α (on top of SFT) | 16 / 32 |
| LR | 5e-6 |
| KL β | 0.04 |
| max_completion | 4,096 tokens |
| Batch / grad_accum | 1 / 4 |
| Loss type | DAPO (TRL default at the time) |
| Importance sampling | Token-level (default) |
| Reward | Binary `Judger.auto_judge` → 1.0 / 0.0 |
| Reward timeout | None (would later cause sympy hang) |

The smoke test surfaced **one operational issue**: a sympy hang inside the
Judger could lock a single reward call for arbitrarily long, stalling the
whole training step. Fix added the same day (commit `45d16f7`,
"GRPO: SIGALRM 15s timeout per reward call (sympy hang fix)"). A second
follow-up (commit `0fd84e3`) raised `grad_accum 4 → 8` so it would be
divisible by `num_generations` after the K change planned for the full run.

### Stage B: full-scale launch (catastrophic failure)

After the smoke test we scaled up to the full training pool and a larger K,
expecting moderate gains:

| Stage B config (delta vs Stage A) | Stage A | Stage B |
|---|---|---|
| Training prompts | 600 | **901** (full public train, val excluded) |
| Rollouts per prompt (K) | 4 | **8** |
| max_completion | 4,096 | ~5,000 (typical generations after K=8 lift) |
| grad_accum | 4 | 8 (divisibility fix) |
| All other hyperparameters | unchanged from Stage A | unchanged |

Within the first few dozen training steps the loop silently degenerated:

| Signal | Observed value |
|---|---|
| `frac_reward_zero_std` (group-wise) | ≈ **1.0** (essentially every K=8 group had zero reward variance) |
| Gradient norm | **~1e-18** (effectively zero — no learning) |
| Mean completion length | **4,070 → 2,574** tokens (**−37%**) over ~hundreds of generations |
| Terminated trajectories (proper EOS / `\boxed{}` close) | **2,680 → 1,345** (**−50%**) |
| MCQ val_225 accuracy | **~75% → ~65%** (**−10 pp**) |
| Overall val_225 vs SFT baseline | **−2.22 pp** |

The model wasn't just stalling — it was **actively unlearning** the SFT
formatting it had been taught, drifting toward shorter and more truncated
outputs.

### Three-bug post-mortem

Independently, three distinct mechanisms were destroying the training signal.
All three needed to be fixed in concert before GRPO would produce any lift.

**Bug 1: Reward saturation (all groups had zero variance)**

The 901-prompt pool was dominated by two regimes the policy could not
benefit from at this stage:
- "Easy" prompts where every one of the K=8 rollouts produced the gold
  answer → reward [1,1,1,1,1,1,1,1] → group-wise std = 0
- "Hard" prompts where none of the K=8 rollouts produced the gold answer →
  reward [0,0,0,0,0,0,0,0] → group-wise std = 0

Because GRPO's policy gradient is proportional to per-rollout advantage
(reward minus group mean) divided by group std, **a group with zero std
contributes zero gradient**. Diagnostic metric: `frac_reward_zero_std → 1.0`
meant nearly every batch was useless. Even when 1–2 prompts in a batch
were in the "1–7 of 8 correct" learnable middle, the per-step signal was
diluted to noise.

The fix in v2 was a difficulty filter: bucket the 901-prompt pool into
all-right (518) / edge (96) / all-wrong (287), keep all 96 edge prompts
plus a random 100-prompt subsample of all-wrong, drop the all-right cases
entirely. Resulting pool: 196 prompts where pass@K=8 would land in the
learnable middle band.

**Bug 2: vLLM importance-sampling overflow (gradients collapsed numerically)**

TRL's default behavior under vLLM colocate applies an importance-sampling
ratio at the per-token level (`sequence_mask` mode) to correct for the
fact that the rollout policy and the policy being optimized are slightly
different. Each token contributes a small log-prob drift to this ratio.
For our setup the per-token drift was roughly **0.015 nats**, which sounds
benign, but it **multiplies across the full completion**:

```
~5,000 tokens × 0.015 nats/token = 75 nats cumulative drift
exp(−75) ≈ 2.7e−33
```

So even when a batch had non-zero advantage, the IS-weighted policy
gradient was being multiplied by ~e^(−75) per sequence and crushed to zero
in floating-point arithmetic. This is what produced the observed grad_norm
of ~1e-18: the gradient was mathematically present but numerically
nonexistent after IS weighting.

The fix in v2 was twofold: disable the per-token IS correction
(`--disable-vllm-is-correction`, relying on PPO clipping to handle drift),
and later switch to `importance_sampling_level="sequence"`, which applies
a single ratio per sequence rather than compounding token-by-token.

**Bug 3: DAPO loss + binary reward → length collapse + MCQ regression**

DAPO loss is not length-normalized. Combined with a binary 0/1 reward that
required emitting `\boxed{}` somewhere in the response, the optimizer found
a degenerate shortcut: **emit short, low-entropy text that ends with a
boxed answer guess**. Shorter completions reduce the per-token KL penalty
and increase the probability of stumbling onto a correct guess within the
reduced reasoning budget, so DAPO actively rewarded length collapse.

The observable consequences on training data after just hundreds of
generations:
- Mean completion length: 4,070 → 2,574 tokens (−37%)
- Terminated trajectories (proper EOS): 2,680 → 1,345 (−50%)
- MCQ accuracy on val: 75% → 65% (−10 pp; the model started skipping
  reasoning and just guessing letters)

The fix in v2 was twofold: switch from DAPO to `dr_grpo`
(length-normalized policy gradient), and add a length-aware reward shaper:

| Outcome | Reward |
|---|---|
| Correct + completion ≥ 2,000 chars | 1.0 |
| Correct + completion < 2,000 chars | 0.5 |
| Wrong / timeout / parse failure | 0.0 |

The shaper makes "correct but skipped reasoning" only half as rewarding as
"correct with full chain of thought", removing the incentive to collapse.

### Outcome and decision

Training was aborted before completing the planned 1 epoch over 901 prompts.
The smoke-test Stage A checkpoint was also discarded because its hyperparams
overlapped with Stage B (same loss, same IS path) and it had only run a
fraction of an epoch on a smaller pool. **No GRPO v1 checkpoint was used in
any downstream stage.**

Three concrete artifacts came out of v1 instead:
1. The three-bug diagnosis above, which became the v2 design specification.
2. The SIGALRM 15 s sympy timeout patch in the reward function, which was
   kept through v2 and v3.
3. The `grad_accum 4 → 8` change for K-divisibility, which was also kept.

Total v1 compute burned: roughly 4–6 hours of H100 time across the smoke
test and the failed full-scale launch. The investment was effectively zero
"model progress" but proportionally large "loop-correctness progress" — every
subsequent GRPO attempt converged because v1 had already exhausted the
failure modes.

---

## Phase 4 — GRPO v2 (Second Attempt, Bugfixes Applied)

This phase succeeded in eliminating all three v1 failure modes, but the
fixes did not translate to a val_225 improvement at this stage. It was run
on the **same Qwen3-4B-Thinking-2507 + LoRA-v1 merged base** that GRPO v1
had used, on the **196-prompt edge-filtered training pool** (96 edge + 100
random all-wrong, see Fix 1 below), with the three v1 bugfixes applied.

### Fix 1: Reward saturation → edge filter

We used K=4 SC scoring on the public training set to bucket prompts by
difficulty, then selected only the learnable middle band:

| Bucket | Definition | Count | Used in pool? |
|---|---|---|---|
| `all_right_ids` | 4 of 4 K=4 samples correct | 518 | Dropped (no signal) |
| `edge_ids` | 1–3 of 4 correct | 96 | All kept (sweet spot) |
| `all_wrong_ids` | 0 of 4 correct | 287 | **100 random subsampled** |

Final pool: **196 prompts**, persisted as `data/grpo_pool_v2.json`. Sampling
only 100 of the 287 all-wrong cases prevented the sparse-reward subset from
dominating the batch.

### Fix 2: IS overflow → disable + later sequence-level

`--disable-vllm-is-correction`; PPO clipping handles drift directly. Later
refined to `importance_sampling_level="sequence"` (sequence-level correction
is much more stable than token-level under vLLM colocate).

### Fix 3: Length collapse → dr_grpo + length-aware reward

Switched from DAPO loss to `dr_grpo` (length-normalized), and added a
length-aware reward shaper:

| Outcome | Reward |
|---|---|
| Correct + completion ≥ 2,000 chars | 1.0 |
| Correct + completion < 2,000 chars | 0.5 |
| Wrong / timeout / parse fail | 0.0 |

The shaper preserves long reasoning chains during RL (anti-collapse).

### Other v1 → v2 hyperparameter changes

| Parameter | v1 | v2 |
|---|---|---|
| LR | 3e-6 cosine | **1e-5 constant** |
| Temperature | 0.7 | **1.0** |
| KL β | 0.04 | **0.0** |
| Epsilon | 0.2 symmetric | **0.3 / 0.4 asymmetric** (DAPO clip) |
| `scale_rewards` | group | **none** |
| Importance sampling | token (default) | **sequence** |
| Loss type | DAPO | **dr_grpo** |
| Judger timeout (sympy hang fix) | none | **15 s SIGALRM** |

### Phase 4 outcome

| Metric | Value |
|---|---|
| Training prompts | 196 (edge-filtered: 96 edge + 100 random all-wrong) |
| K rollouts during training | 4 (matched v1's smoke-test K, since v1's catastrophic K=8 run had not isolated K from the other failure modes) |
| Epochs | 3 |
| Wall clock | ~14–16 h (configured target on the day) |
| **val_225 (K=1)** | **64.00%** — flat vs LoRA-v1 baseline 64.44% |
| Full public K=1 (1126 prompts) | 66.25% (+1.95 pp over LoRA-v1 baseline 64.30%) |
| MCQ pass-rate vs LoRA-v1 baseline | recovered to ~74% (vs 65% during v1's length-collapse) |

The training-loop signals (grad-norm, reward variance, completion length,
MCQ pass-rate) all stabilized in healthy ranges, confirming the three
bugfixes were correct. But the val_225 score was essentially flat against
the LoRA-v1 baseline — the +1.95 pp on the full public set came mostly
from training-set memorization rather than generalization (the
held-out val_225 slice barely moved). Closing this generalization gap was
the explicit motivation for Phase 5: switch to a stronger SFT-merged base,
add hard-prompt boost, add KL anchoring, and align the train/inference
prompts.

---

## Phase 5 — GRPO v3 (Final Pipeline, Current Submission)

This is the production run. Trained on a single H100 PCIe 80GB on RunPod over
~16 hours, on top of the Phase 2 SFT-merged base.

### Training data

`runpod_h100/data/grpo_train_extended_v6.jsonl` — **305 prompts**.

Composition:
- **196 public edge-filtered prompts** — same `grpo_pool_v2.json` selection as
  Phase 4 (96 edge + 100 hard). Real gold answers from the public split.
- **109 additional auxiliary prompts** sourced from past public math
  competition archives (AMC, AIME, USAMO problem sets, selected to match the
  competition's question-type and difficulty distribution). Pseudo-labels were
  generated through a multi-model agreement protocol: for each candidate
  problem we ran our SFT baseline at K=8 self-consistency alongside an
  independent stronger teacher (Qwen3-30B-A3B-Thinking), and retained only
  problems where both pipelines converged on the same Judger-normalized
  boxed answer.

A dedicated subset of **100 "hard" public IDs**
(`runpod_h100/data/hard_prompt_ids.json`) was identified as those producing
**0 / 4 pass rate** at the K=4 baseline. These were treated specially via
hard-prompt boost (below).

### Reward function

Binary correctness via the course Judger (`judger.py`) with the v2
anti-collapse length shaper, plus an MCQ exemption:

```python
ok = judger.judge(rollout, gold, type_sequence, options, timeout=15)
base = 1.0 if ok else 0.0
# Length penalty: free-response only
if base > 0.5 and not is_mcq and len(text) < 2000:
    base = 0.5
return base
```

MCQ rewards 1.0 even for short responses since correct MCQ answers
(`\boxed{C}`) are intrinsically brief.

### Hard-prompt boost (key v3 innovation)

`num_generations=4` globally. But for the 100 hard public IDs, each entry was
**duplicated once in the training data**, so they were visited 2× per epoch.
Effective K_hard = 8 with the same per-step compute cost as K=4 elsewhere.

This concentrates GRPO's learning budget on the sparse-reward frontier:
- "Easy" prompts (pass@4 ≈ 1): low advantage signal, K=4 is enough
- "Hard" prompts (pass@4 ≈ 0): need more rollouts for any positive signal —
  K_eff=8 doubles the chance of producing at least one correct rollout in
  a group

After duplication: 305 + 100 = **405 effective training rows × 3 epochs / batch
8 = 606 total training steps**.

### Configuration table

| Hyperparameter | Value | Why |
|---|---|---|
| Adapter | LoRA r=32, α=64 | Smaller rank than SFT for RL stability |
| Target modules | Same 7 as SFT | |
| K (rollouts/prompt) | 4 (uniform) | + hard-dup = K_eff=8 on 100 IDs |
| Hard-pool dup | 1 (one extra copy) | Sparse-reward concentration |
| KL β | **0.04** | Anchor to SFT — preserve long Thinking chains |
| Loss type | `dr_grpo` | Length-normalized (Phase 4 fix) |
| IS level | `sequence` | Stable under vLLM colocate (Phase 4 fix) |
| `scale_rewards` | `none` | dr_grpo's recipe |
| Epsilon | 0.3 / 0.4 (asymmetric) | DAPO — allows more positive updates |
| Temperature | 1.0 | High exploration |
| LR | 1e-5, constant + 5% warmup | |
| Per-device batch | 1 | |
| Grad accumulation | 8 | Effective batch = 8 prompts |
| Epochs | 3 | 606 steps total |
| max_prompt_length | 2,048 | Run F + question fits with margin |
| max_completion_length | 10,240 | Covers P85 of SFT response distribution |
| vllm_max_model_len | 12,288 | = max_prompt + max_completion |
| vllm_gpu_memory_utilization | **0.35** | Leaves room for KL ref-forward at β > 0 (OOM tuning) |
| PYTORCH_CUDA_ALLOC_CONF | `expandable_segments:True` | Prevent fragmentation OOM on variable seq lens |
| Save | every 20 steps, keep last 10 | Post-hoc best-by-val selection |
| Reference monitoring | TensorBoard | `logging_dir = checkpoints/grpo_v6/logs` |

### Bugfixes specific to Phase 5 (beyond v1+v2 inheritance)

| # | Issue | Fix |
|---|---|---|
| 1 | flash-attn compile fails on RunPod images | Auto-fallback to SDPA in `train_lora_bf16.py` |
| 2 | SFT data prep MAX_SEQ=8,192 misaligned with training cap of 8,192 → 22% of long examples used stitched-up reasoning | Both raised to 16,384, data rebuilt (truncation 21.6% → 2.0%) |
| 3 | GRPO `max_completion_length=6,144` truncated hard-prompt rollouts before `\boxed{}` emit → pass@K locked at 0 | Raised to 10,240 (covers P85 of SFT response lengths) |
| 4 | **GRPO data used a simplified generic system prompt while SFT used Run F** → at GRPO rollout time, model saw an unfamiliar prompt and regressed toward base behavior, losing all format rules | Migrated all 305 GRPO rows to Run F prompts (MCQ + free variants) via `scripts/fix_grpo_v6_prompts.py` |
| 5 | TRL default `vllm_gpu_memory_utilization=0.9` left only 8 GB for training, OOM at step 0 when β > 0 (KL ref-forward adds ~2-3 GB activations) | Lowered to 0.35 after three OOM retries (0.6 → 0.5 → 0.45 → 0.35) |
| 6 | CUDA allocator fragmented on variable seq lengths after 100+ training steps | `expandable_segments:True` |

Bugfix #4 was the highest-leverage discovery. Without prompt alignment we would
likely have seen GRPO collapse the SFT model's format compliance — exactly the
failure mode that hit our earlier GRPO experiments.

### Phase 5 training dynamics

- Total steps: 606 (3 epochs × 405 rows / batch 8 + rounding)
- Wall clock per step: ~96 s (rollout-dominated under K=4 + hard-dup)
- Total wall clock: **~16 h** on H100 PCIe 80GB
- Reward curve: noisy but trending positive after step ~200
- KL curve: stable ~0.001 (β=0.04 effective anchor)
- Clip ratio: 0.0 → 0.6 (asymmetric epsilon allowing positive updates)
- No OOMs after the configuration finalized at vllm_util=0.35 + max_comp=10,240

### Best-checkpoint selection

After Phase 5 completed, we evaluated 10 saved checkpoints
(`checkpoints/grpo_v6/checkpoint-{460,480,500,520,540,560,580,600,606,final}`)
on val_225 and picked the best by overall accuracy:

| Checkpoint | Overall | MCQ | Free single | Free multi |
|---|---|---|---|---|
| **step-606 (selected)** | **66.22%** | 77.33% | 65.67% | 56.63% |
| final | 66.22% | 78.67% | 64.18% | 56.63% |
| step-540 | 65.78% | 77.33% | 64.18% | 56.63% |
| step-460 | 65.78% | 77.33% | 62.69% | 57.83% |
| step-500 | 64.89% | 77.33% | 64.18% | 54.22% |
| step-440 | 64.44% | 77.33% | 62.69% | 54.22% |

step-606 is also the last training step — the val curve was still rising at
the end of the run, suggesting additional epochs might help. We stopped at 3
epochs as planned.

### Phase 5 results vs Phase 2 SFT baseline

| Metric | SFT only | + GRPO v3 (step-606) | Δ |
|---|---|---|---|
| val_225 overall | 64.44% | **66.22%** | **+1.78** |
| MCQ | 74.67% | 77.33% | +2.66 |
| Free single | 62.69% | 65.67% | +2.98 |
| Free multi | 56.63% | 56.63% | 0.00 |

GRPO lifted single-answer questions (both MCQ and FRQ_1) but did not move the
needle on multi-part free-response. Multi-part remains the weakest category
(the model often nails the early sub-answers but loses precision on later ones).

### OOM debugging trace (Phase 5 launch)

The vLLM colocate + KL ref-forward interaction took four configuration retries
before the run stabilized:

| Attempt | `vllm_gpu_memory_utilization` | `max_completion_length` | Outcome |
|---|---|---|---|
| 1 | 0.6 | 14,336 | OOM at step 0 (3.41 GB short) |
| 2 | 0.5 | 14,336 | OOM at step 3 (5.46 GB short) |
| 3 | 0.45 | 14,336 | OOM at step 3 (8.12 GB short) |
| 4 (final) | **0.35** | **10,240** | Stable, ran 16 h to completion |

The decisive change was lowering `max_completion_length` from 14,336 to 10,240
in addition to giving vLLM less GPU. After this point the
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` environment variable
prevented any later fragmentation-induced OOM across the 606 training steps.

---

## Phase 6 — Inference Pipeline

### Configuration for final CSV generation

| Parameter | Value |
|---|---|
| Model | Qwen3-4B-Thinking + merged SFT base + GRPO step-606 LoRA |
| Prompt | Run F template (per-type: MCQ vs free-response) |
| Sampling | K=4 self-consistency, temperature 1.0 |
| Voting | Per-prompt majority vote on normalized `\boxed{}` answers |
| Output budget | 14,336 tokens completion (Thinking + answer) |
| Generation framework | vLLM standalone (not colocate) |

### Post-processing

1. Extract all `\boxed{}` content from each K=4 rollout
2. Apply Judger-style normalization (LaTeX → ASCII forms, common synonyms)
3. Majority-vote across the K=4 rollouts on the normalized answer
4. Backfill: for 10 prompts where the winning rollout had a malformed/truncated
   final boxed, substitute from a parallel `winning_answer` field extracted
   from the same K=4 group

### Final CSV

`runpod_h100/handoff/private_submission.csv`:
- **943 rows** (full private test set)
- **930 of 943 (98.6%) contain `\boxed{...}`**
- Median response length: 13,675 chars
- 0 empty responses

---

## Compute Budget (Phase 2 + Phase 5 final pipeline only)

All training and inference on RunPod **H100 PCIe 80GB** at $2/hr.

| Stage | Time | Cost |
|---|---|---|
| Environment setup + model download | ~30 min | $1 |
| Phase 2 SFT (5 epochs × 737 examples) | ~3 h | $6 |
| Merge LoRA → standalone BF16 | ~5 min | $0.10 |
| val_225 intermediate eval | ~10 min | $0.30 |
| Phase 5 GRPO v3 (3 epochs × 606 steps) | ~16 h | $32 |
| Checkpoint sweep (10 ckpts × val_225) | ~1.5 h | $3 |
| Best-checkpoint merge + HF upload | ~30 min | $1 |
| Public K=1 + private K=4 SC inference | ~3 h | $6 |
| Submission CSV assembly | ~10 min | $0.30 |
| **Total (Phase 2 SFT + Phase 5 GRPO v3)** | **~24.5 h** | **~$50** |

(Phase 1 prompt engineering and the earlier Phase 3 / Phase 4 GRPO attempts
ran on separate machines and budgets earlier; their compute is not included
in the table above.)

---

## Reproducibility

### Model artifacts on HuggingFace Hub

Three separate Hub repositories make the full pipeline reproducible. The
primary artifact (used by `run_inference()`) is the fully merged model; the
two LoRA adapters are also published for transparency and to allow re-creating
the intermediate SFT-merged base.

| Repo | Type | Size | Base | Purpose |
|---|---|---|---|---|
| `JaasonYuu/jason-cse151b-model` | **Merged BF16 model** | ~7.6 GB | n/a (self-contained) | **Primary** — load directly in `run_inference()`. Contains Qwen3-4B-Thinking-2507 + SFT LoRA + GRPO v3 LoRA (step-606) all merged into a single BF16 checkpoint. |
| `JaasonYuu/jason-cse151b-sft-lora` | LoRA adapter | ~250 MB | `Qwen/Qwen3-4B-Thinking-2507` | Phase 2 SFT adapter alone (r=64, α=128). Apply on top of base Qwen3-4B-Thinking and merge to reproduce the SFT-merged intermediate state. |
| `JaasonYuu/jason-cse151b-grpo-lora` | LoRA adapter | ~125 MB | SFT-merged base | Phase 5 GRPO v3 adapter alone (r=32, α=64), the best-by-val_225 step-606 checkpoint. Apply on top of the SFT-merged base to reproduce the final policy. |

For straight inference, the merged model (`JaasonYuu/jason-cse151b-model`) is
all that is needed — it can be loaded as a single `AutoModelForCausalLM` and
served via vLLM directly, with no adapter-merging step at runtime.

### Single-entry-point `run_inference()`

The public submission repository exposes one function that performs the full
inference pipeline end-to-end. When invoked it:

1. Loads the fully merged BF16 model `JaasonYuu/jason-cse151b-model` from
   HuggingFace Hub (no adapter-merging step required at runtime; the model is
   already Qwen3-4B-Thinking + SFT LoRA + GRPO v3 LoRA all merged)
2. Reads `private.jsonl` (943 prompts)
3. For each prompt: builds the Run F system + user prompt (auto-selecting MCQ
   vs free-response variant), generates K=4 samples at temperature 1.0 via
   vLLM
4. Applies post-processing: boxed extraction, Judger-style normalization,
   K=4 majority vote, backfill for malformed final-boxed cases
5. Writes the final `private_submission.csv` in the expected `id,response`
   format

Estimated reproduction time on a single H100 PCIe 80GB: **~30 min** for the
full 943-prompt private set at K=4 SC.

> If a reviewer wishes to reconstruct the merged model from the LoRA
> adapters instead (e.g., for transparency or to inspect each phase's
> contribution), the recipe is:
> 1. `peft.PeftModel.from_pretrained("Qwen/Qwen3-4B-Thinking-2507", "JaasonYuu/jason-cse151b-sft-lora").merge_and_unload()`
> 2. Save → SFT-merged BF16 base
> 3. `peft.PeftModel.from_pretrained(<sft-merged>, "JaasonYuu/jason-cse151b-grpo-lora").merge_and_unload()`
> 4. The result is byte-equivalent (modulo merge-order float-rounding) to
>    `JaasonYuu/jason-cse151b-model`.

---

## Results Summary

| Metric | Score |
|---|---|
| **val_225 overall (held-out public)** | **66.22%** |
| **Kaggle private leaderboard** | _to be added after final upload_ |

Phase progression on val_225:

| # | Stage | val_225 | Cumulative Δ vs baseline |
|---|---|---|---|
| — | Baseline (untuned 4B Thinking + naive prompt, K=1) | ~47% | — |
| 1 | Phase 1 / Run A initial structured prompt (no fine-tune) | ~53–54% | +6 pp |
| 1 | Phase 1 / Run F production prompt (no fine-tune) | **58.67%** | +12 pp |
| 2 | Phase 2 SFT (Run F prompt) | **64.44%** | +17 pp |
| 3 | Phase 3 GRPO v1 (cancelled, regressed) | _−2.22 pp vs SFT_ | _−_ |
| 4 | Phase 4 GRPO v2 (4B + LoRA-v1 merged base, 196 edge-filtered prompts) | **64.00%** (val_225) / 66.25% (full public, training-set-leaky) | +17 pp |
| 5 | **Phase 5 GRPO v3 (4B, step-606)** | **66.22%** | **+19 pp** |

Phase 5 reaches **+2.22 pp val_225 over Phase 4** on the same 4B model
family (64.00% → 66.22%). The improvement came from three orthogonal Phase 5
additions on top of the v2 bugfixes: (a) switching to the SFT-merged base
trained at max_seq=16384, (b) hard-prompt boost via dataset duplication so
the 100 most sparse-reward prompts get K_eff=8 worth of rollouts per
epoch, and (c) re-introducing KL anchoring (β=0.04) to prevent the policy
from drifting away from the SFT-learned reasoning style.

Per-type breakdown (val_225, step-606):

| Question type | Accuracy |
|---|---|
| MCQ | 77.33% |
| Free-response (single answer) | 65.67% |
| Free-response (multi-part) | 56.63% |

---

## Limitations and Honest Notes

- Multi-part free-response remains the weakest category (56.63%). The model
  often answers early sub-parts correctly but degrades on later parts of
  multi-step problems. GRPO v3 did not improve this category over SFT.
- 13 of 943 submissions (1.4%) lacked a clean final `\boxed{}`; post-processing
  fell back to the highest-confidence sub-answer extracted from the K=4 vote.
- The 100 hardest public prompts (pass@K=0 at baseline) are still partially
  beyond the model's reach even with the K_eff=8 boost; some sparse-reward
  prompts contribute 0 gradient throughout training.
- We stopped at 3 GRPO epochs (606 steps) but the val curve was still rising;
  more epochs would likely yield additional gains at proportional GPU cost.

---

## Conclusion

This submission demonstrates that a carefully engineered SFT + GRPO pipeline on
a small (4B) reasoning-trained base can approach the quality of much larger
ensemble baselines on a domain-specific math competition. The contribution is
less in any single algorithmic innovation and more in the cumulative tightening
of the train/inference loop: aligning prompt distributions across phases,
matching data-prep and training caps for max-seq, calibrating vLLM colocate
memory under KL anchoring, and concentrating GRPO's learning budget on
sparse-reward prompts via dataset duplication. Each of these decisions
individually contributed only a fraction of a point, but together they brought
the system from a failing RL run (Phase 3 GRPO v1 collapsed at grad-norm
1e-18) to the final 66.22% val_225 / +1.78 pp improvement over SFT alone.

The remaining headroom is real: multi-part free-response is still the weakest
category, the val curve had not converged at 3 epochs, and roughly 1% of
private-set responses lack a clean final boxed answer. We close with these
limitations recorded honestly, and the full configuration committed in
hyperparameters so that any rerun on the published model artifacts reproduces
the reported numbers within the expected sampling variance.
