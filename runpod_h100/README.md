# CSE 151B — H100 LoRA + GRPO Training on RunPod

Self-contained training pipeline for **Qwen3-4B-Thinking-2507** on H100 80GB.

```
Pipeline:
  1. BF16 LoRA SFT       (~1.5h, $3)   ← lora_sft_h100
  2. Merge to BF16        (~5 min)      ← lora_sft_merged
  3. GRPO on top of SFT  (~20.5h, $41) ← grpo_v6 (K=4 + hard-dup, beta=0.04, max_comp=14336)
  4. Download adapter, run inference locally
```

---

## Quick Start

### 1. Launch RunPod H100 (or H200)

- **GPU**: H100 PCIe 80GB ($2/h) — recommended
- **Template**: PyTorch 2.4+ with CUDA 12.4 (e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4`)
- **Disk**: 100 GB minimum (50 GB for model + checkpoints)
- **Network**: Mount /workspace as persistent volume

### 2. Clone this branch

```bash
cd /workspace
git clone -b runpod-h100-train <repo_url> cse151b
cd cse151b/runpod_h100
```

### 3. Install deps

```bash
pip install -r requirements.txt
# Verify base stack:
python -c "import torch, peft, trl, vllm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

#### flash-attn (OPTIONAL — pipeline auto-falls-back to SDPA)

The SFT script tries to import `flash_attn` and gracefully falls back to PyTorch SDPA if
not available. SDPA is **85–95% as fast as FA2 on H100 for 4B models** — for a 1.5h SFT
job, the difference is ~10–20 min. No reason to compile from source.

If you want the speedup anyway, **install a prebuilt wheel** (instant, no compile):

```bash
# Step 1: check your env
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python --version

# Step 2: match a wheel from https://github.com/Dao-AILab/flash-attention/releases
# Naming: flash_attn-<ver>+cu<CUDA>torch<TORCH>cxx11abiFALSE-cp<PYVER>-cp<PYVER>-linux_x86_64.whl
#
# Example for RunPod's PyTorch 2.4 + CUDA 12 + Python 3.11 image:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

**DO NOT** run `pip install flash-attn` (no wheel) — it tries to compile from source,
takes 30-60 min, often OOMs the build container, and produces broken binaries on
container/host CUDA mismatches.

If a wheel install fails or the URL 404s, just run the pipeline anyway — SDPA fallback
is automatic.

### 4. Run LoRA SFT (1-1.5h)

```bash
python scripts/train_lora_bf16.py
# Output: checkpoints/lora_sft_h100/final/
```

Monitor:
```bash
tail -f logs/train_lora.log  # if you redirected
```

Expected: loss drops from ~0.30 to ~0.08, ~5 epochs.

### 5. Merge LoRA → BF16

```bash
python scripts/merge_lora.py
# Output: checkpoints/lora_sft_merged/  (~8 GB)
```

### 6. (Optional) Eval on val_225

```bash
# Skip if going straight to GRPO; the val score is just an sanity check
# Use the eval_val_225 script from main repo (not included here)
```

### 7. Run GRPO (13-16h)

```bash
python scripts/train_grpo.py \
    --base checkpoints/lora_sft_merged \
    --output checkpoints/grpo_v6 \
    --epochs 3 \
    --num-generations 4
# Output: checkpoints/grpo_v6/final/  (LoRA adapter ~250 MB)
```

In background:
```bash
nohup python scripts/train_grpo.py --base checkpoints/lora_sft_merged > logs/grpo.log 2>&1 &
tail -f logs/grpo.log
```

### 8. Download artifacts

After GRPO completes, download just the adapter (~250 MB):
```bash
# On RunPod, compress
tar czf grpo_v6_adapter.tar.gz checkpoints/grpo_v6/final/

# Download via runpodctl or scp to your local machine
runpodctl send grpo_v6_adapter.tar.gz
```

Or push to HuggingFace Hub:
```bash
huggingface-cli login
huggingface-cli upload <your-username>/qlora-cse151b-grpo-v6 checkpoints/grpo_v6/final/
```

---

## Files Included

```
runpod_h100/
├── README.md                              ← you are here
├── requirements.txt                       ← deps
├── judger.py                              ← course Judger (reward function)
├── src/                                   ← cse151b_comp modules (prompts, evaluate, etc.)
├── data/
│   ├── h100_lora_sft.jsonl                ← 737 SFT training pairs (Run F prompts applied)
│   ├── grpo_train_extended_v6.jsonl       ← 305 GRPO prompts (196 public + 109 private verified)
│   ├── grpo_pool_v2.json                  ← 196 public edge-filtered IDs (reference)
│   ├── val_indices.json                   ← 225 val IDs (for evaluation)
│   ├── public.jsonl                       ← public train+val 1126 questions
│   └── private.jsonl                      ← private test 943 questions (no gold)
└── scripts/
    ├── train_lora_bf16.py                 ← BF16 LoRA SFT
    ├── merge_lora.py                      ← merge adapter into base
    └── train_grpo.py                      ← GRPO with v6 data
```

---

## Configuration Reference

### LoRA SFT (train_lora_bf16.py)

| Param | Value | Notes |
|---|---|---|
| Base | Qwen3-4B-Thinking-2507 | BF16, no quantization |
| LoRA r | 64 | larger than original 32 |
| LoRA alpha | 128 | 2:1 ratio |
| Target modules | q,k,v,o,gate,up,down | 7 modules |
| Batch size | 4 × grad_accum=2 = 8 effective | H100 80GB has headroom |
| Max seq | **16384** | rescues 89% of long-reasoning samples (median 11K tokens). +15 min vs 8192. |
| Epochs | 5 | |
| LR | 2e-4 cosine, warmup 5% | |
| Loss | completion_only | only train on response tokens |
| Expected time | 1-1.5h on H100 PCIe | |

### GRPO (train_grpo.py)

| Param | Value | Notes |
|---|---|---|
| Base | merged LoRA SFT model | |
| Adapter on top | LoRA r=32, alpha=64 | smaller for RL stability |
| K (generations per prompt) | **4 global** | with hard-dup, K_eff = 8 on 100 hard, 4 on rest |
| Hard-pool dup | **1×** | duplicates 100 all-wrong public IDs in training data |
| Temperature | 1.0 | high for exploration |
| Epsilon (DAPO clip) | 0.3 / 0.4 (low/high) | asymmetric — favors positive updates |
| Beta (KL) | **0.04** | gently anchors Thinking model to SFT to preserve chain-of-thought |
| Loss type | dr_grpo | length-normalized |
| IS level | sequence | stabler under vLLM colocate |
| Batch | 1 × grad_accum=8 = 8 effective | divisible by num_generations |
| LR | 1e-5 constant_with_warmup, warmup 5% | warmup avoids early gradient spike |
| Reward | Judger binary + length penalty | 1.0 / 0.5 / 0 — **MCQ exempt from length penalty** |
| max_completion_length | **14336** | covers P95 of SFT reasoning. vLLM stops at EOS so only the ~7% extreme-long prompts pay extra time |
| max_prompt_length | 2048 | Run F + question fit; don't change |
| vllm_max_model_len | **16384** (auto = 2048+14336) | total context budget per rollout |
| Checkpointing | save every 20 steps, keep 10 | for post-hoc best-checkpoint selection |
| Monitoring | TensorBoard | `tensorboard --logdir checkpoints/grpo_v6/logs` |
| Epochs | 3 | ~152 total steps for 405 rows × 3 / batch 8 (after dup) |
| Expected time | **~20.5h on H100 PCIe** | hard prompts get 2× attention + ~95% reasoning fully completes |

### Why "hard-pool dup" instead of K=8 uniform?

The 100 all-wrong public prompts are the sparse-reward "frontier": base model fails them
≥4/4 times under K=4 SC. Giving them K=8 attention (via duplication) doubles their chance
of getting at least one passing rollout in a group, which is the only way GRPO learns from
them. Easy prompts already have high pass@4 so giving them K=8 too is wasteful.

Cost comparison (H100 PCIe @ $2/h, 3 epochs):

| Strategy | Hard K | Easy K | gens | Time | Cost |
|---|---|---|---|---|---|
| `--hard-dup 1` (default) | 8 | 4 | 4860 | ~17h | $35 |
| `--hard-dup 0 --num-generations 4` | 4 | 4 | 3660 | ~13h | $26 |
| `--hard-dup 0 --num-generations 6` | 6 | 6 | 5490 | ~19.5h | $39 |
| `--hard-dup 0 --num-generations 8` | 8 | 8 | 7320 | ~26h | $52 |

The default (`--hard-dup 1`, K_global=4) is the best signal-per-dollar for our pool composition.

---

## Data Provenance

### SFT data (`h100_lora_sft.jsonl`, 737 rows)

```
612 public_K=32_self_distill (n_correct >= 2)
100 private_verified_100 (manually verified by author)
 25 private_dual_verified (hybrid+solved 30B teacher agree)
```

All formatted with **Run F prompt template** (build_prompt_runf from src/cse151b_comp/prompts.py).

### GRPO data (`grpo_train_extended_v6.jsonl`, 305 rows)

```
196 public_edge_filtered (1-3/K samples correct on K=32 SC; real gold)
 80 private_verified_uncertain (hand-verified, sympy-validated, Judger-friendly format)
 25 private_hybrid_solved_agree (dual-pipeline verification)
  4 private_lora_solved_hand_verified (single-pipeline + sympy hand-verification)
```

v6 fixes vs v4: 3 content errors (ID 60, 281, 902), 2 atom-split bugs (ID 473, 396),
4 new high-confidence problems from the 140-solved subset.

---

## Expected Results

| Stage | Val_225 | Private (estimated) |
|---|---|---|
| Local v3.5 baseline (QLoRA) | 60.0% | 0.58-0.62 |
| **H100 LoRA SFT** | **62-65%** | 0.60-0.65 |
| **+ GRPO** | **66-69%** | **0.65-0.71** |

For reference, the original Blackwell training achieved:
- distill_v1 SFT: 63% val
- + GRPO v2: 66.25% val
- → hybrid_sc_grpo_extended.csv: 0.66 leaderboard

---

## Troubleshooting

### Out of memory during SFT
- Reduce `--per-device-bsz` from 4 to 2 or 1
- Enable gradient checkpointing: edit train_lora_bf16.py: `gradient_checkpointing=True`

### OOM during GRPO
- GRPO with K=4 + colocate vLLM needs ~50-70 GB peak
- If under 60GB available, set `--num-generations 2`
- Or split policy + ref to different GPUs

### vLLM startup error
- Add `VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo` env vars
- For RunPod's network, this avoids picking up external IPs

### Tokenizer warning about Mistral regex
- Harmless. Ignore.

---

## Cost Estimate

```
H100 PCIe 80GB @ $2/h (default: K=4 + hard-dup 1, beta=0.04):
  - SFT 1.5h:     $3
  - Merge:        $0.10
  - GRPO ~20.5h:  $41
  ────────────────
  Total:          ~$47

H200 141GB @ $3-5/h:
  - Full pipeline ~12-15h: $36-75 (faster but pricier)

To revert to old cheap config (K=4 uniform, no KL, no hard-dup):
  python scripts/train_grpo.py --num-generations 4 --beta 0.0 --hard-dup 0 \
      --base checkpoints/lora_sft_merged
```

---

## What Comes After GRPO?

Once you have the GRPO adapter:

1. **Download `checkpoints/grpo_v6/final/`** to local machine
2. **Merge GRPO adapter** with SFT-merged base (or use PEFT directly in vLLM)
3. **Run inference on private** with K=8 SC + Run F prompt + Judger-friendly extraction
4. **Generate `submission.csv`**
5. **Submit Kaggle** + GitHub repo on Gradescope

Final inference script template (run on local 4090):
```python
from vllm import LLM, SamplingParams
from cse151b_comp.prompts import build_prompt_runf

llm = LLM(model="path/to/grpo_merged", dtype="bfloat16")
# K=8 SC inference loop
# Apply Run F prompt
# Vote on boxed
# Write CSV
```

---

## Files NOT included (need to fetch separately if needed)

- The base Qwen3-4B-Thinking-2507 (auto-downloaded by HF on first run)
- Local v3.5 merged model (8 GB; if you want to skip SFT and start GRPO from it)
- Previous submission CSVs (for reference / fallback)
