# CSE 151B — H100 LoRA + GRPO Training on RunPod

Self-contained training pipeline for **Qwen3-4B-Thinking-2507** on H100 80GB.

```
Pipeline:
  1. BF16 LoRA SFT       (~1.5h, $3)   ← lora_sft_h100
  2. Merge to BF16        (~5 min)      ← lora_sft_merged
  3. GRPO on top of SFT  (~13-16h, $30) ← grpo_v4
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
# May need: pip install flash-attn --no-build-isolation
# Verify:
python -c "import torch, peft, trl, vllm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

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
    --output checkpoints/grpo_v4 \
    --epochs 3 \
    --num-generations 4
# Output: checkpoints/grpo_v4/final/  (LoRA adapter ~250 MB)
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
tar czf grpo_v4_adapter.tar.gz checkpoints/grpo_v4/final/

# Download via runpodctl or scp to your local machine
runpodctl send grpo_v4_adapter.tar.gz
```

Or push to HuggingFace Hub:
```bash
huggingface-cli login
huggingface-cli upload <your-username>/qlora-cse151b-grpo-v4 checkpoints/grpo_v4/final/
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
│   ├── grpo_train_extended_v4.jsonl       ← 301 GRPO prompts (196 public + 105 private verified)
│   ├── grpo_pool_v2.json                  ← 196 public edge-filtered IDs (reference)
│   ├── val_indices.json                   ← 225 val IDs (for evaluation)
│   ├── public.jsonl                       ← public train+val 1126 questions
│   └── private.jsonl                      ← private test 943 questions (no gold)
└── scripts/
    ├── train_lora_bf16.py                 ← BF16 LoRA SFT
    ├── merge_lora.py                      ← merge adapter into base
    └── train_grpo.py                      ← GRPO with v4 data
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
| Max seq | 8192 | covers 78% of training data |
| Epochs | 5 | |
| LR | 2e-4 cosine, warmup 5% | |
| Loss | completion_only | only train on response tokens |
| Expected time | 1-1.5h on H100 PCIe | |

### GRPO (train_grpo.py)

| Param | Value | Notes |
|---|---|---|
| Base | merged LoRA SFT model | |
| Adapter on top | LoRA r=32, alpha=64 | smaller for RL stability |
| K (generations per prompt) | 4 | edge-filtered → meaningful std |
| Temperature | 1.0 | high for exploration |
| Epsilon (DAPO clip) | 0.3 / 0.4 (low/high) | asymmetric |
| Beta (KL) | 0.0 | no KL drag |
| Loss type | dr_grpo | length-normalized |
| Batch | 1 × grad_accum=8 = 8 effective | divisible by num_generations |
| LR | 1e-5 constant | low to avoid catastrophic forget |
| Reward | course Judger binary + length penalty | 1.0 correct + ≥2000 chars, 0.5 correct + short, 0 wrong |
| Epochs | 3 | |
| Expected time | 13-16h on H100 PCIe | |

---

## Data Provenance

### SFT data (`h100_lora_sft.jsonl`, 737 rows)

```
612 public_K=32_self_distill (n_correct >= 2)
100 private_verified_100 (manually verified by author)
 25 private_dual_verified (hybrid+solved 30B teacher agree)
```

All formatted with **Run F prompt template** (build_prompt_runf from src/cse151b_comp/prompts.py).

### GRPO data (`grpo_train_extended_v4.jsonl`, 301 rows)

```
196 public_edge_filtered (1-3/K samples correct on K=32 SC; real gold)
 80 private_verified_uncertain (model uncertain + manually verified; judge-friendly format)
 25 private_hybrid_solved_agree (dual verification)
```

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
H100 PCIe 80GB @ $2/h:
  - SFT 1.5h:     $3
  - Merge:        $0.10
  - GRPO 13-16h:  $26-32
  ────────────────
  Total:          ~$29-35

H200 141GB @ $3-5/h:
  - Full pipeline ~12-18h: $40-90 (faster but pricier)
```

---

## What Comes After GRPO?

Once you have the GRPO adapter:

1. **Download `checkpoints/grpo_v4/final/`** to local machine
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
