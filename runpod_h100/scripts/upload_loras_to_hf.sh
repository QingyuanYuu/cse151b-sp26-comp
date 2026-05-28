#!/usr/bin/env bash
# Upload SFT and GRPO LoRA adapters as separate HF repos.
set -euo pipefail

REPO=/workspace/cse151b-grpo/cse151b-sp26-comp/runpod_h100
HF_BIN=/workspace/cse151b-grpo/.venv/bin/hf
STAGE_DIR="$REPO/handoff/lora_upload"
SFT_REPO="JaasonYuu/jason-cse151b-sft-lora"
GRPO_REPO="JaasonYuu/jason-cse151b-grpo-lora"

mkdir -p "$STAGE_DIR/sft" "$STAGE_DIR/grpo"

echo "[lora] preparing SFT LoRA staging..."
SFT_SRC="$REPO/checkpoints/lora_sft_h100/final"
cp "$SFT_SRC/adapter_model.safetensors" "$STAGE_DIR/sft/"
cp "$SFT_SRC/adapter_config.json"       "$STAGE_DIR/sft/"
cp "$SFT_SRC/tokenizer.json"            "$STAGE_DIR/sft/"
cp "$SFT_SRC/tokenizer_config.json"     "$STAGE_DIR/sft/"
cp "$SFT_SRC/chat_template.jinja"       "$STAGE_DIR/sft/"

cat > "$STAGE_DIR/sft/README.md" <<'EOF'
---
language: en
license: apache-2.0
tags:
  - lora
  - peft
  - math
  - reasoning
  - qwen3
  - sft
base_model: Qwen/Qwen3-4B-Thinking-2507
library_name: peft
---

# CSE 151B SP26 Math Reasoning — SFT LoRA adapter (r=64)

Stage-1 SFT LoRA for the CSE 151B Spring 2026 math reasoning competition.

Trained on top of `Qwen/Qwen3-4B-Thinking-2507` with `completion_only_loss=True`,
producing the **SFT-merged** base which then powered Stage-2 GRPO.

## Hyperparameters

- LoRA r = **64**, alpha = **128**, dropout = 0.05
- target_modules = `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`
- 5 epochs, LR 2e-4 cosine, warmup 5%
- max_seq = 16384, BF16, gradient checkpointing
- Effective batch size 8 (bsz=1 × grad_accum=8)
- Training data: 737 SFT pairs (self-distill from K=32 SC + private hand-verified)

## val_225 accuracy

After merging into base: **64.44 %** (vs the 60 % QLoRA baseline).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Thinking-2507", dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "JaasonYuu/jason-cse151b-sft-lora")
```

## See also

- Full SFT+GRPO merged BF16: [JaasonYuu/jason-cse151b-model](https://huggingface.co/JaasonYuu/jason-cse151b-model)
- GRPO LoRA: [JaasonYuu/jason-cse151b-grpo-lora](https://huggingface.co/JaasonYuu/jason-cse151b-grpo-lora)
EOF

echo "[lora] preparing GRPO LoRA staging (step-606 best by val_225)..."
GRPO_SRC="$REPO/checkpoints/grpo_v6/checkpoint-606"
cp "$GRPO_SRC/adapter_model.safetensors" "$STAGE_DIR/grpo/"
cp "$GRPO_SRC/adapter_config.json"       "$STAGE_DIR/grpo/"
cp "$GRPO_SRC/tokenizer.json"            "$STAGE_DIR/grpo/"
cp "$GRPO_SRC/tokenizer_config.json"     "$STAGE_DIR/grpo/"
cp "$GRPO_SRC/chat_template.jinja"       "$STAGE_DIR/grpo/"

# Rewrite base_model_name_or_path so it points to a HF-pullable model
/workspace/cse151b-grpo/.venv/bin/python -c "
import json, pathlib
p = pathlib.Path('$STAGE_DIR/grpo/adapter_config.json')
cfg = json.loads(p.read_text())
cfg['base_model_name_or_path'] = 'JaasonYuu/jason-cse151b-model'
p.write_text(json.dumps(cfg, indent=2))
print('[lora] GRPO adapter_config base_model_name_or_path set to JaasonYuu/jason-cse151b-model')
"

cat > "$STAGE_DIR/grpo/README.md" <<'EOF'
---
language: en
license: apache-2.0
tags:
  - lora
  - peft
  - math
  - reasoning
  - qwen3
  - grpo
  - rl
base_model: JaasonYuu/jason-cse151b-model
library_name: peft
---

# CSE 151B SP26 Math Reasoning — GRPO LoRA adapter (r=32, step-606 best)

Stage-2 GRPO LoRA, trained on top of the **SFT-merged** base.

This is the best-by-val_225 checkpoint (step-606) selected from a 27-checkpoint sweep.

> **NOTE**: `base_model_name_or_path` points to `JaasonYuu/jason-cse151b-model`,
> which is the *fully merged* SFT+GRPO model — applying this adapter on top of
> that would double-apply the GRPO delta. The TRUE base of this adapter is the
> SFT-merged BF16 model (Qwen3-4B-Thinking + SFT LoRA merged). To reproduce that
> base, apply [JaasonYuu/jason-cse151b-sft-lora](https://huggingface.co/JaasonYuu/jason-cse151b-sft-lora)
> to `Qwen/Qwen3-4B-Thinking-2507` and merge.

## Hyperparameters

- LoRA r = **32**, alpha = **64**, dropout = 0.05
- target_modules = `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`
- 3 epochs (606 steps), LR 1e-5 constant_with_warmup (5%)
- max_completion_length = 10240, beta (KL) = 0.04
- num_generations K = 4, hard-pool duplication = 1× (effective K=8 on 100 hard prompts)
- Loss: dr_grpo, importance_sampling_level = sequence, scale_rewards = none
- Reward: course Judger binary + length penalty (MCQ exempt)

## val_225 accuracy

Applied on SFT-merged base: **66.22 %** (+1.78 pp over SFT alone, +2.22 pp over base
Qwen3-4B-Thinking-2507 with starter prompts).

## Usage (after reconstructing SFT-merged base)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Step 1: reconstruct SFT-merged base
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Thinking-2507", dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True,
)
sft = PeftModel.from_pretrained(base, "JaasonYuu/jason-cse151b-sft-lora")
sft_merged = sft.merge_and_unload()

# Step 2: apply GRPO LoRA on top
model = PeftModel.from_pretrained(sft_merged, "JaasonYuu/jason-cse151b-grpo-lora")
```

OR just use the [pre-merged SFT+GRPO model](https://huggingface.co/JaasonYuu/jason-cse151b-model).

## See also

- Pre-merged SFT+GRPO BF16: [JaasonYuu/jason-cse151b-model](https://huggingface.co/JaasonYuu/jason-cse151b-model)
- SFT LoRA: [JaasonYuu/jason-cse151b-sft-lora](https://huggingface.co/JaasonYuu/jason-cse151b-sft-lora)
EOF

echo ""
echo "[lora] pushing SFT LoRA → $SFT_REPO"
$HF_BIN repo create "$SFT_REPO" --type model -y 2>&1 | tail -3 || true
$HF_BIN upload "$SFT_REPO" "$STAGE_DIR/sft" --repo-type model \
    --commit-message "SFT LoRA (r=64, val_225 64.44%)"

echo ""
echo "[lora] pushing GRPO LoRA → $GRPO_REPO"
$HF_BIN repo create "$GRPO_REPO" --type model -y 2>&1 | tail -3 || true
$HF_BIN upload "$GRPO_REPO" "$STAGE_DIR/grpo" --repo-type model \
    --commit-message "GRPO LoRA r=32 (step-606 best, val_225 66.22%)"

echo ""
echo "[lora] done. Repos:"
echo "  https://huggingface.co/$SFT_REPO"
echo "  https://huggingface.co/$GRPO_REPO"
