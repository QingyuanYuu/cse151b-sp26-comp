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
