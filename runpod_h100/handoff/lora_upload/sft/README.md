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
