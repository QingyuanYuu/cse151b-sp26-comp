# Pipeline status snapshot — 2026-05-28 03:32

For Claude Code resume / handoff. Read this first.

## TL;DR

- **SFT + Merge + val_225 完成**
- **val_225 baseline: 64.44%** (Run F prompt, SFT-merged BF16)
- **GRPO 训练中** — step ~40/606, ETA ~17h
- 没有阻塞错误
- 一切都在 `tmux session "grpo"` 里跑

## 已完成

| Stage | Result |
|---|---|
| SFT (5 epoch, max_seq=16384, r=64, bsz=1+ga=8+GC) | loss 0.46→0.121, token acc 96% |
| Merge LoRA → BF16 | `checkpoints/lora_sft_merged/` 7.6 GB |
| val_225 eval (Run F + v2 budget) | **64.44%** (MCQ 74.67% / free_single 62.69% / free_multi 56.63%) |

结果落盘：
- `results/val225_sft.jsonl`
- `logs/sft_full.log`, `logs/merge_full.log`, `logs/eval_val225.log`

## GRPO 训练状态（当前）

- tmux session：`grpo`
- log：`logs/grpo_full.log` (raw)，`logs/grpo_pipeline.log` (with stage banners)
- 配置：
  - 3 epochs，K=4，hard-dup=1（hard pool K_eff=8）
  - max_completion=10240，max_prompt=2048，vllm_max=12288
  - vllm_gpu_memory_utilization=0.35
  - beta=0.04 (KL to ref)，dr_grpo，importance_sampling=sequence，scale_rewards=none
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  - bsz=1，grad_accum=8，effective batch 8
  - LR 1e-5 constant_with_warmup 5%
  - **总步数 606**，save_steps=20，save_total_limit=10
- 启动时间：02:31:22
- 当前进度（03:32）：~step 40/606，96 s/step，trl ETA 15h11m

### 历次 OOM 修复轨迹

1. util=0.6 → OOM step 0（3.41 GB）→ 降到 0.5
2. util=0.5 → squeak（作者实测能跑，我们也很紧）→ 我激进改 0.45 → step 3 OOM (5.46 GB)
3. util=0.45 + expandable_segments → step 3 仍 OOM (8.12 GB on max=14336)
4. **util=0.35 + max_completion 14336 → 10240** → 稳定（当前配置）

### Resume 入口（如再 OOM）

```bash
tmux kill-session -t grpo
tmux new-session -d -s grpo -c /workspace/cse151b-grpo/cse151b-sp26-comp/runpod_h100 \
  'bash scripts/run_grpo_only.sh --resume 2>&1 | tee -a logs/grpo_pipeline.log'
```

`--resume` 让 `run_grpo_only.sh` 跳过 `rm -rf grpo_v6`，trainer 自动从最大 checkpoint-N 恢复。

如果再 OOM 想再降参数：
- 第 1 招：`max_completion_length 10240 → 8192`（编辑 `train_grpo.py` 默认）
- 第 2 招：`hard-dup 1 → 0`（hard pool 不重复，损失 hard 信号但内存大幅缓解）

## Pipeline 自动接力（Stage 5 → 9）

GRPO 训完后 `run_grpo_only.sh` 自动跑 Stage 5/6/7/8/9：

- **Stage 5**: best-ckpt-by-val sweep（用 enable_lora 热切换跑 val_225）
- **Stage 6**: merge best GRPO adapter → BF16（`checkpoints/grpo_v6_merged/`）
- **Stage 7**: upload to HF Hub（`JaasonYuu/jason-cse151b-model`）
- **Stage 8**: private K=8 SC inference（`results/private_sc_k8.jsonl`）
- **Stage 9**: Kaggle CSV（`results/private_submission.csv`）

Stage 5 详情：
```bash
$PY scripts/eval_all_ckpts.py \
    --base checkpoints/lora_sft_merged \
    --grpo-dir checkpoints/grpo_v6 \
    --out results/grpo_ckpt_sweep.jsonl \
    --summary results/grpo_ckpt_sweep_summary.json
```

用 vLLM `enable_lora` 热切换所有 GRPO checkpoint，跑 val_225，挑准确率最高的 ckpt。
- 预计 ~1.5-2 h（base 加载 1 次，每个 ckpt eval ~15 min）
- 输出 `results/grpo_ckpt_sweep_summary.json` 含 ranking + best label/path

## 健康指标

| 指标 | 健康范围 | 现在 |
|---|---|---|
| reward 0<r<1 比例 | 50-70% | **67%** ✅ |
| reward=0 (hard) | 20-40% | 33% ✅ |
| kl | < 0.01（beta=0.04 anchor） | ~5e-4 ✅ |
| GPU mem | < 80 GB | 77.9 GB ✅ |
| clipped_ratio | 偶发 ≤ 0.75 | 0-0.75 ✅ |
| step time | 30-110s 波动 | 73-96s avg ✅ |

## 监控

被 Claude Code 维护，不在 tmux 里：
- Cron job `acee4ad1`：每 20 分钟报告 GRPO 进度
- Monitor `bibtq9rsj`：跟踪 grpo_pipeline.log 关键事件

**这两个会随当前 Claude Code 进程死掉。** 新 Claude Code session 需要重建：
```
/loop 20m 汇报GRPO进度
```
（monitor 一般不必重建，cron 报告里 grep log 已经够用）

## 重要文件路径

```
/workspace/cse151b-grpo/cse151b-sp26-comp/runpod_h100/
├── scripts/
│   ├── train_lora_bf16.py     SFT
│   ├── merge_lora.py          Merge
│   ├── eval_val225.py         val_225 评估 (我写的)
│   ├── eval_all_ckpts.py      best-ckpt-by-val sweep (我写的)
│   ├── train_grpo.py          GRPO (origin + 我的 TRL 1.5 修复)
│   ├── run_full_pipeline.sh   SFT + Merge
│   ├── run_post_sft.sh        val_225 + GRPO + sweep
│   └── run_grpo_only.sh       GRPO + sweep（OOM 重试用，支持 --resume）
├── checkpoints/
│   ├── lora_sft_h100/         SFT LoRA adapter
│   ├── lora_sft_merged/       BF16 merged base（GRPO 的 base）
│   └── grpo_v6/               GRPO 输出（训练中，每 20 步存）
│       ├── checkpoint-20/     有 ✅
│       ├── checkpoint-40/     有 ✅
│       └── ... （动态生成）
└── logs/
    ├── sft_full.log
    ├── merge_full.log
    ├── eval_val225.log        val_225 = 64.44% 那个
    ├── grpo_full.log          GRPO trainer log（多次 run 累积）
    ├── grpo_pipeline.log      GRPO pipeline 的 stage banners
    └── eval_all_ckpts.log     Stage 5（GRPO 跑完才有）
```

## 环境

- Python 3.11.10 in `/workspace/cse151b-grpo/.venv`
- torch 2.11.0+cu130, transformers 5.9.0, peft 0.19.1, trl 1.5.1, vllm 0.21.0
- bitsandbytes 0.49.2, antlr4-python3-runtime 4.11, tensorboard 2.20.0
- 没装 flash-attn（vLLM 内置 FA3 用于 rollout；训练用 SDPA）
- HF cache：`/workspace/.cache/huggingface/`（model 已下载）
- HF auth：JaasonYuu

## 重要约定（feedback / 已确认）

- 用中文回复
- 每 20 分钟汇报 GRPO 进度（cron acee4ad1）
- 不要碰 SFT pipeline 的 tmux（已完成，session 已自然退出）
- `vllm_gpu_memory_utilization` 偏激进会 OOM；用 0.35 是当前安全配置
- 用 `tmux attach -t grpo -r` 看进度（-r 只读，按键不影响进程）

## 下一步预期

1. GRPO 跑到 step ~606（明天 ~16:00-20:00）
2. Stage 5 自动接力跑 ckpt sweep（~1.5-2h）
3. 输出 `grpo_ckpt_sweep_summary.json`，选最佳 ckpt
4. 下载 best adapter，scp 给 4090（或 push HF Hub）做 private inference + K=8 SC

## 切换到 tmux Claude Code

```bash
# SSH 进 runpod
tmux new -s claude
claude --resume      # 选这个对话
# 在新 Claude Code 里立刻：
/loop 20m 汇报GRPO进度
```
