# CSE 151B 6 天冲刺 —— 本机职责：LoRA 训练 + 大显存推理

## 这份计划的定位

**机器分工**（用户 2026-05-04 明确）：

- **本机（`jason/dev2` on RTX PRO 6000 Blackwell 96GB）**：只跑需要大显存的活
  —— LoRA / QLoRA 训练、自蒸馏用的大 K SC（K=32 BF16）、必要时的 BF16 推理
- **另一台机器（4090 / DataHub / 其他）**：日常推理、prompt 迭代、val 评分、Kaggle 提交 CSV 生成

`PLAN.md` 是项目总路线图。`PLAN_6DAY.md`（本文件）是**本机本分支**的执行计划，不覆盖推理/提交工作流。

`jason/dev` 已合入 `jason/dev2`（commit `b904aa3`）：v6 prompts、self_consistency / voting、milestone 报告 PDF 都在。**LoRA 全栈待建**。

## 关键约束（再确认）

来自 `PLAN.md` 比赛硬约束：
1. 模型锁 `Qwen/Qwen3-4B-Thinking-2507`；**LoRA / QLoRA 允许**（adapter 不算改 base）
2. 推理时禁外部工具 / API / code interpreter（SymPy 在 prompt 设计阶段可用，generate 时不能跑代码）
3. 提交 CSV `id, response`；943 题
4. Kaggle submission 配额受限

来自 `reports/public_private_gap_analysis.md`：
- public val 和 private leaderboard 反向过 8pp，**variance reduction (SC) 比 prompt 改写靠谱**
- LoRA 是 SC 之后唯一剩下的精度杠杆

## 本机做什么 / 不做什么

| 工作流 | 本机做？ | 备注 |
|---|---|---|
| 单次推理（K=1）on public/private | ❌ | 4090 / DataHub 即可 |
| K=4–8 SC on public/private | ❌ | 4090 24GB 能装 |
| **K=32 SC on public（自蒸馏数据源）** | ✅ | 4B INT4 + n=32 KV cache 在 24GB 上紧；96GB 轻松 |
| **SFT 数据筛选**（K 个样本里挑对的当 target） | ✅ | 跟 K=32 同流程 |
| **LoRA 训练**（4B BF16，bsz=4，r=32） | ✅ | 96GB 关键用途 |
| LoRA + SC 在 val 上 ablate | ✅ | 训完直接评 |
| 最终 submission CSV 生成 | ❌ | 另一台跑（adapter 传过去） |
| Kaggle 提交本身 | ❌ | 另一台跑 |
| Milestone / final 报告写 | ❌ | 另一台 |

## 6 天日程（2026-05-04 → 2026-05-09）

### Day 1（今天，剩约半天）—— SFT 数据池：K=32 SC on public

**目标**：拿到一份 `data/sc_distill_pool.jsonl`，每题 32 个样本 + 抽到的答案 + Judger 判断的正确性。这是 LoRA 训练数据的源头。

1. 起子分支：`git checkout -b day1-distill-pool`（从 `jason/dev2`）
2. 检查 `cse151b-sc` CLI 是否支持"保留全部 K 个 response"输出（看 `self_consistency.py` 的 schema：已经包含 `all_responses` / `all_extracted` / `vote_counts`，✅ 直接能用）
3. 跑：
   ```bash
   uv run cse151b-sc --input data/public.jsonl \
       --output results/sc_v6_k32_public.jsonl \
       --k 32 --temperature 0.7 --top-p 0.95
   ```
   预估时长：4B INT4 + Blackwell + n=32 + 1126 题，**1.5–3 小时**（vLLM 共享 prefix KV cache，K=32 比 K=1 慢约 4–8×，不是 32×）
4. 准备过程中，另一 shell 写 `src/cse151b_comp/prepare_sft_data.py`（新文件）：
   - 读 `results/sc_v6_k32_public.jsonl`
   - 按 `data/val_indices.json` 排除 val_225（只用训练区段，约 901 题）
   - 对每题：从 K 个样本里挑 **judger 判正确**的子集；如果有，按"vote 中标 → 长度最长 → 提取格式最干净"优先级选一个 winning response 当 SFT target
   - 写出 `data/sft_train.jsonl`：`{id, system_prompt, user_prompt, target_response}`
   - target_response 必须包含**完整 `<think>...</think>` + 末尾 `\boxed{...}`**（不能用裸 gold，否则模型学会丢 boxed）
   - 全题对的 `pass@32` 比例 = SFT 数据量上限；预期 60–80 % × 901 = 540–720 examples
5. 配 `tests/test_prepare_sft.py`：覆盖 (a) 全 K 都错的题被丢弃 (b) winning 选择逻辑 (c) target 必含 `<think>` 和 `\boxed{}`

**Gate**：
- `results/sc_v6_k32_public.jsonl` 存在，1126 行
- `data/sft_train.jsonl` 存在，**至少 500 examples**（少于 500 说明 K=32 也没救，得用 NuminaMath 兜底）
- `pytest` 全过

### Day 2 —— LoRA 训练管道（先 dry-run，不冲全量）

**目标**：`lora_train.py` 跑通，100 examples × 1 epoch 在 30 分钟内训完不崩。

1. 装训练 extra：`uv sync --extra dev --extra cu128 --extra train` （pulls peft / trl / datasets / wandb）
2. 写 `src/cse151b_comp/lora_train.py`：
   - 加载 `Qwen3-4B-Thinking-2507` BF16（**不**走 BitsAndBytes INT4，量化误差进梯度会污染训练）
   - LoRA 配置：`r=32, alpha=64, lora_dropout=0.05`，target_modules 用 transformers 的 default for Qwen3（`q_proj, k_proj, v_proj, o_proj`，可加 `gate_proj/up_proj/down_proj` 看显存）
   - SFTTrainer (TRL) 或裸 transformers Trainer。target 用 chat template 格式化，`{system, user}` 部分 mask loss，只训 assistant 段
   - bsz=4, accum=2, lr=2e-4, **不开 gradient_checkpointing**（96GB 够，开了反而慢）
   - wandb 关掉默认上报（`os.environ["WANDB_DISABLED"] = "true"`），需要时再开
3. CLI：`cse151b-train --data data/sft_train.jsonl --output checkpoints/lora_dryrun --epochs 1 --limit 100`
4. dry-run 100 examples / 1 epoch → 检查 loss 下降、显存占用、保存 adapter
5. 在 `pyproject.toml` 注册 `cse151b-train = "cse151b_comp.lora_train:main"`
6. 加 `tests/test_lora_train_smoke.py`（仅模块级 import 测试，不真训）

**Gate**：dry-run 100 examples 训完；adapter 文件 `adapter_model.safetensors` 在 `checkpoints/lora_dryrun/`；dry-run loss 单调下降。

### Day 3 —— LoRA 全量训练 + 本地 val 评估

**目标**：拿到第一份完整 LoRA adapter，知道在 val_225 上是否涨。

1. 全量训：
   ```bash
   uv run cse151b-train --data data/sft_train.jsonl \
       --output checkpoints/lora_v1 --epochs 3 --r 32 --lr 2e-4
   ```
   预估：500–700 examples × 3 epoch × bsz=8(eff) → 200–500 step → **2–4 小时**
2. 评估 LoRA 单 shot on val_225：
   ```bash
   uv run cse151b-infer --data data/public.jsonl --val-only \
       --adapter checkpoints/lora_v1 \
       --out results/lora_v1_val.jsonl
   uv run cse151b-evaluate results/lora_v1_val.jsonl
   ```
   （这里需要 `cse151b-infer` 支持 `--adapter`；如果没实现，先在 `inference.py` 加 4 行 PEFT 加载）
3. 评估 LoRA + SC K=8 on val_225（自蒸馏的真正考察）：
   ```bash
   uv run cse151b-sc --input data/public.jsonl --val-only \
       --adapter checkpoints/lora_v1 \
       --output results/lora_v1_sc8_val.jsonl --k 8
   uv run cse151b-evaluate results/lora_v1_sc8_val.jsonl
   ```
4. 对比 baseline_v0_val.json（Phase 0）和 sc_v6_k8 val（如果另一台已经跑过）

**Gate（本机视角）**：
- LoRA 单 shot val ≥ Phase 0 + 2pp → Day 4 调超参
- LoRA + SC val ≥ v6 + SC val + 3pp → Day 5 直接 ship adapter
- 退步 → Day 4 排查（数据污染？mask loss 错？epoch 太多过拟合？）

### Day 4 —— 超参扫描（条件做）

只有 Day 3 LoRA 涨了才扫；如果退步，Day 4 用来排错重训而不是堆超参。

扫描矩阵（每个 ~3 小时，挑 2–3 个跑）：
- `r ∈ {16, 32, 64}` × `lr ∈ {1e-4, 2e-4}`
- epoch ∈ {2, 3, 5}（早停看 eval loss）
- target_modules：q/k/v/o vs full attn+mlp

每个 config 训完跑 LoRA + SC K=8 val。挑 val 最高的当 `lora_v_final`。

**Gate**：选定 `checkpoints/lora_final/`，比 Day 3 涨 ≥ 1pp，否则用 Day 3 版当 final。

### Day 5 —— 大 K 推理 / final adapter handoff

**目标**：把最终 adapter + 必要的辅助产物给到另一台机器去做 Kaggle submission。

1. 用 `lora_final` 跑 **K=16 SC on val_225** 最后确认数字稳定（两次跑差异 ≤ 1pp）
2. 如果另一台机器不能放 K=16 BF16 + LoRA，本机再跑 **K=16 SC on private** 一次：
   ```bash
   uv run cse151b-sc --input data/private.jsonl \
       --adapter checkpoints/lora_final \
       --output results/lora_final_sc16_private.jsonl \
       --k 16 --temperature 0.7
   ```
   产出包含完整 response，另一台机器只需跑 `cse151b-submit` 转 CSV
3. **handoff 包**：
   ```
   handoff/
   ├── checkpoints/lora_final/    (adapter, ~100MB)
   ├── results/lora_final_sc16_private.jsonl  (如果本机产出)
   ├── results/lora_final_sc16_val.jsonl
   └── HANDOFF.md  (告诉另一台：怎么加载 adapter、跑哪个 CLI、注意事项)
   ```
4. tar + scp 到另一台 / 上传共享盘

**Gate**：handoff 包完整；另一台机器能直接拉起 LoRA + SC 推理；val 数字与本机一致（验证 adapter 没在传输中坏）。

### Day 6 —— buffer + 收尾

1. 本机不再做新实验。等另一台机器汇报最终 leaderboard 分
2. 如果另一台机器请求第二轮 LoRA（比如 leaderboard 没涨想试 r=64），本机响应 1 次重训请求
3. 把 `checkpoints/` 里失败实验删掉，只保留 `lora_final/` 和 dry-run；外部备份
4. 关 GPU 任务，准备退还机器

## 端到端验证清单（每天结束跑）

```bash
unset VIRTUAL_ENV CONDA_PREFIX
uv run pytest && uv run pre-commit run --all-files

# 当日产出
ls -la results/$(date +%Y%m%d)*.jsonl checkpoints/

# val 数字（从 Day 3 起）
uv run cse151b-evaluate results/lora_*_val.jsonl

# 显存监控
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 风险 + 应急

| 风险 | 触发条件 | 应急 |
|---|---|---|
| Day 1 K=32 推理超 4 小时 | vLLM 重批延迟过高 | 降 K=16，或切到 BF16（不再 INT4），KV cache 反而干净 |
| `sft_train.jsonl < 500` | pass@32 太低 | 加 NuminaMath 决方案：用 prepare_sft_data 加一段从 NuminaMath 抽题 + decontaminate（与 public/private 比对）；这会多花半天 |
| LoRA 训不下降 | loss 平 / 涨 | 检查 chat template 对不对、mask 范围对不对（用户 token 不该被 train）、lr 太大 |
| LoRA val 退步 | overfit / 数据污染 | 减 epoch；提高 dropout；确认 val_225 真的从 sft_train 排除了 |
| 96GB OOM | bsz=4 + r=64 + full module | 降 bsz=2 或 r=32 |
| 另一台机器加载 adapter 失败 | 框架版本不一致 | handoff 时附带 `requirements-handoff.txt` 锁版本 |

## 时间余量

| Day | 任务 | 预算 | 弹性 |
|---|---|---|---|
| 1 | K=32 SC + sft_train.jsonl 准备 | 0.5 | +0.5（GPU 挂机时间） |
| 2 | lora_train.py + dry-run | 1 | +0.25 |
| 3 | 全量 LoRA + val 评估 | 1 | +0.5 |
| 4 | 超参扫描 / 排错 | 1 | +0.5 |
| 5 | 大 K 推理 + handoff 包 | 1 | +0.5 |
| 6 | buffer + 第二轮重训 | 1 | +1 |

**总预算**：5.5 工作 + 0.5 buffer + 1 重训应急。

## 起步命令（Day 1 立刻可跑）

```bash
# 0. 每次新 shell
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV CONDA_PREFIX

# 1. Day 1 子分支
git checkout jason/dev2 && git checkout -b day1-distill-pool

# 2. 装 train extra（也顺便准备 Day 2）
uv sync --extra dev --extra cu128 --extra train

# 3. K=32 SC on public（前台跑，挂机 1.5–3 小时）—— 这里需要确认 cse151b-sc 已经支持
#    96GB 该用得上的几个参数（--gpu-mem-util 0.92 等），不行就先看 self_consistency.py CLI
uv run cse151b-sc --input data/public.jsonl \
    --output results/sc_v6_k32_public.jsonl \
    --k 32 --temperature 0.7 --top-p 0.95

# 4. 同时另一 shell：草稿 prepare_sft_data.py（详见 Day 1 第 4 步）
```
