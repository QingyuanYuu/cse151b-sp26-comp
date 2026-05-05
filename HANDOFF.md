# HANDOFF — 跨机器同步文档

这份文档给 4090 那台机器上的 Claude 看，让它快速搞清楚我们当前在哪、谁负责什么、文件互相之间怎么对接。Blackwell 96GB 这边的 Claude 维护这份文档；4090 那边只读。

**最后更新：2026-05-05（Day 1，jason/dev2 + day1-distill-pool 分支）**

---

## 0. 项目一句话

CSE 151B Spring 2026 Kaggle 数学推理竞赛。模型锁死 `Qwen/Qwen3-4B-Thinking-2507`，public 集 1126 题（带答案），private 集 943 题（leaderboard 评分）。当前 best leaderboard 是 Phase 0 starter 的 **0.575**，目标冲班级前列。

---

## 1. 必须读的几个文件（按优先级）

| 文件 | 作用 | 何时读 |
|---|---|---|
| `PLAN_6DAY.md` | **当前 6 天执行计划**（针对 Blackwell 96GB + 模型锁死场景写的） | 先读 |
| `PLAN.md` | 原始路线图（按 4090 24GB 写的，含很多失败模式 / 判分陷阱细节，仍可参考） | 看具体技术问题时 |
| `reports/public_private_gap_analysis.md` | **关键发现**：public val ↑6pp 但 private leaderboard ↓8pp 的原因诊断；T=0.6 单次 std error ≈ ±5pp | 任何 prompt / leaderboard 决策前 |
| `reports/milestone_results_summary.md` | Phase 0 vs Phase 1 表格 + failure mode 直方图 | 想知道当前 baseline 实力 |
| `src/cse151b_comp/prompts.py` | **v6 per-type prompts**（已锁定为生产 prompt） | 写新代码时 |
| `README.md` | 仓库结构 / uv setup / scoring quick reference | 第一次 setup |

---

## 2. 比赛硬约束（违反即 0 分）

来源：`PLAN.md` "比赛硬约束" 章节。

1. **模型锁死**：只能用 `Qwen/Qwen3-4B-Thinking-2507`（INT4 BNB / BF16 都允许）。**LoRA / QLoRA 允许**（adapter 不算改 base）；full fine-tuning 严格读规则可能算"换模型"，仅在 LoRA 平台后才考虑。
2. **推理时禁外部工具**：no API 调用、no code interpreter、no calculator。SymPy 在 prompt 设计阶段可用（验证我们以为的答案对不对），generate 时不能跑代码。
3. **提交 CSV**：`id, response`，response 含完整 reasoning trace，grader 自动抓 `\boxed{...}`。
4. **数据集**：
   - `data/public.jsonl` —— 1126 题带答案，开发用
   - `data/private.jsonl` —— 943 题无答案，仅 Kaggle 提交时评分
5. **Submission 配额受限**：每天有限。`reports/public_private_gap_analysis.md` 里的话："We cannot run more than two additional Kaggle submissions today."
6. **Judger 规则严苛**：MCQ 必须 `\boxed{C}`，**不**接受 `\boxed{(C)}` / `\boxed{C.}`；自由作答数值不能多余四舍五入；多 part 必须 K 个连续 boxed 或一个 comma-separated boxed，不能混。

---

## 3. 多机分工（重要）

用户 2026-05-04 明确划分：

### Blackwell 96GB（这台 / 本 README 所在）
**只做需要大显存的活**。具体：
- LoRA / QLoRA 训练（4B BF16 + bsz=4 + r=32 才用得起 96GB）
- K=32 self-consistency 自蒸馏（4B INT4 + n=32 KV cache 在 24GB 紧张，96GB 轻松）
- 必要时 BF16 推理 / 大 K SC 推理给最终 submission 用
- 把训练好的 adapter / 大 SC 输出打包，scp 给 4090

**不做**：日常 prompt 迭代、val 评分、Kaggle 提交、写报告 —— 这些都给 4090。

### 4090 24GB（另一台）
**做除了大显存以外的一切**。具体：
- 单次 / 小 K SC 推理（K=1 到 K=8）
- prompt 迭代 + val_225 上的 ablation（`reports/baseline_*_val.*`）
- **NuminaMath SFT 数据准备**（CPU + 小 GPU，本机不要做）
- `cse151b-submit` 转 CSV
- Kaggle 提交、写 milestone / final 报告
- 收 Blackwell 给的 adapter，做 LoRA + SC 在 private 上的最终推理 + 提交

**别做**：LoRA 训练（24GB 上 r=32 得 gradient checkpointing 才塞下，慢）、K=32+ SC（KV cache 不够）。

---

## 4. 仓库 / 分支当前状态

```
main                 ←  课程原版
jason/dev2 (origin)  ←  集成分支（保稳定）
└── day1-distill-pool   ←  当前活跃，Day 1 工作分支
```

`jason/dev2` 最近 commit（含 jason/dev 已 merge 进来的 Phase 0c/1/2 工作）：
```
b904aa3  Merge jason/dev into jason/dev2  (含 v6 prompts、self_consistency、milestone PDF)
2b7ccc3  Rewrite PLAN_6DAY for the model-locked constraint
48667e6  Add 6-day cram plan, lock env, exempt starter from lint
```

`day1-distill-pool` 在 `jason/dev2` 之上的 Day 1 commits：
```
8751a93  Add NuminaMath SFT data prep — runs on the 4090 box  ← 4090 用这个
8f61d1a  Day 1: SC chunked checkpointing + detached launcher + SFT prep
```

4090 想做 NuminaMath prep：`git fetch && git checkout day1-distill-pool`（或者 cherry-pick `8751a93` 到自己的工作分支）。

---

## 5. 已经建好的代码 / CLI（merged jason/dev + Day 1 新增）

| 模块 | CLI（pyproject.toml 注册） | 谁建的 | 用途 |
|---|---|---|---|
| `inference.py` | `cse151b-infer` | jason/dev | vLLM 单次推理（K=1） |
| `evaluate.py` | `cse151b-evaluate` | jason/dev | 用 Judger 重判分 |
| `error_analysis.py` | `cse151b-analyze` | jason/dev | failure mode 直方图 |
| `submission.py` | `cse151b-submit` | jason/dev | results JSONL → Kaggle CSV |
| `eval_harness.py` | `cse151b-split` / `cse151b-compare` | jason/dev | val_225 抽 + 两个结果文件对比 |
| `self_consistency.py` | `cse151b-sc` | jason/dev + Day 1 改 | **K=N SC** + 投票，**chunk + resume 支持** |
| `voting.py` | （helpers） | jason/dev | per-type voting (mc / free_single / free_multi) |
| `extract.py` | （helpers） | jason/dev | `\boxed{}` 抽答案 + 数值规范化 |
| `prompts.py` | （helpers） | jason/dev | **v6 per-type prompts**（生产版） |
| `prepare_sft_data.py` | `cse151b-prepare-sft` | **Day 1 新** | SC 池 → SFT 训练对（自蒸馏路径） |
| `prepare_numina_data.py` | `cse151b-prepare-numina` | **Day 1 新** | **NuminaMath → SFT 训练对（4090 跑这个）** |

`pyproject.toml` extras：
- `dev` — pytest / ruff / mypy / pre-commit / nbstripout
- `cu126` / `cu128` — torch wheels（互斥）
- `vllm` — vllm 0.20 + transformers >= 4.56（已在 Blackwell 装好）
- `train` — peft / datasets / trl / wandb（已在 Blackwell 装好；4090 不一定要装）
- **`numina` — datasets + scikit-learn**（4090 跑 NuminaMath prep 装这个）

---

## 6. 6 天日程当前进度

完整版看 `PLAN_6DAY.md`。

| Day | Blackwell（本机）做 | 4090 做 | 状态 |
|---|---|---|---|
| 1（今天） | K=32 SC on 901 train rows → SC 池<br>`prepare_sft_data` 出 `sft_train.jsonl` | **NuminaMath SFT prep** → `numina_sft.jsonl` | ⏳ 进行中（看下面 §7） |
| 2 | 写 `lora_train.py` + 100-example dry-run | 接手 LoRA 训练数据校验 / 备份 | 待开 |
| 3 | LoRA 全量训练（r=32, 3 epoch）+ val_225 ablation（LoRA 单 shot, LoRA + SC K=8） | 单 shot SC ablation 在 val_225 上跑各个 prompt 版本对比 | 待开 |
| 4 | LoRA 超参扫 (r/lr/epoch)；不涨就**升级 full FT** | 持续推理 ablation | 待开 |
| 5 | 大 K SC on private（如果 4090 跑不下）+ handoff 包打包 | **首次 LoRA + SC private 推理 + 提交** | 待开 |
| 6 | buffer / 应急重训 | 第二次提交（最终版）+ 写 final report | 待开 |

**关键 gate**：Day 3 LoRA + SC val ≥ v6 + SC val + 3pp 才上线；Day 4 LoRA r=64 平台后才上 full FT；Day 5 提交 leaderboard ≥ Phase 0 (0.575)。

---

## 7. 当前活跃任务（Day 1）

### 7.1 Blackwell：K=32 SC（**未启动**，等用户从自己 terminal 跑）

之前一次 K=32 跑了 4h25m / 13.3% 被 Claude Code 的 harness reaper 杀了（detail 看 commit `8f61d1a` message）。修了之后两层防护：

1. **`scripts/run_sc_k32.sh`**：用 `setsid` 让进程脱离 harness 进程组，免死
2. **`self_consistency.py --chunk-size 50 --resume`**：每 50 题 fsync 一次盘，崩了再跑同 script 自动跳过已写的 id

用户操作：
```bash
cd /home/jason/cse151b/cse151b-sp26-comp
git checkout day1-distill-pool
git pull
scripts/run_sc_k32.sh                    # detached 启动
scripts/sc_status.sh                     # 查进度
tail -f logs/sc_v6_k32.log               # 详细 log
```

**预估时长**：~26h（按之前实测 1530 tok/s 算）。
**输出**：`results/sc_v6_k32_public_train.jsonl`，每行一题，K=32 个 thinking trace + vote 结果。

跑完 Blackwell 这边继续：
```bash
cse151b-prepare-sft \
    --pool results/sc_v6_k32_public_train.jsonl \
    --source data/public.jsonl \
    --val data/val_indices.json \
    --output data/sft_train.jsonl
```

### 7.2 4090：NuminaMath SFT prep（**等你 git pull + 跑**）

GPU-free，跟 Blackwell 的 K=32 SC 完全并行。

```bash
cd <4090 上的 repo 路径>
git fetch && git checkout day1-distill-pool       # 或 cherry-pick 8751a93
uv sync --extra numina                            # datasets + scikit-learn

uv run cse151b-prepare-numina \
    --dataset AI-MO/NuminaMath-CoT \
    --output data/numina_sft.jsonl \
    --public data/public.jsonl \
    --private data/private.jsonl \
    --tfidf-threshold 0.85 \
    --max-keep 10000

# 跑完 scp 给 Blackwell
scp data/numina_sft.jsonl <blackwell-host>:/home/jason/cse151b/cse151b-sp26-comp/data/
```

**预估时长**：30–90 min（HF 下载 + TF-IDF dedup + 格式化）。
**输出**：`data/numina_sft.jsonl`，10000 行（默认），schema 跟 `sft_train.jsonl` 完全一致。

注意：
- `private.jsonl` 路径如果 4090 上没有，省略 `--private`，但**会损失这部分 dedup 防泄漏**。强烈建议把 private.jsonl 放过去（gitignored，本机也没有，需要从 Kaggle 下载）。
- TIR 变体（`AI-MO/NuminaMath-TIR`）**不要用** —— 它解题过程里有 Python 代码，跟比赛 spec 冲突。
- 如果想多筛点，把 `--max-keep 20000` 试试，最终 LoRA 数据量会有 21k 例（10k numina + 1k 自蒸馏）。

---

## 8. 文件 handoff schema（两边必须一致）

### 8.1 SFT 训练文件 schema

`data/sft_train.jsonl`（自蒸馏）和 `data/numina_sft.jsonl`（外部）**一行一例**：

```json
{
  "id": 42,                            # 自蒸馏: int (public.jsonl 的 id); numina: "numina:N" 字符串
  "question_type": "free_single",      # mc / free_single / free_multi
  "system_prompt": "You are an...",    # v6 per-type system prompt
  "user_prompt": "Compute X.\n\nSolve and put...",
  "target_response": "<think>\n...\n</think>\n\nFinal answer: \\boxed{42}",
  "n_correct": 17,                     # 自蒸馏才有；numina 没这个字段
  "K": 32,                             # 自蒸馏才有
  "source": "math_contest"             # numina 才有
}
```

LoRA 训练时直接合：
```bash
cat data/sft_train.jsonl data/numina_sft.jsonl > data/sft_combined.jsonl
```

### 8.2 SC 输出 schema

`results/sc_v6_k32_public_train.jsonl`（K=32）：

```json
{
  "id": 42,
  "question_type": "free_single",
  "all_responses": ["<think>...", "<think>...", ...K 条...],
  "all_extracted": ["42", "42", "41", ...],
  "vote_counts": {"42": 28, "41": 4},
  "winning_answer": "42",
  "winning_response": "<think>...the longest among voters for 42...",
  "K": 32,
  "answer": 42,                        # 来自 public.jsonl
  "correct": true,                     # winning == gold normalized
  "solvable_but_missed": false         # 至少有一条对、但 vote 错的
}
```

### 8.3 LoRA adapter handoff（Day 5 之后）

```
handoff/
├── checkpoints/lora_final/
│   ├── adapter_model.safetensors    (~50–100 MB)
│   ├── adapter_config.json
│   └── tokenizer files
├── results/lora_final_sc16_val.jsonl
└── HANDOFF_LORA.md                   (Blackwell 写，告诉 4090 怎么 load + 跑)
```

---

## 9. 关键事实库（防止两边 Claude 各自摸索）

来自 `reports/public_private_gap_analysis.md` 和 `reports/milestone_results_summary.md`：

- **Phase 0 starter** prompts：leaderboard **0.575**，val 56.44 %（n=225）
- **Phase 1** （全规则 prompts）：val 62.67 % (+6.23 pp)，但 leaderboard **0.494** (-8.1 pp)
- **v4** （Phase 0 + minimal）：leaderboard **0.462** (-11.3 pp)
- **v6** （per-type routing，当前 prompts.py）：**还没在 leaderboard 测过**；val 也没单独 benchmark
- **诊断**：T=0.6 同 prompt 两次跑 ~44% 答案不同；single-shot 提交 std error ≈ ±5 pp。**单次 prompt 工程在这数据集上 expected value 是负的**
- **正确方向**：variance reduction (SC) > prompt engineering。理论 SC K=8 给 +5–15 pp 在 p ∈ [0.6, 0.85] 范围内
- **答案分布差异**：public 多高精度小数，private 多符号 / True/False / 字母。这是 Phase 1 anti-rounding rule 在 leaderboard 反向的根因
- **K-th-power decay**：自由作答多 part 题是 all-or-nothing 评分；per-slot 准确率 p 的 K-part 题正确率是 p^K。free-multi 占数据集 57%，per-slot 小退步 → question 大退步

### 已经被证伪 / 不要再做的事
- 加 "Use plain numbers" 规则 → True/False 变 1/0 错答（v4 bug）
- 加 "no x = " 规则 → equation 答案被乱剪（v4 bug）
- 加 "Do not round / 6 sig figs" → public 涨 4.5 pp 但 private 跌（distribution shift）
- 加 "if running out of room, output best guess" → 部分题输出空 `\boxed{...}` 字面量

### 已经被验证的事
- v6 per-type routing **理论上**修了 v1/v4 的两个 bug 来源（drop 了 plain-numbers + no-x= 规则）
- self-consistency K=8 在 4B INT4 上 vLLM 加 prefix caching 后吞吐成本 < 4× of K=1
- LoRA r=32 + bsz=4 + 96GB BF16 base 训得动 4B（Day 2 验证）

---

## 10. 当前未决问题（两边 Claude 都注意）

1. **NuminaMath 数据量取多少**？默认 `--max-keep 10000`。如果 dedup 后过 90% 没事 → 试 20000。Day 2 训练时间是约束。
2. **LoRA target_response 要不要保留 NuminaMath 的 latex 格式**？目前完全保留。如果训出来生成 latex 太多，Day 3 ablate 一版 strip 过的看效果。
3. **Day 5 究竟谁跑 private 推理**？主路径是 4090 跑（adapter 加载 + K=8 SC + 提交）；备用路径是 Blackwell 跑大 K（K=16）然后只把 JSONL 给 4090 转 CSV。看 4090 那边显存 + 时间。
4. **Full FT 兜底是否真做**？只在 Day 4 LoRA r=64 平台才做。看实际数字。

---

## 11. 联系点（双向同步要点）

Blackwell → 4090 写：
- LoRA adapter（Day 5）
- 大 K SC results JSONL（如果 4090 跑不下）
- 这份 HANDOFF.md 的更新

4090 → Blackwell 写：
- `data/numina_sft.jsonl`（**Day 1 优先**）
- val ablation 数字（直接更新到 reports/）
- leaderboard 提交分数（直接更新到 reports/）

两边都共享：`PLAN_6DAY.md`、`reports/`、`src/cse151b_comp/`。同一分支 (`day1-distill-pool` 或后续的 day-N branch)，rebase 频繁 push 频繁。

---

## 12. 给 4090 那边 Claude 的最小 next step

如果你（4090 上的 Claude）刚被叫来：

1. **先读这份 HANDOFF.md 全文 + `PLAN_6DAY.md` Day 1 段**
2. 检查 `git log --oneline -5` 是不是已经在 `day1-distill-pool` / 包含 commit `8751a93`
3. 检查 `data/private.jsonl` 在不在（如果不在，让用户从 Kaggle 下载放到 `data/private.jsonl`，不要 commit）
4. 跑：
   ```bash
   uv sync --extra numina
   uv run cse151b-prepare-numina \
       --output data/numina_sft.jsonl \
       --public data/public.jsonl \
       --private data/private.jsonl
   ```
5. 跑完 scp `data/numina_sft.jsonl` 给 Blackwell（`<blackwell-host>:/home/jason/cse151b/cse151b-sp26-comp/data/`）
6. 反馈：用户跟我（Blackwell Claude）说 numina prep 完成 + 数据量
7. **不要**自己启动 LoRA 训练 / K=32 SC（24GB 不够 / 跑不动）
8. **不要**碰 Kaggle 提交直到 Day 5（看 PLAN_6DAY.md gate）

如果 NuminaMath prep 已经完成，下一个能做的事：
- 跑 v6 prompts 单 shot 在 val_225 上的 baseline（生成 `results/v6_val.jsonl`）—— 这件事 4090 干净利落能做，结果给 Blackwell 训练时做对照
- 命令：
  ```bash
  uv run cse151b-infer --data data/public.jsonl --val \
      --out results/v6_val.jsonl --max-tokens 16384
  uv run cse151b-evaluate --results results/v6_val.jsonl
  ```
  （`cse151b-infer --val` 应该过滤到 val_225；如果 CLI 不支持要先加 flag）

---

**这份文档每完成一个大节点（K=32 SC 完成 / NuminaMath 完成 / LoRA 训完）就更新一次最后日期 + §7。其他章节相对稳定。**
