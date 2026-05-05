# CSE 151B 6 天冲刺计划 —— Qwen3-4B-Thinking 锁定 + Blackwell 96GB

## 这份计划的定位

`PLAN.md` 是原始的、按 RTX 4090 24GB 写的完整路线图。本计划 (`PLAN_6DAY.md`) 在三个新约束下重新排程：
1. **6 天硬截止**（含今天）
2. **机器是 Blackwell 96GB 单卡**（不是原 plan 假设的 4090 24GB）
3. **🚫 模型锁死 = `Qwen/Qwen3-4B-Thinking-2507`**，课程规则禁止换模型 —— 不准升 14B / 32B / DeepSeek-R1 / QwQ 等任何替换

**冲突时以本文件为准**；`PLAN.md` 在 prompt 改法、判分陷阱、QLoRA target 格式上仍可参考。

## 硬件 & 截止

- **使用权**：6 天
- **GPU**：RTX PRO 6000 Blackwell 96 GB，sm_120，driver 580 / CUDA 13.0
- **环境**：uv-managed `.venv`（torch 2.11.0+cu130 + transformers 4.57.6 + vllm 0.20.0），已通过烟测
- **conda 坑**：每次开 shell 先 `unset VIRTUAL_ENV CONDA_PREFIX`
- **数据**：1126 题（375 MCQ + 751 自由作答），`data/public.jsonl` 已就位

## 关键策略：4B 锁死下，杠杆全在"测试时计算量 + 训练"

模型不能换 → 唯一能涨分的方向有四个，按杠杆从大到小：

| 杠杆 | 4090 24GB 限制下原 PLAN 的做法 | Blackwell 96GB + 4B 锁定下的做法 |
|---|---|---|
| **测试时投票 n** | n=5（受显存 + 时间限制） | **n=16–32**（4B INT4 才 ~3GB，96GB 可同时跑 32+ 序列；这是最大单笔免费收益） |
| **prompt + 格式** | Phase 1 一次性改 | 一样，但更激进的 few-shot + reflection chain |
| **QLoRA / LoRA 微调** | Phase 4 视情况做 | **Day 5 默认做**（这是除测试时计算外唯一的精度杠杆，4B 训练成本低） |
| **主题路由 + extractor 强化** | Phase 3a | 一样 |

96GB 在 4B 锁定下的真正红利：**同时跑大量并发序列**。`max_num_seqs=256+`、`gpu_memory_utilization=0.9`，n=32 投票一次喂进 vLLM 比 n=5 跑 6 次还快。

## 6 天日程

每天结尾必做：(a) `results/` 下当日 JSONL 存档；(b) `analyze.py` 输出粘到 `notes/dayN.md` 或 PR 描述。

### Day 1（今天，剩约半天）—— 4B INT4 baseline 起跑

**目标**：拿到 Qwen3-4B-Thinking-2507（与 starter 同款）在全 1126 题上的 baseline 分。

1. 起子分支：`git checkout -b day1-baseline-4b`（从 `jason/dev2`）
2. notebook（cell 7）保持 starter 的模型，但放开 vLLM 配置吃满 96GB 红利：
   - `MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"`（**不动**）
   - `MAX_TOKENS = 16384`（thinking trace 偶尔 > 12k，给点余量）
   - `OUTPUT_PATH = "results/baseline_4b.jsonl"`
   - vLLM：`gpu_memory_utilization=0.90`、`max_num_seqs=256`、`max_num_batched_tokens=32768`、`max_model_len=16384`
   - 保留 `os.environ["VLLM_HOST_IP"] = "127.0.0.1"`
3. 先跑 **20 题烟测** → 确认单题平均 < 4 秒（4B INT4 在 Blackwell 上应该非常快）
4. 烟测过 → 全量。预计 < 1 小时（vllm batched，4B 小，显存巨大）
5. 同时另一 shell：实现 `dev_split.py`（200 题分层抽样，固定 seed=151，按 题型 × 长度桶 × 主题正则 分层）
6. 全量结束 → 用 (TODO Day 2 实现的) `cse151b-analyze` 打印分布

**Gate**：4B baseline 总分 ≥ **55 %**（原 PLAN 预测 50–62 %；如果烟测时已发现 prompt 通用性 OK，应能达 60+）。低于 55 % → 检查 vLLM 配置（max_num_seqs 太大可能静默降级 KV cache）、extractor 与 starter prompt 的兼容性。

### Day 2 —— CLI + analyze + dev split

**目标**：把 notebook 逻辑全部抽到 `src/cse151b_comp/`，pytest + pre-commit 干净。

1. `runner.py`：CLI `cse151b-run --data --out --model --max_tokens --temperature --n_samples [--dev]`，复用 notebook 的 `build_prompt()` / `extract_letter()` / `Judger`
2. `analyze.py`：从 results JSONL 重判分。打印：
   - 总体 / MCQ / 自由作答 准确率
   - MCQ 按 `len(options)` 分层；自由作答按 `len(gold_list)` 分层
   - 长度桶：`<150 / 150–500 / 500–1500 / >1500` 字符
   - 主题桶（正则）：calculus / linalg / probability / ODE / complex / combinatorics / other
   - 失败模式：(a) extractor 抽不到 (b) part 数量错 (c) 抽到了但答错 (d) 撞 `MAX_TOKENS`
3. `dev_split.py`：固化 Day 1 用过的 seed，输出 `data/dev_200.jsonl` 和 `data/dev_holdout.jsonl`（剩下 90 % 再分 90/10，10 % 那部分是 holdout，**Day 5 之前禁碰**）
4. `tests/test_runner_extract.py` 和 `tests/test_analyze_buckets.py` 覆盖边界
5. 用新 `analyze.py` 复判 Day 1 baseline，确认数字一致（防 CLI 抽取过程引入回归）

**Gate**：`pytest` 全过 + `pre-commit run --all-files` 干净；`git status --porcelain | grep dev_holdout` 无输出。

### Day 3 —— Prompt 修复 + 温度扫描

**目标**：dev_200 上 ≥ Day 1 baseline + **4 分**。

1. 重写 `prompts.py`，硬性规定（每条都写正例 + 反例）：
   - **MCQ**：`Output ONLY the letter inside \boxed{}, e.g. \boxed{C}. Do NOT write \boxed{(C)}, \boxed{C.}, \boxed{C)}, or \boxed{(C.)}.`
   - **自由作答多 part**：`If the answer has K parts, output exactly K consecutive \boxed{} blocks in order, OR a single \boxed{a, b, ..., k}. Never mix.` + 1 个 few-shot
   - **`[ANS]` 占位（66 %）**：`Replace each [ANS] with \boxed{...}. The final line must end with the boxed answer.`
   - **数值精度**：`Do not round numerical answers. Report at least 6 significant figures or the exact symbolic form. Example: 143.224229 not 143; 2.32625 not 2.33.`
2. 温度扫描，**dev_200 上**：
   - `temperature ∈ {0.0, 0.3, 0.6, 0.9}` × `top_p ∈ {0.9, 0.95}` = 8 个 config
   - 单样本，按 MCQ / 自由作答分别记胜出
   - 4B INT4 + Blackwell 上 dev_200 单 config 应 < 5 分钟，整套扫描半小时内
3. 锁定 config，跑全 1126 → `results/v1_prompt_4b.jsonl`
4. `tests/test_prompts.py` 增正负例字符串断言

**Gate**：dev ≥ baseline + **4 分**。低于：检查 extractor 有没有跟新 prompt 同步（最常见 bug 就是 prompt 改输出格式但 extractor 还按老的抽）。

### Day 4 —— 大 n 投票 + 主题路由 + SymPy 反思

**目标**：dev ≥ Day 3 + **5 分**。这是 4B 锁定下最大杠杆所在。

按杠杆排序：

1. **大 n self-consistency**（上午+下午，+3–7 分）：`voting.py`
   - 4B INT4 在 96GB 上跑 **n=32**（实在不行 n=16）：T=0.6, top_p=0.95
   - dev_200 上对比 n=8 vs n=16 vs n=32 的边际增益，找平台点
   - MCQ：字母众数
   - 自由作答单值：`Judger.is_equal` 聚类后取最大簇
   - 多 part：每 slot 独立投票
   - 全量 1126 估计 n=32 仍 < 3 小时（vllm 批量）
   - 配 `tests/test_voting.py`：tie-break、全部不同、跨等价（`1/2` vs `0.5`）、part 数量不齐
2. **主题路由**（傍晚，+1–3 分）：`topics.py` 正则识别后给 system prompt 加主题后缀
   - calculus / probability / linalg / 其他
   - 配 `tests/test_topics.py`
3. **SymPy 反思 pass**（晚上，+1–2 分）：仅对自由作答中 (a) 第一次抽不到 (b) part 数量错 (c) 撞 MAX_TOKENS 的题，跑第二轮：`Your previous answer was {extracted}. Verify by computing each part numerically and correct any errors. Output corrected \boxed{...}.`，单次反思不递归

跑全 1126 → `results/v3_full_4b.jsonl`。

**Gate**：dev ≥ Day 3 + **5 分**。这是不动模型情况下 prompt + 测试时计算的实际上限。

### Day 5 —— QLoRA on 4B（默认做，不再可选）

模型锁定下，QLoRA 是除了"加大 n"以外**唯一**剩下的精度杠杆，必须做。

**早上 9 点决策**：看 Day 4 失败直方图。
- 如果 reasoning 错 > 格式错 → **训**
- 如果格式错 > reasoning 错 → 仍训，但只用 format-correct 的 target（让模型学输出格式而不是数学）
- 如果 dev 已 ≥ 75 % 且 failure 完全散 → 仍训保险（4B 微调成本低，不训浪费一天）

**训练设置**：
1. 数据：`public.jsonl` 减去 `dev_200`，再分 90/10。10 % 存 `data/dev_holdout.jsonl`，**前 4 天 git status 没碰过**
2. Target：`<think>...reasoning...</think>\n\nFinal answer: \boxed{...}`，**带完整 thinking + boxed**（关键：不能用裸 gold，否则模型学会丢 boxed）
3. 配置（4B 在 Blackwell 上很轻松）：
   - 用 vllm 用的同款 BF16 权重（不是 BitsAndBytes INT4，避免量化误差进训练）
   - LoRA r=32, lr=2e-4, bsz=4, accum=2, **不**开 gradient_checkpointing（96GB 够），3 epoch
   - 96GB 上 batch 可以堆得很大；wallclock 估 3–5 小时
4. 评估：先在 dev_200，再在 dev_holdout
5. 上线条件：**holdout 涨 ≥ 3 分** AND dev 涨 ≥ 2 分。否则丢 LoRA 走 Day 4 best

**Gate**：holdout ≥ Day 4 holdout-benchmark + 3 分；OR LoRA 被丢弃，回 Day 4 config 提交。

### Day 6 —— 提交 + 安全网

1. **早上：锁配置**，dev_200 + dev_holdout 上各跑确认数字
2. **中午：格式校验**
   ```bash
   uv run python -c "import json; rows=[json.loads(l) for l in open('submission.jsonl')]; print('rows', len(rows)); print('ids', len(set(r['id'] for r in rows))); print('boxed', sum(1 for r in rows if '\\\\boxed{' in r['response']))"
   ```
   必须 `1126 / 1126 / 1126`
3. **下午：双提交准备**
   - (a) **best single-pass**（Day 3 v1 prompt + T=0 贪心）
   - (b) **best voted**（Day 4/5 的 n=16 或 n=32，含 LoRA 如果通过 gate）
   - 两份都本地 `judger.py` 评一遍，记预期分；leaderboard 大幅低 = 格式破了，回滚到 (a)
4. **晚上：buffer + 文档** + 备份 `results/*.jsonl` 到外部存储

**Gate**：行数 + boxed 抽检 + 本地预判分；leaderboard 与本地差距 < 5 %。

## 端到端验证清单（每天结束）

```bash
uv run pytest && uv run pre-commit run --all-files
ls -la results/<today>.jsonl
uv run cse151b-analyze results/<today>.jsonl --dev data/dev_200.jsonl
git status --porcelain | grep dev_holdout && echo "ALERT: holdout was touched"  # Day 5 前必查
nvidia-smi --query-gpu=memory.used --format=csv
```

## 风险 + 应急

| 风险 | 触发条件 | 应急 |
|---|---|---|
| 4B baseline < 50 % | Day 1 末 | 检查 vLLM 配置（max_num_seqs=256 + max_model_len=16k 可能 KV cache 静默 OOM 降级），降回 64 / 12k 重跑 |
| Day 3 prompt 改完 dev 不涨 | < +2 分 | extractor 没跟上新格式；先单测 extractor + 新 prompt 输出对得上 |
| Day 4 n=32 全量超 4 小时 | 时间预算溢出 | 降 n=16；优先做反思 pass 和主题路由（边际收益更稳） |
| Day 5 LoRA holdout 跌 | 过拟合 | 直接丢 LoRA，回 Day 4 best；Day 5 剩下时间转 n=64 极限投票（4B + 96GB 撑得住） |
| 提交格式破 | 本地 vs leaderboard > 5 % | 用 (a) single-pass 顶上；查 boxed 字符串里有没有 `(C)` / `C.` / 后空格 |
| 机器突然没了 | 任何时间 | 当下 last-good results 即提交版（永远保留可复现 config） |

## 时间余量

| Day | 任务 | 预算 | 弹性 |
|---|---|---|---|
| 1 | 4B baseline 全量（< 1h 推理）+ dev_split | 0.5 天 | +0.5（GPU 挂机） |
| 2 | CLI + analyze + tests | 1 天 | +0 |
| 3 | Prompt + 温度扫描 + 全量 | 1 天 | +0.25 |
| 4 | n=16/32 投票 + 主题 + 反思 + 全量 | 1 天 | +0.25 |
| 5 | QLoRA 训练 + 评估 | 1 天 | +0.5 |
| 6 | 提交 + buffer | 1 天 | +0.5 |

**总预算**：5.5 天工作 + 0.5 天 buffer。任何 gate 失败 → 回上一天 last-good，把当天预算让给下一环节。

## 起步命令（Day 1 立刻可跑）

```bash
# 0. 每次新 shell
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV CONDA_PREFIX

# 1. 起 Day 1 子分支（从 jason/dev2）
git checkout jason/dev2 && git checkout -b day1-baseline-4b

# 2. 改 notebook 的 vLLM 配置（cell 7），不改 MODEL_ID
#    需要改的字段：MAX_TOKENS, OUTPUT_PATH, vllm 调用参数 gpu_memory_utilization/max_num_seqs/max_num_batched_tokens/max_model_len

# 3. 烟测 + 全量挂机
uv run jupyter notebook starter_code_cse151b_comp.ipynb
```
