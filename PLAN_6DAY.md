# CSE 151B 6 天冲刺计划 —— RTX PRO 6000 Blackwell 96GB

## 这份计划的定位

`PLAN.md` 是原始的、按 RTX 4090 24GB 写的完整路线图（约 4.75 个工作日预算）。本计划 (`PLAN_6DAY.md`) 在 6 天硬截止 + Blackwell 96GB 单卡的约束下重新排程。**冲突时以本文件为准**；`PLAN.md` 只在阶段细节（如 prompt 改法、判分陷阱清单）上保留参考价值。

## 硬件 & 截止

- **机器使用权**：6 天（含今天）
- **GPU**：NVIDIA RTX PRO 6000 Blackwell Max-Q，96 GB VRAM，sm_120，driver 580 / CUDA 13.0
- **环境**：uv-managed `.venv`（torch 2.11.0+cu130 + transformers 4.57.6 + vllm 0.20.0 + bitsandbytes 0.49.2），已通过烟测
- **conda 坑**：每次开 shell 先 `unset VIRTUAL_ENV CONDA_PREFIX`，否则 uv 会把 `.venv/bin/python` 软链到 conda 解释器
- **数据**：1126 题（375 MCQ + 751 自由作答），`data/public.jsonl` 已就位

## 关键策略改动 vs PLAN.md

| 维度 | PLAN.md（4090 24GB） | 本计划（Blackwell 96GB） |
|---|---|---|
| baseline 模型 | Qwen3-4B-Thinking INT4 | **直接 Qwen3-14B-Thinking-2507 BF16**（跳过 4B） |
| `gpu_memory_utilization` | 0.70 | **0.92** |
| `max_model_len` | 12288 | **32768**（让 thinking trace 不被截 truncated） |
| `max_num_seqs` | 64 | **128**（throughput 翻倍） |
| self-consistency `n` | 5 | **8**（再大边际收益小，token 成本线性涨） |
| Phase 3c "14B 升级" | 单独 0.5 天，gate 风险中 | **取消**（Day 1 就用 14B） |
| Phase 4 QLoRA | 默认做 | **条件做**（Day 5 决定，failure 直方图说话） |

理由：14B BF16 直接吃掉 ~28 GB，留 ~65 GB 给 KV cache，比 INT4 4B 期望 +10–15 分 baseline。原 PLAN 的"先 4B 跑通再升 14B"在 96GB 上是没意义的浪费 —— pipeline 在 Phase 0.5 烟测时已经被 5 题验过了。

## 6 天日程

每天结尾必须满足两件事：(a) `results/` 下有当日产出 JSONL；(b) `analyze.py` 打印的 dev 分数 + 失败直方图存档（粘到 PR description 或单独的 `notes/dayN.md`）。

### Day 1（今天，剩约半天）—— 14B baseline 起跑

**目标**：拿到 14B BF16 在全 1126 题上的 baseline 分。

1. 起一个 git branch：`day1-baseline-14b`
2. 改 notebook（先用 notebook 跑通，不忙着抽 CLI）：
   - `MODEL_ID = "Qwen/Qwen3-14B-Thinking-2507"`
   - `MAX_TOKENS = 32768`，`gpu_memory_utilization=0.92`，`max_num_seqs=128`，`max_num_batched_tokens=32768`
   - `OUTPUT_PATH = "results/baseline_14b.jsonl"`
   - 保留 Phase 0.5 的 `os.environ["VLLM_HOST_IP"] = "127.0.0.1"`
3. 先跑 20 题烟测 → 确认生成速度（预期：14B BF16 在 Blackwell 上单题平均 < 6 秒，1126 题约 1.5–2 小时）
4. 烟测过 → 跑全量；同时打开第二个 shell 实现 `dev_split.py`（200 题分层抽样，固定 seed=151）
5. 全量结束 → `cse151b-analyze results/baseline_14b.jsonl` 打印分布

**Gate**：14B baseline 总分 ≥ **65 %**（4B INT4 是 ~55 %，14B BF16 应该天然涨 10+）。低于 65 % 说明 prompt 或解析器有 bug，不是模型不行。

### Day 2 —— CLI + analyze + dev split

**目标**：notebook 里的逻辑全部抽到 `src/cse151b_comp/` 下，`pre-commit` + `pytest` 干净。

1. `runner.py`：CLI `cse151b-run --data --out --model --max_tokens --temperature --n_samples [--dev]`，复用 notebook 的 `build_prompt()` / `extract_letter()`，调 `Judger`
2. `analyze.py`：从 results JSONL 重判分。打印：
   - 总体 / MCQ / 自由作答 准确率
   - MCQ 按 `len(options)` 分层；自由作答按 `len(gold_list)` 分层
   - 长度桶：`<150 / 150–500 / 500–1500 / >1500` 字符
   - 主题桶（正则）：calculus / linalg / probability / ODE / complex / combinatorics / other
   - 失败模式：(a) extractor 抽不到 (b) part 数量错 (c) 抽到了但答错 (d) 撞 `MAX_TOKENS`
3. `dev_split.py`：抽 `data/dev_200.jsonl`（已经 Day 1 跑过；这一步是把代码固化）
4. 把 day1 跑出来的 1126 条结果用新 `analyze.py` 复判，确认数字一致
5. 再实现 `tests/test_runner_extract.py` 和 `tests/test_analyze_buckets.py`，覆盖边界情况

**Gate**：`pytest` 全过 + `pre-commit run --all-files` 干净；analyze 输出与 Day 1 notebook 内同源数据匹配（避免抽 CLI 时引入回归）。

### Day 3 —— Prompt 修复 + 温度扫描（Phase 1 等价）

**目标**：在 dev_200 上跑赢 baseline ≥ 4 分。

1. 重写 `prompts.py`，硬性规定：
   - **MCQ**：`Output ONLY the letter inside \boxed{}, e.g. \boxed{C}. Do NOT write \boxed{(C)}, \boxed{C.}, \boxed{C)}, or \boxed{(C.)}.` 给一个正例 + 三个反例
   - **自由作答多 part**：`If the answer has K parts, output exactly K consecutive \boxed{} blocks in order, OR a single \boxed{a, b, ..., k}. Never mix.` 加 few-shot
   - **`[ANS]` 占位（66 %）**：`Replace each [ANS] with \boxed{...}. The final line must end with the boxed answer.`
   - **数值精度**（Phase 0.5 烟测发现的关键 bug）：`Do not round numerical answers. Report at least 6 significant figures or the exact symbolic form. Example: 143.224229 not 143; 2.32625 not 2.33.`
2. 温度扫描，**dev_200 上**：
   - `temperature ∈ {0.0, 0.3, 0.6}` × `top_p ∈ {0.9, 0.95}` = 6 个 config
   - 单样本，按 MCQ / 自由作答 分别记胜出 config
3. 锁定 config，跑全 1126 → `results/v1_prompt_14b.jsonl`
4. `tests/test_prompts.py` 增加正负例字符串断言（防止 prompt 被误改）

**Gate**：dev ≥ Day 1 baseline + **4 分**。低于这个数：检查 extractor 有没有跟新 prompt 同步（很容易出现 prompt 让模型输出新格式但 extractor 还按老的抽）。

### Day 4 —— self-consistency n=8 + 主题路由 + SymPy 反思（Phase 3a/b/d 合并）

**目标**：dev ≥ Day 3 + 5 分。

按"杠杆 / 工时"排序做：

1. **3b self-consistency** (上午, +2–5 分)：`voting.py` 用 vLLM `n=8`、`T=0.6`、`top_p=0.95`
   - MCQ：字母众数
   - 自由作答单值：用 `Judger.is_equal` 把等价答案聚类，取最大簇
   - 多 part：每 slot 独立投票
   - 配 `tests/test_voting.py` 覆盖：tie-break、全部不同、全部相同、跨等价（`1/2` vs `0.5`）
2. **3a 主题路由** (下午, +1–3 分)：`topics.py` 正则 + system prompt 主题后缀
   - calculus（积分/导数/Taylor/极限）→ 强调 step-by-step 化简
   - probability（条件/期望/分布）→ 强调列出全空间
   - linalg（矩阵/行列式/特征值）→ 强调维度合法性
   - 其他 → 默认 prompt
   - 配 `tests/test_topics.py`
3. **3d SymPy 反思 pass** (晚上, +1–2 分)：仅对自由作答中第一次抽答案为 None / part 数量错 / 撞 MAX_TOKENS 的题，跑第二轮：`Your previous answer was {extracted}. Verify by computing each part numerically and correct any errors. Output corrected \boxed{...}.` 单次反思，不递归

跑全 1126 → `results/v3_full_14b.jsonl`。

**Gate**：dev ≥ Day 3 + **5 分**。这是冲分的最后一个 prompt 层杠杆，做不到说明 14B 已经触顶，QLoRA 需要慎重。

### Day 5 —— 决策日：QLoRA / 32B / 加大 n（三选一）

**早上 9 点决策点**：看 Day 4 的 `analyze.py` 失败直方图：

- **A. failure 主要是"reasoning 错"（抽到了但答错，长链推理 collapse）** → 走 **A1: QLoRA**
  - 训练集：`public.jsonl` 减去 `dev_200`，再分 90/10。10% 存 `data/dev_holdout.jsonl`，**前 4 天没碰过**（git status 应能确认）
  - Target：`<think>...</think>\n\nFinal answer: \boxed{...}`，**带完整 thinking + boxed**，不用裸 gold（裸 gold 会让模型学会丢 boxed）
  - 14B BF16 直接 LoRA（不用 QLoRA，96GB 够）：r=16, lr=2e-4, bsz=2, accum=4, gradient_checkpointing=False（96GB 不需要 ckpt）, 2 epoch
  - 8 小时上下应该跑完
  - 上线条件：**held-out 涨 ≥ 3 分** AND dev 涨 ≥ 2 分。否则丢 LoRA

- **B. failure 主要是"extractor 抽不到 / part 数量错"（格式问题）** → 走 **B1: 加大 n + 投票精修**
  - 把投票从 n=8 提到 **n=16**（96 GB + 32k 上下文塞得下）
  - 多 part 题用 per-slot voting 而不是 whole-answer
  - 不做 QLoRA（reasoning 没问题，训练只会过拟合）

- **C. dev 已经 ≥ 80 % 且 failure 散在所有桶里没有明显 hot spot** → 走 **C1: 32B 试点**
  - 仅当 Day 4 结束时间还早 + 还没用满显存：试 `Qwen3-30B-A3B-Thinking-2507`（如果存在）或 `DeepSeek-R1-Distill-Qwen-32B`（INT4 ~16GB / BF16 ~64GB 都能塞）
  - 仅在 dev_200 上跑，**dev 涨 ≥ 3 分才换**，否则保留 14B
  - 这个分支风险最高（KV cache 翻倍 → 全量推理时间 4–8 小时），不要轻易选

**默认路径**：A 和 B 大部分情况都成立，按 failure 直方图自然分流。**只有 24 GB 内存预算紧的原 PLAN.md 才硬要做 QLoRA**；这台机上"加大 n"是更安全的选择，因为它不引入新参数。

**Gate**：选定方案产出 ≥ Day 4 + 3 分（在该方案目标指标上：QLoRA 看 held-out，加大 n 看 dev，32B 看 dev）。

### Day 6 —— 提交 + 安全网

1. **早上：锁配置**
   - 用 Day 4 / Day 5 选定的 config 在 dev_200 + dev_holdout 上各跑一次确认数字
   - 任何差异 > 1 分 → 排查（通常是 seed 没固定）
2. **中午：提交格式校验**
   - notebook / runner 加 `--submission` 模式，输出 schema `{id, is_mcq, response}`，不带 thinking trace
   - 一行验证：
     ```bash
     uv run python -c "import json; rows=[json.loads(l) for l in open('submission.jsonl')]; print('rows', len(rows)); print('ids', len(set(r['id'] for r in rows))); print('boxed', sum(1 for r in rows if '\\\\boxed{' in r['response']))"
     ```
     必须看到 `1126 / 1126 / 1126`
3. **下午：双提交**
   - 跑两版：(a) **best single-pass**（Day 3 v1 锁定的 prompt + T=0 贪心，确定性最高） (b) **best voted**（Day 4/5 的 n=8 或 n=16）
   - 两份都本地用 `judger.py` 评一遍，记预期分；leaderboard 大幅低于本地分数 = 格式破了，回滚到 (a)
4. **晚上：buffer + 文档**
   - 写 `notes/final_report.md`：每天的 dev 分变化曲线、最终 config、复现命令
   - 把 `results/*.jsonl` 备份到外部存储（不进 git）
   - 关 GPU 任务，准备退还机器

**Gate**：提交文件通过行数 + boxed 抽检 + 本地预判分；leaderboard 与本地预判分差距 < 5 %。

## 端到端验证清单（每天结束跑一次）

```bash
# 1. 测试 + lint 干净
uv run pytest && uv run pre-commit run --all-files

# 2. 当日 results 文件存在且行数对
ls -la results/<today>.jsonl
uv run python -c "import json; print(sum(1 for _ in open('results/<today>.jsonl')))"

# 3. dev 分数 vs 前一天
uv run cse151b-analyze results/<today>.jsonl --dev data/dev_200.jsonl

# 4. holdout 没被偷看（Day 4 之前必查）
git status --porcelain | grep dev_holdout && echo "ALERT: holdout was touched"

# 5. GPU 没漏
nvidia-smi --query-gpu=memory.used --format=csv
```

## 风险 + 应急

| 风险 | 触发条件 | 应急 |
|---|---|---|
| 14B baseline 反而比 4B INT4 低 | Day 1 baseline < 55 % | 检查 vllm config（max_num_seqs=128 + max_model_len=32k 可能 KV cache OOM 静默降级），降回 64 / 16k 重跑 |
| Day 3 prompt 改完 dev 不涨 | < +2 分 | extractor 没跟上新格式；先单独单元测试 extractor + 新 prompt 输出对得上 |
| Day 4 投票成本太高 | n=8 全量预估 > 6 小时 | 降 n=5（仍比单样本好），优先做 3a + 3d |
| Day 5 QLoRA 训了但 held-out 跌 | 过拟合 | 直接丢 LoRA，回 Day 4 best；Day 5 剩下时间转 B1 (n=16) |
| 提交格式破 | 本地分 vs leaderboard > 5 % | 用 (a) single-pass 提交版顶上；查 boxed 字符串里有没有 `(C)` / `C.` / `, ` 后空格 |
| 机器突然没了 | 任何时间 | 当下 results 即提交版（永远保留可复现的 last-good config） |

## 时间余量

| Day | 任务 | 累计预算 | 弹性 |
|---|---|---|---|
| 1 | 14B baseline 全量 | 0.5 天 | +0.5 天（GPU 挂机时间） |
| 2 | CLI + analyze + dev split | 1 天 | +0 |
| 3 | Prompt + 温度扫描 + 全量 | 1 天 | +0.25（晚上跑 1126） |
| 4 | n=8 voting + 主题 + 反思 + 全量 | 1 天 | +0.25 |
| 5 | QLoRA / 32B / n=16 | 1 天 | +0.5（决策日有早停） |
| 6 | 提交 + buffer | 1 天 | +0.75（如果前面顺，下午就能空出来） |

**总预算**：5.5 天工作 + 0.5 天 buffer。任何阶段 gate 失败 → 回到上一天的 last-good config，把当天预算让给后面的环节。

## 起步命令（Day 1 立刻可跑）

```bash
# 0. 每次新 shell 必跑
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV CONDA_PREFIX

# 1. 起 Day 1 分支
git checkout -b day1-baseline-14b

# 2. 改 notebook 里的模型 + vllm config（按上面"关键策略改动"表）
#    或直接 sed：
#    sed -i 's|Qwen3-4B-Thinking-2507|Qwen3-14B-Thinking-2507|' starter_code_cse151b_comp.ipynb
#    （但 ipynb 是 JSON，改完跑 nbstripout 一次）

# 3. 抽 dev split（边等模型加载边跑）
uv run python -m cse151b_comp.dev_split  # 等 Day 1 把这个实现了

# 4. 跑 baseline；GPU 挂着不用看
uv run jupyter notebook starter_code_cse151b_comp.ipynb
```
