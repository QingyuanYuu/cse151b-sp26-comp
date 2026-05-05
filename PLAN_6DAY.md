# CSE 151B 6 天冲刺计划 —— v6 prompts + Self-Consistency 主路线

## 这份计划的定位

`PLAN.md` 是从一开始的完整路线图（含背景知识 / 判分规则 / 失败模式分类 / Phase 4 LoRA 细节，仍可参考）。本计划 (`PLAN_6DAY.md`) 是 **2026-05-04 起 6 天冲刺**的执行版，建立在三件事之上：

- **`jason/dev` 已合入**（commit `b904aa3`）：Phase 0c 模块化 pipeline、Phase 1 prompt 重写、Phase 2 v6 per-type prompts + self-consistency + voting、8 页 milestone 报告 PDF
- **公认的关键发现**（`reports/public_private_gap_analysis.md`）：public val ↑6.23pp 和 private leaderboard ↓8.1pp 同时发生过；single-shot prompt engineering 在这个数据集上的 **expected value 是负的**，因为 T=0.6 下的采样噪声 ~±5pp 已经超过常见 prompt 改进幅度
- **比赛硬约束**：模型锁 Qwen3-4B-Thinking-2507；推理时禁外部工具（不能在 inference 跑 SymPy 代码）；提交 CSV；submission 配额受限

冲突时本文件为准。

## 现状（merge 后）

| 项 | 状态 |
|---|---|
| 模块化 pipeline (`inference.py` / `evaluate.py` / `error_analysis.py` / `eval_harness.py` / `submission.py`) | ✅ 已建 |
| 单测 (`test_extract.py` / `test_evaluate.py` / `test_voting.py` / `test_prompts.py`) | ✅ 已建 |
| Stratified val_indices.json (n=225) | ✅ 已建 |
| Phase 0 starter prompts | ✅ leaderboard **0.575** |
| Phase 1 全规则 prompts | ❌ leaderboard 0.494（-8.1pp，public-overfit） |
| v4 minimal prompts | ❌ leaderboard 0.462（-11.3pp） |
| **v6 per-type prompts** (`src/cse151b_comp/prompts.py` 当前版) | ⏳ **未在 leaderboard 上跑过**；val 也未单独 benchmark |
| **Self-consistency K-sample 投票** (`self_consistency.py` + `voting.py`) | ⏳ **未跑过全量**；CLI 已就绪 |
| LoRA / QLoRA 管道 | ❌ 未建 |
| Milestone 报告 PDF | ✅ 已交 |

## 关键策略 —— 不再做单次 prompt 实验

`reports/public_private_gap_analysis.md` 的诊断必须当作前提。它给出的核心建议：

> "**The right next step is variance reduction, not further prompt rules.** Self-consistency with K independent samples per question, voted via the per-slot or normalized-answer scheme, has a known dependence on per-slot accuracy that produces a ~5-15pp absolute improvement when p ∈ [0.6, 0.85]. Critically, the variance-reduction mechanism is independent of the public/private distribution shift."

所以本周主轴是：

1. **v6 + SC** —— v6 per-type prompts 已经避开了 v1/v4 的 regression 来源，搭配 K=8 / 16 的 voting 直接把方差打下去。**这是冲分主线**
2. **小心使用 leaderboard 配额** —— 每次提交先在 val_225 上做 ablation，只有 val 上 v6+SC ≥ Phase 0+SC 的版本才提交
3. **LoRA 仅在 SC 平台后做** —— Phase 3/4 量级，需要先看 SC 给出的 floor 才知道是否值得训

96GB Blackwell 的红利全用在 **K 可以放大到 16–32 还能跑完全 1126+943** 上，不是更大模型。

## 6 天日程（2026-05-04 → 2026-05-09）

每天结尾产出：(a) `results/` 下当日 JSONL；(b) `reports/dayN_*.md` 短小总结（dev/val 分 + failure 直方图差异 + 决策）；(c) git push 当前分支。

### Day 1（今天，剩约半天）—— v6 在 val_225 上的"本底"

**目标**：知道 v6 单 shot 在 val_225 上跟 Phase 0 的实际差距，决定下一步是直接走 SC 还是先调 v6。

1. 起子分支：`git checkout -b day1-v6-baseline`（从 `jason/dev2`）
2. 用 v6 prompts 在 val_225 上跑 single-shot：
   ```bash
   uv run cse151b-infer --data data/public.jsonl --val-only \
       --out results/v6_val.jsonl --max-tokens 16384 --temperature 0.6
   uv run cse151b-evaluate results/v6_val.jsonl
   ```
3. 用 `eval_harness compare` 对比 v6_val 和 phase0_val（已存于 `reports/baseline_v0_val.json`）：
   - 哪些题 v6 修了 / 哪些题 v6 退步了
   - 按 type / 长度 / topic 分层
4. **如果 val 总分 ≥ Phase 0 + 2pp**：v6 至少没退步，直接进 Day 2 上 SC
5. **如果 val 退步 ≥ 2pp**：v6 有 bug，今天剩下时间找退步根源（很可能是 detect_question_type 的 free_multi 判定漏题），不要急着上 SC
6. **不提交 leaderboard**（v6 single-shot 不值得占配额，反正 SC 后会重测）

**Gate**：`results/v6_val.jsonl` 存在；`reports/day1_v6_val.md` 写明每个 type 的 acc 和决策（v6 走 SC vs. v6 修 bug）。

### Day 2 —— Self-consistency K-tuning on val

**目标**：在 val_225 上扫 K，找出 K 翻倍带来的边际收益曲线，定下全量跑用 K。

1. 用 v6 prompts（除非 Day 1 决定退回 Phase 0），跑 K ∈ {4, 8, 16}：
   ```bash
   uv run cse151b-sc --input data/public.jsonl --val-only \
       --output results/sc_v6_k8_val.jsonl --k 8 --temperature 0.7
   ```
   K=4 / 8 / 16 三个文件分别叫 `sc_v6_k4_val.jsonl` / `_k8_` / `_k16_`
2. **关键诊断 metric** — `solvable_but_missed`（self_consistency 已实现）：K 个样本里至少 1 个对，但投票输了。这个数 = 投票算法本身的 headroom，独立于 base accuracy
3. 看 K=4 → 8 → 16 的边际增益：
   - 收敛快（K=8 → 16 涨 < 1pp）→ K=8 上全量
   - 仍在涨（K=8 → 16 涨 ≥ 2pp）→ 评估全量 K=16 时间预算（4B INT4 + Blackwell + n=16 全 1126 估计 < 2 小时）→ 上 K=16
   - free_multi 的 `solvable_but_missed` 异常高 → 改 voting 到 per-slot（`vote_free_multi` 已经有 fallback，确认它真的被触发）
4. 同时跑一次对照：**Phase 0 prompts + SC K=8** on val。这是 gap_analysis.md 推荐的备选，需要数据决定要不要切

**Gate**：决定 (a) prompt = v6 还是 starter (b) K = 8 还是 16；写进 `reports/day2_sc_tuning.md`；当天**不提交 leaderboard**（继续省配额）。

### Day 3 —— 全量 SC 跑 + 单次 leaderboard 提交

**目标**：拿到 v6+SC 在 public 全集 1126 + private 全集 943 上的稳定数。

1. 用 Day 2 选定的 (prompt, K) 跑全量 public：
   ```bash
   uv run cse151b-sc --input data/public.jsonl \
       --output results/sc_<config>_public.jsonl --k <K> --temperature 0.7
   uv run cse151b-evaluate results/sc_<config>_public.jsonl
   ```
   预计 1–3 小时
2. 同 config 跑 private：
   ```bash
   uv run cse151b-sc --input data/private.jsonl \
       --output results/sc_<config>_private.jsonl --k <K> --temperature 0.7
   uv run cse151b-submit results/sc_<config>_private.jsonl \
       --output submissions/sc_<config>.csv
   ```
3. **校验 submission 文件**（不验证 = 自杀）：
   ```bash
   python -c "import csv; rows=list(csv.DictReader(open('submissions/sc_<config>.csv'))); print('rows', len(rows)); assert len(rows)==943; assert all('\\\\boxed{' in r['response'] for r in rows[:50])"
   ```
4. **提交 leaderboard 1 次**（这是本周第一次也是必跑的一次，验证 SC 真的修了 v1/v4 的 regression）
5. 记录 leaderboard 分到 `reports/day3_sc_submission.md`

**Gate**：leaderboard ≥ Phase 0 (0.575)。
- ≥ 0.60 → 走 Day 4 LoRA 加码
- ≈ 0.575（在 ±5pp 噪声内）→ Day 4 改成"加大 K 到 32"或"对失败子集做更高温采样"
- < 0.55 → SC 也跌了，Day 4 退回 Phase 0 + SC 提交挽救

### Day 4 —— LoRA 训练（条件做） / K-scaling fallback

**早上看 Day 3 leaderboard 决策**：

#### 路线 A — leaderboard ≥ 0.60，LoRA 加码

LoRA 在锁定 base 上是 **唯一**剩下的"动模型"杠杆（adapter 不算改 base，符合 spec）。

1. 数据：`public.jsonl` 减去 val_indices；再分 90/10。10 % 那部分作为 `data/lora_holdout.jsonl`，**Day 5 才用**
2. Target：`<think>...reasoning...</think>\n\nFinal answer: \boxed{...}`，**带完整 thinking + boxed**（不能用裸 gold，否则模型学会丢 boxed）
3. 4B base BF16（不要从 INT4 训，量化误差进梯度）：LoRA r=32, lr=2e-4, bsz=4, accum=2, 不开 ckpt（96GB 够），**3 epoch**
4. 预估 wallclock：4B + 1000 多条 + 96GB → 3–4 小时
5. 评估：先 val_225（v6 prompts），再 lora_holdout
6. **上线条件**：lora_holdout 涨 ≥ 3pp AND val 涨 ≥ 2pp
7. 上线 = "LoRA + v6 + SC" 套件再跑一次 private → 准备 Day 5/6 提交

#### 路线 B — leaderboard ≈ 0.575，加大 K + 失败子集二次采样

1. 把 K 提到 32（4B + 96GB 撑得住）
2. 对 K=N 投票后仍然 `solvable_but_missed=False AND any_correct=False` 的题（约占 dev 的 60–80 %），用 T=1.0 + top_p=0.9 再来 16 samples（"hot tail" sampling）—— 这部分代码可能要新写：`self_consistency.py --boost-failed`
3. 同样在 val_225 上 ablate

#### 路线 C — leaderboard < 0.55，紧急回滚

1. Phase 0 prompts + SC K=16 直接重跑 private 提交（保底版）
2. 当天剩下时间排查 v6 退步

**Gate（A 路线）**：lora_holdout ≥ Day 3 holdout-benchmark + 3pp，且 val 涨 ≥ 2pp。否则丢 LoRA 走 B。

### Day 5 —— Final scaling + ablation 收尾

**目标**：选定最终 config，跑出最终 submission CSV。

1. 把 Day 3 / 4 选定的 best config 在 val_225 + lora_holdout（如果有）上各跑 **2 次**确认数字稳定（差异 ≤ 1pp 才算稳）
2. 最终 config 在 private 上跑一次（如果跟 Day 3 是同 config 不重跑）
3. **第二次 leaderboard 提交**（最终版）—— 提交前再次确认：
   - row count = 943
   - id 唯一
   - 每条 response 含 `\boxed{`
4. 记下 leaderboard 分作为最终成绩

**Gate**：第二次 leaderboard ≥ Day 3 leaderboard。退步 = Day 6 用 Day 3 版当最终。

### Day 6 —— 提交收尾 + buffer + 文档

1. **早上**：根据 Day 3 + Day 5 两次 leaderboard 分，选高的那次作为 **final submission**
2. **中午**：双备份 —— `submissions/final.csv` + 外部存储（Drive / S3）
3. **下午**：写 `reports/final_writeup.md`：
   - 时间线 + 每天 leaderboard 分
   - 最终 config（prompt 版本、K、T、top_p、是否 LoRA）
   - 复现命令（从 fresh checkout 一行能跑出 final.csv 的命令）
   - 学到了什么、什么没用（v1/v4 的 regression、anti-rounding 的反例）
4. **晚上**：buffer。机器还要给到第 6 天结束 → 不做新实验，只保护已有成果。退还机器前确认 GPU 任务全部清掉

**Gate**：final.csv 存在 + 通过 row-count + boxed 抽检；leaderboard 仍可见提交记录；`reports/final_writeup.md` 写完。

## 端到端验证清单（每天结束）

```bash
unset VIRTUAL_ENV CONDA_PREFIX
uv run pytest && uv run pre-commit run --all-files
ls -la results/$(date +%Y%m%d)*.jsonl

# 当日 val 数字 vs 前一天
uv run cse151b-evaluate results/<today>.jsonl

# private 提交文件（仅 Day 3, 5）
test -f submissions/<today>.csv && python -c "
import csv
rows = list(csv.DictReader(open('submissions/<today>.csv')))
assert len(rows) == 943, f'expected 943, got {len(rows)}'
assert all('\\\\boxed{' in r['response'] for r in rows), 'some response missing boxed'
print('submission OK', len(rows), 'rows')
"

nvidia-smi --query-gpu=memory.used --format=csv
```

## 风险 + 应急

| 风险 | 触发条件 | 应急 |
|---|---|---|
| Day 1 v6 val 退步 | val < Phase 0 - 2pp | 确认 detect_question_type 正确（free_multi 漏判会回退到 free_single 规则）；如果改不动，Day 2 用 Phase 0 prompts + SC |
| Day 2 K=4 → 16 边际 < 1pp | K 已经饱和 | 不堆 K，改用 hot-tail sampling 或 LoRA |
| Day 3 leaderboard 退步（< 0.55） | SC 没救 | Phase 0 starter + SC K=16 直接重跑提交 |
| Day 4 LoRA holdout 跌 | 过拟合 | 丢 LoRA，回 Day 3 best config；Day 4 剩下时间用路线 B |
| Submission 格式破 | 本地 vs leaderboard > 5 % | 查 CSV：`(C)` / `C.` / 后空格 / 多余 `\\\\` 转义 |
| 提交配额超 | Kaggle 拒收 | 下一个 UTC 0:00 重置；Day 5/6 各留 1 次 |
| 机器突然没了 | 任何时间 | 当下 last-good submission CSV 即最终；外部备份在 |

## 时间余量

| Day | 任务 | 预算 | 弹性 |
|---|---|---|---|
| 1 | v6 val benchmark + 决策 | 0.5 | +0.5（GPU 挂机） |
| 2 | SC K-tuning on val | 1 | +0.25 |
| 3 | 全量 SC + 第 1 次 leaderboard | 1 | +0.5 |
| 4 | LoRA 训练 OR K-scaling | 1 | +0.5 |
| 5 | Final scaling + 第 2 次 leaderboard | 1 | +0.25 |
| 6 | 提交收尾 + 报告 | 1 | +0.75 |

**总预算**：5.5 天工作 + 0.5 天 buffer。任何 gate 失败 → 不强推，回上一天 last-good，把当天预算让给后面。

## 起步命令（Day 1 立刻可跑）

```bash
# 0. 每次新 shell
export PATH="$HOME/.local/bin:$PATH"
unset VIRTUAL_ENV CONDA_PREFIX

# 1. Day 1 子分支
git checkout jason/dev2 && git checkout -b day1-v6-baseline

# 2. 跑 v6 在 val_225 上的 single-shot
uv run cse151b-infer --data data/public.jsonl --val-only \
    --out results/v6_val.jsonl --max-tokens 16384 --temperature 0.6

# 3. 重判分 + 跟 Phase 0 比较
uv run cse151b-evaluate results/v6_val.jsonl
uv run cse151b-compare reports/baseline_v0_val.json results/v6_val.jsonl
```

(注：上面三行 `uv run` 命令依赖 `cse151b-infer / cse151b-evaluate / cse151b-compare` 的 `--val-only` 开关；如果没实现，先扫一下 `inference.py` / `evaluate.py` / `eval_harness.py` 的 CLI 有没有这些 flag，没有就先加。)
