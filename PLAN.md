# CSE 151B 数学推理竞赛 —— 改进计划（含 repo-template 结构）

## 背景

**Kaggle 比赛**：Submission Competition，提交 CSV（`id, response`）。

**数据集**：
- `data/public.jsonl` —— **1126 题**带答案，本地开发 / 验证用
- `data/private.jsonl` —— **943 题**无答案，leaderboard 评分用

题型分布（public 上统计）：33% MCQ（其中 90% 是 10 选 1），67% 自由作答（99% 是多 part 列表答案）。

模型：**Qwen3-4B-Thinking-2507 INT4** + vLLM 0.20 + BitsAndBytes，单卡 **RTX 4090（24GB）**。

判分：SymPy + LaTeX 符号等价，数学等价宽容，但输出格式严格 —— `\boxed{(C)}` 或 `\boxed{C.}` 会静默判错。

用户目标：**冲班级前列**，**愿意 LoRA 微调**。本计划从"baseline 刚跑完"一路到提交，每个阶段之间设硬门槛（gate），失败就回退。仓库结构遵循 `/home/jason/Desktop/repo-template`（uv + hatchling + ruff + mypy + pytest + pre-commit），方便团队协作和实验追踪。

## 当前进度（2026-05-03 14:30）

- ✅ Phase -1 仓库重构完成
- ✅ Phase 0.5 vLLM 解锁完成
- ✅ **首次 leaderboard 提交完成**（`results/submission_v1.csv`，943 行，779/943=82.6% 有 `\boxed{}`）
- ⚠️ **诊断**：17.4% 缺 `\boxed{}` 主因是 MAX_TOKENS=12288 截断（94/164 = 57% response > 30k 字符）。**MCQ 反而最严重（28% 缺 box）**因为 10 选 1 题逐个分析选项耗尽 tokens
- ⏳ **下一步**：等 leaderboard 出分 + 在 public.jsonl 跑本地基线（**MAX_TOKENS=16384**）
- 📝 Milestone Report 模板已就位（`templates/milestone-report/main.tex`）—— 需要并行写

## 比赛硬约束（spec 列明，违反 = 0 分）

1. 最终推理**只能用 `Qwen/Qwen3-4B-Thinking-2507`**（14B 升级违规）。LoRA 微调允许
2. 推理时**禁外部 API、code interpreter、calculator**（SymPy 只能在 prompt rewrite 阶段用，不能在 inference 时跑代码）
3. 提交是 CSV 含完整 reasoning trace，grader 抓 `\boxed{...}`
4. 硬件：单张 RTX 4090 24GB

## 仓库目标结构

参照 `/home/jason/Desktop/repo-template/`，最终 `151B_SP26_Competition/` 长这样：

```
151B_SP26_Competition/
├── src/cse151b_comp/              # 我们写的代码全部进这里
│   ├── __init__.py                # __version__ = "0.1.0"
│   ├── prompts.py                 # system + few-shot 字符串
│   ├── inference.py               # vLLM 推理入口（CLI），原 runner.py 改名
│   ├── extract.py                 # 从 response 抽 \boxed{...}（标准答案抽取器）
│   ├── evaluate.py                # 对一份 results 文件用 judger 算分（重判分）
│   ├── error_analysis.py          # 失败分类直方图（按 type / 长度 / topic）
│   ├── submission.py              # 把 responses → Kaggle CSV
│   ├── self_consistency.py        # K=N 多采样 + 投票（Phase 2a）
│   ├── prepare_sft_data.py        # NuminaMath + decontamination（Phase 2b）
│   ├── eval_harness.py            # stratified val + compare mode（Phase 2c）
│   ├── topics.py                  # 主题路由（Phase 3a）
│   ├── voting.py                  # voting helpers（被 self_consistency 调用）
│   └── lora_train.py              # Phase 4 训练入口
├── tests/                         # pytest 配置已指向这里
│   ├── test_prompts.py
│   ├── test_extract.py            # answer extractor 单测
│   ├── test_normalize.py          # normalize_answer 30+ 单测
│   ├── test_voting.py
│   └── test_topics.py
├── data/
│   ├── public.jsonl               # 课程提供（1126 题）
│   ├── private.jsonl              # 课程提供（943 题，submission 用）
│   ├── val_indices.json           # stratified 20% val 的 id 列表，固定
│   ├── sft_train.jsonl            # Phase 2b 产出（gitignore）
│   └── dev_holdout.jsonl          # Phase 4 才用（gitignore）
├── results/                       # 全部 gitignore（spec 叫 outputs/，我们保留 results/）
│   ├── baseline_public_v0.jsonl   # Phase 0b 本地 public 全量基线
│   ├── sc_k4.jsonl / sc_k8.jsonl  # Phase 2a self-consistency
│   ├── submission_v1.csv          # 已交
│   └── submission_v2_*.csv        # 后续版本
├── reports/                       # markdown / json 报告
│   ├── phase2_summary.md          # Phase 2 末尾交付
│   └── eval_*.json
├── notebooks/                     # 探索 / 调试 / 出图，不再产 submission
│   └── starter_code_cse151b_comp.ipynb
├── templates/milestone-report/    # Milestone Report LaTeX 模板
├── judger.py utils.py             # 课程提供，根目录保留（不改）
├── pyproject.toml .pre-commit-config.yaml .gitignore README.md LICENSE uv.lock
```

`judger.py` 和 `utils.py` 留在根目录是因为 notebook 里 `sys.path.insert(0, ".")` 直接 import，并且课程评分用的是这两个文件本身——不动它们。我们的代码以 package 形式调用它们：

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))  # repo root
from judger import Judger
```

## 关键文件清单

- 现有（**不改**）：`judger.py`、`utils.py`、`data/public.jsonl`、`starter_code_cse151b_comp.ipynb`
- 新建：上面 `src/cse151b_comp/*.py` 全部 + tests + pyproject.toml + .pre-commit-config.yaml + 更新 .gitignore + README

## Phase 0.5 —— vLLM 解锁 + 参数调整 ✅ 已完成（2026-05-01）

**实际生效配置**（写下来，未来环境重建时照抄）：

| 项 | 计划值 | 实际值 | 原因 |
|---|---|---|---|
| transformers | 4.51–4.55 | **4.57.6** | vllm 0.20.0 实际要求 `>=4.56.0`（pip 解析时报错） |
| vllm | 0.20.0 不动 | 0.20.0 | 同 plan |
| `MAX_TOKENS` | 12288 | 12288 | 同 plan |
| `GPU_ID` | "0" | "0" | 同 plan |
| `gpu_memory_utilization` | 0.85 | **0.70** | 桌面 + WSL 类进程占 4.6 GB；0.85 触发 OOM |
| `max_num_seqs` | 64 | 64 | 同 plan |
| `max_num_batched_tokens` | 16384 | 16384 | 同 plan |
| **`VLLM_HOST_IP=127.0.0.1`** | （计划没提） | **必须设** | vllm V1 engine 默认绑机器主网卡 IP（这台机器是 `100.80.x.x`），跨进程 socket 连不上，永远 retry |

注入位置：`notebook cell 7` 的 `os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID` 之后立刻设：
```python
os.environ["VLLM_HOST_IP"] = "127.0.0.1"
```

**5 题烟测结果**：
- 模型加载 26 秒（torch.compile 15.6s + CUDA graph 2s）
- 生成 + 评分 < 2 分钟
- 准确率 **3/5 = 60%**（MCQ 1/2，free-form 2/3），落在预测区间内
- 关键观察：模型把 `143.224229...` 输出成 `\boxed{143}`、`2.32624...` 输出成 `\boxed{2.33}` —— **过度四舍五入**直接被 1e-8 相对误差判错。Phase 1 prompt 必须明确禁止 rounding

## Phase 2 —— vLLM 提速 ✅ 已合并到 Phase 0.5

原本计划用独立 venv + vllm 0.6.3。实际在主 venv 用 vllm 0.20.0 直接搞定，无独立 venv，**Phase 2 删除**。

## Phase -1 —— 仓库重构 + 模板化 ✅ 已完成（2026-05-01）

已经按模板拷出 `pyproject.toml`、`.pre-commit-config.yaml`、`.gitignore`、`README.md`，建好 `src/cse151b_comp/`（含 `__init__.py`、`prompts.py`、`runner.py` / `analyze.py` / `dev_split.py` 占位）和 `tests/test_prompts.py`。

**剩下没做的善后（Phase 0 跑全量之前可以一起处理）**：

1. **环境从 pip 切到 uv 管理**：当前 `.venv` 是 pip 装的，没 lock。等 baseline 跑完之后：
   ```bash
   rm -rf .venv
   uv sync --extra dev
   uv run pre-commit install
   uv run nbstripout --install
   uv run python -m ipykernel install --user --name cse151b_comp --display-name "Python (cse151b_comp)"
   ```
2. **notebook 移到 `notebooks/`**：现在还在根目录，`notebooks/` 已经空了。等 kernel 关闭再 `mv` + 改 VS Code 标签
3. **GitHub 私有仓库**（可选）：课程允许的话建 Private repo + 加队友 collaborator

## Phase 0 —— 基线 + 错误分析（约半天，GPU 主要在挂机） 🟡 **进行中**

目标：拿到 leaderboard 分 + 本地 public 集分（A/B 基线）+ 失败模式直方图。

**已完成**：
- ✅ Cell 18 跑完 private.jsonl 全 943 题（耗时 3 小时，502 toks/s 输出吞吐）
- ✅ `results/submission_v1.csv` 写出，943 行 / 943 唯一 id / 0 空行 / 779 有 `\boxed{}`（82.6%）
- ✅ 上传 Kaggle leaderboard

**待做**：

### 0c-1. 拆 notebook 到 src/ 模块（命名按 spec）

把 notebook 的逻辑抽到独立 .py 文件，便于 Phase 2 复用 + 单元测试。命名采纳竞赛 spec 风格：

| 文件 | 职责 | 来源 |
|---|---|---|
| `src/cse151b_comp/inference.py` | vLLM 推理（替换 cell 13 + 18），CLI `--data --out --max_tokens --temperature` | notebook cell 13/18 |
| `src/cse151b_comp/extract.py` | 从 response 抽 `\boxed{...}`（MCQ 字母 / free-form list / 多 part），含 `normalize_answer(s)` | notebook cell 22 + 重写 |
| `src/cse151b_comp/evaluate.py` | 对一份 jsonl results 用 `Judger` 重判分，输出准确率 | notebook cell 22 + 24 |
| `src/cse151b_comp/error_analysis.py` | 失败分类直方图（按题型/长度/topic）+ 17.4% 缺 box 题分析 | 新增（之前只在临时脚本里）|
| `src/cse151b_comp/submission.py` | jsonl → Kaggle CSV（QUOTE_ALL 转义） | notebook cell 26 |

`pyproject.toml` 里 `[project.scripts]` 改：
```toml
[project.scripts]
cse151b-infer = "cse151b_comp.inference:main"
cse151b-evaluate = "cse151b_comp.evaluate:main"
cse151b-analyze = "cse151b_comp.error_analysis:main"
cse151b-submit = "cse151b_comp.submission:main"
cse151b-split = "cse151b_comp.eval_harness:make_split_main"
```

### 0c-2. 跑 public 全量基线

`uv run cse151b-infer --data data/public.jsonl --out results/baseline_public_v0.jsonl --max_tokens 16384`

注意 **MAX_TOKENS 提到 16384**（不是 12288），因为 17.4% 缺 box 主要是被 12288 截断。约 4 小时。

### 0c-3. 等 leaderboard 出分

记录到 README + `reports/baseline_v0.md`（leaderboard 截图 + 943 题 boxed rate）。

### 0c-4. 错误分析

`uv run cse151b-analyze --results results/baseline_public_v0.jsonl --report reports/baseline_v0.md` 输出：

- 总体 / MCQ-only / 自由作答-only 准确率
- MCQ 按 `len(options)` 分层；自由作答按 `len(gold_list)` 和有无 `[ANS]` 分层
- 长度桶：`<150 / 150-500 / 500-1500 / >1500` 字符
- 主题桶：正则打标（calculus / linalg / probability / ODE / complex / other）
- **失败模式**：(a) extractor 抽不到/形状错 (b) part 数量错 (c) 抽到了但答错 (d) 撞 `MAX_TOKENS`
- 17.4% 缺 box 题在提到 16384 之后还剩多少 → 决定 Phase 1 prompt 重点
- Top 10 high-confidence-wrong（自由作答里 model 给了清晰 boxed 答案但错的）

**Gate**：
- Leaderboard 公开分记录
- Public 全量本地准确率记录（A/B 基线）
- `error_analysis.py` 失败直方图打印 + 哪类题最弱清楚
- pytest 全过；`extract.py` + `normalize_answer` 至少 30 个单测（spec 要求，cover edge cases：分数、负号、科学记数、unicode minus、千位分隔符等）

## Phase 1 —— Prompt + 采样快赢（约 1 天，目标 +4–10 分）

每个改动只在 stratified val（20% public，~225 题）上 A/B，不在 dev_200 那种东西上（已替换）。

**Prompt 修复（杠杆最大）** —— 全部写进 `src/cse151b_comp/prompts.py`：
- MCQ system prompt：正反例都给 —— `Output ONLY the letter. Example: \boxed{C}. Do NOT write \boxed{(C)}, \boxed{C.}, or \boxed{C)}`
- 多 part 自由作答：`If the answer has K parts, output exactly K \boxed{} blocks in order, OR a single \boxed{a, b, ..., k} comma-separated. Do not mix styles.` 加一个 few-shot 示例
- `[ANS]` 占位题（66%）：`Replace [ANS] with your final \boxed{...}. The final line must be the boxed answer.`
- 数值题：`Use plain numbers (0.5 not "1/2 of pi"), no units, no "x = ", no trailing punctuation.`
- **数值精度（5 题烟测发现的关键 bug）**：`Do not round numerical answers. Report at least 6 significant figures or the exact symbolic form. Example: 143.224229 not 143; 2.32625 not 2.33.`
- **Token 预算自救**：`If you are running out of reasoning room, immediately output your best-guess answer in \boxed{...} before stopping.`

**采样扫描**：val 上扫 `temperature ∈ {0.0, 0.3, 0.6}` × `top_p ∈ {0.9, 0.95}`，单样本。按题型分别锁定胜出 config

**Gate**：val 准确率 ≥ baseline + 4 分；`prompts.py` 改动有对应单元测试。锁定 config 跑全 1126 public → `results/v1_prompt.jsonl`，跑全 943 private → `results/submission_v2_prompt.csv` 提交

## Phase 2 —— Self-Consistency + SFT 准备 + Eval Harness（约 2 天，目标 +5–10 分 + 解锁 Phase 4）

按 spec 拆三个独立模块。

### Phase 2a — `self_consistency.py`（约半天，目标 +3–8 分）

**核心**：vLLM `n=K` 单次调用产生 K 个采样，按题型投票。

- **采样**：`temperature=0.7, top_p=0.95`（spec 推荐，比 greedy 0.6 更多样），`seed` per-sample 变化
- **投票**：
  - **MCQ**：抽 letter，多数票；tie-break 用最长有效 reasoning
  - **free_single**：抽 `\boxed{...}` 后过 `normalize_answer()`，按归一化形态投票，返回最常见原始形式
  - **free_multi**：整个 tuple `\boxed{a, b, c}` 投票（不 per-element），归一化每个元素再 tuple 等价比较
- **`normalize_answer(s)` 纯字符串/正则**（不用 sympy）：strip whitespace + `$`、统一负号（unicode `−`→`-`）、`1/2 → 0.5`、`-512 == -512.0 == -512.00`、`1,000 → 1000`、`2.5e3 → 2500`、`\pi` 不近似、trim trailing zeros。**30+ 单测覆盖 edge cases**

**Ablation**（spec 要求）：在 val 上跑 `K ∈ {1, 4, 8}`（K=16 不强制，看 K=8 显存余量再决定），输出 K-vs-accuracy 曲线，分题型看
- **关键指标**：「Solvable but missed」—— K 个采样里至少一个对、但投票选错的题数。区分"模型能力上限"和"投票质量"

**输出 schema**（每题）：
```json
{"id":42, "question_type":"free_single",
 "all_responses":[...K...], "all_extracted":[...K...],
 "vote_counts":{"norm_a":5, "norm_b":3},
 "winning_answer":"norm_a",
 "winning_response":"<最长 winning sample>"}
```

**显存**：vLLM `n=8` 同 prompt 共享 prefix，KV cache 8×8K ≈ 6GB，加 INT4 模型 5GB ≈ 11GB，4090 24GB 足够。先 dev 验证再全量

### Phase 2b — `prepare_sft_data.py`（约 1 天，为 Phase 4 LoRA 备料）

**数据源**：
- 主：`AI-MO/NuminaMath-CoT`（竞赛级数学 + CoT）
- 拌料 25%：`HuggingFaceTB/smoltalk` 高质量通用指令（防 capability forgetting，Unsloth 75/25 推荐）

**Decontamination（必做，否则训练数据污染）**：
- 8-gram overlap：candidate question 跟 `data/public.jsonl` 每题的 8-gram 集合比 Jaccard，>50% 直接拒绝
- exact substring match（case-insensitive，whitespace-normalized）也要查
- log 拒绝数

**格式转换**（最易出错的一步，做完必抽 5 条人眼检查）：
```
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
<think>
{numina_cot}
</think>

Final answer: \boxed{ground_truth}<|im_end|>
```
- 用 `tokenizer.apply_chat_template(..., enable_thinking=True)` 验证渲染
- 抽 5 条全文打印，肉眼确认 `<think>` 嵌套 + `\boxed{}` 在结尾 + 无特殊 token 损坏

**质量过滤**：
- CoT < 50 token 拒（低质）
- CoT > 4000 token 拒（炸训练上下文）
- 抽不出可靠答案的拒
- MinHash 近似去重（没 datasketch 就退化为 question 完整字符串 hash）

**MC / multi-answer augmentation**：spec 自己说复杂可以跳。**TODO 留注释**，先全用 free_single 跑通

**规模分阶段**：
1. 先 **5K 跑通流程**（约 1 epoch，单卡 4090 ~30 min），验证训练 loss 正常下降、生成不炸
2. **如果 5K 训完在 val 上有提升** → 扩到 50K，2-3 epoch 跑约 6-10 小时
3. 如果 5K 都没用，说明 NuminaMath 跟比赛分布偏太远，回头改 augmentation 策略

输出 `data/sft_train.jsonl`：`{id, messages, question_type, source}`，最后打统计：count / source / type / avg_seq_len / max_seq_len

### Phase 2c — `eval_harness.py`（约半天，所有后续 A/B 都靠它）

**Stratified val/train split**（一次性）：
- public.jsonl 取 20% 作为 val（按 question_type 分层），剩 80% 是 train（Phase 4 用）
- 存 `data/val_indices.json`，固定 seed=42，所有 phase 都用这个

**评估模式**：
1. Greedy（K=1）baseline
2. Self-consistency（K=N）
3. **Compare 模式**：两份 results 文件 side-by-side
   - `python eval_harness.py --compare results/baseline.jsonl results/sc_k8.jsonl`
   - 输出：N 都对、N 仅 baseline 对（**回归！**）、N 仅 SC 对（增益）、N 都错

**指标**：
- 总体 / 按题型 / 按长度桶
- 格式失败率（response 没 `\boxed{}`）
- Top 10 confident-but-wrong（高置信度错答案）
- 「Solvable but missed」（K 里至少一对，投票选错）

**输出**：`reports/eval_{config_name}_{timestamp}.json` + markdown summary

### Phase 2d — `reports/phase2_summary.md`

末尾交付物，含：
1. K-vs-accuracy 曲线（ASCII 图也行）
2. 最佳 K + trade-off 讨论
3. SFT 数据集统计
4. 3 条 SFT 样本（人眼 verify 用）
5. K=8 时 Solvable-but-missed 比率（投票质量上限）
6. **建议下一步**：优先 SFT 还是继续 inference-time 优化（有数据支撑）

**Gate**：
- val 上 K=8 vs K=1 至少 +3 分；否则 self-consistency 不上线
- SFT 5K pilot 训完，val 至少不下降；否则停掉别浪费 50K 训练时间
- `phase2_summary.md` 里"下一步"决定 Phase 3/4 顺序

## Phase 3 —— 针对性优化（约 1.5 天，目标 +4–13 分）

**3a. 主题路由**（约半天，+1–3 分） —— `src/cse151b_comp/topics.py`：正则识别题目主题，给 system prompt 加主题专属后缀。Calculus / Probability / Combinatorics / Linalg 各一套。配 `tests/test_topics.py`

**~~3b. self-consistency~~** —— **已并入 Phase 2a**

**3c. ⚠️ 升级到 Qwen3-14B-Thinking 的合规风险**：spec 第一条说"必须只用 `Qwen/Qwen3-4B-Thinking-2507`"。**14B 升级违规，从 Phase 3 删除**。如果你确认课程允许换更大模型再加回来。

**3d. SymPy 反思 pass**（约半天，+1–2 分）：只对自由作答中第一次抽答案为空、或 part 数量不对的题，跑第二轮 prompt：`Your previous answer was {extracted}. Verify by computing each part numerically and correct any errors.` 单次反思。注意 spec 说"No external API calls, no code interpreters, no calculators"，所以 SymPy 反思**只能用于 prompt rewrite**，不能在 inference 时调用 SymPy 计算（违规）。

**Gate**：val ≥ Phase 2 + 3 分。跑全 1126 public → `results/v3_full.jsonl`，跑全 943 private 提交

## Phase 4 —— QLoRA 微调（约 1.5 天，目标 +3–5 分，过拟合风险真实）

依赖：**Phase 2b 必须完成**（`data/sft_train.jsonl` 已就位 + decontaminated）。

代码进 `src/cse151b_comp/lora_train.py`，依赖加 extra：
```toml
[project.optional-dependencies]
train = ["peft>=0.12", "datasets>=2.20", "trl>=0.10", "wandb"]
```

**两阶段训练**：

**4a. 5K Pilot**（30 分钟，先验证流程）：
- 用 Phase 2b 产出的 `data/sft_train.jsonl` 头 5000 条
- QLoRA r=16 on Qwen3-4B-Thinking（spec 锁死 4B），`bsz=1`、`accum=8`、`grad_ckpt=True`、`lr=2e-4`、1 epoch
- 训完在 stratified val（20% public）上跑 inference，跟 Phase 2 best config 比
- **Gate**：val 至少不下降。下降说明 NuminaMath 跟比赛分布偏差太大，**停掉别浪费 50K 训练时间**，回 Phase 2b 重检 augmentation

**4b. 50K Full**（6-10 小时，5K pilot 通过才做）：
- 全量 50K 跑 2-3 epoch
- 注意训练 target **包含 thinking + boxed 完整形态**（`<think>...</think>\n\nFinal answer: \boxed{...}`）。**不能用裸 `gold` 否则模型学会丢 `\boxed{}`**
- LoRA 在 **dev_holdout（10% public 之前不碰）** 上评估
- **只有 dev_holdout 涨 ≥ 3 分才上线**

**Gate**：dev_holdout ≥ Phase 3 best + 3 分。否则丢 LoRA 走 Phase 3

## Phase 5 —— 提交准备（约 2 小时）

提交格式（已知）：CSV，列 `id,response`，全部用 `csv.QUOTE_ALL` 转义。这部分已经在 cell 26 实装。

每次新版 submission 之前的 checklist：
1. 校验：每条都有 `id`、无重复、`response` 非空、`Has \boxed{}` 接近 100%（缺的太多说明 max_tokens 被截了）
2. 抽 10 条目测，确认 `response` 含 `\boxed{...}` 在最后
3. 用**和提交完全一样的 config** 在 public 上跑本地判分，记录预期分。leaderboard 差距悬殊就是格式破了
4. 命名规则：`results/submission_v{N}_{tag}.csv`，比如 `submission_v2_prompt-fix.csv`
5. Kaggle 提交时 description 写清楚改了啥：`v2 prompt fix: anti-rounding + multi-part instruction`
6. 注意 **5 次/天** 限额，重要 config 改动留次数

## Milestone Report（并行轨道，截止日期前 1 周开写）

模板已就位：`templates/milestone-report/main.tex`（NeurIPS 2024 格式）

**写作策略**：
- 实验做到哪写到哪，不要 Phase 4 全做完才动笔
- 主要章节：Problem / Approach（pipeline 图）/ Experiments（按 Phase 0-1-3 列 dev 分对比表）/ Discussion（哪类题最弱）
- 编译用 Overleaf（最简单）：上传整个 `templates/milestone-report/` 文件夹
- 队友协作时所有图表数据从 `results/*.jsonl` 来，可复现
- LaTeX 产物（`.aux`/`.log`/`.pdf`/`.bbl` 等）已经在 `.gitignore` 里 / 准备加进去

## 端到端验证

每阶段成功 = 全部满足：
- `results/` 下新文件存在，能解析 JSONL，行数对（1126 全量 / 200 dev）
- `analyze.py` 打印准确率 + 失败直方图，gate 指标达成
- `pytest` 全过；`pre-commit run --all-files` 干净
- `dev_holdout.jsonl` 在 Phase 4 最终评估前**不被读写**（`git status` 不应看到对它的写入）
- 最终提交通过行数 + box 抽检

## 时间 + 风险预算

| Phase | Wall time | 预期增益 | 风险 | 状态 |
|---|---|---|---|---|
| -1 模板重构 | 0.25 天 | — | 低 | ✅ 完成 |
| 0.5 vLLM 解锁 | 0.25 天 | — | 中（环境） | ✅ 完成 |
| ~~2 vLLM~~ | ~~0.5 天~~ | — | — | 删除（合并入 0.5） |
| 0a private 提交 | 0.5 天 | — | 低 | ✅ 完成（leaderboard 待出分）|
| 0b public 全量基线（MAX_TOKENS=16384）| 0.5 天 | +5 至 +8（修截断）| 低 | ⏳ 下一步 |
| 0c-1 拆 notebook → src/ 模块 | 0.5 天 | — | 低 | 待开 |
| 0c-2 error_analysis 失败直方图 | 0.25 天 | — | 低 | 待开 |
| 1 prompts + 采样扫描 | 1 天 | +4 至 +10 | 低 | 待开 |
| **2a self_consistency K-ablation** | 0.5 天 | +3 至 +8 | 低 | 待开 |
| **2b prepare_sft_data + decontam** | 1 天 | — | 中（数据源） | 待开 |
| **2c eval_harness + compare mode** | 0.5 天 | — | 低 | 待开 |
| **2d phase2_summary.md** | 0.25 天 | — | 低 | 待开 |
| 3a 主题路由 | 0.5 天 | +1 至 +3 | 低 | 待开 |
| 3c 14B 升级 | 0.5 天 | +3 至 +8 | 中（显存） | 待开 |
| 3d SymPy 反思 | 0.5 天 | +1 至 +2 | 低 | 待开 |
| 4 QLoRA（5K pilot → 50K） | 1.5 天 | +3 至 +5 | 中（过拟合 + 时间） | 待开（依赖 2b） |
| 5 提交（每次新版前） | 0.25 天 | — | 低 | 流程化 |
| Milestone Report | 1 天（并行） | — | 低 | 待开 |

**注**：Phase 3b（投票）已合并入 **Phase 2a**（同一个 self_consistency.py）。

从约 55% baseline 起步：现实落点 **70–80% 总分**，每个 gate 都有早停机制。

## 团队协作约定

- 实验分支化：`main` 保稳定可跑版本，每次大改开 `phaseN-feature` 分支
- 每个 PR 必须：(a) `pre-commit` 干净 (b) `pytest` 全过 (c) 至少一个队友 review (d) PR description 写 dev 分变化
- `results/*.jsonl` 太大不进 git；放团队共享盘（Google Drive / OneDrive / S3）；PR 里贴上 results 路径和 dev 分
- 大文件（模型 checkpoint、中间数据）走 `git-lfs` 或外部存储，不进 git
- 课程禁止公开代码 → repo 必须 **Private**，提交前再次确认 visibility
