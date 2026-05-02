# CSE 151B 数学推理竞赛 —— 改进计划（含 repo-template 结构）

## 背景

Starter code（`starter_code_cse151b_comp.ipynb`）跑通的是 **Qwen3-4B-Thinking-2507 INT4** + Transformers + BitsAndBytes，单卡 **RTX 4090（24GB）**。数据集 1126 题：33% MCQ（其中 90% 是 10 选 1，随机正确率仅 10%），67% 自由作答（其中 99% 是多 part 列表答案）。判分用 SymPy + LaTeX 做符号等价，**数学等价很宽容**，但**输出格式很严格** —— `\boxed{(C)}` 或 `\boxed{C.}` 会静默判错。

用户目标：**冲班级前列**，**愿意 LoRA 微调**。本计划从"baseline 刚跑完"一路到提交，每个阶段之间设硬门槛（gate），失败就回退。仓库结构遵循 `/home/jason/Desktop/repo-template`（uv + hatchling + ruff + mypy + pytest + pre-commit），方便团队协作和实验追踪。

## 仓库目标结构

参照 `/home/jason/Desktop/repo-template/`，最终 `151B_SP26_Competition/` 长这样：

```
151B_SP26_Competition/
├── src/cse151b_comp/              # 我们写的代码全部进这里
│   ├── __init__.py                # __version__ = "0.1.0"
│   ├── prompts.py                 # system + few-shot 字符串
│   ├── runner.py                  # 推理入口（CLI）
│   ├── analyze.py                 # 重判分 + 失败分类
│   ├── dev_split.py               # 抽 dev_200
│   ├── voting.py                  # n=5 self-consistency
│   ├── topics.py                  # 主题路由
│   └── lora_train.py              # Phase 4 训练入口
├── tests/                         # pytest 配置已指向这里
│   ├── test_prompts.py
│   ├── test_voting.py
│   └── test_topics.py
├── data/
│   ├── public.jsonl               # 课程提供
│   ├── dev_200.jsonl              # Phase 0 抽出，gitignore
│   └── dev_holdout.jsonl          # Phase 4 才用，gitignore
├── results/                       # 全部 gitignore
│   ├── baseline_v0.jsonl
│   ├── v1_prompt.jsonl
│   └── v3_full.jsonl
├── notebooks/
│   └── starter_code_cse151b_comp.ipynb   # 课程原文件，移过来
├── judger.py                      # 课程提供，根目录保留（不改）
├── utils.py                       # 课程提供，根目录保留（不改）
├── pyproject.toml                 # 用模板那份改
├── .pre-commit-config.yaml        # 直接拷模板
├── .gitignore                     # 模板 + 加 .venv-vllm/、*.safetensors 等
├── README.md                      # 项目说明
├── LICENSE                        # 模板 MIT
└── uv.lock                        # commit 进 git（应用级仓库要锁版本）
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

## Phase 0 —— 基线 + 错误分析（约半天，GPU 主要在挂机） ⏳ **下一步**

目标：拿到一个可信的基线分数 + 失败模式直方图。

1. ✅ ~~`GPU_ID = "0"`~~ 已改
2. **从 notebook 跑全量先拿 baseline**（最快路径）：
   - Cell 18 把 `data[:5]` 改成 `data`
   - Cell 7 把 `OUTPUT_PATH = "results/starter_results.jsonl"` 改成 `"results/baseline_v0.jsonl"`
   - 跑一次。预计 1-3 小时（vllm 0.20 + 4090 + INT4，n=1126，每题 thinking 平均 13k token）
3. 第二轮：把 generate / score 抽到 `src/cse151b_comp/runner.py`，CLI 参数 `--data --out --max_tokens --temperature --n_samples`。复用 notebook 现有的 `build_prompt()` 和 `extract_letter()`，加上 `Judger` 评分
4. 在 `pyproject.toml` 的 `[project.scripts]` 注册（已经写过）：
   ```toml
   [project.scripts]
   cse151b-run = "cse151b_comp.runner:main"
   cse151b-analyze = "cse151b_comp.analyze:main"
   cse151b-split = "cse151b_comp.dev_split:main"
   ```
5. `analyze.py` 计算：
   - 总体 / MCQ-only / 自由作答-only 准确率
   - MCQ 按 `len(options)` 分层；自由作答按 `len(gold_list)` 和有无 `[ANS]` 分层
   - 长度桶：`<150 / 150-500 / 500-1500 / >1500` 字符
   - 主题桶：正则打标（calculus / linalg / probability / ODE / complex / other）
   - 失败模式：(a) extractor 抽不到/形状错 (b) part 数量错 (c) 抽到了但答错 (d) 撞 `MAX_TOKENS`
6. `dev_split.py` 抽 `data/dev_200.jsonl`（200 题，固定 seed，按 type+topic+长度分层）。**Phase 1–3 的所有 A/B 只在 dev 上跑**
7. 给 `prompts.py` 写 unit test：`tests/test_prompts.py` 验证 MCQ 和 free-form 的 prompt 模板包含必要的格式指令

**Gate**：基线分数记录 + 失败直方图打印；pytest 全过。预期：50–62%

## Phase 1 —— Prompt + 采样快赢（约 1 天，目标 +4–10 分）

每个改动只在 dev 上 A/B。

**Prompt 修复（杠杆最大）** —— 全部写进 `src/cse151b_comp/prompts.py`：
- MCQ system prompt：正反例都给 —— `Output ONLY the letter. Example: \boxed{C}. Do NOT write \boxed{(C)}, \boxed{C.}, or \boxed{C)}`
- 多 part 自由作答：`If the answer has K parts, output exactly K \boxed{} blocks in order, OR a single \boxed{a, b, ..., k} comma-separated. Do not mix styles.` 加一个 few-shot 示例
- `[ANS]` 占位题（66%）：`Replace [ANS] with your final \boxed{...}. The final line must be the boxed answer.`
- 数值题：`Use plain numbers (0.5 not "1/2 of pi"), no units, no "x = ", no trailing punctuation.`
- **数值精度（5 题烟测发现的关键 bug）**：`Do not round numerical answers. Report at least 6 significant figures or the exact symbolic form. Example: 143.224229 not 143; 2.32625 not 2.33.`

**采样扫描**：dev 上扫 `temperature ∈ {0.0, 0.3, 0.6}` × `top_p ∈ {0.9, 0.95}`，单样本。按题型分别锁定胜出 config

**self-consistency**（Phase 2 之后再做，依赖 vLLM）：n=5 在 T=0.6 下采样，多 part 题逐 slot 投票，MCQ 取字母众数

**Gate**：dev 准确率 ≥ baseline + 4 分；`prompts.py` 改动有对应单元测试。锁定 config 跑全 1126 → `results/v1_prompt.jsonl`

## Phase 3 —— 针对性优化（约 2–3 天，目标 +5–13 分）

**3a. 主题路由**（约半天，+1–3 分） —— `src/cse151b_comp/topics.py`：正则识别题目主题，给 system prompt 加主题专属后缀。Calculus / Probability / Combinatorics / Linalg 各一套。配 `tests/test_topics.py`

**3b. self-consistency / 多数投票**（约半天，+2–5 分） —— `src/cse151b_comp/voting.py`：vLLM 下 n=5，T=0.6
- MCQ：字母众数
- 自由作答单值：用 `Judger.is_equal` 把等价答案聚类，取最大簇
- 多 part：每 slot 独立投票
配 `tests/test_voting.py`

**3c. 升级到 Qwen3-14B-Thinking**（约半天，+3–8 分 —— 单步增益最大）：INT4 权重 ~9GB，24GB 卡设 `max_model_len=8192`、`gpu_memory_utilization=0.9` 能塞下。先 dev 验证，dev 涨 ≥ 3 分就升级

**3d. SymPy 反思 pass**（约半天，+1–2 分）：只对自由作答中第一次抽答案为空、或 part 数量不对的题，跑第二轮 prompt：`Your previous answer was {extracted}. Verify by computing each part numerically and correct any errors.` 单次反思

**Gate**：dev ≥ Phase 1 + 5 分。跑全 1126 → `results/v3_full.jsonl`

## Phase 4 —— QLoRA 微调（约 1 天，目标 +3–5 分，过拟合风险真实）

只有 Phase 3 之后 dev 还有空间（失败直方图里仍有 reasoning 错误而不只是 format 错误）才做。

代码进 `src/cse151b_comp/lora_train.py`，依赖加 extra：
```toml
[project.optional-dependencies]
train = ["peft>=0.12", "datasets>=2.20", "trl>=0.10", "wandb"]
```

1. 训练集：`public.jsonl` 减去 `dev_200`，再分 90/10。10% 那部分单独存为 `dev_holdout.jsonl`，**前面所有阶段都不许碰**
2. Target 包含**思考 + 装箱**完整形态：`<think>...reasoning...</think>\n\nFinal answer: \boxed{...}`。**重要**：不能用裸 `gold` 列表，否则模型学会丢 `\boxed{}`
3. QLoRA r=16（14B 用 r=8）on Qwen3-(4B|14B)-Thinking，`bsz=1`、`accum=8`、`gradient_checkpointing=True`、`lr=2e-4`、2–3 epoch
4. LoRA 在 **held-out + dev** 上评估。**只有 held-out 涨 ≥ 3 分才上线**

**Gate**：held-out ≥ Phase 3 best held-out + 3 分。否则丢 LoRA 走 Phase 3

## Phase 5 —— 提交准备（约 2 小时）

1. notebook 里 `SAVE_EVAL=False`（输出 schema 变成 `{id, is_mcq, response}`）
2. 校验：每条都有 `id`、无重复、`response` 非空。一行：`uv run python -c "import json; rows=[json.loads(l) for l in open('submission.jsonl')]; print(len(rows), len(set(r['id'] for r in rows)))"`
3. 抽 10 条目测，确认 `response` 含 `\boxed{...}`
4. 用**和提交完全一样的 config** 在 public 上跑本地判分，记录预期分。leaderboard 差距悬殊就是格式破了
5. 准备两份提交：best single-pass 和 best n=5-voted

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
| 0 baseline 全量 | 0.5 天 | — | 低 | ⏳ 下一步 |
| 1 prompts + 采样 | 1 天 | +4 至 +10 | 低 | 待开 |
| 3a 主题路由 | 0.5 天 | +1 至 +3 | 低 | 待开 |
| 3b n=5 投票 | 0.5 天 | +2 至 +5 | 低 | 待开 |
| 3c 14B 升级 | 0.5 天 | +3 至 +8 | 中（显存） | 待开 |
| 3d SymPy 反思 | 0.5 天 | +1 至 +2 | 低 | 待开 |
| 4 QLoRA | 1 天 | +3 至 +5 | 中（过拟合） | 待开 |
| 5 提交 | 0.25 天 | — | 低 | 待开 |

从约 55% baseline 起步：现实落点 **70–80% 总分**，每个 gate 都有早停机制。

## 团队协作约定

- 实验分支化：`main` 保稳定可跑版本，每次大改开 `phaseN-feature` 分支
- 每个 PR 必须：(a) `pre-commit` 干净 (b) `pytest` 全过 (c) 至少一个队友 review (d) PR description 写 dev 分变化
- `results/*.jsonl` 太大不进 git；放团队共享盘（Google Drive / OneDrive / S3）；PR 里贴上 results 路径和 dev 分
- 大文件（模型 checkpoint、中间数据）走 `git-lfs` 或外部存储，不进 git
- 课程禁止公开代码 → repo 必须 **Private**，提交前再次确认 visibility
