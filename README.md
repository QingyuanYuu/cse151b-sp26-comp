# CSE 151B Spring 2026 — Math Reasoning Competition

Inference + scoring pipeline for the CSE 151B math reasoning competition. Built on top of the
course-provided `judger.py` / `utils.py` and the Qwen3-4B-Thinking-2507 starter notebook.

## Quickstart

```bash
uv sync --extra dev --extra cu126   # or --extra cu128 for RTX 50 series
uv run pre-commit install
uv run nbstripout --install         # strip notebook outputs from git diffs
```

After the first install, register the kernel for the notebook:

```bash
uv run python -m ipykernel install --user --name cse151b_comp --display-name "Python (cse151b_comp)"
```

Open `notebooks/starter_code_cse151b_comp.ipynb` and select the **Python (cse151b_comp)** kernel.

## Layout

```
src/cse151b_comp/      # our code (prompts, runner, analyze, dev_split, ...)
tests/                 # pytest, runs against src/cse151b_comp
notebooks/             # starter notebook + experiment notebooks
data/public.jsonl      # course-provided dataset (committed)
data/dev_*.jsonl       # local stratified splits (gitignored)
results/               # generated JSONL output (gitignored)
judger.py utils.py     # course-provided scoring (do NOT modify)
pyproject.toml         # uv + hatchling + ruff + mypy + pytest config
```

## Entry points

After `uv sync`, these commands are on `PATH` inside `.venv`:

| Command | Purpose |
|---|---|
| `cse151b-run` | Run inference over a JSONL dataset |
| `cse151b-analyze` | Re-judge a results file and print failure-mode breakdown |
| `cse151b-split` | Build a stratified dev split |

## Optional dependency groups

| Extra | What it pulls in |
|---|---|
| `dev` | `pytest`, `ruff`, `mypy`, `pre-commit`, `nbstripout` |
| `cu126` / `cu128` | PyTorch wheels for CUDA 12.6 / 12.8 (mutually exclusive) |
| `vllm` | `vllm==0.6.3` + `transformers==4.46.3` (Phase 2; use a separate `.venv-vllm`) |
| `train` | `peft`, `datasets`, `trl`, `wandb` (Phase 4 QLoRA) |

## Scoring quick reference

`judger.py` is generous on math equivalence (SymPy + LaTeX) but strict on output format:

- MCQ: emit **`\boxed{C}`** — `\boxed{(C)}`, `\boxed{C.}`, `\boxed{C)}` all silently fail.
- Free-form multi-part: emit either K consecutive `\boxed{}` blocks **or** one
  `\boxed{a, b, ..., k}`. Don't mix.
- Numeric: plain numbers, no units, no `x = ` prefix, no trailing punctuation.
- Thinking tags `<think>...</think>` before the final boxed answer are stripped automatically.
