"""Build runpod_h100/handoff/ — small, git-committable artifacts for the final report.

What it does:
  - Copies val_225 results, ckpt sweep summary, Kaggle CSV
  - Builds a "lite" version of private_sc_k8 (drops all_responses, keeps winning + extracted)
  - gzips the full private_sc_k8 jsonl as backup (still small enough for git after compression)
  - Extracts loss/reward traces from SFT + GRPO logs (compact .tsv files)
  - Writes a README index

Usage:
    python scripts/build_handoff.py
"""
from __future__ import annotations

import gzip
import json
import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
HANDOFF = REPO / "handoff"
RESULTS = REPO / "results"
LOGS = REPO / "logs"


def _copy_if_exists(src: pathlib.Path, dst: pathlib.Path, label: str) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        sz = dst.stat().st_size
        print(f"  [+] {label:<40s} {sz/1024:7.1f} KB → {dst.relative_to(REPO)}")
        return True
    print(f"  [-] {label:<40s} (skipped, not found: {src})")
    return False


def _build_private_lite(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Strip all_responses; keep winning_response + extracted summary."""
    if not src.exists():
        print(f"  [-] private_sc_k8_lite skipped (not found: {src})")
        return
    n_in = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            n_in += 1
            r = json.loads(line)
            lite = {
                "id": r.get("id"),
                "question_type": r.get("question_type"),
                "K": r.get("K"),
                "all_extracted": r.get("all_extracted"),
                "vote_counts": r.get("vote_counts"),
                "winning_answer": r.get("winning_answer"),
                "winning_response": r.get("winning_response"),
            }
            fout.write(json.dumps(lite) + "\n")
    sz = dst.stat().st_size
    print(f"  [+] private_sc_k8_lite.jsonl (no all_responses) {sz/(1024*1024):.1f} MB ← {n_in} rows")


def _gzip_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    if not src.exists():
        print(f"  [-] gzip skipped (not found: {src})")
        return
    with open(src, "rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, length=1024 * 1024)
    sz = dst.stat().st_size
    print(f"  [+] {dst.name} (gzipped)  {sz/(1024*1024):.1f} MB")


def _extract_log_metrics(log: pathlib.Path, dst: pathlib.Path, fields: list[str]) -> None:
    if not log.exists():
        print(f"  [-] log trace skipped (not found: {log})")
        return
    n = 0
    with open(dst, "w") as fout:
        fout.write("\t".join(fields) + "\n")
        with open(log) as fin:
            for line in fin:
                m = re.search(r"\{.*?\}", line)
                if not m:
                    continue
                try:
                    d = eval(m.group(), {"__builtins__": {}}, {})
                    if not isinstance(d, dict) or "loss" not in d:
                        continue
                except Exception:
                    continue
                fout.write("\t".join(str(d.get(f, "")) for f in fields) + "\n")
                n += 1
    print(f"  [+] {dst.name}: {n} entries")


def main() -> None:
    HANDOFF.mkdir(parents=True, exist_ok=True)
    print(f"[handoff] building artifacts in {HANDOFF}")

    # Small copyable artifacts
    _copy_if_exists(RESULTS / "val225_sft.jsonl", HANDOFF / "val225_sft.jsonl", "val_225 (SFT) results")
    _copy_if_exists(RESULTS / "grpo_ckpt_sweep_summary.json",
                    HANDOFF / "grpo_ckpt_sweep_summary.json", "GRPO ckpt sweep summary")
    _copy_if_exists(RESULTS / "grpo_ckpt_sweep.jsonl",
                    HANDOFF / "grpo_ckpt_sweep.jsonl", "GRPO ckpt sweep raw")
    _copy_if_exists(RESULTS / "public_k1.jsonl",
                    HANDOFF / "public_k1.jsonl", "public K=1 (GRPO model, full 1126)")
    # Kaggle CSVs — new naming "grpov3_{public,private}.csv"
    for tag in ("grpov3_public", "grpov3_private"):
        _copy_if_exists(RESULTS / f"{tag}.csv", HANDOFF / f"{tag}.csv", f"Kaggle {tag} CSV")
    # Backward compat: old name if it exists
    _copy_if_exists(RESULTS / "private_submission.csv",
                    HANDOFF / "private_submission.csv", "Kaggle private (legacy name)")

    # Private SC: lite version + gzipped full (try k4 then k8 path)
    for tag in ("k4", "k8"):
        src = RESULTS / f"private_sc_{tag}.jsonl"
        if src.exists():
            _build_private_lite(src, HANDOFF / f"private_sc_{tag}_lite.jsonl")
            _gzip_file(src, HANDOFF / f"private_sc_{tag}.jsonl.gz")
            break

    # Training metric traces (compact TSV from log JSON dicts)
    sft_fields = ["epoch", "loss", "grad_norm", "learning_rate", "entropy", "mean_token_accuracy", "num_tokens"]
    grpo_fields = ["epoch", "loss", "grad_norm", "learning_rate", "reward",
                   "rewards/reward_correct/mean", "rewards/reward_correct/std",
                   "completions/mean_length", "completions/clipped_ratio", "kl", "step_time"]
    _extract_log_metrics(LOGS / "sft_full.log", HANDOFF / "sft_loss_trace.tsv", sft_fields)
    _extract_log_metrics(LOGS / "grpo_full.log", HANDOFF / "grpo_step_trace.tsv", grpo_fields)

    # Pipeline status memo
    _copy_if_exists(REPO / "STATUS.md", HANDOFF / "STATUS.md", "Pipeline STATUS memo")

    # README index
    val_acc = None
    try:
        with open(RESULTS / "val225_sft.jsonl") as f:
            rows = [json.loads(l) for l in f]
        ncorrect = sum(1 for r in rows if r.get("correct"))
        val_acc = ncorrect / len(rows) * 100
    except Exception:
        pass

    # public_k1 (GRPO model) — split into train(901) vs val(225)
    public_k1_overall = public_k1_train = public_k1_val = None
    try:
        val_ids = set(json.load(open(REPO / "data" / "val_indices.json"))["val_ids"])
        with open(RESULTS / "public_k1.jsonl") as f:
            rows = [json.loads(l) for l in f]
        oc = sum(1 for r in rows if r.get("correct"))
        public_k1_overall = (oc / len(rows) * 100, oc, len(rows))
        vrows = [r for r in rows if r["id"] in val_ids]
        trows = [r for r in rows if r["id"] not in val_ids]
        if vrows:
            vc = sum(1 for r in vrows if r.get("correct"))
            public_k1_val = (vc / len(vrows) * 100, vc, len(vrows))
        if trows:
            tc = sum(1 for r in trows if r.get("correct"))
            public_k1_train = (tc / len(trows) * 100, tc, len(trows))
    except Exception:
        pass

    best_label = best_acc = None
    try:
        s = json.loads(open(RESULTS / "grpo_ckpt_sweep_summary.json").read())
        best_label = s["best"]["label"]
        best_acc = s["best"]["overall_acc"] * 100
    except Exception:
        pass

    readme = HANDOFF / "README.md"
    readme.write_text(f"""# CSE 151B SP26 — Handoff artifacts

Final pipeline outputs for the GRPO-trained Qwen3-4B-Thinking model.

## Headline metrics

- **val_225 (SFT-merged)**:  {f'{val_acc:.2f}%' if val_acc else '(not yet)'}
- **Best GRPO ckpt by val_225**: {best_label or '(not yet)'} @ {f'{best_acc:.2f}%' if best_acc else '(not yet)'}
- **Public K=1 (GRPO-merged, full 1126)**: {f'{public_k1_overall[0]:.2f}% ({public_k1_overall[1]}/{public_k1_overall[2]})' if public_k1_overall else '(not yet)'}
  - train_split (901, seen in SFT/GRPO): {f'{public_k1_train[0]:.2f}%' if public_k1_train else '(not yet)'}
  - **val_225 (true holdout, compare vs SFT 64.44%)**: {f'{public_k1_val[0]:.2f}%' if public_k1_val else '(not yet)'}
- **HF Hub**: https://huggingface.co/JaasonYuu/jason-cse151b-model
- **Kaggle CSV**: `private_submission.csv`

## Files

| File | What |
|---|---|
| `val225_sft.jsonl`              | 225-row val accuracy of the SFT-merged base (pre-GRPO) |
| `grpo_ckpt_sweep_summary.json`  | Ranking of every GRPO checkpoint by val_225 accuracy |
| `grpo_ckpt_sweep.jsonl`         | Per-ckpt × per-row scoring details |
| `public_k1.jsonl`               | Full public K=1 inference with GRPO-merged model (1126 questions, scored) |
| `private_sc_k8_lite.jsonl`      | Private K=8 SC voting results (winning answer + reasoning only) |
| `private_sc_k8.jsonl.gz`        | Full private K=8 SC (all 8 responses per question, gzip) |
| `private_submission.csv`        | Kaggle-format CSV |
| `sft_loss_trace.tsv`            | SFT loss/accuracy per logging step |
| `grpo_step_trace.tsv`           | GRPO loss/reward/length per step |
| `STATUS.md`                     | Full pipeline status, config, OOM history |

## Pipeline

```
1. SFT (5 epoch, r=64, max_seq=16384, completion_only_loss)
2. Merge LoRA → BF16 base
3. val_225 (Run F prompt + v2 budget)
4. GRPO (3 epoch, K=4 + hard-dup=1, beta=0.04, dr_grpo)
5. Best ckpt by val_225 sweep (vLLM enable_lora hot-swap)
6. Merge best GRPO LoRA → BF16
7. Upload to HuggingFace Hub
8. Private K=8 SC inference (Run F + v2 budget)
9. Build Kaggle CSV
10. Build this handoff/
11. git push
```
""")
    print(f"  [+] README.md")

    print(f"\n[handoff] done.")
    subprocess.run(["du", "-sh", str(HANDOFF)])


if __name__ == "__main__":
    main()
