# CSE 151B SP26 — Handoff artifacts

Final pipeline outputs for the GRPO-trained Qwen3-4B-Thinking model.

## Headline metrics

- **val_225 (SFT-merged)**:  64.89%
- **Best GRPO ckpt by val_225**: step-606 @ 66.22%
- **Public K=1 (GRPO-merged, full 1126)**: 65.63% (739/1126)
  - train_split (901, seen in SFT/GRPO): 65.93%
  - **val_225 (true holdout, compare vs SFT 64.44%)**: 64.44%
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
