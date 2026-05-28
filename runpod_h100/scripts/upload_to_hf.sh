#!/usr/bin/env bash
# Upload the final merged GRPO model to HuggingFace Hub.
# Repo: JaasonYuu/jason-cse151b-model
set -euo pipefail

REPO_DIR=/workspace/cse151b-grpo/cse151b-sp26-comp/runpod_h100
HF_REPO="${HF_REPO:-JaasonYuu/jason-cse151b-model}"
MODEL_DIR="${1:-$REPO_DIR/checkpoints/grpo_v6_merged}"
HF_BIN=/workspace/cse151b-grpo/.venv/bin/hf

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: model dir not found: $MODEL_DIR" >&2
    exit 1
fi

echo "[hf-upload] target repo: $HF_REPO"
echo "[hf-upload] source:      $MODEL_DIR"
echo "[hf-upload] size:        $(du -sh "$MODEL_DIR" | cut -f1)"

# Create the repo if it doesn't exist (idempotent, --exist-ok-equivalent via || true)
$HF_BIN repo create "$HF_REPO" --type model -y 2>&1 | tail -3 || true

# Upload — default privacy is whatever the repo currently is set to.
$HF_BIN upload "$HF_REPO" "$MODEL_DIR" --repo-type model --commit-message "GRPO + SFT merged (Qwen3-4B-Thinking-2507 + LoRA chain)"

echo "[hf-upload] done → https://huggingface.co/$HF_REPO"
