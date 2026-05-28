#!/usr/bin/env bash
# Commit + push runpod_h100/handoff/ to GitHub.
set -euo pipefail

REPO_ROOT=/workspace/cse151b-grpo/cse151b-sp26-comp
HANDOFF_DIR="$REPO_ROOT/runpod_h100/handoff"
STATUS_FILE="$REPO_ROOT/runpod_h100/STATUS.md"
BRANCH="${BRANCH:-runpod-h100-train}"

cd "$REPO_ROOT"

if [[ ! -d "$HANDOFF_DIR" ]]; then
    echo "[git-push] ERROR: $HANDOFF_DIR not found — run build_handoff.py first" >&2
    exit 1
fi

echo "[git-push] target branch: $BRANCH"
echo "[git-push] handoff size: $(du -sh "$HANDOFF_DIR" | cut -f1)"

# Bail if individual file > 100 MB (GitHub hard limit). 50 MB shows warning but accepts.
large=$(find "$HANDOFF_DIR" -type f -size +100M)
if [[ -n "$large" ]]; then
    echo "[git-push] ERROR: file > 100 MB (GitHub limit):" >&2
    echo "$large" >&2
    exit 1
fi

# Stay on the current branch; warn if it diverged from origin
git fetch origin "$BRANCH" 2>&1 | tail -3 || true

git add "$HANDOFF_DIR" "$STATUS_FILE" 2>&1 | tail -3

if git diff --cached --quiet; then
    echo "[git-push] no changes staged — nothing to commit."
    exit 0
fi

COMMIT_MSG="Handoff: GRPO + private K=8 SC results

Pipeline complete:
- SFT (val_225 baseline in handoff/val225_sft.jsonl)
- GRPO 3-epoch, K=4 + hard-dup=1 (ckpt sweep in handoff/grpo_ckpt_sweep_summary.json)
- Private K=8 SC inference (handoff/private_sc_k8_lite.jsonl + .gz)
- Kaggle CSV: handoff/private_submission.csv
- HF Hub: JaasonYuu/jason-cse151b-model
"
git commit -m "$COMMIT_MSG" 2>&1 | tail -5

echo "[git-push] pushing to origin/$BRANCH ..."
git push origin "HEAD:$BRANCH" 2>&1 | tail -5

echo "[git-push] done. Commit:"
git log -1 --oneline
