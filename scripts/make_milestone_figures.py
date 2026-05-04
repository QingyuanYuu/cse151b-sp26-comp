"""Generate all figures for the milestone report.

Reads:
  - reports/baseline_v0_val.json  (Phase 0 = starter prompts on val 225)
  - reports/baseline_v1_val.json  (Phase 1 = new prompts on val 225)

Writes (both PDF + PNG):
  - templates/milestone-report/figures/fig_acc_by_type.pdf
  - templates/milestone-report/figures/fig_acc_by_length.pdf
  - templates/milestone-report/figures/fig_acc_by_topic.pdf
  - templates/milestone-report/figures/fig_failure_modes.pdf
  - templates/milestone-report/figures/fig_phase_compare.pdf
"""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np


REPO = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "templates" / "milestone-report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Style — clean academic look
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

PHASE0_COLOR = "#888888"   # neutral gray
PHASE1_COLOR = "#1f77b4"   # blue


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=200)
    print(f"  → {FIG_DIR / name}.{{pdf,png}}")


# ─── Load data ──────────────────────────────────────────────────────────────

p0 = json.loads((REPO / "reports/baseline_v0_val.json").read_text())
p1 = json.loads((REPO / "reports/baseline_v1_val.json").read_text())


# ─── Figure 1: accuracy by question type ────────────────────────────────────


def fig_by_type() -> None:
    types = ["mcq", "free_single", "free_multi"]
    labels = ["MCQ\n(n=75)", "Free-form single\n(n=67)", "Free-form multi\n(n=83)"]
    p0_acc = [p0["by_type"][t]["acc"] * 100 for t in types]
    p1_acc = [p1["by_type"][t]["acc"] * 100 for t in types]

    x = np.arange(len(types))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w / 2, p0_acc, w, label="Phase 0 (starter prompts)", color=PHASE0_COLOR)
    ax.bar(x + w / 2, p1_acc, w, label="Phase 1 (our prompts)", color=PHASE1_COLOR)

    for i, (a, b) in enumerate(zip(p0_acc, p1_acc)):
        delta = b - a
        ax.text(i, max(a, b) + 2, f"{delta:+.1f}", ha="center", fontsize=9,
                color="green" if delta > 0 else "red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by question type (validation set, n=225)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_acc_by_type")
    plt.close(fig)


# ─── Figure 2: accuracy by question length ──────────────────────────────────


def fig_by_length() -> None:
    buckets = ["short(<150)", "medium(150-500)", "long(500-1500)", "vlong(>=1500)"]
    bucket_names = ["short\n<150 chars", "medium\n150–500", "long\n500–1500", "vlong\n≥1500"]
    p0_acc = [p0["by_length"].get(b, {"acc": 0})["acc"] * 100 for b in buckets]
    p1_acc = [p1["by_length"].get(b, {"acc": 0})["acc"] * 100 for b in buckets]
    n_per = [p1["by_length"].get(b, {"n": 0})["n"] for b in buckets]

    labels = [f"{name}\n(n={n})" for name, n in zip(bucket_names, n_per)]

    x = np.arange(len(buckets))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, p0_acc, w, label="Phase 0", color=PHASE0_COLOR)
    ax.bar(x + w / 2, p1_acc, w, label="Phase 1", color=PHASE1_COLOR)

    for i, (a, b) in enumerate(zip(p0_acc, p1_acc)):
        delta = b - a
        ax.text(i, max(a, b) + 2, f"{delta:+.1f}", ha="center", fontsize=9,
                color="green" if delta > 0 else "red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy degrades on long questions (validation, n=225)")
    ax.set_ylim(0, 95)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_acc_by_length")
    plt.close(fig)


# ─── Figure 3: accuracy by topic ────────────────────────────────────────────


def fig_by_topic() -> None:
    # Show the 6 most populous topics, sorted by Phase-1 accuracy
    topics_data = []
    for t in p1["by_topic"]:
        if p1["by_topic"][t]["n"] < 5:
            continue
        topics_data.append({
            "topic": t,
            "n": p1["by_topic"][t]["n"],
            "p0": p0["by_topic"].get(t, {"acc": 0})["acc"] * 100,
            "p1": p1["by_topic"][t]["acc"] * 100,
        })
    topics_data.sort(key=lambda d: -d["p1"])

    x = np.arange(len(topics_data))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, [d["p0"] for d in topics_data], w, label="Phase 0", color=PHASE0_COLOR)
    ax.bar(x + w / 2, [d["p1"] for d in topics_data], w, label="Phase 1", color=PHASE1_COLOR)

    for i, d in enumerate(topics_data):
        delta = d["p1"] - d["p0"]
        ax.text(i, max(d["p0"], d["p1"]) + 2, f"{delta:+.1f}", ha="center", fontsize=9,
                color="green" if delta > 0 else "red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d['topic']}\n(n={d['n']})" for d in topics_data])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by topic (validation, n=225 — topic via regex tagger)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_acc_by_topic")
    plt.close(fig)


# ─── Figure 4: failure modes (Phase 1 on full public 1126) ──────────────────


def fig_failure_modes() -> None:
    # Use the FULL Phase 1 stats here, not just val (more dramatic narrative)
    full = json.loads((REPO / "reports/baseline_public_v1.json").read_text())
    fb = full["failure_breakdown"]
    n_total = full["n_total"]
    n_correct = full["n_correct"]
    n_wrong = n_total - n_correct

    modes = [
        ("Truncated\n(hit MAX_TOKENS)", fb["truncated"], "#d62728"),
        ("No \\boxed{}", fb["no_box"], "#ff7f0e"),
        ("Wrong shape\n(multi-part)", fb["wrong_shape"], "#9467bd"),
        ("Wrong answer\n(reasoning error)", fb["numeric_but_wrong"], "#1f77b4"),
    ]
    labels, counts, colors = zip(*modes)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, width=0.6)

    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * 0.02,
                f"{c}\n({c/n_wrong*100:.1f}% of wrong)",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title(f"Failure mode breakdown on Phase 1 (full public, "
                 f"{n_wrong} wrong / {n_total} total)")
    ax.set_ylim(0, max(counts) * 1.35)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig_failure_modes")
    plt.close(fig)


# ─── Figure 5: Phase 0 vs Phase 1 confusion ────────────────────────────────


def fig_phase_compare() -> None:
    # From the compare result
    counts = {
        "both_right": 119,
        "only_baseline_right (regression)": 8,
        "only_candidate_right (gain)": 22,
        "both_wrong": 76,
    }
    matrix = np.array([[counts["both_right"], counts["only_baseline_right (regression)"]],
                       [counts["only_candidate_right (gain)"], counts["both_wrong"]]])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())

    cell_labels = [
        ["both right\n119", "Phase 0 only\n(regression: 8)"],
        ["Phase 1 only\n(gain: 22)", "both wrong\n76"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cell_labels[i][j], ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Phase 0\nright", "Phase 0\nwrong"])
    ax.set_yticklabels(["Phase 1\nright", "Phase 1\nwrong"])
    ax.set_title("Phase 0 vs Phase 1 outcomes on val (n=225)\n"
                 "Net gain: +14 questions (22 gain − 8 regression)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "fig_phase_compare")
    plt.close(fig)


if __name__ == "__main__":
    print("Saving figures to:", FIG_DIR)
    fig_by_type()
    fig_by_length()
    fig_by_topic()
    fig_failure_modes()
    fig_phase_compare()
    print("Done.")
