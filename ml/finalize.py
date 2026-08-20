#!/usr/bin/env python3
"""
Fit the selected models on all available data, persist them, and render the
figures used in the review deck.

Run `train.py` first — this script reads its results.json to know which
configuration won each task.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train import OUT, build_pipelines, load_task  # noqa: E402

INK, GAIN, REGRESS, MARK = "#14201F", "#1F6F5C", "#8E2C4E", "#B47C14"
PAPER, RULE, INK3 = "#F7F8F5", "#C7CDC4", "#6B7C82"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": PAPER,
    "axes.facecolor": "white",
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK3,
    "ytick.color": INK3,
    "axes.grid": True,
    "grid.color": RULE,
    "grid.alpha": 0.5,
    "font.size": 9,
})


def confusion_figure(cm, labels, title, path, accent=GAIN):
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(norm, cmap="BuGn" if accent == GAIN else "RdPu", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=12)
    ax.grid(False)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.0%}",
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else INK)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row-normalised", fontsize=8)
    cbar.outline.set_edgecolor(RULE)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def ablation_figure(tasks, path):
    order = [
        ("majority", "Majority baseline", INK3),
        ("lexicon_only", "Lexicon only (10 features)", MARK),
        ("tfidf_only", "TF-IDF only", INK3),
        ("hybrid", "TF-IDF + lexicon", GAIN),
        ("hybrid_plus_context", "+ previous-turn context", REGRESS),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, (key, t) in zip(axes, tasks.items()):
        ab = t["ablation"]
        names = [n for _, n, _ in order]
        vals = [ab[k] for k, _, _ in order]
        colors = [c for _, _, c in order]
        bars = ax.barh(names, vals, color=colors, edgecolor=INK, linewidth=0.8, height=0.6)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.28)
        ax.set_xlabel("Cross-validated macro-F1")
        ax.set_title(t["label"], fontsize=10, fontweight="bold")
        ax.grid(axis="y", visible=False)
        for b, v in zip(bars, vals):
            ax.text(v + max(vals) * 0.02, b.get_y() + b.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")
    fig.suptitle("Feature ablation — grouped by conversation, 5-fold CV",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    results = json.loads((OUT / "results.json").read_text())

    for task_key, t in results["tasks"].items():
        X, y, groups = load_task(task_key)
        best_name = t["selected_model"]
        print(f"[{task_key}] fitting {best_name} on all {len(X)} utterances...")
        model = build_pipelines()[best_name].fit(X, y)

        joblib.dump(
            {"model": model, "task": task_key, "classes": sorted(y.unique()),
             "selected": best_name, "metrics": {
                 "test_accuracy": t["test_accuracy"],
                 "test_macro_f1": t["test_macro_f1"]}},
            OUT / f"model_{task_key}.joblib",
        )

        cm = t["confusion_matrix"]
        confusion_figure(
            cm["matrix"], cm["labels"],
            f"{t['label']} — held-out conversations\n"
            f"accuracy {t['test_accuracy']:.2f} · macro-F1 {t['test_macro_f1']:.2f}",
            OUT / f"confusion_{task_key}.png",
            accent=GAIN if task_key == "therapist" else REGRESS,
        )
        print(f"[{task_key}] saved model + confusion matrix")

    ablation_figure(results["tasks"], OUT / "ablation.png")
    print("wrote ablation.png")


if __name__ == "__main__":
    main()
