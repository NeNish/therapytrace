#!/usr/bin/env python3
"""
TherapyTrace — figures for the M21 multi-model benchmark.

Reads ml/artifacts/benchmark.json and writes six PNGs sized for slides.

Design notes, so the choices are not mistaken for defaults:

  * Magnitude panels (model comparison, cost, folds) use a single hue. Colour
    there carries no identity, so a second hue would be decoration. The
    selected model is picked out in the accent; everything else is recessive
    ink. The majority-class floor is drawn as a dashed rule rather than a bar,
    because it is a reference level, not a competitor.

  * Only the per-class panel needs categorical colour, because there colour
    does carry identity (which class). It uses the first four slots of a
    CVD-validated categorical order. Those slots sit below 3:1 contrast on this
    surface, so every bar is directly labelled — identity never rests on colour
    alone.

  * Heatmaps and confusion matrices use one hue, light to dark, with the value
    printed in every cell. No rainbow, no diverging scale: these quantities have
    a floor at zero and no meaningful midpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ART = Path(__file__).resolve().parent / "artifacts"
DATA = ART / "benchmark.json"

# Deck palette, so the figures sit inside the Review III slides without clashing.
PAPER = "#F2F0EB"
INK = "#16181D"
MUTED = "#6B6F76"
GRID = "#D9D5CC"
ACC = "#C9502A"          # accent: the selected model
VER = "#2E7D5B"          # sequential hue
BAR = "#9AA0A6"          # recessive bar fill

# First four slots of a CVD-validated categorical order (adjacent pairs pass
# both the CVD and normal-vision floors on this surface).
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

SEQ = LinearSegmentedColormap.from_list("ver", ["#FBFAF7", "#CFE0D7", "#8FBCA7", "#4C9478", VER, "#1B4F39"])

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

B = json.loads(DATA.read_text())
TASKS = ["client", "therapist"]
ORDER = B["model_order"]
DISPLAY = B["display_names"]
REAL = [m for m in ORDER if m != "majority_class"]


def panel_title(ax, task: str, pad: int = 10) -> None:
    """Two lines, so the right-hand panel is never clipped by the figure edge."""
    t = B["tasks"][task]
    ax.set_title(
        f"{t['label']}  ·  {len(t['class_names'])} classes\n"
        f"{t['n_test']:,} test utterances · {t['n_test_conversations']} unseen conversations",
        loc="left",
        pad=pad,
        fontsize=11,
    )


# ---------------------------------------------------------------------------
# 1. model comparison — CV macro-F1 with fold spread
# ---------------------------------------------------------------------------

def fig_models() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, task in zip(axes, TASKS):
        t = B["tasks"][task]
        best = t["selected_model"]
        names = REAL[::-1]
        means = [t["models"][n]["cv"]["macro_f1"]["mean"] for n in names]
        stds = [t["models"][n]["cv"]["macro_f1"]["std"] for n in names]
        colors = [ACC if n == best else BAR for n in names]
        y = np.arange(len(names))

        ax.barh(y, means, height=0.6, color=colors, zorder=3)
        ax.errorbar(
            means, y, xerr=stds, fmt="none", ecolor=INK, elinewidth=1.4,
            capsize=4, capthick=1.4, alpha=0.75, zorder=4,
        )
        floor = t["models"]["majority_class"]["cv"]["macro_f1"]["mean"]
        ax.axvline(floor, color=INK, ls="--", lw=1.2, zorder=2)
        ax.text(
            floor, len(names) - 0.3, f"  majority class {floor:.2f}",
            va="center", ha="left", fontsize=8.5, color=INK, style="italic",
        )

        for yi, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + s + 0.012, yi, f"{m:.3f}", va="center", fontsize=9.5,
                    color=INK, fontweight="bold")

        ax.set_yticks(y, [DISPLAY[n] for n in names])
        ax.set_xlim(0, max(means) + max(stds) + 0.11)
        ax.set_xlabel("macro-F1, 5-fold conversation-grouped CV  (± 1 SD across folds)")
        ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        panel_title(ax, task)

    fig.suptitle(
        "Seven classifiers, one identical feature matrix",
        x=0.008, ha="left", fontsize=15, fontweight="bold",
    )
    fig.text(
        0.008, 0.015,
        "Every model sees TF-IDF word 1–2 grams + character 3–5 grams + 12 lexicon features. "
        "Differences are therefore attributable to the learner. Splits are grouped by conversation.",
        fontsize=8.5, color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.93])
    fig.savefig(ART / "bench_models.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. held-out metric table as a heatmap
# ---------------------------------------------------------------------------

METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("weighted_f1", "Weighted-F1"),
    ("macro_precision", "Macro\nprecision"),
    ("macro_recall", "Macro\nrecall"),
    ("macro_roc_auc", "ROC-AUC\n(OvR)"),
]


def fig_metrics() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, task in zip(axes, TASKS):
        t = B["tasks"][task]
        best = t["selected_model"]
        M = np.array(
            [[(t["models"][n]["test"][k] or np.nan) for k, _ in METRICS] for n in ORDER],
            dtype=float,
        )
        # Shade across the range the real models actually occupy. A fixed 0-1
        # ramp would push every value into one or two indistinguishable steps,
        # because nothing here scores below 0.4 or above 0.9. The majority-class
        # row is excluded from the range so its near-zero scores do not stretch
        # the scale and flatten the comparison that matters.
        real = M[1:, :]
        lo = float(np.nanmin(real))
        hi = float(np.nanmax(real))
        ax.imshow(M, cmap=SEQ, vmin=lo, vmax=hi, aspect="auto")

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=8.5, color=MUTED)
                    continue
                shade = (v - lo) / (hi - lo) if hi > lo else 0.5
                ax.text(
                    j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="#FFFFFF" if shade > 0.55 else INK,
                    fontweight="bold" if ORDER[i] == best else "normal",
                )

        ax.set_xticks(range(len(METRICS)), [lbl for _, lbl in METRICS], fontsize=9)
        labels = [
            (DISPLAY[n] + "  ◂ selected") if n == best else DISPLAY[n] for n in ORDER
        ]
        ax.set_yticks(range(len(ORDER)), labels, fontsize=9.5)
        for tick, n in zip(ax.get_yticklabels(), ORDER):
            if n == best:
                tick.set_color(ACC)
                tick.set_fontweight("bold")
        ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ORDER), 1), minor=True)
        ax.grid(which="minor", color=PAPER, lw=2.0)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        panel_title(ax, task)

    fig.suptitle(
        "Held-out results — every model scored once on conversations it never saw",
        x=0.008, ha="left", fontsize=15, fontweight="bold",
    )
    fig.text(
        0.008, 0.015,
        "Selection happened on development conversations only; the test conversations were scored a single time, for all models,\n"
        "so the runner-up's number is shown rather than hidden. Shading is scaled within each panel — compare numbers, not colours, across panels.",
        fontsize=8.5, color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.065, 1, 0.93])
    fig.savefig(ART / "bench_metrics.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. per-class F1 — where every model struggles
# ---------------------------------------------------------------------------

def fig_per_class() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, task in zip(axes, TASKS):
        t = B["tasks"][task]
        classes = t["class_names"]
        x = np.arange(len(REAL))
        w = 0.8 / len(classes)
        for ci, c in enumerate(classes):
            vals = [t["models"][n]["test_per_class_f1"][c] for n in REAL]
            pos = x - 0.4 + w * (ci + 0.5)
            ax.bar(pos, vals, width=w * 0.88, color=CLASS_COLORS[ci], label=c, zorder=3)
            for p, v in zip(pos, vals):
                ax.text(p, v + 0.014, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7.4, color=INK, rotation=90)

        ax.set_xticks(x, [DISPLAY[n].replace(" (128 hidden)", "") for n in REAL],
                      rotation=28, ha="right", fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("F1 on held-out conversations")
        ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        # The legend sits above the plotting area; inside it would land on the
        # rotated value labels of the tallest bars.
        ax.legend(
            title="class", frameon=False, fontsize=9, title_fontsize=9,
            ncol=len(classes), loc="lower center", bbox_to_anchor=(0.5, 1.005),
            handlelength=1.1, columnspacing=1.4,
        )
        panel_title(ax, task, pad=44)

    fig.suptitle(
        "Per-class F1 — the minority classes are the hard part, for every learner",
        x=0.008, ha="left", fontsize=15, fontweight="bold",
    )
    fig.text(
        0.008, 0.015,
        "Each bar is directly labelled, so class identity never depends on colour alone.",
        fontsize=8.5, color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.93])
    fig.savefig(ART / "bench_perclass.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. confusion matrices for the selected model
# ---------------------------------------------------------------------------

def fig_confusion() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    for ax, task in zip(axes, TASKS):
        t = B["tasks"][task]
        best = t["selected_model"]
        cm = np.array(t["models"][best]["confusion_matrix"], dtype=float)
        classes = t["class_names"]
        row = cm / cm.sum(axis=1, keepdims=True)

        ax.imshow(row, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j, i, f"{row[i, j]*100:.0f}%\n{int(cm[i, j])}",
                    ha="center", va="center", fontsize=9.5,
                    color="#FFFFFF" if row[i, j] > 0.55 else INK,
                    fontweight="bold" if i == j else "normal",
                )
        ax.set_xticks(range(len(classes)), classes, fontsize=9.5)
        ax.set_yticks(range(len(classes)), classes, fontsize=9.5)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
        ax.grid(which="minor", color=PAPER, lw=2.0)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(
            f"{t['label']}  ·  {DISPLAY[best]}\n"
            f"accuracy {t['models'][best]['test']['accuracy']:.3f}   "
            f"macro-F1 {t['models'][best]['test']['macro_f1']:.3f}",
            loc="left", pad=10,
        )

    fig.suptitle(
        "Where the selected model actually goes wrong",
        x=0.008, ha="left", fontsize=15, fontweight="bold",
    )
    fig.text(
        0.008, 0.015,
        "Rows are normalised to the true class; the count is printed underneath each percentage.",
        fontsize=8.5, color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.93])
    fig.savefig(ART / "bench_confusion.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. accuracy against training cost
# ---------------------------------------------------------------------------

def fig_cost() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, task in zip(axes, TASKS):
        t = B["tasks"][task]
        best = t["selected_model"]
        xs = [t["models"][n]["fit_seconds_clean"] for n in REAL]
        ys = [t["models"][n]["cv"]["macro_f1"]["mean"] for n in REAL]
        ax.set_xscale("log")

        # Labels are placed by hand rather than by matplotlib's default offset,
        # because several learners land close together in both axes and the
        # straight "13 points above the dot" rule collides. Points are walked in
        # x order; a label flips below the dot whenever the previous label is
        # near enough in normalised space to overlap it.
        lxs = np.log10(np.array(xs))
        span_x = max(lxs.max() - lxs.min(), 1e-6)
        span_y = max(max(ys) - min(ys), 1e-6)
        order = np.argsort(lxs)
        placed: list[tuple[float, float, int]] = []
        for idx in order:
            n, x, y = REAL[idx], xs[idx], ys[idx]
            nx, ny = (lxs[idx] - lxs.min()) / span_x, (y - min(ys)) / span_y
            above = True
            for px, py, pdir in placed:
                if abs(px - nx) < 0.18 and abs(py - ny) < 0.16 and pdir == 1:
                    above = False
                    break
            placed.append((nx, ny, 1 if above else -1))
            sel = n == best
            ax.scatter(
                x, y, s=190 if sel else 110,
                color=ACC if sel else BAR,
                edgecolor=PAPER, linewidth=2.0, zorder=4,
            )
            ax.annotate(
                DISPLAY[n].replace(" (128 hidden)", ""),
                (x, y), textcoords="offset points",
                xytext=(0, 14 if above else -22),
                ha="center", fontsize=9, zorder=5,
                color=ACC if sel else INK,
                fontweight="bold" if sel else "normal",
            )
        # Room at both ends so the outermost labels are not cut by the spine.
        ax.set_xlim(10 ** (lxs.min() - 0.55 * span_x), 10 ** (lxs.max() + 0.55 * span_x))
        ax.set_xlabel("classifier fit time, seconds on one fold, idle machine (log scale)")
        ax.set_ylabel("CV macro-F1")
        lo, hi = min(ys), max(ys)
        pad = max(0.28 * (hi - lo), 0.02)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        panel_title(ax, task)

    fig.suptitle(
        "Cost against benefit — the cheapest learners are also the best here",
        x=0.008, ha="left", fontsize=15, fontweight="bold",
    )
    fig.text(
        0.008, 0.015,
        "Boosted trees cost one to two orders of magnitude more compute on this feature matrix and do not repay it. "
        "With 3,000-odd short utterances and tens of thousands of sparse features, a linear decision boundary is the right capacity.",
        fontsize=8.5, color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.93])
    fig.savefig(ART / "bench_cost.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. fold-level spread — are the gaps real?
# ---------------------------------------------------------------------------

def fig_folds() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    rng = np.random.default_rng(7)
    for ax, task in zip(axes, TASKS):
        t = B["tasks"][task]
        best = t["selected_model"]
        names = REAL[::-1]
        for yi, n in enumerate(names):
            folds = t["models"][n]["cv"]["macro_f1"]["folds"]
            sel = n == best
            jitter = rng.uniform(-0.13, 0.13, len(folds))
            ax.scatter(
                folds, np.full(len(folds), yi) + jitter, s=46,
                color=ACC if sel else BAR, alpha=0.9,
                edgecolor=PAPER, linewidth=1.2, zorder=3,
            )
            m = t["models"][n]["cv"]["macro_f1"]["mean"]
            ax.plot([m, m], [yi - 0.3, yi + 0.3], color=INK, lw=2.0, zorder=4)
            p = t["models"][n]["vs_best"]["p_value"]
            note = "selected" if sel else f"Δ {t['models'][n]['vs_best']['delta']:+.3f}   p = {p:.3f}"
            ax.text(
                ax.get_xlim()[1], yi, "", va="center",
            )
            ax.annotate(
                note, (1.0, yi), xycoords=("axes fraction", "data"),
                xytext=(-6, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8.6,
                color=ACC if sel else MUTED,
                fontweight="bold" if sel else "normal",
            )
        ax.set_yticks(range(len(names)), [DISPLAY[n] for n in names])
        ax.set_xlabel("macro-F1 of each individual CV fold")
        ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        lo = min(min(t["models"][n]["cv"]["macro_f1"]["folds"]) for n in REAL)
        hi = max(max(t["models"][n]["cv"]["macro_f1"]["folds"]) for n in REAL)
        span = hi - lo
        ax.set_xlim(lo - 0.04 * span, hi + 0.42 * span)
        panel_title(ax, task)

    fig.suptitle(
        "Fold-by-fold spread, and whether the gaps survive a paired test",
        x=0.008, ha="left", fontsize=15, fontweight="bold",
    )
    fig.text(
        0.008, 0.015,
        "Each dot is one fold of five; the vertical rule is the mean. Δ and p come from a paired t-test against the "
        "selected model across the same five folds. Fold-to-fold variation is large because folds differ by whole conversations.",
        fontsize=8.5, color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.93])
    fig.savefig(ART / "bench_folds.png")
    plt.close(fig)


def main() -> None:
    fig_models()
    fig_metrics()
    fig_per_class()
    fig_confusion()
    fig_cost()
    fig_folds()
    for p in sorted(ART.glob("bench_*.png")):
        print(f"wrote {p.name}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
