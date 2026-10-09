#!/usr/bin/env python3
"""
M22 figures — classifier error as the interface would experience it.

Two panels, because there are two separate decisions to support:

  left   does the number on screen track reality, and does correcting it help?
         Colour carries identity here (naive vs corrected), so it uses two
         slots of a CVD-validated categorical order, each directly labelled
         and additionally separated by line style — identity never rests on
         colour alone. A 1:1 reference line gives the eye the thing both
         series are being judged against.

  right  which model should actually be deployed? Magnitude only, so a single
         hue, with the recommended model in the accent.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path(__file__).resolve().parent / "artifacts"

PAPER, INK, MUTED, GRID = "#F2F0EB", "#16181D", "#6B6F76", "#D9D5CC"
ACC, BAR = "#C9502A", "#9AA0A6"
NAIVE, CORRECTED = "#eb6834", "#2a78d6"   # validated categorical slots

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

R = json.loads((ART / "reliability.json").read_text())
B = json.loads((ART / "benchmark.json").read_text())

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6))

# ---------------------------------------------------------------- left panel
t = R["attenuation"]["true"]
nv = R["attenuation"]["naive"]
cr = R["attenuation"]["corrected"]

axL.plot([0, 1], [0, 1], ls=(0, (2, 3)), lw=1.3, color=MUTED, zorder=2)
axL.annotate("perfect display", (0.52, 0.52), xytext=(9, -15), textcoords="offset points",
             ha="left", va="top", fontsize=8.6, color=MUTED, style="italic", rotation=36)

axL.plot(t, nv, "-o", lw=2, ms=7, color=NAIVE, markeredgecolor=PAPER,
         markeredgewidth=1.6, zorder=4, label="as built today")
axL.plot(t, cr, "--s", lw=2, ms=7, color=CORRECTED, markeredgecolor=PAPER,
         markeredgewidth=1.6, zorder=4, label="with bias correction")

axL.annotate("as built today", (t[1], nv[1]), xytext=(6, -20), textcoords="offset points",
             fontsize=9.5, color=NAIVE, fontweight="bold")
axL.annotate("with bias correction", (t[2], cr[2]), xytext=(8, -22),
             textcoords="offset points", fontsize=9.5, color=CORRECTED, fontweight="bold")

axL.set_xlim(0.03, 0.97)
axL.set_ylim(0.03, 0.97)
axL.set_xlabel("the client's true change-talk ratio")
axL.set_ylabel("what the website would display")
axL.grid(True, color=GRID, lw=0.8, zorder=0)
axL.set_axisbelow(True)
axL.set_title(
    f"A client who changes a lot and one who barely changes\n"
    f"look almost identical on screen   ·   slope {R['attenuation']['slope_naive']:.2f} "
    f"→ {R['attenuation']['slope_corrected']:.2f} once corrected",
    loc="left", pad=10,
)

# --------------------------------------------------------------- right panel
order = [m for m in B["model_order"] if m != "majority_class"]
rec = {m: B["tasks"]["client"]["models"][m]["per_class_recall"]["sustain"] for m in order}
ranked = sorted(order, key=lambda m: rec[m])
best = max(order, key=lambda m: rec[m])
vals = [rec[m] for m in ranked]
y = np.arange(len(ranked))
colors = [ACC if m == best else BAR for m in ranked]

axR.barh(y, vals, height=0.62, color=colors, zorder=3)
for yi, v in enumerate(vals):
    axR.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=9.5,
             color=INK, fontweight="bold")

sel = B["tasks"]["client"]["selected_model"]
labels = []
for m in ranked:
    lab = B["display_names"][m]
    if m == best:
        lab += "  ◂ deploy this"
    if m == sel:
        lab += "  (macro-F1 pick)"
    labels.append(lab)
axR.set_yticks(y, labels, fontsize=9.5)
for tick, m in zip(axR.get_yticklabels(), ranked):
    if m == best:
        tick.set_color(ACC)
        tick.set_fontweight("bold")

axR.set_xlim(0, max(vals) * 1.22)
axR.set_xlabel("sustain-talk recall on held-out conversations")
axR.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
axR.set_axisbelow(True)
axR.set_title(
    "Sustain talk is what the display gets wrong,\n"
    "and macro-F1 does not select for catching it",
    loc="left", pad=10,
)

fig.suptitle(
    "What classifier error does to the number on the screen",
    x=0.008, ha="left", fontsize=15, fontweight="bold",
)
fig.text(
    0.008, 0.015,
    "Simulated from each model's held-out confusion matrix: 20,000 sessions of 40 directional client turns. "
    "Missed sustain talk is read as change talk, so the displayed ratio is inflated by a margin that does not\n"
    "shrink with longer sessions — it is systematic, not random. Correcting it with the confusion matrix "
    "(adjusted classify-and-count) restores most of the real variation.",
    fontsize=8.5, color=MUTED,
)
fig.tight_layout(rect=[0, 0.075, 1, 0.93])
fig.savefig(ART / "bench_reliability.png")
print(f"wrote {ART / 'bench_reliability.png'}")
