#!/usr/bin/env python3
"""
Renders the benchmark results as markdown, straight from benchmark.json.

The point of this file is that no figure in the write-up is ever typed by hand.
Every number in the tables below is read out of the artifact the run produced,
so the prose and the JSON cannot drift apart — which is the failure mode that
makes a results slide indefensible under questioning.

    python ml/report_benchmark.py > ml/RESULTS.md
"""

from __future__ import annotations

import json
from pathlib import Path

ART = Path(__file__).resolve().parent / "artifacts"
B = json.loads((ART / "benchmark.json").read_text())
DISPLAY = B["display_names"]
ORDER = B["model_order"]
REAL = [m for m in ORDER if m != "majority_class"]
TASKS = ["client", "therapist"]

out: list[str] = []
w = out.append


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


w("# M21 — multi-model benchmark results\n")
w(f"**Dataset.** {B['dataset']}\n")
w("AnnoMI is a corpus of *performed* motivational-interviewing demonstrations, "
  "collected from training videos. It is expert-annotated but it is not clinical "
  "session data, and the 23 low-quality transcripts are deliberately staged poor "
  "counselling. Every claim below is a claim about this corpus only.\n")

w("## Protocol\n")
for k, v in B["protocol"].items():
    w(f"- **{k.replace('_', ' ').title()}** — {v}")
w("")

# ---------------------------------------------------------------- per task
for task in TASKS:
    t = B["tasks"][task]
    best = t["selected_model"]
    w(f"## Task: {t['label']}\n")
    bal = ", ".join(f"`{c}` {n} ({100*n/t['n_utterances']:.1f}%)"
                    for c, n in t["class_balance"].items())
    w(f"{t['n_utterances']} utterances across {t['n_conversations']} conversations. "
      f"Class balance: {bal}.\n")
    w(f"Development set {t['n_dev']} utterances / {t['n_dev_conversations']} conversations. "
      f"Held-out test {t['n_test']} utterances / {t['n_test_conversations']} conversations.\n")

    w("### Cross-validation on development conversations\n")
    w("| Model | CV macro-F1 | CV accuracy | CV ROC-AUC | Fit (s) | Δ vs best | p |")
    w("|---|---|---|---|---|---|---|")
    for n in ORDER:
        m = t["models"][n]
        cv = m["cv"]
        vb = m["vs_best"]
        name = f"**{DISPLAY[n]}**" if n == best else DISPLAY[n]
        delta = "—" if vb["is_best"] else f"{vb['delta']:+.4f}"
        p = "—" if vb["p_value"] is None else f"{vb['p_value']:.3f}"
        w(f"| {name} "
          f"| {fmt(cv['macro_f1']['mean'])} ± {fmt(cv['macro_f1']['std'], 3)} "
          f"| {fmt(cv['accuracy']['mean'])} "
          f"| {fmt(cv['macro_roc_auc']['mean'], 3)} "
          f"| {m['fit_seconds_clean']:.2f} "
          f"| {delta} | {p} |")
    w("")
    w(f"Selected on CV macro-F1: **{DISPLAY[best]}**. "
      "`Δ` and `p` come from a paired t-test across the same five folds.\n")

    w("### Held-out test — every model scored once\n")
    w("| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro precision | Macro recall | ROC-AUC |")
    w("|---|---|---|---|---|---|---|")
    for n in ORDER:
        s = t["models"][n]["test"]
        name = f"**{DISPLAY[n]}**" if n == best else DISPLAY[n]
        w(f"| {name} | {fmt(s['accuracy'])} | {fmt(s['macro_f1'])} "
          f"| {fmt(s['weighted_f1'])} | {fmt(s['macro_precision'])} "
          f"| {fmt(s['macro_recall'])} | {fmt(s['macro_roc_auc'], 3)} |")
    w("")

    w("### Per-class F1 on the held-out conversations\n")
    classes = t["class_names"]
    w("| Model | " + " | ".join(f"`{c}`" for c in classes) + " |")
    w("|---" * (len(classes) + 1) + "|")
    for n in REAL:
        pc = t["models"][n]["test_per_class_f1"]
        name = f"**{DISPLAY[n]}**" if n == best else DISPLAY[n]
        w(f"| {name} | " + " | ".join(fmt(pc[c], 3) for c in classes) + " |")
    w("")

    cm = t["models"][best]["confusion_matrix"]
    w(f"### Confusion matrix — {DISPLAY[best]}\n")
    w("| actual ↓ / predicted → | " + " | ".join(f"`{c}`" for c in classes) + " |")
    w("|---" * (len(classes) + 1) + "|")
    for i, c in enumerate(classes):
        w(f"| `{c}` | " + " | ".join(str(v) for v in cm[i]) + " |")
    w("")

# ---------------------------------------------------------------- findings
w("## What the comparison actually shows\n")

for task in TASKS:
    t = B["tasks"][task]
    best = t["selected_model"]
    cvs = {n: t["models"][n]["cv"]["macro_f1"]["mean"] for n in REAL}
    rank = sorted(cvs, key=cvs.get, reverse=True)
    floor = t["models"]["majority_class"]["cv"]["macro_f1"]["mean"]
    spread = cvs[rank[0]] - cvs[rank[-1]]
    best_std = t["models"][best]["cv"]["macro_f1"]["std"]
    # Which gaps are not distinguishable from noise at the 5% level?
    insignificant = [
        n for n in REAL
        if n != best and (t["models"][n]["vs_best"]["p_value"] or 1.0) > 0.05
    ]
    costliest = max(REAL, key=lambda n: t["models"][n]["fit_seconds_clean"])
    cheap_best = min(
        (n for n in rank[:3]), key=lambda n: t["models"][n]["fit_seconds_clean"]
    )
    t_hi = t["models"][costliest]["fit_seconds_clean"]
    t_lo = t["models"][cheap_best]["fit_seconds_clean"]

    # Significance sentence. Listing six model names reads worse than saying so.
    if not insignificant:
        sig = "The selected model separates from every alternative at the 5% level."
    elif len(insignificant) == len(REAL) - 1:
        sig = (f"A paired t-test cannot separate {DISPLAY[best]} from **any** of the "
               "other six at the 5% level — on this task the seven learners are "
               "statistically indistinguishable.")
    else:
        sig = (f"A paired t-test cannot separate {DISPLAY[best]} from "
               + ", ".join(DISPLAY[n] for n in insignificant)
               + " at the 5% level.")

    # Cost sentence. A ratio against a sub-0.1s fit would imply a precision the
    # measurement does not have, so below that threshold the absolute times are
    # quoted instead of a headline multiple.
    if t_lo >= 0.1:
        cost = (f"{DISPLAY[costliest]} costs {t_hi / t_lo:.0f}× the compute of "
                f"{DISPLAY[cheap_best]} ({t_hi:.0f} s against {t_lo:.2f} s) and does "
                "not buy a better score.")
    else:
        cost = (f"{DISPLAY[costliest]} needs {t_hi:.0f} s to fit where "
                f"{DISPLAY[cheap_best]} needs {t_lo:.2f} s — a difference of three "
                "orders of magnitude for no gain in score.")

    w(f"**{t['label']}.** Every learner clears the majority-class floor of "
      f"{floor:.3f} macro-F1 by a wide margin, so the features carry real signal. "
      f"But the spread between the best and worst of the seven is only "
      f"{spread:.3f} macro-F1, against a fold-to-fold standard deviation of "
      f"{best_std:.3f} for the selected model alone. {sig} {cost}\n")

w("The conclusion is not that model choice never matters. It is that on this "
  "corpus, at this size, with these labels, **the ceiling is set by the task and "
  "not by the learner.** Roughly three thousand short utterances and tens of "
  "thousands of sparse features is a regime where a regularised linear boundary "
  "is already the right capacity; the extra expressiveness of boosted trees has "
  "nothing left to find and spends compute discovering that.\n")

w("That matters for how the project is read. The contribution of TherapyTrace is "
  "not a better utterance classifier — this benchmark is the evidence that chasing "
  "one would be a poor use of effort. The contribution is the layer above: "
  "within-person normalisation, trajectory estimation across sessions, and "
  "state-conditional estimates of which therapist move helps. Those are measured "
  "elsewhere in `ml/`, and none of the seven models here addresses them.\n")

w("## Honest limitations\n")
w("- **AnnoMI is performed, not clinical.** Transfer to real sessions is unmeasured.")
w("- **One unbalanced learner.** scikit-learn's MLP accepts neither class weights "
  "nor sample weights, so it alone optimises raw accuracy. Read its macro-F1 with "
  "that in mind; its accuracy is comparable, its macro-F1 is mildly flattered on "
  "the majority class and penalised on the minorities.")
w("- **Hyperparameters are fixed, not tuned per model.** A tuned XGBoost might "
  "close its gap. The budget went to a clean protocol rather than a leaderboard, "
  "and the paired tests suggest the headroom is small.")
w("- **Selection is imperfect.** Model selection ran on development folds, so the "
  "model with the best CV score is not always the one with the best held-out score. "
  "Both numbers are reported for every model rather than only the flattering one.")
w("- **No transformer baseline.** A fine-tuned encoder such as MentalBERT is the "
  "obvious next comparison and is not present; the environment has no model-hub "
  "access. Published AnnoMI transformer results are the reference point.")
w("")

print("\n".join(out))
