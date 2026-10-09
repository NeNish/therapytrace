#!/usr/bin/env python3
"""
M22 — how far does classifier error propagate into what the website shows?

"Macro-F1 is 0.46" is not an answer to "can we trust the screen". The number
the UI would actually display is not a turn label, it is the session-level
change-talk ratio

    ratio = change / (change + sustain)

aggregated over every directional client turn in a session. That is the
quantity the MI meta-analysis supports (Magill et al., 2018: proportion change
talk relates to reduced risk behaviour, r = -.16; raw change-talk counts do
not reach significance), and it is already computed in
`app/nlp/ml_models.classify_client_turns`.

A noisy per-turn classifier can still produce a usable session-level ratio,
because independent per-turn errors partly cancel in aggregate. "Partly" is
doing real work in that sentence, and the size of the residual error is an
empirical question, not a matter of opinion. This file measures it.

Method. The held-out confusion matrix from the M21 benchmark gives the
empirical P(predicted | true) for the selected model. For a session with n
directional client turns and a true change-talk ratio p, we:

    1. draw n true labels from the implied change/sustain mix,
    2. corrupt each one by sampling from its row of the confusion matrix,
    3. compute the ratio the website would display,
    4. repeat, and summarise bias and spread.

Two questions follow, and both matter for the interface:

    * How wrong is a single displayed number?
    * Can a real session-to-session change in a client be distinguished from
      classifier noise? That is what decides whether plotting this series over
      time is informative or is drawing trends in noise.

    python ml/reliability_study.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parent / "artifacts"
RNG = np.random.default_rng(20260101)
N_SIMS = 20000


def confusion_probabilities(model: str | None = None) -> tuple[np.ndarray, list[str], str]:
    """Row-normalised P(predicted | true) from the held-out benchmark."""
    b = json.loads((ART / "benchmark.json").read_text())
    task = b["tasks"]["client"]
    name = model or task["selected_model"]
    cm = np.array(task["models"][name]["confusion_matrix"], dtype=float)
    labels = task["class_names"]
    return cm / cm.sum(axis=1, keepdims=True), labels, name


def adjusted_count(pred_change: int, pred_sustain: int, P: np.ndarray, idx: dict) -> float:
    """
    Adjusted Classify & Count — de-bias the ratio using the confusion matrix.

    Counting predicted labels and dividing is a biased estimator of prevalence
    whenever the classifier is imperfect, and the bias does not vanish with more
    turns because it is systematic rather than random. The standard correction
    inverts the misclassification matrix: if the predicted distribution is
    q = Mᵀ p for true prevalence p, then p is recovered by solving that system
    and projecting back onto the simplex.

    This is the difference between a screen that reads 0.75 for a client who is
    evenly split, and one that reads roughly 0.5.
    """
    c, s, nt = idx["change"], idx["sustain"], idx["neutral"]
    total = pred_change + pred_sustain
    if total == 0:
        return float("nan")

    # Restrict to the directional two-class sub-problem the ratio is built from.
    # M[i][j] = P(predict j | true i), renormalised over the directional outputs.
    M = np.array(
        [
            [P[c, c], P[c, s]],
            [P[s, c], P[s, s]],
        ],
        dtype=float,
    )
    M = M / M.sum(axis=1, keepdims=True)

    q = np.array([pred_change / total, pred_sustain / total])
    try:
        p = np.linalg.solve(M.T, q)
    except np.linalg.LinAlgError:
        return float("nan")

    p = np.clip(p, 0.0, None)
    return float(p[0] / p.sum()) if p.sum() > 0 else float("nan")


def simulate(p_true: float, n_turns: int, P: np.ndarray, idx: dict, sims: int = N_SIMS):
    """
    Returns the distribution of the ratio the UI would show.

    Only directional turns are simulated, because neutral turns are excluded
    from the denominator by construction. A predicted-neutral turn therefore
    drops out of the estimate rather than biasing it, which is itself part of
    why the aggregate survives a weak per-turn model.
    """
    c, s, nt = idx["change"], idx["sustain"], idx["neutral"]

    # True labels for this session: change with probability p_true.
    truth = RNG.random((sims, n_turns)) < p_true           # True -> change
    u = RNG.random((sims, n_turns))

    # P(predicted = change) and P(predicted = sustain) given each true label.
    p_pred_change = np.where(truth, P[c, c], P[s, c])
    p_pred_sustain = np.where(truth, P[c, s], P[s, s])

    pred_change = u < p_pred_change
    pred_sustain = (u >= p_pred_change) & (u < p_pred_change + p_pred_sustain)
    # everything else lands on neutral and leaves the denominator

    n_change = pred_change.sum(axis=1)
    n_sustain = pred_sustain.sum(axis=1)
    denom = n_change + n_sustain
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(denom > 0, n_change / np.maximum(denom, 1), np.nan)
    return ratio, n_change, n_sustain


def simulate_corrected(p_true, n_turns, P, idx, sims=N_SIMS):
    """Same simulation, with the Adjusted Classify & Count correction applied."""
    _, n_change, n_sustain = simulate(p_true, n_turns, P, idx, sims)
    return np.array(
        [adjusted_count(int(c), int(s), P, idx) for c, s in zip(n_change, n_sustain)]
    )


def main() -> None:
    P, labels, chosen = confusion_probabilities()
    idx = {l: i for i, l in enumerate(labels)}

    print(f"Held-out P(predicted | true) — client talk type, {chosen}")
    print(f"{'true \\ pred':<14}" + "".join(f"{l:>11}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:<14}" + "".join(f"{P[i, j]:>11.3f}" for j in range(len(labels))))
    print(f"\nsustain talk is recovered as sustain only {P[idx['sustain'], idx['sustain']]:.0%} "
          f"of the time, and read as change {P[idx['sustain'], idx['change']]:.0%} of the time.")

    out = {"source": "benchmark.json held-out confusion matrix, client talk type",
           "model": chosen, "naive": {}, "corrected": {}, "by_model": {}}

    # ---- 1. naive display vs corrected -----------------------------------
    print("\n\nWhat the screen would show for a client who is genuinely 50/50\n")
    print(f"{'turns':>7}{'naive':>9}{'bias':>8}{'SD':>7}   |{'corrected':>11}{'bias':>8}{'SD':>7}")
    print("-" * 60)
    for n in (10, 20, 30, 50, 80, 120):
        raw, _, _ = simulate(0.50, n, P, idx)
        raw = raw[~np.isnan(raw)]
        adj = simulate_corrected(0.50, n, P, idx)
        adj = adj[~np.isnan(adj)]
        print(f"{n:>7}{raw.mean():>9.3f}{raw.mean()-0.5:>8.3f}{raw.std():>7.3f}   |"
              f"{adj.mean():>11.3f}{adj.mean()-0.5:>8.3f}{adj.std():>7.3f}")
        out["naive"][n] = {"mean": round(float(raw.mean()),4), "bias": round(float(raw.mean()-0.5),4), "sd": round(float(raw.std()),4)}
        out["corrected"][n] = {"mean": round(float(adj.mean()),4), "bias": round(float(adj.mean()-0.5),4), "sd": round(float(adj.std()),4)}

    # ---- 2. attenuation, naive vs corrected ------------------------------
    print("\n\nDoes the display track reality? (n = 40 directional turns)\n")
    print(f"{'true':>7}{'naive':>9}{'corrected':>12}")
    print("-" * 28)
    truths = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]
    nv, cr = [], []
    for p_ in truths:
        raw, _, _ = simulate(p_, 40, P, idx); raw = raw[~np.isnan(raw)]
        adj = simulate_corrected(p_, 40, P, idx); adj = adj[~np.isnan(adj)]
        nv.append(float(raw.mean())); cr.append(float(adj.mean()))
        print(f"{p_:>7.2f}{raw.mean():>9.3f}{adj.mean():>12.3f}")
    s_nv = float(np.polyfit(truths, nv, 1)[0]); s_cr = float(np.polyfit(truths, cr, 1)[0])
    print(f"\nslope  naive {s_nv:.3f}   corrected {s_cr:.3f}   (1.00 = faithful)")
    out["attenuation"] = {"true": truths, "naive": [round(v,4) for v in nv],
                          "corrected": [round(v,4) for v in cr],
                          "slope_naive": round(s_nv,4), "slope_corrected": round(s_cr,4)}

    # ---- 3. which model should actually be deployed? ---------------------
    # Macro-F1 picked naive bayes. The product needs sustain talk detected and
    # the displayed ratio to track reality, which is a different criterion.
    print("\n\nWhich model serves the interface best?")
    print("(sustain recall drives the bias; slope measures how faithfully the UI tracks reality)\n")
    print(f"{'model':<22}{'sustain recall':>16}{'naive slope':>13}{'bias @0.5':>11}")
    print("-" * 62)
    b = json.loads((ART / "benchmark.json").read_text())
    for name in b["model_order"]:
        if name == "majority_class":
            continue
        cm = np.array(b["tasks"]["client"]["models"][name].get("confusion_matrix") or [], dtype=float)
        if cm.size == 0:
            continue
        Pm = cm / cm.sum(axis=1, keepdims=True)
        rec = Pm[idx["sustain"], idx["sustain"]]
        vals = []
        for p_ in (0.25, 0.5, 0.75):
            r, _, _ = simulate(p_, 40, Pm, idx); r = r[~np.isnan(r)]
            vals.append(float(r.mean()))
        sl = float(np.polyfit([0.25, 0.5, 0.75], vals, 1)[0])
        print(f"{b['display_names'][name]:<22}{rec:>16.3f}{sl:>13.3f}{vals[1]-0.5:>11.3f}")
        out["by_model"][name] = {"sustain_recall": round(float(rec),4), "slope": round(sl,4),
                                 "bias_at_half": round(vals[1]-0.5,4)}

    (ART / "reliability.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ART / 'reliability.json'}")


if __name__ == "__main__":
    main()
