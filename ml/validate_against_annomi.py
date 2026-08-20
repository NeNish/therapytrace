#!/usr/bin/env python3
"""
Validation check 3 — does the TPI agree with expert judgement?

The five process dimensions were designed from clinical theory, not fitted to
anything. AnnoMI gives us an independent test: experienced counsellors labelled
each client utterance as change talk, sustain talk, or neutral. If the lexicon
is measuring what it claims to measure, then utterances experts called *change
talk* should score higher on the process dimensions than utterances they called
*sustain talk* — without the lexicon ever having seen those labels.

Two analyses:

  A. Utterance level. Mean process score by expert label, with Cohen's d and a
     Mann-Whitney U test for change vs sustain.

  B. Conversation level. Each of the 133 dialogues is treated as a session:
     correlate its aggregated process score against its expert change-talk
     ratio. This is the closest available proxy for "does the TPI track
     therapeutic momentum", using real therapy dialogue and expert labels.

Nothing here is trained. It is a test of the rule-based layer against an
external gold standard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "therapytrace" / "backend"))

from app.nlp.features import DIMENSIONS, score_utterance, session_features  # noqa: E402

DATA = HERE / "data" / "AnnoMI-full.csv"
OUT = HERE / "artifacts"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else 0.0


def main() -> None:
    df = pd.read_csv(DATA)
    df = df[(df.interlocutor == "client") & df.client_talk_type.notna()]
    df = df[df.utterance_text.notna()]
    df = df[df.utterance_text.astype(str).str.split().str.len() >= 5]
    print(f"client utterances: {len(df)}   conversations: {df.transcript_id.nunique()}")

    # ---------------------------------------------------------------- A
    print("\n" + "=" * 74)
    print("A. UTTERANCE LEVEL — mean process score by expert label")
    print("=" * 74)

    scored = [score_utterance(str(t)) for t in df.utterance_text]
    for d in list(DIMENSIONS) + ["process_mean"]:
        df[d] = [getattr(s, d) if d != "process_mean" else s.process_mean for s in scored]

    table = df.groupby("client_talk_type")[list(DIMENSIONS) + ["process_mean"]].mean()
    print(f"\n{'dimension':<24}{'change':>10}{'neutral':>10}{'sustain':>10}"
          f"{'d (ch-su)':>12}{'p':>12}")
    print("-" * 78)

    results = {}
    for d in list(DIMENSIONS) + ["process_mean"]:
        ch = df.loc[df.client_talk_type == "change", d].to_numpy()
        su = df.loc[df.client_talk_type == "sustain", d].to_numpy()
        d_val = cohens_d(ch, su)
        u, p = stats.mannwhitneyu(ch, su, alternative="two-sided")
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{d:<24}{table.loc['change', d]:>10.4f}{table.loc['neutral', d]:>10.4f}"
              f"{table.loc['sustain', d]:>10.4f}{d_val:>12.3f}{p:>11.2e}{star}")
        results[d] = {
            "mean_change": round(float(table.loc["change", d]), 4),
            "mean_neutral": round(float(table.loc["neutral", d]), 4),
            "mean_sustain": round(float(table.loc["sustain", d]), 4),
            "cohens_d": round(d_val, 4),
            "mannwhitney_p": float(p),
            "direction_correct": bool(table.loc["change", d] > table.loc["sustain", d]),
        }

    n_correct = sum(1 for d in DIMENSIONS if results[d]["direction_correct"])
    print(f"\ndimensions ordered correctly (change > sustain): {n_correct} / {len(DIMENSIONS)}")

    # ---------------------------------------------------------------- B
    print("\n" + "=" * 74)
    print("B. CONVERSATION LEVEL — process score vs expert change-talk ratio")
    print("=" * 74)

    rows = []
    for tid, g in df.groupby("transcript_id"):
        n_ch = int((g.client_talk_type == "change").sum())
        n_su = int((g.client_talk_type == "sustain").sum())
        if n_ch + n_su < 5:
            continue
        feats, _ = session_features(list(g.utterance_text.astype(str)))
        rows.append({
            "transcript_id": tid,
            "change_ratio": n_ch / (n_ch + n_su),
            "mi_quality": g.mi_quality.iloc[0],
            **{d: feats[d] for d in DIMENSIONS},
            "process_mean": float(np.mean([feats[d] for d in DIMENSIONS])),
        })

    cdf = pd.DataFrame(rows)
    print(f"\nconversations with >=5 directional turns: {len(cdf)}")
    print(f"\n{'dimension':<24}{'pearson r':>12}{'p':>12}{'spearman':>12}{'p':>12}")
    print("-" * 72)

    conv = {}
    for d in list(DIMENSIONS) + ["process_mean"]:
        r, pr = stats.pearsonr(cdf[d], cdf.change_ratio)
        rho, ps = stats.spearmanr(cdf[d], cdf.change_ratio)
        star = "***" if pr < 0.001 else "**" if pr < 0.01 else "*" if pr < 0.05 else ""
        print(f"{d:<24}{r:>12.4f}{pr:>11.2e}{star:<3}{rho:>9.4f}{ps:>11.2e}")
        conv[d] = {
            "pearson_r": round(float(r), 4), "pearson_p": float(pr),
            "spearman_rho": round(float(rho), 4), "spearman_p": float(ps),
        }

    hi = cdf.loc[cdf.mi_quality == "high", "process_mean"]
    lo = cdf.loc[cdf.mi_quality == "low", "process_mean"]
    quality = None
    if len(lo) >= 3:
        d_val = cohens_d(hi.to_numpy(), lo.to_numpy())
        u, p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
        print(f"\nHigh- vs low-quality MI sessions (client process score)")
        print(f"  high  n={len(hi):>3}  mean={hi.mean():.4f}")
        print(f"  low   n={len(lo):>3}  mean={lo.mean():.4f}")
        print(f"  Cohen's d = {d_val:.3f}   p = {p:.2e}")
        quality = {
            "n_high": int(len(hi)), "n_low": int(len(lo)),
            "mean_high": round(float(hi.mean()), 4),
            "mean_low": round(float(lo.mean()), 4),
            "cohens_d": round(d_val, 4), "p": float(p),
        }

    out = {
        "n_utterances": int(len(df)),
        "n_conversations": int(cdf.transcript_id.nunique()),
        "utterance_level": results,
        "conversation_level": conv,
        "dimensions_ordered_correctly": f"{n_correct}/{len(DIMENSIONS)}",
        "mi_quality_contrast": quality,
    }
    (OUT / "validation_annomi.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'validation_annomi.json'}")


if __name__ == "__main__":
    main()
