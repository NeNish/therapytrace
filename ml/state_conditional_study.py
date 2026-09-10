#!/usr/bin/env python3
"""
Empirical study — is the effect of a therapist intervention conditional on the
state the client is already in?

The motivational-interviewing literature reports which therapist behaviours are
associated with change talk *on average*. This asks a question that is not, to
our knowledge, answered anywhere in the corpus of work on AnnoMI:

    Does the same intervention produce a different response depending on
    whether the client is currently stuck or already moving?

Design, using real expert annotations throughout:

  * 133 real therapy conversations, expert-coded.
  * For each therapist turn with a client turn on each side, take:
        - the expert label of the therapist turn (question / reflection /
          therapist_input / other)
        - the client's state *before*, measured as the process score of the
          preceding client turn, binned into terciles across the corpus
        - the outcome, measured two ways:
            (i)  process lift  = process(after) - process(before)
            (ii) expert change talk in the following client turn (binary)
  * Report mean outcome per intervention x state cell.

Outcome (ii) is the important one: it uses the *expert's* label as the outcome,
so the finding does not depend on our lexicon being right.

Confounding by indication is unavoidable and is stated with the result: a
therapist chooses a challenge because of what the client just said. This is a
description of what happens in real sessions, not an experiment.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "therapytrace" / "backend"))

from app.nlp.features import score_utterance  # noqa: E402

DATA = HERE / "data" / "AnnoMI-full.csv"
OUT = HERE / "artifacts"

STATES = ["stuck", "middling", "moving"]
INTERVENTIONS = ["question", "reflection", "therapist_input", "other"]


def build_triples() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df = df.sort_values(["transcript_id", "utterance_id"]).reset_index(drop=True)
    df = df[df.utterance_text.notna()]

    cache: dict[int, float] = {}

    def process_of(idx: int) -> float:
        if idx not in cache:
            cache[idx] = score_utterance(str(df.at[idx, "utterance_text"])).process_mean
        return cache[idx]

    rows = []
    for tid, g in df.groupby("transcript_id"):
        idx = list(g.index)
        for pos in range(1, len(idx) - 1):
            i_prev, i_t, i_next = idx[pos - 1], idx[pos], idx[pos + 1]
            if df.at[i_t, "interlocutor"] != "therapist":
                continue
            if df.at[i_prev, "interlocutor"] != "client":
                continue
            if df.at[i_next, "interlocutor"] != "client":
                continue

            behaviour = df.at[i_t, "main_therapist_behaviour"]
            if pd.isna(behaviour):
                continue
            if len(str(df.at[i_next, "utterance_text"]).split()) < 5:
                continue
            if len(str(df.at[i_prev, "utterance_text"]).split()) < 5:
                continue

            before = process_of(i_prev)
            after = process_of(i_next)
            next_label = df.at[i_next, "client_talk_type"]

            rows.append({
                "transcript_id": tid,
                "intervention": behaviour,
                "before": before,
                "after": after,
                "lift": after - before,
                "next_change": 1 if next_label == "change" else 0,
                "next_sustain": 1 if next_label == "sustain" else 0,
                "next_label": next_label,
                "mi_quality": df.at[i_t, "mi_quality"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    t = build_triples()
    print(f"turn triples: {len(t)}   conversations: {t.transcript_id.nunique()}")

    lo, hi = t.before.quantile([1 / 3, 2 / 3])
    t["state"] = np.where(t.before <= lo, "stuck",
                  np.where(t.before >= hi, "moving", "middling"))
    print(f"state thresholds: stuck <= {lo:.4f}   moving >= {hi:.4f}")
    print(t.state.value_counts().to_string())

    # ---------------------------------------------------------------- main
    print("\n" + "=" * 88)
    print("P(next client turn is CHANGE TALK)  —  expert label as the outcome")
    print("=" * 88)
    print(f"\n{'intervention':<20}" + "".join(f"{s:>14}" for s in STATES)
          + f"{'spread':>12}{'n':>8}")
    print("-" * 88)

    findings = []
    for interv in INTERVENTIONS:
        sub = t[t.intervention == interv]
        if len(sub) < 30:
            continue
        cells, ns = {}, {}
        for s in STATES:
            c = sub[sub.state == s]
            cells[s] = float(c.next_change.mean()) if len(c) >= 10 else np.nan
            ns[s] = int(len(c))
        vals = [v for v in cells.values() if not np.isnan(v)]
        spread = max(vals) - min(vals) if len(vals) > 1 else np.nan
        print(f"{interv:<20}" + "".join(
            f"{cells[s]:>13.1%}" if not np.isnan(cells[s]) else f"{'—':>14}"
            for s in STATES
        ) + f"{spread:>11.1%}{len(sub):>8}")
        findings.append({
            "intervention": interv, "by_state": cells, "n_by_state": ns,
            "n_total": int(len(sub)), "spread": None if np.isnan(spread) else round(spread, 4),
        })

    # ---------------------------------------------------------------- tests
    print("\n" + "=" * 88)
    print("Is the effect of an intervention conditional on state?")
    print("=" * 88)

    tests = []
    for interv in INTERVENTIONS:
        sub = t[t.intervention == interv]
        stuck = sub[sub.state == "stuck"].next_change
        moving = sub[sub.state == "moving"].next_change
        if len(stuck) < 15 or len(moving) < 15:
            continue
        table = [[int(stuck.sum()), int(len(stuck) - stuck.sum())],
                 [int(moving.sum()), int(len(moving) - moving.sum())]]
        chi2, p, _, _ = stats.chi2_contingency(table)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {interv:<20} stuck {stuck.mean():.1%} (n={len(stuck)})  vs  "
              f"moving {moving.mean():.1%} (n={len(moving)})   "
              f"chi2={chi2:.2f}  p={p:.2e} {star}")
        tests.append({
            "intervention": interv,
            "p_change_when_stuck": round(float(stuck.mean()), 4),
            "n_stuck": int(len(stuck)),
            "p_change_when_moving": round(float(moving.mean()), 4),
            "n_moving": int(len(moving)),
            "chi2": round(float(chi2), 3),
            "p_value": float(p),
            "significant": bool(p < 0.05),
        })

    # ---------------------------------------------------------- best per state
    print("\n" + "=" * 88)
    print("Best intervention, conditional on the state the client is in")
    print("=" * 88)
    policy = []
    for s in STATES:
        sub = t[t.state == s]
        scores = {
            i: float(sub[sub.intervention == i].next_change.mean())
            for i in INTERVENTIONS
            if len(sub[sub.intervention == i]) >= 15
        }
        if not scores:
            continue
        best = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)
        print(f"  client is {s.upper():<10} -> best: {best:<18} "
              f"({scores[best]:.1%} change talk)   worst: {worst} ({scores[worst]:.1%})")
        policy.append({
            "state": s, "best": best, "best_rate": round(scores[best], 4),
            "worst": worst, "worst_rate": round(scores[worst], 4),
            "all": {k: round(v, 4) for k, v in scores.items()},
        })

    # global comparison, for contrast with the conditional view
    print("\nUnconditional (what the field usually reports):")
    for i in INTERVENTIONS:
        sub = t[t.intervention == i]
        if len(sub) >= 30:
            print(f"  {i:<20} {sub.next_change.mean():.1%} change talk  (n={len(sub)})")

    out = {
        "n_triples": int(len(t)),
        "n_conversations": int(t.transcript_id.nunique()),
        "state_thresholds": {"stuck_below": round(float(lo), 4),
                             "moving_above": round(float(hi), 4)},
        "change_talk_by_intervention_and_state": findings,
        "conditional_tests": tests,
        "state_conditional_policy": policy,
        "unconditional": {
            i: round(float(t[t.intervention == i].next_change.mean()), 4)
            for i in INTERVENTIONS if len(t[t.intervention == i]) >= 30
        },
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "state_conditional.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'state_conditional.json'}")


if __name__ == "__main__":
    main()
