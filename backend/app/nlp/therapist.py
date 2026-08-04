"""
Therapist contribution analysis.

The question: after a given kind of therapist move, does the client's very next
turn shift toward more agency, more specific affect, more ownership, more
reflection?

Method (deliberately simple and honest about its limits):

  1. Tag each therapist turn with an intervention type (rule-based, MITI/MI
     inspired taxonomy).
  2. For each therapist turn, take the client turn immediately after it and
     the client turn immediately before it.
  3. lift = process_mean(next) - process_mean(previous)
     i.e. within the same session, against the client's own local level, so
     between-client differences cancel out.
  4. Aggregate lift per intervention type, with a mean, an SD, an n, and a
     bootstrap-free normal CI.

Limits, which belong in any write-up: this is an association, not a causal
estimate. Therapists choose interventions in response to what the client just
said, so confounding by indication is guaranteed. Treat the output as a
reflective prompt for supervision, not a ranking of techniques.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from .features import score_utterance
from .lexicons import THERAPIST_INTERVENTIONS
from .transcript import Turn

COMPILED = [
    (name, [re.compile(p, re.IGNORECASE) for p in pats])
    for name, pats in THERAPIST_INTERVENTIONS
]

INTERVENTION_LABELS = {
    "complex_reflection": "Complex reflection",
    "simple_reflection": "Simple reflection",
    "open_question": "Open question",
    "closed_question": "Closed question",
    "affirmation": "Affirmation",
    "challenge": "Challenge",
    "psychoeducation": "Psychoeducation",
    "directive": "Directive / task",
    "summary": "Summary",
    "minimal_encourager": "Minimal encourager",
    "other": "Other",
}


def tag_intervention(text: str) -> str:
    t = text.strip()
    for name, patterns in COMPILED:
        for rx in patterns:
            if rx.search(t):
                return name
    return "open_question" if t.endswith("?") else "other"


def analyse_turn_pairs(turns: list[Turn]) -> list[dict]:
    """One record per therapist turn that has a client turn on both sides."""
    records: list[dict] = []
    cache: dict[int, float] = {}

    def pm(turn: Turn) -> float:
        if turn.idx not in cache:
            cache[turn.idx] = score_utterance(turn.text).process_mean
        return cache[turn.idx]

    for i, turn in enumerate(turns):
        if turn.speaker != "therapist":
            continue
        prev_client = next(
            (turns[j] for j in range(i - 1, -1, -1) if turns[j].speaker == "client"),
            None,
        )
        next_client = next(
            (turns[j] for j in range(i + 1, len(turns)) if turns[j].speaker == "client"),
            None,
        )
        if prev_client is None or next_client is None:
            continue
        if len(next_client.text.split()) < 5:
            continue  # a two-word answer carries no measurable shift

        before, after = pm(prev_client), pm(next_client)
        records.append({
            "therapist_turn_idx": turn.idx,
            "intervention": tag_intervention(turn.text),
            "therapist_text": turn.text,
            "client_next_text": next_client.text,
            "before": round(before, 4),
            "after": round(after, 4),
            "lift": round(after - before, 4),
        })
    return records


def aggregate_impact(records: list[dict], min_n: int = 3) -> list[dict]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in records:
        buckets[r["intervention"]].append(r["lift"])

    out = []
    for name, lifts in buckets.items():
        n = len(lifts)
        mean = sum(lifts) / n
        sd = (
            math.sqrt(sum((x - mean) ** 2 for x in lifts) / (n - 1))
            if n > 1 else 0.0
        )
        se = sd / math.sqrt(n) if n else 0.0
        out.append({
            "intervention": name,
            "label": INTERVENTION_LABELS.get(name, name),
            "n": n,
            "mean_lift": round(mean, 4),
            "sd": round(sd, 4),
            "ci_low": round(mean - 1.96 * se, 4),
            "ci_high": round(mean + 1.96 * se, 4),
            "reliable": n >= min_n and abs(mean) > 1.96 * se,
        })
    out.sort(key=lambda r: r["mean_lift"], reverse=True)
    return out


def top_moments(records: list[dict], k: int = 5) -> dict[str, list[dict]]:
    """The single exchanges that moved the needle most, in both directions."""
    ranked = sorted(records, key=lambda r: r["lift"], reverse=True)
    trim = lambda r: {
        "intervention": INTERVENTION_LABELS.get(r["intervention"], r["intervention"]),
        "therapist_text": r["therapist_text"][:400],
        "client_next_text": r["client_next_text"][:600],
        "lift": r["lift"],
    }
    return {
        "openings": [trim(r) for r in ranked[:k]],
        "closings": [trim(r) for r in reversed(ranked[-k:])],
    }
