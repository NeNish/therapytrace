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


# Which coarse behaviour family each fine-grained label belongs to. The trained
# model predicts the family; the regex taxonomy names the subtype inside it.
FAMILY_OF = {
    "open_question": "question",
    "closed_question": "question",
    "simple_reflection": "reflection",
    "complex_reflection": "reflection",
    "summary": "reflection",
    "affirmation": "therapist_input",
    "psychoeducation": "therapist_input",
    "directive": "therapist_input",
    "challenge": "therapist_input",
    "minimal_encourager": "other",
    "other": "other",
}

# Used when the model is confident about the family but no pattern matches
# inside it — better than falling back to a label from the wrong family.
FAMILY_DEFAULT = {
    "question": "open_question",
    "reflection": "simple_reflection",
    "therapist_input": "psychoeducation",
    "other": "other",
}


def tag_intervention(text: str, family: str | None = None) -> str:
    """
    Name the intervention, optionally constrained to a behaviour family.

    Without `family` this is the original regex taxonomy, kept intact so the
    system still works on a machine with no trained artifacts.

    With `family` — supplied by the AnnoMI-trained behaviour model — the search
    is restricted to patterns belonging to that family. This is the division of
    labour each method is actually good at. The model was trained on thousands
    of expert-annotated turns and decides reliably whether something is a
    question, a reflection or the therapist supplying content; it has no notion
    of *open* versus *closed*. The patterns cannot tell a reflection from a
    statement, but they are precise about the subtype once the family is known.
    Neither alone is as good as the pair.
    """
    t = text.strip()
    for name, patterns in COMPILED:
        if family is not None and FAMILY_OF.get(name) != family:
            continue
        for rx in patterns:
            if rx.search(t):
                return name

    if family is not None:
        return FAMILY_DEFAULT.get(family, "other")
    return "open_question" if t.endswith("?") else "other"


def _behaviour_families(turns: list[Turn]) -> dict[int, str]:
    """
    Model-predicted family for every therapist turn, in one batched call.

    Batched deliberately: classifying turn by turn would re-enter the sklearn
    pipeline once per utterance and dominate the cost of analysing a session.
    Returns an empty mapping when no model is loaded, and every caller treats
    that as "fall back to the patterns" rather than as an error.
    """
    idxs = [t.idx for t in turns if t.speaker == "therapist"]
    texts = [t.text for t in turns if t.speaker == "therapist"]
    if not texts:
        return {}
    try:
        from .ml_models import classify_therapist_turns

        out = classify_therapist_turns(texts)
    except Exception:
        return {}
    if not out or not out.get("labels"):
        return {}
    return dict(zip(idxs, out["labels"]))


def analyse_turn_pairs(turns: list[Turn]) -> list[dict]:
    """One record per therapist turn that has a client turn on both sides."""
    records: list[dict] = []
    cache: dict[int, float] = {}
    families = _behaviour_families(turns)

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
        family = families.get(turn.idx)
        records.append({
            "therapist_turn_idx": turn.idx,
            "intervention": tag_intervention(turn.text, family),
            # Recorded so a reader can see which component produced the label,
            # rather than having to infer it.
            "behaviour": family,
            "tagged_by": "model+lexicon" if family else "lexicon",
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
