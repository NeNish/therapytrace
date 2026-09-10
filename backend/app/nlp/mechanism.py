"""
M11 — Idiographic Mechanism & Response Profiling.

This module is the part of TherapyTrace that has no analogue in the
speech-to-outcome literature it otherwise resembles. Financial linguistics asks
whether a CEO's language predicts a price. It never asks *which analyst question
should be asked next*, because the analyst is not trying to change the CEO.

In therapy the second speaker is the intervention. That makes two questions
askable that are not askable elsewhere:

  A. MECHANISM — within one person, does one process dimension move *before*
     another? Clinical theory asserts orderings ("insight precedes behavioural
     change") but they are asserted, not measured. If, for a given client,
     reflection depth at session s predicts self-agency at s+1 better than the
     reverse, that is a directed, testable mechanism claim recovered from
     language alone.

  B. RESPONSE POLICY — the field reports which interventions work *on average*.
     The clinically useful question is which intervention works *for this
     client, in the state they are in right now*. A challenge delivered to a
     client already showing agency is a different act from the same challenge
     delivered to a client who is stuck, even word for word.

Both are estimated per person. Neither pools across a population, because the
whole premise of the system is that population baselines are invalid here.

Honest statistical position: these are lagged associations from short series.
With 10-12 sessions there is no causal identification, and the module reports
effect sizes with explicit sample counts rather than significance theatre. It is
hypothesis generation for supervision, not proof of mechanism.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import permutations

from .features import DIMENSIONS


# ==========================================================================
# A. MECHANISM — which dimension leads which, within one person
# ==========================================================================

def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return num / (dx * dy) if dx > 1e-12 and dy > 1e-12 else None


def _diff(xs: list[float]) -> list[float]:
    """First differences. We model *change*, not level, because levels within a
    person are dominated by their stable speaking style."""
    return [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]


def lead_lag_matrix(dimension_series: dict[str, list[float]], min_sessions: int = 6) -> dict:
    """
    For every ordered pair of dimensions (A, B), estimate whether change in A
    at session s precedes change in B at s+1.

        lead(A -> B) = corr( dA[:-1], dB[1:] )

    The asymmetry is what carries information: if lead(A->B) is substantially
    larger than lead(B->A), A moves first for this person.
    """
    dims = [d for d in DIMENSIONS if d in dimension_series]
    n = min((len(dimension_series[d]) for d in dims), default=0)

    if n < min_sessions:
        return {
            "available": False,
            "reason": f"Needs at least {min_sessions} sessions; this case has {n}.",
            "n_sessions": n,
        }

    deltas = {d: _diff(dimension_series[d][:n]) for d in dims}
    edges = []

    for a, b in permutations(dims, 2):
        fwd = _pearson(deltas[a][:-1], deltas[b][1:])
        rev = _pearson(deltas[b][:-1], deltas[a][1:])
        if fwd is None or rev is None:
            continue
        edges.append({
            "from": a,
            "to": b,
            "lead_corr": round(fwd, 4),
            "reverse_corr": round(rev, 4),
            "asymmetry": round(fwd - rev, 4),
            "n_pairs": len(deltas[a]) - 1,
        })

    edges.sort(key=lambda e: e["asymmetry"], reverse=True)

    # Keep only edges where the forward direction is both positive and clearly
    # stronger than its reverse. The 0.25 floor is deliberately conservative for
    # series this short.
    strong = [e for e in edges if e["asymmetry"] > 0.25 and e["lead_corr"] > 0.2]

    order = None
    if strong:
        wins: dict[str, float] = defaultdict(float)
        for e in strong:
            wins[e["from"]] += e["asymmetry"]
            wins[e["to"]] -= e["asymmetry"]
        order = sorted(dims, key=lambda d: wins[d], reverse=True)

    return {
        "available": True,
        "n_sessions": n,
        "edges": edges,
        "strong_edges": strong,
        "inferred_order": order,
        "interpretation": (
            _describe_order(order, strong)
            if order else
            "No dimension consistently leads another for this client. Change "
            "appears to move together rather than in sequence."
        ),
        "caveat": (
            "Lagged association from a short series. This is a hypothesis about "
            "the order of change for this person, not a causal claim."
        ),
    }


def _describe_order(order: list[str], strong: list[dict]) -> str:
    pretty = lambda d: d.replace("_", " ")
    lead = pretty(order[0])
    top = strong[0]
    return (
        f"For this client, {pretty(top['from'])} tends to move a session before "
        f"{pretty(top['to'])} (asymmetry {top['asymmetry']:+.2f}). "
        f"{lead.capitalize()} appears to be the earliest-moving dimension, which "
        f"suggests it is where change starts for them."
    )


# ==========================================================================
# B. RESPONSE POLICY — what works for this client, in the state they are in
# ==========================================================================

STATE_BINS = ("stuck", "middling", "moving")


def _state_of(process_mean: float, lo: float, hi: float) -> str:
    if process_mean <= lo:
        return "stuck"
    if process_mean >= hi:
        return "moving"
    return "middling"


def response_profile(
    records: list[dict], min_per_cell: int = 3
) -> dict:
    """
    records: the turn-pair rows produced by M6 — each has an intervention type,
    the client's process score before it, and the score after.

    Splits every intervention by the client's state *immediately before* it, and
    reports the lift within each state. The clinically interesting output is not
    the best intervention overall; it is where an intervention's effect changes
    sign depending on the state it was delivered into.
    """
    if len(records) < 12:
        return {
            "available": False,
            "reason": f"Needs at least 12 turn pairs; this case has {len(records)}.",
        }

    befores = sorted(r["before"] for r in records)
    lo = befores[len(befores) // 3]
    hi = befores[2 * len(befores) // 3]

    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        cells[(r["intervention"], _state_of(r["before"], lo, hi))].append(r["lift"])

    rows = []
    for (interv, state), lifts in cells.items():
        if len(lifts) < min_per_cell:
            continue
        mean = sum(lifts) / len(lifts)
        sd = (
            math.sqrt(sum((x - mean) ** 2 for x in lifts) / (len(lifts) - 1))
            if len(lifts) > 1 else 0.0
        )
        rows.append({
            "intervention": interv,
            "state": state,
            "n": len(lifts),
            "mean_lift": round(mean, 4),
            "sd": round(sd, 4),
            "se": round(sd / math.sqrt(len(lifts)), 4) if lifts else 0.0,
        })

    # State-dependence: an intervention whose effect flips sign between states
    # is the finding worth surfacing to a supervisor.
    by_interv: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_interv[r["intervention"]][r["state"]] = r

    flips = []
    for interv, states in by_interv.items():
        if len(states) < 2:
            continue
        best = max(states.values(), key=lambda s: s["mean_lift"])
        worst = min(states.values(), key=lambda s: s["mean_lift"])
        spread = best["mean_lift"] - worst["mean_lift"]
        if spread > 0.02:
            flips.append({
                "intervention": interv,
                "best_state": best["state"],
                "best_lift": best["mean_lift"],
                "worst_state": worst["state"],
                "worst_lift": worst["mean_lift"],
                "state_dependence": round(spread, 4),
                "sign_flip": best["mean_lift"] > 0 > worst["mean_lift"],
                "n_total": sum(s["n"] for s in states.values()),
            })
    flips.sort(key=lambda f: f["state_dependence"], reverse=True)

    recommendations = []
    for state in STATE_BINS:
        in_state = [r for r in rows if r["state"] == state]
        if not in_state:
            continue
        best = max(in_state, key=lambda r: r["mean_lift"])
        if best["mean_lift"] > 0:
            recommendations.append({
                "when_client_is": state,
                "intervention": best["intervention"],
                "mean_lift": best["mean_lift"],
                "n": best["n"],
            })

    return {
        "available": True,
        "state_thresholds": {"stuck_below": round(lo, 4), "moving_above": round(hi, 4)},
        "cells": sorted(rows, key=lambda r: (r["state"], -r["mean_lift"])),
        "state_dependent": flips,
        "recommendations": recommendations,
        "n_pairs": len(records),
        "caveat": (
            "Observational and confounded by indication: therapists choose "
            "interventions in response to what was just said. Read this as a "
            "description of what has happened with this client, not a "
            "prescription for what to do next."
        ),
    }


def profile_client(dimension_series: dict[str, list[float]], turn_records: list[dict]) -> dict:
    return {
        "mechanism": lead_lag_matrix(dimension_series),
        "response": response_profile(turn_records),
    }
