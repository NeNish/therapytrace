"""
The Therapeutic Progress Index (TPI).

This is the part of TherapyTrace that is genuinely different from symptom
classifiers, so it is worth being precise about it.

A cross-sectional model asks: how does this client compare to other people?
TPI asks: how does this session compare to *this client's own* earlier
sessions? Every dimension is standardised against a personal baseline built
from the client's first `baseline_k` sessions, so a naturally withdrawn client
and a naturally expressive client can both show movement.

    z_d(s) = (x_d(s) - mu_d) / sigma_d          per dimension d, session s
    TPI(s) = 50 + 10 * sum_d ( w_d * z_d(s) )   clipped to [0, 100]

sigma_d is pooled: the client's own baseline SD, shrunk toward a cohort-level
SD, because two or three baseline sessions give a very unstable variance
estimate on their own. Shrinkage weight follows a standard James-Stein style
rule, lambda = n / (n + k0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .features import DIMENSIONS

# Weights are exposed so they can be re-fit against outcome measures
# (PHQ-9 / MADRS change scores, AnnoMI change-talk labels) rather than
# left as the authors' guesses. See `calibration.py`.
DEFAULT_WEIGHTS: dict[str, float] = {
    "self_agency": 0.24,
    "future_orientation": 0.18,
    "emotional_granularity": 0.20,
    "problem_ownership": 0.20,
    "reflection_depth": 0.18,
}

# Cohort priors, used until enough real sessions exist to estimate them.
COHORT_SD: dict[str, float] = {
    "self_agency": 0.075,
    "future_orientation": 0.070,
    "emotional_granularity": 0.080,
    "problem_ownership": 0.085,
    "reflection_depth": 0.070,
}

SHRINKAGE_K0 = 3.0  # sessions of prior weight given to the cohort SD
MIN_SD = 0.015


@dataclass
class Baseline:
    means: dict[str, float]
    sds: dict[str, float]
    n_sessions: int
    provisional: bool  # True until baseline_k sessions exist

    def to_dict(self) -> dict:
        return {
            "means": self.means,
            "sds": self.sds,
            "n_sessions": self.n_sessions,
            "provisional": self.provisional,
        }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def build_baseline(
    feature_rows: list[dict[str, float]], baseline_k: int = 2
) -> Baseline:
    """feature_rows must be in session order, earliest first."""
    rows = feature_rows[:baseline_k] if feature_rows else []
    n = len(rows)
    means, sds = {}, {}
    for d in DIMENSIONS:
        vals = [r[d] for r in rows if d in r]
        means[d] = round(_mean(vals), 4) if vals else 0.5
        own_sd = _sd(vals)
        lam = n / (n + SHRINKAGE_K0)
        pooled = lam * own_sd + (1 - lam) * COHORT_SD[d]
        sds[d] = round(max(MIN_SD, pooled), 4)
    return Baseline(means, sds, n, provisional=n < baseline_k)


def z_scores(features: dict[str, float], baseline: Baseline) -> dict[str, float]:
    return {
        d: round((features.get(d, baseline.means[d]) - baseline.means[d]) / baseline.sds[d], 4)
        for d in DIMENSIONS
    }


def tpi_from_z(
    z: dict[str, float], weights: dict[str, float] | None = None
) -> tuple[float, dict[str, float]]:
    """Returns (TPI, per-dimension contribution in TPI points)."""
    w = weights or DEFAULT_WEIGHTS
    contributions = {d: round(10.0 * w[d] * z.get(d, 0.0), 3) for d in DIMENSIONS}
    tpi = 50.0 + sum(contributions.values())
    return round(max(0.0, min(100.0, tpi)), 2), contributions


def confidence(features: dict[str, float], baseline: Baseline) -> tuple[float, list[str]]:
    """
    How much weight should a clinician put on this score?
    Short sessions and provisional baselines both reduce it.
    """
    notes: list[str] = []
    conf = 1.0
    words = features.get("n_client_words", 0.0)
    turns = features.get("n_client_turns", 0.0)

    if words < 300:
        conf *= 0.55
        notes.append("Fewer than 300 client words: the score is noisy.")
    elif words < 800:
        conf *= 0.8
        notes.append("Short session; interpret movement cautiously.")
    if turns < 10:
        conf *= 0.7
        notes.append("Fewer than 10 client turns.")
    if baseline.provisional:
        conf *= 0.6
        notes.append(
            "Baseline is provisional — it firms up once two full sessions exist."
        )
    return round(min(1.0, conf), 3), notes


def absolute_process_index(
    features: dict[str, float], weights: dict[str, float] | None = None
) -> tuple[float, dict[str, float]]:
    """
    Standalone process score when no client baseline exists yet.

    Maps the weighted mean of raw dimension scores (0–1) to the same 0–100
    scale: 50 means neutral language on every dimension, higher means more
    therapeutic process markers in the text itself.
    """
    w = weights or DEFAULT_WEIGHTS
    weighted = sum(w[d] * features.get(d, 0.5) for d in DIMENSIONS)
    contributions = {
        d: round(100.0 * w[d] * features.get(d, 0.5), 3) for d in DIMENSIONS
    }
    tpi = round(max(0.0, min(100.0, 100.0 * weighted)), 2)
    return tpi, contributions


def score_session(
    features: dict[str, float],
    baseline: Baseline,
    weights: dict[str, float] | None = None,
) -> dict:
    z = z_scores(features, baseline)
    tpi, contrib = tpi_from_z(z, weights)
    conf, notes = confidence(features, baseline)
    return {
        "tpi": tpi,
        "z": z,
        "contributions": contrib,
        "confidence": conf,
        "confidence_notes": notes,
    }


def score_session_standalone(
    features: dict[str, float], weights: dict[str, float] | None = None
) -> dict:
    """Score a single session with no prior history — used by the multimodal page."""
    tpi, contrib = absolute_process_index(features, weights)
    conf, notes = confidence(features, Baseline(
        means={d: 0.5 for d in DIMENSIONS},
        sds=COHORT_SD,
        n_sessions=0,
        provisional=True,
    ))
    notes.insert(0, "No client baseline yet — score reflects language content, not change from prior sessions.")
    return {
        "tpi": tpi,
        "z": {d: round((features.get(d, 0.5) - 0.5) / COHORT_SD[d], 4) for d in DIMENSIONS},
        "contributions": contrib,
        "confidence": conf,
        "confidence_notes": notes,
        "scoring_mode": "absolute",
    }
