"""
Feature extraction.

Two levels:
  * `score_utterance`  -> raw counts + normalised sub-scores for one client turn
  * `session_features` -> aggregated, length-normalised feature vector for a session

Every construct is expressed on a 0..1 scale where higher = more therapeutic.
Nothing here is trained; it is deterministic and auditable. The learned tier in
`encoders.py` can override any dimension when a fine-tuned checkpoint is present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Iterable

from . import lexicons as LX

WORD_RE = re.compile(r"[a-z']+")

DIMENSIONS = (
    "self_agency",
    "future_orientation",
    "emotional_granularity",
    "problem_ownership",
    "reflection_depth",
)

AUXILIARY = (
    "hopelessness",
    "solution_focus",
    "absolutism",
    "self_focus_balance",
    "rumination",
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _count_phrases(text_lc: str, phrases: Iterable[str]) -> int:
    """Count occurrences of multi-word phrases and single words alike."""
    total = 0
    for p in phrases:
        if " " in p or "'" in p:
            total += text_lc.count(p)
        else:
            total += len(re.findall(rf"\b{re.escape(p)}\b", text_lc))
    return total


def _count_patterns(text_lc: str, patterns: Iterable[str]) -> int:
    return sum(len(re.findall(p, text_lc)) for p in patterns)


def _ratio(pos: float, neg: float, prior: float = 1.0) -> float:
    """
    Smoothed positive share. With no evidence either way it returns 0.5,
    so a short turn never looks like a collapse in functioning.
    """
    return (pos + prior) / (pos + neg + 2 * prior)


def _saturate(count: float, per_100_words: float, n_words: int) -> float:
    """Map a rate to 0..1 with diminishing returns (saturating at ~2x target)."""
    if n_words == 0:
        return 0.0
    rate = 100.0 * count / n_words
    return min(1.0, rate / (rate + per_100_words))


# --------------------------------------------------------------------------
# utterance level
# --------------------------------------------------------------------------

@dataclass
class UtteranceScore:
    text: str
    n_words: int
    self_agency: float
    future_orientation: float
    emotional_granularity: float
    problem_ownership: float
    reflection_depth: float
    hopelessness: float
    solution_focus: float
    absolutism: float
    self_focus_balance: float
    rumination: float
    evidence: dict[str, list[str]] = field(default_factory=dict)

    @property
    def process_mean(self) -> float:
        return sum(getattr(self, d) for d in DIMENSIONS) / len(DIMENSIONS)

    def to_dict(self) -> dict:
        return asdict(self)


def _collect_evidence(text_lc: str, phrases: Iterable[str], cap: int = 4) -> list[str]:
    hits = []
    for p in phrases:
        if " " in p or "'" in p:
            if p in text_lc:
                hits.append(p)
        elif re.search(rf"\b{re.escape(p)}\b", text_lc):
            hits.append(p)
        if len(hits) >= cap:
            break
    return hits


def score_utterance(text: str, keep_evidence: bool = False) -> UtteranceScore:
    text_lc = " " + text.lower().strip() + " "
    tokens = tokenize(text_lc)
    n = len(tokens)

    # --- 1. self-agency ---------------------------------------------------
    act = _count_phrases(text_lc, LX.AGENCY_ACTIVE)
    pas = _count_phrases(text_lc, LX.AGENCY_PASSIVE)
    self_agency = _ratio(act, pas)

    # --- 2. future orientation -------------------------------------------
    fut = _count_phrases(text_lc, LX.FUTURE_MARKERS)
    pst = _count_phrases(text_lc, LX.PAST_MARKERS)
    rum = _count_phrases(text_lc, LX.RUMINATION_MARKERS)
    future_orientation = _ratio(fut, pst + 2 * rum)

    # --- 3. emotional granularity ----------------------------------------
    t1 = _count_phrases(text_lc, LX.EMOTION_TIER_1)
    t2 = _count_phrases(text_lc, LX.EMOTION_TIER_2)
    t3 = _count_phrases(text_lc, LX.EMOTION_TIER_3)
    mixed = _count_patterns(text_lc, LX.MIXED_EMOTION_PATTERNS)
    distinct = len({w for w in tokens if w in LX.EMOTION_TIER_2 | LX.EMOTION_TIER_3})
    weighted = 0.25 * t1 + 1.0 * t2 + 2.0 * t3 + 1.5 * mixed
    specificity = _ratio(1.0 * t2 + 2.0 * t3 + 1.5 * mixed, 1.5 * t1)
    density = _saturate(weighted, per_100_words=3.0, n_words=n)
    variety = min(1.0, distinct / 4.0)
    emotional_granularity = 0.5 * specificity + 0.3 * density + 0.2 * variety

    # --- 4. problem ownership --------------------------------------------
    own = _count_phrases(text_lc, LX.OWNERSHIP_INTERNAL)
    ext = _count_phrases(text_lc, LX.OWNERSHIP_EXTERNAL)
    blame = _count_phrases(text_lc, LX.SELF_BLAME)
    # self-blame is penalised: it is neither ownership nor externalisation
    problem_ownership = _ratio(own, ext + 1.5 * blame)

    # --- 5. reflection depth ---------------------------------------------
    ins = _count_phrases(text_lc, LX.INSIGHT_TERMS)
    cau = _count_phrases(text_lc, LX.CAUSAL_TERMS)
    ten = _count_phrases(text_lc, LX.TENTATIVE_TERMS)
    abso = _count_phrases(text_lc, LX.ABSOLUTIST_TERMS)
    dist = _count_patterns(text_lc, LX.SELF_DISTANCING_PATTERNS)
    depth_raw = 2.0 * ins + 1.0 * cau + 0.75 * ten + 1.0 * dist
    depth_density = _saturate(depth_raw, per_100_words=4.0, n_words=n)
    depth_balance = _ratio(depth_raw, 1.5 * abso)
    reflection_depth = 0.6 * depth_density + 0.4 * depth_balance

    # --- auxiliary --------------------------------------------------------
    hope_neg = _count_phrases(text_lc, LX.HOPELESSNESS_TERMS)
    hopelessness = _saturate(hope_neg, per_100_words=1.5, n_words=n)
    solution_focus = _saturate(
        _count_phrases(text_lc, LX.SOLUTION_FOCUS_TERMS), per_100_words=2.0, n_words=n
    )
    absolutism = _saturate(abso, per_100_words=2.5, n_words=n)
    rumination = _saturate(rum, per_100_words=1.5, n_words=n)

    fps = sum(1 for w in tokens if w in LX.FIRST_PERSON_SINGULAR)
    oth = sum(1 for w in tokens if w in LX.OTHER_PERSON)
    # Balanced self-reference peaks around a 55/45 I-to-other split. Both
    # total self-absorption and total self-erasure score low.
    share = _ratio(fps, oth)
    self_focus_balance = max(0.0, 1.0 - abs(share - 0.55) / 0.45)

    evidence: dict[str, list[str]] = {}
    if keep_evidence:
        evidence = {
            "self_agency+": _collect_evidence(text_lc, LX.AGENCY_ACTIVE),
            "self_agency-": _collect_evidence(text_lc, LX.AGENCY_PASSIVE),
            "future_orientation+": _collect_evidence(text_lc, LX.FUTURE_MARKERS),
            "future_orientation-": _collect_evidence(text_lc, LX.RUMINATION_MARKERS),
            "emotional_granularity+": _collect_evidence(text_lc, LX.EMOTION_TIER_3),
            "emotional_granularity-": _collect_evidence(text_lc, LX.EMOTION_TIER_1),
            "problem_ownership+": _collect_evidence(text_lc, LX.OWNERSHIP_INTERNAL),
            "problem_ownership-": _collect_evidence(text_lc, LX.OWNERSHIP_EXTERNAL),
            "reflection_depth+": _collect_evidence(text_lc, LX.INSIGHT_TERMS),
            "reflection_depth-": _collect_evidence(text_lc, LX.ABSOLUTIST_TERMS),
            "hopelessness": _collect_evidence(text_lc, LX.HOPELESSNESS_TERMS),
        }
        evidence = {k: v for k, v in evidence.items() if v}

    return UtteranceScore(
        text=text,
        n_words=n,
        self_agency=round(self_agency, 4),
        future_orientation=round(future_orientation, 4),
        emotional_granularity=round(min(1.0, emotional_granularity), 4),
        problem_ownership=round(problem_ownership, 4),
        reflection_depth=round(min(1.0, reflection_depth), 4),
        hopelessness=round(hopelessness, 4),
        solution_focus=round(solution_focus, 4),
        absolutism=round(absolutism, 4),
        self_focus_balance=round(self_focus_balance, 4),
        rumination=round(rumination, 4),
        evidence=evidence,
    )


# --------------------------------------------------------------------------
# session level
# --------------------------------------------------------------------------

def session_features(
    client_utterances: list[str], min_words: int = 5
) -> tuple[dict[str, float], list[UtteranceScore]]:
    """
    Aggregate client turns into one session-level vector.

    Turns are weighted by sqrt(length): a 200-word narrative should count for
    more than a two-word "yeah", but not 100x more, or one long monologue
    would swamp the session.
    """
    scored = [score_utterance(u, keep_evidence=True) for u in client_utterances]
    usable = [s for s in scored if s.n_words >= min_words]
    if not usable:
        usable = scored or [score_utterance("")]

    weights = [max(1.0, s.n_words) ** 0.5 for s in usable]
    wsum = sum(weights) or 1.0

    feats: dict[str, float] = {}
    for dim in DIMENSIONS + AUXILIARY:
        feats[dim] = round(
            sum(getattr(s, dim) * w for s, w in zip(usable, weights)) / wsum, 4
        )

    feats["n_client_turns"] = float(len(scored))
    feats["n_client_words"] = float(sum(s.n_words for s in scored))
    feats["mean_turn_length"] = round(
        feats["n_client_words"] / max(1, feats["n_client_turns"]), 2
    )
    return feats, scored
