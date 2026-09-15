"""
M10 — supervised classifiers (AnnoMI).

Two models trained on expert annotations, loaded lazily so the system runs
unchanged when the artifacts are absent (fresh clone, CI, or a deployment that
chooses not to ship them).

  change-talk model     client utterance -> change / sustain / neutral
  behaviour model       therapist utterance -> question / reflection /
                                               therapist_input / other

Why this matters beyond adding a number:

  * Change talk is the field's hand-labelled proxy for therapeutic momentum.
    Correlating the session-level change-talk ratio with the TPI is external
    validation of the five lexicon dimensions against expert judgement —
    validation check 3 in the methodology.

  * The behaviour model replaces the regex intervention taxonomy in M6 with a
    classifier trained on MISC-coded data, so therapist attribution stops
    depending on hand-written patterns.

Both models are calibrated on English motivational-interviewing demonstrations.
Applying them to other modalities is an assumption, not a guarantee, and the
`available` flag plus the returned confidence let callers act accordingly.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

ARTIFACTS = Path(__file__).resolve().parents[3] / "ml" / "artifacts"

CLIENT_MODEL = ARTIFACTS / "model_client.joblib"
THERAPIST_MODEL = ARTIFACTS / "model_therapist.joblib"

# Held-out performance, recorded so callers can weight predictions honestly.
REPORTED_METRICS = {
    "client": {"accuracy": 0.54, "macro_f1": 0.49},
    "therapist": {"accuracy": 0.65, "macro_f1": 0.61},
}


@functools.lru_cache(maxsize=2)
def _load(path_str: str) -> dict[str, Any] | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        return None


def available() -> dict[str, bool]:
    return {
        "change_talk": _load(str(CLIENT_MODEL)) is not None,
        "therapist_behaviour": _load(str(THERAPIST_MODEL)) is not None,
    }


def _frame(texts: list[str], prev: list[str] | None = None):
    """
    Built lazily so that pandas is only required when a trained model is
    actually being used. The text pipeline must start on a machine that has
    never installed the ML stack.
    """
    import pandas as pd

    prev = prev or [""] * len(texts)
    return pd.DataFrame({"text": texts, "prev_text": prev})


def classify_client_turns(texts: list[str], prev: list[str] | None = None) -> dict | None:
    """
    Returns per-utterance labels plus the session-level change-talk ratio:

        ratio = change / (change + sustain)

    which is the quantity motivational-interviewing research associates with
    outcome. Neutral turns are excluded from the denominator because they carry
    no directional information.
    """
    bundle = _load(str(CLIENT_MODEL))
    if bundle is None or not texts:
        return None

    model = bundle["model"]
    labels = list(model.predict(_frame(texts, prev)))

    probs = None
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(_frame(texts, prev))
            classes = list(model.classes_)
            probs = [
                {c: round(float(row[i]), 4) for i, c in enumerate(classes)} for row in p
            ]
        except Exception:
            probs = None

    n_change = labels.count("change")
    n_sustain = labels.count("sustain")
    denom = n_change + n_sustain

    return {
        "labels": labels,
        "probabilities": probs,
        "n_change": n_change,
        "n_sustain": n_sustain,
        "n_neutral": labels.count("neutral"),
        "change_talk_ratio": round(n_change / denom, 4) if denom else None,
        "directional_turns": denom,
        "model_metrics": REPORTED_METRICS["client"],
        "caveat": (
            "Trained on English motivational-interviewing demonstrations. "
            "Macro-F1 0.49 on unseen conversations — treat individual turn "
            "labels as noisy and read the session-level ratio instead."
        ),
    }


def classify_therapist_turns(texts: list[str]) -> dict | None:
    bundle = _load(str(THERAPIST_MODEL))
    if bundle is None or not texts:
        return None

    labels = list(bundle["model"].predict(_frame(texts)))
    counts: dict[str, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1

    total = len(labels) or 1
    return {
        "labels": labels,
        "counts": counts,
        "proportions": {k: round(v / total, 4) for k, v in counts.items()},
        "reflection_to_question_ratio": (
            round(counts.get("reflection", 0) / counts["question"], 3)
            if counts.get("question") else None
        ),
        "model_metrics": REPORTED_METRICS["therapist"],
    }


def session_ml_summary(
    client_texts: list[str],
    therapist_texts: list[str],
    prev_for_client: list[str] | None = None,
) -> dict:
    """Everything M10 contributes to one session analysis."""
    client = classify_client_turns(client_texts, prev_for_client)
    therapist = classify_therapist_turns(therapist_texts)
    return {
        "available": available(),
        "client_talk": client,
        "therapist_behaviour": therapist,
        "dataset": "AnnoMI (Wu et al., 2023) — 133 expert-annotated MI dialogues",
    }
