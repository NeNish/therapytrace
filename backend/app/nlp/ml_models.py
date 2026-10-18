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

  * The behaviour model sets the coarse family in M6 — question, reflection,
    therapist input or other — and the regex taxonomy then names the subtype
    inside that family. Therapist attribution therefore no longer depends on
    hand-written patterns to decide *what kind of thing* a turn was, only to
    distinguish open from closed questions and simple from complex reflections
    once the family is settled. With no artifacts present the patterns handle
    both jobs, as before.

Both models are calibrated on English motivational-interviewing demonstrations.
Applying them to other modalities is an assumption, not a guarantee, and the
`available` flag plus the returned confidence let callers act accordingly.
"""

from __future__ import annotations

import functools
import json
import math
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

# Held-out confusion matrix for the deployed client model (logistic regression),
# rows = true label, columns = predicted, order change / neutral / sustain.
# Source: ml/artifacts/benchmark.json, 1,656 utterances from 34 conversations
# the model never saw.
CLIENT_CONFUSION = {
    "labels": ["change", "neutral", "sustain"],
    "matrix": [
        [240, 147, 62],
        [253, 561, 103],
        [89, 104, 97],
    ],
}


@functools.lru_cache(maxsize=1)
def _directional_matrix() -> tuple[float, float, float, float]:
    """
    P(predicted | true) restricted to the two directional classes.

    Counting predicted labels and dividing is a biased estimate of how much
    change talk a session contained, because the classifier is imperfect and
    its errors are not symmetric. Sustain talk is the weak class: it is
    recovered as sustain about a third of the time and read as *change* about
    a third of the time, and both of those mistakes push the ratio the same
    way. The bias that follows is systematic, so it does not shrink as a
    session gets longer — a client who is evenly split would read as roughly
    three-quarters change talk however much they said.

    Returns (cc, cs, sc, ss): the probability of predicting change or sustain
    given each true directional label, renormalised over the two.
    """
    m = CLIENT_CONFUSION["matrix"]
    i_c, i_s = 0, 2  # change, sustain
    c_row = m[i_c][i_c] + m[i_c][i_s]
    s_row = m[i_s][i_c] + m[i_s][i_s]
    return (
        m[i_c][i_c] / c_row, m[i_c][i_s] / c_row,
        m[i_s][i_c] / s_row, m[i_s][i_s] / s_row,
    )


def corrected_change_ratio(n_change: int, n_sustain: int) -> dict:
    """
    De-bias the change-talk ratio with adjusted classify-and-count.

    If the predicted split is q and the true prevalence is p, then q = Mᵀp for
    the misclassification matrix M. M is known from held-out data, so p is
    recovered by solving that 2x2 system and clipping back onto the simplex.

    On simulated sessions this moves the displayed value from 0.75 to about
    0.57 for a client who is genuinely evenly split, and restores the slope
    between true and displayed ratio from 0.27 to 0.80 — meaning a client who
    is changing a great deal stops looking like one who is barely changing.
    The estimate becomes noisier in exchange, which is the honest trade: it is
    better to be imprecise than to be confidently wrong in one direction.

    Returns {"ratio", "lo", "hi", "wide"}; `ratio` is None with nothing to
    estimate from.

    The estimate is a Bayesian posterior rather than a direct matrix inversion,
    and that choice came from running the inversion on real sessions. Solving
    the 2x2 system algebraically is the textbook adjusted classify-and-count,
    and on long sessions it is fine — but therapy sessions here carry seven to
    fifteen directional turns, and at that size the solve routinely lands
    outside [0, 1] and has to be clamped. Eight of twelve demo sessions pinned
    to exactly 0.00 or 1.00: a step function between two extremes, stated with
    total confidence. That is a worse failure than the upward bias it was meant
    to remove.

    Instead: the classifier predicts change with probability

        theta(p) = p * P(change|change) + (1 - p) * P(change|sustain)

    so the observed count is binomial in theta(p). A flat prior over p and a
    grid evaluation give a posterior whose mean is always inside [0, 1] by
    construction, shrinks toward the middle when the evidence is thin, and
    yields a credible interval the interface can show. No clamping anywhere.

    `wide` marks an interval too broad to read as a point estimate, which is
    the honest state for a short session rather than something to hide.
    """
    total = n_change + n_sustain
    if total == 0:
        return {"ratio": None, "lo": None, "hi": None, "wide": False}

    cc, _cs, sc, _ss = _directional_matrix()
    if abs(cc - sc) < 1e-6:
        # The classifier cannot separate the classes at all; nothing to invert.
        return {"ratio": None, "lo": None, "hi": None, "wide": True}

    steps = 400
    grid = [i / steps for i in range(steps + 1)]
    logliks = []
    for p in grid:
        theta = min(max(p * cc + (1.0 - p) * sc, 1e-9), 1.0 - 1e-9)
        logliks.append(n_change * math.log(theta) + n_sustain * math.log1p(-theta))

    peak = max(logliks)
    weights = [math.exp(l - peak) for l in logliks]
    mass = sum(weights)
    if mass <= 0:
        return {"ratio": None, "lo": None, "hi": None, "wide": True}

    mean = sum(p * w for p, w in zip(grid, weights)) / mass

    # Central 80% credible interval, walked off the cumulative posterior.
    cum, lo, hi = 0.0, grid[0], grid[-1]
    got_lo = False
    for p, w in zip(grid, weights):
        cum += w / mass
        if not got_lo and cum >= 0.10:
            lo, got_lo = p, True
        if cum >= 0.90:
            hi = p
            break

    return {
        "ratio": round(mean, 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "wide": (hi - lo) > 0.45,
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

    # Both numbers are returned. The raw count is what the classifier literally
    # said; the corrected one is what the interface should show. Keeping both
    # makes the correction auditable rather than invisible, and lets a reviewer
    # see how large it is on any given session.
    raw = round(n_change / denom, 4) if denom else None
    adj = corrected_change_ratio(n_change, n_sustain)
    corrected = adj["ratio"]

    return {
        "labels": labels,
        "probabilities": probs,
        "n_change": n_change,
        "n_sustain": n_sustain,
        "n_neutral": labels.count("neutral"),
        "change_talk_ratio": corrected if corrected is not None else raw,
        "change_talk_ratio_raw": raw,
        "change_talk_lo": adj["lo"],
        "change_talk_hi": adj["hi"],
        "bias_corrected": corrected is not None,
        "estimate_is_wide": adj["wide"],
        "directional_turns": denom,
        "model_metrics": REPORTED_METRICS["client"],
        "caveat": (
            "Trained on English motivational-interviewing demonstrations. "
            "Macro-F1 0.49 on unseen conversations — treat individual turn "
            "labels as noisy and read the session-level ratio instead. The "
            "ratio is corrected for known classifier bias; uncorrected, it "
            "reads about 0.75 for a client who is genuinely evenly split."
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


# ---------------------------------------------------------------- M26
_RESOLVABILITY_PATH = (
    Path(__file__).resolve().parents[3] / "ml" / "artifacts" / "resolvability.json"
)


@functools.lru_cache(maxsize=1)
def _resolvability_table() -> dict | None:
    """The measured precision table, or None if it has not been generated."""
    try:
        return json.loads(_RESOLVABILITY_PATH.read_text())
    except (OSError, ValueError):
        return None


def resolvability(n_turns: int) -> dict:
    """
    Say whether a session is long enough for its ratio to mean anything.

    The estimate is always computable; that is not the same as it being
    readable. At the session lengths this corpus actually contains, the 80%
    credible interval spans roughly two-thirds of the scale, so the midpoint
    carries almost no information about where the client sits. Showing it
    without saying so invites a reader to treat noise as a finding, which is
    the failure this whole project exists to avoid.

    The verdict comes from ml/artifacts/resolvability.json, a table measured by
    simulating sessions through the estimator that actually ships. It is not a
    fitted curve and not a rule of thumb. If the table is missing the function
    says so rather than guessing: no table, no verdict.

    Returns a dict carrying the verdict, the measured interval width, how many
    more turns would be needed, and which reading the data does support.
    """
    tab = _resolvability_table()
    if tab is None:
        return {
            "available": False,
            "reason": "resolvability table not generated; run ml/resolvability.py",
        }

    rows = tab["table"]
    row = min(rows, key=lambda r: abs(r["turns"] - n_turns))
    need = tab.get("turns_for_readable_estimate")
    readable = bool(need is not None and n_turns >= need)

    if readable:
        verdict = "This session is long enough to read as a single value."
    elif need is not None:
        verdict = (
            f"Too short to read on its own: {n_turns} directional turns give an "
            f"80% interval about {row['ci80_width']:.2f} wide on a 0-1 scale. "
            f"A single-session reading needs roughly {need}. Read the direction "
            f"across sessions instead."
        )
    else:
        verdict = "No session length in the calibration reaches a readable interval."

    return {
        "available": True,
        "n_turns": int(n_turns),
        "readable": readable,
        "ci80_width": row["ci80_width"],
        "posterior_sd": row["posterior_sd"],
        "min_detectable_difference": row["min_detectable_difference"],
        "turns_for_readable_estimate": need,
        "turns_short_by": (None if readable or need is None else max(0, need - int(n_turns))),
        "supported_reading": "trend across sessions" if not readable else "single session",
        "verdict": verdict,
        "assumptions": tab.get("assumptions"),
    }
