"""
M17 — Recommendation Engine.

The objection this module answers: a progress bar tells you where you are, not
what to do. Measurement without a next step is a dashboard, and dashboards get
looked at twice and then ignored.

Three outputs, in increasing order of how much they stick their neck out:

  1. DIMENSION TARGETING — which of the five is actually stuck. "The index fell"
     is not actionable; "ownership has sat below baseline for four sessions
     while everything else improved" is.

  2. EARLY WARNING — flag a likely decline one session ahead rather than
     reporting it after. Built from the slope of recent change plus the
     auxiliary risk signals, deliberately tuned to be specific rather than
     sensitive: a warning that fires often gets ignored, and an ignored warning
     is worse than none.

  3. NEXT-SESSION SUGGESTION — what to try, grounded in two things this system
     already knows. First, the AnnoMI finding that intervention effect is
     conditional on client state: questions and reflections roughly double
     their yield when a client already has momentum (26.8% -> 47.9% and
     26.0% -> 45.5% change talk) while advice-giving is flat across states
     (25.3% -> 29.6%, n.s.). Second, this client's own response profile from
     M11, which overrides the corpus finding wherever enough of their own data
     exists.

Standing caveat, carried in every response: these are observational patterns,
confounded by indication, from short series. The engine suggests what to
consider, and says why, so a clinician can disagree with the reasoning rather
than just the conclusion. It never prescribes.
"""

from __future__ import annotations

from typing import Any

from .features import DIMENSIONS
from .narrative import DIM_CONCRETE, DIM_WORDS

# From the AnnoMI study: P(next client turn is change talk), by state.
CORPUS_PRIOR = {
    "stuck": {"question": 0.268, "reflection": 0.260, "other": 0.244, "therapist_input": 0.253},
    "middling": {"question": 0.412, "reflection": 0.382, "other": 0.331, "therapist_input": 0.265},
    "moving": {"question": 0.479, "reflection": 0.455, "other": 0.415, "therapist_input": 0.296},
}

READABLE = {
    "question": "open questions",
    "reflection": "reflections",
    "therapist_input": "advice or information giving",
    "other": "other moves",
    "complex_reflection": "complex reflections",
    "simple_reflection": "simple reflections",
    "open_question": "open questions",
    "closed_question": "closed questions",
    "affirmation": "affirmations",
    "challenge": "challenges",
    "directive": "directives or tasks",
    "summary": "summaries",
    "psychoeducation": "psychoeducation",
    "minimal_encourager": "minimal encouragers",
}


# --------------------------------------------------------------------------
# 1. dimension targeting
# --------------------------------------------------------------------------

def target_dimensions(dimension_series: dict[str, list[float]], baseline: dict) -> dict:
    """Which dimension is holding the client back, and which is carrying them."""
    means = (baseline or {}).get("means", {})
    sds = (baseline or {}).get("sds", {})
    rows = []

    for d in DIMENSIONS:
        series = dimension_series.get(d) or []
        if len(series) < 3 or d not in means:
            continue
        recent = series[-3:]
        z = (sum(recent) / len(recent) - means[d]) / (sds.get(d) or 1e-6)
        early = series[: max(2, len(series) // 3)]
        drift = (sum(recent) / len(recent)) - (sum(early) / len(early))
        rows.append({
            "dimension": d,
            "label": DIM_WORDS[d],
            "recent_z": round(z, 3),
            "drift_since_start": round(drift, 4),
            "sessions_below_baseline": sum(1 for v in series if v < means[d]),
            "n_sessions": len(series),
        })

    if not rows:
        return {"available": False, "reason": "needs at least three scored sessions"}

    rows.sort(key=lambda r: r["recent_z"])
    stuck, strongest = rows[0], rows[-1]
    up, down = DIM_CONCRETE[stuck["dimension"]]

    # The laggard is only "below baseline" if it actually is. When every
    # dimension has risen, saying otherwise would misrepresent a good session.
    if stuck["recent_z"] < -0.3:
        summary = (
            f"{stuck['label'].capitalize()} is sitting furthest below this client's "
            f"baseline ({stuck['recent_z']:+.1f} SD over the last three sessions). "
            f"In practice that means {down}. "
            f"{strongest['label'].capitalize()} is the one carrying them "
            f"({strongest['recent_z']:+.1f} SD)."
        )
        suggestion = (
            f"If you want a focus for the next session, {stuck['label']} is where the "
            f"movement is missing — you would be looking for {up}."
        )
    elif strongest["recent_z"] > 0.3:
        summary = (
            f"Every dimension is at or above this client's baseline. "
            f"{strongest['label'].capitalize()} has moved most "
            f"({strongest['recent_z']:+.1f} SD); {stuck['label']} has moved least "
            f"({stuck['recent_z']:+.1f} SD), though it is not below baseline."
        )
        suggestion = (
            f"Nothing is holding this client back at present. If you want somewhere "
            f"to push, {stuck['label']} has the most headroom — you would be looking "
            f"for {up}."
        )
    else:
        summary = (
            f"All five dimensions are sitting close to this client's baseline, with "
            f"{strongest['label']} highest ({strongest['recent_z']:+.1f} SD) and "
            f"{stuck['label']} lowest ({stuck['recent_z']:+.1f} SD)."
        )
        suggestion = (
            f"No dimension stands out either way. {stuck['label'].capitalize()} is "
            f"the least active if you want a focus."
        )

    return {
        "available": True,
        "all": rows,
        "stuck_dimension": stuck,
        "strongest_dimension": strongest,
        "all_above_baseline": bool(stuck["recent_z"] >= -0.3),
        "summary": summary,
        "suggestion": suggestion,
    }


# --------------------------------------------------------------------------
# 2. early warning
# --------------------------------------------------------------------------

def early_warning(tpi_series: list[float], features_series: list[dict],
                  trajectory: dict | None = None) -> dict:
    """
    Deliberately specific rather than sensitive. Each condition is a clinically
    meaningful pattern on its own, and the alert only fires when enough of them
    coincide that a supervisor would want to know.
    """
    if len(tpi_series) < 4:
        return {"available": False, "reason": "needs at least four scored sessions"}

    reasons: list[str] = []
    risk = 0.0

    recent = tpi_series[-3:]
    slope = (recent[-1] - recent[0]) / 2
    if slope < -2.0:
        risk += 0.35
        reasons.append(f"the index has dropped {abs(slope) * 2:.0f} points over three sessions")
    elif slope < -0.8:
        risk += 0.18
        reasons.append("the index has been drifting down over recent sessions")

    if tpi_series[-1] < 44:
        risk += 0.2
        reasons.append(f"the latest session ({tpi_series[-1]:.0f}) sits below their own baseline")

    if len(tpi_series) >= 5:
        spread = max(tpi_series[-4:]) - min(tpi_series[-4:])
        if spread > 18:
            risk += 0.15
            reasons.append("session-to-session variation has become large")

    last = features_series[-1] if features_series else {}
    if last.get("hopelessness", 0) > 0.3:
        risk += 0.25
        reasons.append("hopelessness-associated language rose in the latest session")
    if last.get("rumination", 0) > 0.3:
        risk += 0.12
        reasons.append("repetitive backward-looking language rose")

    if len(features_series) >= 3:
        wl = [f.get("n_client_words", 0) for f in features_series]
        prior = sum(wl[:-1]) / max(1, len(wl) - 1)
        if prior and wl[-1] < 0.6 * prior:
            risk += 0.2
            reasons.append(f"the client spoke {100 - int(100 * wl[-1] / prior)}% less than usual")

    if trajectory and trajectory.get("momentum", {}).get("state") == "plateauing":
        if len(tpi_series) >= 6:
            risk += 0.1
            reasons.append("the index has been flat for several sessions")

    risk = min(1.0, risk)
    level = "elevated" if risk >= 0.55 else "watch" if risk >= 0.3 else "none"

    return {
        "available": True,
        "risk_score": round(risk, 3),
        "level": level,
        "reasons": reasons,
        "message": (
            "Nothing in the recent pattern stands out." if level == "none"
            else (f"Worth keeping an eye on: " + "; ".join(reasons) + ".")
            if level == "watch"
            else (f"Worth raising in supervision before the next session: "
                  + "; ".join(reasons) + ".")
        ),
        "caveat": (
            "A pattern-matching flag on a short series, not a clinical "
            "prediction. It is a prompt to look, never a reason to act alone."
        ),
    }


# --------------------------------------------------------------------------
# 3. next-session suggestion
# --------------------------------------------------------------------------

def _current_state(tpi: float) -> str:
    if tpi <= 45:
        return "stuck"
    if tpi >= 56:
        return "moving"
    return "middling"


def next_session_suggestion(
    tpi_series: list[float],
    response_profile: dict | None = None,
    therapist_impact: list[dict] | None = None,
) -> dict:
    """
    Prefers this client's own response history. Falls back to the corpus prior
    only when their own data is too thin, and says which one it used.
    """
    if not tpi_series:
        return {"available": False, "reason": "no scored sessions yet"}

    state = _current_state(tpi_series[-1])
    source = "corpus"
    ranked: list[tuple[str, float, int]] = []

    # this client's own profile, if M11 produced one
    if response_profile and response_profile.get("available"):
        cells = [c for c in response_profile.get("cells", [])
                 if c["state"] == state and c["n"] >= 4]
        # Only trust this client's own history if something in it actually
        # worked. If every move has a negative mean lift, recommending the
        # least-bad one would be worse than falling back to the corpus.
        if len(cells) >= 2 and max(c["mean_lift"] for c in cells) > 0.01:
            source = "this client's own history"
            ranked = [(c["intervention"], c["mean_lift"], c["n"]) for c in cells]
            ranked.sort(key=lambda r: r[1], reverse=True)

    if not ranked:
        prior = CORPUS_PRIOR[state]
        ranked = [(k, v, 0) for k, v in sorted(prior.items(), key=lambda kv: -kv[1])]

    best = ranked[0]
    avoid = ranked[-1] if len(ranked) > 2 else None
    nothing_worked = source != "corpus" and best[1] <= 0.01

    if source == "corpus":
        if state == "stuck":
            rationale = (
                "Across 123 annotated sessions, no therapist move worked well with a "
                "client in this state — all four converged around 24-27% change talk. "
                "The evidence says technique choice matters least here, so the "
                "question is less which move to pick than what might shift the state "
                "at all."
            )
        else:
            rationale = (
                f"Across 123 annotated sessions, {READABLE[best[0]]} were followed by "
                f"change talk {best[1]:.0%} of the time when a client was "
                f"{'already moving' if state == 'moving' else 'in this middle range'}, "
                f"against {ranked[-1][1]:.0%} for {READABLE[ranked[-1][0]]}."
            )
    else:
        rationale = (
            f"With this client specifically, {READABLE.get(best[0], best[0])} have been "
            f"followed by the largest shift in their next turn when they were "
            f"{state} (mean lift {best[1]:+.3f} over {best[2]} instances)."
        )

    return {
        "available": True,
        "client_state": state,
        "latest_tpi": tpi_series[-1],
        "evidence_source": source,
        "ranked": [
            {"intervention": k, "label": READABLE.get(k, k),
             "score": round(v, 4), "n": n}
            for k, v, n in ranked
        ],
        "suggestion": (
            f"The client is currently {state}. "
            + ("No move in this client's own history has been reliably followed by "
               "a shift in this state — worth discussing in supervision rather than "
               "picking a technique."
               if nothing_worked
               else f"Consider leaning on {READABLE.get(best[0], best[0])}."
               if state != "stuck" or source != "corpus"
               else "Consider what might move them out of a stuck state before "
                    "choosing a technique.")
        ),
        "rationale": rationale,
        "avoid": (
            f"{READABLE.get(avoid[0], avoid[0]).capitalize()} showed the weakest "
            f"association in this state." if avoid else None
        ),
        "caveat": (
            "Observational and confounded by indication — therapists choose moves "
            "in response to what was just said. This describes what has been "
            "followed by change, not what causes it."
        ),
    }


# --------------------------------------------------------------------------

def recommend(
    tpi_series: list[float],
    features_series: list[dict],
    dimension_series: dict[str, list[float]],
    baseline: dict,
    trajectory: dict | None = None,
    profile: dict | None = None,
) -> dict:
    """Everything M17 produces for one client."""
    resp = (profile or {}).get("response")
    return {
        "targeting": target_dimensions(dimension_series, baseline),
        "early_warning": early_warning(tpi_series, features_series, trajectory),
        "next_session": next_session_suggestion(tpi_series, resp),
        "principle": (
            "Decision support for reflection and supervision. The system "
            "suggests what to consider and shows its reasoning; the clinical "
            "decision remains entirely with the clinician."
        ),
    }
