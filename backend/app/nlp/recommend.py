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



def _clean_excerpt(text: str | None, max_len: int = 80) -> str:
    """Clean a client excerpt to read smoothly without cut-offs or prompt artifacts."""
    if not text:
        return ""
    t = text.strip().replace("\n", " ")
    # Pick the first sentence if multiple
    for end_punct in [". ", "? ", "! "]:
        if end_punct in t:
            t = t.split(end_punct)[0] + end_punct.strip()
            break
    if len(t) > max_len:
        t = t[:max_len].rsplit(" ", 1)[0] + "…"
    return t


def generate_suggested_questions(
    tpi_series: list[float],
    features_series: list[dict],
    dimension_series: dict[str, list[float]],
    baseline: dict,
    trajectory: dict | None = None,
    session_data: dict | None = None,
) -> dict:
    """
    Generate authentic, clinically attuned follow-up questions for the therapist.
    Rather than robotic formulaic strings or artificial word-frequency snippets,
    these are framed as warm, conversational inquiries rooted in the client's actual
    narrative and present moment.
    """
    if not tpi_series or not features_series:
        return {"available": False, "questions": [], "rationale": "Requires at least one completed session."}

    session_idx = len(tpi_series)
    curr_tpi = tpi_series[-1]
    last_feats = features_series[-1]
    prev_tpi = tpi_series[-2] if len(tpi_series) >= 2 else 50.0
    tpi_delta = curr_tpi - prev_tpi

    session_data = session_data or {}
    drivers = session_data.get("drivers") or {}
    weakest_turns = drivers.get("weakest") or []
    strongest_turns = drivers.get("strongest") or []

    # Get the cleanest meaningful client excerpts from distinct turns
    excerpt_stuck = _clean_excerpt(weakest_turns[0]["text"] if weakest_turns else None)
    excerpt_stuck2 = _clean_excerpt(weakest_turns[1]["text"] if len(weakest_turns) > 1 else None)
    excerpt_gain = _clean_excerpt(strongest_turns[0]["text"] if strongest_turns else None)

    # Read z-scores directly from session_data if provided, or compute against baseline
    z_scores_dict = session_data.get("z_scores") or {}
    means = baseline.get("means") or baseline.get("mean") or {}
    sds = baseline.get("sds") or baseline.get("sd") or {}

    dim_deltas = []
    for d in DIMENSIONS:
        if d in z_scores_dict:
            z = float(z_scores_dict[d])
            val = last_feats.get(d, 0.0)
        else:
            val = last_feats.get(d, 0.0)
            m = means.get(d, val)
            s = sds.get(d, 1.0) or 1.0
            z = (val - m) / s
        dim_deltas.append((d, z, val))

    dim_deltas.sort(key=lambda x: x[1])
    lagging_dim, lowest_z, lowest_raw = dim_deltas[0]
    second_dim, second_z, _ = dim_deltas[1]
    surging_dim, highest_z, _ = dim_deltas[-1]

    questions: list[dict] = []

    # -------------------------------------------------------------
    # 1. PRIMARY FOCUS: Grounded in what felt most difficult / constrained
    # -------------------------------------------------------------
    if lagging_dim == "self_agency":
        if excerpt_stuck:
            q_text = (
                f"You mentioned last time that {excerpt_stuck.lower().rstrip('.')} — "
                f"when things start feeling that overwhelming or out of your hands, what is one small thing you can still decide for yourself?"
            )
        else:
            q_text = (
                "When you find yourself feeling boxed in by situations or other people, "
                "where do you feel you still have a little room to breathe and make your own choice?"
            )
        questions.append({
            "category": "Exploring Autonomy & Choices",
            "question": q_text,
            "clinical_intent": (
                "Gently explores the client's sense of personal agency without putting pressure on them "
                f"(Agency z = {lowest_z:+.2f} SD vs baseline)."
            ),
            "metric_driver": f"Self-Agency ({lowest_z:+.2f} SD)",
        })

    # -------------------------------------------------------------
    # 1. PRIMARY CLINICAL QUESTION: Tailored to what was most constrained/stuck
    # -------------------------------------------------------------
    if lagging_dim == "future_orientation":
        if last_feats.get("rumination", 0) > 0.18:
            q_text = (
                f"It sounded like your thoughts were spending a lot of energy replaying past events. "
                f"If you were to gently look toward the upcoming week, is there one moment you'd like to approach a little differently?"
            )
            intent = "Invites the client to step out of retrospective loops into proactive mental rehearsal."
        elif excerpt_stuck and "numb" in excerpt_stuck.lower():
            q_text = (
                f"You spoke about how '{excerpt_stuck.rstrip('.')}' — when feelings start returning, even painful ones, "
                f"what helps you feel steady enough to stay with them rather than shutting down?"
            )
            intent = "Supports emotional tolerance and pacing in grief/avoidance processing."
        elif excerpt_stuck and ("decision" in excerpt_stuck.lower() or "asking" in excerpt_stuck.lower() or "over me" in excerpt_stuck.lower()):
            q_text = (
                f"Thinking back to when '{excerpt_stuck.rstrip('.')}' — in the week ahead, "
                f"is there a conversation where you'd like to practice naming what you need earlier, before it builds up?"
            )
            intent = "Scaffolds forward assertiveness in relational conflict."
        elif excerpt_stuck and ("reached out" in excerpt_stuck.lower() or "got outside" in excerpt_stuck.lower()):
            q_text = (
                f"You mentioned that '{excerpt_stuck.rstrip('.')}' — even if that felt small or exhausting, "
                f"what is one gentle step you could see yourself taking in the next few days that honors that same effort?"
            )
            intent = "Gently builds behavioural activation momentum against depressive fatigue."
        elif excerpt_stuck:
            q_text = (
                f"Reflecting on when you shared that '{excerpt_stuck.rstrip('.')}' — "
                f"if you imagine a day in the near future where that feels even slightly lighter, what does that day look like?"
            )
            intent = "Encourages future horizon imagery to counterbalance stuckness."
        else:
            q_text = (
                "Looking ahead to the coming days, what is one hopeful or practical thing you would like to keep your eyes on?"
            )
            intent = "Encourages forward-looking momentum."

        questions.append({
            "category": "Looking Forward & Anticipation",
            "question": q_text,
            "clinical_intent": intent,
            "metric_driver": f"Future Horizon ({lowest_z:+.2f} SD)",
        })

    elif lagging_dim == "self_agency":
        if excerpt_stuck:
            q_text = (
                f"When you reflected that '{excerpt_stuck.rstrip('.')}', it sounded like a situation where you felt quite constrained. "
                f"Looking back at that moment, what is one small boundary or response that could still belong purely to you?"
            )
        else:
            q_text = (
                "When you find yourself feeling boxed in by situations or other people, "
                "where do you feel you still have a little room to breathe and make your own choice?"
            )
        questions.append({
            "category": "Exploring Autonomy & Choices",
            "question": q_text,
            "clinical_intent": (
                "Gently explores the client's sense of personal agency without putting pressure on them "
                f"(Agency z = {lowest_z:+.2f} SD vs baseline)."
            ),
            "metric_driver": f"Self-Agency ({lowest_z:+.2f} SD)",
        })

    elif lagging_dim == "problem_ownership":
        if excerpt_stuck:
            q_text = (
                f"When describing how '{excerpt_stuck.rstrip('.')}' — "
                f"in dynamics like that, what feels like your responsibility to carry, and what feels like it belongs to someone else?"
            )
        else:
            q_text = (
                "When difficulties arise, what part of the situation feels like it's within your power to influence, "
                "and what parts do you need permission to let go of?"
            )
        questions.append({
            "category": "Clarifying Boundaries & Responsibility",
            "question": q_text,
            "clinical_intent": (
                "Helps differentiate healthy responsibility from external helplessness or inappropriate self-blame "
                f"(Problem Ownership z = {lowest_z:+.2f} SD)."
            ),
            "metric_driver": f"Problem Ownership ({lowest_z:+.2f} SD)",
        })

    elif lagging_dim == "emotional_granularity":
        if excerpt_stuck:
            q_text = (
                f"When you described feeling that way — '{excerpt_stuck.rstrip('.')}' — "
                f"if you were to pause and really check in with your body, what feelings were sitting right beneath the surface?"
            )
        else:
            q_text = (
                "When that heavy or uneasy feeling shows up, what specific emotions do you notice coming along with it — "
                "is it frustration, exhaustion, fear, or something else?"
            )
        questions.append({
            "category": "Untangling Underlying Emotions",
            "question": q_text,
            "clinical_intent": (
                "Helps the client develop richer emotional vocabulary rather than staying in diffuse distress "
                f"(Emotional Granularity z = {lowest_z:+.2f} SD)."
            ),
            "metric_driver": f"Emotional Granularity ({lowest_z:+.2f} SD)",
        })

    elif lagging_dim == "reflection_depth":
        if excerpt_stuck:
            q_text = (
                f"Looking back at when you noticed that '{excerpt_stuck.rstrip('.')}' — "
                f"what do you think that moment was trying to tell you about what you genuinely needed?"
            )
        else:
            q_text = (
                "Stepping back from the busy sequence of what happened this week, "
                "what pattern or recurring feeling stands out most clearly to you?"
            )
        questions.append({
            "category": "Deepening Insight & Meaning",
            "question": q_text,
            "clinical_intent": (
                "Invites psychological reflection beyond storytelling to uncover underlying patterns "
                f"(Reflection Depth z = {lowest_z:+.2f} SD)."
            ),
            "metric_driver": f"Reflection Depth ({lowest_z:+.2f} SD)",
        })

    # -------------------------------------------------------------
    # 2. SECONDARY / COMPLEMENTARY CLINICAL INQUIRY
    # -------------------------------------------------------------
    if second_dim == "problem_ownership":
        q2 = (
            "When navigating stress this week, what helped you decide which problems were genuinely yours to solve "
            "versus things outside your control?"
        )
        cat2 = "Discerning Personal Influence"
        intent2 = "Encourages healthy boundary discernment and realistic locus of control."
    elif second_dim == "reflection_depth":
        q2 = (
            "When you stepped back from the daily rush between our sessions, "
            "what realization or pattern about yourself came into view most clearly?"
        )
        cat2 = "Reflective Pattern Recognition"
        intent2 = "Fosters self-observing capacity and internal psychological attunement."
    elif second_dim == "emotional_granularity":
        q2 = (
            "When you felt most unsettled this past week, what was the physical sensation or mood that you carried most strongly?"
        )
        cat2 = "Emotional Awareness & Grounding"
        intent2 = "Encourages the client to name somatic and affective states with greater nuance."
    elif second_dim == "self_agency":
        q2 = (
            "In moments where the week felt chaotic, were there any quiet boundaries or small choices that helped you stay anchored?"
        )
        cat2 = "Recognizing Inner Strengths"
        intent2 = "Reinforces client autonomy and small self-protective boundaries."
    else:
        q2 = (
            "Outside of the immediate difficulties we've been working through, "
            "is there something in the coming week you're quietly hoping goes well?"
        )
        cat2 = "Fostering Tomorrow's Hopes"
        intent2 = "Provides balance by inquiring about forward anticipation and everyday resilience."

    questions.append({
        "category": cat2,
        "question": q2,
        "clinical_intent": intent2,
        "metric_driver": f"Complementary Focus ({second_dim.replace('_', ' ').title()})",
    })

    # -------------------------------------------------------------
    # 3. CONTEXTUAL SIGNAL (Velocity Shift, Rupture Check, or Breakthrough Anchor)
    # -------------------------------------------------------------
    if tpi_delta <= -6.0:
        questions.append({
            "category": "Checking In on the Relationship",
            "question": (
                "Before we jump into new things today, I wanted to pause and check in: "
                "how did our last session sit with you after you left? Did anything feel rushed or misunderstood?"
            ),
            "clinical_intent": (
                f"Addresses an observed drop in session momentum ({tpi_delta:+.1f} pts) through supportive metacommunication."
            ),
            "metric_driver": f"Momentum Shift ({tpi_delta:+.1f} pts)",
        })
    elif tpi_delta >= 7.0 or curr_tpi >= 60.0:
        questions.append({
            "category": "Consolidating Breakthrough Moments",
            "question": (
                "You expressed a great deal of clarity and openness in our last session. "
                "What allowed you to tap into that strength, and how has that felt in your day-to-day life since?"
            ),
            "clinical_intent": (
                f"Helps anchor the client's recent positive breakthrough (+{tpi_delta:.1f} pts gain) into lasting self-concept."
            ),
            "metric_driver": f"Positive Shift (+{tpi_delta:.1f} pts)",
        })
    elif last_feats.get("hopelessness", 0) > 0.25:
        questions.append({
            "category": "Nurturing Resilience in Low Moments",
            "question": (
                "When that familiar feeling tells you that things won't change, "
                "what is the quiet part of you that still chooses to show up and give therapy a chance?"
            ),
            "clinical_intent": "Gently explores the client's innate resilience and underlying values during discouraging phases.",
            "metric_driver": "Presence of Depressive Strain",
        })

    # Ensure 2 to 3 high-quality questions
    selected_questions = questions[:3]

    rationale = (
        f"Prepared for Session {session_idx}: focused on supportive exploration of {lagging_dim.replace('_', ' ')} "
        f"and the client's current emotional state, framed in natural clinical language."
    )

    return {
        "available": True,
        "session_number": session_idx,
        "target_dimension": lagging_dim,
        "target_z": round(lowest_z, 2),
        "tpi": curr_tpi,
        "tpi_delta": round(tpi_delta, 1),
        "questions": selected_questions,
        "clinical_rationale": rationale,
    }



# --------------------------------------------------------------------------

def recommend(
    tpi_series: list[float],
    features_series: list[dict],
    dimension_series: dict[str, list[float]],
    baseline: dict,
    trajectory: dict | None = None,
    profile: dict | None = None,
    session_data: dict | None = None,
) -> dict:
    """Everything M17 produces for one client, strictly based on recent data."""
    resp = (profile or {}).get("response")
    return {
        "targeting": target_dimensions(dimension_series, baseline),
        "early_warning": early_warning(tpi_series, features_series, trajectory),
        "next_session": next_session_suggestion(tpi_series, resp),
        "suggested_questions": generate_suggested_questions(
            tpi_series, features_series, dimension_series, baseline, trajectory, session_data
        ),
        "principle": (
            "Decision support for reflection and supervision. The system "
            "suggests what to consider and shows its reasoning; the clinical "
            "decision remains entirely with the clinician."
        ),
    }
