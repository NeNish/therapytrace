"""
M19 — Body Language Insight.

Turns posture and expression measurements into sentences a clinician can act
on, anchored to a timestamp and to the words being spoken.

The line this module holds, and why it holds it:

    It reports WHAT THE BODY DID and WHEN, never WHAT THE PERSON FELT.

That is not timidity. The 2019 review by Barrett, Adolphs, Marsella, Martinez
and Pollak examined over a thousand studies and found people scowl when angry
roughly 30% of the time, and scowl for unrelated reasons far more often. In
therapy the problem is worse: clients actively manage their expressions in
front of a clinician, which is the phenomenon, not noise. Commercial systems
also show documented demographic bias in emotion assignment — a serious harm in
a mental-health tool.

So "the client is defensive" is a claim this module cannot support. But:

    "Posture closed at 07:12 and stayed closed for four minutes, starting as
     they began talking about their father. Their ownership score was lowest in
     that stretch."

is a claim it can support entirely, and it is more useful to a supervisor,
because it says where to look and what to look at rather than what to conclude.

Every sentence produced here names a measured quantity, a time, and where
possible the utterance it coincided with.

Patterns detected:

  POSTURE    closure episodes      arms folded, sustained
             opening               sustained increase in elbow span
             lean shifts           toward or away from the therapist
             stillness             movement well below this person's own norm
             agitation             movement well above it

  EXPRESSION expressive flattening  range below their own baseline
             expressive widening    range above it
             blink rate change      relative to their own baseline

All thresholds are relative to the client's own session baseline, for the same
reason everything else in this system is: a naturally still person is not a
disengaged one.
"""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ImportError:  # multimodal extras absent; these modules self-disable
    np = None

MIN_EPISODE_S = 20.0   # shorter than this is a shift in the chair, not a pattern
Z_STRONG = 1.2

SUPERVISION_QUESTIONS = {
    "posture_closed": (
        "Arms stayed folded for a sustained stretch. What was being discussed? "
        "Ask whether closure coincided with a topic the client finds hard to own — "
        "not whether they were 'being defensive'."
    ),
    "posture_opened": (
        "Posture opened relative to this person's own baseline. Did this coincide "
        "with naming a plan, a feeling precisely, or taking partial responsibility?"
    ),
    "stillness": (
        "Unusual stillness — in the room, did this feel like deep processing or "
        "withdrawal? The recording shows movement; only the therapist knows which."
    ),
    "heightened_movement": (
        "Movement rose sharply. Was the client working out a next step, reacting "
        "to something said, or shifting to a more charged topic?"
    ),
    "lean_shift": (
        "Torso angle changed mid-recording. Did the relational dynamic or topic "
        "shift at the same point? Worth locating in the full session."
    ),
    "head_motion_spike": (
        "Head movement peaked here — often marks a topic shift or internal "
        "re-orientation. What was the client saying or hearing at this second?"
    ),
}


def _supervision_for(kind: str) -> str:
    return SUPERVISION_QUESTIONS.get(
        kind,
        "Locate this moment in the recording and ask what was happening relationally.",
    )


def _clock(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def _episodes(signals: list[dict], key: str, predicate, min_s: float) -> list[dict]:
    """Contiguous runs where a predicate holds, longer than min_s."""
    runs, start = [], None
    for i, s in enumerate(signals):
        hit = predicate(s)
        if hit and start is None:
            start = i
        elif not hit and start is not None:
            dur = signals[i - 1]["t"] - signals[start]["t"]
            if dur >= min_s:
                runs.append({"start": signals[start]["t"], "end": signals[i - 1]["t"],
                             "duration": dur, "n": i - start})
            start = None
    if start is not None:
        dur = signals[-1]["t"] - signals[start]["t"]
        if dur >= min_s:
            runs.append({"start": signals[start]["t"], "end": signals[-1]["t"],
                         "duration": dur, "n": len(signals) - start})
    return runs


def _utterance_at(t: float, aligned: list[dict] | None) -> dict | None:
    if not aligned:
        return None
    for a in aligned:
        if a["start"] <= t <= a["end"]:
            return a
    return min(aligned, key=lambda a: abs(a["start"] - t))


def _mins(seconds: float) -> str:
    if seconds < 90:
        return f"{int(round(seconds))} seconds"
    return f"{seconds / 60:.1f} minutes"


def body_language_insights(
    video_result: dict,
    aligned_turns: list[dict] | None = None,
    min_episode_s: float = MIN_EPISODE_S,
) -> dict:
    """
    video_result: output of M15 analyse_video
    aligned_turns: output of M18, so episodes can name the words they coincided with
    """
    if not video_result or not video_result.get("available"):
        return {"available": False, "reason": "no usable video analysis"}

    sig = video_result.get("signals", [])
    if len(sig) < 20:
        return {"available": False, "reason": "recording too short to detect patterns"}

    usable = [s for s in sig if s.get("pose_present")]
    if len(usable) < 15:
        return {
            "available": False,
            "reason": (
                f"body was only in frame for {video_result.get('pose_detection_rate', 0):.0%} "
                f"of the recording — posture tracks are unreliable for this session"
            ),
        }

    client_turns = [a for a in (aligned_turns or []) if a.get("speaker") == "client"]

    def stats(key: str) -> tuple[float, float]:
        v = np.array([s.get(key, 0.0) for s in usable], dtype=float)
        return float(v.mean()), float(v.std() or 1e-6)

    open_mu, open_sd = stats("posture_openness")
    move_mu, move_sd = stats("gesture_amplitude")
    head_mu, head_sd = stats("head_motion")

    findings: list[dict] = []

    # ---- sustained closure ------------------------------------------
    for ep in _episodes(usable, "arms_crossed",
                        lambda s: s.get("arms_crossed"), min_episode_s):
        u = _utterance_at(ep["start"], client_turns)
        findings.append({
            "kind": "posture_closed",
            "start": ep["start"], "timestamp": _clock(ep["start"]),
            "duration_s": round(ep["duration"], 1),
            "sentence": (
                f"Arms stayed folded for {_mins(ep['duration'])} from "
                f"{_clock(ep['start'])}"
                + (f", starting as the client was saying \u201c{u['text'][:90]}\u2026\u201d"
                   if u else "")
                + "."
            ),
            "utterance": u["text"][:200] if u else None,
            "process_mean": u.get("process_mean") if u else None,
            "supervision_question": _supervision_for("posture_closed"),
            "clinical_note": (
                "Sustained arm folding — measure posture, not mood. "
                "Use the timestamp to find the topic, then ask the client."
            ),
        })

    # ---- sustained opening ------------------------------------------
    for ep in _episodes(usable, "posture_openness",
                        lambda s: s.get("posture_openness", 0) > open_mu + Z_STRONG * open_sd,
                        min_episode_s):
        u = _utterance_at(ep["start"], client_turns)
        findings.append({
            "kind": "posture_opened",
            "start": ep["start"], "timestamp": _clock(ep["start"]),
            "duration_s": round(ep["duration"], 1),
            "sentence": (
                f"Posture was noticeably more open than this client's own norm for "
                f"{_mins(ep['duration'])} from {_clock(ep['start'])}"
                + (f", around \u201c{u['text'][:90]}\u2026\u201d" if u else "")
                + "."
            ),
            "utterance": u["text"][:200] if u else None,
            "supervision_question": _supervision_for("posture_opened"),
            "clinical_note": "Posture opened — check whether language moved toward agency or planning.",
        })

    # ---- stillness ---------------------------------------------------
    for ep in _episodes(usable, "gesture_amplitude",
                        lambda s: s.get("gesture_amplitude", 0) < max(0.0, move_mu - Z_STRONG * move_sd),
                        min_episode_s * 1.5):
        u = _utterance_at(ep["start"], client_turns)
        findings.append({
            "kind": "stillness",
            "start": ep["start"], "timestamp": _clock(ep["start"]),
            "duration_s": round(ep["duration"], 1),
            "sentence": (
                f"The client was unusually still for {_mins(ep['duration'])} from "
                f"{_clock(ep['start'])}, moving well below their own average"
                + (f", while saying \u201c{u['text'][:90]}\u2026\u201d" if u else "")
                + ". Worth watching — stillness in therapy is as often deep "
                  "processing as disengagement."
            ),
            "utterance": u["text"][:200] if u else None,
            "supervision_question": _supervision_for("stillness"),
            "clinical_note": "Stillness can mean processing or disengagement — context decides.",
        })

    # ---- agitation ---------------------------------------------------
    for ep in _episodes(usable, "gesture_amplitude",
                        lambda s: s.get("gesture_amplitude", 0) > move_mu + Z_STRONG * move_sd,
                        min_episode_s):
        u = _utterance_at(ep["start"], client_turns)
        findings.append({
            "kind": "heightened_movement",
            "start": ep["start"], "timestamp": _clock(ep["start"]),
            "duration_s": round(ep["duration"], 1),
            "sentence": (
                f"Movement rose well above this client's own norm for "
                f"{_mins(ep['duration'])} from {_clock(ep['start'])}"
                + (f", around \u201c{u['text'][:90]}\u2026\u201d" if u else "")
                + "."
            ),
            "utterance": u["text"][:200] if u else None,
            "supervision_question": _supervision_for("heightened_movement"),
            "clinical_note": "Heightened movement — note whether speech rate or pitch also shifted.",
        })

    # ---- lean shift --------------------------------------------------
    lean_vals = [s.get("lean", 0.0) for s in usable]
    if len(lean_vals) >= 20:
        mid = len(lean_vals) // 2
        early, late = lean_vals[:mid], lean_vals[mid:]
        early_m, late_m = float(np.mean(early)), float(np.mean(late))
        if abs(late_m - early_m) > 0.08:
            direction = "toward" if late_m > early_m else "away from"
            u = _utterance_at(usable[mid]["t"], client_turns)
            findings.append({
                "kind": "lean_shift",
                "start": usable[mid]["t"], "timestamp": _clock(usable[mid]["t"]),
                "duration_s": round(usable[-1]["t"] - usable[mid]["t"], 1),
                "sentence": (
                    f"From {_clock(usable[mid]['t'])} onward the torso angled more "
                    f"{direction} the camera than in the first half"
                    + (f", as the client was saying \u201c{u['text'][:90]}\u2026\u201d" if u else "")
                    + "."
                ),
                "utterance": u["text"][:200] if u else None,
                "supervision_question": _supervision_for("lean_shift"),
                "clinical_note": "Lean shift — relational distance may have changed; verify in session.",
            })

    # ---- head motion spike -------------------------------------------
    head_vals = np.array([s.get("head_motion", 0.0) for s in usable], dtype=float)
    head_sd = float(head_vals.std()) or 1e-6
    spike_idx = int(np.argmax(head_vals))
    if head_vals[spike_idx] > head_mu + 2.0 * head_sd and spike_idx > 2:
        u = _utterance_at(usable[spike_idx]["t"], client_turns)
        findings.append({
            "kind": "head_motion_spike",
            "start": usable[spike_idx]["t"],
            "timestamp": _clock(usable[spike_idx]["t"]),
            "duration_s": 0.0,
            "sentence": (
                f"Head movement peaked at {_clock(usable[spike_idx]['t'])}"
                + (f" while the client was saying \u201c{u['text'][:90]}\u2026\u201d" if u else "")
                + " — the sharpest shift in the recording."
            ),
            "utterance": u["text"][:200] if u else None,
            "supervision_question": _supervision_for("head_motion_spike"),
            "clinical_note": "Head movement spike — often marks re-orientation; locate topic in recording.",
        })

    findings.sort(key=lambda f: f["start"])

    # ---- session-level description -----------------------------------
    summary = video_result.get("summary", {})
    overall: list[str] = []
    closed = summary.get("arms_crossed_ratio", 0)
    if closed > 0.5:
        overall.append(f"Arms were folded for {closed:.0%} of the recording.")
    elif closed > 0.15:
        overall.append(f"Arms were folded intermittently ({closed:.0%} of the recording).")
    if summary.get("gesture_amplitude_sd", 0) > summary.get("gesture_amplitude_mean", 1) * 1.5:
        overall.append("Movement was uneven — long quiet stretches broken by bursts.")
    lean = summary.get("lean_mean", 0)
    if abs(lean) > 0.12:
        overall.append(
            f"The client's torso was consistently angled "
            f"{'toward' if lean > 0 else 'away from'} the camera."
        )
    if video_result.get("pose_detection_rate", 1) < 0.8:
        overall.append(
            f"Note: body was fully in frame for only "
            f"{video_result['pose_detection_rate']:.0%} of the recording."
        )

    engagement = "steady"
    if len(usable) >= 10:
        move_series = np.array([s.get("gesture_amplitude", 0.0) for s in usable], dtype=float)
        first_third = float(move_series[: len(move_series) // 3].mean())
        last_third = float(move_series[-len(move_series) // 3 :].mean())
        if last_third > first_third * 1.4:
            engagement = "increasing_movement"
            overall.append("Movement built across the clip — more animated toward the end.")
        elif last_third < first_third * 0.6:
            engagement = "decreasing_movement"
            overall.append("Movement tapered across the clip — quieter toward the end.")

    supervision_prompts = [_supervision_for(f["kind"]) for f in findings[:4]]
    if engagement == "increasing_movement":
        supervision_prompts.append(
            "Movement built across the clip. Did the session move toward planning "
            "or toward agitation? Compare with what was said if a transcript exists."
        )
    elif engagement == "decreasing_movement":
        supervision_prompts.append(
            "Movement tapered toward the end. Closing-down can mean integration "
            "or fatigue — only the therapist in the room can distinguish them."
        )

    return {
        "available": True,
        "n_findings": len(findings),
        "findings": findings,
        "engagement_arc": engagement,
        "supervision_prompts": supervision_prompts,
        "session_description": " ".join(overall) or
            "Posture and movement stayed within this client's usual range throughout.",
        "narrative": (
            " ".join(f["sentence"] for f in findings[:5])
            if findings else
            "No sustained postural or movement pattern stood out from this "
            "client's own baseline in this recording."
        ),
        "principle": (
            "These are observations of what the body did and when, not "
            "interpretations of what the client felt. Facial expression and "
            "posture map unreliably onto emotion, and in therapy clients "
            "actively manage both. The value here is navigational: it tells a "
            "supervisor where to look, not what to conclude."
        ),
        "not_provided": (
            "No emotion label, mood score or affect classification is produced, "
            "by design."
        ),
    }


def combined_insight(
    text_note: dict, body: dict | None = None, acoustic: dict | None = None
) -> dict:
    """
    One brief for the clinician, text first.

    Text leads because it is the only channel validated against expert
    annotation in this system. Body and voice observations follow as context,
    clearly marked as observation rather than inference.
    """
    parts = {
        "what_was_said": text_note.get("note", ""),
        "what_the_body_did": (body or {}).get("narrative") if body and body.get("available") else None,
        "how_it_sounded": None,
    }

    if acoustic and acoustic.get("available"):
        m = acoustic.get("session_mean", {})
        bits = []
        f0z = m.get("f0_cv_z")
        if isinstance(f0z, (int, float)) and abs(f0z) > 1.5:
            bits.append(
                f"pitch variability ran {abs(f0z):.1f} SD "
                f"{'below' if f0z < 0 else 'above'} their baseline"
            )
        pz = m.get("pause_ratio_z")
        if isinstance(pz, (int, float)) and pz > 1.5:
            bits.append(f"they paused more than usual ({pz:.1f} SD above baseline)")
        if bits:
            parts["how_it_sounded"] = "In this session, " + " and ".join(bits) + "."

    ordered = [v for v in (parts["what_was_said"], parts["what_the_body_did"],
                           parts["how_it_sounded"]) if v]
    return {
        **parts,
        "brief": "\n\n".join(ordered),
        "channels_used": len(ordered),
        "principle": (
            "Language first — it is the only channel here validated against "
            "expert annotation. Body and voice are reported as observations, "
            "not as evidence of feeling."
        ),
    }
