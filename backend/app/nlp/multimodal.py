"""
M14 — Multimodal fusion and unified analysis.

Combines text (M3/M4), acoustic (M12) and visual/pose (M15) into one session
record with mode-aware insights.
"""

from __future__ import annotations

from typing import Any

from .features import DIMENSIONS

W_ACOUSTIC = 0.15
W_VISUAL = 0.10
MAX_SHIFT = 4.0

ACOUSTIC_SIGNS = {
    "f0_cv_z": +1.0,
    "f0_range_z": +1.0,
    "energy_cv_z": +1.0,
    "pause_ratio_z": -0.6,
    "longest_pause_s_z": -0.4,
    "hnr_z": +0.5,
    "syllable_rate_z": +0.3,
}
VISUAL_SIGNS = {
    "expressivity_range_z": +1.0,
    "au_activity_z": +0.6,
    "gesture_amplitude_z": +0.8,
    "head_motion_z": +0.3,
    "posture_openness_z": +0.5,
    "arms_crossed_ratio_z": -0.7,
}

# Population reference values for z-scoring a single audio clip with no baseline.
ACOUSTIC_NORMS: dict[str, tuple[float, float]] = {
    "f0_cv": (0.18, 0.06),
    "f0_range": (45.0, 20.0),
    "energy_cv": (0.35, 0.12),
    "pause_ratio": (0.22, 0.10),
    "longest_pause_s": (1.2, 0.8),
    "hnr": (8.0, 4.0),
    "syllable_rate": (3.5, 1.2),
    "jitter": (0.015, 0.010),
    "shimmer": (0.04, 0.025),
}


def _z_from_norm(value: float | None, key: str) -> float | None:
    if value is None or key not in ACOUSTIC_NORMS:
        return None
    mu, sd = ACOUSTIC_NORMS[key]
    return round((float(value) - mu) / sd, 4)


def prepare_acoustic_for_fusion(raw: dict[str, Any]) -> dict[str, Any]:
    """Add _z keys so fusion can use a whole-file acoustic extract."""
    session_mean = dict(raw)
    for key in ACOUSTIC_NORMS:
        z = _z_from_norm(raw.get(key), key)
        if z is not None:
            session_mean[f"{key}_z"] = z
    return {"available": True, "session_mean": session_mean, "n_segments": 1}


def prepare_visual_for_fusion(video_result: dict[str, Any]) -> dict[str, Any]:
    """Map pose-track output into z-scored features fusion understands."""
    if not video_result.get("available"):
        return {"available": False, "reason": video_result.get("reason", "no video")}

    signals = video_result.get("signals", [])
    usable = [s for s in signals if s.get("pose_present")]
    if len(usable) < 5:
        return {"available": False, "reason": "insufficient pose detections"}

    try:
        import numpy as np
    except ImportError:
        return {"available": False, "reason": "numpy not available"}

    def _within_z(key: str) -> float:
        vals = np.array([s.get(key, 0.0) for s in usable], dtype=float)
        sd = float(vals.std()) or 1e-6
        return round(float((vals.mean() - 0.0) / sd), 4)

    summary = video_result.get("summary", {})
    gesture_sd = summary.get("gesture_amplitude_sd", 0.0)
    gesture_mu = summary.get("gesture_amplitude_mean", 0.0) or 1e-6

    session_mean = {
        "expressivity_range_z": round(gesture_sd / (gesture_mu + 1e-6), 4),
        "gesture_amplitude_z": _within_z("gesture_amplitude"),
        "head_motion_z": _within_z("head_motion"),
        "posture_openness_z": _within_z("posture_openness"),
        "arms_crossed_ratio_z": round(
            (summary.get("arms_crossed_ratio", 0) - 0.15) / 0.12, 4
        ),
    }
    return {"available": True, "session_mean": session_mean}


def _channel_score(session_mean: dict[str, Any], signs: dict[str, float]) -> tuple[float | None, int]:
    used = {k: w for k, w in signs.items() if isinstance(session_mean.get(k), (int, float))}
    if not used:
        return None, 0
    total = sum(abs(w) for w in used.values())
    z = sum(w * session_mean[k] for k, w in used.items()) / total
    return max(-3.0, min(3.0, z)), len(used)


def fuse(
    tpi: float,
    acoustic: dict | None = None,
    visual: dict | None = None,
    confidence: float = 1.0,
) -> dict:
    parts: dict[str, Any] = {"tpi_text": round(tpi, 2), "acoustic": None, "visual": None}
    shift = 0.0

    if acoustic and acoustic.get("available"):
        z, n = _channel_score(acoustic.get("session_mean", {}), ACOUSTIC_SIGNS)
        if z is not None:
            contrib = 10.0 * W_ACOUSTIC * z
            shift += contrib
            parts["acoustic"] = {"z": round(z, 3), "features_used": n, "contribution": round(contrib, 2)}

    if visual and visual.get("available"):
        z, n = _channel_score(visual.get("session_mean", {}), VISUAL_SIGNS)
        if z is not None:
            contrib = 10.0 * W_VISUAL * z
            shift += contrib
            parts["visual"] = {"z": round(z, 3), "features_used": n, "contribution": round(contrib, 2)}

    shift = max(-MAX_SHIFT, min(MAX_SHIFT, shift)) * confidence
    fused = max(0.0, min(100.0, tpi + shift))
    modalities = 1 + (parts["acoustic"] is not None) + (parts["visual"] is not None)

    agreement = "text_only"
    if modalities > 1:
        if abs(shift) < 1.0:
            agreement = "converging"
        elif shift > 0:
            agreement = "channels_support_higher"
        else:
            agreement = "channels_suggest_caution"

    return {
        **parts,
        "tpi_multimodal": round(fused, 2),
        "total_shift": round(shift, 2),
        "modalities_used": modalities,
        "agreement": agreement,
        "note": (
            f"Text process score {round(tpi, 1)}"
            + (f"; acoustic/visual channels adjusted by {shift:+.1f} points." if shift else ".")
            + " Audio and video enter as bounded modifiers (max ±4 combined)."
        ),
    }


def detect_input_mode(transcript: str | None, audio_path: str | None, video_path: str | None) -> str:
    has_t = bool(transcript and len(transcript.strip()) >= 20)
    has_a = bool(audio_path)
    has_v = bool(video_path)
    if has_t and has_a and has_v:
        return "full_multimodal"
    if has_t and has_a:
        return "text_audio"
    if has_t and has_v:
        return "text_video"
    if has_a and has_v:
        return "audio_video"
    if has_t:
        return "text_only"
    if has_v:
        return "video_only"
    if has_a:
        return "audio_only"
    return "none"


INPUT_MODE_LABELS = {
    "full_multimodal": "Full session — transcript, audio and video",
    "text_audio": "Transcript + audio recording",
    "text_video": "Transcript + video recording",
    "audio_video": "Audio + video (no transcript)",
    "text_only": "Transcript only",
    "video_only": "Video only",
    "audio_only": "Audio only",
    "none": "No input supplied",
}


def _nearest_preview(preview_map: dict[float, dict], t: float, tol: float = 1.5) -> dict | None:
    if not preview_map:
        return None
    closest = min(preview_map.keys(), key=lambda k: abs(k - t))
    if abs(closest - t) <= tol:
        return preview_map[closest]
    return None


def _attach_preview_to_items(items: list[dict], preview_map: dict[float, dict]) -> None:
    for item in items:
        t = item.get("t")
        if t is None:
            t = item.get("start")
        if t is None:
            continue
        prev = _nearest_preview(preview_map, float(t))
        if prev:
            item["preview"] = prev


def acoustic_insights(acoustic: dict | None) -> dict:
    if not acoustic or not acoustic.get("available"):
        return {"available": False, "bullets": [], "summary": None}

    m = acoustic.get("session_mean", {})
    bullets: list[str] = []

    f0 = m.get("f0_mean")
    f0_cv = m.get("f0_cv")
    if f0:
        bullets.append(f"Mean pitch {f0:.0f} Hz — reference only; vocal range varies by speaker.")
    if isinstance(f0_cv, (int, float)):
        if f0_cv < 0.12:
            bullets.append("Pitch variability is low — voice stayed relatively flat throughout the clip.")
        elif f0_cv > 0.24:
            bullets.append("Pitch variability is high — wide vocal range with expressive contour.")
        else:
            bullets.append("Pitch variability is within a typical range.")

    pr = m.get("pause_ratio")
    if isinstance(pr, (int, float)):
        if pr > 0.35:
            bullets.append(f"Pauses account for {pr:.0%} of the recording — longer silences than typical.")
        elif pr < 0.12:
            bullets.append(f"Very few pauses ({pr:.0%}) — speech ran continuously.")
        else:
            bullets.append(f"Pause ratio {pr:.0%} — normal conversational pacing.")

    lp = m.get("longest_pause_s")
    if isinstance(lp, (int, float)) and lp > 2.0:
        bullets.append(f"Longest pause {lp:.1f}s — worth checking whether a hesitation preceded reformulation.")

    sr = m.get("syllable_rate")
    if isinstance(sr, (int, float)):
        if sr < 2.5:
            bullets.append(f"Speech rate {sr:.1f} syllables/s — slower than typical conversation.")
        elif sr > 5.0:
            bullets.append(f"Speech rate {sr:.1f} syllables/s — faster than typical conversation.")

    hnr = m.get("hnr")
    if isinstance(hnr, (int, float)) and hnr < 5:
        bullets.append("Voice quality is breathy (low harmonics-to-noise ratio).")

    jitter = m.get("jitter")
    if isinstance(jitter, (int, float)) and jitter > 0.025:
        bullets.append("Elevated jitter — cycle-to-cycle pitch instability detected.")

    if not bullets:
        bullets.append("Acoustic features extracted; no strong deviations from typical ranges.")

    supervision: list[str] = []
    pr = m.get("pause_ratio")
    if isinstance(pr, (int, float)) and pr > 0.35:
        supervision.append(
            "Long pauses detected — in supervision, ask whether silences were "
            "productive (client thinking) or stuck (client unable to answer)."
        )
    f0cv = m.get("f0_cv")
    if isinstance(f0cv, (int, float)) and f0cv < 0.12:
        supervision.append(
            "Flat pitch contour — voice stayed monotone. Did the client's language "
            "also show low future orientation or emotional granularity?"
        )
    sr = m.get("syllable_rate")
    if isinstance(sr, (int, float)) and (sr < 2.5 or sr > 5.0):
        supervision.append(
            "Speech rate was atypical. Compare with session content: slow can "
            "mean careful reflection; fast can mean urgency or anxiety — "
            "the words decide, not the rate alone."
        )
    if not supervision:
        supervision.append(
            "Listen back at timestamps where the client named difficulty — "
            "did pitch or pacing shift on those words specifically?"
        )

    return {
        "available": True,
        "bullets": bullets,
        "summary": " ".join(bullets[:3]),
        "supervision_prompts": supervision,
        "metrics": {
            k: m[k] for k in (
                "f0_mean", "f0_cv", "pause_ratio", "longest_pause_s",
                "syllable_rate", "hnr", "jitter", "shimmer", "duration_s",
            ) if m.get(k) is not None
        },
    }


def video_insights(
    body: dict | None,
    video: dict | None,
    moments: list | None,
    preview_map: dict[float, dict] | None = None,
) -> dict:
    if not body or not body.get("available"):
        reason = (body or {}).get("reason") or (video or {}).get("reason") or "no video analysis"
        return {"available": False, "bullets": [], "summary": None, "reason": reason}

    bullets: list[str] = []
    findings = body.get("findings", [])

    if body.get("session_description"):
        bullets.append(body["session_description"])

    for f in findings[:6]:
        bullets.append(f["sentence"])

    if not bullets:
        bullets.append(body.get("narrative") or "Posture and movement stayed within this person's usual range.")

    timeline = []
    for f in findings:
        timeline.append({
            "t": f.get("start"),
            "timestamp": f.get("timestamp"),
            "kind": f.get("kind"),
            "duration_s": f.get("duration_s"),
            "sentence": f.get("sentence"),
            "utterance": f.get("utterance"),
            "process_mean": f.get("process_mean"),
            "clinical_note": f.get("clinical_note"),
            "supervision_question": f.get("supervision_question"),
        })

    enriched_moments = list(moments or [])[:8]
    for m in enriched_moments:
        if not m.get("description"):
            m["description"] = m.get("observation") or m.get("why", "")

    pmap = preview_map or {}
    _attach_preview_to_items(timeline, pmap)
    _attach_preview_to_items(enriched_moments, pmap)

    preview_gallery = [
        pmap[k] for k in sorted(pmap.keys())
    ]

    arc = body.get("engagement_arc", "steady")
    arc_labels = {
        "increasing_movement": "Movement built across the clip — more animated toward the end.",
        "decreasing_movement": "Movement tapered — quieter toward the end.",
        "steady": "Movement stayed relatively even throughout.",
    }

    return {
        "available": True,
        "bullets": bullets,
        "summary": body.get("narrative"),
        "timeline": timeline,
        "moments": enriched_moments,
        "preview_gallery": preview_gallery,
        "supervision_prompts": body.get("supervision_prompts", []),
        "engagement_arc": arc,
        "engagement_label": arc_labels.get(arc, arc_labels["steady"]),
        "n_findings": len(findings),
        "pose_detection_rate": (video or {}).get("pose_detection_rate"),
        "duration_s": (video or {}).get("duration_s"),
    }


def text_insights(text_result: dict | None) -> dict:
    if not text_result:
        return {"available": False}

    from .narrative import session_note

    score = text_result.get("score", {})
    note = session_note(text_result)
    features = text_result.get("features", {})

    dim_high = sorted(DIMENSIONS, key=lambda d: features.get(d, 0), reverse=True)[:2]
    dim_low = sorted(DIMENSIONS, key=lambda d: features.get(d, 0))[:2]

    from .narrative import DIM_WORDS
    bullets = [
        f"Process index {score.get('tpi', 50):.1f} "
        f"(confidence {score.get('confidence', 1) * 100:.0f}%).",
        f"Strongest dimensions: {DIM_WORDS.get(dim_high[0], dim_high[0])}, "
        f"{DIM_WORDS.get(dim_high[1], dim_high[1])}.",
        f"Lowest dimensions: {DIM_WORDS.get(dim_low[0], dim_low[0])}, "
        f"{DIM_WORDS.get(dim_low[1], dim_low[1])}.",
    ]
    if score.get("scoring_mode") == "absolute":
        bullets.insert(1, "Scored from language content — add to a case for within-person change tracking.")

    from .narrative import DIM_WORDS, DIM_CONCRETE
    supervision: list[str] = []
    for d in dim_low:
        pos, neg = DIM_CONCRETE.get(d, ("", ""))
        supervision.append(
            f"{DIM_WORDS.get(d, d)} was low — consider interventions that invite "
            f"{pos}, e.g. open questions about {pos.split('more ')[-1] if 'more ' in pos else 'process'}."
        )
    if text_result.get("drivers", {}).get("weakest"):
        w = text_result["drivers"]["weakest"][0]
        supervision.append(
            f"The weakest client turn (process {w.get('process_mean', 0):.2f}) "
            f"may be worth exploring: \"{w.get('text', '')[:100]}…\""
        )
    if not supervision:
        supervision.append(
            "Review the strongest and weakest client turns — where did language "
            "show agency, ownership, or reflection?"
        )

    return {
        "available": True,
        "tpi": score.get("tpi"),
        "confidence": score.get("confidence"),
        "scoring_mode": score.get("scoring_mode", "relative"),
        "note": note.get("note"),
        "flags": note.get("flags", []),
        "bullets": bullets,
        "supervision_prompts": supervision[:4],
        "dimensions": {d: round(features.get(d, 0), 3) for d in DIMENSIONS},
        "n_client_turns": text_result.get("parse", {}).get("n_client_turns"),
        "strongest_turns": (text_result.get("drivers") or {}).get("strongest", [])[:2],
        "weakest_turns": (text_result.get("drivers") or {}).get("weakest", [])[:2],
    }


def run_multimodal_analysis(payload: dict) -> dict:
    """
    Unified multimodal analysis — accepts any combination of transcript,
    audio_path and video_path.
    """
    from pathlib import Path

    from .acoustic import extract_acoustic
    from .align import align_session
    from .body import body_language_insights, combined_insight
    from .narrative import session_note
    from .pipeline import analyse_session
    from .review import attach_frame_previews, find_moments, review_session
    from .scoring import score_session_standalone

    transcript = (payload.get("transcript") or "").strip()
    audio_path = payload.get("audio_path")
    video_path = payload.get("video_path")
    max_seconds = payload.get("max_seconds", 30)

    mode = detect_input_mode(transcript, audio_path, video_path)
    if mode == "none":
        return {"error": "Provide at least one of: transcript, audio file, or video file."}

    text_result = None
    text_score = None
    channels: dict[str, Any] = {}
    acoustic_raw = acoustic_fusion = None
    video_raw = visual_fusion = body = aligned = aligned_result = None
    moments: list = []

    # ---- TEXT ----
    if transcript and len(transcript) >= 20:
        text_result = analyse_session(transcript)
        text_result["score"] = score_session_standalone(text_result["features"])
        text_score = text_result["score"]["tpi"]
        channels["text"] = {
            "available": True,
            "tpi": text_score,
            "confidence": text_result["score"]["confidence"],
            "scoring_mode": "absolute",
            "features": text_result["features"],
            "n_client_turns": text_result["parse"]["n_client_turns"],
            "role": "primary — language process markers",
        }
    else:
        channels["text"] = {
            "available": False,
            "reason": "no transcript supplied",
            "role": "primary when supplied",
        }

    # ---- VIDEO ----
    if video_path and Path(str(video_path)).exists():
        review = review_session(video_path, max_seconds=max_seconds)
        if review.get("video", {}).get("available"):
            video_raw = {**review["video"], "moments": review.get("moments", [])}
            moments = find_moments(review) or review.get("moments", [])
            if transcript and len(transcript) >= 20:
                aligned_result = align_session(transcript, video_result=video_raw)
                aligned = aligned_result.get("turns") if aligned_result.get("available") else None
            body = body_language_insights(video_raw, aligned)
            visual_fusion = prepare_visual_for_fusion(video_raw)
            channels["video"] = {
                "available": True,
                "pose_detection_rate": video_raw.get("pose_detection_rate"),
                "face_detection_rate": video_raw.get("face_detection_rate"),
                "duration_s": video_raw.get("duration_s"),
                "n_findings": (body or {}).get("n_findings", 0),
                "timing_source": (aligned_result or {}).get("timing_source") if transcript else "video_only",
                "role": "posture and movement observation",
            }
        else:
            channels["video"] = {"available": False, "reason": review.get("reason", "video unusable")}
    else:
        channels["video"] = {
            "available": False,
            "reason": "no video supplied" if not video_path else "video file not found",
        }

    # ---- AUDIO ----
    if audio_path and Path(str(audio_path)).exists():
        raw = extract_acoustic(audio_path, duration=max_seconds)
        if raw:
            acoustic_raw = prepare_acoustic_for_fusion(raw)
            acoustic_fusion = acoustic_raw
            channels["audio"] = {
                "available": True,
                "n_features": len([k for k, v in raw.items() if isinstance(v, (int, float))]),
                "f0_mean": raw.get("f0_mean"),
                "pause_ratio": raw.get("pause_ratio"),
                "syllable_rate": raw.get("syllable_rate"),
                "duration_s": raw.get("duration_s"),
                "role": "vocal prosody and pacing",
            }
        else:
            channels["audio"] = {"available": False, "reason": "audio unreadable"}
    else:
        channels["audio"] = {
            "available": False,
            "reason": "no audio supplied" if not audio_path else "audio file not found",
        }

    # ---- FUSION ----
    base_tpi = text_score if text_score is not None else 50.0
    conf = text_result["score"]["confidence"] if text_result else 0.7
    fusion = fuse(base_tpi, acoustic_fusion, visual_fusion, conf)

    # When no text, derive a descriptive index label from channels only
    if text_score is None:
        channel_z = 0.0
        if acoustic_fusion and acoustic_fusion.get("available"):
            z, _ = _channel_score(acoustic_fusion["session_mean"], ACOUSTIC_SIGNS)
            if z is not None:
                channel_z += z * 0.6
        if visual_fusion and visual_fusion.get("available"):
            z, _ = _channel_score(visual_fusion["session_mean"], VISUAL_SIGNS)
            if z is not None:
                channel_z += z * 0.4
        fusion["tpi_text"] = None
        fusion["tpi_multimodal"] = round(max(0, min(100, 50 + 10 * channel_z)), 2)
        fusion["note"] = (
            "No transcript — index estimated from acoustic/visual channels only. "
            "Add a transcript for language-based process scoring."
        )

    # ---- FRAME PREVIEWS ----
    preview_map: dict[float, dict] = {}
    if video_path and video_raw and body:
        preview_map = attach_frame_previews(
            video_path,
            moments,
            (body or {}).get("findings"),
            video_raw.get("signals"),
        )

    # ---- INSIGHTS ----
    text_note = session_note(text_result) if text_result else None
    a_insights = acoustic_insights(acoustic_raw)
    v_insights = video_insights(body, video_raw, moments, preview_map)
    t_insights = text_insights(text_result)

    brief = combined_insight(text_note or {}, body, acoustic_raw)

    return {
        "input_mode": mode,
        "input_mode_label": INPUT_MODE_LABELS.get(mode, mode),
        "channels": channels,
        "channels_active": sum(1 for c in channels.values() if c.get("available")),
        "fusion": fusion,
        "insights": {
            "text": t_insights,
            "audio": a_insights,
            "video": v_insights,
        },
        "brief": brief,
        "body": body,
        "moments": v_insights.get("moments") or moments,
        "preview_gallery": v_insights.get("preview_gallery") or [],
        "aligned_turns": (aligned or [])[:60],
        "principle": (
            "Insights are generated per channel based on what you supplied. "
            "Text scores therapeutic language; audio reports vocal prosody; "
            "video reports posture and movement — never inferred emotion."
        ),
    }


def analyse_multimodal_session(
    transcript: str,
    audio_path: str | None = None,
    video_path: str | None = None,
    turns: list[dict] | None = None,
    baseline=None,
) -> dict:
    """Full multimodal analysis with optional client baseline."""
    from .acoustic import session_acoustics
    from .pipeline import analyse_session
    from .visual import session_visual

    base = analyse_session(transcript, baseline)

    acoustic = visual = None
    if audio_path and turns:
        acoustic = session_acoustics(audio_path, turns)
    if video_path and turns:
        visual = session_visual(video_path, turns)

    if acoustic and acoustic.get("available"):
        acoustic = prepare_acoustic_for_fusion(acoustic.get("session_mean", {}))

    base["acoustic"] = acoustic
    base["visual"] = visual
    base["fusion"] = fuse(
        base["score"]["tpi"], acoustic, visual, base["score"]["confidence"]
    )
    return base
