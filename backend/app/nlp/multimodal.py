"""
M14 — Multimodal fusion.

Combines the text tier (M3/M4), the acoustic tier (M12) and the visual tier
(M13) into one session record.

Design position, which is the defensible one:

  LATE FUSION, not early. Each modality is scored independently and combined at
  the decision layer. Early fusion (concatenating raw features) would let 48
  acoustic and 15 visual dimensions swamp the 5 process dimensions that carry
  the clinical meaning, and would destroy the explainability that is the whole
  reason a clinician would trust the index.

  TEXT IS PRIMARY AND STAYS PRIMARY. On our own evidence, coarse timing added
  nothing and made the text model worse (AUC 0.624 -> 0.567). Acoustic and
  visual channels therefore enter as *modifiers* with bounded influence, never
  as equal votes. A modality that cannot be shown to add signal does not get to
  move the score.

  EVERYTHING IS WITHIN-PERSON. Vocal tract length sets f0; face geometry sets
  every landmark ratio. Raw values are as incomparable between clients as raw
  word counts, so all three tiers z-score against the client's own baseline.

The fused index keeps the TPI's scale: 50 is this client's own norm.
"""

from __future__ import annotations

from typing import Any

# Bounded modifiers. Text dominates by construction; the other channels can
# together move the index by at most +/- 4 points, which is well under the
# session-to-session noise floor of a typical case.
W_ACOUSTIC = 0.15
W_VISUAL = 0.10
MAX_SHIFT = 4.0

# Direction of each z-scored feature with respect to therapeutic engagement,
# taken from the clinical speech literature rather than fitted here.
ACOUSTIC_SIGNS = {
    "f0_cv_z": +1.0,          # pitch variability; flattening is a depression marker
    "f0_range_z": +1.0,
    "energy_cv_z": +1.0,      # vocal dynamism
    "pause_ratio_z": -0.6,    # psychomotor slowing
    "longest_pause_s_z": -0.4,
    "hnr_z": +0.5,            # breathiness falls as engagement rises
    "syllable_rate_z": +0.3,
}
VISUAL_SIGNS = {
    "expressivity_range_z": +1.0,   # range, deliberately not valence
    "au_activity_z": +0.6,
    "gaze_aversion_ratio_z": -0.5,
    "head_motion_z": +0.3,
}


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
    """
    Returns the fused index alongside the text-only TPI, so a clinician can
    always see what the extra channels did — and disagree with them.
    """
    parts: dict[str, Any] = {
        "tpi_text": round(tpi, 2),
        "acoustic": None,
        "visual": None,
    }
    shift = 0.0

    if acoustic and acoustic.get("available"):
        z, n = _channel_score(acoustic.get("session_mean", {}), ACOUSTIC_SIGNS)
        if z is not None:
            contrib = 10.0 * W_ACOUSTIC * z
            shift += contrib
            parts["acoustic"] = {"z": round(z, 3), "features_used": n,
                                 "contribution": round(contrib, 2)}

    if visual and visual.get("available"):
        z, n = _channel_score(visual.get("session_mean", {}), VISUAL_SIGNS)
        if z is not None:
            contrib = 10.0 * W_VISUAL * z
            shift += contrib
            parts["visual"] = {"z": round(z, 3), "features_used": n,
                               "contribution": round(contrib, 2)}

    # Bounded, and further damped when the text tier itself is low-confidence.
    shift = max(-MAX_SHIFT, min(MAX_SHIFT, shift)) * confidence
    fused = max(0.0, min(100.0, tpi + shift))

    modalities = 1 + (parts["acoustic"] is not None) + (parts["visual"] is not None)
    return {
        **parts,
        "tpi_multimodal": round(fused, 2),
        "total_shift": round(shift, 2),
        "modalities_used": modalities,
        "agreement": (
            "text_only" if modalities == 1
            else "converging" if abs(shift) < 1.5
            else "diverging"
        ),
        "note": (
            "Late fusion with bounded modifiers. Text remains primary because "
            "it is the only channel shown on our data to carry process signal; "
            "acoustic and visual channels can move the index by at most "
            f"{MAX_SHIFT} points combined. When they diverge from the text "
            "score that disagreement is surfaced, not averaged away."
        ),
    }


def analyse_multimodal_session(
    transcript: str,
    audio_path: str | None = None,
    video_path: str | None = None,
    turns: list[dict] | None = None,
    baseline=None,
) -> dict:
    """Full multimodal analysis. Audio and video are optional throughout."""
    from .acoustic import session_acoustics
    from .pipeline import analyse_session
    from .visual import session_visual

    base = analyse_session(transcript, baseline)

    acoustic = visual = None
    if audio_path and turns:
        acoustic = session_acoustics(audio_path, turns)
    if video_path and turns:
        visual = session_visual(video_path, turns)

    base["acoustic"] = acoustic
    base["visual"] = visual
    base["fusion"] = fuse(
        base["score"]["tpi"], acoustic, visual, base["score"]["confidence"]
    )
    return base
