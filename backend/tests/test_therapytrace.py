from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault(
    "THERAPYTRACE_DB", f"sqlite:///{tempfile.gettempdir()}/therapytrace_test.db"
)

from app.nlp.features import score_utterance, session_features  # noqa: E402
from app.nlp.pipeline import analyse_session  # noqa: E402
from app.nlp.scoring import build_baseline, score_session  # noqa: E402
from app.nlp.trajectory import detect_change_point, ols_trend, summarise  # noqa: E402
from app.nlp.transcript import parse_transcript  # noqa: E402

STUCK = (
    "It's all my fault and I ruin everything. He always does this and there's "
    "nothing I can do. I had to go, I have no choice. I just feel bad. What's "
    "the point, nothing works and nothing ever changes."
)
MOVING = (
    "I decided not to go and I told her why. I felt resentful and also a bit "
    "ashamed of being resentful, both at once. I noticed I shut down because I "
    "was scared, and I can see how I escalate it. Next week I'm going to ask "
    "for ten minutes before we talk."
)


# --------------------------------------------------------------------------
# feature ordering — the core claim of the lexicon tier
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "dim",
    ["self_agency", "future_orientation", "emotional_granularity",
     "problem_ownership", "reflection_depth"],
)
def test_moving_language_scores_above_stuck_language(dim):
    lo = getattr(score_utterance(STUCK), dim)
    hi = getattr(score_utterance(MOVING), dim)
    assert hi > lo, f"{dim}: expected {hi} > {lo}"


def test_short_turn_does_not_look_like_collapse():
    """Smoothing means a two-word turn lands near the neutral 0.5, not at 0."""
    s = score_utterance("Yeah, okay.")
    assert 0.35 < s.self_agency < 0.65
    assert 0.35 < s.problem_ownership < 0.65


def test_self_blame_is_not_counted_as_ownership():
    blame = score_utterance("It's all my fault. I'm the problem. I ruin everything.")
    owned = score_utterance("My part in it is that I get defensive and shut the conversation down.")
    assert owned.problem_ownership > blame.problem_ownership


# --------------------------------------------------------------------------
# transcript parsing
# --------------------------------------------------------------------------

def test_parses_standard_labels():
    p = parse_transcript("THERAPIST: How was it?\nCLIENT: Hard.\nTHERAPIST: Say more.")
    assert len(p.therapist_turns) == 2
    assert len(p.client_turns) == 1
    assert not p.inferred


def test_infers_roles_for_diarised_speakers():
    raw = (
        "SPEAKER_00: What was that like?\n"
        "SPEAKER_01: It was hard, I spent the whole week going over it and I "
        "couldn't settle at all, it kept coming back.\n"
        "SPEAKER_00: How so?\n"
        "SPEAKER_01: Because every time I thought about it I felt that same "
        "tightness and I didn't know what to do with it.\n"
    )
    p = parse_transcript(raw)
    assert p.inferred
    assert p.speaker_map["speaker_00"] == "therapist"
    assert p.warnings


def test_continuation_lines_join_previous_turn():
    p = parse_transcript("CLIENT: I went to the shop\nand then I came home.")
    assert len(p.client_turns) == 1
    assert "came home" in p.client_turns[0].text


# --------------------------------------------------------------------------
# within-person scoring
# --------------------------------------------------------------------------

def test_tpi_is_fifty_at_own_baseline():
    feats, _ = session_features([STUCK] * 6)
    base = build_baseline([feats], baseline_k=1)
    assert score_session(feats, base)["tpi"] == pytest.approx(50.0, abs=0.01)


def test_tpi_rises_when_language_improves_relative_to_baseline():
    low, _ = session_features([STUCK] * 6)
    high, _ = session_features([MOVING] * 6)
    base = build_baseline([low, low], baseline_k=2)
    assert score_session(high, base)["tpi"] > 55.0


def test_confidence_drops_for_thin_sessions():
    thin, _ = session_features(["Yeah.", "I guess.", "Fine."])
    base = build_baseline([thin], baseline_k=1)
    assert score_session(thin, base)["confidence"] < 0.5


# --------------------------------------------------------------------------
# trajectory
# --------------------------------------------------------------------------

def test_trend_detects_a_rising_series():
    t = ols_trend([40, 44, 47, 52, 55, 61])
    assert t.slope > 3 and t.p_approx < 0.01


def test_change_point_finds_the_step():
    cp = detect_change_point([48, 49, 47, 50, 66, 68, 65, 67])
    assert cp and cp["index"] == 4 and cp["direction"] == "improvement"


def test_flat_series_has_no_change_point():
    assert detect_change_point([50, 50.4, 49.8, 50.2, 50.1, 49.9]) is None


def test_plateau_is_labelled_as_plateau():
    s = summarise([55, 55.5, 54.8, 55.2, 55.1, 55.3])
    assert s["momentum"]["state"] == "plateauing"


# --------------------------------------------------------------------------
# therapist impact
# --------------------------------------------------------------------------

def test_reflection_before_an_insight_turn_scores_positive_lift():
    raw = (
        f"CLIENT: {STUCK}\n"
        "THERAPIST: It sounds like part of you wanted to speak and part of you froze.\n"
        f"CLIENT: {MOVING}\n"
    )
    out = analyse_session(raw)
    rows = {r["intervention"]: r for r in out["therapist_impact"]}
    assert "complex_reflection" in rows
    assert rows["complex_reflection"]["mean_lift"] > 0


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_full_case_flow(client):
    code = "TEST-001"
    client.delete_existing = None
    existing = [c for c in client.get("/api/clients").json() if c["code"] == code]
    for c in existing:
        client.delete(f"/api/clients/{c['id']}")

    created = client.post("/api/clients", json={"code": code}).json()
    cid = created["id"]

    for i in range(1, 7):
        text = STUCK if i <= 3 else MOVING
        transcript = "\n".join(
            f"THERAPIST: What was that like?\nCLIENT: {text}" for _ in range(6)
        )
        r = client.post(
            f"/api/clients/{cid}/sessions",
            json={"session_number": i, "transcript": transcript},
        )
        assert r.status_code == 201
        client.post(
            f"/api/clients/{cid}/measures",
            json={"session_number": i, "instrument": "PHQ-9", "score": 20 - 2 * i},
        )

    traj = client.get(f"/api/clients/{cid}/trajectory").json()
    assert len(traj["sessions"]) == 6
    assert traj["tpi_series"][-1] > traj["tpi_series"][0]
    assert traj["validation"]["pearson_r"] < 0  # language up, symptoms down

    impact = client.get(f"/api/clients/{cid}/therapist-impact").json()
    assert impact["interventions"]

    client.delete(f"/api/clients/{cid}")


def test_stateless_analyze(client):
    r = client.post("/api/analyze", json={"transcript": f"CLIENT: {MOVING}"})
    assert r.status_code == 200
    assert 0 <= r.json()["score"]["tpi"] <= 100


def test_insert_time_score_matches_recomputed_series(client):
    """
    Regression: the analysis must be attached through the relationship, or the
    recompute silently skips the session it just scored and every POST returns
    a flat 50. Session 1 is exempt — it legitimately moves once a second
    session widens the baseline.
    """
    code = "TEST-REGRESSION"
    for c in [x for x in client.get("/api/clients").json() if x["code"] == code]:
        client.delete(f"/api/clients/{c['id']}")
    cid = client.post("/api/clients", json={"code": code}).json()["id"]

    returned = []
    for i in range(1, 6):
        text = STUCK if i <= 2 else MOVING
        transcript = "\n".join(
            f"THERAPIST: What was that like?\nCLIENT: {text}" for _ in range(6)
        )
        r = client.post(
            f"/api/clients/{cid}/sessions",
            json={"session_number": i, "transcript": transcript},
        ).json()
        returned.append(r["tpi"])

    series = client.get(f"/api/clients/{cid}/trajectory").json()["tpi_series"]
    assert returned[1:] == pytest.approx(series[1:], abs=0.01)
    assert len(set(returned)) > 1, "every POST returned the same score"
    client.delete(f"/api/clients/{cid}")


# --------------------------------------------------------------------------
# M10 — supervised models
# --------------------------------------------------------------------------

def test_ml_module_degrades_gracefully_without_artifacts():
    """The pipeline must run identically on a clone that never trained."""
    from app.nlp.ml_models import available, session_ml_summary

    out = session_ml_summary(["I decided to go and I told her why."], ["What was that like?"])
    assert set(out["available"]) == {"change_talk", "therapist_behaviour"}
    for key in ("client_talk", "therapist_behaviour"):
        assert key in out


@pytest.mark.skipif(
    not __import__("app.nlp.ml_models", fromlist=["available"]).available()["change_talk"],
    reason="trained artifacts not present",
)
def test_change_talk_model_separates_clear_cases():
    from app.nlp.ml_models import classify_client_turns

    out = classify_client_turns([
        "I decided to stop drinking on weekdays and I told my wife about it.",
        "I don't think I can change, it is just who I am and it always has been.",
    ])
    assert out["labels"][0] != "sustain"
    assert out["change_talk_ratio"] is not None


@pytest.mark.skipif(
    not __import__("app.nlp.ml_models", fromlist=["available"]).available()["therapist_behaviour"],
    reason="trained artifacts not present",
)
def test_therapist_model_recognises_a_question():
    from app.nlp.ml_models import classify_therapist_turns

    out = classify_therapist_turns(["What would it look like if you did make that change?"])
    assert out["labels"][0] == "question"


def test_models_endpoint(client):
    body = client.get("/api/models").json()
    assert len(body["models"]) == 2
    assert "GroupKFold" in body["split_protocol"]


# --------------------------------------------------------------------------
# M12-M14 — multimodal tier
# --------------------------------------------------------------------------

def test_acoustic_extracts_expected_feature_set(tmp_path):
    import numpy as np, soundfile as sf
    from app.nlp.acoustic import extract_acoustic

    sr = 16000
    t = np.arange(sr * 3) / sr
    y = (np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)).astype("float32")
    p = tmp_path / "a.wav"
    sf.write(p, y, sr)

    f = extract_acoustic(p)
    assert f is not None
    for k in ("f0_mean", "energy_mean", "pause_ratio", "jitter", "hnr", "mfcc1_mean"):
        assert k in f
    assert 100 < f["f0_mean"] < 220


def test_acoustic_missing_file_returns_none():
    from app.nlp.acoustic import extract_acoustic
    assert extract_acoustic("/nonexistent/file.wav") is None


def test_within_person_standardisation_centres_the_baseline():
    from app.nlp.acoustic import standardise_within_client

    base = [{"f0_mean": 100.0}, {"f0_mean": 110.0}, {"f0_mean": 120.0}]
    out = standardise_within_client([{"f0_mean": 110.0}], base)
    assert abs(out[0]["f0_mean_z"]) < 0.01  # the baseline mean maps to zero


def test_fusion_leaves_text_score_alone_without_other_modalities():
    from app.nlp.multimodal import fuse
    out = fuse(63.5)
    assert out["tpi_multimodal"] == 63.5
    assert out["modalities_used"] == 1
    assert out["agreement"] == "text_only"


def test_fusion_shift_is_bounded():
    """No combination of acoustic evidence may override the text tier."""
    from app.nlp.multimodal import MAX_SHIFT, fuse

    extreme = {"available": True, "session_mean": {
        "f0_cv_z": 50.0, "f0_range_z": 50.0, "energy_cv_z": 50.0,
        "hnr_z": 50.0, "syllable_rate_z": 50.0}}
    out = fuse(50.0, extreme, None)
    assert abs(out["total_shift"]) <= MAX_SHIFT + 1e-6


def test_visual_missing_file_degrades_gracefully():
    from app.nlp.visual import session_visual
    out = session_visual("/nonexistent/v.mp4", [{"speaker": "client", "start": 0, "end": 2}])
    assert out["available"] is False


# --------------------------------------------------------------------------
# M15 — session review renderer
# --------------------------------------------------------------------------

def _synth_clip(path, seconds=4, fps=15, w=320, h=240):
    import cv2, numpy as np
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(seconds * fps):
        f = np.full((h, w, 3), 205, np.uint8)
        cx = w // 2
        cv2.ellipse(f, (cx, 90), (34, 42), 0, 0, 360, (180, 152, 142), -1)
        cv2.circle(f, (cx - 12, 82), 4, (25, 25, 25), -1)
        cv2.circle(f, (cx + 12, 82), 4, (25, 25, 25), -1)
        cv2.ellipse(f, (cx, 108), (14, 5), 0, 0, 180, (70, 45, 45), 2)
        cv2.rectangle(f, (cx - 48, 136), (cx + 48, h), (70, 90, 120), -1)
        vw.write(f)
    vw.release()


def test_review_analyses_a_clip_and_renders_overlays(tmp_path):
    from app.nlp.review import review_session

    src, dst = tmp_path / "s.mp4", tmp_path / "s_out.mp4"
    _synth_clip(src)
    r = review_session(src, output_video=dst)

    assert r["video"]["available"] is True
    assert r["video"]["n_sampled"] > 0
    assert r["video"]["face_detection_rate"] > 0.5   # a drawn face is detectable
    assert r["render"]["available"] is True
    assert dst.exists() and dst.stat().st_size > 0


def test_review_missing_video_degrades_gracefully():
    from app.nlp.review import review_session
    out = review_session("/nonexistent/clip.mp4")
    assert out["available"] is False


def test_moments_are_spread_across_the_session(tmp_path):
    from app.nlp.review import analyse_video, find_moments

    src = tmp_path / "s.mp4"
    _synth_clip(src, seconds=6)
    a = analyse_video(src)
    moments = find_moments(a, k=3)
    times = [m["t"] for m in moments]
    assert times == sorted(times)          # chronological
    assert len(set(times)) == len(times)   # non-maximum suppression worked


# --------------------------------------------------------------------------
# M16 / M17 — narrative and recommendation
# --------------------------------------------------------------------------

def test_narrative_reports_measurements_not_inferences():
    from app.nlp.narrative import session_note

    note = session_note(
        {"score": {"tpi": 38.0, "confidence": 0.9,
                   "contributions": {"self_agency": -1.2, "reflection_depth": 0.1}},
         "features": {"n_client_words": 900, "hopelessness": 0.4}, "ml": {}},
    )
    text = note["note"].lower()
    assert "38" in text
    # the module must never assert a feeling or a diagnosis
    for banned in ("depressed", "is sad", "is anxious", "feels ", "diagnos"):
        assert banned not in text
    assert any(f["kind"] == "hopelessness_language" for f in note["flags"])


def test_narrative_flags_low_confidence():
    from app.nlp.narrative import session_note
    note = session_note({"score": {"tpi": 50.0, "confidence": 0.4, "contributions": {}},
                         "features": {}, "ml": {}})
    assert any(f["kind"] == "low_confidence" for f in note["flags"])


def test_early_warning_stays_quiet_on_a_healthy_series():
    from app.nlp.recommend import early_warning
    out = early_warning([50, 53, 56, 58, 61, 63], [{"n_client_words": 900}] * 6)
    assert out["level"] == "none"


def test_early_warning_fires_on_decline_plus_risk_language():
    from app.nlp.recommend import early_warning
    feats = [{"n_client_words": 900}] * 5 + [{"n_client_words": 880, "hopelessness": 0.45}]
    out = early_warning([62, 60, 55, 50, 46, 41], feats)
    assert out["level"] in ("watch", "elevated")
    assert out["reasons"]


def test_targeting_wording_matches_the_sign():
    """A client above baseline on everything must not be told they are below it."""
    from app.nlp.recommend import target_dimensions

    dims = {d: [0.60, 0.62, 0.64, 0.66] for d in
            ["self_agency", "future_orientation", "emotional_granularity",
             "problem_ownership", "reflection_depth"]}
    baseline = {"means": {d: 0.50 for d in dims}, "sds": {d: 0.05 for d in dims}}
    out = target_dimensions(dims, baseline)
    assert out["all_above_baseline"] is True
    assert "below this client's baseline" not in out["summary"]


def test_recommender_falls_back_when_nothing_in_history_worked():
    """Never recommend the least-bad of several harmful moves."""
    from app.nlp.recommend import next_session_suggestion

    profile = {"available": True, "cells": [
        {"intervention": "challenge", "state": "moving", "n": 6, "mean_lift": -0.05},
        {"intervention": "directive", "state": "moving", "n": 5, "mean_lift": -0.09},
    ]}
    out = next_session_suggestion([58.0], profile)
    assert out["evidence_source"] == "corpus"


def test_recommender_prefers_the_clients_own_history_when_it_is_positive():
    from app.nlp.recommend import next_session_suggestion

    profile = {"available": True, "cells": [
        {"intervention": "affirmation", "state": "moving", "n": 7, "mean_lift": 0.08},
        {"intervention": "directive", "state": "moving", "n": 5, "mean_lift": -0.04},
    ]}
    out = next_session_suggestion([58.0], profile)
    assert out["evidence_source"] == "this client's own history"
    assert out["ranked"][0]["intervention"] == "affirmation"


@pytest.mark.skipif(
    not __import__("pathlib").Path("/tmp/head-pose-face-detection-female.mp4").exists(),
    reason="sample video not downloaded; see ml/README.md",
)
def test_review_on_real_human_footage():
    """
    Regression against a real recording rather than a synthetic figure.

    The synthetic clips used elsewhere in this suite are enough to prove the
    code path runs, but MediaPipe Pose is trained on real people and cannot
    find a drawn stick figure — so pose tracking was unverified until this
    test. Sample: a seated person facing the camera, the framing a counselling
    recording actually has.
    """
    from app.nlp.review import review_session

    r = review_session("/tmp/head-pose-face-detection-female.mp4", max_seconds=8)
    v = r["video"]
    assert v["available"] is True
    assert v["face_detection_rate"] > 0.8
    assert v["pose_detection_rate"] > 0.8
    assert v["summary"]["posture_openness_mean"] > 0
    assert len(r["moments"]) > 0


# --------------------------------------------------------------------------
# M18 — session alignment
# --------------------------------------------------------------------------

def test_explicit_timestamps_are_preferred_over_estimation():
    from app.nlp.align import align_session

    t = ("[00:00:05] THERAPIST: How was the week?\n"
         "[00:00:20] CLIENT: It was hard, I went quiet again like I always do.\n"
         "[00:00:45] THERAPIST: What happened in that moment?\n"
         "[00:01:10] CLIENT: I decided to say something on Thursday and I did say it.\n")
    out = align_session(t, duration_s=120)
    assert out["timing_source"] == "explicit"
    assert out["timing_warning"] is None
    assert out["turns"][1]["start"] == 20.0


def test_estimation_is_flagged_as_approximate():
    """A supervisor who jumps to a wrong timestamp stops trusting all of them."""
    from app.nlp.align import align_session

    t = ("THERAPIST: How was the week?\n"
         "CLIENT: It was hard and I went quiet again, the way I always seem to.\n")
    out = align_session(t, duration_s=60)
    assert out["timing_source"] == "estimated"
    assert "estimated" in out["timing_warning"]


def test_turn_times_are_ordered_and_cover_the_recording():
    from app.nlp.align import align_session

    t = "\n".join(
        f"{'THERAPIST' if i % 2 == 0 else 'CLIENT'}: This is turn number {i} with some words in it."
        for i in range(8)
    )
    out = align_session(t, duration_s=100)
    starts = [x["start"] for x in out["turns"]]
    assert starts == sorted(starts)
    assert out["turns"][-1]["end"] <= 101


def test_moments_are_linked_to_the_utterance_being_spoken():
    """The whole point: a movement signal must acquire a referent."""
    from app.nlp.align import align_session

    t = ("THERAPIST: How was the week?\n"
         "CLIENT: I don't know, it is just how my father was so it is just how I am.\n"
         "THERAPIST: What would you want to say instead?\n"
         "CLIENT: I decided I am going to ask for ten minutes before we talk on Tuesday.\n")
    video = {"duration_s": 40, "signals": [
        {"t": float(i), "gesture_amplitude": 0.5 if 10 < i < 20 else 0.05,
         "arms_crossed": False, "posture_openness": 1.2, "head_motion": 0.02,
         "lean": 0.0, "pose_present": True, "face_present": True}
        for i in range(40)],
        "moments": [{"t": 15.0, "timestamp": "00:15", "salience": 2.0}]}
    out = align_session(t, video_result=video)
    linked = out["linked_moments"]
    assert len(linked) == 1
    assert linked[0]["utterance"]          # a moment now carries words
    assert "turn_idx" in linked[0]


def test_alignment_without_duration_declines_rather_than_guessing():
    from app.nlp.align import align_session
    out = align_session("CLIENT: something happened this week.\n")
    assert out["available"] is False


# --------------------------------------------------------------------------
# M19 — body language insight
# --------------------------------------------------------------------------

def _planted_signals():
    sig = []
    for i in range(240):
        t = float(i)
        crossed, still = 40 <= t <= 95, 120 <= t <= 175
        sig.append({"t": t, "pose_present": True, "face_present": True,
                    "arms_crossed": crossed,
                    "posture_openness": 0.9 if crossed else 1.6,
                    "gesture_amplitude": 0.002 if still else 0.06,
                    "head_motion": 0.02, "lean": 0.02})
    return {"available": True, "signals": sig, "pose_detection_rate": 1.0,
            "summary": {"arms_crossed_ratio": 0.23, "gesture_amplitude_mean": 0.05,
                        "gesture_amplitude_sd": 0.03, "lean_mean": 0.02}}


def test_body_detects_a_sustained_closure_episode():
    from app.nlp.body import body_language_insights
    out = body_language_insights(_planted_signals())
    assert out["available"] is True
    kinds = [f["kind"] for f in out["findings"]]
    assert "posture_closed" in kinds
    ep = next(f for f in out["findings"] if f["kind"] == "posture_closed")
    assert 50 <= ep["duration_s"] <= 60


def test_body_anchors_findings_to_the_utterance():
    """A posture observation is only useful if it names what was being said."""
    from app.nlp.body import body_language_insights

    turns = [{"speaker": "client", "start": 35, "end": 100,
              "text": "It is just how my father was, so it is just how I am.",
              "process_mean": 0.40}]
    out = body_language_insights(_planted_signals(), turns)
    ep = next(f for f in out["findings"] if f["kind"] == "posture_closed")
    assert "father" in ep["sentence"]
    assert ep["process_mean"] == 0.40


def test_body_never_emits_an_emotion_label():
    """The design commitment, enforced: measurement not mind-reading."""
    from app.nlp.body import body_language_insights

    out = body_language_insights(_planted_signals())
    blob = (out["narrative"] + out["session_description"]
            + " ".join(f["sentence"] for f in out["findings"])).lower()
    for banned in ("defensive", "anxious", "angry", "sad", "depressed",
                   "closed off emotionally", "feels ", "is feeling"):
        assert banned not in blob
    assert "no emotion label" in out["not_provided"].lower()


def test_body_declines_when_the_subject_is_out_of_frame():
    from app.nlp.body import body_language_insights

    sig = [{"t": float(i), "pose_present": False, "arms_crossed": False,
            "posture_openness": 0, "gesture_amplitude": 0, "head_motion": 0, "lean": 0}
           for i in range(60)]
    out = body_language_insights(
        {"available": True, "signals": sig, "pose_detection_rate": 0.05, "summary": {}}
    )
    assert out["available"] is False
    assert "frame" in out["reason"]


def test_combined_brief_puts_language_first():
    from app.nlp.body import body_language_insights, combined_insight

    note = {"note": "This session scored 61, above this client's own baseline."}
    body = body_language_insights(_planted_signals())
    brief = combined_insight(note, body)
    assert brief["brief"].startswith("This session scored 61")
    assert brief["channels_used"] >= 2


# --------------------------------------------------------------------------
# M20 — model-generated insight
# --------------------------------------------------------------------------

_NOTE = {
    "sentences": [
        "This session scored 49 on the progress index.",
        "The index has been falling about 1.3 points per session.",
    ],
    "flags": [{"kind": "plateau"}],
    "note": "(deterministic fallback)",
}
_TRAJ = {"momentum": {"state": "regressing"},
         "recent_trend": {"slope_per_session": -1.3}, "n_sessions": 11}


def test_model_draft_is_accepted_when_every_figure_traces_to_an_input():
    from app.nlp.generate import generate_insight

    draft = ("The session scored 49 on the progress index. Across 11 sessions the "
             "index has fallen about 1.3 points per session.")
    out = generate_insight(_NOTE, trajectory=_TRAJ, _transport=lambda p: draft)
    assert out["source"] == "model"
    assert out["verified"] is True


def test_invented_figures_are_rejected():
    """The guard that makes generation usable: no number may be invented."""
    from app.nlp.generate import generate_insight

    draft = "The session scored 49. Engagement fell 37% and alliance dropped to 62."
    out = generate_insight(_NOTE, trajectory=_TRAJ, _transport=lambda p: draft)
    assert out["source"] == "template"
    assert "unsupported figures" in out["reason"]
    assert "62" in out["reason"] and "37" in out["reason"]
    assert out["note"] == "(deterministic fallback)"
    assert "rejected_draft" in out          # failures are visible, not silent


def test_inferential_language_is_rejected():
    from app.nlp.generate import generate_insight

    draft = "The session scored 49. The client appeared defensive and withdrawn."
    out = generate_insight(_NOTE, trajectory=_TRAJ, _transport=lambda p: draft)
    assert out["source"] == "template"
    assert "inferential language" in out["reason"]


def test_falls_back_cleanly_with_no_api_key(monkeypatch):
    from app.nlp.generate import generate_insight

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = generate_insight(_NOTE)
    assert out["source"] == "template"
    assert out["verified"] is True


def test_generation_failure_falls_back_rather_than_raising():
    from app.nlp.generate import generate_insight

    def boom(_):
        raise RuntimeError("network down")

    out = generate_insight(_NOTE, _transport=boom)
    assert out["source"] == "template"
    assert "generation failed" in out["reason"]


def test_payload_carries_only_measured_values():
    """The model must never see free narrative it could pattern-match on."""
    from app.nlp.generate import build_payload

    p = build_payload(_NOTE, features={"n_client_words": 900, "tpi": 49.0},
                      trajectory=_TRAJ)
    assert set(p) <= {"measurements", "flags", "session_features",
                      "trajectory", "body_observations", "voice"}
    assert all(isinstance(v, (int, float)) for v in p["session_features"].values())


def test_multimodal_endpoint_runs_on_text_alone(client):
    """Text is primary: the other two channels must be genuinely optional."""
    body = client.post("/api/sessions/multimodal", json={"transcript": (
        "THERAPIST: How was the week?\n"
        "CLIENT: I decided to tell her how I felt and I did say it, which surprised me.\n"
    )}).json()
    assert body["channels"]["text"]["available"] is True
    assert body["channels"]["audio"]["available"] is False
    assert body["channels"]["video"]["available"] is False
    assert body["channels_active"] == 1
    assert body["fusion"]["tpi_multimodal"] == body["fusion"]["tpi_text"]


def test_multimodal_requires_a_transcript(client):
    assert client.post("/api/sessions/multimodal", json={"video_path": "/x.mp4"}).status_code == 400


def test_missing_media_degrades_only_that_channel(client):
    body = client.post("/api/sessions/multimodal", json={
        "transcript": "CLIENT: I decided to go and I told her why, which was hard.\n",
        "audio_path": "/nonexistent.wav", "video_path": "/nonexistent.mp4",
    }).json()
    assert body["channels"]["text"]["available"] is True
    assert body["channels"]["audio"]["available"] is False
    assert body["channels"]["video"]["available"] is False
