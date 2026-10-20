"""Tests for the decline alert, session goals, prep card and multilingual input."""

from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault(
    "THERAPYTRACE_DB", f"sqlite:///{tempfile.gettempdir()}/therapytrace_test.db"
)

from app.nlp.care import decline_alert, evaluate_goal  # noqa: E402
from app.nlp import translate as tr  # noqa: E402

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


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def _make_case(client, code, texts):
    for c in client.get("/api/clients").json():
        if c["code"] == code:
            client.delete(f"/api/clients/{c['id']}")
    cid = client.post("/api/clients", json={"code": code}).json()["id"]
    for i, text in enumerate(texts, start=1):
        transcript = "\n".join(
            f"THERAPIST: What was that like for you?\nCLIENT: {text}" for _ in range(6)
        )
        r = client.post(f"/api/clients/{cid}/sessions",
                        json={"session_number": i, "transcript": transcript})
        assert r.status_code == 201
    return cid


# --------------------------------------------------------------------------
# 15. decline alert
# --------------------------------------------------------------------------

def test_decline_alert_fires_after_three_consecutive_drops():
    a = decline_alert([55, 58, 54, 50, 46])
    assert a["active"] and a["run_length"] == 3
    assert "supervision" in a["message"]


def test_decline_alert_stays_quiet_when_the_fall_is_interrupted():
    assert not decline_alert([60, 55, 57, 52, 50])["active"]


def test_decline_alert_ignores_rounding_noise():
    assert not decline_alert([50.0, 49.9, 49.8, 49.7, 49.6])["active"]


def test_decline_alert_needs_enough_sessions():
    assert not decline_alert([60, 55, 50])["active"]


def test_trajectory_carries_the_alert(client):
    cid = _make_case(client, "TEST-ALERT", [MOVING, MOVING, MOVING, STUCK, STUCK, STUCK])
    traj = client.get(f"/api/clients/{cid}/trajectory").json()
    assert "decline_alert" in traj
    client.delete(f"/api/clients/{cid}")


# --------------------------------------------------------------------------
# 16. session goals
# --------------------------------------------------------------------------

def test_goal_moved_toward_when_dimension_rises():
    sessions = [
        {"session_number": 1, "z": {"future_orientation": -0.5}},
        {"session_number": 2, "z": {"future_orientation": 0.6}},
    ]
    ev = evaluate_goal("future_orientation", 2, sessions)
    assert ev["status"] == "moved_toward" and ev["delta_sd"] > 0


def test_goal_pending_until_session_is_scored():
    assert evaluate_goal("self_agency", 5, [{"session_number": 1, "z": {}}])["status"] == "pending"


def test_goal_held_inside_noise_band():
    sessions = [
        {"session_number": 1, "z": {"self_agency": 0.10}},
        {"session_number": 2, "z": {"self_agency": 0.20}},
    ]
    assert evaluate_goal("self_agency", 2, sessions)["status"] == "held"


def test_goal_endpoints_round_trip(client):
    cid = _make_case(client, "TEST-GOAL", [STUCK, STUCK, MOVING])
    r = client.post(f"/api/clients/{cid}/goals",
                    json={"session_number": 3, "dimension": "future_orientation"})
    assert r.status_code == 201
    goals = r.json()["goals"]
    assert goals[0]["evaluation"]["status"] == "moved_toward"

    r = client.post(f"/api/clients/{cid}/goals",
                    json={"session_number": 4, "dimension": "self_agency", "note": "decisions"})
    pending = [g for g in r.json()["goals"] if g["session_number"] == 4][0]
    assert pending["evaluation"]["status"] == "pending"

    bad = client.post(f"/api/clients/{cid}/goals",
                      json={"session_number": 4, "dimension": "mood"})
    assert bad.status_code == 400

    assert client.delete(f"/api/goals/{pending['id']}").status_code == 204
    assert len(client.get(f"/api/clients/{cid}/goals").json()["goals"]) == 1
    client.delete(f"/api/clients/{cid}")


# --------------------------------------------------------------------------
# 17. prep card
# --------------------------------------------------------------------------

def test_prep_card_contents(client):
    cid = _make_case(client, "TEST-PREP", [MOVING, MOVING, STUCK, STUCK])
    client.post(f"/api/clients/{cid}/goals",
                json={"session_number": 5, "dimension": "problem_ownership"})
    card = client.get(f"/api/clients/{cid}/prep-card").json()
    assert card["available"]
    assert card["next_session"] == 5
    assert card["focus"]["dimension"]
    assert len(card["revisit"]) <= 2
    assert card["open_goals"] and card["open_goals"][0]["dimension"] == "problem_ownership"
    assert "decline_alert" in card and "caveat" in card
    client.delete(f"/api/clients/{cid}")


def test_prep_card_without_sessions(client):
    for c in client.get("/api/clients").json():
        if c["code"] == "TEST-EMPTY":
            client.delete(f"/api/clients/{c['id']}")
    cid = client.post("/api/clients", json={"code": "TEST-EMPTY"}).json()["id"]
    assert client.get(f"/api/clients/{cid}/prep-card").json()["available"] is False
    client.delete(f"/api/clients/{cid}")


# --------------------------------------------------------------------------
# 22. multilingual input
# --------------------------------------------------------------------------

TAMIL = "சிகிச்சையாளர்: இந்த வாரம் எப்படி இருந்தது?\nவாடிக்கையாளர்: நான் அவளிடம் பேச முடிவு செய்தேன்."
HINDI = "चिकित्सक: यह हफ्ता कैसा रहा?\nक्लाइंट: मैंने उससे बात करने का फैसला किया।"


def test_language_detection():
    assert tr.detect_language(TAMIL) == "ta"
    assert tr.detect_language(HINDI) == "hi"
    assert tr.detect_language("CLIENT: I decided to talk to her.") == "en"


def test_native_speaker_labels_are_mapped():
    turns = tr.split_turns(TAMIL)
    assert [t["label"] for t in turns] == ["THERAPIST", "CLIENT"]
    turns = tr.split_turns(HINDI)
    assert [t["label"] for t in turns] == ["THERAPIST", "CLIENT"]


def test_translation_keeps_turn_alignment():
    fake = lambda texts, lang: [f"EN[{lang}] {t[:5]}" for t in texts]  # noqa: E731
    out = tr.translate_transcript(TAMIL, _transport=fake)
    assert out["available"] and out["language"] == "ta"
    assert len(out["turns"]) == 2
    assert out["english_transcript"].splitlines()[1].startswith("CLIENT: EN[ta]")


def test_translation_rejects_misaligned_output():
    out = tr.translate_transcript(HINDI, _transport=lambda texts, lang: ["only one"])
    assert out["available"] is False


def test_translate_endpoint_scores_the_english(client, monkeypatch):
    monkeypatch.setattr(
        tr, "translate_texts",
        lambda texts, lang, _transport=None: (
            ["How was this week?", MOVING], "test"),
    )
    r = client.post("/api/analyze/translate", json={"transcript": HINDI})
    assert r.status_code == 200
    body = r.json()
    assert body["language"] == "hi"
    assert body["turns"][1]["original"].startswith("मैंने")
    assert 0 <= body["analysis"]["score"]["tpi"] <= 100
    assert body["caveat"]


def test_translate_endpoint_reports_missing_backend(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(tr, "_argos", lambda texts, lang: (_ for _ in ()).throw(ImportError()))
    r = client.post("/api/analyze/translate", json={"transcript": TAMIL})
    assert r.status_code == 503
    assert "backend" in r.json()["detail"]
