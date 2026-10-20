"""Tests for NLLB translation routing and audio transcription (models stubbed)."""

from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault(
    "THERAPYTRACE_DB", f"sqlite:///{tempfile.gettempdir()}/therapytrace_test.db"
)

from app.nlp import speech as sp  # noqa: E402
from app.nlp import translate as tr  # noqa: E402

MOVING = (
    "I decided not to go and I told her why. I felt resentful and also a bit "
    "ashamed of being resentful. Next week I'm going to ask for ten minutes "
    "before we talk."
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


# --------------------------------------------------------------------------
# NLLB routing
# --------------------------------------------------------------------------

def test_sentence_split_handles_tamil_and_hindi_marks():
    assert tr.split_sentences("நான் போனேன். அவள் வந்தாள்!") == ["நான் போனேன்.", "அவள் வந்தாள்!"]
    assert tr.split_sentences("मैं गया। वह आई।") == ["मैं गया।", "वह आई।"]


def test_nllb_rejoins_sentences_per_turn(monkeypatch):
    seen = {}

    def fake(sentences, lang):
        seen["n"], seen["lang"] = len(sentences), lang
        return [f"<{s}>" for s in sentences]

    monkeypatch.setattr(tr, "_nllb_batch", fake)
    out = tr._nllb(["அ. ஆ.", "இ"], "ta")
    assert seen == {"n": 3, "lang": "ta"}
    assert out == ["<அ.> <ஆ.>", "<இ>"]


def test_nllb_is_used_offline_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("THERAPYTRACE_TRANSLATOR", raising=False)
    monkeypatch.setattr(tr, "_nllb", lambda texts, lang: [f"EN {t}" for t in texts])
    out, provider = tr.translate_texts(["வணக்கம்"], "ta")
    assert provider == "nllb" and out == ["EN வணக்கம்"]


def test_falls_back_to_argos_when_nllb_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("THERAPYTRACE_TRANSLATOR", raising=False)

    def missing(texts, lang):
        raise ImportError("transformers")

    monkeypatch.setattr(tr, "_nllb", missing)
    monkeypatch.setattr(tr, "_argos", lambda texts, lang: ["argos"] * len(texts))
    assert tr.translate_texts(["नमस्ते"], "hi")[1] == "argos"


def test_forced_translator_is_respected(monkeypatch):
    monkeypatch.setenv("THERAPYTRACE_TRANSLATOR", "argos")
    monkeypatch.setattr(tr, "_nllb", lambda texts, lang: ["nllb"] * len(texts))
    monkeypatch.setattr(tr, "_argos", lambda texts, lang: ["argos"] * len(texts))
    assert tr.translate_texts(["x"], "hi")[1] == "argos"


def test_error_names_the_install_command(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("THERAPYTRACE_TRANSLATOR", raising=False)

    def missing(texts, lang):
        raise ImportError()

    monkeypatch.setattr(tr, "_nllb", missing)
    monkeypatch.setattr(tr, "_argos", missing)
    with pytest.raises(RuntimeError, match="requirements-multilingual"):
        tr.translate_texts(["x"], "ta")


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def test_speaker_guess_and_turn_merging():
    assert sp.guess_speaker("How was the week?", 3) == "THERAPIST"
    assert sp.guess_speaker("Welcome back", 0) == "THERAPIST"
    assert sp.guess_speaker("I went to work.", 2) == "CLIENT"
    turns = sp.turns_from_lines([
        {"speaker": "THERAPIST", "original": "a", "english": "A"},
        {"speaker": "CLIENT", "original": "b", "english": "B"},
        {"speaker": "CLIENT", "original": "c", "english": "C"},
    ])
    assert [t["speaker"] for t in turns] == ["THERAPIST", "CLIENT"]
    assert turns[1]["english"] == "B C"


def _fake_segments(path, language):
    rows = [
        {"start": 0.0, "end": 2.0, "text": "இந்த வாரம் எப்படி இருந்தது?"},
        {"start": 2.5, "end": 6.0, "text": "நான் அவளிடம் பேச முடிவு செய்தேன்."},
    ]
    return rows, "ta", 0.97


def test_transcribe_translates_and_guesses(monkeypatch):
    monkeypatch.setattr(tr, "translate_texts", lambda texts, lang, _transport=None: (
        ["How was this week?", "I decided to talk to her."], "nllb"))
    monkeypatch.setattr(sp, "translate_texts", tr.translate_texts)
    out = sp.transcribe_audio("x.wav", _segments=_fake_segments)
    assert out["available"] and out["language"] == "ta"
    assert [ln["speaker"] for ln in out["lines"]] == ["THERAPIST", "CLIENT"]
    assert out["lines"][1]["english"] == "I decided to talk to her."
    assert out["translation_provider"] == "nllb"


def test_transcribe_reports_missing_whisper():
    def missing(path, language):
        raise ImportError("faster_whisper")

    out = sp.transcribe_audio("x.wav", _segments=missing)
    assert out["available"] is False and "requirements-multilingual" in out["reason"]


def test_unsupported_language_is_declined():
    out = sp.transcribe_audio(
        "x.wav", _segments=lambda p, l: ([{"start": 0, "end": 1, "text": "hola"}], "es", 0.9))
    assert out["available"] is False and "Tamil, Hindi and English" in out["reason"]


def test_transcribe_endpoint_deletes_upload_and_returns_lines(client, monkeypatch):
    seen = {}

    def fake(path, language):
        seen["path"] = path
        return _fake_segments(path, language)

    monkeypatch.setattr(sp, "_whisper_segments", fake)
    monkeypatch.setattr(sp, "translate_texts", lambda texts, lang, _transport=None: (
        ["How was this week?", "I decided to talk to her."], "nllb"))
    r = client.post("/api/audio/transcribe",
                    files={"file": ("s.wav", b"RIFF0000WAVE", "audio/wav")},
                    data={"language": "auto"})
    assert r.status_code == 200, r.text
    assert len(r.json()["lines"]) == 2
    assert not os.path.exists(seen["path"])  # recording removed after use


def test_transcribe_endpoint_rejects_other_files(client):
    r = client.post("/api/audio/transcribe", files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 400


def test_reviewed_turns_are_scored(client):
    r = client.post("/api/analyze/turns", json={"language": "ta", "lines": [
        {"speaker": "THERAPIST", "original": "q", "english": "How was this week?"},
        {"speaker": "CLIENT", "original": "a", "english": MOVING},
    ]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["analysis"]["score"]["scoring_mode"] == "absolute"
    assert body["caveat"] and len(body["turns"]) == 2


def test_reviewed_turns_need_a_client_line(client):
    r = client.post("/api/analyze/turns", json={"language": "en", "lines": [
        {"speaker": "THERAPIST", "original": "", "english": "How was this week for you?"},
    ]})
    assert r.status_code == 400


def test_load_audio_decodes_without_whisper_decoder(tmp_path):
    """Our own decoder works with current PyAV, which faster-whisper's does not."""
    pytest.importorskip("av")  # optional dependency, not installed in CI
    import wave

    import numpy as np

    p = tmp_path / "tone.wav"
    rate = 22050
    t = np.arange(rate) / rate
    pcm = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    audio = sp.load_audio(str(p))
    assert audio.dtype == np.float32
    assert abs(len(audio) / 16000 - 1.0) < 0.05  # resampled to 16 kHz, ~1 s
    assert 0.2 < float(np.abs(audio).max()) < 0.4
