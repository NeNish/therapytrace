"""
Audio input: a session recording in Tamil, Hindi or English is transcribed
with Whisper (via faster-whisper, fully offline after a one-time model
download), translated to English where needed, and handed back as a list of
timed lines for the clinician to label before anything is scored.

Whisper writes down what was said, not who said it. TherapyTrace only scores
client turns, so speaker labels matter. Rather than trust an automatic voice
separator, every line comes back with a first guess (a line ending in a
question is guessed as the therapist) that the user confirms or flips in the
interface. Nothing is scored until that review is done, and the recording is
deleted as soon as it has been transcribed.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from .translate import LANG_NAMES, translate_texts

WHISPER_SIZE = os.getenv("THERAPYTRACE_WHISPER_MODEL", "small")
_MODEL: dict[str, Any] = {}

SUPPORTED = {"ta", "hi", "en"}


def load_audio(path: str, rate: int = 16000):
    """
    Decode any audio file to 16 kHz mono float32 with PyAV.

    faster-whisper has its own decoder, but it passes an argument that PyAV 16+
    no longer accepts, so on a fresh install every file fails before
    transcription starts. Decoding here keeps the app independent of that.
    """
    import av
    import numpy as np

    chunks = []
    with av.open(path) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _whisper_segments(path: str, language: str | None) -> tuple[list[dict], str, float]:
    """Run faster-whisper; returns (segments, detected_language, probability)."""
    from faster_whisper import WhisperModel  # optional dependency

    audio = load_audio(path)
    if audio.size == 0:
        return [], language or "en", 0.0
    if "m" not in _MODEL:
        _MODEL["m"] = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    segments, info = _MODEL["m"].transcribe(
        audio, language=language, task="transcribe", vad_filter=True, beam_size=5,
    )
    rows = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments if s.text.strip()
    ]
    return rows, info.language, float(info.language_probability)


def guess_speaker(text: str, index: int) -> str:
    """
    A deliberately simple first guess for the reviewer to correct: questions
    are usually the therapist's, and sessions usually open with the therapist.
    """
    t = text.strip()
    if t.endswith(("?", "？")):
        return "THERAPIST"
    if index == 0:
        return "THERAPIST"
    return "CLIENT"


def turns_from_lines(lines: list[dict]) -> list[dict]:
    """Merge consecutive lines from the same speaker into one turn."""
    turns: list[dict] = []
    for ln in lines:
        spk = (ln.get("speaker") or "CLIENT").upper()
        orig = (ln.get("original") or "").strip()
        eng = (ln.get("english") or orig).strip()
        if not eng:
            continue
        if turns and turns[-1]["speaker"] == spk:
            turns[-1]["original"] = (turns[-1]["original"] + " " + orig).strip()
            turns[-1]["english"] = (turns[-1]["english"] + " " + eng).strip()
        else:
            turns.append({"speaker": spk, "original": orig, "english": eng})
    return turns


def transcribe_audio(
    path: str,
    language: str = "auto",
    _segments: Callable[[str, str | None], tuple[list[dict], str, float]] | None = None,
) -> dict[str, Any]:
    run = _segments or _whisper_segments
    want = None if language in (None, "", "auto") else language
    try:
        rows, detected, prob = run(path, want)
    except ImportError:
        return {
            "available": False,
            "reason": "Speech recognition is not installed. Run "
                      "`pip install -r requirements-multilingual.txt` in the backend.",
        }
    except Exception as e:
        return {"available": False, "reason": f"Transcription failed: {type(e).__name__}: {e}"}

    if not rows:
        return {"available": False, "reason": "No speech was found in the recording."}

    lang = want or detected
    if lang not in SUPPORTED:
        return {
            "available": False,
            "language": lang,
            "reason": f"Detected language '{lang}'. This page supports Tamil, Hindi and English.",
        }

    originals = [r["text"] for r in rows]
    translation_provider, translation_error = None, None
    if lang == "en":
        english = originals
        translation_provider = "none (already English)"
    else:
        try:
            english, translation_provider = translate_texts(originals, lang)
        except Exception as e:
            english = [None] * len(rows)
            translation_error = str(e)

    lines = []
    for i, (r, en) in enumerate(zip(rows, english)):
        lines.append({
            "id": i,
            "start": r["start"],
            "end": r["end"],
            "original": r["text"],
            "english": en,
            "speaker": guess_speaker(r["text"], i),
        })

    return {
        "available": True,
        "language": lang,
        "language_name": LANG_NAMES.get(lang, lang),
        "language_probability": round(prob, 3) if want is None else None,
        "speech_model": f"whisper-{WHISPER_SIZE}",
        "translation_provider": translation_provider,
        "translation_error": translation_error,
        "lines": lines,
        "note": (
            "Speaker labels are a first guess (questions are assumed to be the "
            "therapist's). Check each line before scoring: only client turns "
            "are scored."
        ),
    }
