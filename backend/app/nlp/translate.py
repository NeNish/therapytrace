"""
Multilingual input: Tamil or Hindi transcripts are translated to English turn
by turn, then scored by the unchanged English pipeline.

This is a first step toward the "beyond English" direction in the paper, and it
is labelled as such everywhere it appears. The lexicon was built for English,
so scores computed on a machine translation measure the translation as much as
the client: idioms, honorifics and the way agency is grammatically expressed
can all shift in transit. The original text is always returned next to the
English so a reader can check what was scored.

Translation backends, tried in order:
  1. a test transport (unit tests only),
  2. the Anthropic API when ANTHROPIC_API_KEY is set (same switch as the
     model-written notes),
  3. Argos Translate, if installed with the language pack (fully offline).
If none is available the endpoint says so rather than scoring untranslated text.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

LANG_NAMES = {"ta": "Tamil", "hi": "Hindi", "en": "English"}

# Native-script speaker labels mapped to the labels the parser understands.
NATIVE_LABELS = {
    # Tamil
    "சிகிச்சையாளர்": "THERAPIST", "ஆலோசகர்": "THERAPIST", "மருத்துவர்": "THERAPIST",
    "வாடிக்கையாளர்": "CLIENT", "நோயாளி": "CLIENT", "பயனர்": "CLIENT",
    # Hindi
    "चिकित्सक": "THERAPIST", "थेरेपिस्ट": "THERAPIST", "परामर्शदाता": "THERAPIST",
    "काउंसलर": "THERAPIST", "क्लाइंट": "CLIENT", "ग्राहक": "CLIENT", "रोगी": "CLIENT",
    "मरीज़": "CLIENT", "मरीज": "CLIENT",
}

_LINE = re.compile(r"^\s*([^\s:：][^:：]{0,24}?)\s*[:：]\s*(.+?)\s*$")

SYSTEM = (
    "You translate psychotherapy transcript turns into English for a research "
    "tool that measures how the client speaks. Translate faithfully and "
    "literally. Keep first-person and passive constructions, tense, hedges, "
    "repetitions and uncertainty exactly as spoken; do not tidy, summarise, "
    "soften or add anything. Return only a JSON array of strings, one English "
    "string per input string, in the same order."
)


def detect_language(text: str) -> str:
    """Script-based detection: Tamil, Devanagari (Hindi) or English."""
    tamil = sum(1 for c in text if "஀" <= c <= "௿")
    deva = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if max(tamil, deva) == 0:
        return "en"
    if tamil >= deva and tamil >= 0.2 * (latin + tamil):
        return "ta"
    if deva > tamil and deva >= 0.2 * (latin + deva):
        return "hi"
    return "en"


def split_turns(text: str) -> list[dict]:
    """
    Split a transcript into (label, content) turns. Lines without a label are
    attached to the previous turn, as the English parser does.
    """
    turns: list[dict] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = _LINE.match(raw)
        if m and len(m.group(1).split()) <= 3:
            label = m.group(1).strip()
            turns.append({
                "label": NATIVE_LABELS.get(label, label),
                "original_label": label,
                "text": m.group(2).strip(),
            })
        elif turns:
            turns[-1]["text"] += " " + raw.strip()
        else:
            turns.append({"label": "", "original_label": "", "text": raw.strip()})
    return turns


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

def _anthropic(texts: list[str], lang: str, key: str) -> list[str]:
    import urllib.request

    from .generate import MODEL

    prompt = (
        f"Source language: {LANG_NAMES.get(lang, lang)}.\n"
        f"Input ({len(texts)} strings):\n{json.dumps(texts, ensure_ascii=False)}"
    )
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({
            "model": MODEL,
            "max_tokens": 4000,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        body = json.loads(r.read().decode())
    out = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
    m = re.search(r"\[.*\]", out, re.S)
    return json.loads(m.group(0) if m else out)


def _argos(texts: list[str], lang: str) -> list[str]:
    import argostranslate.translate as at  # optional dependency

    return [at.translate(t, lang, "en") for t in texts]


def translate_texts(
    texts: list[str],
    lang: str,
    _transport: Callable[[list[str], str], list[str]] | None = None,
) -> tuple[list[str], str]:
    """Translate a list of strings; returns (translations, provider)."""
    if _transport is not None:
        out, provider = _transport(texts, lang), "test"
    elif os.getenv("ANTHROPIC_API_KEY"):
        out, provider = _anthropic(texts, lang, os.environ["ANTHROPIC_API_KEY"]), "anthropic"
    else:
        try:
            out, provider = _argos(texts, lang), "argos"
        except Exception as e:  # not installed, or no language pack
            raise RuntimeError(
                "No translation backend is available. Set ANTHROPIC_API_KEY, or "
                "install Argos Translate with the "
                f"{LANG_NAMES.get(lang, lang)}→English package."
            ) from e
    if not isinstance(out, list) or len(out) != len(texts):
        raise RuntimeError("The translation came back with a different number of turns.")
    return [str(s).strip() for s in out], provider


# --------------------------------------------------------------------------

def translate_transcript(
    text: str,
    language: str = "auto",
    _transport: Callable[[list[str], str], list[str]] | None = None,
) -> dict[str, Any]:
    lang = detect_language(text) if language in (None, "", "auto") else language
    turns = split_turns(text)
    if not turns:
        return {"available": False, "reason": "no readable turns in the transcript"}

    if lang == "en":
        english = [t["text"] for t in turns]
        provider = "none (already English)"
    else:
        try:
            english, provider = translate_texts([t["text"] for t in turns], lang, _transport)
        except Exception as e:
            return {"available": False, "language": lang,
                    "language_name": LANG_NAMES.get(lang, lang), "reason": str(e)}

    rows = []
    for t, en in zip(turns, english):
        rows.append({
            "speaker": t["label"] or None,
            "original_speaker": t["original_label"] or None,
            "original": t["text"],
            "english": en,
        })

    transcript_en = "\n".join(
        f"{r['speaker']}: {r['english']}" if r["speaker"] else r["english"] for r in rows
    )
    return {
        "available": True,
        "language": lang,
        "language_name": LANG_NAMES.get(lang, lang),
        "provider": provider,
        "turns": rows,
        "english_transcript": transcript_en,
    }


CAVEAT = (
    "Scored on a machine translation. The lexicon was built for English, so "
    "idiom, politeness forms and how agency is expressed grammatically can "
    "shift in translation. Treat these scores as exploratory and check the "
    "original text alongside them."
)
