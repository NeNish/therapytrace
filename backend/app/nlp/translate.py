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
  3. Meta's NLLB-200 through Hugging Face transformers, if installed. Fully
     offline after a one-time model download, and covers both Tamil and Hindi,
  4. Argos Translate, if installed with the language pack (offline; Hindi only,
     since Argos publishes no Tamil model).
THERAPYTRACE_TRANSLATOR=anthropic|nllb|argos forces one backend. If none is
available the endpoint says so rather than scoring untranslated text.
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


# NLLB-200 language codes for the languages this page accepts.
NLLB_CODES = {"ta": "tam_Taml", "hi": "hin_Deva"}
NLLB_MODEL = os.getenv("THERAPYTRACE_NLLB_MODEL", "facebook/nllb-200-distilled-600M")
_NLLB: dict[str, Any] = {}

_SENT = re.compile(r"(?<=[.!?\u0964|])\s+")


def split_sentences(text: str) -> list[str]:
    """
    NLLB was trained on single sentences and degrades on long paragraphs, so a
    turn is translated sentence by sentence and joined back together.
    """
    parts = [s.strip() for s in _SENT.split(text.strip()) if s.strip()]
    return parts or [text.strip()]


def _nllb_load():
    if "model" not in _NLLB:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # optional

        _NLLB["tok"] = AutoTokenizer.from_pretrained(NLLB_MODEL)
        _NLLB["model"] = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
        _NLLB["model"].eval()
    return _NLLB["tok"], _NLLB["model"]


def _nllb_batch(sentences: list[str], lang: str) -> list[str]:
    """Translate a flat list of sentences into English with NLLB."""
    import torch

    tok, model = _nllb_load()
    tok.src_lang = NLLB_CODES[lang]
    eng = tok.convert_tokens_to_ids("eng_Latn")
    out: list[str] = []
    for i in range(0, len(sentences), 16):
        chunk = sentences[i:i + 16]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            gen = model.generate(**enc, forced_bos_token_id=eng, max_new_tokens=256, num_beams=4)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


def _nllb(texts: list[str], lang: str) -> list[str]:
    if lang not in NLLB_CODES:
        raise RuntimeError(f"NLLB is not configured for language '{lang}'.")
    pieces = [split_sentences(t) for t in texts]
    flat = [s for p in pieces for s in p]
    done = _nllb_batch(flat, lang) if flat else []
    result, k = [], 0
    for p in pieces:
        result.append(" ".join(done[k:k + len(p)]).strip())
        k += len(p)
    return result


def translate_texts(
    texts: list[str],
    lang: str,
    _transport: Callable[[list[str], str], list[str]] | None = None,
) -> tuple[list[str], str]:
    """Translate a list of strings; returns (translations, provider)."""
    forced = os.getenv("THERAPYTRACE_TRANSLATOR", "").strip().lower()
    out = None
    provider = ""
    errors: list[str] = []

    if _transport is not None:
        out, provider = _transport(texts, lang), "test"
    else:
        order = [forced] if forced else ["anthropic", "nllb", "argos"]
        for name in order:
            try:
                if name == "anthropic":
                    key = os.getenv("ANTHROPIC_API_KEY")
                    if not key:
                        errors.append("anthropic: no ANTHROPIC_API_KEY")
                        continue
                    out = _anthropic(texts, lang, key)
                elif name == "nllb":
                    out = _nllb(texts, lang)
                elif name == "argos":
                    out = _argos(texts, lang)
                else:
                    errors.append(f"unknown translator '{name}'")
                    continue
                provider = name
                break
            except Exception as e:  # not installed, no model, or no language pack
                errors.append(f"{name}: {type(e).__name__}")
                out = None

    if out is None:
        raise RuntimeError(
            "No translation backend is available for "
            f"{LANG_NAMES.get(lang, lang)}. Install the offline translator with "
            "`pip install -r requirements-multilingual.txt` (NLLB, covers Tamil "
            "and Hindi), or set ANTHROPIC_API_KEY. Tried: " + "; ".join(errors)
        )
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
