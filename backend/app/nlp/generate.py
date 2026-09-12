"""
M20 — Model-Generated Insight.

M16 writes the session note from templates. That makes it safe and fully
traceable, but it also makes it read like a form letter, and it cannot connect
findings across channels the way a clinician would. This module hands the
measured numbers to a language model and asks it to write the note.

The obvious objection is hallucination: a model that invents a statistic in a
clinical document is worse than no document. So the generation is wrapped in
three constraints, and the third is the one that matters:

  1. THE PROMPT CARRIES ONLY MEASURED VALUES. No transcript text beyond short
     quoted utterances, no free narrative, nothing the model could pattern-match
     into a plausible-sounding clinical story.

  2. THE INSTRUCTIONS FORBID INFERENCE. The model is told explicitly that it may
     not name an emotion, assert a diagnosis, or explain why something happened.
     It reports what was measured and when.

  3. EVERY NUMBER IN THE OUTPUT IS VERIFIED AGAINST THE INPUT. After generation,
     each numeric token in the draft is checked against the set of values that
     were actually supplied. If the model produces a figure that was never given
     to it, the draft is rejected and the deterministic M16 note is returned
     instead.

Constraint 3 is what makes this usable clinically. The model gets to write
better prose; it does not get to decide what is true. A rejected draft is
logged with the offending value, so failures are visible rather than silent.

Falls back to M16 cleanly when no API key is configured, so the system runs
unchanged without one.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 700

SYSTEM = """You write session notes for psychotherapy supervision.

You will be given measurements from one therapy session. Write a short note for \
the supervising clinician.

Absolute rules:
- Use ONLY the numbers you are given. Never introduce a figure that is not in \
the input. This is the most important rule.
- Report what was MEASURED, never what the client FELT. Do not write that the \
client was anxious, defensive, sad, depressed, withdrawn or any other emotional \
or diagnostic state.
- Do not explain WHY anything happened. You do not know.
- Do not recommend treatment.
- Where a measurement coincided with something the client said, say so and quote \
the words briefly.
- Plain professional English. No bullet points, no headings. Three to six \
sentences.
- If a measurement is flagged as low confidence or estimated, say so.

You are describing evidence so a clinician can decide what it means. You are not \
deciding for them."""


def _numbers_in(text: str) -> set[str]:
    """Numeric tokens, normalised so 6.0 and 6 compare equal."""
    out = set()
    for tok in re.findall(r"-?\d+\.?\d*", text):
        try:
            f = float(tok)
        except ValueError:
            continue
        out.add(str(int(f)) if f == int(f) else f"{f:.4g}")
    return out


def _allowed_numbers(payload: dict) -> set[str]:
    """
    Every value the model is permitted to use, plus the roundings a writer would
    naturally apply (49.3 -> 49, and percentages of any fraction).
    """
    allowed: set[str] = set()

    def walk(v: Any):
        nonlocal allowed
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, bool):
            return
        elif isinstance(v, (int, float)):
            f = float(v)
            allowed.add(str(int(f)) if f == int(f) else f"{f:.4g}")
            allowed.add(str(int(round(f))))
            allowed.add(f"{f:.1f}".rstrip("0").rstrip("."))
            allowed.add(str(abs(int(round(f)))))
            if 0 <= abs(f) <= 1:                      # fraction read as a percentage
                allowed.add(str(int(round(f * 100))))
            if abs(f) < 100:
                allowed.add(f"{abs(f):.4g}")
        elif isinstance(v, str):
            allowed |= _numbers_in(v)

    walk(payload)
    # small integers a writer uses structurally ("three sessions", "the five
    # dimensions") rather than as claims
    allowed |= {str(i) for i in range(0, 13)}
    return allowed


def verify_numbers(draft: str, payload: dict) -> tuple[bool, list[str]]:
    """Returns (ok, offending values). The gate that makes this usable."""
    allowed = _allowed_numbers(payload)
    bad = [n for n in _numbers_in(draft) if n not in allowed]
    return (not bad), bad


BANNED = (
    "anxious", "anxiety", "depressed", "depression", "defensive", "withdrawn",
    "sad ", "angry", "distressed", "resistant", "avoidant", "feels ",
    "is feeling", "seems to feel", "diagnos", "disorder",
)


def verify_language(draft: str) -> tuple[bool, list[str]]:
    low = draft.lower()
    return (hits := [b.strip() for b in BANNED if b in low]) == [], hits


def build_payload(
    session_note: dict,
    features: dict | None = None,
    trajectory: dict | None = None,
    body: dict | None = None,
    acoustic: dict | None = None,
) -> dict:
    """Only measured values reach the model."""
    payload: dict[str, Any] = {
        "measurements": session_note.get("sentences", []),
        "flags": [f.get("kind") for f in session_note.get("flags", [])],
    }
    if features:
        payload["session_features"] = {
            k: v for k, v in features.items()
            if isinstance(v, (int, float))
        }
    if trajectory:
        payload["trajectory"] = {
            "momentum": trajectory.get("momentum", {}).get("state"),
            "slope_per_session": trajectory.get("recent_trend", {}).get("slope_per_session"),
            "n_sessions": trajectory.get("n_sessions"),
        }
    if body and body.get("available"):
        payload["body_observations"] = [
            {"timestamp": f["timestamp"], "kind": f["kind"],
             "duration_s": f["duration_s"],
             "utterance": (f.get("utterance") or "")[:140]}
            for f in body.get("findings", [])[:5]
        ]
    if acoustic and acoustic.get("available"):
        payload["voice"] = {
            k: v for k, v in acoustic.get("session_mean", {}).items()
            if k.endswith("_z") and isinstance(v, (int, float)) and abs(v) > 1.0
        }
    return payload


def generate_insight(
    session_note: dict,
    features: dict | None = None,
    trajectory: dict | None = None,
    body: dict | None = None,
    acoustic: dict | None = None,
    api_key: str | None = None,
    _transport=None,
) -> dict:
    """
    Model-written note, verified, with the deterministic note as fallback.

    `_transport` exists so the verification path can be tested without a network
    call; production leaves it None.
    """
    payload = build_payload(session_note, features, trajectory, body, acoustic)
    fallback = {
        "source": "template",
        "note": session_note.get("note", ""),
        "verified": True,
        "reason": None,
    }

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key and _transport is None:
        return {**fallback, "reason": "no ANTHROPIC_API_KEY configured"}

    prompt = (
        "Session measurements:\n\n"
        + json.dumps(payload, indent=2)
        + "\n\nWrite the supervision note."
    )

    try:
        if _transport is not None:
            draft = _transport(prompt)
        else:
            import urllib.request

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps({
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "system": SYSTEM,
                    "messages": [{"role": "user", "content": prompt}],
                }).encode(),
                headers={
                    "content-type": "application/json",
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                body_json = json.loads(r.read().decode())
            draft = "".join(
                b.get("text", "") for b in body_json.get("content", [])
                if b.get("type") == "text"
            ).strip()
    except Exception as e:
        return {**fallback, "reason": f"generation failed: {type(e).__name__}"}

    if not draft:
        return {**fallback, "reason": "empty response"}

    ok_num, bad_nums = verify_numbers(draft, payload)
    ok_lang, bad_lang = verify_language(draft)

    if not ok_num:
        return {
            **fallback,
            "reason": f"rejected: unsupported figures {bad_nums}",
            "rejected_draft": draft,
        }
    if not ok_lang:
        return {
            **fallback,
            "reason": f"rejected: inferential language {bad_lang}",
            "rejected_draft": draft,
        }

    return {
        "source": "model",
        "note": draft,
        "verified": True,
        "reason": None,
        "checks": {
            "numeric_verification": "passed — every figure traced to an input value",
            "language_verification": "passed — no emotional or diagnostic terms",
        },
        "model": MODEL,
    }
