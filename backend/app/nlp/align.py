"""
M18 — Session Alignment.

The module that makes the visual and acoustic tiers mean something.

Until now the channels ran in parallel and never met. The video knew that
movement changed sharply at 22:10. The transcript knew that turn 34 was the
client's lowest-scoring utterance of the session. Neither knew about the other,
so the video could report movement but never *what the movement was about* —
which is why a gesture-amplitude number told a supervisor nothing.

Aligning them on a shared clock changes the kind of claim the system can make:

    before:  "movement changed sharply at 22:10"
    after:   "the client's posture closed at 22:10, while they were saying
              'I don't know, it's just how my father was' — their lowest
              ownership score of the session"

The second is worth a supervisor's attention. The first is not.

Alignment sources, in order of preference:

  1. EXPLICIT TIMESTAMPS in the transcript (WebVTT, SRT, or "[00:12:34]"
     inline markers). Exact, and what most transcription tools already emit.
  2. DIARISATION SEGMENTS supplied by the caller as turn boundaries.
  3. PROPORTIONAL ESTIMATION as a last resort — distribute turns across the
     recording by word count. Approximate, and flagged as such in the output,
     because a supervisor who jumps to a timestamp and finds the wrong moment
     will stop trusting every timestamp.

Nothing here infers meaning. It puts a measurement next to the words that were
being spoken at the time, and leaves the interpretation where it belongs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .features import DIMENSIONS, score_utterance

# [00:12:34]  (00:12:34)  00:12:34  12:34
TS_PATTERNS = [
    re.compile(r"[\[(]?(\d{1,2}):(\d{2}):(\d{2})[\])]?"),
    re.compile(r"[\[(](\d{1,2}):(\d{2})[\])]"),
]


@dataclass
class AlignedTurn:
    idx: int
    speaker: str
    text: str
    start: float
    end: float
    source: str            # "explicit" | "diarisation" | "estimated"
    process_mean: float = 0.0
    scores: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "speaker": self.speaker,
            "text": self.text,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "timestamp": _clock(self.start),
            "timing_source": self.source,
            "process_mean": round(self.process_mean, 4),
            "scores": self.scores or {},
        }


def _clock(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def parse_timestamp(line: str) -> float | None:
    for i, rx in enumerate(TS_PATTERNS):
        m = rx.search(line)
        if m:
            g = m.groups()
            return (int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2])) if len(g) == 3 \
                else (int(g[0]) * 60 + int(g[1]))
    return None


def align_turns(
    turns: list[Any],
    duration_s: float,
    diarisation: list[dict] | None = None,
    raw_lines: list[str] | None = None,
) -> tuple[list[AlignedTurn], str]:
    """
    turns: Turn objects from M1 (idx, speaker, text)
    Returns the aligned turns and which source was used.
    """
    n = len(turns)
    if not n:
        return [], "none"

    # ---- 1. explicit timestamps -------------------------------------
    stamps: dict[int, float] = {}
    if raw_lines:
        ti = 0
        for line in raw_lines:
            if not line.strip():
                continue
            ts = parse_timestamp(line)
            if ts is not None and ti < n:
                stamps[ti] = ts
            if ":" in line or "\t" in line:
                ti += 1
    if len(stamps) >= max(3, n // 3):
        aligned = []
        keys = sorted(stamps)
        for i, t in enumerate(turns):
            start = stamps.get(i)
            if start is None:
                prev = max((k for k in keys if k < i), default=None)
                nxt = min((k for k in keys if k > i), default=None)
                if prev is not None and nxt is not None:
                    frac = (i - prev) / (nxt - prev)
                    start = stamps[prev] + frac * (stamps[nxt] - stamps[prev])
                elif prev is not None:
                    start = stamps[prev]
                else:
                    start = 0.0
            aligned.append(AlignedTurn(i, t.speaker, t.text, start, 0.0, "explicit"))
        for i in range(len(aligned) - 1):
            aligned[i].end = aligned[i + 1].start
        aligned[-1].end = max(duration_s, aligned[-1].start + 1)
        return _score(aligned), "explicit"

    # ---- 2. diarisation ----------------------------------------------
    if diarisation and len(diarisation) >= n:
        aligned = [
            AlignedTurn(i, t.speaker, t.text,
                        float(diarisation[i]["start"]), float(diarisation[i]["end"]),
                        "diarisation")
            for i, t in enumerate(turns)
        ]
        return _score(aligned), "diarisation"

    # ---- 3. proportional estimate ------------------------------------
    # Speaking time tracks word count far better than turn count does: a
    # 200-word answer does not take the same time as "mm-hm".
    words = [max(1, len(t.text.split())) for t in turns]
    total = sum(words)
    aligned, cursor = [], 0.0
    for i, t in enumerate(turns):
        span = duration_s * words[i] / total
        aligned.append(AlignedTurn(i, t.speaker, t.text, cursor, cursor + span, "estimated"))
        cursor += span
    return _score(aligned), "estimated"


def _score(aligned: list[AlignedTurn]) -> list[AlignedTurn]:
    for a in aligned:
        if a.speaker == "client":
            s = score_utterance(a.text)
            a.process_mean = s.process_mean
            a.scores = {d: getattr(s, d) for d in DIMENSIONS}
    return aligned


# --------------------------------------------------------------------------
# the payoff: what was being said when the signal moved
# --------------------------------------------------------------------------

def link_moments(
    moments: list[dict], aligned: list[AlignedTurn], window_s: float = 6.0
) -> list[dict]:
    """Attach the utterance active at each flagged moment."""
    out = []
    for m in moments:
        t = float(m.get("t", 0))
        active = next((a for a in aligned if a.start <= t <= a.end), None)
        if active is None:
            active = min(aligned, key=lambda a: min(abs(a.start - t), abs(a.end - t)))

        nearby = [a for a in aligned
                  if abs((a.start + a.end) / 2 - t) <= window_s and a.speaker == "client"]
        context = min(nearby, key=lambda a: a.process_mean) if nearby else active

        out.append({
            **m,
            "speaker": active.speaker,
            "utterance": active.text[:400],
            "turn_idx": active.idx,
            "process_mean": round(active.process_mean, 4) if active.speaker == "client" else None,
            "timing_source": active.source,
            "notable_nearby": (
                {"text": context.text[:300], "process_mean": round(context.process_mean, 4)}
                if context is not active and context.speaker == "client" else None
            ),
        })
    return out


def find_convergences(
    aligned: list[AlignedTurn], video_signals: list[dict],
    acoustic_turns: list[dict] | None = None, k: int = 5,
) -> list[dict]:
    """
    Moments where a channel and the language move together.

    This is the reason for aligning at all. Movement alone is noise — people
    shift in their seats. Movement that coincides with the client's lowest
    ownership score of the session is a moment worth watching.
    """
    client = [a for a in aligned if a.speaker == "client" and len(a.text.split()) >= 8]
    if not client or not video_signals:
        return []

    pm = [a.process_mean for a in client]
    mu = sum(pm) / len(pm)
    sd = (sum((x - mu) ** 2 for x in pm) / max(1, len(pm) - 1)) ** 0.5 or 1e-6

    g = [s.get("gesture_amplitude", 0.0) for s in video_signals]
    gmu = sum(g) / len(g)
    gsd = (sum((x - gmu) ** 2 for x in g) / max(1, len(g) - 1)) ** 0.5 or 1e-6

    rows = []
    for a in client:
        seg = [s for s in video_signals if a.start <= s.get("t", -1) <= a.end]
        if not seg:
            continue
        gz = (sum(s.get("gesture_amplitude", 0.0) for s in seg) / len(seg) - gmu) / gsd
        lz = (a.process_mean - mu) / sd
        closed = sum(1 for s in seg if s.get("arms_crossed")) / len(seg)

        # Disagreement between channels is more informative than agreement:
        # low language score with high movement, or vice versa.
        rows.append({
            "timestamp": _clock(a.start),
            "t": round(a.start, 2),
            "turn_idx": a.idx,
            "utterance": a.text[:300],
            "language_z": round(lz, 2),
            "gesture_z": round(gz, 2),
            "arms_crossed_ratio": round(closed, 3),
            "divergence": round(abs(lz - gz), 3),
            "pattern": (
                "low language score with heightened movement" if lz < -0.7 < gz
                else "strong language with unusual stillness" if lz > 0.7 and gz < -0.7
                else "language and movement agree"
            ),
        })

    rows.sort(key=lambda r: r["divergence"], reverse=True)
    return rows[:k]


def align_session(
    transcript: str,
    video_result: dict | None = None,
    audio_result: dict | None = None,
    diarisation: list[dict] | None = None,
    duration_s: float | None = None,
) -> dict:
    """Full alignment for one session."""
    from .transcript import parse_transcript

    parsed = parse_transcript(transcript)
    if not parsed.turns:
        return {"available": False, "reason": "no turns parsed"}

    dur = duration_s or (video_result or {}).get("duration_s") \
        or (audio_result or {}).get("duration_s") or 0.0
    if not dur:
        return {"available": False,
                "reason": "no duration available; pass duration_s or a video result"}

    aligned, source = align_turns(
        parsed.turns, dur, diarisation, transcript.splitlines()
    )

    signals = (video_result or {}).get("signals", [])
    moments = (video_result or {}).get("moments", []) or []
    linked = link_moments(moments, aligned) if moments else []
    conv = find_convergences(aligned, signals) if signals else []

    return {
        "available": True,
        "duration_s": round(dur, 2),
        "n_turns": len(aligned),
        "timing_source": source,
        "timing_warning": (
            "Turn times are estimated from word counts, not measured. Use them "
            "to navigate approximately; they can drift by several seconds."
            if source == "estimated" else None
        ),
        "turns": [a.to_dict() for a in aligned],
        "linked_moments": linked,
        "convergences": conv,
        "principle": (
            "Alignment places a measurement beside the words spoken at the time. "
            "It does not infer why the two coincided."
        ),
    }
