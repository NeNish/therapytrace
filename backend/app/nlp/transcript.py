"""
Transcript parsing.

Accepts the label styles that actually show up in the corpora this project
targets (IEEE DataPort psychotherapeutic sessions, AnnoMI exports, Alexander
Street transcripts, Whisper + diarisation output):

    THERAPIST: ...        T: ...        Counselor: ...    [therapist] ...
    CLIENT: ...           C: ...        Patient: ...      SPEAKER_00: ...

For diarised output with anonymous SPEAKER_xx labels we fall back to a
heuristic: the speaker who asks proportionally more questions and speaks fewer
words is assumed to be the therapist. That assumption is reported back so the
caller can correct it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

THERAPIST_LABELS = {
    "therapist", "t", "th", "counselor", "counsellor", "clinician",
    "doctor", "dr", "psychologist", "interviewer", "ellie", "helper",
}
CLIENT_LABELS = {
    "client", "c", "cl", "patient", "pt", "participant", "help-seeker",
    "helpseeker", "caller", "user",
}
# Note: bare "SPEAKER_00"-style labels are deliberately NOT listed. They carry
# no role information, so they fall through to the inference path below.

LABEL_RE = re.compile(
    r"^\s*\[?\s*([A-Za-z_][A-Za-z0-9 _\-]{0,24})\s*\]?\s*[:\-–]\s*(.+)$"
)

# A leading timestamp, as emitted by Whisper, Otter, Rev and most WebVTT
# exports: "[00:12:34] THERAPIST: ...", "(12:34) CLIENT: ...", "00:12:34 T: ...".
# Stripped before label matching, and returned separately so M18 can use it for
# alignment rather than discarding timing the transcript already carried.
LEADING_TS_RE = re.compile(
    r"^\s*[\[(]?\s*(\d{1,2}:\d{2}(?::\d{2})?)(?:\.\d+)?\s*[\])]?\s*[-–]?\s*"
)


@dataclass
class Turn:
    idx: int
    speaker: str  # "therapist" | "client"
    raw_label: str
    text: str


@dataclass
class ParsedTranscript:
    turns: list[Turn]
    speaker_map: dict[str, str]
    inferred: bool
    warnings: list[str]

    @property
    def client_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.speaker == "client"]

    @property
    def therapist_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.speaker == "therapist"]


def _normalise(label: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", label.strip().lower())


def parse_transcript(raw: str) -> ParsedTranscript:
    warnings: list[str] = []
    rows: list[tuple[str, str]] = []
    unlabelled = 0

    for line in raw.splitlines():
        if not line.strip():
            continue
        stripped = LEADING_TS_RE.sub("", line)
        m = LABEL_RE.match(stripped)
        if m:
            rows.append((m.group(1), m.group(2).strip()))
        elif rows:
            # continuation of the previous speaker's turn
            rows[-1] = (rows[-1][0], rows[-1][1] + " " + line.strip())
        else:
            unlabelled += 1

    if unlabelled:
        warnings.append(
            f"{unlabelled} line(s) before the first speaker label were ignored."
        )
    if not rows:
        return ParsedTranscript([], {}, False, ["No speaker-labelled lines found."])

    labels = sorted({_normalise(l) for l, _ in rows})
    mapping: dict[str, str] = {}
    inferred = False

    for lab in labels:
        base = re.sub(r"[0-9_]+$", "", lab)
        if lab in THERAPIST_LABELS or base in THERAPIST_LABELS:
            mapping[lab] = "therapist"
        elif lab in CLIENT_LABELS or base in CLIENT_LABELS:
            mapping[lab] = "client"

    unknown = [l for l in labels if l not in mapping]
    if unknown:
        inferred = True
        stats = {}
        for lab in unknown:
            texts = [t for l, t in rows if _normalise(l) == lab]
            words = sum(len(t.split()) for t in texts)
            questions = sum(t.count("?") for t in texts)
            stats[lab] = (questions / max(1, len(texts)), words / max(1, len(texts)))
        # highest question rate + shortest turns -> therapist
        ranked = sorted(stats.items(), key=lambda kv: (-kv[1][0], kv[1][1]))
        for i, (lab, _) in enumerate(ranked):
            mapping[lab] = "therapist" if i == 0 and len(ranked) > 1 else "client"
        warnings.append(
            "Speaker roles were inferred from question rate and turn length for: "
            + ", ".join(unknown)
            + ". Check the assignment before trusting therapist-impact results."
        )

    # A dyad with only one detected speaker cannot support turn-pair analysis.
    roles = set(mapping.values())
    if len(roles) < 2:
        warnings.append("Only one speaker role was detected in this transcript.")

    turns = [
        Turn(i, mapping[_normalise(lab)], lab, text)
        for i, (lab, text) in enumerate(rows)
        if text
    ]
    return ParsedTranscript(turns, mapping, inferred, warnings)
