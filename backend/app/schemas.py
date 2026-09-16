from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ClientCreate(BaseModel):
    code: str = Field(min_length=1, max_length=32)
    presenting_issue: str | None = None
    modality: str | None = None
    therapist_code: str | None = None
    notes: str | None = None

    @field_validator("code")
    @classmethod
    def no_identifiers(cls, v: str) -> str:
        if " " in v.strip():
            raise ValueError("Use a pseudonymous code with no spaces, e.g. CL-014.")
        return v.strip()


class ClientOut(BaseModel):
    id: int
    code: str
    presenting_issue: str | None
    modality: str | None
    therapist_code: str | None
    n_sessions: int
    latest_tpi: float | None
    momentum: str | None
    created_at: dt.datetime


class SessionCreate(BaseModel):
    session_number: int | None = None
    session_date: dt.date | None = None
    source: str = "upload"
    transcript: str = Field(min_length=20)
    audio_path: str | None = None
    video_path: str | None = None
    max_seconds: float | None = 120


class SessionSummary(BaseModel):
    id: int
    session_number: int
    session_date: dt.date | None
    tpi: float
    confidence: float
    features: dict[str, float]
    z_scores: dict[str, float]
    contributions: dict[str, float]


class SessionDetail(SessionSummary):
    transcript: str
    drivers: dict[str, Any]
    utterance_series: list[dict[str, Any]]
    therapist_impact: list[dict[str, Any]]
    therapist_moments: dict[str, Any]
    parse_info: dict[str, Any]


class TrajectoryOut(BaseModel):
    client: ClientOut
    baseline: dict[str, Any] | None
    sessions: list[SessionSummary]
    tpi_series: list[float]
    dimension_series: dict[str, list[float]]
    trajectory: dict[str, Any]
    measures: list[dict[str, Any]] = []
    validation: dict[str, Any] | None = None


class AnalyzeRequest(BaseModel):
    transcript: str = Field(min_length=20)


class MeasureCreate(BaseModel):
    session_number: int
    instrument: str = Field(max_length=24)
    score: float
