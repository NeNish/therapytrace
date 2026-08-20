from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Client(Base):
    """
    A de-identified case. `code` is a pseudonymous study ID — no names, no
    contact details, ever. Deleting a Client cascades to every transcript.
    """

    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    presenting_issue: Mapped[str | None] = mapped_column(String(160), default=None)
    modality: Mapped[str | None] = mapped_column(String(64), default=None)
    therapist_code: Mapped[str | None] = mapped_column(String(32), default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)

    sessions: Mapped[list["TherapySession"]] = relationship(
        back_populates="client",
        cascade="all, delete-orphan",
        order_by="TherapySession.session_number",
    )
    measures: Mapped[list["OutcomeMeasure"]] = relationship(
        back_populates="client", cascade="all, delete-orphan"
    )


class TherapySession(Base):
    __tablename__ = "therapy_sessions"
    __table_args__ = (UniqueConstraint("client_id", "session_number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    session_number: Mapped[int] = mapped_column(Integer)
    session_date: Mapped[dt.date | None] = mapped_column(Date, default=None)
    source: Mapped[str] = mapped_column(String(64), default="upload")
    transcript: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)

    client: Mapped[Client] = relationship(back_populates="sessions")
    analysis: Mapped["SessionAnalysis | None"] = relationship(
        back_populates="session", cascade="all, delete-orphan", uselist=False
    )


class SessionAnalysis(Base):
    """Cached analysis output. TPI is recomputed whenever the baseline moves."""

    __tablename__ = "session_analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("therapy_sessions.id", ondelete="CASCADE"), unique=True
    )
    features: Mapped[dict] = mapped_column(JSON)
    tpi: Mapped[float] = mapped_column(Float)
    z_scores: Mapped[dict] = mapped_column(JSON)
    contributions: Mapped[dict] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    drivers: Mapped[dict] = mapped_column(JSON, default=dict)
    utterance_series: Mapped[list] = mapped_column(JSON, default=list)
    therapist_impact: Mapped[list] = mapped_column(JSON, default=list)
    therapist_moments: Mapped[dict] = mapped_column(JSON, default=dict)
    parse_info: Mapped[dict] = mapped_column(JSON, default=dict)
    ml: Mapped[dict] = mapped_column(JSON, default=dict)
    engine_version: Mapped[str] = mapped_column(String(16), default="1.0.0")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)

    session: Mapped[TherapySession] = relationship(back_populates="analysis")


class OutcomeMeasure(Base):
    """
    Self-report scores (PHQ-9, GAD-7, MADRS, ORS...) used to validate the TPI.
    This table is what turns the project from a demo into a study.
    """

    __tablename__ = "outcome_measures"

    id: Mapped[int] = mapped_column(primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("clients.id", ondelete="CASCADE"))
    session_number: Mapped[int] = mapped_column(Integer)
    instrument: Mapped[str] = mapped_column(String(24))
    score: Mapped[float] = mapped_column(Float)

    client: Mapped[Client] = relationship(back_populates="measures")
