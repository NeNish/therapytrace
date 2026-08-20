from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from .models import Client, OutcomeMeasure, SessionAnalysis, TherapySession
from .nlp.calibration import validate_against_measure
from .nlp.pipeline import analyse_session, rollup_client
from .nlp.scoring import build_baseline, score_session

ENGINE_VERSION = "1.0.0"


def _ordered_sessions(db: DbSession, client_id: int) -> list[TherapySession]:
    """Every analysed session for a case, earliest first, read fresh from the DB."""
    rows = db.scalars(
        select(TherapySession)
        .where(TherapySession.client_id == client_id)
        .order_by(TherapySession.session_number)
    ).all()
    return [s for s in rows if s.analysis]


def add_session(
    db: DbSession, client: Client, payload_transcript: str,
    session_number: int | None, session_date=None, source: str = "upload",
) -> TherapySession:
    if session_number is None:
        session_number = (
            max((s.session_number for s in client.sessions), default=0) + 1
        )

    existing = db.scalar(
        select(TherapySession).where(
            TherapySession.client_id == client.id,
            TherapySession.session_number == session_number,
        )
    )
    if existing:
        existing.transcript = payload_transcript
        existing.session_date = session_date or existing.session_date
        sess = existing
    else:
        sess = TherapySession(
            client_id=client.id,
            session_number=session_number,
            session_date=session_date,
            source=source,
            transcript=payload_transcript,
        )
        db.add(sess)
    db.flush()

    result = analyse_session(payload_transcript)
    analysis = sess.analysis or SessionAnalysis(session_id=sess.id)
    analysis.features = result["features"]
    analysis.tpi = result["score"]["tpi"]
    analysis.z_scores = result["score"]["z"]
    analysis.contributions = result["score"]["contributions"]
    analysis.confidence = result["score"]["confidence"]
    analysis.drivers = result["drivers"]
    analysis.utterance_series = result["utterance_series"]
    analysis.therapist_impact = result["therapist_impact"]
    analysis.therapist_moments = result["therapist_moments"]
    analysis.ml = result.get("ml", {})
    analysis.parse_info = {
        **result["parse"],
        "confidence_notes": result["score"]["confidence_notes"],
    }
    analysis.engine_version = ENGINE_VERSION
    # Assign through the relationship, not just the FK. `sess.analysis` was read
    # a few lines above while it was still None, and SQLAlchemy caches that; a
    # bare db.add() would leave the relationship reporting None until the next
    # expiry, so the recompute below would skip the session it just scored.
    sess.analysis = analysis
    db.add(analysis)
    db.flush()

    recompute_client(db, client)
    db.commit()
    db.refresh(sess)
    db.refresh(sess.analysis)
    return sess


def recompute_client(db: DbSession, client: Client) -> None:
    """
    Rebuild the personal baseline from the earliest sessions and rescore all of
    them. Called whenever a session is added, edited or deleted, so a client's
    history is always internally consistent.

    Sessions are read straight from the database rather than through
    client.sessions: after a flush the in-memory collection can be stale, and a
    stale collection here silently leaves the newest session scored against a
    one-session baseline — which pins it at exactly 50 forever.
    """
    ordered = _ordered_sessions(db, client.id)
    if not ordered:
        return
    base = build_baseline([s.analysis.features for s in ordered], baseline_k=2)
    for s in ordered:
        scored = score_session(s.analysis.features, base)
        s.analysis.tpi = scored["tpi"]
        s.analysis.z_scores = scored["z"]
        s.analysis.contributions = scored["contributions"]
        s.analysis.confidence = scored["confidence"]
        s.analysis.parse_info = {
            **(s.analysis.parse_info or {}),
            "confidence_notes": scored["confidence_notes"],
        }
        db.add(s.analysis)


def client_trajectory(db: DbSession, client: Client) -> dict:
    ordered = _ordered_sessions(db, client.id)
    rows = [
        {
            "id": s.id,
            "session_number": s.session_number,
            "session_date": s.session_date,
            "features": s.analysis.features,
        }
        for s in ordered
    ]
    roll = rollup_client(rows)

    measures = db.scalars(
        select(OutcomeMeasure).where(OutcomeMeasure.client_id == client.id)
    ).all()
    validation = None
    if measures:
        instrument = measures[0].instrument
        by_session = {
            m.session_number: m.score for m in measures if m.instrument == instrument
        }
        tpi_by_session = {
            r["session_number"]: r["tpi"] for r in roll.get("sessions", [])
        }
        v = validate_against_measure(tpi_by_session, by_session)
        if v:
            validation = {"instrument": instrument, **v}

    roll["measures"] = [
        {"session_number": m.session_number, "instrument": m.instrument, "score": m.score}
        for m in measures
    ]
    roll["validation"] = validation
    return roll


def client_summary(db: DbSession, client: Client) -> dict:
    ordered = _ordered_sessions(db, client.id)
    traj = client_trajectory(db, client) if ordered else None
    return {
        "id": client.id,
        "code": client.code,
        "presenting_issue": client.presenting_issue,
        "modality": client.modality,
        "therapist_code": client.therapist_code,
        "n_sessions": len(ordered),
        "latest_tpi": traj["tpi_series"][-1] if traj and traj["tpi_series"] else None,
        "momentum": traj["trajectory"]["momentum"]["state"] if traj else None,
        "created_at": client.created_at,
    }
