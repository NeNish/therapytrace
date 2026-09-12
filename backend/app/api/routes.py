from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from ..db import get_db
from ..models import Client, OutcomeMeasure, TherapySession
from ..nlp.calibration import refit_weights
from ..nlp.features import DIMENSIONS
from ..nlp.pipeline import analyse_session
from ..nlp.therapist import INTERVENTION_LABELS
from ..schemas import (
    AnalyzeRequest, ClientCreate, MeasureCreate, SessionCreate,
)
from ..services import add_session, client_summary, client_trajectory, recompute_client

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------
# clients
# --------------------------------------------------------------------------

@router.get("/clients")
def list_clients(db: DbSession = Depends(get_db)):
    clients = db.scalars(select(Client).order_by(Client.code)).all()
    return [client_summary(db, c) for c in clients]


@router.post("/clients", status_code=201)
def create_client(payload: ClientCreate, db: DbSession = Depends(get_db)):
    if db.scalar(select(Client).where(Client.code == payload.code)):
        raise HTTPException(409, f"A case with the code {payload.code} already exists.")
    client = Client(**payload.model_dump())
    db.add(client)
    db.commit()
    db.refresh(client)
    return client_summary(db, client)


def _get_client(db: DbSession, client_id: int) -> Client:
    client = db.get(Client, client_id)
    if not client:
        raise HTTPException(404, "No case with that ID.")
    return client


@router.delete("/clients/{client_id}", status_code=204)
def delete_client(client_id: int, db: DbSession = Depends(get_db)):
    db.delete(_get_client(db, client_id))
    db.commit()


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------

@router.post("/clients/{client_id}/sessions", status_code=201)
def create_session(
    client_id: int, payload: SessionCreate, db: DbSession = Depends(get_db)
):
    client = _get_client(db, client_id)
    sess = add_session(
        db, client, payload.transcript, payload.session_number,
        payload.session_date, payload.source,
    )
    return _session_detail(sess)


@router.post("/clients/{client_id}/sessions/upload", status_code=201)
async def upload_session(
    client_id: int, file: UploadFile = File(...), db: DbSession = Depends(get_db)
):
    client = _get_client(db, client_id)
    raw = (await file.read()).decode("utf-8", errors="replace")
    if len(raw.strip()) < 20:
        raise HTTPException(400, "That file has no readable transcript in it.")
    sess = add_session(db, client, raw, None, None, f"file:{file.filename}")
    return _session_detail(sess)


@router.get("/clients/{client_id}/trajectory")
def get_trajectory(client_id: int, db: DbSession = Depends(get_db)):
    client = _get_client(db, client_id)
    roll = client_trajectory(db, client)
    return {
        "client": client_summary(db, client),
        "baseline": roll.get("baseline"),
        "sessions": [
            {
                "id": r["id"],
                "session_number": r["session_number"],
                "session_date": r["session_date"],
                "tpi": r["tpi"],
                "confidence": r["confidence"],
                "features": r["features"],
                "z_scores": r["z"],
                "contributions": r["contributions"],
            }
            for r in roll.get("sessions", [])
        ],
        "tpi_series": roll.get("tpi_series", []),
        "dimension_series": roll.get("dimension_series", {}),
        "trajectory": roll.get("trajectory"),
        "measures": roll.get("measures", []),
        "validation": roll.get("validation"),
        "dimensions": list(DIMENSIONS),
    }


def _session_detail(sess: TherapySession) -> dict:
    a = sess.analysis
    return {
        "id": sess.id,
        "client_id": sess.client_id,
        "session_number": sess.session_number,
        "session_date": sess.session_date,
        "transcript": sess.transcript,
        "tpi": a.tpi,
        "confidence": a.confidence,
        "features": a.features,
        "z_scores": a.z_scores,
        "contributions": a.contributions,
        "drivers": a.drivers,
        "utterance_series": a.utterance_series,
        "therapist_impact": a.therapist_impact,
        "therapist_moments": a.therapist_moments,
        "ml": a.ml or {},
        "parse_info": a.parse_info,
    }


@router.get("/sessions/{session_id}")
def get_session(session_id: int, db: DbSession = Depends(get_db)):
    sess = db.get(TherapySession, session_id)
    if not sess or not sess.analysis:
        raise HTTPException(404, "No analysed session with that ID.")
    return _session_detail(sess)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: int, db: DbSession = Depends(get_db)):
    sess = db.get(TherapySession, session_id)
    if not sess:
        raise HTTPException(404, "No session with that ID.")
    client = sess.client
    db.delete(sess)
    db.flush()
    db.refresh(client)
    recompute_client(db, client)
    db.commit()


# --------------------------------------------------------------------------
# therapist view
# --------------------------------------------------------------------------

@router.get("/clients/{client_id}/therapist-impact")
def therapist_impact(client_id: int, db: DbSession = Depends(get_db)):
    client = _get_client(db, client_id)
    pooled: dict[str, list[float]] = defaultdict(list)
    moments: list[dict] = []

    for s in sorted(client.sessions, key=lambda x: x.session_number):
        if not s.analysis:
            continue
        for row in s.analysis.therapist_impact or []:
            pooled[row["intervention"]].extend([row["mean_lift"]] * row["n"])
        for m in (s.analysis.therapist_moments or {}).get("openings", [])[:2]:
            moments.append({**m, "session_number": s.session_number})

    summary = []
    for name, lifts in pooled.items():
        n = len(lifts)
        mean = sum(lifts) / n
        summary.append({
            "intervention": name,
            "label": INTERVENTION_LABELS.get(name, name),
            "n": n,
            "mean_lift": round(mean, 4),
        })
    summary.sort(key=lambda r: r["mean_lift"], reverse=True)
    moments.sort(key=lambda m: m["lift"], reverse=True)

    return {
        "client_code": client.code,
        "therapist_code": client.therapist_code,
        "interventions": summary,
        "best_moments": moments[:8],
        "caveat": (
            "Association only. Therapists choose interventions in response to "
            "what was just said, so these figures are confounded by indication. "
            "Use them to start a supervision conversation, not to rank techniques."
        ),
    }


# --------------------------------------------------------------------------
# outcome measures + validation
# --------------------------------------------------------------------------

@router.post("/clients/{client_id}/measures", status_code=201)
def add_measure(client_id: int, payload: MeasureCreate, db: DbSession = Depends(get_db)):
    client = _get_client(db, client_id)
    existing = db.scalar(
        select(OutcomeMeasure).where(
            OutcomeMeasure.client_id == client.id,
            OutcomeMeasure.session_number == payload.session_number,
            OutcomeMeasure.instrument == payload.instrument,
        )
    )
    if existing:
        existing.score = payload.score
    else:
        db.add(OutcomeMeasure(client_id=client.id, **payload.model_dump()))
    db.commit()
    return {"ok": True}


@router.get("/calibration/weights")
def calibration(db: DbSession = Depends(get_db)):
    """Re-fit dimension weights across every case that has outcome measures."""
    rows, targets = [], []
    for client in db.scalars(select(Client)).all():
        measures = {
            (m.session_number, m.instrument): m.score for m in client.measures
        }
        if not measures:
            continue
        instrument = client.measures[0].instrument
        for s in client.sessions:
            key = (s.session_number, instrument)
            if s.analysis and key in measures:
                rows.append(s.analysis.features)
                targets.append(measures[key])
    weights = refit_weights(rows, targets) if len(rows) >= 8 else None
    return {
        "n_paired_sessions": len(rows),
        "refit_weights": weights,
        "note": (
            "Weights are re-fit by least squares against symptom change. "
            "Fewer than 8 paired sessions returns nothing — the fit would be noise."
        ),
    }


# --------------------------------------------------------------------------
# stateless preview
# --------------------------------------------------------------------------

@router.post("/analyze")
def analyze(payload: AnalyzeRequest):
    """Score one transcript without saving anything. Useful for a quick look."""
    return analyse_session(payload.transcript)


# --------------------------------------------------------------------------
# M10 — supervised models
# --------------------------------------------------------------------------

@router.get("/models")
def model_info():
    """
    What supervised models are loaded, and how well they actually perform.

    Reported figures come from held-out *conversations*, never held-out
    utterances — utterances from one conversation share a speaker and a topic,
    so an utterance-level split leaks and inflates every metric.
    """
    from ..nlp.ml_models import REPORTED_METRICS, available

    avail = available()
    return {
        "dataset": "AnnoMI (Wu et al., 2023) — 133 expert-annotated MI dialogues",
        "split_protocol": "GroupKFold / GroupShuffleSplit on transcript_id",
        "models": [
            {
                "id": "change_talk",
                "task": "Client utterance -> change / sustain / neutral",
                "loaded": avail["change_talk"],
                "algorithm": "TF-IDF (1-2 gram) + TherapyTrace lexicon -> logistic regression",
                "held_out_accuracy": REPORTED_METRICS["client"]["accuracy"],
                "held_out_macro_f1": REPORTED_METRICS["client"]["macro_f1"],
                "role": "External validation of the five process dimensions "
                        "against expert change-talk annotations.",
            },
            {
                "id": "therapist_behaviour",
                "task": "Therapist utterance -> question / reflection / therapist_input / other",
                "loaded": avail["therapist_behaviour"],
                "algorithm": "TF-IDF (1-2 gram) + TherapyTrace lexicon -> logistic regression",
                "held_out_accuracy": REPORTED_METRICS["therapist"]["accuracy"],
                "held_out_macro_f1": REPORTED_METRICS["therapist"]["macro_f1"],
                "role": "Replaces the regex intervention taxonomy in M6 with a "
                        "classifier trained on MISC-coded data.",
            },
        ],
    }


@router.get("/clients/{client_id}/profile")
def client_profile(client_id: int, db: DbSession = Depends(get_db)):
    """M11 — mechanism ordering and state-conditional response profile."""
    client = _get_client(db, client_id)
    from ..services import client_trajectory

    roll = client_trajectory(db, client)
    return {
        "client_code": client.code,
        "profile": roll.get("profile"),
        "note": (
            "Both analyses are per-person and observational. They generate "
            "hypotheses for supervision; they do not identify causal effects."
        ),
    }


@router.post("/review/video")
def review_video(payload: dict):
    """
    M15 — analyse a recorded session and surface the moments worth watching.

    Expects {"video_path": "...", "audio_path": "...", "output_video": "...",
             "max_seconds": 300}. Paths are server-side; this is a research
    tool operating on files already placed on the machine, not an upload
    endpoint, and it does not capture from any device.
    """
    from ..nlp.review import review_session

    video = payload.get("video_path")
    if not video:
        raise HTTPException(400, "video_path is required.")
    return review_session(
        video,
        output_video=payload.get("output_video"),
        audio_path=payload.get("audio_path"),
        max_seconds=payload.get("max_seconds"),
    )


@router.get("/clients/{client_id}/insights")
def client_insights(client_id: int, db: DbSession = Depends(get_db)):
    """
    M16 + M17 — the plain-English session note, plus what to consider next.

    Every sentence here traces to a measured number. Nothing is generated by a
    language model, so the same inputs always produce the same note and nothing
    can be fabricated.
    """
    client = _get_client(db, client_id)
    from ..services import client_trajectory

    roll = client_trajectory(db, client)
    return {
        "client_code": client.code,
        "narrative": roll.get("narrative"),
        "recommendation": roll.get("recommendation"),
        "principle": (
            "Decision support for reflection and supervision. Measurements are "
            "reported; interpretation and clinical decisions remain with the "
            "clinician."
        ),
    }
