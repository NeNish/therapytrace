from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from .api.routes import router
from .db import get_db, init_db
from .models import Client, OutcomeMeasure
from .nlp.features import AUXILIARY, DIMENSIONS
from .seed.generator import DEMO_CASES, generate_case
from .services import add_session

app = FastAPI(
    title="TherapyTrace",
    version="1.0.0",
    description=(
        "Longitudinal linguistic process monitoring for psychotherapy. "
        "Scores how a client's language changes against their own baseline, "
        "session over session. Decision support for reflection and supervision "
        "— not a diagnostic device."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("THERAPYTRACE_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


init_db()


@app.get("/api/health")
def health():
    return {"status": "ok", "engine_version": "1.0.0"}


@app.get("/api/meta")
def meta():
    return {
        "dimensions": [
            {"key": "self_agency", "label": "Self-agency",
             "question": "Does the client speak as someone who acts, or as someone acted upon?"},
            {"key": "future_orientation", "label": "Future orientation",
             "question": "Is attention on what comes next, or looping on what already happened?"},
            {"key": "emotional_granularity", "label": "Emotional granularity",
             "question": "Is affect named precisely, or left as undifferentiated 'bad'?"},
            {"key": "problem_ownership", "label": "Problem ownership",
             "question": "Is the difficulty located inside the client's reach, or entirely outside it?"},
            {"key": "reflection_depth", "label": "Reflection depth",
             "question": "Is the client describing events, or working out why they happen?"},
        ],
        "auxiliary": list(AUXILIARY),
        "dimension_keys": list(DIMENSIONS),
    }


@app.post("/api/demo/seed")
def seed_demo(db: DbSession = Depends(get_db)):
    """Load four synthetic cases with known trajectories. Safe to re-run."""
    created = []
    for i, (code, shape, n, issue, modality, therapist) in enumerate(DEMO_CASES):
        if db.scalar(select(Client).where(Client.code == code)):
            continue
        client = Client(
            code=code, presenting_issue=issue, modality=modality,
            therapist_code=therapist,
            notes=f"Synthetic case, programmed trajectory: {shape}. Not real clinical data.",
        )
        db.add(client)
        db.commit()
        db.refresh(client)

        for row in generate_case(shape, n, seed=1000 + i):
            add_session(
                db, client, row["transcript"], row["session_number"],
                source=f"synthetic:{shape}",
            )
            db.add(OutcomeMeasure(
                client_id=client.id, session_number=row["session_number"],
                instrument="PHQ-9", score=row["phq9"],
            ))
        db.commit()
        created.append({"code": code, "shape": shape, "sessions": n})

    return {
        "created": created,
        "note": "Synthetic data with programmed trajectories, for pipeline validation only.",
    }


# --------------------------------------------------------------------------
# serve the built frontend if it is present (single-container deployment)
# --------------------------------------------------------------------------

_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        return FileResponse(_dist / "index.html")
