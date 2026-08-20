"""End-to-end analysis for a single session, plus a client-level roll-up."""

from __future__ import annotations

from .features import DIMENSIONS, session_features
from .ml_models import session_ml_summary
from .scoring import Baseline, build_baseline, score_session
from .therapist import aggregate_impact, analyse_turn_pairs, top_moments
from .trajectory import summarise
from .transcript import parse_transcript


def analyse_session(raw_transcript: str, baseline: Baseline | None = None) -> dict:
    parsed = parse_transcript(raw_transcript)
    client_texts = [t.text for t in parsed.client_turns]
    feats, utt_scores = session_features(client_texts)

    base = baseline or build_baseline([feats], baseline_k=1)
    scored = score_session(feats, base)

    pairs = analyse_turn_pairs(parsed.turns)

    # M10 — supervised classifiers. Returns availability flags and None-valued
    # sections when the trained artifacts are absent, so the pipeline behaves
    # identically on a fresh clone that has not run training.
    prev_for_client = []
    for t in parsed.client_turns:
        earlier = [x for x in parsed.turns if x.idx < t.idx]
        prev_for_client.append(earlier[-1].text if earlier else "")
    ml = session_ml_summary(
        client_texts=client_texts,
        therapist_texts=[t.text for t in parsed.therapist_turns],
        prev_for_client=prev_for_client,
    )

    # Utterance-level drivers: which client turns pulled each dimension up or down
    ranked = sorted(
        [(i, s) for i, s in enumerate(utt_scores) if s.n_words >= 12],
        key=lambda kv: kv[1].process_mean,
        reverse=True,
    )
    drivers = {
        "strongest": [
            {
                "turn": i,
                "text": s.text[:600],
                "process_mean": round(s.process_mean, 4),
                "scores": {d: getattr(s, d) for d in DIMENSIONS},
                "evidence": s.evidence,
            }
            for i, s in ranked[:5]
        ],
        "weakest": [
            {
                "turn": i,
                "text": s.text[:600],
                "process_mean": round(s.process_mean, 4),
                "scores": {d: getattr(s, d) for d in DIMENSIONS},
                "evidence": s.evidence,
            }
            for i, s in list(reversed(ranked))[:5]
        ],
    }

    return {
        "features": feats,
        "score": scored,
        "baseline": base.to_dict(),
        "utterance_series": [
            {
                "turn": i,
                "speaker": "client",
                "n_words": s.n_words,
                "process_mean": round(s.process_mean, 4),
                **{d: getattr(s, d) for d in DIMENSIONS},
            }
            for i, s in enumerate(utt_scores)
        ],
        "drivers": drivers,
        "ml": ml,
        "therapist_impact": aggregate_impact(pairs),
        "therapist_moments": top_moments(pairs),
        "parse": {
            "n_turns": len(parsed.turns),
            "n_client_turns": len(parsed.client_turns),
            "n_therapist_turns": len(parsed.therapist_turns),
            "speaker_map": parsed.speaker_map,
            "roles_inferred": parsed.inferred,
            "warnings": parsed.warnings,
        },
    }


def rollup_client(session_rows: list[dict]) -> dict:
    """
    session_rows: ordered list of {"session_number", "features", "tpi", "z", ...}
    Recomputes the baseline from the earliest sessions and rescores everything,
    so adding session 9 correctly leaves sessions 1-8 unchanged.
    """
    if not session_rows:
        return {"sessions": [], "trajectory": summarise([]), "baseline": None}

    base = build_baseline([r["features"] for r in session_rows], baseline_k=2)
    rescored = []
    for row in session_rows:
        s = score_session(row["features"], base)
        rescored.append({**row, **s})

    tpi_series = [r["tpi"] for r in rescored]
    dim_series = {
        d: [round(r["features"].get(d, 0.0), 4) for r in rescored] for d in DIMENSIONS
    }
    return {
        "baseline": base.to_dict(),
        "sessions": rescored,
        "tpi_series": tpi_series,
        "dimension_series": dim_series,
        "trajectory": summarise(tpi_series),
    }
