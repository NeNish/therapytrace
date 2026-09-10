"""
M13 — Visual tier.

Facial and head-motion features from session video, using MediaPipe Face Mesh
(478 landmarks, CPU, no licence constraints — unlike OpenFace, which is
research-licensed and awkward to deploy).

A clinical caution that belongs in the code, not just the paper: facial
expressivity is a poor proxy for therapeutic work. A client looking away, face
still, may be doing the deepest processing of the session; a client smiling and
animated may be avoiding. So this module deliberately does NOT emit a valence
score. It emits *variability and movement* measures, which index expressive
range and engagement rather than a positive/negative reading:

  EXPRESSIVITY   au_activity           mean landmark displacement from neutral
                 expressivity_range    variability of that displacement
                 brow_raise, brow_furrow, mouth_open, smile_index
                                       geometric proxies for AU1/2, AU4,
                                       AU25/26 and AU12
  GAZE / HEAD    head_pitch/yaw/roll   pose from the canonical face model
                 head_motion           frame-to-frame pose change
                 gaze_aversion_ratio   proportion of frames looking away
  ENGAGEMENT     face_present_ratio    detection rate; also a quality check
                 blink_rate            per minute

As with acoustics, raw values are not comparable between people — face shape
alone shifts every geometric ratio. Everything is z-scored against the client's
own baseline before use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# canonical landmark indices in MediaPipe Face Mesh
L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]
BROW_L, BROW_R = [70, 63, 105, 66, 107], [300, 293, 334, 296, 336]
MOUTH_OUTER = [61, 291, 0, 17]
MOUTH_CORNERS = [61, 291]
NOSE_TIP, CHIN = 1, 199


def _ear(pts: np.ndarray, idx: list[int]) -> float:
    """Eye aspect ratio — low values indicate a closed lid."""
    p = pts[idx]
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return float((a + b) / (2 * c + 1e-9))


def extract_visual(video_path: str | Path, start: float = 0.0,
                   end: float | None = None, sample_fps: float = 6.0) -> dict[str, Any] | None:
    """
    Features for one turn. Sampled at `sample_fps` rather than full frame rate —
    facial dynamics relevant here operate well under 6 Hz, and sampling keeps a
    50-minute session tractable on CPU.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        return None
    if not Path(video_path).exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))
    first = int(start * fps)
    last = int(end * fps) if end else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    )

    disp, brows, mouths, smiles, ears, poses = [], [], [], [], [], []
    seen = detected = 0
    neutral = None
    idx = first

    try:
        while idx < last:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - first) % step:
                idx += 1
                continue
            seen += 1
            idx += 1

            res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                continue
            detected += 1

            lm = res.multi_face_landmarks[0].landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm])

            # scale-normalise by inter-ocular distance so face size cancels
            iod = np.linalg.norm(pts[33] - pts[263]) + 1e-9
            pts_n = (pts - pts[NOSE_TIP]) / iod

            if neutral is None:
                neutral = pts_n.copy()
            disp.append(float(np.mean(np.linalg.norm(pts_n - neutral, axis=1))))

            brows.append(float(np.mean(pts_n[BROW_L + BROW_R, 1])))
            mouths.append(float(np.linalg.norm(pts_n[MOUTH_OUTER[2]] - pts_n[MOUTH_OUTER[3]])))
            smiles.append(float(np.linalg.norm(pts_n[MOUTH_CORNERS[0]] - pts_n[MOUTH_CORNERS[1]])))
            ears.append((_ear(pts_n, L_EYE) + _ear(pts_n, R_EYE)) / 2)

            fwd = pts_n[CHIN] - pts_n[NOSE_TIP]
            poses.append([
                float(np.arctan2(fwd[1], fwd[2] + 1e-9)),
                float(np.arctan2(pts_n[263, 2] - pts_n[33, 2], pts_n[263, 0] - pts_n[33, 0])),
                float(np.arctan2(pts_n[263, 1] - pts_n[33, 1], pts_n[263, 0] - pts_n[33, 0])),
            ])
    finally:
        cap.release()
        mesh.close()

    if detected < 3:
        return {"available": False, "reason": "face not reliably detected",
                "face_present_ratio": round(detected / max(1, seen), 4)}

    poses = np.array(poses)
    ear_arr = np.array(ears)
    thresh = float(np.mean(ear_arr) - 1.2 * np.std(ear_arr))
    blinks = int(np.sum((ear_arr[1:] < thresh) & (ear_arr[:-1] >= thresh)))
    dur_min = max(1e-6, (last - first) / fps / 60.0)
    yaw = poses[:, 1]

    return {
        "available": True,
        "n_frames_sampled": seen,
        "face_present_ratio": round(detected / max(1, seen), 4),
        "au_activity": round(float(np.mean(disp)), 5),
        "expressivity_range": round(float(np.std(disp)), 5),
        "brow_raise": round(float(-np.mean(brows)), 5),
        "brow_variability": round(float(np.std(brows)), 5),
        "mouth_open": round(float(np.mean(mouths)), 5),
        "smile_index": round(float(np.mean(smiles)), 5),
        "smile_variability": round(float(np.std(smiles)), 5),
        "head_pitch": round(float(np.mean(poses[:, 0])), 4),
        "head_yaw": round(float(np.mean(yaw)), 4),
        "head_roll": round(float(np.mean(poses[:, 2])), 4),
        "head_motion": round(float(np.mean(np.abs(np.diff(poses, axis=0)))), 5)
        if len(poses) > 1 else 0.0,
        "gaze_aversion_ratio": round(float(np.mean(np.abs(yaw - np.median(yaw)) > 0.25)), 4),
        "blink_rate_per_min": round(blinks / dur_min, 2),
        "caveat": (
            "Expressive range and movement only. This module deliberately emits "
            "no valence score: stillness in therapy is as often deep processing "
            "as it is disengagement, and a smile is as often avoidance as "
            "affect. Interpreting these as mood would be a category error."
        ),
    }


def session_visual(video_path: str | Path, turns: list[dict],
                   baseline_rows: list[dict] | None = None) -> dict:
    from .acoustic import standardise_within_client

    if not Path(video_path).exists():
        return {"available": False, "reason": f"video not found: {video_path}"}

    rows = []
    for t in turns:
        if t.get("speaker") != "client":
            continue
        if float(t["end"]) - float(t["start"]) < 1.0:
            continue
        f = extract_visual(video_path, float(t["start"]), float(t["end"]))
        if f and f.get("available"):
            rows.append(f)

    if not rows:
        return {"available": False, "reason": "no usable client video segments"}

    rows = standardise_within_client(rows, baseline_rows or rows)
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    return {
        "available": True,
        "n_segments": len(rows),
        "per_turn": rows,
        "session_mean": {
            k: round(float(np.mean([r[k] for r in rows
                                    if isinstance(r.get(k), (int, float))])), 5)
            for k in numeric
        },
    }
