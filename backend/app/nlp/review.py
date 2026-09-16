"""
M15 — Session Review Renderer.

Produces a supervision artefact from a recorded session: the video with
measurement overlays, plus a time-aligned signal track that can be scrubbed
alongside it.

A distinction this module is built around, because it is the one a reviewer
will press on:

    The overlay is DETECTION. The timeline is MEASUREMENT.
    Neither is interpretation, and the module never claims to supply it.

A box drawn around a face proves a face was found. It tells a supervisor
nothing. What is useful is the synchronised track underneath — gesture
amplitude, postural openness, pitch variability, speech rate, and the client's
process score — because that lets a supervisor jump straight to the moment
where something changed and watch it. The judgement stays with the human; the
system's contribution is finding the moment worth watching.

Signals rendered, all per second of session time:

  VISUAL     gesture_amplitude    wrist motion relative to shoulder width
             posture_openness     shoulder span, arms-crossed detection
             lean                 torso inclination toward or away
             head_motion          pose change frame to frame
             face_present         detection quality, also a QC track

  AUDIO      f0_variability       pitch range in the window
             energy               vocal effort
             speech_rate          syllable nuclei per second
             pause_ratio          silence proportion

  TEXT       process_score        the client's five-dimension score for the
                                  utterance active at that moment

Everything is scale-normalised by body geometry (shoulder width) or z-scored
within person, for the same reason the rest of the system is: raw values are
not comparable between people.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # multimodal extras absent; these modules self-disable
    np = None

# MediaPipe Pose landmark indices
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24

INK = (31, 32, 20)
GAIN = (92, 111, 31)
REGRESS = (78, 44, 142)
MARK = (20, 124, 180)
PAPER = (231, 237, 234)


@dataclass
class FrameSignals:
    t: float
    face_present: bool = False
    pose_present: bool = False
    gesture_amplitude: float = 0.0
    posture_openness: float = 0.0
    lean: float = 0.0
    head_motion: float = 0.0
    arms_crossed: bool = False
    boxes: list[tuple] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "t": round(self.t, 2),
            "face_present": self.face_present,
            "pose_present": self.pose_present,
            "gesture_amplitude": round(self.gesture_amplitude, 4),
            "posture_openness": round(self.posture_openness, 4),
            "lean": round(self.lean, 4),
            "head_motion": round(self.head_motion, 4),
            "arms_crossed": self.arms_crossed,
        }


def analyse_video(
    video_path: str | Path,
    sample_fps: float = 5.0,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """
    Walks the video once, extracting pose and face signals per sampled frame.

    Sampling at 5 fps rather than full rate: the postural and gestural dynamics
    that matter in a counselling session operate well below 5 Hz, and sampling
    keeps a 50-minute recording tractable on CPU.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        return {"available": False, "reason": "opencv or mediapipe not installed"}

    if not Path(video_path).exists():
        return {"available": False, "reason": f"video not found: {video_path}"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"available": False, "reason": "could not open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_seconds:
        total = min(total, int(max_seconds * fps))
    step = max(1, int(round(fps / sample_fps)))

    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

    signals: list[FrameSignals] = []
    prev_wrists = prev_nose = None
    idx = 0

    try:
        while idx < total:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            t = idx / fps
            idx += 1

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            s = FrameSignals(t=t)

            fres = face.process(rgb)
            if fres.detections:
                s.face_present = True
                for det in fres.detections:
                    bb = det.location_data.relative_bounding_box
                    s.boxes.append((
                        "face",
                        int(bb.xmin * w), int(bb.ymin * h),
                        int(bb.width * w), int(bb.height * h),
                        float(det.score[0]),
                    ))

            pres = pose.process(rgb)
            if pres.pose_landmarks:
                s.pose_present = True
                lm = pres.pose_landmarks.landmark
                pts = np.array([[p.x * w, p.y * h] for p in lm])

                shoulder_w = float(np.linalg.norm(pts[L_SH] - pts[R_SH])) + 1e-6
                wrists = np.array([pts[L_WR], pts[R_WR]])

                if prev_wrists is not None:
                    s.gesture_amplitude = float(
                        np.mean(np.linalg.norm(wrists - prev_wrists, axis=1)) / shoulder_w
                    )
                prev_wrists = wrists

                if prev_nose is not None:
                    s.head_motion = float(np.linalg.norm(pts[NOSE] - prev_nose) / shoulder_w)
                prev_nose = pts[NOSE]

                hip_mid = (pts[L_HIP] + pts[R_HIP]) / 2
                sh_mid = (pts[L_SH] + pts[R_SH]) / 2
                s.lean = float((hip_mid[0] - sh_mid[0]) / shoulder_w)

                # arms crossed: wrists on the opposite side of the body midline
                mid_x = sh_mid[0]
                s.arms_crossed = bool(
                    (pts[L_WR][0] < mid_x - 0.05 * shoulder_w)
                    and (pts[R_WR][0] > mid_x + 0.05 * shoulder_w)
                )
                elbow_span = float(np.linalg.norm(pts[L_EL] - pts[R_EL]))
                s.posture_openness = round(elbow_span / shoulder_w, 4)

                x = pts[[L_SH, R_SH, L_HIP, R_HIP], 0]
                y = pts[[L_SH, R_SH, L_HIP, R_HIP], 1]
                s.boxes.append((
                    "torso", int(x.min()), int(y.min()),
                    int(x.max() - x.min()), int(y.max() - y.min()), 1.0,
                ))

            signals.append(s)
    finally:
        cap.release()
        pose.close()
        face.close()

    if not signals:
        return {"available": False, "reason": "no frames processed"}

    rows = [s.to_dict() for s in signals]
    det = float(np.mean([r["pose_present"] for r in rows]))

    return {
        "available": True,
        "fps": fps,
        "duration_s": round(len(signals) * step / fps, 2),
        "n_sampled": len(signals),
        "pose_detection_rate": round(det, 4),
        "face_detection_rate": round(float(np.mean([r["face_present"] for r in rows])), 4),
        "signals": rows,
        "_frames": signals,
        "summary": {
            "gesture_amplitude_mean": round(
                float(np.mean([r["gesture_amplitude"] for r in rows])), 4),
            "gesture_amplitude_sd": round(
                float(np.std([r["gesture_amplitude"] for r in rows])), 4),
            "posture_openness_mean": round(
                float(np.mean([r["posture_openness"] for r in rows if r["pose_present"]] or [0])), 4),
            "arms_crossed_ratio": round(
                float(np.mean([r["arms_crossed"] for r in rows])), 4),
            "lean_mean": round(float(np.mean([r["lean"] for r in rows])), 4),
            "head_motion_mean": round(float(np.mean([r["head_motion"] for r in rows])), 4),
        },
        "caveat": (
            "Detection and measurement only. Gesture amplitude and postural "
            "openness describe movement, not meaning: crossed arms are as often "
            "cold as defensive, and stillness is as often concentration as "
            "disengagement. These tracks exist to help a supervisor find the "
            "moment worth watching, not to interpret it for them."
        ),
    }


def render_annotated_video(
    video_path: str | Path,
    output_path: str | Path,
    analysis: dict,
    sample_fps: float = 5.0,
) -> dict:
    """
    Writes a copy of the video with detection overlays and a live signal strip.

    What is drawn is deliberately limited to what was actually measured — boxes
    where something was detected, and the current value of each track. No
    inferred emotion labels are ever drawn on the frame.
    """
    try:
        import cv2
    except ImportError:
        return {"available": False, "reason": "opencv not installed"}

    frames: list[FrameSignals] = analysis.get("_frames", [])
    if not frames:
        return {"available": False, "reason": "no analysis frames"}

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    strip = 100
    out = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h + strip)
    )

    by_time = {round(f.t, 2): f for f in frames}
    times = sorted(by_time)
    gmax = max([f.gesture_amplitude for f in frames] + [1e-6])
    idx = 0
    last = frames[0]

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            idx += 1

            nearest = min(times, key=lambda x: abs(x - t))
            if abs(nearest - t) < 1.0 / sample_fps:
                last = by_time[nearest]

            for kind, x, y, bw, bh, score in last.boxes:
                col = GAIN if kind == "face" else MARK
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 2)
                label = f"{kind} {score:.2f}" if kind == "face" else "torso"
                cv2.rectangle(frame, (x, max(0, y - 20)), (x + 118, y), col, -1)
                cv2.putText(frame, label, (x + 4, max(12, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            canvas = np.full((h + strip, w, 3), 30, np.uint8)
            canvas[:h] = frame

            def bar(i, name, val, vmax, col):
                x0 = 14 + i * (w - 28) // 4
                bw_ = (w - 28) // 4 - 18
                cv2.putText(canvas, name, (x0, h + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 205, 200), 1)
                cv2.rectangle(canvas, (x0, h + 32), (x0 + bw_, h + 46), (70, 74, 70), -1)
                filled = int(bw_ * min(1.0, val / (vmax + 1e-9)))
                cv2.rectangle(canvas, (x0, h + 32), (x0 + filled, h + 46), col, -1)
                cv2.putText(canvas, f"{val:.3f}", (x0, h + 66),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 240, 235), 1)

            bar(0, "GESTURE", last.gesture_amplitude, gmax, GAIN)
            bar(1, "OPENNESS", last.posture_openness, 3.0, MARK)
            bar(2, "HEAD MOTION", last.head_motion, 0.4, GAIN)
            bar(3, "LEAN", abs(last.lean), 0.5, REGRESS)

            cv2.rectangle(canvas, (w - 108, h + 4), (w - 8, h + 24), (52, 56, 52), -1)
            cv2.putText(canvas, f"t={t:6.2f}s", (w - 102, h + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (215, 220, 215), 1)
            if last.arms_crossed:
                cv2.rectangle(canvas, (w - 150, h + 72), (w - 8, h + 90), REGRESS, -1)
                cv2.putText(canvas, "ARMS CROSSED", (w - 144, h + 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            out.write(canvas)
    finally:
        cap.release()
        out.release()

    return {"available": True, "output": str(output_path), "frames_written": idx}


def _clock_short(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def _describe_moment(rows: list[dict], idx: int) -> tuple[str, str, list[str]]:
    """
    Returns (observation, supervision_question, signal_tags) for one sampled
    frame. Observations name measured quantities; questions stay non-inferential.
    """
    r = rows[idx]
    ga_vals = [x["gesture_amplitude"] for x in rows]
    po_vals = [x["posture_openness"] for x in rows if x.get("pose_present")]
    hm_vals = [x["head_motion"] for x in rows]
    ga_mu = float(np.mean(ga_vals)) if ga_vals else 0.0
    po_mu = float(np.mean(po_vals)) if po_vals else 0.0
    hm_mu = float(np.mean(hm_vals)) if hm_vals else 0.0

    tags: list[str] = []
    parts: list[str] = []

    if r.get("arms_crossed"):
        tags.append("arms_crossed")
        parts.append("arms were folded at this point")
    ga = r.get("gesture_amplitude", 0.0)
    if ga_mu and ga > ga_mu * 1.6:
        tags.append("high_gesture")
        parts.append("hand and arm movement ran well above this clip's average")
    elif ga_mu and ga < ga_mu * 0.45:
        tags.append("low_gesture")
        parts.append("movement dropped to unusually low levels")
    po = r.get("posture_openness", 0.0)
    if po_mu and po > po_mu * 1.35:
        tags.append("open_posture")
        parts.append("elbow span widened — posture more open than elsewhere in the clip")
    elif po_mu and po < po_mu * 0.65:
        tags.append("closed_posture")
        parts.append("shoulder span narrowed relative to the rest of the recording")
    hm = r.get("head_motion", 0.0)
    if hm_mu and hm > hm_mu * 2.0:
        tags.append("head_shift")
        parts.append("head position shifted abruptly")
    lean = r.get("lean", 0.0)
    if abs(lean) > 0.14:
        tags.append("leaned_away" if lean < 0 else "leaned_toward")
        parts.append(
            f"torso angled {'away from' if lean < 0 else 'toward'} the camera"
        )

    if not parts:
        parts.append("multiple movement channels changed together")

    observation = f"At {_clock_short(r['t'])}, " + "; ".join(parts) + "."

    if "arms_crossed" in tags or "closed_posture" in tags:
        question = (
            "What topic was active here? In supervision, ask whether closure "
            "coincided with difficulty the client was naming — not whether they "
            "were 'being defensive'."
        )
    elif "low_gesture" in tags:
        question = (
            "Stillness at this timestamp — in the room, did this feel like deep "
            "processing or disengagement? The recording cannot answer that; "
            "the therapist can."
        )
    elif "high_gesture" in tags or "head_shift" in tags:
        question = (
            "Movement peaked here. Was the client working out a plan, reacting "
            "to something the therapist said, or shifting to a new topic?"
        )
    elif "open_posture" in tags or "leaned_toward" in tags:
        question = (
            "Posture opened or leaned in. Did this coincide with the client "
            "taking ownership of their part, or naming a next step?"
        )
    else:
        question = (
            "Several channels shifted at once — worth locating this moment in "
            "the session recording and asking what was happening relationally."
        )

    return observation, question, tags


def find_moments(analysis: dict, audio_summary: dict | None = None, k: int = 8) -> list[dict]:
    """
    Timestamps where several channels changed at once, with therapist-readable
    observations and non-inferential supervision questions.
    """
    rows = analysis.get("signals", [])
    if len(rows) < 10:
        return []

    def z(key: str) -> np.ndarray:
        v = np.array([r[key] for r in rows], dtype=float)
        sd = v.std() or 1e-6
        return (v - v.mean()) / sd

    tracks = [z("gesture_amplitude"), z("head_motion"), z("posture_openness")]
    change = np.zeros(len(rows))
    for tr in tracks:
        change[1:] += np.abs(np.diff(tr))

    order = np.argsort(change)[::-1]
    chosen: list[int] = []
    min_gap = max(3, len(rows) // (k * 2 + 1))
    for i in order:
        if all(abs(i - c) > min_gap for c in chosen):
            chosen.append(int(i))
        if len(chosen) >= k:
            break

    out = []
    for i in sorted(chosen, key=lambda j: rows[j]["t"]):
        obs, question, tags = _describe_moment(rows, i)
        out.append({
            "t": rows[i]["t"],
            "timestamp": _clock_short(rows[i]["t"]),
            "salience": round(float(change[i]), 3),
            "gesture_amplitude": rows[i]["gesture_amplitude"],
            "posture_openness": rows[i]["posture_openness"],
            "head_motion": rows[i]["head_motion"],
            "arms_crossed": rows[i]["arms_crossed"],
            "lean": rows[i].get("lean"),
            "tags": tags,
            "observation": obs,
            "supervision_question": question,
            "description": obs,
            "why": obs,
        })
    return out


def extract_preview_frames(
    video_path: str | Path,
    timestamps: list[float],
    signals: list[dict] | None = None,
    max_width: int = 420,
) -> list[dict]:
    """
    Grab JPEG preview frames at the given timestamps. Returns base64-encoded
    thumbnails with optional measurement overlay for supervision review.
    """
    try:
        import cv2
    except ImportError:
        return []

    path = Path(video_path)
    if not path.exists() or not timestamps:
        return []

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sig_by_t = {}
    if signals:
        for s in signals:
            sig_by_t[round(s["t"], 1)] = s

    previews: list[dict] = []
    try:
        for t in sorted(set(timestamps))[:12]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            scale = min(1.0, max_width / w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            sig = min(sig_by_t.keys(), key=lambda x: abs(x - t), default=None)
            nearest = sig_by_t.get(sig) if sig is not None and abs(sig - t) < 2.0 else None

            ts_label = _clock_short(t)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (31, 32, 20), -1)
            cv2.putText(frame, ts_label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (231, 237, 234), 1, cv2.LINE_AA)

            if nearest:
                bar_y = frame.shape[0] - 36
                cv2.rectangle(frame, (0, bar_y), (frame.shape[1], frame.shape[0]), (31, 32, 20), -1)
                ga = nearest.get("gesture_amplitude", 0)
                bar_w = int(min(frame.shape[1] - 20, ga * 200))
                cv2.rectangle(frame, (10, bar_y + 10), (10 + bar_w, bar_y + 22), (92, 111, 31), -1)
                label = "ARMS CROSSED" if nearest.get("arms_crossed") else f"gesture {ga:.2f}"
                cv2.putText(frame, label, (10, bar_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (200, 205, 200), 1, cv2.LINE_AA)

            ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok_enc:
                continue

            previews.append({
                "t": round(t, 2),
                "timestamp": ts_label,
                "preview_jpeg_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                "width": frame.shape[1],
                "height": frame.shape[0],
                "arms_crossed": nearest.get("arms_crossed") if nearest else None,
                "gesture_amplitude": nearest.get("gesture_amplitude") if nearest else None,
            })
    finally:
        cap.release()

    return previews


def attach_frame_previews(
    video_path: str | Path,
    moments: list[dict],
    findings: list[dict] | None = None,
    signals: list[dict] | None = None,
) -> dict[float, dict]:
    """Extract previews for moments and body findings; return map by timestamp."""
    times: list[float] = [m["t"] for m in moments if "t" in m]
    for f in findings or []:
        if f.get("start") is not None:
            times.append(float(f["start"]))
    previews = extract_preview_frames(video_path, times, signals)
    return {p["t"]: p for p in previews}


def review_session(
    video_path: str | Path,
    output_video: str | Path | None = None,
    audio_path: str | Path | None = None,
    max_seconds: float | None = None,
) -> dict:
    """One call: analyse, optionally render, and surface the moments."""
    analysis = analyse_video(video_path, max_seconds=max_seconds)
    if not analysis.get("available"):
        return analysis

    result = {
        "video": {k: v for k, v in analysis.items() if k != "_frames"},
        "moments": find_moments(analysis),
    }

    if audio_path and Path(str(audio_path)).exists():
        from .acoustic import extract_acoustic

        result["audio"] = extract_acoustic(audio_path)

    if output_video:
        result["render"] = render_annotated_video(video_path, output_video, analysis)

    return result
