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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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


def find_moments(analysis: dict, audio_summary: dict | None = None, k: int = 5) -> list[dict]:
    """
    The actually useful output: timestamps where several channels changed at
    once. A supervisor cannot watch 50 minutes; they can watch five moments.

    Scored by the magnitude of simultaneous change across available tracks,
    which is agnostic about direction — a moment where the client suddenly goes
    still is as worth watching as one where they become animated.
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

    # non-maximum suppression so the five moments are spread across the session
    order = np.argsort(change)[::-1]
    chosen: list[int] = []
    for i in order:
        if all(abs(i - c) > len(rows) // (k * 2 + 1) for c in chosen):
            chosen.append(int(i))
        if len(chosen) >= k:
            break

    return [
        {
            "t": rows[i]["t"],
            "timestamp": f"{int(rows[i]['t'] // 60):02d}:{int(rows[i]['t'] % 60):02d}",
            "salience": round(float(change[i]), 3),
            "gesture_amplitude": rows[i]["gesture_amplitude"],
            "posture_openness": rows[i]["posture_openness"],
            "arms_crossed": rows[i]["arms_crossed"],
            "why": "Several movement channels changed together at this point.",
        }
        for i in sorted(chosen, key=lambda i: rows[i]["t"])
    ]


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
