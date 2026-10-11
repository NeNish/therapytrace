r"""
Fastest way to tell whether the multimodal tier can actually run.

Checks the three things that silently break video and audio analysis, and
prints what to do about each. Run it from the backend directory with the
project's own interpreter:

    .venv\Scripts\python.exe check_media.py            (Windows)
    .venv/bin/python check_media.py                    (macOS / Linux)

Optionally pass a video file to run a real analysis through the pipeline:

    .venv\Scripts\python.exe check_media.py clip.mp4
"""
import sys

ok = True

print(f"python      : {sys.version.split()[0]}")
if sys.version_info >= (3, 13):
    print("              PROBLEM: no MediaPipe release with the Solutions API")
    print("              ships a 3.13 wheel. Rebuild the venv on Python 3.12.")
    ok = False

try:
    import cv2
    print(f"opencv      : {cv2.__version__}")
except ImportError:
    print("opencv      : MISSING -> pip install opencv-python-headless")
    ok = False

try:
    import mediapipe as mp
    v = getattr(mp, "__version__", "unknown")
    if hasattr(mp, "solutions"):
        print(f"mediapipe   : {v}  (Solutions API present)")
    else:
        print(f"mediapipe   : {v}  PROBLEM: no Solutions API (removed in 0.10.30)")
        print("              -> pip install mediapipe==0.10.14")
        ok = False
except ImportError:
    print("mediapipe   : MISSING -> pip install mediapipe==0.10.14")
    ok = False

for mod, fix in (("librosa", "pip install librosa"), ("soundfile", "pip install soundfile")):
    try:
        __import__(mod)
        print(f"{mod:<12}: present")
    except ImportError:
        print(f"{mod:<12}: MISSING -> {fix}")
        ok = False

print()
if not ok:
    print("RESULT: the multimodal tier will report 'not available'.")
    print("        Fix the lines marked above, then run this again.")
    sys.exit(1)

print("RESULT: dependencies are correct.")

if len(sys.argv) > 1:
    clip = sys.argv[1]
    print(f"\nRunning {clip} through the pipeline...")
    from app.nlp.multimodal import run_multimodal_analysis

    r = run_multimodal_analysis({"video_path": clip, "max_seconds": 20})
    for name, ch in r.get("channels", {}).items():
        state = "OK" if ch.get("available") else "unavailable"
        print(f"  {name:<6}: {state}  {ch.get('reason', '')}")
    vid = r.get("channels", {}).get("video", {})
    if vid.get("available"):
        print(f"\n  pose detected in {vid.get('pose_detection_rate')} of sampled frames")
        print(f"  findings: {vid.get('n_findings')}")
        print("\nVideo analysis is working.")
    else:
        print("\nVideo still unavailable — the reason above names the cause.")
        sys.exit(1)
