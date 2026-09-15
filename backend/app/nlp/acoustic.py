"""
M12 — Acoustic tier.

Why this module exists, stated precisely, because the distinction matters:

We tested whether *timestamp-derived* timing (speech rate, turn duration, pause
approximations at 1-second resolution) predicts therapeutic process on 2,343
real utterances. It does not — AUC 0.51-0.54 against a 0.50 floor, and adding
it to the text model made that model worse. Gradient boosting did not rescue it,
and neither did within-person standardisation.

That result rules out coarse timing. It says nothing about genuine acoustics.
Pitch contour, vocal energy, jitter, shimmer and true pause structure live at
millisecond resolution and are invisible to a transcript timestamp. The
depression-speech literature reports real effects for exactly these features,
so the honest position is: coarse timing is dead, fine acoustics are untested
here, and this module is what makes testing them possible.

Features extracted per utterance, all standard in clinical speech analysis:

  PROSODIC   f0 mean/sd/range        pitch level and variability; flattened
                                      pitch is a well-replicated depression marker
             energy mean/sd          vocal effort
             rate_syllables          syllable nuclei per second

  TEMPORAL   pause_count             silences over 250 ms
             pause_ratio             proportion of the turn spent silent
             longest_pause           the hesitation that precedes reformulation
             phonation_ratio         voiced fraction

  QUALITY    jitter, shimmer         cycle-to-cycle instability
             hnr                     harmonics-to-noise; breathiness
             spectral_centroid/rolloff/flux
             mfcc_1..13 mean/sd      standard timbral basis

The module is written so a missing or unreadable audio file degrades to None
rather than raising, because the text pipeline must keep working regardless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # multimodal extras absent; these modules self-disable
    np = None

SR = 16_000
MIN_PAUSE_S = 0.25
TOP_DB = 30


def _safe(fn, default=0.0):
    try:
        v = fn()
        return float(v) if np.isfinite(v) else default
    except Exception:
        return default


def extract_acoustic(path: str | Path, offset: float = 0.0,
                     duration: float | None = None) -> dict[str, Any] | None:
    """
    Feature vector for one utterance.

    `offset` and `duration` let a whole-session recording be sliced per turn
    using the diarisation the transcript already provides, so a single audio
    file serves every utterance in the session.
    """
    try:
        import librosa
    except ImportError:
        return None

    try:
        y, sr = librosa.load(str(path), sr=SR, offset=offset, duration=duration, mono=True)
    except Exception:
        return None

    if y.size < sr * 0.3:
        return None

    y = librosa.util.normalize(y)
    out: dict[str, Any] = {"duration_s": round(len(y) / sr, 3)}

    # ---------------- pitch ----------------
    try:
        f0, voiced, _ = librosa.pyin(
            y, sr=sr, fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C6")), frame_length=1024,
        )
        vf = f0[~np.isnan(f0)]
        if vf.size >= 5:
            out["f0_mean"] = round(float(np.mean(vf)), 2)
            out["f0_sd"] = round(float(np.std(vf)), 2)
            out["f0_range"] = round(float(np.percentile(vf, 95) - np.percentile(vf, 5)), 2)
            # Flattened pitch: variability relative to level, so it is
            # comparable across speakers with different vocal ranges.
            out["f0_cv"] = round(float(np.std(vf) / np.mean(vf)), 4)
            out["phonation_ratio"] = round(float(np.mean(voiced)), 4)
        else:
            out.update({k: None for k in
                        ("f0_mean", "f0_sd", "f0_range", "f0_cv", "phonation_ratio")})
    except Exception:
        out.update({k: None for k in
                    ("f0_mean", "f0_sd", "f0_range", "f0_cv", "phonation_ratio")})

    # ---------------- energy ----------------
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    out["energy_mean"] = round(float(np.mean(rms)), 5)
    out["energy_sd"] = round(float(np.std(rms)), 5)
    out["energy_cv"] = round(float(np.std(rms) / (np.mean(rms) + 1e-9)), 4)

    # ---------------- pauses ----------------
    intervals = librosa.effects.split(y, top_db=TOP_DB, frame_length=1024, hop_length=256)
    total = len(y) / sr
    if len(intervals):
        speech_s = float(sum(b - a for a, b in intervals) / sr)
        gaps = [
            (intervals[i + 1][0] - intervals[i][1]) / sr
            for i in range(len(intervals) - 1)
        ]
        long_gaps = [g for g in gaps if g >= MIN_PAUSE_S]
        out["speech_ratio"] = round(speech_s / total, 4)
        out["pause_ratio"] = round(1 - speech_s / total, 4)
        out["pause_count"] = len(long_gaps)
        out["pause_rate_per_s"] = round(len(long_gaps) / total, 4)
        out["longest_pause_s"] = round(max(long_gaps), 3) if long_gaps else 0.0
        out["mean_pause_s"] = round(float(np.mean(long_gaps)), 3) if long_gaps else 0.0
    else:
        out.update({"speech_ratio": 0.0, "pause_ratio": 1.0, "pause_count": 0,
                    "pause_rate_per_s": 0.0, "longest_pause_s": 0.0, "mean_pause_s": 0.0})

    # ---------------- rate ----------------
    try:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
        out["syllable_rate"] = round(len(onsets) / total, 3)
    except Exception:
        out["syllable_rate"] = None

    # ---------------- voice quality ----------------
    out["jitter"] = _safe(lambda: _jitter(y, sr))
    out["shimmer"] = _safe(lambda: _shimmer(y, sr))
    out["hnr"] = _safe(lambda: _hnr(y, sr))

    # ---------------- spectral ----------------
    out["spectral_centroid"] = round(
        float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))), 2)
    out["spectral_rolloff"] = round(
        float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))), 2)
    out["spectral_flux"] = round(
        float(np.mean(librosa.onset.onset_strength(y=y, sr=sr))), 4)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        out[f"mfcc{i + 1}_mean"] = round(float(np.mean(mfcc[i])), 3)
        out[f"mfcc{i + 1}_sd"] = round(float(np.std(mfcc[i])), 3)

    return out


def _jitter(y: np.ndarray, sr: int) -> float:
    """Cycle-to-cycle period variability, from zero-crossing period estimates."""
    zc = np.where(np.diff(np.signbit(y)))[0]
    if zc.size < 12:
        return 0.0
    periods = np.diff(zc) / sr
    periods = periods[(periods > 1 / 500) & (periods < 1 / 60)]
    if periods.size < 6:
        return 0.0
    return float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-12))


def _shimmer(y: np.ndarray, sr: int) -> float:
    """Cycle-to-cycle amplitude variability."""
    import librosa

    peaks = librosa.util.peak_pick(
        np.abs(y), pre_max=20, post_max=20, pre_avg=40,
        post_avg=40, delta=0.01, wait=20,
    )
    if peaks.size < 6:
        return 0.0
    amps = np.abs(y[peaks])
    return float(np.mean(np.abs(np.diff(amps))) / (np.mean(amps) + 1e-12))


def _hnr(y: np.ndarray, sr: int) -> float:
    """Harmonics-to-noise ratio in dB, via harmonic/percussive separation."""
    import librosa

    harm, perc = librosa.effects.hpss(y)
    ph, pp = float(np.mean(harm ** 2)), float(np.mean(perc ** 2))
    return 10.0 * np.log10((ph + 1e-12) / (pp + 1e-12))


# --------------------------------------------------------------------------
# within-person standardisation — the same correction the TPI applies
# --------------------------------------------------------------------------

def standardise_within_client(
    rows: list[dict], baseline_rows: list[dict], keys: list[str] | None = None
) -> list[dict]:
    """
    Acoustic features are strongly speaker-specific — vocal tract length alone
    shifts f0 by an octave. Raw values are therefore not comparable between
    clients, exactly as raw word counts are not. Every feature is z-scored
    against the client's own baseline sessions before it reaches any model.
    """
    if not baseline_rows:
        return rows
    keys = keys or [
        k for k, v in baseline_rows[0].items() if isinstance(v, (int, float))
    ]
    stats_: dict[str, tuple[float, float]] = {}
    for k in keys:
        vals = [r[k] for r in baseline_rows if isinstance(r.get(k), (int, float))]
        if len(vals) >= 2:
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) or 1e-6
            stats_[k] = (mu, sd)

    out = []
    for r in rows:
        z = dict(r)
        for k, (mu, sd) in stats_.items():
            if isinstance(r.get(k), (int, float)):
                z[f"{k}_z"] = round((r[k] - mu) / sd, 4)
        out.append(z)
    return out


def session_acoustics(
    audio_path: str | Path, turns: list[dict], baseline_rows: list[dict] | None = None
) -> dict:
    """
    turns: [{"speaker": "client", "start": 12.4, "end": 31.0}, ...]
    Slices one session recording per turn and aggregates the client's features.
    """
    if not Path(audio_path).exists():
        return {"available": False, "reason": f"audio not found: {audio_path}"}

    rows = []
    for t in turns:
        if t.get("speaker") != "client":
            continue
        dur = float(t["end"]) - float(t["start"])
        if dur < 0.5:
            continue
        f = extract_acoustic(audio_path, offset=float(t["start"]), duration=dur)
        if f:
            rows.append(f)

    if not rows:
        return {"available": False, "reason": "no usable client audio segments"}

    rows = standardise_within_client(rows, baseline_rows or rows)
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    agg = {
        k: round(float(np.mean([r[k] for r in rows if isinstance(r.get(k), (int, float))])), 4)
        for k in numeric
    }
    return {
        "available": True,
        "n_segments": len(rows),
        "per_turn": rows,
        "session_mean": agg,
        "caveat": (
            "Acoustic features are speaker-specific; values are z-scored "
            "against the client's own baseline before use. Coarse timing was "
            "tested and found uninformative (AUC 0.51); these are the "
            "millisecond-resolution features that test could not reach."
        ),
    }
