"""
Calibrate and report the false-alarm rate of the alliance-strain screen.

An alert that fires on clients who are not changing is worse than no alert,
because a clinician cannot tell which kind they are looking at. This script
measures the rate directly: it simulates clients whose true TPI never moves,
feeds the resulting noise through the real detector, and counts how often it
speaks.

The noise scale is not invented here. It is the standard error of measurement
implied by the same assumptions calculate_rci uses, so the screen is held to
the same standard as the reliable-change statistic beside it.

    python ml/calibrate_rupture.py            # report current rates
    python ml/calibrate_rupture.py --sweep    # show rates across cut-offs

A second simulation reports sensitivity: how often the screen catches a client
who IS genuinely declining. A screen that never fires is trivially safe and
useless, so both numbers belong in any claim made about it.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.nlp.trajectory import (  # noqa: E402
    ELEVATED_AT,
    MODERATE_AT,
    SE_MEAS,
    assess_rupture_risk,
)

N_SIM = 20_000
N_SESSIONS = 12
SEED = 42


def simulate(true_slope: float, n_sim: int = N_SIM, n_sess: int = N_SESSIONS) -> dict:
    """Run the real detector over series with a known underlying slope."""
    rng = random.Random(SEED)
    levels = {"low": 0, "moderate": 0, "elevated": 0}
    scores = []
    for _ in range(n_sim):
        series = [
            50.0 + true_slope * i + rng.gauss(0.0, SE_MEAS) for i in range(n_sess)
        ]
        r = assess_rupture_risk(series)
        levels[r["level"]] += 1
        scores.append(r["score"])
    flagged = levels["moderate"] + levels["elevated"]
    return {
        "levels": {k: v / n_sim for k, v in levels.items()},
        "flagged": flagged / n_sim,
        "mean_score": statistics.fmean(scores),
    }


def sweep(n_sim: int = 5_000) -> None:
    """Rates at a range of cut-offs, for choosing MODERATE_AT deliberately."""
    # One generator per SERIES, not per value. Constructing random.Random(seed)
    # inside the inner comprehension reseeds on every draw, so all twelve
    # sessions receive an identical noise value: the null series comes out
    # perfectly flat and the declining series perfectly monotonic. That made
    # the sweep report a 0% false-alarm rate and 100% sensitivity, neither of
    # which is real.
    def _series(seed: int, slope: float) -> list[float]:
        rng = random.Random(seed)
        return [50.0 + slope * j + rng.gauss(0.0, SE_MEAS) for j in range(N_SESSIONS)]

    rng_null = [_series(SEED + i, 0.0) for i in range(n_sim)]
    rng_real = [_series(SEED + 10_000 + i, -1.2) for i in range(n_sim)]
    null_scores = [assess_rupture_risk(s)["score"] for s in rng_null]
    real_scores = [assess_rupture_risk(s)["score"] for s in rng_real]
    print(f"\n{'cut-off':>9} {'false alarm':>13} {'caught real decline':>22}")
    for cut in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        fa = sum(1 for s in null_scores if s >= cut) / len(null_scores)
        tp = sum(1 for s in real_scores if s >= cut) / len(real_scores)
        mark = "  <- MODERATE_AT" if abs(cut - MODERATE_AT) < 1e-9 else ""
        print(f"{cut:>9.2f} {fa:>12.1%} {tp:>21.1%}{mark}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="rates across score cut-offs")
    args = ap.parse_args()

    print(f"Measurement noise SD = {SE_MEAS:.2f} TPI points per session")
    print(f"Cut-offs in use: moderate >= {MODERATE_AT}, elevated >= {ELEVATED_AT}")

    null = simulate(true_slope=0.0)
    print(f"\nClients with NO true change ({N_SIM:,} simulated, {N_SESSIONS} sessions):")
    for k in ("low", "moderate", "elevated"):
        print(f"  {k:>9}: {null['levels'][k]:>6.1%}")
    print(f"  FALSE ALARM (moderate or above): {null['flagged']:.1%}")

    real = simulate(true_slope=-1.2)
    print(f"\nClients genuinely declining at 1.2 points/session:")
    for k in ("low", "moderate", "elevated"):
        print(f"  {k:>9}: {real['levels'][k]:>6.1%}")
    print(f"  CAUGHT (moderate or above): {real['flagged']:.1%}")

    if args.sweep:
        sweep()

    print(
        "\nBoth numbers matter. A screen that never fires has a perfect false-alarm\n"
        "rate and no use. Quote them together or not at all."
    )


if __name__ == "__main__":
    main()
