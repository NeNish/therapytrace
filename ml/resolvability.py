"""
M26 - How much can a single session actually resolve?

The interface shows a change-talk ratio for every session. Whether that number
can be read at all depends on how many directional turns the session contained,
and until now nothing told the user which case they were looking at.

This measures it directly. For a grid of session lengths it simulates sessions
at known true ratios, runs them through the estimator the application actually
ships, and records how wide the resulting credible interval is and how far
apart two sessions must be before the difference means anything.

The output is a table, not a fitted curve. A formula would be tidier and would
also be a second modelling assumption layered on the first; the table is what
was measured.

    python ml/resolvability.py           # writes ml/artifacts/resolvability.json

Read the result as an upper bound on the instrument's precision. The simulation
assumes turns are independent and that the confusion matrix measured on held-out
AnnoMI conversations carries over to the session being scored. Both are
optimistic, so real sessions resolve no better than this and probably worse.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.nlp.ml_models import (  # noqa: E402
    _directional_matrix,
    corrected_change_ratio,
)

OUT = Path(__file__).resolve().parent / "artifacts" / "resolvability.json"
GRID = [4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120]
N_SIM = 3000
SEED = 5

# An interval wider than this covers so much of the 0-1 scale that reading the
# midpoint as "the client's ratio" is not supportable. Chosen, not derived:
# half the scale is the point at which the estimate stops narrowing anything.
READABLE_CI = 0.50


def measure(n_turns: int, cc: float, sc: float, rng: random.Random) -> dict:
    widths, errors = [], []
    for _ in range(N_SIM):
        p = rng.random()
        theta = p * cc + (1.0 - p) * sc
        n_change = sum(1 for _ in range(n_turns) if rng.random() < theta)
        res = corrected_change_ratio(n_change, n_turns - n_change)
        if res["ratio"] is None:
            continue
        widths.append(res["hi"] - res["lo"])
        errors.append(res["ratio"] - p)

    mean_err = sum(errors) / len(errors)
    sd = math.sqrt(sum((e - mean_err) ** 2 for e in errors) / len(errors))
    # Smallest difference between two independent sessions that a two-sided
    # test at 5% would detect 80% of the time: (1.96 + 0.84) * sqrt(2) * sd.
    mdd = 2.80 * math.sqrt(2.0) * sd
    return {
        "turns": n_turns,
        "ci80_width": round(sum(widths) / len(widths), 4),
        "posterior_sd": round(sd, 4),
        "bias": round(mean_err, 4),
        "min_detectable_difference": round(mdd, 4),
        "readable": bool(sum(widths) / len(widths) <= READABLE_CI),
    }


def main() -> None:
    cc, _cs, sc, _ss = _directional_matrix()
    rng = random.Random(SEED)
    rows = [measure(n, cc, sc, rng) for n in GRID]

    readable = [r["turns"] for r in rows if r["readable"]]
    pairwise = [r["turns"] for r in rows if r["min_detectable_difference"] <= 0.25]

    payload = {
        "estimator": "bayesian posterior over true ratio, flat prior",
        "confusion_source": "held-out AnnoMI conversations, logistic regression",
        "p_change_given_change": round(cc, 4),
        "p_change_given_sustain": round(sc, 4),
        "readable_ci_threshold": READABLE_CI,
        "n_simulations_per_point": N_SIM,
        "turns_for_readable_estimate": min(readable) if readable else None,
        "turns_for_quarter_scale_comparison": min(pairwise) if pairwise else None,
        "assumptions": (
            "Turns independent; confusion matrix transfers to the scored "
            "session. Both optimistic, so these are upper bounds on precision."
        ),
        "table": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"{'turns':>6} {'80% CI width':>14} {'post SD':>9} {'min detectable diff':>21} {'readable':>9}")
    for r in rows:
        print(
            f"{r['turns']:>6} {r['ci80_width']:>14.3f} {r['posterior_sd']:>9.3f} "
            f"{r['min_detectable_difference']:>21.3f} {'yes' if r['readable'] else 'no':>9}"
        )
    print(f"\nA single session needs {payload['turns_for_readable_estimate']} directional "
          f"turns before its interval is narrower than {READABLE_CI} of the scale.")
    tq = payload["turns_for_quarter_scale_comparison"]
    print(
        "Comparing two sessions to a quarter of the scale needs "
        + (f"{tq} turns each." if tq else "more turns than any session provides.")
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
