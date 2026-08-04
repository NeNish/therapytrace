"""
Trajectory layer.

Turns a series of per-session TPI values into something a clinician can act on:

  * slope       - OLS trend in TPI points per session, with a t-test
  * momentum    - gaining / steady / plateauing / regressing
  * change_point- the session index that best splits the series into two
                  regimes with different means (max-t / binary segmentation)
  * corridor    - the client's own baseline +/- 1 SD band, so "progress" is
                  visibly defined as departure from their own norm
  * streaks     - consecutive sessions above or below baseline
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TrendResult:
    slope: float
    intercept: float
    t_stat: float
    p_approx: float
    r_squared: float
    n: int


def ols_trend(y: list[float]) -> TrendResult:
    n = len(y)
    if n < 3:
        return TrendResult(0.0, y[0] if y else 50.0, 0.0, 1.0, 0.0, n)
    x = list(range(n))
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    resid = [yi - (intercept + slope * xi) for xi, yi in zip(x, y)]
    sse = sum(r * r for r in resid)
    sst = sum((yi - my) ** 2 for yi in y)
    r2 = 1 - sse / sst if sst else 0.0
    df = n - 2
    se = math.sqrt(sse / df / sxx) if df > 0 and sxx else 0.0
    t = slope / se if se else 0.0
    return TrendResult(
        round(slope, 4), round(intercept, 3), round(t, 3),
        round(_p_from_t(abs(t), df), 4), round(r2, 4), n,
    )


def _p_from_t(t: float, df: int) -> float:
    """Two-tailed p, normal approximation with a small-sample correction.
    Good enough for a UI badge; swap in scipy.stats if exactness matters."""
    if df <= 0:
        return 1.0
    z = t * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def detect_change_point(y: list[float], min_seg: int = 2) -> dict | None:
    """
    Binary segmentation with a two-sample t statistic. Returns the split that
    maximises |t|, or None if no split is defensible.
    """
    n = len(y)
    if n < 2 * min_seg + 1:
        return None
    best = None
    for k in range(min_seg, n - min_seg + 1):
        a, b = y[:k], y[k:]
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        va = sum((v - ma) ** 2 for v in a) / max(1, len(a) - 1)
        vb = sum((v - mb) ** 2 for v in b) / max(1, len(b) - 1)
        pooled = math.sqrt(va / len(a) + vb / len(b))
        if pooled < 1e-9:
            continue
        t = (mb - ma) / pooled
        if best is None or abs(t) > abs(best["t"]):
            best = {
                "index": k,
                "t": round(t, 3),
                "mean_before": round(ma, 2),
                "mean_after": round(mb, 2),
                "shift": round(mb - ma, 2),
            }
    if best is None or abs(best["t"]) < 2.0:
        return None
    best["direction"] = "improvement" if best["shift"] > 0 else "deterioration"
    best["p_approx"] = round(_p_from_t(abs(best["t"]), n - 2), 4)
    return best


def classify_momentum(
    y: list[float], trend: TrendResult, overall: TrendResult | None = None
) -> dict:
    """
    Four states, deliberately conservative. The label is a prompt for a
    conversation in supervision, never a verdict.

    A six-session window has very little statistical power, so a direction is
    called either when the recent window is convincing on its own, or when it
    is suggestive AND the whole-series trend agrees with its sign.
    """
    n = len(y)
    if n < 3:
        return {
            "state": "insufficient_data",
            "label": "Not enough sessions yet",
            "detail": "Momentum needs at least three scored sessions.",
        }

    recent = y[-3:]
    spread = max(y) - min(y)
    volatile = _stdev(y) > 6.0

    def _directional(sign: int) -> bool:
        strong = sign * trend.slope > 1.0 and trend.p_approx < 0.20
        corroborated = (
            sign * trend.slope > 0.6
            and overall is not None
            and sign * overall.slope > 0.4
            and overall.p_approx < 0.15
        )
        return strong or corroborated

    if _directional(1):
        state, label = "gaining", "Gaining momentum"
        detail = f"TPI is rising about {trend.slope:.1f} points per session."
    elif _directional(-1):
        state, label = "regressing", "Losing ground"
        detail = f"TPI is falling about {abs(trend.slope):.1f} points per session."
    elif spread < 6.0 or abs(trend.slope) <= 0.25:
        state, label = "plateauing", "Plateau"
        detail = "Language has been stable for several sessions with little movement."
    else:
        state, label = "steady", "Mixed / steady"
        detail = "Movement session to session, but no clear direction yet."

    if volatile and state in ("plateauing", "steady"):
        detail += " Session-to-session variance is high."

    return {
        "state": state,
        "label": label,
        "detail": detail,
        "recent_mean": round(sum(recent) / len(recent), 2),
        "volatility": round(_stdev(y), 2),
    }


def _stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def streaks(y: list[float], reference: float = 50.0) -> dict:
    above = below = 0
    for v in reversed(y):
        if v >= reference and below == 0:
            above += 1
        elif v < reference and above == 0:
            below += 1
        else:
            break
    return {"above_baseline": above, "below_baseline": below}


RECENT_WINDOW = 6


def summarise(tpi_series: list[float]) -> dict:
    """
    Two trends are reported. The overall trend answers "has therapy moved this
    person since intake"; the recent trend answers "is it moving now". Momentum
    is classified on the recent window, because a client who improved for six
    sessions and has been sliding for four is not "gaining" — and a whole-series
    slope would say they were.
    """
    trend = ols_trend(tpi_series)
    window = tpi_series[-RECENT_WINDOW:]
    recent_trend = ols_trend(window)
    return {
        "n_sessions": len(tpi_series),
        "trend": {
            "slope_per_session": trend.slope,
            "t_stat": trend.t_stat,
            "p_approx": trend.p_approx,
            "r_squared": trend.r_squared,
        },
        "recent_trend": {
            "window": len(window),
            "slope_per_session": recent_trend.slope,
            "p_approx": recent_trend.p_approx,
        },
        "momentum": classify_momentum(window, recent_trend, trend),
        "change_point": detect_change_point(tpi_series),
        "streaks": streaks(tpi_series),
        "first": tpi_series[0] if tpi_series else None,
        "latest": tpi_series[-1] if tpi_series else None,
        "net_change": round(tpi_series[-1] - tpi_series[0], 2) if len(tpi_series) > 1 else 0.0,
    }
