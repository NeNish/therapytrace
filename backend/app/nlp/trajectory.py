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


def calculate_rci(
    tpi_series: list[float],
    baseline_sd: float = 10.0,
    reliability: float = 0.82,
) -> dict:
    """
    Jacobson & Truax (1991) Reliable Change Index.

        S_diff = sqrt(2 * (SD * sqrt(1 - r_xx))^2)

    With SD = 10 by construction of the TPI and r_xx = 0.82, S_diff = 6.0, so a
    change of at least 1.96 * S_diff = 11.76 points is reliable at p < 0.05.

    ASSUMPTION, NOT A MEASUREMENT. r_xx is the test-retest reliability of the
    TPI and it has never been measured: doing so needs repeated sessions from
    clients who are known to be stable, which this project does not have. The
    0.82 here is a stand-in, and the only 0.82 elsewhere in the repository is a
    single cross-validation fold's macro ROC-AUC, which is a different quantity
    and no justification for this one.

    Everything downstream scales with it: the 11.76-point threshold, and every
    alliance-strain threshold in assess_rupture_risk, which is expressed in
    units of S_diff. A lower true reliability widens the band and makes both
    more conservative. Treat the threshold as provisional and say so in any
    write-up; do not report it as a validated property of the instrument.
    """
    if len(tpi_series) < 2:
        return {
            "available": False,
            "reliable_change_threshold": 11.76,
            "status": "baseline_only",
            "net_change": 0.0,
            "rci_score": 0.0,
            "clinical_significance": "Insufficient sessions to calculate reliable change.",
        }

    s_diff = math.sqrt(2.0 * (baseline_sd ** 2) * (1.0 - reliability))
    net_change = tpi_series[-1] - tpi_series[0]
    rci_stat = net_change / s_diff if s_diff else 0.0
    crit = 1.96

    if rci_stat >= crit:
        status = "reliable_improvement"
        desc = (
            f"Client shows statistically reliable improvement (RCI = +{rci_stat:.2f}, p < 0.05). "
            f"Net gain of +{net_change:.1f} exceeds measurement error (threshold ±{crit * s_diff:.1f} pts)."
        )
    elif rci_stat <= -crit:
        status = "reliable_deterioration"
        desc = (
            f"Client shows statistically reliable deterioration (RCI = {rci_stat:.2f}, p < 0.05). "
            f"Net loss of {net_change:.1f} exceeds measurement error (threshold ±{crit * s_diff:.1f} pts)."
        )
    else:
        status = "no_reliable_change"
        desc = (
            f"Change of {net_change:+.1f} pts sits within the expected measurement noise band "
            f"(RCI = {rci_stat:.2f}, threshold ±{crit * s_diff:.1f} pts)."
        )

    return {
        "available": True,
        "reliable_change_threshold": round(crit * s_diff, 2),
        "s_diff": round(s_diff, 2),
        "status": status,
        "rci_score": round(rci_stat, 2),
        "net_change": round(net_change, 2),
        "clinical_significance": desc,
    }


def forecast_trajectory(
    tpi_series: list[float],
    horizon: int = 3,
    baseline_sd: float = 10.0,
) -> dict:
    """
    Damped linear-trend projection with bands that widen as the horizon grows.

    Not Bayesian, despite an earlier docstring saying so: there is no prior and
    no posterior here. The recent OLS slope is extrapolated and multiplied by
    0.85 per step, so the projection flattens toward the present value instead
    of running away. The bands are heuristic — hist_sd * sqrt(1 + step/2) — and
    describe how wrong the extrapolation has tended to be, not a posterior over
    future values. They are labelled "band" in the interface for that reason.

    A projection is the weakest thing on the page. It assumes the recent slope
    persists, which is exactly what a change point breaks.
    """
    n = len(tpi_series)
    if n < 3:
        return {
            "available": False,
            "reason": "Forecast requires at least 3 completed sessions.",
            "projections": [],
        }

    window = tpi_series[-RECENT_WINDOW:]
    trend = ols_trend(window)
    base_slope = trend.slope
    latest_val = tpi_series[-1]
    hist_sd = _stdev(window) if len(window) >= 2 else (baseline_sd * 0.5)
    hist_sd = max(2.5, min(12.0, hist_sd))

    projections = []
    curr_mean = latest_val

    for step in range(1, horizon + 1):
        # Damping factor: 0.85^step shrinks slope towards mean reversion
        damping = 0.85 ** step
        step_slope = base_slope * damping
        curr_mean = max(15.0, min(88.0, curr_mean + step_slope))

        # Standard error grows with forecast horizon: sqrt(step) * residual variance
        se_forecast = hist_sd * math.sqrt(1.0 + step * 0.5)
        # 80% CI (z = 1.282), 95% CI (z = 1.96)
        ci_80_lo = max(10.0, curr_mean - 1.282 * se_forecast)
        ci_80_hi = min(95.0, curr_mean + 1.282 * se_forecast)
        ci_95_lo = max(5.0, curr_mean - 1.96 * se_forecast)
        ci_95_hi = min(100.0, curr_mean + 1.96 * se_forecast)

        session_num = n + step
        projections.append({
            "session_number": session_num,
            "projected_tpi": round(curr_mean, 1),
            "ci_80": [round(ci_80_lo, 1), round(ci_80_hi, 1)],
            "ci_95": [round(ci_95_lo, 1), round(ci_95_hi, 1)],
            "expected_change": round(curr_mean - latest_val, 1),
        })

    # Summary outlook
    final_proj = projections[-1]["projected_tpi"]
    delta = final_proj - latest_val
    if delta >= 4.0:
        outlook = "Upward trajectory projected if the recent slope persists."
    elif delta <= -4.0:
        outlook = "Downward trajectory projected over the next three sessions."
    else:
        outlook = "Plateau/steady trajectory projected across the next 3 sessions."

    return {
        "available": True,
        "horizon": horizon,
        "baseline_slope": round(base_slope, 2),
        "outlook": outlook,
        "projections": projections,
    }


# Measurement-error scale for the TPI, from the same assumptions calculate_rci
# uses. Every threshold below is expressed as a multiple of this, so the alert
# cannot fire on a movement the instrument is unable to resolve.
S_DIFF = math.sqrt(2.0 * (10.0 ** 2) * (1.0 - 0.82))      # 6.0 TPI points
SE_MEAS = 10.0 * math.sqrt(1.0 - 0.82)                     # 4.24 points, single session

# Score cut-offs calibrated by simulation against series with no true change.
# ml/calibrate_rupture.py reproduces these and prints the resulting rates.
MODERATE_AT = 0.30
ELEVATED_AT = 0.55
FALSE_ALARM_RATE = 0.050   # measured, see ml/calibrate_rupture.py
DETECTION_RATE = 0.998     # measured, genuine decline of 1.2 TPI points/session


def assess_rupture_risk(
    tpi_series: list[float],
) -> dict:
    """
    Alliance-strain screen over the TPI series.

    This is a screening heuristic, not a clinical instrument, and it is
    deliberately conservative. An earlier version flagged a 6-point
    single-session drop, which is inside the band calculate_rci defines as
    measurement noise; on simulated clients with no true change it raised an
    alert for roughly one in three. Thresholds are now expressed in units of
    the measurement error itself and the score cut-offs are calibrated so that
    a non-changing client reaches "moderate" about 5% of the time.

    What it reports is that the series looks unlike noise, which is a reason to
    ask the client how the work is going. It is not a finding about the
    therapeutic alliance, and it does not recommend an intervention.
    """
    n = len(tpi_series)
    if n < 2:
        return {
            "level": "low",
            "score": 0.0,
            "alerts": [],
            "note": "Baseline is still being established.",
            "false_alarm_rate": FALSE_ALARM_RATE,
            "detection_rate": DETECTION_RATE,
            "threshold_points": round(1.96 * S_DIFF, 1),
        }

    alerts = []
    score = 0.0

    # 1. A single-session drop large enough to be distinguishable from noise.
    #    1.96 * S_DIFF is the same bar calculate_rci uses to call a change real.
    last_drop = tpi_series[-1] - tpi_series[-2]
    if last_drop <= -1.96 * S_DIFF:
        score += 0.50
        alerts.append(
            f"Drop of {abs(last_drop):.1f} points exceeds the "
            f"{1.96 * S_DIFF:.1f}-point reliable-change threshold."
        )
    elif last_drop <= -1.40 * S_DIFF:
        score += 0.25
        alerts.append(f"Drop of {abs(last_drop):.1f} points approaches the reliable-change threshold.")

    # 2. A run of declines longer than noise comfortably produces. Under pure
    #    noise a run of k declines has probability about 2^-k at each position,
    #    so 3 is common across a dozen sessions and 4 is not.
    consec_down = 0
    for i in range(len(tpi_series) - 1, 0, -1):
        if tpi_series[i] < tpi_series[i - 1]:
            consec_down += 1
        else:
            break
    if consec_down >= 5:
        score += 0.45
        alerts.append(f"Declines in {consec_down} consecutive sessions.")
    elif consec_down == 4:
        score += 0.25
        alerts.append("Declines in 4 consecutive sessions.")

    # 3. Recent sessions sitting below the personal baseline by more than the
    #    standard error of their mean, rather than any single dip below 45.
    recent = tpi_series[-3:]
    if len(recent) == 3:
        se_mean = SE_MEAS / math.sqrt(3)
        if (sum(recent) / 3.0) < 50.0 - 1.96 * se_mean:
            score += 0.30
            alerts.append("Last three sessions average below the client's own baseline.")

    # 4. Variance materially above what measurement noise alone produces.
    if n >= 4:
        recent_sd = _stdev(tpi_series[-4:])
        if recent_sd > 2.0 * SE_MEAS:
            score += 0.20
            alerts.append(f"Session-to-session variation (SD = {recent_sd:.1f}) is unusually wide.")

    score = min(1.0, round(score, 2))
    if score >= ELEVATED_AT:
        level = "elevated"
        note = (
            "The series is moving in a way noise alone rarely produces. "
            "Worth asking the client how they feel the work is going."
        )
    elif score >= MODERATE_AT:
        level = "moderate"
        note = "Some downward movement beyond measurement error. Worth keeping an eye on."
    else:
        level = "low"
        note = "Nothing in the series stands out from ordinary measurement variation."

    return {
        "level": level,
        "score": score,
        "alerts": alerts,
        "note": note,
        "false_alarm_rate": FALSE_ALARM_RATE,
        "detection_rate": DETECTION_RATE,
        "threshold_points": round(1.96 * S_DIFF, 1),
    }


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
        "rci": calculate_rci(tpi_series),
        "forecast": forecast_trajectory(tpi_series),
        "rupture_risk": assess_rupture_risk(tpi_series),
    }
