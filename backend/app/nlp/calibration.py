"""
Validation.

A progress index nobody has checked is just a number. This module provides the
minimum evidence a reviewer will ask for:

  * within-person correlation between TPI and a symptom measure (PHQ-9, GAD-7,
    MADRS...). Expected sign is NEGATIVE — as language improves, symptoms fall.
  * lead-lag: does a TPI change at session s predict the symptom change at
    session s+1 better than the reverse? If language moves first, the index is
    doing something a questionnaire is not.
  * weight re-fitting by ridge-free least squares against the measure, so the
    dimension weights stop being the authors' guesses.
"""

from __future__ import annotations

import math

from .features import DIMENSIONS


def pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3 or n != len(y):
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return round(num / (dx * dy), 4) if dx and dy else None


def _rank(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    return pearson(_rank(x), _rank(y))


def validate_against_measure(
    tpi_by_session: dict[int, float], measure_by_session: dict[int, float]
) -> dict | None:
    shared = sorted(set(tpi_by_session) & set(measure_by_session))
    if len(shared) < 3:
        return None
    t = [tpi_by_session[s] for s in shared]
    m = [measure_by_session[s] for s in shared]

    result = {
        "n_paired_sessions": len(shared),
        "pearson_r": pearson(t, m),
        "spearman_rho": spearman(t, m),
        "expected_sign": "negative",
    }

    # lead-lag on first differences
    if len(shared) >= 5:
        dt_ = [t[i + 1] - t[i] for i in range(len(t) - 1)]
        dm = [m[i + 1] - m[i] for i in range(len(m) - 1)]
        result["language_leads_symptoms_r"] = pearson(dt_[:-1], dm[1:])
        result["symptoms_lead_language_r"] = pearson(dm[:-1], dt_[1:])
    return result


def refit_weights(
    feature_rows: list[dict[str, float]], target: list[float]
) -> dict[str, float] | None:
    """
    Least squares of z-scored dimensions on a (sign-flipped) symptom measure,
    normalised to sum to 1. Returns None if the system is underdetermined.
    """
    n = len(feature_rows)
    if n < len(DIMENSIONS) + 2 or n != len(target):
        return None

    cols = {d: [r.get(d, 0.0) for r in feature_rows] for d in DIMENSIONS}
    means = {d: sum(v) / n for d, v in cols.items()}
    sds = {
        d: math.sqrt(sum((x - means[d]) ** 2 for x in v) / (n - 1)) or 1e-6
        for d, v in cols.items()
    }
    X = [[(cols[d][i] - means[d]) / sds[d] for d in DIMENSIONS] for i in range(n)]
    y = [-t for t in target]  # improvement = falling symptoms

    p = len(DIMENSIONS)
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(p)] for a in range(p)]
    Xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(p)]
    for a in range(p):
        XtX[a][a] += 1e-3  # tiny ridge for stability

    beta = _solve(XtX, Xty)
    if beta is None:
        return None
    pos = [max(0.0, b) for b in beta]
    total = sum(pos)
    if total <= 0:
        return None
    return {d: round(pos[i] / total, 4) for i, d in enumerate(DIMENSIONS)}


def _solve(A: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [vr - f * vc for vr, vc in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]
