"""
Clinician-facing helpers built on top of the trajectory.

  * decline_alert      -- a gentle prompt when the index has fallen several
                          sessions in a row.
  * evaluate_goal      -- did the dimension a therapist chose to work on move in
                          the session they chose?
  * prep_card          -- a one-page brief for the next session.

None of these diagnoses or predicts. They re-present measurements the system
already holds, in the form a clinician would use before or after a session,
and each one says what it is based on.
"""

from __future__ import annotations

from typing import Any

from .features import DIMENSIONS
from .narrative import DIM_CONCRETE, DIM_WORDS
from .recommend import READABLE

# A session-to-session change smaller than this (in the client's own SD units)
# is treated as noise rather than movement.
GOAL_MOVE_SD = 0.3


# --------------------------------------------------------------------------
# early-warning banner
# --------------------------------------------------------------------------

def decline_alert(tpi_series: list[float], run: int = 3, min_step: float = 0.5) -> dict:
    """
    Fire when the TPI has fallen in each of the last `run` sessions.

    `min_step` ignores drops smaller than half a point, so rounding noise on a
    flat series cannot trigger it. The wording is a prompt for supervision,
    never a statement that the client is deteriorating.
    """
    needed = run + 1
    if len(tpi_series) < needed:
        return {
            "active": False,
            "run_length": 0,
            "reason": f"needs at least {needed} scored sessions",
        }

    streak = 0
    for prev, cur in zip(reversed(tpi_series[:-1]), reversed(tpi_series[1:])):
        if cur < prev - min_step:
            streak += 1
        else:
            break

    active = streak >= run
    start = len(tpi_series) - streak  # 1-based session index where the fall began
    total_drop = tpi_series[start - 1] - tpi_series[-1] if streak else 0.0

    return {
        "active": active,
        "run_length": streak,
        "total_drop": round(total_drop, 1),
        "from_session": start if streak else None,
        "to_session": len(tpi_series) if streak else None,
        "message": (
            f"The index has fallen in each of the last {streak} sessions "
            f"(down {total_drop:.0f} points since session {start}). "
            "Consider discussing this client in supervision."
            if active else None
        ),
        "caveat": (
            "A pattern in the measurements, not a judgement about the client "
            "or the therapy. Short sessions and topic changes can move the "
            "index too."
        ),
    }


# --------------------------------------------------------------------------
# session goals
# --------------------------------------------------------------------------

def evaluate_goal(
    dimension: str,
    session_number: int,
    sessions: list[dict],
) -> dict:
    """
    Compare the goal dimension in the target session with the session before.

    Uses the within-person z-scores already stored for each session, so the
    comparison is on the client's own scale. Returns status "pending" until
    the target session has been scored.
    """
    if dimension not in DIMENSIONS:
        return {"status": "invalid", "detail": f"unknown dimension {dimension}"}

    by_num = {s["session_number"]: s for s in sessions}
    target = by_num.get(session_number)
    if target is None:
        return {"status": "pending", "detail": "session not scored yet"}

    earlier = [n for n in by_num if n < session_number]
    if not earlier:
        return {"status": "no_comparison",
                "detail": "this is the first session, so there is nothing to compare with"}
    prev = by_num[max(earlier)]

    z_now = (target.get("z") or target.get("z_scores") or {}).get(dimension)
    z_prev = (prev.get("z") or prev.get("z_scores") or {}).get(dimension)
    if z_now is None or z_prev is None:
        return {"status": "pending", "detail": "scores not available yet"}

    delta = z_now - z_prev
    if delta >= GOAL_MOVE_SD:
        status = "moved_toward"
    elif delta <= -GOAL_MOVE_SD:
        status = "moved_away"
    else:
        status = "held"

    up, down = DIM_CONCRETE[dimension]
    words = {
        "moved_toward": f"{DIM_WORDS[dimension].capitalize()} rose by {abs(delta):.1f} SD "
                        f"against session {max(earlier)}: {up}.",
        "moved_away": f"{DIM_WORDS[dimension].capitalize()} fell by {abs(delta):.1f} SD "
                      f"against session {max(earlier)}: {down}.",
        "held": f"{DIM_WORDS[dimension].capitalize()} stayed within "
                f"±{GOAL_MOVE_SD} SD of the previous session ({delta:+.1f} SD).",
    }[status]

    return {
        "status": status,
        "z_before": round(z_prev, 3),
        "z_after": round(z_now, 3),
        "delta_sd": round(delta, 3),
        "compared_with_session": max(earlier),
        "detail": words,
    }


# --------------------------------------------------------------------------
# next-session prep card
# --------------------------------------------------------------------------

def _phrases_for(dimension: str, drivers: dict, limit: int = 2) -> list[dict]:
    """
    Client turns from the latest session that scored lowest on the dimension.
    These are the moments worth returning to.
    """
    pool = (drivers or {}).get("weakest", []) + (drivers or {}).get("strongest", [])
    seen, rows = set(), []
    for d in sorted(pool, key=lambda r: (r.get("scores") or {}).get(dimension, 1.0)):
        text = (d.get("text") or "").strip()
        if not text or any(text in s or s in text for s in seen):
            continue
        seen.add(text)
        short = text if len(text) <= 220 else text[:217].rsplit(" ", 1)[0] + "…"
        rows.append({
            "turn": d.get("turn"),
            "text": short,
            "score": round((d.get("scores") or {}).get(dimension, 0.0), 3),
        })
        if len(rows) >= limit:
            break
    return rows


def prep_card(
    client_code: str,
    sessions: list[dict],
    tpi_series: list[float],
    latest_drivers: dict | None,
    therapist_moves: list[dict],
    recommendation: dict | None,
    goals: list[dict] | None = None,
) -> dict:
    """Everything a clinician might want on one page before the next session."""
    if not sessions:
        return {"available": False, "reason": "no scored sessions yet"}

    last = sessions[-1]
    z = last.get("z") or last.get("z_scores") or {}
    ranked = sorted(((d, z[d]) for d in DIMENSIONS if d in z), key=lambda kv: kv[1])
    low_dim, low_z = ranked[0] if ranked else (None, None)
    high_dim, high_z = ranked[-1] if ranked else (None, None)

    moves = [m for m in therapist_moves if m.get("n", 0) >= 2]
    best_moves = [
        {
            "intervention": m["intervention"],
            "label": READABLE.get(m["intervention"], m.get("label", m["intervention"])),
            "mean_lift": m["mean_lift"],
            "n": m["n"],
        }
        for m in sorted(moves, key=lambda r: r["mean_lift"], reverse=True)[:2]
        if m["mean_lift"] > 0
    ]

    rec = recommendation or {}
    warning = rec.get("early_warning") or {}
    nxt = rec.get("next_session") or {}

    open_goals = [g for g in (goals or []) if g.get("evaluation", {}).get("status") == "pending"]
    done_goals = [g for g in (goals or []) if g.get("evaluation", {}).get("status") not in (None, "pending")]

    up, _down = DIM_CONCRETE[low_dim] if low_dim else ("", "")

    return {
        "available": True,
        "client_code": client_code,
        "based_on_session": last["session_number"],
        "next_session": last["session_number"] + 1,
        "latest_tpi": round(tpi_series[-1], 1) if tpi_series else None,
        "focus": {
            "dimension": low_dim,
            "label": DIM_WORDS.get(low_dim, low_dim),
            "z": round(low_z, 2) if low_z is not None else None,
            "look_for": up,
        } if low_dim else None,
        "strength": {
            "dimension": high_dim,
            "label": DIM_WORDS.get(high_dim, high_dim),
            "z": round(high_z, 2) if high_z is not None else None,
        } if high_dim else None,
        "revisit": _phrases_for(low_dim, latest_drivers or {}) if low_dim else [],
        "what_has_worked": best_moves,
        "suggestion": nxt.get("suggestion"),
        "suggestion_source": nxt.get("evidence_source"),
        "watch": warning.get("message") if warning.get("level") in ("watch", "elevated") else None,
        "decline_alert": decline_alert(tpi_series),
        "open_goals": open_goals,
        "recent_goal_results": done_goals[-3:],
        "caveat": (
            "Prepared from measurements of this client's own language. It "
            "suggests where to look, not what to do; associations with "
            "therapist moves are confounded by indication."
        ),
    }
