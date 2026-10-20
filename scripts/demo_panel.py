"""
Create CL-DEMO: a synthetic case whose index falls in each of its last three
sessions, with two session goals set, so the downward-run banner, the goal
tracker and the prep card can all be shown in a live demonstration.

Usage (backend running on port 8000):
    python scripts/demo_panel.py [http://localhost:8000]

Synthetic data for demonstration only. It is not part of the validation
reported in the paper.
"""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/") + "/api"

MOVING = (
    "I decided not to go and I told her why. I felt resentful and also a bit "
    "ashamed of being resentful, both at once. I noticed I shut down because I "
    "was scared, and I can see how I escalate it. Next week I'm going to ask "
    "for ten minutes before we talk."
)
MIDDLE = (
    "I sort of told her, I guess. I felt bad about it. Things are a bit better "
    "maybe, I don't know what happens next."
)
STUCK = (
    "It's all my fault and I ruin everything. He always does this and there's "
    "nothing I can do. I had to go, I have no choice. I just feel bad. What's "
    "the point, nothing works and nothing ever changes."
)

# Each session alternates two kinds of client turn; the mix sets its level.
SESSIONS = [(STUCK, STUCK), (STUCK, MIDDLE), (MOVING, MIDDLE), (MOVING, MOVING),
            (MOVING, MIDDLE), (MIDDLE, STUCK), (STUCK, STUCK)]


def call(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def main() -> None:
    for c in call("GET", "/clients"):
        if c["code"] == "CL-DEMO":
            call("DELETE", f"/clients/{c['id']}")

    cid = call("POST", "/clients", {
        "code": "CL-DEMO", "presenting_issue": "Demonstration: downward run",
        "modality": "CBT", "therapist_code": "TH-01",
    })["id"]

    for i, (a, b) in enumerate(SESSIONS, start=1):
        transcript = "\n".join(
            f"THERAPIST: What was that like for you?\nCLIENT: {a if k % 2 else b}"
            for k in range(8)
        )
        call("POST", f"/clients/{cid}/sessions",
             {"session_number": i, "transcript": transcript})

    call("POST", f"/clients/{cid}/goals",
         {"session_number": 6, "dimension": "self_agency", "note": "decisions at work"})
    call("POST", f"/clients/{cid}/goals",
         {"session_number": 8, "dimension": "future_orientation", "note": "plans for the week"})

    traj = call("GET", f"/clients/{cid}/trajectory")
    print("CL-DEMO created. TPI:", [round(x, 1) for x in traj["tpi_series"]])
    print("Banner active:", traj["decline_alert"]["active"])
    print(f"Open the case at /case/{cid} and the prep card at /case/{cid}/prep")


if __name__ == "__main__":
    main()
