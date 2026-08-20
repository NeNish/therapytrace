#!/usr/bin/env python3
"""
Bulk ingest for TherapyTrace.

Loads a directory of transcripts into the database, one case per client folder
or one case per filename pattern. Designed for the corpora this project
targets, none of which ship in the same shape:

  IEEE DataPort psychotherapeutic sessions
      A flat CSV/TSV of separate therapist and client remarks. Use --format csv
      with --speaker-col and --text-col.

  AnnoMI
      A CSV with transcript_id, utterance_text, interlocutor. One conversation
      per transcript_id, each treated as a separate single-session case unless
      --group-by maps several to one client.

  VoiceDiaryMood / CUEMPATHY / Alexander Street exports
      Plain text files, one per session, named so that the client ID and the
      session order can be read off the filename. Use --format txt with
      --pattern.

Nothing here uploads anywhere. It talks to a local API over HTTP.

Examples
--------
  # one folder per case, files named 01.txt, 02.txt ...
  python ingest.py --format txt --root ./corpus/by_client

  # flat folder, filenames like CL014_s03.txt
  python ingest.py --format txt --root ./corpus/flat \\
      --pattern '(?P<client>[A-Z]{2}\\d+)_s(?P<session>\\d+)'

  # AnnoMI style csv
  python ingest.py --format csv --file annomi.csv \\
      --client-col transcript_id --speaker-col interlocutor \\
      --text-col utterance_text
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

DEFAULT_API = "http://localhost:8000/api"

THERAPIST_ALIASES = {
    "therapist", "counselor", "counsellor", "clinician", "interviewer",
    "t", "th", "doctor", "dr",
}


# --------------------------------------------------------------------------
# tiny HTTP client (no dependencies, so this runs anywhere Python does)
# --------------------------------------------------------------------------

def _call(api: str, path: str, payload: dict | None = None, method: str = "GET"):
    url = f"{api}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300]
        raise SystemExit(f"API error {e.code} on {method} {path}: {detail}")
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Could not reach the API at {api}. Is the backend running?\n  {e.reason}"
        )


def ensure_client(api: str, code: str, meta: dict) -> int:
    existing = _call(api, "/clients")
    for c in existing:
        if c["code"] == code:
            return c["id"]
    created = _call(api, "/clients", {"code": code, **meta}, method="POST")
    return created["id"]


def push_session(api: str, client_id: int, number: int, transcript: str, source: str):
    return _call(
        api,
        f"/clients/{client_id}/sessions",
        {"session_number": number, "transcript": transcript, "source": source},
        method="POST",
    )


# --------------------------------------------------------------------------
# format: plain text files
# --------------------------------------------------------------------------

def ingest_txt(args) -> None:
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory.")

    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)

    if args.pattern:
        rx = re.compile(args.pattern)
        for path in sorted(root.rglob("*.txt")):
            m = rx.search(path.name)
            if not m:
                print(f"  skipped (no match): {path.name}")
                continue
            client = m.groupdict().get("client") or path.parent.name
            session = int(m.groupdict().get("session") or 0)
            groups[client].append((session, path))
    else:
        # one sub-directory per case, files ordered by name
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            for i, path in enumerate(sorted(sub.glob("*.txt")), start=1):
                groups[sub.name][:] = groups[sub.name] + [(i, path)]

    if not groups:
        raise SystemExit("Nothing matched. Check --root and --pattern.")

    for code, items in groups.items():
        items.sort(key=lambda kv: kv[0])
        cid = ensure_client(
            args.api, code,
            {"presenting_issue": args.issue, "modality": args.modality,
             "therapist_code": args.therapist},
        )
        print(f"{code}: {len(items)} session(s)")
        for order, (num, path) in enumerate(items, start=1):
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text.strip()) < 20:
                print(f"  {path.name}: too short, skipped")
                continue
            res = push_session(
                args.api, cid, num or order, text, f"corpus:{args.corpus or root.name}"
            )
            print(f"  session {res['session_number']:>3}  TPI {res['tpi']:.1f}"
                  f"  conf {res['confidence']:.2f}  ({path.name})")


# --------------------------------------------------------------------------
# format: utterance-level CSV
# --------------------------------------------------------------------------

def ingest_csv(args) -> None:
    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"{path} does not exist.")

    rows: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        dialect = csv.Sniffer().sniff(sample) if args.delimiter is None else None
        reader = (
            csv.DictReader(fh, delimiter=args.delimiter)
            if args.delimiter
            else csv.DictReader(fh, dialect=dialect)
        )
        missing = [
            c for c in (args.client_col, args.speaker_col, args.text_col)
            if c not in (reader.fieldnames or [])
        ]
        if missing:
            raise SystemExit(
                f"Columns not found: {', '.join(missing)}.\n"
                f"Available: {', '.join(reader.fieldnames or [])}"
            )
        for row in reader:
            key = str(row[args.client_col]).strip()
            speaker = str(row[args.speaker_col]).strip().lower()
            text = (row[args.text_col] or "").strip()
            if not text:
                continue
            role = "THERAPIST" if speaker in THERAPIST_ALIASES else "CLIENT"
            rows[key].append((role, text))

    # optional grouping: several conversation ids belong to one client
    grouping: dict[str, list[str]] = defaultdict(list)
    if args.group_by:
        gx = re.compile(args.group_by)
        for key in rows:
            m = gx.search(key)
            grouping[m.group(1) if m and m.groups() else key].append(key)
    else:
        for key in rows:
            grouping[key].append(key)

    for code, keys in grouping.items():
        cid = ensure_client(
            args.api, str(code)[:32],
            {"presenting_issue": args.issue, "modality": args.modality,
             "therapist_code": args.therapist},
        )
        print(f"{code}: {len(keys)} conversation(s)")
        for n, key in enumerate(sorted(keys), start=1):
            transcript = "\n".join(f"{role}: {text}" for role, text in rows[key])
            if len(transcript) < 20:
                continue
            res = push_session(
                args.api, cid, n, transcript, f"corpus:{args.corpus or path.stem}"
            )
            print(f"  session {res['session_number']:>3}  TPI {res['tpi']:.1f}"
                  f"  conf {res['confidence']:.2f}  ({key})")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=DEFAULT_API, help="Base API URL")
    ap.add_argument("--format", choices=["txt", "csv"], required=True)
    ap.add_argument("--root", help="Directory of .txt transcripts (format=txt)")
    ap.add_argument("--file", help="CSV file of utterances (format=csv)")
    ap.add_argument("--pattern", help="Regex with named groups 'client' and 'session'")
    ap.add_argument("--client-col", default="transcript_id")
    ap.add_argument("--speaker-col", default="interlocutor")
    ap.add_argument("--text-col", default="utterance_text")
    ap.add_argument("--delimiter", default=None, help="Force a CSV delimiter, e.g. '\\t'")
    ap.add_argument("--group-by", help="Regex whose first group maps ids to one case")
    ap.add_argument("--corpus", help="Label recorded as the session source")
    ap.add_argument("--issue", default=None)
    ap.add_argument("--modality", default=None)
    ap.add_argument("--therapist", default=None)
    args = ap.parse_args()

    _call(args.api, "/health")  # fail fast if the backend is down

    if args.format == "txt":
        if not args.root:
            ap.error("--root is required for --format txt")
        ingest_txt(args)
    else:
        if not args.file:
            ap.error("--file is required for --format csv")
        ingest_csv(args)

    print("\nDone. Open the case list to see the traces.")


if __name__ == "__main__":
    sys.exit(main())
