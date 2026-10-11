#!/usr/bin/env python3
"""
Read the rendered PDF and work out what page number each heading prints on.

Physical page index is not the printed number: the cover and title carry none,
the front matter is lower-case roman, and the body restarts at arabic 1. The
offsets are derived from where each section actually begins rather than
assumed, so that inserting a page anywhere upstream cannot silently shift the
whole contents table.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PDF = HERE / "TherapyTrace_Project_Report.pdf"

ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
         "xi", "xii", "xiii", "xiv", "xv"]

# key -> regex identifying the page the item's own heading sits on.
#
# Every pattern is anchored to a whole line. Without the anchors these match
# the contents page too, because the contents lists each of these strings; the
# first match then lands on the contents page and every number in the table
# comes out shifted. That failure is silent and plausible-looking, which is
# what makes it worth guarding against explicitly.
WANTED = {
    "ABSTRACT": r"^\s*ABSTRACT\s*$",
    "ACKNOWLEDGEMENT": r"^\s*ACKNOWLEDGEMENT\s*$",
    "LIST OF FIGURES": r"^\s*LIST OF FIGURES\s*$",
    "LIST OF TABLES": r"^\s*LIST OF TABLES\s*$",
    "LIST OF ACRONYMS": r"^\s*LIST OF ACRONYMS\s*$",
    "REFERENCES": r"^\s*REFERENCES\s*$",
    "APPENDICES": r"^\s*APPENDICES\s*$",
}
for c in range(1, 8):
    WANTED[f"CHAPTER {c}"] = rf"^\s*CHAPTER {c}\s*$"


def page_texts():
    out = []
    n = int(re.search(r"Pages:\s+(\d+)",
            subprocess.run(["pdfinfo", str(PDF)], capture_output=True,
                           text=True).stdout).group(1))
    for i in range(1, n + 1):
        t = subprocess.run(
            ["pdftotext", "-f", str(i), "-l", str(i), "-layout", str(PDF), "-"],
            capture_output=True, text=True).stdout
        out.append(t)
    return out


def main():
    pages = page_texts()
    n = len(pages)

    # Where does the arabic-numbered body begin? The first page carrying
    # "CHAPTER 1". Everything before it is cover (2 pages) then roman.
    body_start = next(i for i, t in enumerate(pages)
                      if re.search(r"^\s*CHAPTER 1\s*$", t, re.M))
    front_start = 2  # two unnumbered pages: cover, title

    def printed(idx):
        if idx < front_start:
            return ""
        if idx < body_start:
            r = idx - front_start
            return ROMAN[r] if r < len(ROMAN) else str(r + 1)
        return str(idx - body_start + 1)

    found = {}
    for key, rx in WANTED.items():
        for i, t in enumerate(pages):
            if re.search(rx, t, re.M):
                found[key] = printed(i)
                break

    # Numbered sections, figures and tables — scanned only from the body
    # onward, since the contents pages repeat every one of these strings.
    for i in range(body_start, n):
        t = pages[i]
        for m in re.finditer(r"^\s*(\d\.\d)\s+[A-Z]", t, re.M):
            found.setdefault(m.group(1), printed(i))
        for m in re.finditer(r"Figure\s+(\d\.\d)\s*:", t):
            found.setdefault("FIG" + m.group(1), printed(i))
        for m in re.finditer(r"Table\s+(A?\d+\.\d)\s*:", t):
            found.setdefault("TAB" + m.group(1), printed(i))

    (HERE / "pagemap.json").write_text(json.dumps(found, indent=1, sort_keys=True))
    print(f"{n} physical pages; body starts at index {body_start} "
          f"(printed 1); {len(found)} entries mapped")
    missing = [k for k in WANTED if k not in found]
    if missing:
        print("NOT FOUND:", missing)
    print("last printed page:", printed(n - 1))


if __name__ == "__main__":
    main()
