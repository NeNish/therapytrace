# Panel and publication material

Sources for the three documents produced from this codebase. The generated
artefacts (`.pptx`, `.docx`, `.pdf`) are deliberately not committed — they are
build outputs, and the generators here reproduce them.

Every figure and every number in all three comes from `ml/benchmark.json`,
`ml/reliability.json` and the CSV exports in `ml/results/`. Nothing is
hand-entered, so re-running the benchmark and re-running these generators keeps
the documents in step with the code.

| Directory | What it builds | How |
|---|---|---|
| `paper/` | IEEE conference manuscript, 6 pages | Upload to Overleaf, compile with pdfLaTeX. See `paper/README.md` |
| `deck/` | Final review deck, 20 slides | `node deck/build.js` (needs `pptxgenjs`) |
| `report/` | University project report, 61 pages | `python report/build.py` (needs `python-docx`) |

## Before the report is submitted

`report/front.py` carries four placeholders that only the author can fill:

- `<<REGISTER NUMBER>>` — appears four times
- `<<GUIDE NAME>>` — appears twice
- `<<DESIGNATION>>`
- `<<HOD NAME>>`

## Before the paper is submitted anywhere

Four references in `paper/refs.bib` — `cohen2022`, `hoang2024`, `kumar2024`,
`m3tcm2024` — were located through secondary citations rather than read in
full. Verify author lists, venues and page numbers against the originals.
A wrong citation costs credibility cheaply, and reviewers do check.
