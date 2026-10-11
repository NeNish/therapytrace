# TherapyTrace — IEEE conference paper

## Uploading to Overleaf

1. Go to Overleaf → **New Project** → **Upload Project**
2. Upload `TherapyTrace_IEEE_Overleaf.zip` (this whole archive)
3. Overleaf sets `main.tex` as the root document automatically; if it does
   not, open the Menu and set **Main document** to `main.tex`
4. Menu → **Compiler** → **pdfLaTeX** (not XeLaTeX or LuaLaTeX)
5. Press **Recompile**

Nothing else needs installing. `IEEEtran.cls` and `IEEEtran.bst` are part of
Overleaf's TeX Live, so they resolve automatically.

**Do not add an `IEEEtran.cls` file to the project.** LaTeX searches the
project directory before the system tree, so a local copy would override
Overleaf's real class and silently change the layout.

## Files

| File | Purpose |
|---|---|
| `main.tex` | The manuscript |
| `refs.bib` | 18 references, BibTeX |
| `figures/TherapyTrace_Architecture.png` | Fig. 1, system architecture |
| `figures/bench_reliability.png` | Fig. 2, error propagation and correction |
| `figures/bench_cost.png` | Spare — accuracy against fit time, not currently placed |

## Before submitting anywhere

- Add co-authors and your supervisor to the `\author` block if the submission
  is joint.
- Check the target venue's page limit. The paper is currently 6 pages
  including references.
- Some venues require anonymisation for review. If so, replace the author
  block with `\author{\IEEEauthorblockN{Anonymous Submission}}` and remove
  identifying statements from the Acknowledgements (there are none at
  present).
- Four references (`hoang2024`, `kumar2024`, `m3tcm2024`, `cohen2022`) were
  located through secondary citations rather than read in full. Verify the
  author lists, venues and page numbers against the originals before
  submission — reviewers check these, and a wrong citation is the kind of
  error that costs credibility cheaply.

## Compilation note

This manuscript was validated locally against a stand-in class, because CTAN
was unreachable from the machine it was written on. The LaTeX itself is clean:
no errors, no undefined references or citations, no overfull boxes. The exact
IEEE typography is what Overleaf will apply when you compile there.
