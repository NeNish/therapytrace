# Data and datasets

This repository contains **no clinical data**. The four demo cases in
`backend/app/seed/generator.py` are synthetic, generated from templates with
programmed trajectories, and exist so the pipeline can be validated against a
known ground truth.

The corpora this project is designed to work with are distributed by their own
holders under their own terms. None of them are redistributed here.

| Dataset | Access |
|---|---|
| IEEE DataPort — Psychotherapeutic Sessions | IEEE DataPort subscription |
| IEEE DataPort — VoiceDiaryMood | Restricted; application required |
| IEEE DataPort — Kangning Clinical Interview | IEEE DataPort |
| IEEE DataPort — Text-based Depression Detection (Chinese) | IEEE DataPort |
| IEEE DataPort — Chinese Multimodal Depression Corpus | EULA by email |
| AnnoMI | Open, see the AnnoMI repository |
| Alexander Street Counseling & Psychotherapy Transcripts | Institutional library subscription |
| CUEMPATHY | On request from the authors |

If you load real transcripts, do not commit them. `.gitignore` excludes `*.db`,
but a `corpus/` directory is your responsibility — add it to `.gitignore`
before your first commit.
