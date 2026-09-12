# TherapyTrace

**Longitudinal linguistic process monitoring for psychotherapy.**

Most computational mental-health work asks *"is this person depressed?"*
TherapyTrace asks a different question: *"is this person talking differently
than they used to?"*

It reads speaker-labelled session transcripts and scores five dimensions of
therapeutic process in the client's language, standardises each one against
that client's **own** earliest sessions, and reports a **Therapeutic Progress
Index (TPI)** per session plus a trajectory — gaining, plateauing, or losing
ground. It also flags which therapist moves were followed by a shift in the
client's very next turn.

Everyone else built a thermometer. This is a progress bar.

---

## Contents

- [What it does](#what-it-does)
- [How it differs from prior work](#how-it-differs-from-prior-work)
- [Running it](#running-it)
- [The method](#the-method)
- [Loading real corpora](#loading-real-corpora)
- [Validation](#validation)
- [API reference](#api-reference)
- [Project layout](#project-layout)
- [Limits and ethics](#limits-and-ethics)

---

## What it does

Five dimensions, each on 0–1, all extracted from client turns only:

| Dimension | The question it asks | Movement looks like |
|---|---|---|
| **Self-agency** | Does the client speak as someone who acts, or as someone acted upon? | "it just happens to me" → "I decided to" |
| **Future orientation** | Is attention on what comes next, or looping on what already happened? | replaying the argument → naming a plan for Tuesday |
| **Emotional granularity** | Is affect named precisely, or left as undifferentiated "bad"? | "I feel bad" → "resentful, and a bit ashamed of that" |
| **Problem ownership** | Is the difficulty inside the client's reach, or entirely outside it? | "he always" → "my part is that I go quiet then punish him for it" |
| **Reflection depth** | Is the client describing events, or working out why they happen? | a narrated week → a noticed pattern with a cause attached |

Auxiliary signals reported alongside but excluded from the composite:
hopelessness, rumination, absolutism, solution focus, self-reference balance.

**Ownership is deliberately not self-blame.** "It's all my fault, I ruin
everything" scores *down*, not up. Collapsing into fault is a symptom, not
insight. There is a test enforcing this.

---

## How it differs from prior work

| Existing depression/anxiety detection | TherapyTrace |
|---|---|
| "Is this person depressed? Yes/No" | "Is this person changing?" |
| Compares the client against **other people** | Compares them against **their own past self** |
| One snapshot in time | A curve across many sessions |
| Measures **symptoms** | Measures **therapy process** |
| Output is a label | Output is an explainable trajectory |
| Ignores the therapist | Scores the therapist's contribution too |

Three things combined make the framework novel: within-person change instead of
cross-sectional classification, process markers instead of symptom markers, and
session-level explanations a clinician can actually read.

---

## Running it

### Docker (one command)

```bash
docker compose up --build
# app  http://localhost:8080
# api  http://localhost:8000/docs
```

### Local development

```bash
./run.sh
# api  http://localhost:8000/docs
# app  http://localhost:5173
```

Or by hand:

```bash
# backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# frontend, in a second terminal
cd frontend
npm install
npm run dev
```

### First run

Open the app, click **Load synthetic cases**. That creates four cases with
programmed trajectories — improving, plateauing, relapsing, late breakthrough —
each with 10–12 sessions and matching PHQ-9 scores. It's the fastest way to see
what a trace does and to confirm the pipeline recovers a known signal.

### Tests

```bash
cd backend && python -m pytest tests -q
# 22 passed
```

---

## Machine learning (M10)

Two classifiers trained on **AnnoMI** — 133 expert-annotated MI dialogues,
6,725 labelled client utterances.

| Task | Accuracy | Macro-F1 |
|---|---|---|
| Client talk type (change / sustain / neutral) | 0.54 | 0.49 |
| Therapist behaviour (question / reflection / input / other) | 0.65 | 0.61 |

Splits are grouped by conversation, never by utterance — an utterance-level
split leaks speaker and topic between train and test and inflates every metric.

**Convergent validation:** the rule-based lexicon never saw the expert labels,
so AnnoMI independently tests it. All five dimensions order correctly
(change talk > sustain talk); the composite scores d = 0.389, p = 7.4e-18 at
utterance level, and correlates r = 0.344 (p < 0.001) with the expert
change-talk ratio across 93 conversations.

See `ml/README.md` for the full ablation and the honest negative results.

## Multimodal tier (M12–M14)

Three channels, all standardised **within person** — because vocal tract length
sets f0 and face geometry sets every landmark ratio, exactly as speaking style
sets word counts.

| Module | Channel | Features | Library |
|---|---|---|---|
| M12 | Acoustic | 48 per utterance: f0 mean/sd/range/CV, energy, pause count & ratio, longest pause, jitter, shimmer, HNR, spectral, 13 MFCCs | librosa |
| M13 | Visual | 15 per turn: expressivity range, AU proxies, head pose & motion, gaze aversion, blink rate | MediaPipe Face Mesh |
| M14 | Fusion | Late fusion, bounded modifiers | — |

### Why late fusion, and why text stays primary

We tested the multimodal hypothesis before building on it. Timestamp-derived
timing (speech rate, duration, pause approximations) on 2,343 real utterances:

| Model | AUC (grouped 5-fold) |
|---|---|
| Text process score only | **0.624** |
| Timing only, logistic regression | 0.539 |
| Timing only, gradient boosting | 0.545 |
| Timing, within-person z-scored, GB | 0.512 |
| Text + timing | 0.567 — **worse than text alone** |

Coarse timing carries essentially nothing, and a stronger model does not rescue
it. That result constrains the design: acoustic and visual channels enter as
**bounded modifiers**, able to move the index by at most ±4 points combined,
never as equal votes. When they disagree with the text tier, the disagreement is
surfaced rather than averaged away.

What that experiment does *not* show is that genuine acoustics are useless —
pitch contour, jitter and true pause structure live at millisecond resolution
and are invisible to a 1-second transcript timestamp. M12 is what makes testing
them possible once session audio is available.

### A deliberate omission

M13 emits **no valence score**. Stillness in therapy is as often deep processing
as disengagement, and a smile is as often avoidance as affect. Reading facial
expression as mood would be a category error in a *process* instrument, so the
module reports expressive range and movement only.

## Insights and recommendations (M16–M17)

The system does not stop at a score. `GET /api/clients/{id}/insights` returns a
plain-English session note and a next step.

**Session note (M16)** — template-driven, not model-generated, so every
sentence traces to a number, nothing can be fabricated, and the same inputs
always produce the same note:

> *"This session scored 49, close to this client's usual level. The clearest
> movement was in precision in naming feelings. Future orientation sat below
> their baseline: more replaying of what already happened. Across recent
> sessions the index has been falling, about 1.3 points per session. Worth
> raising in supervision."*

The rule the module obeys: **report the measurement, never the inference.** It
says "pitch variability ran 2 SD below this client's baseline", never "the
client sounded depressed."

**Recommendations (M17)** — three outputs:

| Output | Example |
|---|---|
| Dimension targeting | "Future orientation has the most headroom — you would be looking for more talk of what comes next." |
| Early warning | "[watch] The index has dropped 6 points over three sessions." |
| Next-session suggestion | "Client is middling. Consider open questions — across 123 annotated sessions they were followed by change talk 41% of the time in this range, against 26% for advice-giving." |

The suggestion prefers **this client's own response history** and falls back to
the corpus prior only when their own data is thin — and says which it used. If
no move in their history has a positive mean lift, it declines to recommend one
rather than suggesting the least-bad option.

## The method

### Within-person standardisation

Every dimension is z-scored against the client's own baseline, built from their
first two sessions:

```
z_d(s) = ( x_d(s) − μ_d ) / σ_d
TPI(s) = 50 + 10 · Σ_d ( w_d · z_d(s) )      clipped to [0, 100]
```

`σ_d` is **shrunk toward a cohort prior**, `λ = n / (n + k₀)` with `k₀ = 3`,
because two baseline sessions give a terrible variance estimate on their own
and an unshrunk σ makes early sessions swing wildly.

A TPI of 50 means *talking the way they always have*. 60 means one standard
deviation above their own norm. **Nobody is ever compared to anybody else** —
that is the entire point.

### Trajectory layer

- OLS trend with a t-test, over the full series and over a recent 6-session window
- Momentum classified on the **recent** window, corroborated by the full trend
- Change-point detection by binary segmentation, maximising a two-sample t statistic
- Baseline corridor (±1 SD) rendered as a band, so "progress" is visually defined as departure from one's own norm

> **A trap worth knowing about.** An earlier version classified momentum on the
> whole-series slope and called the *relapsing* case "steady" — it rose then
> fell, so the overall slope was still positive. Recent-window classification
> fixes it. Any longitudinal index needs to handle rise-then-fall shapes.

### Therapist contribution

For each therapist turn with a client turn on both sides:

```
lift = process_mean(client turn after) − process_mean(client turn before)
```

Within the same session, against the client's own local level, so between-client
differences cancel. Aggregated per intervention type (complex reflection, open
question, affirmation, challenge, directive, summary, …) with means, SDs and
normal CIs.

**This is association, not causation.** Therapists choose interventions in
response to what was just said, so the figures are confounded by indication by
construction. The UI says so on the page.

### Explainability

Every score traces back to text. The session view shows a turn-by-turn ribbon,
the turns that lifted and depressed each session, and *evidence chips* naming
the exact phrases that fired. A clinician can disagree with the score and see
precisely why it landed where it did.

---

## Loading real corpora

`scripts/ingest.py` handles the shapes the target corpora actually ship in.

```bash
# one folder per case, files named 01.txt, 02.txt ...
python scripts/ingest.py --format txt --root ./corpus/by_client

# flat folder, filenames like CL014_s03.txt
python scripts/ingest.py --format txt --root ./corpus/flat \
    --pattern '(?P<client>[A-Z]{2}\d+)_s(?P<session>\d+)'

# utterance-level CSV (AnnoMI style)
python scripts/ingest.py --format csv --file annomi.csv \
    --client-col transcript_id --speaker-col interlocutor \
    --text-col utterance_text
```

The transcript parser accepts `THERAPIST:` / `CLIENT:`, `T:` / `C:`,
`Counselor:` / `Patient:`, bracketed labels, and diarised `SPEAKER_00` output.
When labels carry no role information it infers roles from question rate and
turn length — and **says so in the response**, so you can correct it before
trusting any therapist-impact result.

### Datasets this was built for

| Dataset | Use |
|---|---|
| [IEEE DataPort — Psychotherapeutic Sessions (separate therapist/client remarks)](https://ieee-dataport.org/documents/psychotherapeutic-sessions-dataset-separate-remarks-therapist-and-their-clients) | Primary. Speaker separation is exactly what both modules need. |
| [IEEE DataPort — VoiceDiaryMood](https://ieee-dataport.org/documents/voicediarymood-psychiatrist-annotated-chinese-voice-diary-transcripts-suicide-risk-and-0) | 450 transcripts from 37 patients over 4 years — genuine within-person repeated measures. Restricted access; apply early. |
| [IEEE DataPort — Kangning Clinical Interview](https://ieee-dataport.org/documents/kangning-dataset-clinical-interview-depression) | Clinical-validity anchor (MADRS). |
| [IEEE DataPort — Text-based Depression Detection (Chinese)](https://ieee-dataport.org/documents/text-based-depression-detection-dataset-chinese) | Cross-lingual generalisation (PHQ-9 labelled). |
| [IEEE DataPort — Chinese Multimodal Depression Corpus](https://ieee-dataport.org/open-access/chinese-multimodal-depression-corpus) | Multimodal extension. EULA by email. |
| [IEEE DataPort — AI-Driven Psychological Therapy Data](https://ieee-dataport.org/documents/ai-driven-psychological-therapy-data) | Prototyping while restricted access is pending. |
| AnnoMI | Free and public. Client turns labelled **change talk vs sustain talk** — effectively hand-annotated ground truth for therapeutic momentum. |
| Alexander Street Counseling & Psychotherapy Transcripts | ~950 sessions; institutional library subscription. |
| CUEMPATHY | 39 dyads × 4 sessions — the cleanest small longitudinal structure available. |

**The honest gap:** almost no public corpus contains many sessions from the same
client, which is exactly what a within-person index needs. That gap is the
project's motivation, not a flaw in it — say so explicitly in any write-up.
The synthetic generator exists so the method can be validated against a known
ground truth while restricted-access applications are pending.

---

## Validation

Record PHQ-9 / GAD-7 / MADRS scores against sessions and the case page reports:

- **Within-person Pearson r and Spearman ρ** between TPI and the measure. The expected sign is *negative* — language up, symptoms down.
- **Lead-lag test** on first differences: does a TPI change at session *s* predict the symptom change at *s+1* better than the reverse? If language moves first, the transcript is carrying information the questionnaire hasn't caught up with.
- **Weight re-fitting** (`GET /api/calibration/weights`): least squares of z-scored dimensions against symptom change, so the composite weights stop being the author's guesses.

Results on the four synthetic cases:

| Case | Programmed | Detected | r with PHQ-9 |
|---|---|---|---|
| CL-014 | improving | gaining ✅ | −0.76 |
| CL-027 | plateauing | steady ✅ | −0.81 |
| CL-031 | relapsing | regressing ✅ | −0.70 |
| CL-046 | late breakthrough | gaining ✅ | −0.74 |

Change point in CL-014 detected at session 4, *t* = 4.99, *p* ≈ 0.001.

An index nobody has checked against an outcome is decoration. Do this part.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness |
| `GET` | `/api/meta` | Dimension keys, labels and definitions |
| `POST` | `/api/demo/seed` | Load the four synthetic cases |
| `GET` `POST` | `/api/clients` | List / create cases |
| `DELETE` | `/api/clients/{id}` | Delete a case and everything under it |
| `POST` | `/api/clients/{id}/sessions` | Add and score a session |
| `POST` | `/api/clients/{id}/sessions/upload` | Same, from a file upload |
| `GET` | `/api/clients/{id}/trajectory` | TPI series, dimensions, momentum, change point, validation |
| `GET` | `/api/clients/{id}/therapist-impact` | Pooled intervention lift + best exchanges |
| `POST` | `/api/clients/{id}/measures` | Record a self-report score |
| `GET` | `/api/sessions/{sid}` | Full session analysis with drivers and evidence |
| `DELETE` | `/api/sessions/{sid}` | Delete a session and rescore the case |
| `GET` | `/api/calibration/weights` | Re-fit dimension weights across all cases |
| `POST` | `/api/analyze` | Score a transcript without saving anything |

Interactive docs at `/docs`.

---

## Project layout

```
therapytrace/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app, demo seeding, SPA serving
│   │   ├── db.py models.py schemas.py services.py
│   │   ├── api/routes.py      every endpoint
│   │   ├── nlp/
│   │   │   ├── lexicons.py    the five constructs + MI intervention taxonomy
│   │   │   ├── features.py    utterance and session scoring
│   │   │   ├── transcript.py  multi-format parser with role inference
│   │   │   ├── scoring.py     within-person z-scores and the TPI
│   │   │   ├── trajectory.py  trend, momentum, change points
│   │   │   ├── therapist.py   next-turn lift per intervention
│   │   │   ├── calibration.py correlation, lead-lag, weight re-fitting
│   │   │   └── pipeline.py    end-to-end orchestration
│   │   └── seed/generator.py  synthetic cases with programmed trajectories
│   └── tests/                 22 tests
├── frontend/
│   └── src/
│       ├── components/Trace.jsx        the signature chart
│       ├── components/DimensionGrid.jsx
│       └── pages/                     Cases · CaseTrace · SessionView ·
│                                      TherapistView · Bench · Method
├── scripts/ingest.py          bulk corpus loader
├── docker-compose.yml
└── run.sh
```

### Design notes

The interface is built as a **clinical instrument readout**: chart-paper ground,
IBM Plex Condensed / Serif / Mono, no rounded corners, nothing floating. Colour
is diagnostic and means exactly one thing each — teal gaining, plum losing
ground, slate holding, amber change point. The trace itself is hand-rolled SVG
rather than a charting library, because the corridor-and-departure shape *is*
the thesis of the project and no chart library draws it.

---

## Limits and ethics

- **It cannot diagnose.** Nothing in the pipeline maps to a disorder and it must never be read as if it did.
- **Therapist impact is correlational**, confounded by indication. It is a supervision prompt, never a ranking of clinicians or techniques.
- **Short sessions are noisy.** Confidence drops below 50% under ~300 client words. Believe it.
- **The lexicon is English and culturally situated.** Directness, emotion vocabulary and how ownership gets expressed all vary. Cross-cultural transfer is an open question, not a solved one.
- **A falling trace is a prompt for a conversation.** It is not evidence that a therapist is failing or that a client is.
- Cases are stored under pseudonymous codes. Never enter names, dates of birth, addresses or anything else identifying. Real clinical material needs ethics approval, informed consent covering secondary analysis, and encryption at rest.

The research question, stated plainly: **can therapeutic progress be quantified
from language change across psychotherapy sessions?** This is the instrument for
finding out.
