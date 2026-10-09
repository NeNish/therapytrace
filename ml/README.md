# ML component (M10, M21)

Supervised classifiers trained on **AnnoMI** (Wu et al., 2023) — 133
expert-annotated motivational-interviewing dialogues.

> **What AnnoMI is.** Performed demonstration videos used for MI training, not
> recordings of clinical sessions, and the 23 low-quality transcripts are
> deliberately staged poor counselling. Every number here is a claim about this
> corpus and nothing more.

## Reproducing

```bash
curl -L -o ml/data/AnnoMI-full.csv \
  https://raw.githubusercontent.com/uccollab/AnnoMI/main/AnnoMI-full.csv

pip install scikit-learn pandas scipy joblib matplotlib xgboost lightgbm

# M10 — does the lexicon add signal? (features vary, classifier fixed)
python ml/train.py                     # trains, cross-validates, ablates
python ml/finalize.py                  # fits final models, saves figures
python ml/validate_against_annomi.py   # validates the TPI against experts

# M21 — does the classifier choice matter? (features fixed, learner varies)
python ml/benchmark.py                 # 7 learners × 2 tasks, grouped CV   (~40 min)
python ml/time_models.py               # clean fit-time measurement, idle CPU
python ml/figure_benchmark.py          # six figures into ml/artifacts/
python ml/report_benchmark.py > ml/RESULTS.md
```

The dataset is **not** committed — download it from the link above.

## Results (held-out conversations)

Selected model per task, from the M21 benchmark:

| Task | Classes | Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|
| Client talk type | change / sustain / neutral | Naive Bayes | 0.525 | 0.457 | 0.674 |
| Therapist behaviour | question / reflection / therapist_input / other | Logistic Regression | 0.664 | 0.608 | 0.850 |

Full tables for all seven learners — accuracy, macro-F1, weighted-F1, macro
precision and recall, ROC-AUC, per-class F1, confusion matrices and paired
significance tests — are in **[`RESULTS.md`](RESULTS.md)**, generated directly
from `artifacts/benchmark.json` so the prose cannot drift from the data.

## M21 — seven classifiers, one identical feature matrix

Every learner sees the same TF-IDF word 1–2 grams + character 3–5 grams + 12
lexicon features, and every learner is class-balanced by whatever mechanism it
offers, so a difference in score is attributable to the learner and not to
feature engineering or to one model quietly optimising raw accuracy.

| Model | Client macro-F1 (CV) | Therapist macro-F1 (CV) | Fit (s) |
|---|---|---|---|
| Majority class | 0.228 | 0.145 | 0.00 |
| Naive Bayes | **0.465** | 0.585 | 0.01 |
| Logistic Regression | 0.464 | **0.639** | 2.47 |
| Linear SVM | 0.459 | 0.627 | 0.34 |
| Random Forest | 0.420 | 0.584 | 11.49 |
| XGBoost | 0.463 | 0.625 | 56.00 |
| LightGBM | 0.424 | 0.629 | 35.84 |
| MLP (128 hidden) | 0.414 | 0.613 | 7.47 |

**The finding is a negative one, and it is the useful part.** On client talk
type a paired t-test across the five folds cannot separate the best learner from
any of the other six. The spread across all seven is 0.051 macro-F1 against a
fold-to-fold standard deviation of 0.043 — the gaps are inside the noise.
XGBoost needs 56 s to fit where Naive Bayes needs 0.01 s and scores no better.

That is evidence, not an excuse: with ~3,000 short utterances and tens of
thousands of sparse features, a regularised linear boundary is already the right
capacity, and the ceiling is set by the task rather than by the model. Chasing a
better utterance classifier would be a poor use of this project's effort. The
contribution sits in the layer above — within-person normalisation, trajectory
across sessions, and state-conditional therapist effects — which none of these
seven models addresses.

Model selection is also shown to be imperfect rather than hidden: the model with
the best CV score is not always the one with the best held-out score, so
`RESULTS.md` reports the held-out number for *every* model, not only the winner.

### Figures

| File | Shows |
|---|---|
| `artifacts/bench_models.png` | CV macro-F1 per learner, ± 1 SD, against the majority-class floor |
| `artifacts/bench_metrics.png` | Six held-out metrics × eight models, both tasks |
| `artifacts/bench_perclass.png` | Per-class F1 — the minority classes are the hard part |
| `artifacts/bench_confusion.png` | Confusion matrices for the selected model |
| `artifacts/bench_cost.png` | Macro-F1 against fit time, log scale |
| `artifacts/bench_folds.png` | Fold-level spread with paired t-test Δ and p |

## The methodological point

**Splits are grouped by `transcript_id`, never by utterance.** Utterances from
one conversation share a client, a therapist and a topic; an utterance-level
split lets the model recognise a conversation it memorised in training and
inflates every metric substantially. All figures above come from conversations
the model has never seen.

## Ablation (5-fold grouped CV, macro-F1)

| Features | Client talk | Therapist behaviour |
|---|---|---|
| Majority baseline | 0.228 | 0.145 |
| TherapyTrace lexicon only (10 features) | 0.451 | 0.523 |
| TF-IDF only (thousands of features) | 0.471 | 0.607 |
| TF-IDF + lexicon | **0.475** | **0.644** |
| + previous-turn context | 0.470 | 0.633 |

Two findings worth stating plainly:

- **Ten interpretable lexicon features come within 0.02 macro-F1 of thousands
  of TF-IDF features** on the client task. The rule-based layer is carrying real
  signal, not acting as decoration.
- **Adding the previous turn as context did not help** (−0.004, −0.011). A
  negative result: raw TF-IDF of the preceding utterance dilutes the signal.
  Encoding *what kind* of move preceded it, rather than its words, is the
  version worth trying next.

## Convergent validation

The lexicon never saw the expert labels, so AnnoMI is an independent test of it.

- **5 / 5** dimensions order correctly (change talk > sustain talk)
- Composite process score: **d = 0.389, p = 7.4 × 10⁻¹⁸** at utterance level
- Across 93 conversations, process score vs expert change-talk ratio:
  **r = 0.344, p = 7.3 × 10⁻⁴**
- Honest weakness: `emotional_granularity` (d = 0.010) and `problem_ownership`
  (d = 0.033) do **not** discriminate. Those two dimensions need the transformer
  tier; word lists are not enough for them.

## What is deliberately not here

No transformer. Fine-tuning MentalBERT needs HuggingFace access, which this
build environment does not have. `train.py` is structured so the transformer
slots in as one more configuration in `build_pipelines()`.

---

## Empirical finding — intervention effect is conditional on client state

`state_conditional_study.py`, run over 1,815 turn triples from 123 real
conversations. Client state is measured by the TherapyTrace process score; the
**outcome is the expert's own change-talk label**, so the result does not depend
on our lexicon being right about the outcome.

P(next client turn is change talk):

| Therapist move | Client stuck | Middling | Already moving | Swing |
|---|---|---|---|---|
| Open question | 26.8% | 41.2% | **47.9%** | +21 pts *** |
| Reflection | 26.0% | 38.2% | **45.5%** | +19 pts *** |
| Advice / information | 25.3% | 26.5% | 29.6% | +4 pts n.s. |
| Other | 24.4% | 33.1% | 41.5% | +17 pts *** |

Three things fall out of this table:

1. **Questions and reflections are amplifiers, not causes.** They roughly double
   their yield depending on the state they land in. The unconditional figure the
   field usually reports — questions 36.9%, reflections 34.4% — averages over a
   twenty-point spread and hides the effect entirely.

2. **Advice-giving is inert to state** (+4 pts, p = 0.73). It returns the same
   low rate whether the client is stuck or moving. It does not amplify.

3. **When a client is stuck, intervention choice barely matters** — all four
   converge to 24–27%. The differences between techniques only appear once the
   client already has momentum. That reframes the clinical question from "which
   technique is best" to "what moves a client out of a stuck state at all".

A fourth result is incidental but useful: the process-score terciles predict
expert change-talk rate monotonically for every intervention type. That is a
third independent validation of the lexicon, on a different quantity from the
earlier two.

**Confounding by indication is severe and unresolved.** Therapists ask questions
of clients who are already engaged. This is a description of what happens in
real sessions, not an experiment, and no causal claim is made.


---

## Sample video for the review module (M15)

M15 is verified against real footage, not only synthetic clips. MediaPipe Pose
is trained on real people and cannot find a drawn figure, so pose tracking
stayed unverified until a genuine recording was used.

```bash
curl -L -o /tmp/head-pose-face-detection-female.mp4 \
  https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/head-pose-face-detection-female.mp4

python -m pytest backend/tests -k real_human_footage
```

Measured on 20 s of that clip (768x432, 12 fps, seated person facing camera —
the framing a counselling recording actually has):

| | |
|---|---|
| Face detection | **100%** |
| Pose detection | **100%** |
| Throughput | 1.7x realtime on CPU |
| Posture openness | 1.275 (elbow span / shoulder width) |
| Gesture amplitude | 0.041 mean, 0.094 SD |

A useful negative control: the same pipeline on `vtest.avi` (distant walking
figures, surveillance framing) detects **0%** for both face and pose. The module
needs the subject reasonably close and facing the camera. That is a real
deployment constraint and it is stated rather than discovered later.
