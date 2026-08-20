# ML component (M10)

Two supervised classifiers trained on **AnnoMI** (Wu et al., 2023) — 133
expert-annotated motivational-interviewing dialogues.

## Reproducing

```bash
curl -L -o ml/data/AnnoMI-full.csv \
  https://raw.githubusercontent.com/uccollab/AnnoMI/main/AnnoMI-full.csv

pip install scikit-learn pandas scipy joblib matplotlib
python ml/train.py                     # trains, cross-validates, ablates
python ml/finalize.py                  # fits final models, saves figures
python ml/validate_against_annomi.py   # validates the TPI against experts
```

The dataset is **not** committed — download it from the link above.

## Results (held-out conversations)

| Task | Classes | Accuracy | Macro-F1 |
|---|---|---|---|
| Client talk type | change / sustain / neutral | 0.54 | 0.49 |
| Therapist behaviour | question / reflection / therapist_input / other | 0.65 | 0.61 |

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
