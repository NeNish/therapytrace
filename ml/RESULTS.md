# M21 — multi-model benchmark results

**Dataset.** AnnoMI (Wu et al., 2023) — 133 expert-annotated MI demonstrations

AnnoMI is a corpus of *performed* motivational-interviewing demonstrations, collected from training videos. It is expert-annotated but it is not clinical session data, and the 23 low-quality transcripts are deliberately staged poor counselling. Every claim below is a claim about this corpus only.

## Protocol

- **Split** — GroupShuffleSplit by transcript_id, 25% of conversations held out
- **Cv** — 5-fold GroupKFold by transcript_id over development conversations
- **Features** — TF-IDF word 1-2gram + TF-IDF char_wb 3-5gram + 12 lexicon features, identical for every model
- **Selection** — CV macro-F1 on development conversations; the test set is scored once
- **Class Balancing** — equalised across learners: class_weight='balanced' (linear models, LightGBM), balanced_subsample (random forest), uniform priors (naive Bayes), injected balanced sample weights (XGBoost). The MLP cannot be balanced in scikit-learn and is the one unbalanced learner.
- **Random State** — 42
- **Timing** — fit_seconds_clean is the classifier's own fit time on fold 0 of the development split, measured on an idle machine with 2 vCPUs; feature_build_seconds is the shared TF-IDF and lexicon build, timed separately. The timings recorded during the benchmark run itself (mean_fit_seconds) are left in place but are not used for any claim, because they were collected under variable CPU load.

## Task: Client talk type

4740 utterances across 133 conversations. Class balance: `change` 1425 (30.1%), `neutral` 2529 (53.4%), `sustain` 786 (16.6%).

Development set 3084 utterances / 99 conversations. Held-out test 1656 utterances / 34 conversations.

### Cross-validation on development conversations

| Model | CV macro-F1 | CV accuracy | CV ROC-AUC | Fit (s) | Δ vs best | p |
|---|---|---|---|---|---|---|
| Majority class | 0.2283 ± 0.015 | 0.5227 | 0.500 | 0.00 | +0.2369 | 0.000 |
| **Naive Bayes** | 0.4652 ± 0.043 | 0.5127 | 0.656 | 0.01 | — | — |
| Logistic Regression | 0.4637 ± 0.026 | 0.5204 | 0.648 | 2.47 | +0.0015 | 0.938 |
| Linear SVM | 0.4587 ± 0.029 | 0.5247 | 0.643 | 0.34 | +0.0065 | 0.741 |
| Random Forest | 0.4203 ± 0.035 | 0.5266 | 0.670 | 11.49 | +0.0449 | 0.068 |
| XGBoost | 0.4625 ± 0.039 | 0.5279 | 0.650 | 56.00 | +0.0027 | 0.880 |
| LightGBM | 0.4240 ± 0.018 | 0.5247 | 0.633 | 35.84 | +0.0412 | 0.092 |
| MLP (128 hidden) | 0.4139 ± 0.040 | 0.5490 | 0.647 | 7.47 | +0.0513 | 0.264 |

Selected on CV macro-F1: **Naive Bayes**. `Δ` and `p` come from a paired t-test across the same five folds.

### Held-out test — every model scored once

| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro precision | Macro recall | ROC-AUC |
|---|---|---|---|---|---|---|
| Majority class | 0.5537 | 0.2376 | 0.3947 | 0.1846 | 0.3333 | 0.500 |
| **Naive Bayes** | 0.5254 | 0.4565 | 0.5233 | 0.4603 | 0.4691 | 0.674 |
| Logistic Regression | 0.5423 | 0.4886 | 0.5471 | 0.4912 | 0.4936 | 0.678 |
| Linear SVM | 0.5664 | 0.4925 | 0.5637 | 0.4964 | 0.4919 | 0.675 |
| Random Forest | 0.6135 | 0.5002 | 0.5895 | 0.5780 | 0.5044 | 0.690 |
| XGBoost | 0.5495 | 0.4846 | 0.5508 | 0.4841 | 0.4907 | 0.679 |
| LightGBM | 0.5411 | 0.4254 | 0.5239 | 0.4305 | 0.4327 | 0.651 |
| MLP (128 hidden) | 0.6184 | 0.4624 | 0.5705 | 0.6193 | 0.4766 | 0.685 |

### Per-class F1 on the held-out conversations

| Model | `change` | `neutral` | `sustain` |
|---|---|---|---|
| **Naive Bayes** | 0.508 | 0.620 | 0.242 |
| Logistic Regression | 0.466 | 0.649 | 0.351 |
| Linear SVM | 0.472 | 0.686 | 0.320 |
| Random Forest | 0.522 | 0.731 | 0.248 |
| XGBoost | 0.508 | 0.653 | 0.292 |
| LightGBM | 0.441 | 0.682 | 0.153 |
| MLP (128 hidden) | 0.524 | 0.732 | 0.131 |

### Confusion matrix — Naive Bayes

| actual ↓ / predicted → | `change` | `neutral` | `sustain` |
|---|---|---|---|
| `change` | 277 | 129 | 43 |
| `neutral` | 274 | 532 | 111 |
| `sustain` | 90 | 139 | 61 |

## Task: Therapist behaviour

5078 utterances across 133 conversations. Class balance: `other` 526 (10.4%), `question` 1904 (37.5%), `reflection` 1665 (32.8%), `therapist_input` 983 (19.4%).

Development set 3284 utterances / 99 conversations. Held-out test 1794 utterances / 34 conversations.

### Cross-validation on development conversations

| Model | CV macro-F1 | CV accuracy | CV ROC-AUC | Fit (s) | Δ vs best | p |
|---|---|---|---|---|---|---|
| Majority class | 0.1451 ± 0.020 | 0.4135 | 0.500 | 0.00 | +0.4942 | 0.000 |
| Naive Bayes | 0.5853 ± 0.038 | 0.6738 | 0.847 | 0.01 | +0.0540 | 0.011 |
| **Logistic Regression** | 0.6393 ± 0.024 | 0.7140 | 0.866 | 4.03 | — | — |
| Linear SVM | 0.6273 ± 0.025 | 0.7083 | 0.862 | 0.40 | +0.0120 | 0.115 |
| Random Forest | 0.5836 ± 0.035 | 0.6668 | 0.845 | 11.80 | +0.0557 | 0.009 |
| XGBoost | 0.6254 ± 0.033 | 0.7003 | 0.865 | 84.42 | +0.0139 | 0.127 |
| LightGBM | 0.6288 ± 0.037 | 0.7101 | 0.863 | 44.81 | +0.0105 | 0.244 |
| MLP (128 hidden) | 0.6126 ± 0.032 | 0.7049 | 0.858 | 14.15 | +0.0267 | 0.015 |

Selected on CV macro-F1: **Logistic Regression**. `Δ` and `p` come from a paired t-test across the same five folds.

### Held-out test — every model scored once

| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro precision | Macro recall | ROC-AUC |
|---|---|---|---|---|---|---|
| Majority class | 0.3043 | 0.1167 | 0.1420 | 0.0761 | 0.2500 | 0.500 |
| Naive Bayes | 0.6505 | 0.5983 | 0.6405 | 0.6887 | 0.5847 | 0.836 |
| **Logistic Regression** | 0.6639 | 0.6077 | 0.6560 | 0.6245 | 0.6032 | 0.850 |
| Linear SVM | 0.6778 | 0.6117 | 0.6640 | 0.6396 | 0.6078 | 0.842 |
| Random Forest | 0.6433 | 0.6000 | 0.6360 | 0.6474 | 0.5875 | 0.847 |
| XGBoost | 0.6878 | 0.6511 | 0.6852 | 0.6627 | 0.6478 | 0.860 |
| LightGBM | 0.6867 | 0.6298 | 0.6741 | 0.6669 | 0.6227 | 0.851 |
| MLP (128 hidden) | 0.6845 | 0.6202 | 0.6673 | 0.7224 | 0.6096 | 0.856 |

### Per-class F1 on the held-out conversations

| Model | `other` | `question` | `reflection` | `therapist_input` |
|---|---|---|---|---|
| Naive Bayes | 0.377 | 0.779 | 0.656 | 0.582 |
| **Logistic Regression** | 0.362 | 0.802 | 0.697 | 0.570 |
| Linear SVM | 0.348 | 0.813 | 0.719 | 0.567 |
| Random Forest | 0.426 | 0.795 | 0.646 | 0.533 |
| XGBoost | 0.488 | 0.814 | 0.713 | 0.589 |
| LightGBM | 0.406 | 0.811 | 0.712 | 0.591 |
| MLP (128 hidden) | 0.378 | 0.818 | 0.696 | 0.589 |

### Confusion matrix — Logistic Regression

| actual ↓ / predicted → | `other` | `question` | `reflection` | `therapist_input` |
|---|---|---|---|---|
| `other` | 69 | 14 | 92 | 45 |
| `question` | 8 | 431 | 62 | 45 |
| `reflection` | 43 | 32 | 458 | 46 |
| `therapist_input` | 41 | 52 | 123 | 233 |

## What the comparison actually shows

**Client talk type.** Every learner clears the majority-class floor of 0.228 macro-F1 by a wide margin, so the features carry real signal. But the spread between the best and worst of the seven is only 0.051 macro-F1, against a fold-to-fold standard deviation of 0.043 for the selected model alone. A paired t-test cannot separate Naive Bayes from **any** of the other six at the 5% level — on this task the seven learners are statistically indistinguishable. XGBoost needs 56 s to fit where Naive Bayes needs 0.01 s — a difference of three orders of magnitude for no gain in score.

**Therapist behaviour.** Every learner clears the majority-class floor of 0.145 macro-F1 by a wide margin, so the features carry real signal. But the spread between the best and worst of the seven is only 0.056 macro-F1, against a fold-to-fold standard deviation of 0.024 for the selected model alone. A paired t-test cannot separate Logistic Regression from Linear SVM, XGBoost, LightGBM at the 5% level. XGBoost costs 211× the compute of Linear SVM (84 s against 0.40 s) and does not buy a better score.

The conclusion is not that model choice never matters. It is that on this corpus, at this size, with these labels, **the ceiling is set by the task and not by the learner.** Roughly three thousand short utterances and tens of thousands of sparse features is a regime where a regularised linear boundary is already the right capacity; the extra expressiveness of boosted trees has nothing left to find and spends compute discovering that.

That matters for how the project is read. The contribution of TherapyTrace is not a better utterance classifier — this benchmark is the evidence that chasing one would be a poor use of effort. The contribution is the layer above: within-person normalisation, trajectory estimation across sessions, and state-conditional estimates of which therapist move helps. Those are measured elsewhere in `ml/`, and none of the seven models here addresses them.

## Honest limitations

- **AnnoMI is performed, not clinical.** Transfer to real sessions is unmeasured.
- **One unbalanced learner.** scikit-learn's MLP accepts neither class weights nor sample weights, so it alone optimises raw accuracy. Read its macro-F1 with that in mind; its accuracy is comparable, its macro-F1 is mildly flattered on the majority class and penalised on the minorities.
- **Hyperparameters are fixed, not tuned per model.** A tuned XGBoost might close its gap. The budget went to a clean protocol rather than a leaderboard, and the paired tests suggest the headroom is small.
- **Selection is imperfect.** Model selection ran on development folds, so the model with the best CV score is not always the one with the best held-out score. Both numbers are reported for every model rather than only the flattering one.
- **No transformer baseline.** A fine-tuned encoder such as MentalBERT is the obvious next comparison and is not present; the environment has no model-hub access. Published AnnoMI transformer results are the reference point.

