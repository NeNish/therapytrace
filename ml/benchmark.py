#!/usr/bin/env python3
"""
TherapyTrace — M21: a multi-model benchmark on AnnoMI.

`train.py` asks a *feature* question: does the TherapyTrace lexicon add signal
on top of bag-of-words? It answers that by varying the features and holding the
classifier fixed at logistic regression.

This file asks the complementary *model* question: given one fixed feature
representation, how much does the choice of classifier actually matter? Seven
learners are compared, from a majority-class floor through linear models and
tree ensembles to a small neural network.

    0. Majority class            the floor every model must clear
    1. Multinomial Naive Bayes   the classic text-classification baseline
    2. Logistic Regression       linear, L2, class-weighted
    3. Linear SVM                hinge loss, class-weighted
    4. Random Forest             bagged trees
    5. XGBoost                   gradient-boosted trees, histogram split
    6. LightGBM                  leaf-wise gradient boosting
    7. MLP                       one hidden layer, 128 units

Every model sees the SAME matrix:

    TF-IDF word 1-2 grams  +  TF-IDF character 3-5 grams  +  12 lexicon features

so a difference in score is attributable to the learner and nothing else. The
lexicon block is min-max scaled rather than standardised, which keeps the whole
matrix non-negative and therefore admits Naive Bayes without special-casing it.

Class balancing is equalised for the same reason. The corpus is 53% neutral on
the client task, so a learner left to optimise raw accuracy will under-predict
the minority classes and lose macro-F1 for a reason that is not about the
algorithm. Every model is therefore balanced by whatever mechanism it offers:
`class_weight="balanced"` on the linear models and LightGBM,
`balanced_subsample` on the forest, uniform priors on Naive Bayes, and injected
per-row weights on XGBoost. The MLP is the exception — scikit-learn's
implementation accepts neither a class weight nor a sample weight — and that
asymmetry is stated in the results rather than glossed over.

METHODOLOGY — the choice that matters most in this file, unchanged from
train.py:

    Splits are grouped by transcript_id, never by utterance.

Utterances inside one conversation share a client, a therapist, a topic and a
speaking style. An utterance-level split lets a model memorise a conversation
during training and recognise it at test time, which inflates every metric
substantially and is a well-known flaw in published dialogue-classification
results. Here, model selection runs as 5-fold GroupKFold over the development
conversations, and the final number comes from a quarter of the conversations
that were held out before any model was fitted.

Reported per model: accuracy, macro-F1, weighted-F1, macro precision and
recall, one-vs-rest macro ROC-AUC, fit time, and a paired t-test against the
best model across the five CV folds.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import softmax
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.nlp.ml_features import Column, LexiconFeatures  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "AnnoMI-full.csv"
OUT = Path(__file__).resolve().parent / "artifacts"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.25

# Human-readable names, in the order they should appear in every table and chart.
MODEL_ORDER = [
    "majority_class",
    "naive_bayes",
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "xgboost",
    "lightgbm",
    "mlp",
]
DISPLAY = {
    "majority_class": "Majority class",
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP (128 hidden)",
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_task(task: str):
    """
    Returns (frame, y_int, groups, class_names).

    Labels are integer-encoded because XGBoost and LightGBM require it; the
    encoder's classes_ are carried through so every report reads in words.
    """
    df = pd.read_csv(DATA)
    df = df.sort_values(["transcript_id", "utterance_id"]).reset_index(drop=True)
    df["prev_text"] = df.groupby("transcript_id").utterance_text.shift(1).fillna("")

    if task == "client":
        df = df[df.interlocutor == "client"].dropna(subset=["client_talk_type"])
        y_raw = df.client_talk_type
    else:
        df = df[df.interlocutor == "therapist"].dropna(subset=["main_therapist_behaviour"])
        y_raw = df.main_therapist_behaviour

    df = df[df.utterance_text.notna()]
    df = df[df.utterance_text.astype(str).str.split().str.len() >= 3]
    y_raw = y_raw.loc[df.index]

    enc = LabelEncoder().fit(y_raw)
    frame = pd.DataFrame(
        {
            "text": df.utterance_text.astype(str),
            "prev_text": df.prev_text.astype(str),
        }
    )
    y = pd.Series(enc.transform(y_raw), index=frame.index)
    return frame, y, df.transcript_id, list(enc.classes_)


# ---------------------------------------------------------------------------
# one shared feature representation
# ---------------------------------------------------------------------------

def features() -> FeatureUnion:
    """
    Word n-grams catch vocabulary, character n-grams catch morphology and the
    misspellings that survive transcription, and the lexicon block carries the
    five process dimensions the rest of the system is built on.

    MinMaxScaler, not StandardScaler: Naive Bayes rejects negative inputs, and
    a bounded [0, 1] block also sits on the same scale as the TF-IDF values,
    which keeps the MLP's first layer well conditioned.
    """
    word = Pipeline(
        [
            ("col", Column("text")),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.85,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    max_features=20000,
                ),
            ),
        ]
    )
    char = Pipeline(
        [
            ("col", Column("text")),
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=5,
                    sublinear_tf=True,
                    max_features=20000,
                ),
            ),
        ]
    )
    lex = Pipeline(
        [
            ("col", Column("text")),
            ("lex", LexiconFeatures()),
            ("scale", MinMaxScaler()),
        ]
    )
    return FeatureUnion([("word", word), ("char", char), ("lex", lex)])


class BalancedXGB(XGBClassifier):
    """
    XGBoost with the class imbalance corrected.

    This exists for a fairness reason rather than a performance one. Every other
    learner in the comparison is told to balance the classes — `class_weight=
    "balanced"` on the linear models and LightGBM, `balanced_subsample` on the
    forest. XGBoost has no such parameter; it only accepts per-row weights at
    fit time. Left as-is it quietly optimises raw accuracy, which on a corpus
    that is 53% neutral means it under-predicts the minority classes and loses
    macro-F1 for a reason that has nothing to do with the algorithm.

    So the balanced weights are computed here and injected, putting XGBoost on
    the same footing as the rest. The one learner that still cannot be balanced
    is the MLP: scikit-learn's implementation accepts neither a class weight nor
    a sample weight, and that asymmetry is reported rather than hidden.
    """

    def fit(self, X, y, **kwargs):
        kwargs.setdefault("sample_weight", compute_sample_weight("balanced", y))
        return super().fit(X, y, **kwargs)


def classifiers(n_classes: int) -> dict:
    """One entry per learner. Hyperparameters are modest and fixed, not tuned
    on the test set."""
    import lightgbm as lgb

    return {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        # fit_prior=False gives uniform class priors, which is the Naive Bayes
        # analogue of the balanced class weighting every other model receives.
        "naive_bayes": MultinomialNB(alpha=0.3, fit_prior=False),
        "logistic_regression": LogisticRegression(
            max_iter=3000, C=4.0, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "linear_svm": LinearSVC(
            C=0.5, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": BalancedXGB(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.4,
            reg_lambda=1.0,
            tree_method="hist",
            objective="multi:softprob",
            num_class=n_classes,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.1,
            min_child_samples=10,
            colsample_bytree=0.4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=60,
            early_stopping=True,
            n_iter_no_change=6,
            random_state=RANDOM_STATE,
        ),
    }


def build(name: str, n_classes: int) -> Pipeline:
    return Pipeline([("feat", features()), ("clf", classifiers(n_classes)[name])])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def scores_for(fitted, X):
    """
    Class scores on a comparable scale.

    LinearSVC has no predict_proba; its decision function is pushed through a
    softmax so a one-vs-rest AUC can still be computed. That is a monotone
    transform per row, so the ranking AUC depends on is preserved.
    """
    if hasattr(fitted, "predict_proba"):
        return fitted.predict_proba(X)
    d = fitted.decision_function(X)
    if d.ndim == 1:
        d = np.column_stack([-d, d])
    return softmax(d, axis=1)


def macro_auc(y_true, score, n_classes) -> float | None:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, score[:, 1]))
        return float(
            roc_auc_score(y_true, score, multi_class="ovr", average="macro")
        )
    except Exception:
        return None


def metric_block(y_true, pred, score, n_classes) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, pred, average="weighted", zero_division=0)), 4),
        "macro_precision": round(
            float(precision_score(y_true, pred, average="macro", zero_division=0)), 4
        ),
        "macro_recall": round(
            float(recall_score(y_true, pred, average="macro", zero_division=0)), 4
        ),
        "macro_roc_auc": (
            round(a, 4) if (a := macro_auc(y_true, score, n_classes)) is not None else None
        ),
    }


def cross_validate(name, X, y, groups, n_classes) -> dict:
    """5-fold GroupKFold. Each fold's test set is whole conversations."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    per_fold, fit_times = [], []
    for tr, te in gkf.split(X, y, groups):
        pipe = build(name, n_classes)
        t0 = time.perf_counter()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        fit_times.append(time.perf_counter() - t0)
        pred = pipe.predict(X.iloc[te])
        per_fold.append(
            metric_block(y.iloc[te], pred, scores_for(pipe, X.iloc[te]), n_classes)
        )

    def agg(key):
        vals = [f[key] for f in per_fold if f[key] is not None]
        if not vals:
            return {"mean": None, "std": None, "folds": []}
        return {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "folds": [round(float(v), 4) for v in vals],
        }

    return {
        "cv": {k: agg(k) for k in per_fold[0]},
        "mean_fit_seconds": round(float(np.mean(fit_times)), 2),
    }


# ---------------------------------------------------------------------------
# one task end to end
# ---------------------------------------------------------------------------

def run_task(task: str) -> dict:
    label = "Client talk type" if task == "client" else "Therapist behaviour"
    X, y, groups, class_names = load_task(task)
    n_classes = len(class_names)

    print(f"\n{'=' * 86}\nTASK: {label}   ({n_classes} classes)\n{'=' * 86}")
    print(f"utterances {len(X)}   conversations {groups.nunique()}")
    counts = y.value_counts().sort_index()
    for i, c in enumerate(class_names):
        n = int(counts.get(i, 0))
        print(f"   {c:<18}{n:>6}  ({100 * n / len(y):.1f}%)")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    dev_idx, test_idx = next(gss.split(X, y, groups))
    X_dev, y_dev, g_dev = X.iloc[dev_idx], y.iloc[dev_idx], groups.iloc[dev_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    print(
        f"\ndev {len(X_dev)} utt / {g_dev.nunique()} conv"
        f"   |   held-out test {len(X_test)} utt / {groups.iloc[test_idx].nunique()} conv"
    )

    models: dict[str, dict] = {}
    header = f"{'model':<22}{'CV macro-F1':>18}{'CV acc':>10}{'AUC':>8}{'fit s':>9}"
    print(f"\n{header}\n{'-' * len(header)}")

    for name in MODEL_ORDER:
        res = cross_validate(name, X_dev, y_dev, g_dev, n_classes)
        models[name] = res
        cv = res["cv"]
        auc = cv["macro_roc_auc"]["mean"]
        print(
            f"{DISPLAY[name]:<22}"
            f"{cv['macro_f1']['mean']:.4f} ± {cv['macro_f1']['std']:.3f}"
            f"{cv['accuracy']['mean']:>10.4f}"
            f"{(f'{auc:.3f}' if auc is not None else '  n/a'):>8}"
            f"{res['mean_fit_seconds']:>9.1f}"
        )

    # Selection happens on CV macro-F1 over development conversations only.
    best = max(
        (n for n in MODEL_ORDER if n != "majority_class"),
        key=lambda n: models[n]["cv"]["macro_f1"]["mean"],
    )
    print(f"\nselected on CV macro-F1: {DISPLAY[best]}")

    # Paired t-test across the five folds, best against each other model.
    best_folds = models[best]["cv"]["macro_f1"]["folds"]
    for name in MODEL_ORDER:
        if name == best:
            models[name]["vs_best"] = {"delta": 0.0, "p_value": None, "is_best": True}
            continue
        other = models[name]["cv"]["macro_f1"]["folds"]
        t, p = stats.ttest_rel(best_folds, other)
        models[name]["vs_best"] = {
            "delta": round(
                models[best]["cv"]["macro_f1"]["mean"]
                - models[name]["cv"]["macro_f1"]["mean"],
                4,
            ),
            "p_value": round(float(p), 4),
            "is_best": False,
        }

    # Every model is then scored once on the untouched test conversations, so
    # the table shows what the runner-up would have delivered too.
    print(f"\nHELD-OUT TEST — all models, {len(X_test)} utterances from unseen conversations")
    header = f"{'model':<22}{'accuracy':>10}{'macro-F1':>11}{'weighted-F1':>13}{'AUC':>8}"
    print(f"{header}\n{'-' * len(header)}")
    for name in MODEL_ORDER:
        pipe = build(name, n_classes).fit(X_dev, y_dev)
        pred = pipe.predict(X_test)
        score = scores_for(pipe, X_test)
        block = metric_block(y_test, pred, score, n_classes)
        models[name]["test"] = block
        models[name]["test_per_class_f1"] = {
            class_names[i]: round(float(v), 4)
            for i, v in enumerate(
                f1_score(y_test, pred, average=None, labels=range(n_classes), zero_division=0)
            )
        }
        if name == best:
            models[name]["test_report"] = classification_report(
                y_test,
                pred,
                labels=range(n_classes),
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            models[name]["confusion_matrix"] = confusion_matrix(
                y_test, pred, labels=range(n_classes)
            ).tolist()
        auc = block["macro_roc_auc"]
        print(
            f"{DISPLAY[name]:<22}{block['accuracy']:>10.4f}{block['macro_f1']:>11.4f}"
            f"{block['weighted_f1']:>13.4f}"
            f"{(f'{auc:.3f}' if auc is not None else '  n/a'):>8}"
        )

    print(f"\nper-class F1 on the held-out conversations ({DISPLAY[best]})")
    for c, v in models[best]["test_per_class_f1"].items():
        print(f"   {c:<18}{v:.4f}")

    return {
        "task": task,
        "label": label,
        "n_utterances": int(len(X)),
        "n_conversations": int(groups.nunique()),
        "class_names": class_names,
        "class_balance": {class_names[i]: int(counts.get(i, 0)) for i in range(n_classes)},
        "n_dev": int(len(X_dev)),
        "n_dev_conversations": int(g_dev.nunique()),
        "n_test": int(len(X_test)),
        "n_test_conversations": int(groups.iloc[test_idx].nunique()),
        "selected_model": best,
        "models": models,
    }


def main() -> None:
    if not DATA.exists():
        raise SystemExit(
            f"AnnoMI not found at {DATA}\n"
            "  curl -L -o ml/data/AnnoMI-full.csv "
            "https://raw.githubusercontent.com/uccollab/AnnoMI/main/AnnoMI-full.csv"
        )

    out = {
        "dataset": "AnnoMI (Wu et al., 2023) — 133 expert-annotated MI demonstrations",
        "protocol": {
            "split": "GroupShuffleSplit by transcript_id, 25% of conversations held out",
            "cv": f"{N_FOLDS}-fold GroupKFold by transcript_id over development conversations",
            "features": "TF-IDF word 1-2gram + TF-IDF char_wb 3-5gram + 12 lexicon features, identical for every model",
            "selection": "CV macro-F1 on development conversations; the test set is scored once",
            "class_balancing": (
                "equalised across learners: class_weight='balanced' (linear models, LightGBM), "
                "balanced_subsample (random forest), uniform priors (naive Bayes), "
                "injected balanced sample weights (XGBoost). The MLP cannot be balanced in "
                "scikit-learn and is the one unbalanced learner."
            ),
            "random_state": RANDOM_STATE,
        },
        "display_names": DISPLAY,
        "model_order": MODEL_ORDER,
        "tasks": {},
    }
    for task in ("client", "therapist"):
        out["tasks"][task] = run_task(task)

    path = OUT / "benchmark.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
