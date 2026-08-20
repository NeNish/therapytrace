#!/usr/bin/env python3
"""
TherapyTrace — M10: supervised classifiers trained on AnnoMI.

Two tasks, both from expert annotations:

  Task A  client_talk_type      change / sustain / neutral      (6,725 utterances)
  Task B  main_therapist_behaviour
                                question / reflection /
                                therapist_input / other         (6,826 utterances)

Task A matters most: change talk vs sustain talk is the closest thing the field
has to a hand-labelled ground truth for "therapeutic momentum", which is exactly
what the TPI claims to measure. A trained change-talk classifier therefore does
double duty — it is a sixth signal for the index, and it is external validation
of the five lexicon dimensions.

METHODOLOGY NOTE — the single most important choice in this file:

    Splits are grouped by transcript_id, never by utterance.

Utterances from one conversation share a client, a therapist, a topic and a
speaking style. Splitting at the utterance level lets the model memorise a
conversation in training and recognise it in test, which inflates accuracy by a
large margin and is a well-known flaw in dialogue-classification papers. Every
number reported here comes from conversations the model has never seen.

Four feature configurations are compared so the contribution of the
TherapyTrace lexicon can be measured rather than assumed:

    1. TF-IDF only                    bag of words baseline
    2. Lexicon only                   the 5 process dimensions + auxiliaries
    3. TF-IDF + lexicon (hybrid)      does the lexicon add anything?
    4. Majority-class                 the floor any model must beat
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "therapytrace" / "backend"))
from app.nlp.features import AUXILIARY, DIMENSIONS, score_utterance  # noqa: E402
from app.nlp.ml_features import Column, LexiconFeatures  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "AnnoMI-full.csv"
OUT = Path(__file__).resolve().parent / "artifacts"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_task(task: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns a frame with `text` and `prev_text`.

    `prev_text` is the immediately preceding utterance in the same conversation.
    For the client task this is almost always the therapist's turn — and what a
    therapist just said is a strong prior on whether the client's reply is
    change talk. Including it tests a real clinical hypothesis rather than just
    adding features: that client language is a *response*, not an isolated
    utterance.
    """
    df = pd.read_csv(DATA)
    df = df.sort_values(["transcript_id", "utterance_id"]).reset_index(drop=True)
    df["prev_text"] = (
        df.groupby("transcript_id").utterance_text.shift(1).fillna("")
    )

    if task == "client":
        df = df[df.interlocutor == "client"].dropna(subset=["client_talk_type"])
        y = df.client_talk_type
    else:
        df = df[df.interlocutor == "therapist"].dropna(subset=["main_therapist_behaviour"])
        y = df.main_therapist_behaviour

    df = df[df.utterance_text.notna()]
    df = df[df.utterance_text.astype(str).str.split().str.len() >= 3]
    y = y.loc[df.index]
    frame = pd.DataFrame({
        "text": df.utterance_text.astype(str),
        "prev_text": df.prev_text.astype(str),
    })
    return frame, y, df.transcript_id


# ---------------------------------------------------------------------------
# model definitions
# ---------------------------------------------------------------------------

def tfidf_block() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def _text(): return Pipeline([("col", Column("text")), ("tfidf", tfidf_block())])
def _prev(): return Pipeline([("col", Column("prev_text")), ("tfidf", tfidf_block())])
def _lex():  return Pipeline([("col", Column("text")), ("lex", LexiconFeatures()),
                              ("scale", StandardScaler())])


def build_pipelines() -> dict[str, Pipeline]:
    return {
        "majority_baseline": Pipeline([
            ("feat", _lex()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "lexicon_only__logreg": Pipeline([
            ("feat", _lex()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        "tfidf__logreg": Pipeline([
            ("feat", _text()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", C=2.0, random_state=RANDOM_STATE
            )),
        ]),
        "tfidf__linearsvc": Pipeline([
            ("feat", _text()),
            ("clf", LinearSVC(class_weight="balanced", C=0.5, random_state=RANDOM_STATE)),
        ]),
        "tfidf__randomforest": Pipeline([
            ("feat", _text()),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                class_weight="balanced_subsample", random_state=RANDOM_STATE
            )),
        ]),
        "hybrid_tfidf_lexicon__logreg": Pipeline([
            ("feat", FeatureUnion([("tfidf", _text()), ("lex", _lex())])),
            ("clf", LogisticRegression(
                max_iter=3000, class_weight="balanced", C=2.0, random_state=RANDOM_STATE
            )),
        ]),
        "hybrid_plus_context__logreg": Pipeline([
            ("feat", FeatureUnion([
                ("tfidf", _text()), ("lex", _lex()), ("prev", _prev()),
            ])),
            ("clf", LogisticRegression(
                max_iter=3000, class_weight="balanced", C=2.0, random_state=RANDOM_STATE
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def cross_validate_grouped(pipe, X, y, groups) -> dict:
    """GroupKFold: every fold's test set is whole conversations, unseen in training."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    accs, f1s = [], []
    for tr, te in gkf.split(X, y, groups):
        p = pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = p.predict(X.iloc[te])
        accs.append(accuracy_score(y.iloc[te], pred))
        f1s.append(f1_score(y.iloc[te], pred, average="macro"))
    return {
        "cv_accuracy_mean": round(float(np.mean(accs)), 4),
        "cv_accuracy_std": round(float(np.std(accs)), 4),
        "cv_macro_f1_mean": round(float(np.mean(f1s)), 4),
        "cv_macro_f1_std": round(float(np.std(f1s)), 4),
        "fold_macro_f1": [round(f, 4) for f in f1s],
    }


def run_task(task: str) -> dict:
    label = "Client talk type" if task == "client" else "Therapist behaviour"
    print(f"\n{'=' * 78}\nTASK: {label}\n{'=' * 78}")

    X, y, groups = load_task(task)
    print(f"utterances: {len(X)}   conversations: {groups.nunique()}")
    print("class balance:")
    for k, v in y.value_counts().items():
        print(f"   {k:<18} {v:>5}  ({100 * v / len(y):.1f}%)")

    # Held-out test set: whole conversations, never touched during selection.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    dev_idx, test_idx = next(gss.split(X, y, groups))
    X_dev, y_dev, g_dev = X.iloc[dev_idx], y.iloc[dev_idx], groups.iloc[dev_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    print(f"\ndev: {len(X_dev)} utterances / {g_dev.nunique()} conversations")
    print(f"held-out test: {len(X_test)} utterances / {groups.iloc[test_idx].nunique()} conversations")

    results = {}
    print(f"\n{'model':<32}{'CV macro-F1':>16}{'CV accuracy':>16}")
    print("-" * 64)
    for name, pipe in build_pipelines().items():
        cv = cross_validate_grouped(pipe, X_dev, y_dev, g_dev)
        results[name] = cv
        print(f"{name:<32}"
              f"{cv['cv_macro_f1_mean']:.4f} ± {cv['cv_macro_f1_std']:.3f}"
              f"{cv['cv_accuracy_mean']:>10.4f}")

    # Select on CV macro-F1, then evaluate once on the held-out conversations.
    best_name = max(
        (n for n in results if n != "majority_baseline"),
        key=lambda n: results[n]["cv_macro_f1_mean"],
    )
    print(f"\nselected: {best_name}")

    best = build_pipelines()[best_name].fit(X_dev, y_dev)
    pred = best.predict(X_test)

    labels = sorted(y.unique())
    report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=labels)

    print(f"\nHELD-OUT TEST ({len(X_test)} utterances from unseen conversations)")
    print(classification_report(y_test, pred, labels=labels, zero_division=0))
    print("confusion matrix (rows = true, cols = predicted)")
    print(f"{'':<18}" + "".join(f"{l:>12}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l:<18}" + "".join(f"{v:>12}" for v in cm[i]))

    # Ablation: what does the lexicon add on top of bag-of-words?
    lex_f1 = results["lexicon_only__logreg"]["cv_macro_f1_mean"]
    tfidf_f1 = results["tfidf__logreg"]["cv_macro_f1_mean"]
    hyb_f1 = results["hybrid_tfidf_lexicon__logreg"]["cv_macro_f1_mean"]
    ctx_f1 = results["hybrid_plus_context__logreg"]["cv_macro_f1_mean"]
    maj_f1 = results["majority_baseline"]["cv_macro_f1_mean"]
    print(f"\nABLATION (CV macro-F1)")
    print(f"  majority baseline        {maj_f1:.4f}")
    print(f"  lexicon only             {lex_f1:.4f}   (+{lex_f1 - maj_f1:.4f} over baseline)")
    print(f"  tf-idf only              {tfidf_f1:.4f}")
    print(f"  tf-idf + lexicon         {hyb_f1:.4f}   ({hyb_f1 - tfidf_f1:+.4f} from adding lexicon)")
    print(f"  + previous turn context  {ctx_f1:.4f}   ({ctx_f1 - hyb_f1:+.4f} from adding context)")

    return {
        "task": task,
        "label": label,
        "n_utterances": int(len(X)),
        "n_conversations": int(groups.nunique()),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().items()},
        "cv_results": results,
        "selected_model": best_name,
        "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "test_macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
        "test_report": report,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "ablation": {
            "majority": maj_f1, "lexicon_only": lex_f1,
            "tfidf_only": tfidf_f1, "hybrid": hyb_f1,
            "hybrid_plus_context": ctx_f1,
            "lexicon_gain_over_tfidf": round(hyb_f1 - tfidf_f1, 4),
            "context_gain_over_hybrid": round(ctx_f1 - hyb_f1, 4),
        },
        "n_test": int(len(X_test)),
        "n_test_conversations": int(groups.iloc[test_idx].nunique()),
    }


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"AnnoMI not found at {DATA}")

    out = {"dataset": "AnnoMI (Wu et al., 2023)", "tasks": {}}
    for task in ("client", "therapist"):
        out["tasks"][task] = run_task(task)

    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
