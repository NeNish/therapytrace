#!/usr/bin/env python3
"""
M25 — is the three-class framing the thing holding the number down?

The benchmark reports roughly 0.54 accuracy on client talk type, which looks
poor until you notice the majority-class floor is 0.554: the corpus is 53%
neutral, so always answering "neutral" scores higher than any real model. That
is a sign the metric is wrong for the task, not that the models are useless —
macro-F1 goes from 0.238 at the floor to 0.489, which is the part that moved.

But there is a better question hiding behind it. The product never uses the
neutral class. The change-talk ratio is

    change / (change + sustain)

computed over directional turns only; neutral turns are excluded from the
denominator by construction. So the three-class problem is not the problem the
interface needs solved. The interface needs: given a turn that IS directional,
is it change or sustain?

That is a different task — two classes, 1,425 against 786, and no 53% neutral
class soaking up the probability mass. This file measures it under exactly the
benchmark's protocol so the numbers are comparable:

    * same conversation-grouped splits, same held-out conversations
    * same feature matrix
    * same seven learners, same class balancing

If the binary framing scores substantially better, then the honest headline for
the project is the binary number, because that is the classifier the product
actually runs on.

    python ml/binary_study.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"


def load_directional():
    """Client turns labelled change or sustain. Neutral is dropped entirely."""
    df = pd.read_csv(B.DATA)
    df = df.sort_values(["transcript_id", "utterance_id"]).reset_index(drop=True)
    df["prev_text"] = df.groupby("transcript_id").utterance_text.shift(1).fillna("")

    df = df[df.interlocutor == "client"].dropna(subset=["client_talk_type"])
    df = df[df.client_talk_type.isin(["change", "sustain"])]
    df = df[df.utterance_text.notna()]
    df = df[df.utterance_text.astype(str).str.split().str.len() >= 3]

    enc = LabelEncoder().fit(df.client_talk_type)
    X = pd.DataFrame({
        "text": df.utterance_text.astype(str),
        "prev_text": df.prev_text.astype(str),
    })
    y = pd.Series(enc.transform(df.client_talk_type), index=X.index)
    return X, y, df.transcript_id, list(enc.classes_)


def main() -> None:
    X, y, groups, classes = load_directional()
    counts = y.value_counts().sort_index()
    print(f"directional client turns: {len(X)} across {groups.nunique()} conversations")
    for i, c in enumerate(classes):
        print(f"   {c:<10}{int(counts[i]):>6}  ({100 * counts[i] / len(y):.1f}%)")
    floor = max(counts) / len(y)
    print(f"\nmajority-class floor: {floor:.4f} accuracy\n")

    gss = GroupShuffleSplit(n_splits=1, test_size=B.TEST_SIZE, random_state=B.RANDOM_STATE)
    dev_idx, test_idx = next(gss.split(X, y, groups))
    X_dev, y_dev, g_dev = X.iloc[dev_idx], y.iloc[dev_idx], groups.iloc[dev_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    print(f"dev {len(X_dev)} / {g_dev.nunique()} conv   |   "
          f"held-out {len(X_test)} / {groups.iloc[test_idx].nunique()} conv\n")

    header = f"{'model':<22}{'CV macro-F1':>16}{'CV acc':>10}"
    print(header + "\n" + "-" * len(header))

    results = {}
    for name in B.MODEL_ORDER:
        gkf = GroupKFold(n_splits=B.N_FOLDS)
        f1s, accs = [], []
        for tr, te in gkf.split(X_dev, y_dev, g_dev):
            pipe = B.build(name, 2).fit(X_dev.iloc[tr], y_dev.iloc[tr])
            pred = pipe.predict(X_dev.iloc[te])
            f1s.append(f1_score(y_dev.iloc[te], pred, average="macro"))
            accs.append(accuracy_score(y_dev.iloc[te], pred))
        results[name] = {
            "cv_macro_f1": round(float(np.mean(f1s)), 4),
            "cv_macro_f1_sd": round(float(np.std(f1s)), 4),
            "cv_accuracy": round(float(np.mean(accs)), 4),
        }
        print(f"{B.DISPLAY[name]:<22}"
              f"{np.mean(f1s):.4f} ± {np.std(f1s):.3f}{np.mean(accs):>10.4f}")

    print(f"\n{'HELD-OUT':<22}{'accuracy':>10}{'macro-F1':>11}{'ROC-AUC':>10}")
    print("-" * 53)
    for name in B.MODEL_ORDER:
        pipe = B.build(name, 2).fit(X_dev, y_dev)
        pred = pipe.predict(X_test)
        score = B.scores_for(pipe, X_test)
        acc = accuracy_score(y_test, pred)
        mf1 = f1_score(y_test, pred, average="macro")
        try:
            auc = roc_auc_score(y_test, score[:, 1])
        except Exception:
            auc = float("nan")
        results[name].update({
            "test_accuracy": round(float(acc), 4),
            "test_macro_f1": round(float(mf1), 4),
            "test_roc_auc": round(float(auc), 4),
            "per_class_f1": {
                classes[i]: round(float(v), 4)
                for i, v in enumerate(
                    f1_score(y_test, pred, average=None, labels=[0, 1], zero_division=0)
                )
            },
            "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        })
        print(f"{B.DISPLAY[name]:<22}{acc:>10.4f}{mf1:>11.4f}{auc:>10.4f}")

    best = max(
        (n for n in B.MODEL_ORDER if n != "majority_class"),
        key=lambda n: results[n]["cv_macro_f1"],
    )
    print(f"\nselected on CV macro-F1: {B.DISPLAY[best]}")
    print(f"\nHeld-out report — {B.DISPLAY[best]}")
    pipe = B.build(best, 2).fit(X_dev, y_dev)
    print(classification_report(y_test, pipe.predict(X_test),
                                target_names=classes, zero_division=0))

    out = {
        "task": "client directional talk: change vs sustain",
        "rationale": (
            "The interface computes change / (change + sustain) over directional "
            "turns only, so the neutral class is never used. This measures the "
            "classifier the product actually depends on."
        ),
        "n_utterances": int(len(X)),
        "n_conversations": int(groups.nunique()),
        "class_balance": {classes[i]: int(counts[i]) for i in range(2)},
        "majority_floor_accuracy": round(float(floor), 4),
        "n_test": int(len(X_test)),
        "n_test_conversations": int(groups.iloc[test_idx].nunique()),
        "selected_model": best,
        "models": results,
    }
    (ART / "binary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ART / 'binary.json'}")


if __name__ == "__main__":
    main()
