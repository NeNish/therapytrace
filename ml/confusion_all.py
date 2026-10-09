#!/usr/bin/env python3
"""
Store a held-out confusion matrix for every model, not only the selected one.

The benchmark saves a confusion matrix for the winner alone, which is enough to
report accuracy but not enough to answer a product question: the interface
displays a *ratio* built from the change and sustain classes, so the quantity
that decides whether the screen is trustworthy is each model's per-class
confusion, especially its sustain recall. A model can win on macro-F1 and still
be the wrong thing to deploy.

This refits each model once on the development conversations and records its
confusion matrix on the held-out conversations, writing the result back into
benchmark.json. No cross-validation, so it costs one fit per model rather than
six.

    python ml/confusion_all.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
PATH = ART / "benchmark.json"


def main() -> None:
    data = json.loads(PATH.read_text())

    for task in ("client", "therapist"):
        X, y, groups, class_names = B.load_task(task)
        n_classes = len(class_names)

        gss = GroupShuffleSplit(n_splits=1, test_size=B.TEST_SIZE, random_state=B.RANDOM_STATE)
        dev_idx, test_idx = next(gss.split(X, y, groups))
        X_dev, y_dev = X.iloc[dev_idx], y.iloc[dev_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        print(f"\n{data['tasks'][task]['label']}  —  per-class recall on held-out conversations")
        print(f"{'model':<22}" + "".join(f"{c[:11]:>13}" for c in class_names))
        print("-" * (22 + 13 * n_classes))

        for name in B.MODEL_ORDER:
            pipe = B.build(name, n_classes).fit(X_dev, y_dev)
            pred = pipe.predict(X_test)
            cm = confusion_matrix(y_test, pred, labels=range(n_classes))
            data["tasks"][task]["models"][name]["confusion_matrix"] = cm.tolist()

            with np.errstate(invalid="ignore"):
                rec = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
            data["tasks"][task]["models"][name]["per_class_recall"] = {
                class_names[i]: round(float(rec[i]), 4) for i in range(n_classes)
            }
            print(f"{B.DISPLAY[name]:<22}" + "".join(f"{v:>13.3f}" for v in rec))

    data["protocol"]["confusion_all"] = (
        "A held-out confusion matrix and per-class recall are stored for every "
        "model, from a single fit on the development conversations, so the "
        "deployment choice can be made on the per-class behaviour the interface "
        "depends on rather than on macro-F1 alone."
    )
    PATH.write_text(json.dumps(data, indent=2))
    print(f"\nupdated {PATH}")


if __name__ == "__main__":
    main()
