#!/usr/bin/env python3
"""
Clean fit-time measurement for the M21 benchmark.

Timings collected during the benchmark run itself are only trustworthy if
nothing else is competing for the CPU, and in practice that is hard to
guarantee — a stray background job inflates whichever models happened to run
while it was alive, which silently corrupts any cost-versus-benefit claim.

So cost is measured separately here: one representative fold per model per
task, on an otherwise idle machine, with the feature build timed apart from the
classifier so the two are not conflated. The result is written as
`fit_seconds_clean` into benchmark.json and is the only timing the figures and
the write-up use.

    python ml/time_models.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

from sklearn.model_selection import GroupKFold, GroupShuffleSplit

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
        dev_idx, _ = next(gss.split(X, y, groups))
        X_dev, y_dev, g_dev = X.iloc[dev_idx], y.iloc[dev_idx], groups.iloc[dev_idx]

        # Fold 0 of the same GroupKFold the benchmark used.
        tr, _ = next(GroupKFold(n_splits=B.N_FOLDS).split(X_dev, y_dev, g_dev))
        Xtr, ytr = X_dev.iloc[tr], y_dev.iloc[tr]

        # Feature build, timed once and shared by every model below.
        t0 = time.perf_counter()
        feat = B.features().fit(Xtr, ytr)
        M = feat.transform(Xtr)
        feat_s = time.perf_counter() - t0

        print(f"\n{data['tasks'][task]['label']}  —  {M.shape[0]} rows × {M.shape[1]} features")
        print(f"{'model':<22}{'classifier s':>14}{'+ features s':>15}")
        print("-" * 51)
        data["tasks"][task]["feature_build_seconds"] = round(feat_s, 2)

        for name in B.MODEL_ORDER:
            clf = B.classifiers(n_classes)[name]
            t0 = time.perf_counter()
            clf.fit(M, ytr.values)
            clf_s = time.perf_counter() - t0
            data["tasks"][task]["models"][name]["fit_seconds_clean"] = round(clf_s, 2)
            data["tasks"][task]["models"][name]["fit_seconds_total_clean"] = round(
                clf_s + feat_s, 2
            )
            print(f"{B.DISPLAY[name]:<22}{clf_s:>14.2f}{clf_s + feat_s:>15.2f}")

    data["protocol"]["timing"] = (
        "fit_seconds_clean is the classifier's own fit time on fold 0 of the "
        "development split, measured on an idle machine with 2 vCPUs; "
        "feature_build_seconds is the shared TF-IDF and lexicon build, timed "
        "separately. The timings recorded during the benchmark run itself "
        "(mean_fit_seconds) are left in place but are not used for any claim, "
        "because they were collected under variable CPU load."
    )
    PATH.write_text(json.dumps(data, indent=2))
    print(f"\nupdated {PATH}")


if __name__ == "__main__":
    main()
