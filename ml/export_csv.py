#!/usr/bin/env python3
"""
Export the benchmark results as CSV, for a report table or a spreadsheet.

Three files rather than one, because the three things have different shapes and
forcing them into a single sheet would mean either empty columns or repeated
values:

    model_metrics.csv    one row per task x model — the headline table
    per_class.csv        long form: task, model, class, F1, recall
    confusion.csv        long form: task, model, actual, predicted, count

Per-class figures are in long form because the two tasks have different class
names (three for client talk, four for therapist behaviour). A wide sheet would
need columns that are blank for half the rows; long form stays rectangular and
pivots cleanly in Excel or pandas.

Everything is read from ml/artifacts/benchmark.json, so the CSVs cannot drift
from what the training run measured.

    python ml/export_csv.py [output_dir]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ART = Path(__file__).resolve().parent / "artifacts"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ART / "csv"

DEPLOYED = "logistic_regression"


def main() -> None:
    b = json.loads((ART / "benchmark.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    names = b["display_names"]
    order = b["model_order"]

    # ---------------------------------------------------------------- headline
    path = OUT / "model_metrics.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "task", "model", "role",
            "cv_macro_f1", "cv_macro_f1_sd", "cv_accuracy",
            "test_accuracy", "test_macro_f1", "test_weighted_f1",
            "test_macro_precision", "test_macro_recall", "test_roc_auc",
            "fit_seconds", "p_value_vs_selected",
            "n_test_utterances", "n_test_conversations",
        ])
        for key in ("client", "therapist"):
            t = b["tasks"][key]
            for n in order:
                m = t["models"][n]
                cv, test = m["cv"], m["test"]
                role = []
                if n == "majority_class":
                    role.append("baseline")
                if n == t["selected_model"]:
                    role.append("selected on CV macro-F1")
                if n == DEPLOYED:
                    role.append("deployed")
                w.writerow([
                    t["label"], names[n], "; ".join(role),
                    cv["macro_f1"]["mean"], cv["macro_f1"]["std"], cv["accuracy"]["mean"],
                    test["accuracy"], test["macro_f1"], test["weighted_f1"],
                    test["macro_precision"], test["macro_recall"],
                    "" if test["macro_roc_auc"] is None else test["macro_roc_auc"],
                    "" if m.get("fit_seconds_clean") is None else m["fit_seconds_clean"],
                    "" if (m.get("vs_best") or {}).get("p_value") is None
                    else m["vs_best"]["p_value"],
                    t["n_test"], t["n_test_conversations"],
                ])
    print(f"wrote {path}")

    # --------------------------------------------------------------- per class
    path = OUT / "per_class.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "model", "class", "f1", "recall", "support"])
        for key in ("client", "therapist"):
            t = b["tasks"][key]
            # support = the true count per class in the test set, read off the
            # confusion matrix rather than assumed from the overall balance
            for n in order:
                m = t["models"][n]
                f1s = m.get("test_per_class_f1", {})
                rec = m.get("per_class_recall", {})
                cm = m.get("confusion_matrix")
                for i, c in enumerate(t["class_names"]):
                    support = sum(cm[i]) if cm else ""
                    w.writerow([
                        t["label"], names[n], c,
                        f1s.get(c, ""), rec.get(c, ""), support,
                    ])
    print(f"wrote {path}")

    # --------------------------------------------------------------- confusion
    path = OUT / "confusion.csv"
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "model", "actual", "predicted", "count", "row_share"])
        for key in ("client", "therapist"):
            t = b["tasks"][key]
            for n in order:
                cm = t["models"][n].get("confusion_matrix")
                if not cm:
                    continue
                for i, actual in enumerate(t["class_names"]):
                    row_total = sum(cm[i]) or 1
                    for j, predicted in enumerate(t["class_names"]):
                        w.writerow([
                            t["label"], names[n], actual, predicted,
                            cm[i][j], round(cm[i][j] / row_total, 4),
                        ])
    print(f"wrote {path}")

    print(
        f"\ndataset: {b['dataset']}\n"
        f"protocol: {b['protocol']['split']}\n"
        f"deployed: {names[DEPLOYED]} — chosen on sustain recall, not macro-F1"
    )


if __name__ == "__main__":
    main()
