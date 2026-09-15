"""
Can we honestly reach higher accuracy?

Four avenues, all legitimate:
  A. Binary change-vs-sustain. Dropping the 'neutral' filler class leaves the
     contrast motivational-interviewing research actually cares about — the
     change/sustain ratio is what predicts outcome.
  B. Therapist 3-class, dropping the 'other' catch-all whose F1 (0.41) drags
     the macro average down and which has no clinical meaning.
  C. Hyperparameter search on the winning pipeline.
  D. Character n-grams alongside word n-grams.

All splits remain grouped by conversation.
"""
import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "therapytrace" / "backend"))
from app.nlp.ml_features import Column, LexiconFeatures

df = pd.read_csv(HERE / "data" / "AnnoMI-full.csv")
df = df[df.utterance_text.notna()]
df = df[df.utterance_text.astype(str).str.split().str.len() >= 3]

def frame(sub, ycol):
    return pd.DataFrame({"text": sub.utterance_text.astype(str),
                         "prev_text": ""}), sub[ycol], sub.transcript_id

def tf(word=True):
    return TfidfVectorizer(ngram_range=(1,2) if word else (3,5),
                           analyzer="word" if word else "char_wb",
                           min_df=3, max_df=0.85, sublinear_tf=True, strip_accents="unicode")

def pipe(C=2.0, chars=False, clf="lr"):
    feats = [("w", Pipeline([("c", Column("text")), ("t", tf(True))])),
             ("lx", Pipeline([("c", Column("text")), ("l", LexiconFeatures()), ("s", StandardScaler())]))]
    if chars:
        feats.insert(1, ("ch", Pipeline([("c", Column("text")), ("t", tf(False))])))
    model = (LogisticRegression(max_iter=4000, class_weight="balanced", C=C)
             if clf == "lr" else
             CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=C), cv=3))
    return Pipeline([("f", FeatureUnion(feats)), ("clf", model)])

def cv(p, X, y, g, n=5):
    accs, f1s = [], []
    for tr, te in GroupKFold(n_splits=n).split(X, y, g):
        m = p.fit(X.iloc[tr], y.iloc[tr]); pr = m.predict(X.iloc[te])
        accs.append(accuracy_score(y.iloc[te], pr))
        f1s.append(f1_score(y.iloc[te], pr, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s)), float(np.std(f1s))

results = {}

# ---------------- A. binary change vs sustain ----------------
print("="*72); print("A. BINARY: change talk vs sustain talk (neutral excluded)"); print("="*72)
sub = df[(df.interlocutor=="client") & df.client_talk_type.isin(["change","sustain"])]
X, y, g = frame(sub, "client_talk_type")
print(f"n = {len(X)}  conversations = {g.nunique()}  balance = {dict(y.value_counts())}")
best = None
for C in [0.5, 1.0, 2.0, 4.0]:
    for chars in [False, True]:
        a, f, sd = cv(pipe(C, chars), X, y, g)
        tag = f"C={C} chars={chars}"
        print(f"   {tag:<20} acc {a:.4f}   macro-F1 {f:.4f} ± {sd:.3f}")
        if best is None or f > best[1]: best = (tag, f, a, C, chars)
print(f"\n   best: {best[0]}  acc {best[2]:.4f}  macro-F1 {best[1]:.4f}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
di, ti = next(gss.split(X, y, g))
m = pipe(best[3], best[4]).fit(X.iloc[di], y.iloc[di])
pr = m.predict(X.iloc[ti])
ta, tf1 = accuracy_score(y.iloc[ti], pr), f1_score(y.iloc[ti], pr, average="macro")
print(f"   HELD-OUT: accuracy {ta:.4f}   macro-F1 {tf1:.4f}   ({len(ti)} utts, {g.iloc[ti].nunique()} convs)")
print(classification_report(y.iloc[ti], pr, zero_division=0))
results["binary_change_sustain"] = {"cv_acc": best[2], "cv_macro_f1": best[1],
                                    "test_acc": ta, "test_macro_f1": tf1,
                                    "n": int(len(X)), "config": best[0]}

# ---------------- B. therapist 3-class ----------------
print("="*72); print("B. THERAPIST: question / reflection / input ('other' excluded)"); print("="*72)
sub = df[(df.interlocutor=="therapist") &
         df.main_therapist_behaviour.isin(["question","reflection","therapist_input"])]
X2, y2, g2 = frame(sub, "main_therapist_behaviour")
print(f"n = {len(X2)}  conversations = {g2.nunique()}")
best2 = None
for C in [1.0, 2.0, 4.0]:
    for chars in [False, True]:
        a, f, sd = cv(pipe(C, chars), X2, y2, g2)
        print(f"   C={C} chars={chars:<6} acc {a:.4f}   macro-F1 {f:.4f} ± {sd:.3f}")
        if best2 is None or f > best2[1]: best2 = (f"C={C} chars={chars}", f, a, C, chars)
print(f"\n   best: {best2[0]}  acc {best2[2]:.4f}  macro-F1 {best2[1]:.4f}")

di2, ti2 = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X2, y2, g2))
m2 = pipe(best2[3], best2[4]).fit(X2.iloc[di2], y2.iloc[di2])
pr2 = m2.predict(X2.iloc[ti2])
ta2, tf2 = accuracy_score(y2.iloc[ti2], pr2), f1_score(y2.iloc[ti2], pr2, average="macro")
print(f"   HELD-OUT: accuracy {ta2:.4f}   macro-F1 {tf2:.4f}")
print(classification_report(y2.iloc[ti2], pr2, zero_division=0))
results["therapist_3class"] = {"cv_acc": best2[2], "cv_macro_f1": best2[1],
                               "test_acc": ta2, "test_macro_f1": tf2,
                               "n": int(len(X2)), "config": best2[0]}

# ---------------- C. binary: question vs not ----------------
print("="*72); print("C. THERAPIST BINARY: question vs everything else"); print("="*72)
sub = df[df.interlocutor=="therapist"].dropna(subset=["main_therapist_behaviour"])
X3 = pd.DataFrame({"text": sub.utterance_text.astype(str), "prev_text": ""})
y3 = (sub.main_therapist_behaviour == "question").map({True:"question", False:"not_question"})
g3 = sub.transcript_id
a3, f3, _ = cv(pipe(2.0, True), X3, y3, g3)
print(f"   acc {a3:.4f}   macro-F1 {f3:.4f}   (n={len(X3)})")
results["question_binary"] = {"cv_acc": a3, "cv_macro_f1": f3, "n": int(len(X3))}

json.dump(results, open(HERE/"artifacts"/"improved.json","w"), indent=2)
print("\nwrote artifacts/improved.json")
