"""
Paralinguistic analysis without audio files.

AnnoMI timestamps each utterance, so speaking *rate* is recoverable directly:
    duration  = next utterance's start - this utterance's start
    rate      = words / duration

Rate is a genuine paralinguistic feature. Slowed, effortful speech is
associated in clinical linguistics with deliberate processing; rapid speech
with activation or rehearsed narrative. The question here is a process
question, not a symptom one:

    Does HOW FAST a client speaks carry information about WHETHER they are
    doing therapeutic work, over and above WHAT they say?

If rate adds predictive value on top of the text model, that is the empirical
case for a full multimodal tier. If it does not, that is worth knowing before
anyone builds one.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "therapytrace" / "backend"))
from app.nlp.features import score_utterance

df = pd.read_csv(HERE / "data" / "AnnoMI-full.csv")
df = df.sort_values(["transcript_id", "utterance_id"]).reset_index(drop=True)

def secs(t):
    try:
        h, m, s = str(t).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return np.nan

df["t"] = df.timestamp.map(secs)
df["dur"] = df.groupby("transcript_id").t.shift(-1) - df.t
df["nwords"] = df.utterance_text.astype(str).str.split().str.len()

c = df[(df.interlocutor == "client") & df.client_talk_type.notna()].copy()
c = c[(c.dur >= 2) & (c.dur <= 120) & (c.nwords >= 8)]
c["rate"] = c.nwords / c.dur
c = c[(c.rate > 0.4) & (c.rate < 6)]
print(f"client utterances with usable timing: {len(c)}  ({c.transcript_id.nunique()} conversations)")

print("\n" + "=" * 70)
print("Speech rate (words/sec) by expert label")
print("=" * 70)
for lab in ["change", "neutral", "sustain"]:
    g = c[c.client_talk_type == lab].rate
    print(f"  {lab:<10} n={len(g):>5}   mean={g.mean():.3f}   median={g.median():.3f}   sd={g.std():.3f}")

ch = c[c.client_talk_type == "change"].rate
su = c[c.client_talk_type == "sustain"].rate
pooled = np.sqrt(((len(ch)-1)*ch.var()+(len(su)-1)*su.var())/(len(ch)+len(su)-2))
d = (ch.mean()-su.mean())/pooled
u, p = stats.mannwhitneyu(ch, su, alternative="two-sided")
print(f"\n  change vs sustain:  Cohen's d = {d:+.3f}   p = {p:.3e}")

# does rate add anything beyond the text-derived process score?
c["process"] = [score_utterance(str(t)).process_mean for t in c.utterance_text]
r_proc, p_proc = stats.pointbiserialr((c.client_talk_type == "change").astype(int), c.process)
r_rate, p_rate = stats.pointbiserialr((c.client_talk_type == "change").astype(int), c.rate)
print(f"\n  correlation with change talk:")
print(f"    text process score   r = {r_proc:+.4f}  (p = {p_proc:.2e})")
print(f"    speech rate          r = {r_rate:+.4f}  (p = {p_rate:.2e})")
print(f"    rate vs process      r = {stats.pearsonr(c.rate, c.process)[0]:+.4f}   <- overlap between the two")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

y = (c.client_talk_type == "change").astype(int).values
groups = c.transcript_id.values
sets = {"text only": ["process"], "rate only": ["rate"], "text + rate": ["process", "rate"]}
print(f"\n  incremental value (AUC, 5-fold grouped by conversation):")
aucs = {}
for name, cols in sets.items():
    X = StandardScaler().fit_transform(c[cols].values)
    scores = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    aucs[name] = float(np.mean(scores))
    print(f"    {name:<14} AUC = {np.mean(scores):.4f} ± {np.std(scores):.3f}")
gain = aucs["text + rate"] - aucs["text only"]
print(f"\n  incremental AUC from adding speech rate: {gain:+.4f}")

json.dump({
    "n": int(len(c)), "n_conversations": int(c.transcript_id.nunique()),
    "rate_by_label": {l: round(float(c[c.client_talk_type==l].rate.mean()),4)
                      for l in ["change","neutral","sustain"]},
    "cohens_d_change_vs_sustain": round(float(d),4), "p": float(p),
    "r_process": round(float(r_proc),4), "r_rate": round(float(r_rate),4),
    "auc": {k: round(v,4) for k,v in aucs.items()},
    "incremental_auc": round(gain,4),
}, open(HERE/"artifacts"/"prosody.json","w"), indent=2)
print("\nwrote artifacts/prosody.json")
