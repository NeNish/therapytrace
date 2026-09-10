"""
Paralinguistic tier, done properly.

The first attempt tested RAW speech rate and found nothing. That test was
wrong, and wrong in a way this project should have caught: raw rate is a
between-person quantity. A client who habitually speaks at 4 words/sec and one
who speaks at 2 are not comparable, exactly as their vocabularies are not.

The same correction the TPI applies to lexical features applies here:
standardise every timing feature against that client's own baseline. If the
within-person version carries signal where the raw version did not, that is
both a working paralinguistic tier AND independent evidence for the central
methodological claim of the whole system.

Features (all derivable from timestamps, no audio required):
    dur              utterance duration
    rate             words per second
    pause_before     gap left by the previous speaker's turn
    dur_z, rate_z    the same, z-scored within client
    rate_dev         |rate - client's own median rate|   (departure either way)
    rate_delta       change from this client's previous turn
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "therapytrace" / "backend"))
from app.nlp.features import score_utterance

df = pd.read_csv(HERE / "data" / "AnnoMI-full.csv").sort_values(
    ["transcript_id", "utterance_id"]).reset_index(drop=True)

def secs(t):
    try:
        h, m, s = str(t).split(":"); return int(h)*3600 + int(m)*60 + int(s)
    except Exception: return np.nan

df["t"] = df.timestamp.map(secs)
df["dur"] = df.groupby("transcript_id").t.shift(-1) - df.t
df["nwords"] = df.utterance_text.astype(str).str.split().str.len()
df["prev_dur"] = df.groupby("transcript_id").dur.shift(1)
df["prev_words"] = df.groupby("transcript_id").nwords.shift(1)
# how much of the previous speaker's slot was NOT speech -> approximate pause
df["pause_before"] = df.prev_dur - df.prev_words / 3.0

c = df[(df.interlocutor == "client") & df.client_talk_type.notna()].copy()
c = c[(c.dur >= 2) & (c.dur <= 180) & (c.nwords >= 8)]
c["rate"] = c.nwords / c.dur
c = c[(c.rate > 0.3) & (c.rate < 7)]

# ---- within-person standardisation, the whole point -----------------------
g = c.groupby("transcript_id")
for col in ["rate", "dur", "nwords"]:
    mu = g[col].transform("mean"); sd = g[col].transform("std").replace(0, np.nan)
    c[f"{col}_z"] = ((c[col] - mu) / sd).fillna(0)
c["rate_dev"] = (c.rate - g.rate.transform("median")).abs()
c["rate_delta"] = g.rate.diff().fillna(0)
c["pause_z"] = ((c.pause_before - g.pause_before.transform("mean"))
                / g.pause_before.transform("std").replace(0, np.nan)).fillna(0)
c["process"] = [score_utterance(str(t)).process_mean for t in c.utterance_text]

print(f"n = {len(c)} client utterances, {c.transcript_id.nunique()} conversations")
y = (c.client_talk_type == "change").astype(int).values
groups = c.transcript_id.values

print("\n" + "="*72)
print("Raw vs within-person timing features — correlation with change talk")
print("="*72)
for f in ["rate","dur","nwords","rate_z","dur_z","nwords_z","rate_dev","rate_delta","pause_z"]:
    r, p = stats.pointbiserialr(y, c[f].values)
    tag = "within-person" if f.endswith(("_z","_dev","_delta")) else "raw"
    star = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    print(f"  {f:<12} ({tag:<13}) r = {r:+.4f}   p = {p:.2e} {star}")

TIMING_RAW = ["rate","dur","nwords","pause_before"]
TIMING_WP  = ["rate_z","dur_z","nwords_z","rate_dev","rate_delta","pause_z"]

def evaluate(cols, model="logreg", name=""):
    X = c[cols].replace([np.inf,-np.inf], np.nan).fillna(0).values
    X = StandardScaler().fit_transform(X)
    aucs=[]
    for tr,te in GroupKFold(n_splits=5).split(X,y,groups):
        m = (LogisticRegression(max_iter=2000, class_weight="balanced")
             if model=="logreg" else
             HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                            max_depth=4, random_state=42))
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:,1]))
    print(f"  {name:<38} AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.3f}")
    return float(np.mean(aucs))

print("\n" + "="*72)
print("Does a paralinguistic tier add anything?  (5-fold, grouped by conversation)")
print("="*72)
res={}
res["timing_raw"]      = evaluate(TIMING_RAW, "logreg", "timing, RAW (logreg)")
res["timing_raw_gb"]   = evaluate(TIMING_RAW, "gb",     "timing, RAW (gradient boosting)")
res["timing_wp"]       = evaluate(TIMING_WP,  "logreg", "timing, WITHIN-PERSON (logreg)")
res["timing_wp_gb"]    = evaluate(TIMING_WP,  "gb",     "timing, WITHIN-PERSON (gradient boosting)")
res["text"]            = evaluate(["process"],"logreg", "text process score only")
res["text_timing_raw"] = evaluate(["process"]+TIMING_RAW,"gb","text + timing RAW (GB)")
res["text_timing_wp"]  = evaluate(["process"]+TIMING_WP, "gb","text + timing WITHIN-PERSON (GB)")

print(f"\n  raw timing vs chance          : {res['timing_raw_gb']-0.5:+.4f}")
print(f"  within-person timing vs chance: {res['timing_wp_gb']-0.5:+.4f}")
print(f"  gain from within-person norm  : {res['timing_wp_gb']-res['timing_raw_gb']:+.4f}")
print(f"  text alone                    : {res['text']:.4f}")
print(f"  text + within-person timing   : {res['text_timing_wp']:.4f}  ({res['text_timing_wp']-res['text']:+.4f})")

json.dump({"n":int(len(c)),"n_conversations":int(c.transcript_id.nunique()),
           "auc":{k:round(v,4) for k,v in res.items()},
           "within_person_gain":round(res["timing_wp_gb"]-res["timing_raw_gb"],4),
           "multimodal_gain_over_text":round(res["text_timing_wp"]-res["text"],4)},
          open(HERE/"artifacts"/"prosody_v2.json","w"), indent=2)
print("\nwrote artifacts/prosody_v2.json")
