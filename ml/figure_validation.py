import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "therapytrace" / "backend"))
from app.nlp.features import DIMENSIONS, session_features, score_utterance

INK, GAIN, REGRESS, MARK = "#14201F", "#1F6F5C", "#8E2C4E", "#B47C14"
PAPER, RULE, INK3 = "#F7F8F5", "#C7CDC4", "#6B7C82"
plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":PAPER,"axes.facecolor":"white",
    "axes.edgecolor":RULE,"text.color":INK,"xtick.color":INK3,"ytick.color":INK3,
    "axes.grid":True,"grid.color":RULE,"grid.alpha":0.5,"font.size":9})

v = json.loads((HERE/"artifacts"/"validation_annomi.json").read_text())
df = pd.read_csv(HERE/"data"/"AnnoMI-full.csv")
df = df[(df.interlocutor=="client") & df.client_talk_type.notna() & df.utterance_text.notna()]
df = df[df.utterance_text.astype(str).str.split().str.len()>=5]

rows=[]
for tid,g in df.groupby("transcript_id"):
    nch=int((g.client_talk_type=="change").sum()); nsu=int((g.client_talk_type=="sustain").sum())
    if nch+nsu<5: continue
    f,_=session_features(list(g.utterance_text.astype(str)))
    rows.append({"ratio":nch/(nch+nsu),"pm":float(np.mean([f[d] for d in DIMENSIONS]))})
c=pd.DataFrame(rows)

fig,axes=plt.subplots(1,2,figsize=(11,4.2))

# left: effect sizes
ax=axes[0]
dims=list(DIMENSIONS)
ds=[v["utterance_level"][d]["cohens_d"] for d in dims]
ps=[v["utterance_level"][d]["mannwhitney_p"] for d in dims]
cols=[GAIN if p<0.05 else INK3 for p in ps]
names=[d.replace("_"," ").title() for d in dims]
b=ax.barh(names,ds,color=cols,edgecolor=INK,linewidth=0.8,height=0.6)
ax.axvline(0,color=INK,lw=1)
ax.invert_yaxis(); ax.grid(axis="y",visible=False)
ax.set_xlabel("Cohen's d  (change talk − sustain talk)")
ax.set_title("Expert-labelled utterances\n5/5 dimensions ordered correctly",fontsize=10,fontweight="bold")
ax.set_xlim(-0.05,max(ds)*1.35)
for bar,d,p in zip(b,ds,ps):
    star="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
    ax.text(d+max(ds)*0.03,bar.get_y()+bar.get_height()/2,f"{d:.3f} {star}",va="center",fontsize=8.5,fontweight="bold")

# right: scatter
ax=axes[1]
ax.scatter(c.ratio,c.pm,s=34,c=GAIN,alpha=0.65,edgecolor=INK,linewidth=0.5)
m,bb=np.polyfit(c.ratio,c.pm,1); xs=np.linspace(c.ratio.min(),c.ratio.max(),50)
ax.plot(xs,m*xs+bb,color=REGRESS,lw=2)
r,p=stats.pearsonr(c.pm,c.ratio)
ax.set_xlabel("Expert change-talk ratio  (change / [change+sustain])")
ax.set_ylabel("TherapyTrace process score")
ax.set_title(f"93 real therapy conversations\nr = {r:.3f},  p = {p:.1e}",fontsize=10,fontweight="bold")
fig.suptitle("Convergent validation against AnnoMI expert annotations",fontsize=11.5,fontweight="bold")
fig.tight_layout()
fig.savefig(HERE/"artifacts"/"validation.png",dpi=200)
print("ok")
