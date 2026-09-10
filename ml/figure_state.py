import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
d=json.loads((HERE/"artifacts"/"state_conditional.json").read_text())
INK,GAIN,REGRESS,MARK="#14201F","#1F6F5C","#8E2C4E","#B47C14"
PAPER,RULE,INK3="#F7F8F5","#C7CDC4","#6B7C82"
plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":PAPER,"axes.facecolor":"white",
 "axes.edgecolor":RULE,"text.color":INK,"xtick.color":INK3,"ytick.color":INK3,
 "axes.grid":True,"grid.color":RULE,"grid.alpha":0.5,"font.size":9})

fig,axes=plt.subplots(1,2,figsize=(11.5,4.4))
states=["stuck","middling","moving"]
xlab=["Stuck","Middling","Moving"]
names={"question":"Open question","reflection":"Reflection",
       "therapist_input":"Advice / information","other":"Other"}
cols={"question":GAIN,"reflection":MARK,"therapist_input":REGRESS,"other":INK3}

ax=axes[0]
for f in d["change_talk_by_intervention_and_state"]:
    k=f["intervention"]; ys=[f["by_state"][s]*100 for s in states]
    ax.plot(xlab,ys,marker="s",ms=7,lw=2.4,color=cols[k],label=names[k],
            markeredgecolor=INK,markeredgewidth=0.8)
    ax.annotate(f"{ys[-1]:.0f}%",(2,ys[-1]),xytext=(6,0),textcoords="offset points",
                fontsize=8.5,fontweight="bold",color=cols[k],va="center")
ax.set_ylabel("Next client turn is change talk (%)")
ax.set_xlabel("Client's state before the intervention\n(TherapyTrace process score, terciles)")
ax.set_title("The same intervention, different states",fontsize=10.5,fontweight="bold")
ax.legend(frameon=False,fontsize=8.5,loc="upper left")
ax.set_ylim(18,54)

ax=axes[1]
labels=[names[t["intervention"]] for t in d["conditional_tests"]]
stuck=[t["p_change_when_stuck"]*100 for t in d["conditional_tests"]]
moving=[t["p_change_when_moving"]*100 for t in d["conditional_tests"]]
sig=[t["significant"] for t in d["conditional_tests"]]
y=np.arange(len(labels)); h=0.36
ax.barh(y-h/2,stuck,h,color=REGRESS,edgecolor=INK,linewidth=0.7,label="Client stuck")
ax.barh(y+h/2,moving,h,color=GAIN,edgecolor=INK,linewidth=0.7,label="Client already moving")
ax.set_yticks(y,labels); ax.invert_yaxis(); ax.grid(axis="y",visible=False)
ax.set_xlabel("Next client turn is change talk (%)")
ax.set_title("Amplifiers vs. inert moves",fontsize=10.5,fontweight="bold")
ax.legend(frameon=False,fontsize=8.5,loc="lower right")
for i,(s,m,g) in enumerate(zip(stuck,moving,sig)):
    ax.text(max(s,m)+1.2,i,f"+{m-s:.0f} pts {'***' if g else 'n.s.'}",
            va="center",fontsize=8.5,fontweight="bold",
            color=GAIN if g else INK3)
ax.set_xlim(0,62)
fig.suptitle("Therapist intervention effect is conditional on client state — 1,815 turn triples, 123 real conversations",
             fontsize=11,fontweight="bold")
fig.tight_layout()
fig.savefig(HERE/"artifacts"/"state_conditional.png",dpi=200)
print("ok")
