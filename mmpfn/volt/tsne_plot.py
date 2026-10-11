"""t-SNE of clean vs triggered test-sample embeddings (BAPLe Fig. 3 style), from the .npz dumps run_baple writes
with MMPFN_DUMP_EMB. Usage: python -m mmpfn.volt.tsne_plot <dump1.npz> <dump2.npz> ... -o fig.png"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

out = sys.argv[sys.argv.index("-o")+1] if "-o" in sys.argv else "tsne.png"
skip = {"-o", out}
files=[a for a in sys.argv[1:] if not a.startswith("-") and a != out]
ncol=3; nrow=(len(files)+ncol-1)//ncol
fig,axes=plt.subplots(nrow,ncol,figsize=(5*ncol,4.6*nrow),squeeze=False)
for ax in axes.flat: ax.axis("off")
for i,f in enumerate(files):
    d=np.load(f, allow_pickle=True)
    cl,tr,y,tgt=d["clean"],d["trig"],d["y"],int(d["target"])
    X=np.concatenate([cl,tr]); 
    Z=TSNE(n_components=2,init="pca",perplexity=30,random_state=0).fit_transform(X)
    zc,zt=Z[:len(cl)],Z[len(cl):]
    ax=axes.flat[i]; ax.axis("on")
    classes=sorted(set(y.tolist())); cmap=plt.cm.tab10
    for ci,c in enumerate(classes):
        m=y==c
        ax.scatter(zc[m,0],zc[m,1],s=8,color=cmap(ci%10),alpha=0.5,label=f"class {c}" + (" (target)" if c==tgt else ""))
    # triggered points (non-target true label) as black x -> should land in the target cluster
    nt=y!=tgt
    ax.scatter(zt[nt,0],zt[nt,1],s=14,marker="x",color="black",alpha=0.7,label="triggered")
    ax.set_title(f"{str(d['backbone'])} / {str(d['dataset'])}  (target={tgt})",fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    if i==0: ax.legend(fontsize=6,markerscale=1.2,loc="best")
plt.tight_layout(); plt.savefig(out,dpi=150,bbox_inches="tight"); print("saved",out)
