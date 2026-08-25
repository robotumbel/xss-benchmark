"""Regenerate the 5 manuscript figures as black-and-white / grayscale-safe versions
for the Informatics and Automation (SPIIRAS) journal (color not allowed).
Outputs overwrite: paper q4/figs/*.png
"""
import os, json, itertools, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
warnings.filterwarnings("ignore")

V2 = "output_v2"
OUT = r"D:/Code-PhD/Deteksi XXS/paper q4/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.35,
                     "grid.color": "0.7", "figure.dpi": 200,
                     "axes.edgecolor": "black", "savefig.facecolor": "white"})

results = json.load(open(os.path.join(V2, "results.json")))
pdir = os.path.join(V2, "preds")
y = np.load(os.path.join(pdir, "_y_test.npy"))
def load(n): return np.load(os.path.join(pdir, n.replace(" ", "_") + ".npy"))

order = sorted(results, key=lambda k: results[k]["AUPRC"], reverse=True)

# B&W style cycles
LS = ['-', '--', '-.', ':']
MK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
GRAY = ['0.0', '0.45', '0.0', '0.45', '0.0', '0.45', '0.0', '0.45', '0.0', '0.45']

# ---- FIG 1: PR curves (grayscale, linestyle+marker per model) ----
plt.figure(figsize=(7, 6))
for i, n in enumerate(order):
    if not os.path.exists(os.path.join(pdir, n.replace(" ", "_") + ".npy")):
        continue
    p = load(n)
    pr, rc, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    lw = 2.4 if i == 0 else 1.3
    plt.plot(rc, pr, ls=LS[i % 4], color=GRAY[i % len(GRAY)], lw=lw,
             marker=MK[i % len(MK)], markevery=max(len(rc)//10, 1), markersize=4,
             markerfacecolor='white', label=f"{n} (AP={ap:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall curves (100:1 imbalance)")
plt.ylim(0, 1.02); plt.legend(fontsize=7.5, loc="lower left")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_pr_curves.png")); plt.close()

# ---- FIG 2: metric comparison (grayscale + hatches) ----
mets = ["AUPRC", "F1", "MCC", "Precision", "Recall"]
mo = [n for n in order if os.path.exists(os.path.join(pdir, n.replace(" ", "_") + ".npy"))]
hatches = ['', '///', '...', 'xxx', '\\\\\\', '++', 'ooo', '--', '||', '**']
grays = np.linspace(0.15, 0.9, len(mo))
x = np.arange(len(mets)); w = 0.8 / len(mo)
plt.figure(figsize=(11, 6))
for i, n in enumerate(mo):
    vals = [results[n][m] for m in mets]
    plt.bar(x + i*w, vals, w, label=n, facecolor=str(grays[i]),
            edgecolor='black', linewidth=0.6, hatch=hatches[i % len(hatches)])
plt.xticks(x + w*len(mo)/2, mets); plt.ylim(0, 1.05)
plt.ylabel("Score"); plt.title("Model comparison (imbalance-aware metrics)")
plt.legend(fontsize=7.5, ncol=3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_metric_comparison.png")); plt.close()

# ---- FIG 3: confusion matrices top-3 (Greys) ----
top3 = mo[:3]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, n in zip(axes, top3):
    thr = results[n]["threshold"]
    cm = confusion_matrix(y, (load(n) >= thr).astype(int))
    ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max())
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, f"{v:,}", ha="center", va="center", fontsize=12,
                color="white" if v > cm.max()/2 else "black")
    ax.set_title(f"{n}\nF1={results[n]['F1']:.3f}  MCC={results[n]['MCC']:.3f}")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "XSS"]); ax.set_yticklabels(["Benign", "XSS"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_confusion.png")); plt.close()

# ---- FIG 4: SHAP mean|value| bar (grayscale) ----
try:
    import shap
    from xss_experiment_v2 import load_splits, sklearn_models
    D = load_splits()
    tname = [m for m in ["LightGBM", "Random Forest", "XGBoost"] if m in results][0]
    clf = sklearn_models(D["y_tr"])[tname]; clf.fit(D["X_tr"], D["y_tr"])
    expl = shap.TreeExplainer(clf)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(D["X_te"]), min(2000, len(D["X_te"])), replace=False)
    sv = expl.shap_values(D["X_te"][idx])
    if isinstance(sv, list):
        sv = sv[1]
    imp = np.abs(sv).mean(0)
    o = np.argsort(imp)[::-1][:20][::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(range(len(o)), imp[o], color="0.35", edgecolor="black", linewidth=0.6)
    plt.yticks(range(len(o)), [D["feat_cols"][k] for k in o], fontsize=8)
    plt.xlabel("Mean |SHAP value| (impact on model output)")
    plt.title(f"Feature importance (SHAP) - {tname}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_shap_summary.png")); plt.close()
    print("  fig4 SHAP (bar, grayscale) done")
except Exception as e:
    print("  [fig4 SHAP skipped]", repr(e))

# ---- FIG 5: reliability diagram (grayscale, markers) ----
def ece(yt, p, nb=10):
    bins = np.linspace(0, 1, nb+1); e = 0.0
    for i in range(nb):
        m = (p > bins[i]) & (p <= bins[i+1])
        if m.sum():
            e += (m.sum()/len(p)) * abs(yt[m].mean() - p[m].mean())
    return e
cal_models = [m for m in ["Random Forest", "LightGBM", "XGBoost", "MLP", "BiLSTM"]
              if os.path.exists(os.path.join(pdir, m.replace(" ", "_") + ".npy"))]
plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], color="0.5", ls=":", lw=1.2, label="Perfect calibration")
for i, n in enumerate(cal_models):
    p = load(n); bins = np.linspace(0, 1, 11); xs, ys = [], []
    for b in range(10):
        m = (p > bins[b]) & (p <= bins[b+1])
        if m.sum():
            xs.append(p[m].mean()); ys.append(y[m].mean())
    plt.plot(xs, ys, ls=LS[i % 4], color=GRAY[i % len(GRAY)], lw=1.6,
             marker=MK[i % len(MK)], markersize=6, markerfacecolor='white',
             label=f"{n} (ECE={ece(y, p):.3f})")
plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraction positive")
plt.title("Reliability diagram (Study A, test set)")
plt.legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig5_calibration.png")); plt.close()

print("B&W figures written to:", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
