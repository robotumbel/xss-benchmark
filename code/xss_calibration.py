"""
Calibration analysis for Study A (uses saved test predictions in output_v2/preds).
Computes Brier score + Expected Calibration Error (ECE) and a reliability diagram.
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

OUT = "output_v2"; FIG = os.path.join(OUT, "figs")
pred_dir = os.path.join(OUT, "preds")
y = np.load(os.path.join(pred_dir, "_y_test.npy"))
models = ["Random Forest", "LightGBM", "XGBoost", "MLP", "BiLSTM"]

def ece(y_true, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0:
            continue
        conf = p[m].mean(); acc = y_true[m].mean()
        e += (m.sum() / len(p)) * abs(acc - conf)
    return e

plt.figure(figsize=(7, 6))
plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
rows = []
for name in models:
    f = os.path.join(pred_dir, name.replace(" ", "_") + ".npy")
    if not os.path.exists(f):
        continue
    p = np.load(f)
    bs = brier_score_loss(y, p); e = ece(y, p)
    rows.append({"Model": name, "Brier": round(float(bs), 4), "ECE": round(float(e), 4)})
    # reliability curve (manual, 10 bins)
    bins = np.linspace(0, 1, 11); xs, ys = [], []
    for i in range(10):
        m = (p > bins[i]) & (p <= bins[i + 1])
        if m.sum() > 0:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    plt.plot(xs, ys, "o-", label=f"{name} (ECE={e:.3f}, Brier={bs:.3f})", markersize=4)

plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraction positive")
plt.title("Reliability diagram (Study A, test set)")
plt.legend(fontsize=8, loc="upper left"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig5_calibration.png"), dpi=150)
plt.close()

import pandas as pd
pd.DataFrame(rows).to_csv(os.path.join(OUT, "calibration.csv"), index=False)
print("Calibration (Brier / ECE):")
for r in rows:
    print(f"  {r['Model']:<16} Brier={r['Brier']:.4f}  ECE={r['ECE']:.4f}")
print("Saved fig5_calibration.png + calibration.csv")
