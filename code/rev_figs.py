"""Two additional grayscale-safe manuscript figures for the revised paper.

fig6 - AUPRC and MCC as a function of the benign:XSS ratio (R2),
fig7 - detection recall under the three evasion tiers for every model family
       on raw payloads (R7/R8).
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figs")
REV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "rev_out")
os.makedirs(OUT_FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.35,
                     "grid.color": "0.7", "figure.dpi": 200,
                     "axes.edgecolor": "black", "savefig.facecolor": "white"})

LS = ["-", "--", "-.", ":", "-", "--"]
MK = ["o", "s", "^", "D", "v", "P"]
GRAY = ["0.0", "0.35", "0.0", "0.35", "0.55", "0.55"]


def fig6():
    df = pd.read_csv(os.path.join(REV, "r2_imbalance_sweep_summary.csv"))
    models = ["Random Forest", "LightGBM", "XGBoost", "MLP", "GRU-1L",
              "Logistic Regression"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for metric, ax in zip(("AUPRC", "MCC"), axes):
        for i, m in enumerate(models):
            sub = df[df["model"] == m].sort_values("ratio")
            ax.errorbar(sub["ratio"], sub[f"{metric}_mean"],
                        yerr=sub[f"{metric}_std"], label=m,
                        color=GRAY[i], linestyle=LS[i], marker=MK[i],
                        markersize=5, linewidth=1.4, capsize=2.5)
        ax.set_xscale("log")
        ax.set_xticks([100, 250, 500, 1000, 2000, 5000])
        ax.set_xticklabels(["100:1", "250:1", "500:1", "1000:1", "2000:1",
                            "5000:1"], rotation=30)
        ax.set_xlabel("benign : XSS ratio")
        ax.set_ylabel(f"test {metric}")
        ax.set_ylim(0.2, 1.02)
    axes[0].legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    p = os.path.join(OUT_FIG, "fig6_imbalance_sweep.png")
    fig.savefig(p, bbox_inches="tight")
    print("saved", p)


def fig7():
    df = pd.read_csv(os.path.join(REV, "r7_modern_architectures.csv"))
    labels = ["clean", "escape-based\nobfuscation",
              "artifact-suppressing\nevasion"]
    models = list(df["model"])
    x = np.arange(len(labels))
    w = 0.8 / len(models)
    hatches = ["", "//", "\\\\", "xx", "..", "++", "oo"]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    for i, m in enumerate(models):
        r = df[df["model"] == m].iloc[0]
        vals = [r["Recall_clean"], r["Recall_obf"], r["Recall_tier3"]]
        errs = [r.get("Recall_std", 0) or 0, r.get("Recall_obf_std", 0) or 0,
                r.get("Recall_tier3_std", 0) or 0]
        errs = [0 if (e != e) else e for e in errs]
        ax.bar(x + (i - len(models) / 2 + 0.5) * w, vals, w, label=m,
               yerr=errs, capsize=2.5, error_kw={"linewidth": 0.8},
               color=str(0.28 + 0.11 * i), edgecolor="black", linewidth=0.6,
               hatch=hatches[i % len(hatches)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("recall on malicious payloads")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, frameon=False)
    fig.tight_layout()
    p = os.path.join(OUT_FIG, "fig7_evasion_tiers.png")
    fig.savefig(p, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    fig6()
    fig7()
