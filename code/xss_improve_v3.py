"""
================================================================
 XSS Study A — rigor & deployment upgrades (for Q2/Q3)
   1) Multi-seed evaluation (mean +/- std) for the strong models
   2) Imbalance-strategy ablation on LightGBM
   3) Inference-latency measurement
   4) Compact top-k SHAP-feature model (deployable) vs full 67-feature

 Dataset: XSS_dataset1.csv (67 features, 100k:1k)
 Run: python xss_improve_v3.py
================================================================
"""
import os, time, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef, precision_recall_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

OUT = "output_v3"; os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 7, 123, 2024, 99]
DATA = "XSS_dataset1.csv"


def make_splits(seed):
    df = pd.read_csv(DATA)
    fc = [c for c in df.columns if c != "Class"]
    X = df[fc].values.astype(np.float32); y = df["Class"].values.astype(int)
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.125, stratify=ytv, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    return (sc.transform(Xtr).astype(np.float32), ytr,
            sc.transform(Xva).astype(np.float32), yva,
            sc.transform(Xte).astype(np.float32), yte, fc)


def thr_val(yv, pv):
    pr, rc, th = precision_recall_curve(yv, pv)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[int(np.nanargmax(f1[:-1]))]) if len(th) else 0.5


def mets(yt, p, thr):
    yh = (p >= thr).astype(int)
    return dict(AUPRC=average_precision_score(yt, p),
                F1=f1_score(yt, yh, zero_division=0),
                MCC=matthews_corrcoef(yt, yh),
                Precision=precision_score(yt, yh, zero_division=0),
                Recall=recall_score(yt, yh))


def build_models(ytr, seed):
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                   n_jobs=-1, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.1,
                                 subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
                                 eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=seed),
        "LightGBM": LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                                   subsample=0.9, colsample_bytree=0.9, class_weight="balanced",
                                   n_jobs=-1, random_state=seed, verbose=-1),
    }


# ---------- 1) Multi-seed evaluation ----------
def multiseed():
    print("\n[1] Multi-seed evaluation (%d seeds)" % len(SEEDS))
    acc = {}
    for seed in SEEDS:
        Xtr, ytr, Xva, yva, Xte, yte, _ = make_splits(seed)
        for name, clf in build_models(ytr, seed).items():
            clf.fit(Xtr, ytr)
            thr = thr_val(yva, clf.predict_proba(Xva)[:, 1])
            m = mets(yte, clf.predict_proba(Xte)[:, 1], thr)
            acc.setdefault(name, []).append(m)
        print(f"  seed {seed} done")
    rows = []
    print(f"\n  {'Model':<22}{'AUPRC':>16}{'MCC':>16}{'F1':>16}")
    for name, lst in acc.items():
        r = {"Model": name}
        for k in ["AUPRC", "MCC", "F1", "Precision", "Recall"]:
            v = np.array([d[k] for d in lst])
            r[k+"_mean"] = round(float(v.mean()), 4); r[k+"_std"] = round(float(v.std()), 4)
        rows.append(r)
        print(f"  {name:<22}{r['AUPRC_mean']:.4f}+/-{r['AUPRC_std']:.4f}   "
              f"{r['MCC_mean']:.4f}+/-{r['MCC_std']:.4f}   {r['F1_mean']:.4f}+/-{r['F1_std']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "multiseed.csv"), index=False)
    return rows


# ---------- 2) Imbalance-strategy ablation (LightGBM) ----------
def imbalance_ablation():
    print("\n[2] Imbalance-strategy ablation (LightGBM, %d seeds)" % len(SEEDS))
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    from imblearn.under_sampling import RandomUnderSampler
    strategies = ["none", "class_weight", "SMOTE", "SMOTE-Tomek", "undersample"]
    acc = {s: [] for s in strategies}
    for seed in SEEDS:
        Xtr, ytr, Xva, yva, Xte, yte, _ = make_splits(seed)
        for s in strategies:
            cw = "balanced" if s == "class_weight" else None
            Xt, yt = Xtr, ytr
            if s == "SMOTE":
                Xt, yt = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
            elif s == "SMOTE-Tomek":
                Xt, yt = SMOTETomek(random_state=seed).fit_resample(Xtr, ytr)
            elif s == "undersample":
                Xt, yt = RandomUnderSampler(random_state=seed).fit_resample(Xtr, ytr)
            clf = LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9, class_weight=cw,
                                 n_jobs=-1, random_state=seed, verbose=-1)
            clf.fit(Xt, yt)
            thr = thr_val(yva, clf.predict_proba(Xva)[:, 1])
            acc[s].append(mets(yte, clf.predict_proba(Xte)[:, 1], thr))
    rows = []
    print(f"\n  {'Strategy':<16}{'AUPRC':>16}{'MCC':>16}{'F1':>16}")
    for s in strategies:
        r = {"Strategy": s}
        for k in ["AUPRC", "MCC", "F1", "Precision", "Recall"]:
            v = np.array([d[k] for d in acc[s]])
            r[k+"_mean"] = round(float(v.mean()), 4); r[k+"_std"] = round(float(v.std()), 4)
        rows.append(r)
        print(f"  {s:<16}{r['AUPRC_mean']:.4f}+/-{r['AUPRC_std']:.4f}   "
              f"{r['MCC_mean']:.4f}+/-{r['MCC_std']:.4f}   {r['F1_mean']:.4f}+/-{r['F1_std']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "imbalance_ablation.csv"), index=False)
    return rows


# ---------- 3) Latency + 4) compact top-k model ----------
def latency_and_compact():
    print("\n[3+4] Inference latency & compact top-k SHAP model")
    import shap
    Xtr, ytr, Xva, yva, Xte, yte, fc = make_splits(42)
    full = LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, class_weight="balanced",
                          n_jobs=-1, random_state=42, verbose=-1).fit(Xtr, ytr)
    # SHAP importance -> top-k
    expl = shap.TreeExplainer(full)
    sv = expl.shap_values(Xte[:2000])
    if isinstance(sv, list): sv = sv[1]
    imp = np.abs(sv).mean(0)
    order = np.argsort(imp)[::-1]
    rows = []
    for k in [67, 20, 15, 10, 5]:
        idx = order[:k]
        clf = LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                             subsample=0.9, colsample_bytree=0.9, class_weight="balanced",
                             n_jobs=-1, random_state=42, verbose=-1).fit(Xtr[:, idx], ytr)
        thr = thr_val(yva, clf.predict_proba(Xva[:, idx])[:, 1])
        pte = clf.predict_proba(Xte[:, idx])[:, 1]
        m = mets(yte, pte, thr)
        # latency: per-sample predict_proba over test set
        t0 = time.time()
        for _ in range(3):
            clf.predict_proba(Xte[:, idx])
        lat_us = (time.time() - t0) / 3 / len(Xte) * 1e6
        rows.append({"k_features": k, "AUPRC": round(m["AUPRC"], 4),
                     "MCC": round(m["MCC"], 4), "F1": round(m["F1"], 4),
                     "latency_us_per_sample": round(lat_us, 2)})
        print(f"  top-{k:<3} feats: AUPRC={m['AUPRC']:.4f} MCC={m['MCC']:.4f} "
              f"F1={m['F1']:.4f}  latency={lat_us:.2f} us/sample")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "compact_latency.csv"), index=False)
    top = [fc[i] for i in order[:15]]
    json.dump(top, open(os.path.join(OUT, "top15_features.json"), "w"), indent=2)
    print("  top-15 features:", ", ".join(top[:8]), "...")
    return rows


if __name__ == "__main__":
    print("=" * 60); print("  XSS Study A — rigor & deployment upgrades"); print("=" * 60)
    r1 = multiseed()
    r2 = imbalance_ablation()
    r3 = latency_and_compact()
    print("\nDone. Outputs in", OUT)
