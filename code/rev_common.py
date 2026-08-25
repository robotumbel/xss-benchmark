"""Shared helpers for the revision experiments (R1-R9).

Mirrors the protocol of xss_experiment_v2.py so that every new result is
produced under exactly the same leakage-free rules as the original study:
stratified splits, scaler fit on train only, decision threshold selected on
validation and applied unchanged to test.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, balanced_accuracy_score,
    precision_recall_curve, confusion_matrix,
)
from scipy import stats as scipy_stats

# Directory holding XSS_dataset1.csv and data_raw/. Defaults to two levels
# above this file, which is the layout of the authors' working tree; set
# XSS_DATA_DIR to point at the datasets anywhere else.
ROOT = os.environ.get(
    "XSS_DATA_DIR",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_A = os.path.join(ROOT, "XSS_dataset1.csv")
DATA_RAW_A = os.path.join(ROOT, "data_raw", "xss_payloads_fmereani.csv")
DATA_RAW_B = os.path.join(ROOT, "data_raw", "xss_syedsaqlain.csv")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rev_out")
os.makedirs(OUT, exist_ok=True)

SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.10


def load_structural(seed=SEED, drop_duplicates=False, pos_ratio=None):
    """Load the 67-feature structural dataset.

    pos_ratio: if given, subsample the positive class so that the resulting
    negative:positive ratio equals pos_ratio (used for the imbalance sweep).
    """
    df = pd.read_csv(DATA_A)
    info = {"n_raw": len(df)}
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    info["n_used"] = len(df)

    feat_cols = [c for c in df.columns if c != "Class"]
    X = df[feat_cols].values.astype(np.float32)
    y = df["Class"].values.astype(np.int64)

    if pos_ratio is not None:
        rng = np.random.RandomState(seed)
        neg_idx = np.where(y == 0)[0]
        pos_idx = np.where(y == 1)[0]
        keep_pos = max(int(round(len(neg_idx) / pos_ratio)), 5)
        keep_pos = min(keep_pos, len(pos_idx))
        pos_idx = rng.choice(pos_idx, keep_pos, replace=False)
        sel = np.sort(np.concatenate([neg_idx, pos_idx]))
        X, y = X[sel], y[sel]
        info["ratio"] = len(neg_idx) / keep_pos
        info["n_pos"] = keep_pos

    X_tv, X_te, y_tv, y_te, idx_tv, idx_te = train_test_split(
        X, y, np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=seed)
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_tr, X_va, y_tr, y_va, idx_tr, idx_va = train_test_split(
        X_tv, y_tv, idx_tv, test_size=val_ratio, stratify=y_tv, random_state=seed)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_va_s, X_te_s = (scaler.transform(a).astype(np.float32)
                              for a in (X_tr, X_va, X_te))
    return dict(X_tr=X_tr_s, y_tr=y_tr, X_va=X_va_s, y_va=y_va,
                X_te=X_te_s, y_te=y_te, feat_cols=feat_cols,
                X_tr_raw=X_tr, X_va_raw=X_va, X_te_raw=X_te,
                idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te, info=info)


def best_threshold_on_val(y_val, p_val):
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    best_i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[best_i]) if len(thr) else 0.5


def metrics_at(y_true, p, thr):
    y_hat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "AUPRC": float(average_precision_score(y_true, p)),
        "ROC_AUC": float(roc_auc_score(y_true, p)),
        "F1": float(f1_score(y_true, y_hat, zero_division=0)),
        "Precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "Recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_hat)),
        "BalAcc": float(balanced_accuracy_score(y_true, y_hat)),
        "threshold": float(thr),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def mcnemar(y_true, pred_a, pred_b):
    """Exact-corrected McNemar test on the discordant pairs."""
    b = int(np.sum((pred_a == y_true) & (pred_b != y_true)))
    c = int(np.sum((pred_a != y_true) & (pred_b == y_true)))
    if b + c == 0:
        return 0.0, 1.0, b, c
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(stat), float(1 - scipy_stats.chi2.cdf(stat, df=1)), b, c


def holm_bonferroni(pvals):
    """Holm step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def tree_models(y_tr, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    spw = n_neg / max(n_pos, 1)
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed),
        "XGBoost": XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.1, subsample=0.9,
            colsample_bytree=0.9, scale_pos_weight=spw, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=seed),
        "LightGBM": LGBMClassifier(
            n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, class_weight="balanced", n_jobs=-1,
            random_state=seed, verbose=-1),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=seed),
    }


def fit_eval_sklearn(clf, D):
    t0 = time.time()
    clf.fit(D["X_tr"], D["y_tr"])
    elapsed = time.time() - t0
    p_va = clf.predict_proba(D["X_va"])[:, 1]
    p_te = clf.predict_proba(D["X_te"])[:, 1]
    thr = best_threshold_on_val(D["y_va"], p_va)
    m = metrics_at(D["y_te"], p_te, thr)
    m["time_s"] = round(elapsed, 2)
    return m, p_te


def save(df, name):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(f"  saved -> {path}")
    return path
