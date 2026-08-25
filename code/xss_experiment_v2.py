"""
================================================================
 XSS Detection v2 — Honest, leakage-free, imbalance-aware study
 (rebuild for Scopus Q2/Q3 submission)

 Fixes vs original xss_rnn_detection.py:
  * Fair baselines at matched capacity (no deliberately-weak models)
  * Strong tabular baselines added: RandomForest, XGBoost, LightGBM
  * Threshold calibrated on VALIDATION, reported on TEST (no leakage)
  * Imbalance-correct metrics: AUPRC (PR-AUC), MCC, balanced acc
  * Honest model selection — data decides the winner
  * Bootstrap 95% CI on test with a FIXED (val-calibrated) threshold

 Dataset: XSS_dataset1.csv — 67 URL/HTML/JS structural features,
          100,000 benign : 1,000 XSS  (100:1 imbalance)

 Run:
   python xss_experiment_v2.py fast   # classical + trees + MLP (minutes)
   python xss_experiment_v2.py rnn    # recurrent family (slower, CPU)
   python xss_experiment_v2.py all    # everything
================================================================
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, balanced_accuracy_score,
    precision_recall_curve, confusion_matrix,
)
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

CFG = {
    "data_path":   "XSS_dataset1.csv",
    "seed":        42,
    "test_size":   0.20,
    "val_size":    0.10,   # fraction of the FULL dataset
    "output_dir":  "output_v2",
    "boot_n":      1000,
}
SEED = CFG["seed"]
np.random.seed(SEED)
os.makedirs(CFG["output_dir"], exist_ok=True)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_splits():
    df = pd.read_csv(CFG["data_path"])
    feat_cols = [c for c in df.columns if c != "Class"]
    X = df[feat_cols].values.astype(np.float32)
    y = df["Class"].values.astype(np.int64)

    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=CFG["test_size"], stratify=y, random_state=SEED)
    val_ratio = CFG["val_size"] / (1 - CFG["test_size"])
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=val_ratio, stratify=y_tv, random_state=SEED)

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = (scaler.transform(a).astype(np.float32)
                        for a in (X_tr, X_va, X_te))
    return dict(X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va,
                X_te=X_te, y_te=y_te, feat_cols=feat_cols)


# ----------------------------------------------------------------------
# Evaluation helpers  (leakage-free)
# ----------------------------------------------------------------------
def best_threshold_on_val(y_val, p_val):
    """Threshold that maximises F1 on the VALIDATION set (no test peeking)."""
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    # precision_recall_curve returns len(thr)=len(prec)-1
    best_i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[best_i]) if len(thr) else 0.5


def metrics_at(y_true, p, thr):
    y_hat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "AUPRC":     float(average_precision_score(y_true, p)),
        "ROC_AUC":   float(roc_auc_score(y_true, p)),
        "F1":        float(f1_score(y_true, y_hat, zero_division=0)),
        "Precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "Recall":    float(recall_score(y_true, y_hat, zero_division=0)),
        "MCC":       float(matthews_corrcoef(y_true, y_hat)),
        "BalAcc":    float(balanced_accuracy_score(y_true, y_hat)),
        "threshold": float(thr),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def bootstrap_ci(y_true, p, thr, metric="AUPRC", n=1000, ci=0.95):
    rng = np.random.RandomState(SEED)
    N = len(y_true); vals = []
    for _ in range(n):
        idx = rng.choice(N, N, replace=True)
        yt, pp = y_true[idx], p[idx]
        if len(np.unique(yt)) < 2:
            continue
        if metric == "AUPRC":
            vals.append(average_precision_score(yt, pp))
        elif metric == "ROC_AUC":
            vals.append(roc_auc_score(yt, pp))
        elif metric == "F1":
            vals.append(f1_score(yt, (pp >= thr).astype(int), zero_division=0))
        elif metric == "MCC":
            vals.append(matthews_corrcoef(yt, (pp >= thr).astype(int)))
    a = (1 - ci) / 2
    return float(np.percentile(vals, a * 100)), float(np.percentile(vals, (1 - a) * 100))


def mcnemar(y_true, pred_a, pred_b):
    b = int(np.sum((pred_a == y_true) & (pred_b != y_true)))
    c = int(np.sum((pred_a != y_true) & (pred_b == y_true)))
    if b + c == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(stat), float(1 - scipy_stats.chi2.cdf(stat, df=1))


# ----------------------------------------------------------------------
# Model zoo
# ----------------------------------------------------------------------
def sklearn_models(y_tr):
    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    spw = n_neg / n_pos
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=None, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=spw,
            eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED)
    except Exception as e:
        print("  [skip XGBoost]", e)
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=600, num_leaves=63, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, class_weight="balanced",
            n_jobs=-1, random_state=SEED, verbose=-1)
    except Exception as e:
        print("  [skip LightGBM]", e)
    return models


def run_sklearn(D, results, preds):
    for name, clf in sklearn_models(D["y_tr"]).items():
        t0 = time.time()
        clf.fit(D["X_tr"], D["y_tr"])
        elapsed = time.time() - t0
        p_va = clf.predict_proba(D["X_va"])[:, 1]
        p_te = clf.predict_proba(D["X_te"])[:, 1]
        thr = best_threshold_on_val(D["y_va"], p_va)
        m = metrics_at(D["y_te"], p_te, thr)
        m["time_s"] = round(elapsed, 2)
        results[name] = m
        preds[name] = p_te
        print(f"  {name:<22} AUPRC={m['AUPRC']:.4f} F1={m['F1']:.4f} "
              f"MCC={m['MCC']:.4f} P={m['Precision']:.3f} R={m['Recall']:.3f} "
              f"({elapsed:.1f}s)")


# ----------------------------------------------------------------------
# Deep learning (PyTorch): fair MLP + recurrent family, matched capacity
# ----------------------------------------------------------------------
def run_torch(D, results, preds, which):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_neg, n_pos = int((D["y_tr"] == 0).sum()), int((D["y_tr"] == 1).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=dev)

    def loaders(seq):
        def mk(X, y, train=False):
            xt = torch.tensor(X[:, :, None] if seq else X, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.float32)
            ds = TensorDataset(xt, yt)
            if train:
                w = 1.0 / np.bincount(y)[y]
                s = WeightedRandomSampler(torch.tensor(w, dtype=torch.float32),
                                          len(w), replacement=True)
                return DataLoader(ds, batch_size=512, sampler=s)
            return DataLoader(ds, batch_size=1024)
        return (mk(D["X_tr"], D["y_tr"], True),
                mk(D["X_va"], D["y_va"]), mk(D["X_te"], D["y_te"]))

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, 1))
        def forward(self, x): return self.net(x).squeeze(1)

    class RNN(nn.Module):
        # matched capacity across variants (hidden=64); no deliberate weakening
        def __init__(self, kind, hidden=64, layers=1, bidir=False):
            super().__init__()
            cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
            self.rnn = cls(1, hidden, layers, batch_first=True,
                           bidirectional=bidir,
                           dropout=0.3 if layers > 1 else 0.0)
            feat = hidden * (2 if bidir else 1)
            self.head = nn.Sequential(
                nn.BatchNorm1d(feat), nn.Dropout(0.3),
                nn.Linear(feat, feat // 2), nn.ReLU(),
                nn.BatchNorm1d(feat // 2), nn.Dropout(0.3),
                nn.Linear(feat // 2, 1))
        def forward(self, x):
            o, _ = self.rnn(x)
            return self.head(o[:, -1, :]).squeeze(1)

    def train_eval(name, model, seq, epochs=30, patience=6):
        tr, va, te = loaders(seq)
        model = model.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_ap, best_state, wait = -1, None, 0
        t0 = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in tr:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            # val AUPRC
            model.eval(); pv, yv = [], []
            with torch.no_grad():
                for xb, yb in va:
                    pv.extend(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
                    yv.extend(yb.numpy())
            ap = average_precision_score(yv, pv)
            if ap > best_ap:
                best_ap, wait = ap, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
        model.load_state_dict(best_state)
        elapsed = time.time() - t0
        # probs on val (for threshold) and test
        def probs(loader):
            model.eval(); out = []
            with torch.no_grad():
                for xb, _ in loader:
                    out.extend(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
            return np.array(out)
        p_va, p_te = probs(va), probs(te)
        thr = best_threshold_on_val(D["y_va"], p_va)
        m = metrics_at(D["y_te"], p_te, thr)
        m["time_s"] = round(elapsed, 2)
        results[name] = m
        preds[name] = p_te
        print(f"  {name:<22} AUPRC={m['AUPRC']:.4f} F1={m['F1']:.4f} "
              f"MCC={m['MCC']:.4f} P={m['Precision']:.3f} R={m['Recall']:.3f} "
              f"({elapsed:.0f}s)")

    d = D["X_tr"].shape[1]
    if which in ("fast", "all"):
        train_eval("MLP", MLP(d), seq=False)
    if which in ("rnn", "all"):
        train_eval("Vanilla RNN", RNN("rnn",  64, 1, False), seq=True)
        train_eval("LSTM-1L",     RNN("lstm", 64, 1, False), seq=True)
        train_eval("GRU-1L",      RNN("gru",  64, 1, False), seq=True)
        train_eval("BiLSTM",      RNN("lstm", 64, 1, True),  seq=True)


# ----------------------------------------------------------------------
def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "fast"
    print("=" * 64)
    print(f"  XSS Detection v2 | tier={which}")
    print("=" * 64)
    D = load_splits()
    print(f"  Train {len(D['y_tr']):>6,} (XSS {int(D['y_tr'].sum())}) | "
          f"Val {len(D['y_va']):>5,} (XSS {int(D['y_va'].sum())}) | "
          f"Test {len(D['y_te']):>6,} (XSS {int(D['y_te'].sum())})")
    print("-" * 64)

    # resume/merge previous results if present
    res_path = os.path.join(CFG["output_dir"], "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    preds = {}

    if which in ("fast", "all"):
        print("[classical + trees]")
        run_sklearn(D, results, preds)
    print("[deep learning]")
    run_torch(D, results, preds, which)

    # save predictions
    pred_dir = os.path.join(CFG["output_dir"], "preds")
    os.makedirs(pred_dir, exist_ok=True)
    for n, p in preds.items():
        np.save(os.path.join(pred_dir, n.replace(" ", "_") + ".npy"), p)
    np.save(os.path.join(pred_dir, "_y_test.npy"), D["y_te"])

    json.dump(results, open(res_path, "w"), indent=2)

    # ranking table
    print("\n" + "=" * 64)
    print("  RANKING by AUPRC (primary metric for 100:1 imbalance)")
    print("=" * 64)
    order = sorted(results, key=lambda k: results[k]["AUPRC"], reverse=True)
    print(f"  {'Model':<22}{'AUPRC':>8}{'F1':>8}{'MCC':>8}{'Prec':>7}{'Rec':>7}")
    for n in order:
        m = results[n]
        print(f"  {n:<22}{m['AUPRC']:>8.4f}{m['F1']:>8.4f}{m['MCC']:>8.4f}"
              f"{m['Precision']:>7.3f}{m['Recall']:>7.3f}")

    pd.DataFrame(results).T.to_csv(os.path.join(CFG["output_dir"], "results.csv"))
    print(f"\n  Saved: {res_path}")
    print(f"  Saved: {os.path.join(CFG['output_dir'], 'results.csv')}")


if __name__ == "__main__":
    main()
