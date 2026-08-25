"""
================================================================
 XSS Detection — Phase 2: RAW PAYLOAD pipeline (honest NLP/DL)

 Realizes the "TF-IDF / embedding + sequence model on raw payloads"
 story truthfully, on a genuine raw-payload dataset (fmereani):
   42,671 unique payloads | 28,068 benign : 15,149 malicious

 Here character sequences ARE genuinely sequential, so char-level
 CNN / BiLSTM are well-motivated (unlike the tabular Phase-1 features).

 Models:
   * TF-IDF (char n-grams 2-4) + Logistic Regression / LinearSVM / XGBoost
   * Char-CNN  (embedding -> Conv1d -> global max pool -> dense)
   * Char-BiLSTM (embedding -> BiLSTM -> dense)

 Protocol (same rigor as Phase 1): stratified train/val/test,
 threshold calibrated on validation, metrics AUPRC/MCC/F1.

 Run:
   python xss_phase2_raw.py tfidf   # TF-IDF + classical/XGBoost (fast)
   python xss_phase2_raw.py dl      # char-CNN + char-BiLSTM
   python xss_phase2_raw.py all
================================================================
"""
import sys, os, re, time, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_score, recall_score, matthews_corrcoef,
                             balanced_accuracy_score, precision_recall_curve,
                             confusion_matrix)
warnings.filterwarnings("ignore")

CFG = {"data": "data_raw/xss_payloads_clean.parquet", "seed": 42,
       "maxlen": 500, "out": "output_p2"}
SEED = CFG["seed"]
np.random.seed(SEED)
os.makedirs(CFG["out"], exist_ok=True)


def clean(s):
    s = str(s).replace("<br>", "")          # scraping artifact
    return s


def load():
    df = pd.read_parquet(CFG["data"])
    payloads = [clean(s) for s in df["payload"].tolist()]
    labels = [int(v) for v in df["label"].tolist()]
    pairs = [(p, l) for p, l in zip(payloads, labels) if len(p) > 0]
    X = np.array([p for p, _ in pairs], dtype=object)
    y = np.array([l for _, l in pairs], dtype=int)
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.125,
                                          stratify=ytv, random_state=SEED)
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)


def thr_on_val(yv, pv):
    pr, rc, th = precision_recall_curve(yv, pv)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[int(np.nanargmax(f1[:-1]))]) if len(th) else 0.5


def metrics(yt, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yh, labels=[0, 1]).ravel()
    return {"AUPRC": float(average_precision_score(yt, p)),
            "ROC_AUC": float(roc_auc_score(yt, p)),
            "F1": float(f1_score(yt, yh, zero_division=0)),
            "Precision": float(precision_score(yt, yh, zero_division=0)),
            "Recall": float(recall_score(yt, yh, zero_division=0)),
            "MCC": float(matthews_corrcoef(yt, yh)),
            "BalAcc": float(balanced_accuracy_score(yt, yh)),
            "threshold": float(thr),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def report(name, res, m, t):
    res[name] = {**m, "time_s": round(t, 2)}
    print(f"  {name:<26} AUPRC={m['AUPRC']:.4f} F1={m['F1']:.4f} "
          f"MCC={m['MCC']:.4f} P={m['Precision']:.3f} R={m['Recall']:.3f} ({t:.1f}s)")


# ---------------- TF-IDF + classical ----------------
def run_tfidf(D, res, preds):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          min_df=3, max_features=50000, sublinear_tf=True)
    t0 = time.time()
    Xtr = vec.fit_transform(D["Xtr"]); Xva = vec.transform(D["Xva"])
    Xte = vec.transform(D["Xte"])
    print(f"  [TF-IDF char(2-4)] vocab={len(vec.vocabulary_):,} "
          f"({time.time()-t0:.1f}s)")
    models = {
        "TFIDF + LogReg": LogisticRegression(max_iter=2000, C=5,
                                             class_weight="balanced", n_jobs=-1),
        "TFIDF + LinearSVM": CalibratedClassifierCV(
            LinearSVC(C=1, class_weight="balanced"), cv=3),
    }
    try:
        from xgboost import XGBClassifier
        spw = (D["ytr"] == 0).sum() / (D["ytr"] == 1).sum()
        models["TFIDF + XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.2,
            subsample=0.9, colsample_bytree=0.6, scale_pos_weight=spw,
            eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=SEED)
    except Exception as e:
        print("  [skip XGBoost]", e)
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(Xtr, D["ytr"])
        pva = clf.predict_proba(Xva)[:, 1]; pte = clf.predict_proba(Xte)[:, 1]
        thr = thr_on_val(D["yva"], pva)
        report(name, res, metrics(D["yte"], pte, thr), time.time() - t0)
        preds[name] = pte


# ---------------- char-level DL ----------------
def run_dl(D, res, preds):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ML = CFG["maxlen"]

    # build char vocab from training data
    chars = set()
    for s in D["Xtr"]:
        chars.update(s[:ML])
    vocab = {c: i + 2 for i, c in enumerate(sorted(chars))}  # 0=pad,1=unk
    V = len(vocab) + 2
    print(f"  [char vocab] size={V}")

    def enc(arr):
        out = np.zeros((len(arr), ML), dtype=np.int64)
        for i, s in enumerate(arr):
            for j, c in enumerate(s[:ML]):
                out[i, j] = vocab.get(c, 1)
        return out

    Xtr, Xva, Xte = enc(D["Xtr"]), enc(D["Xva"]), enc(D["Xte"])

    def loader(X, y, bs, shuffle=False):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.float32))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    ltr = loader(Xtr, D["ytr"], 256, True)
    lva = loader(Xva, D["yva"], 512); lte = loader(Xte, D["yte"], 512)
    spw = torch.tensor([(D["ytr"] == 0).sum() / (D["ytr"] == 1).sum()],
                       dtype=torch.float32, device=dev)

    class CharCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, 32, padding_idx=0)
            self.convs = nn.ModuleList([nn.Conv1d(32, 128, k, padding=k//2)
                                        for k in (3, 5, 7)])
            self.drop = nn.Dropout(0.3); self.fc = nn.Linear(128*3, 1)
        def forward(self, x):
            e = self.emb(x).transpose(1, 2)
            hs = [torch.relu(c(e)).max(dim=2)[0] for c in self.convs]
            return self.fc(self.drop(torch.cat(hs, 1))).squeeze(1)

    class CharBiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, 32, padding_idx=0)
            self.rnn = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3); self.fc = nn.Linear(128, 1)
        def forward(self, x):
            o, _ = self.rnn(self.emb(x))
            return self.fc(self.drop(o.max(dim=1)[0])).squeeze(1)

    def train_eval(name, model, epochs=8, patience=3):
        model = model.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss(pos_weight=spw)
        best, best_state, wait, t0 = -1, None, 0, time.time()
        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in ltr:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss = lossf(model(xb), yb)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval(); pv = []
            with torch.no_grad():
                for xb, _ in lva:
                    pv.extend(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
            ap = average_precision_score(D["yva"], pv)
            if ap > best:
                best, wait = ap, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience: break
        model.load_state_dict(best_state)
        def probs(ld):
            model.eval(); o = []
            with torch.no_grad():
                for xb, _ in ld:
                    o.extend(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
            return np.array(o)
        pva, pte = probs(lva), probs(lte)
        thr = thr_on_val(D["yva"], pva)
        report(name, res, metrics(D["yte"], pte, thr), time.time() - t0)
        preds[name] = pte

    train_eval("Char-CNN", CharCNN())
    train_eval("Char-BiLSTM", CharBiLSTM())


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("=" * 64); print(f"  XSS Phase-2 (raw payload) | tier={which}"); print("=" * 64)
    D = load()
    print(f"  Train {len(D['ytr']):,} (pos {int(D['ytr'].sum())}) | "
          f"Val {len(D['yva']):,} | Test {len(D['yte']):,} "
          f"(pos rate {D['yte'].mean():.3f})")
    print("-" * 64)
    rp = os.path.join(CFG["out"], "results.json")
    res = json.load(open(rp)) if os.path.exists(rp) else {}
    preds = {}
    if which in ("tfidf", "all"):
        print("[TF-IDF + classical]"); run_tfidf(D, res, preds)
    if which in ("dl", "all"):
        print("[char-level deep learning]"); run_dl(D, res, preds)

    pd_dir = os.path.join(CFG["out"], "preds"); os.makedirs(pd_dir, exist_ok=True)
    for n, p in preds.items():
        np.save(os.path.join(pd_dir, n.replace(" ", "_").replace("+", "") + ".npy"), p)
    np.save(os.path.join(pd_dir, "_y_test.npy"), D["yte"])
    json.dump(res, open(rp, "w"), indent=2)

    print("\n" + "=" * 64); print("  RANKING by AUPRC"); print("=" * 64)
    order = sorted(res, key=lambda k: res[k]["AUPRC"], reverse=True)
    print(f"  {'Model':<26}{'AUPRC':>8}{'F1':>8}{'MCC':>8}{'Prec':>7}{'Rec':>7}")
    for n in order:
        m = res[n]
        print(f"  {n:<26}{m['AUPRC']:>8.4f}{m['F1']:>8.4f}{m['MCC']:>8.4f}"
              f"{m['Precision']:>7.3f}{m['Recall']:>7.3f}")
    pd.DataFrame(res).T.to_csv(os.path.join(CFG["out"], "results.csv"))
    print(f"\n  Saved: {rp}")


if __name__ == "__main__":
    main()
