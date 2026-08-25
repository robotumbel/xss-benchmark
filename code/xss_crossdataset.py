"""
================================================================
 XSS Study B — CROSS-DATASET generalization (external validity)

 Dataset A: fmereani  (42,671 payloads, mostly URL-form)
 Dataset B: syedsaqlain (13,686 payloads, HTML-form)

 Two directions: train on A -> test on B, and train on B -> test on A.
 Also within-dataset reference (train/test same source).
 Threshold calibrated on a validation split of the TRAINING source.

 Models: TF-IDF char(2-4)+LogReg (simple), Char-CNN, Char-BiLSTM.
 Run: python xss_crossdataset.py
================================================================
"""
import os, re, time, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef, precision_recall_curve)
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
OUT = "output_p2"; os.makedirs(OUT, exist_ok=True)
MAXLEN = 400


def clean(s): return str(s).replace("<br>", "")

def load_A():
    df = pd.read_parquet("data_raw/xss_payloads_clean.parquet")
    X = np.array([clean(s) for s in df["payload"].tolist()], dtype=object)
    y = np.array([int(v) for v in df["label"].tolist()], dtype=int)
    return X, y

def load_B():
    df = pd.read_csv("data_raw/xss_syedsaqlain.csv")
    df = df.drop_duplicates(subset=["Sentence"]).dropna(subset=["Sentence"])
    X = np.array([str(s) for s in df["Sentence"].tolist()], dtype=object)
    y = np.array([int(v) for v in df["Label"].tolist()], dtype=int)
    return X, y


def thr_val(yv, pv):
    pr, rc, th = precision_recall_curve(yv, pv)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[int(np.nanargmax(f1[:-1]))]) if len(th) else 0.5

def mets(yt, p, thr):
    yh = (p >= thr).astype(int)
    return dict(AUPRC=round(float(average_precision_score(yt, p)), 4),
                F1=round(float(f1_score(yt, yh, zero_division=0)), 4),
                MCC=round(float(matthews_corrcoef(yt, yh)), 4),
                Precision=round(float(precision_score(yt, yh, zero_division=0)), 4),
                Recall=round(float(recall_score(yt, yh, zero_division=0)), 4))


def run_direction(Xtr_all, ytr_all, Xte, yte, tag, results):
    # hold out val from training source for threshold
    Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.12,
                                          stratify=ytr_all, random_state=SEED)
    # ---- TF-IDF char + LogReg ----
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                          max_features=50000, sublinear_tf=True)
    Xt = vec.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=2000, C=5, class_weight="balanced", n_jobs=-1).fit(Xt, ytr)
    thr = thr_val(yva, clf.predict_proba(vec.transform(Xva))[:, 1])
    m = mets(yte, clf.predict_proba(vec.transform(Xte))[:, 1], thr)
    results[f"{tag} | TFIDF-char+LogReg"] = m
    print(f"  {tag:<16} TFIDF-char+LogReg  AUPRC={m['AUPRC']} F1={m['F1']} MCC={m['MCC']} R={m['Recall']}")

    # ---- char DL ----
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chars = set()
    for s in Xtr:
        chars.update(s[:MAXLEN])
    vocab = {c: i + 2 for i, c in enumerate(sorted(chars))}; Vs = len(vocab) + 2
    def enc(arr):
        out = np.zeros((len(arr), MAXLEN), dtype=np.int64)
        for i, s in enumerate(arr):
            for j, c in enumerate(s[:MAXLEN]):
                out[i, j] = vocab.get(c, 1)
        return out
    Xtr_e, Xva_e, Xte_e = enc(Xtr), enc(Xva), enc(Xte)
    spw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32, device=dev)

    class CharCNN(nn.Module):
        def __init__(s):
            super().__init__(); s.emb = nn.Embedding(Vs, 32, padding_idx=0)
            s.convs = nn.ModuleList([nn.Conv1d(32, 128, k, padding=k//2) for k in (3, 5, 7)])
            s.drop = nn.Dropout(0.3); s.fc = nn.Linear(384, 1)
        def forward(s, x):
            e = s.emb(x).transpose(1, 2)
            h = [torch.relu(c(e)).max(2)[0] for c in s.convs]
            return s.fc(s.drop(torch.cat(h, 1))).squeeze(1)

    class CharBiLSTM(nn.Module):
        def __init__(s):
            super().__init__(); s.emb = nn.Embedding(Vs, 32, padding_idx=0)
            s.rnn = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
            s.drop = nn.Dropout(0.3); s.fc = nn.Linear(128, 1)
        def forward(s, x):
            o, _ = s.rnn(s.emb(x)); return s.fc(s.drop(o.max(1)[0])).squeeze(1)

    def loader(X, y, bs, sh=False):
        return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.float32)),
                          batch_size=bs, shuffle=sh)
    ltr = loader(Xtr_e, ytr, 256, True)
    def probs(model, X):
        model.eval(); o = []
        with torch.no_grad():
            for xb, _ in loader(X, np.zeros(len(X)), 512):
                o.extend(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
        return np.array(o)

    for name, Net in [("Char-CNN", CharCNN), ("Char-BiLSTM", CharBiLSTM)]:
        model = Net().to(dev); opt = torch.optim.Adam(model.parameters(), 1e-3)
        lossf = nn.BCEWithLogitsLoss(pos_weight=spw); best, bstate, wait = -1, None, 0
        for ep in range(1, 9):
            model.train()
            for xb, yb in ltr:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ap = average_precision_score(yva, probs(model, Xva_e))
            if ap > best: best, wait, bstate = ap, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 3: break
        model.load_state_dict(bstate)
        thr = thr_val(yva, probs(model, Xva_e))
        m = mets(yte, probs(model, Xte_e), thr)
        results[f"{tag} | {name}"] = m
        print(f"  {tag:<16} {name:<12} AUPRC={m['AUPRC']} F1={m['F1']} MCC={m['MCC']} R={m['Recall']}")


def main():
    print("=" * 66); print("  XSS Study B — Cross-dataset generalization"); print("=" * 66)
    XA, yA = load_A(); XB, yB = load_B()
    print(f"  A (fmereani): {len(XA):,} ({yA.mean():.2f} mal) | B (syedsaqlain): {len(XB):,} ({yB.mean():.2f} mal)")
    print("-" * 66)
    results = {}
    # within-dataset references (train/test split same source)
    XAtr, XAte, yAtr, yAte = train_test_split(XA, yA, test_size=0.2, stratify=yA, random_state=SEED)
    XBtr, XBte, yBtr, yBte = train_test_split(XB, yB, test_size=0.2, stratify=yB, random_state=SEED)
    print("[within A]"); run_direction(XAtr, yAtr, XAte, yAte, "A->A", results)
    print("[within B]"); run_direction(XBtr, yBtr, XBte, yBte, "B->B", results)
    print("[cross A->B]"); run_direction(XA, yA, XB, yB, "A->B", results)
    print("[cross B->A]"); run_direction(XB, yB, XA, yA, "B->A", results)

    json.dump(results, open(os.path.join(OUT, "crossdataset.json"), "w"), indent=2)
    pd.DataFrame(results).T.to_csv(os.path.join(OUT, "crossdataset.csv"))
    print("\n  Saved crossdataset.json / .csv")


if __name__ == "__main__":
    main()
