"""R7b - cross-dataset transfer for the two additional architectures.

The attention BiLSTM and the char-level Transformer introduced in R7 are put
through exactly the protocol of Section 5.4 of the manuscript (train on the
whole source corpus, calibrate the threshold on a held-out validation split of
the source, evaluate on the other corpus), so that their transfer numbers are
directly comparable with the previously published ones rather than measured
under a different setup.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from rev_common import ROOT, SEED, best_threshold_on_val, metrics_at, save
from rev_r7_modern_dl import CharBiLSTMAttn, CharTransformer, MAXLEN

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean(s):
    return str(s).replace("<br>", "")


def load_A():
    df = pd.read_parquet(os.path.join(ROOT, "data_raw", "xss_payloads_clean.parquet"))
    X = np.array([clean(s) for s in df["payload"].tolist()], dtype=object)
    y = np.array([int(v) for v in df["label"].tolist()], dtype=int)
    return X, y


def load_B():
    df = pd.read_csv(os.path.join(ROOT, "data_raw", "xss_syedsaqlain.csv"))
    df = df.drop_duplicates(subset=["Sentence"]).dropna(subset=["Sentence"])
    X = np.array([str(s) for s in df["Sentence"].tolist()], dtype=object)
    y = np.array([int(v) for v in df["Label"].tolist()], dtype=int)
    return X, y


def run_direction(Xtr_all, ytr_all, Xte, yte, tag, rows):
    Xtr, Xva, ytr, yva = train_test_split(Xtr_all, ytr_all, test_size=0.12,
                                          stratify=ytr_all, random_state=SEED)
    chars = set()
    for s in Xtr:
        chars.update(s[:MAXLEN])
    vocab = {c: i + 2 for i, c in enumerate(sorted(chars))}
    V = len(vocab) + 2

    def enc(arr):
        out = np.zeros((len(arr), MAXLEN), dtype=np.int64)
        for i, s in enumerate(arr):
            for j, c in enumerate(s[:MAXLEN]):
                out[i, j] = vocab.get(c, 1)
        return out

    Etr, Eva, Ete = enc(Xtr), enc(Xva), enc(Xte)
    spw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                       dtype=torch.float32, device=DEV)

    def loader(X, y, bs, sh=False):
        return DataLoader(TensorDataset(torch.tensor(X),
                                        torch.tensor(y, dtype=torch.float32)),
                          batch_size=bs, shuffle=sh)

    def probs(model, X):
        model.eval()
        o = []
        with torch.no_grad():
            for xb, _ in loader(X, np.zeros(len(X)), 512):
                o.extend(torch.sigmoid(model(xb.to(DEV))).cpu().numpy())
        return np.array(o)

    ltr = loader(Etr, ytr, 256, True)
    for name, mk in (("Char-BiLSTM + attention", lambda: CharBiLSTMAttn(V)),
                     ("Char-Transformer (2L, 4 heads)", lambda: CharTransformer(V))):
        torch.manual_seed(SEED)
        model = mk().to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        lossf = nn.BCEWithLogitsLoss(pos_weight=spw)
        best, bstate, wait = -1, None, 0
        for _ in range(8):
            model.train()
            for xb, yb in ltr:
                xb, yb = xb.to(DEV), yb.to(DEV)
                opt.zero_grad()
                lossf(model(xb), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            ap = average_precision_score(yva, probs(model, Eva))
            if ap > best:
                best, wait = ap, 0
                bstate = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 3:
                    break
        model.load_state_dict(bstate)
        thr = best_threshold_on_val(yva, probs(model, Eva))
        m = metrics_at(yte, probs(model, Ete), thr)
        rows.append({"direction": tag, "model": name,
                     "AUPRC": round(m["AUPRC"], 4), "F1": round(m["F1"], 4),
                     "MCC": round(m["MCC"], 4),
                     "Precision": round(m["Precision"], 4),
                     "Recall": round(m["Recall"], 4)})
        print(rows[-1])


def main():
    XA, yA = load_A()
    XB, yB = load_B()
    rows = []
    run_direction(XA, yA, XB, yB, "A -> B", rows)
    run_direction(XB, yB, XA, yA, "B -> A", rows)
    save(pd.DataFrame(rows), "r7b_crossdataset_modern.csv")


if __name__ == "__main__":
    main()
