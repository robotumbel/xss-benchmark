"""R9 - external validation of Study A on a second, independent feature schema.

Answers Reviewer C comment 1 (and Reviewer A comment 13). The structural study
used a single corpus with a fixed 67-feature schema, so the "trees beat
recurrent models" conclusion could in principle be an artefact of that schema.

Here we build a *different* feature schema, with a different author, a
different extraction philosophy and a different dimensionality (32 lexical and
entropy-based descriptors), from a *different* corpus (the raw-payload data of
Study B), and re-run the same comparison. If the ordering of the model
families is reproduced on an independent schema and corpus, the conclusion is
not schema-specific.

To make the second schema a real test rather than a restatement of the first,
the imbalance of the source corpus is also re-shaped to 100:1 by subsampling
the malicious class, so that both structural studies are run under the same
operational skew.
"""
import math
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from rev_common import (ROOT, SEED, best_threshold_on_val, metrics_at,
                        tree_models, fit_eval_sklearn, save)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARQUET = os.path.join(ROOT, "data_raw", "xss_payloads_clean.parquet")
SEEDS = [42, 7, 2024]
TARGET_RATIO = 100


def entropy(s):
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


SPECIALS = "<>\"'`;()[]{}=&%#\\/|+*:,?!$@~^"
JS_TOKENS = ["eval", "document", "window", "cookie", "location", "innerhtml",
             "fromcharcode", "settimeout", "onerror", "onload", "onmouseover",
             "javascript", "iframe", "svg", "img", "src", "href", "atob",
             "unescape", "prompt", "confirm", "expression"]


def featurise(s):
    low = s.lower()
    n = max(len(s), 1)
    f = {
        "length": len(s),
        "entropy": entropy(s),
        "n_special": sum(s.count(c) for c in SPECIALS),
        "ratio_special": sum(s.count(c) for c in SPECIALS) / n,
        "ratio_digit": sum(ch.isdigit() for ch in s) / n,
        "ratio_upper": sum(ch.isupper() for ch in s) / n,
        "ratio_alpha": sum(ch.isalpha() for ch in s) / n,
        "n_angle": s.count("<") + s.count(">"),
        "n_quote": s.count('"') + s.count("'"),
        "n_paren": s.count("(") + s.count(")"),
        "n_equals": s.count("="),
        "n_semicolon": s.count(";"),
        "n_percent_escape": len(re.findall(r"%[0-9a-fA-F]{2}", s)),
        "n_hex_escape": len(re.findall(r"\\x[0-9a-fA-F]{2}", s)),
        "n_unicode_escape": len(re.findall(r"\\u[0-9a-fA-F]{4}", s)),
        "n_html_entity": len(re.findall(r"&#?\w+;", s)),
        "n_tags": len(re.findall(r"<\s*/?\s*\w+", s)),
        "n_attrs": len(re.findall(r"\w+\s*=\s*[\"']", s)),
        "n_event_handlers": len(re.findall(r"\bon\w+\s*=", low)),
        "n_urls": len(re.findall(r"https?://", low)),
        "max_token_len": max([len(t) for t in re.split(r"\W+", s) if t] or [0]),
        "n_comment_marks": low.count("/*") + low.count("*/") + low.count("<!--"),
        "n_concat_ops": low.count("+'") + low.count("'+") + low.count('+"'),
        "longest_digit_run": max([len(m) for m in re.findall(r"\d+", s)] or [0]),
    }
    for t in JS_TOKENS[:8]:
        f["tok_" + t] = low.count(t)
    return f


def build_dataset(seed=SEED):
    df = pd.read_parquet(PARQUET)
    texts = [str(s).replace("<br>", "") for s in df["payload"].tolist()]
    labels = np.array([int(v) for v in df["label"].tolist()])
    keep = np.array([len(t) > 0 for t in texts])
    texts = np.array(texts, dtype=object)[keep]
    labels = labels[keep]

    # re-shape to the 100:1 operational skew used in Study A
    rng = np.random.RandomState(seed)
    neg = np.where(labels == 0)[0]
    pos = np.where(labels == 1)[0]
    keep_pos = max(int(round(len(neg) / TARGET_RATIO)), 20)
    pos = rng.choice(pos, min(keep_pos, len(pos)), replace=False)
    sel = np.sort(np.concatenate([neg, pos]))
    texts, labels = texts[sel], labels[sel]

    t0 = time.time()
    F = pd.DataFrame([featurise(t) for t in texts])
    print(f"  extracted {F.shape[1]} features from {len(F):,} payloads "
          f"in {time.time()-t0:.1f}s | pos={int(labels.sum())} "
          f"ratio={int((labels == 0).sum() / max(labels.sum(), 1))}:1")
    return F, labels


def splits(F, y, seed):
    X = F.values.astype(np.float32)
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.125,
                                          stratify=ytv, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    return dict(X_tr=sc.transform(Xtr).astype(np.float32), y_tr=ytr,
                X_va=sc.transform(Xva).astype(np.float32), y_va=yva,
                X_te=sc.transform(Xte).astype(np.float32), y_te=yte)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class RNNNet(nn.Module):
    def __init__(self, kind, hidden=64, bidir=False):
        super().__init__()
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = cls(1, hidden, 1, batch_first=True, bidirectional=bidir)
        feat = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat), nn.Dropout(0.3),
            nn.Linear(feat, feat // 2), nn.ReLU(),
            nn.BatchNorm1d(feat // 2), nn.Dropout(0.3),
            nn.Linear(feat // 2, 1))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1, :]).squeeze(1)


def torch_eval(model, D, seq, seed, epochs=30, patience=6):
    torch.manual_seed(seed)
    n_neg, n_pos = int((D["y_tr"] == 0).sum()), int((D["y_tr"] == 1).sum())
    pw = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEV)

    def mk(X, y, train=False):
        xt = torch.tensor(X[:, :, None] if seq else X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(xt, yt)
        if train:
            w = 1.0 / np.bincount(y)[y]
            s = WeightedRandomSampler(torch.tensor(w, dtype=torch.float32),
                                      len(w), replacement=True)
            return DataLoader(ds, batch_size=512, sampler=s)
        return DataLoader(ds, batch_size=2048)

    tr, va, te = (mk(D["X_tr"], D["y_tr"], True), mk(D["X_va"], D["y_va"]),
                  mk(D["X_te"], D["y_te"]))
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, best_state, wait, t0 = -1, None, 0, time.time()
    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        pv, yv = [], []
        with torch.no_grad():
            for xb, yb in va:
                pv.extend(torch.sigmoid(model(xb.to(DEV))).cpu().numpy())
                yv.extend(yb.numpy())
        ap = average_precision_score(yv, pv)
        if ap > best:
            best, wait = ap, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    def probs(loader):
        model.eval()
        o = []
        with torch.no_grad():
            for xb, _ in loader:
                o.extend(torch.sigmoid(model(xb.to(DEV))).cpu().numpy())
        return np.array(o)

    thr = best_threshold_on_val(D["y_va"], probs(va))
    m = metrics_at(D["y_te"], probs(te), thr)
    m["time_s"] = round(elapsed, 1)
    return m


def main():
    F, y = build_dataset()
    rows = []
    for seed in SEEDS:
        D = splits(F, y, seed)
        print(f"\n=== seed {seed} | train_pos={int(D['y_tr'].sum())} "
              f"test_pos={int(D['y_te'].sum())}")
        for name, clf in tree_models(D["y_tr"], seed=seed).items():
            m, _ = fit_eval_sklearn(clf, D)
            rows.append(dict(seed=seed, model=name, **{k: m[k] for k in
                        ("AUPRC", "F1", "MCC", "Precision", "Recall", "time_s")}))
            print(f"  {name:<20} AUPRC={m['AUPRC']:.4f} MCC={m['MCC']:.4f} "
                  f"({m['time_s']}s)")
        d = D["X_tr"].shape[1]
        for name, mk_model, seq in (("MLP", lambda: MLP(d), False),
                                    ("GRU-1L", lambda: RNNNet("gru", 64), True),
                                    ("BiLSTM", lambda: RNNNet("lstm", 64, True), True),
                                    ("Vanilla RNN", lambda: RNNNet("rnn", 64), True)):
            m = torch_eval(mk_model(), D, seq, seed)
            rows.append(dict(seed=seed, model=name, **{k: m[k] for k in
                        ("AUPRC", "F1", "MCC", "Precision", "Recall", "time_s")}))
            print(f"  {name:<20} AUPRC={m['AUPRC']:.4f} MCC={m['MCC']:.4f} "
                  f"({m['time_s']}s)")

    df = pd.DataFrame(rows)
    save(df, "r9_alt_schema_raw.csv")
    agg = (df.groupby("model")[["AUPRC", "MCC", "F1", "Precision", "Recall", "time_s"]]
             .agg(["mean", "std"]).round(4))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.sort_values("AUPRC_mean", ascending=False).reset_index()
    save(agg, "r9_alt_schema_summary.csv")
    print("\n", agg.to_string(index=False))
    save(pd.DataFrame({"feature": list(F.columns)}), "r9_alt_schema_features.csv")


if __name__ == "__main__":
    main()
