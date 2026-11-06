"""R7 - modern deep architectures on raw payloads.

Answers Reviewer C comment 4: the raw-payload study only contained a plain
Char-CNN and a plain Char-BiLSTM, so it could not speak to attention-based or
transformer models. Here we add

  * Char-BiLSTM + additive attention pooling,
  * a char-level Transformer encoder (multi-head self-attention,
    2 layers x 4 heads, learned positional embeddings),

and evaluate them under exactly the same protocol as the existing models, on
the standard split, the escape-based obfuscated test set, the harder
artifact-suppressing evasion set of R8, and the cross-dataset direction A->B,
so that "no advantage from deep models" is tested against a current-generation
attention architecture rather than only against basic ones.
"""
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from rev_common import OUT, ROOT, SEED, best_threshold_on_val, metrics_at, save

SEEDS = [42, 7, 2024]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAXLEN = 400
PARQUET = os.path.join(ROOT, "data_raw", "xss_payloads_clean.parquet")
CSV_B = os.path.join(ROOT, "data_raw", "xss_syedsaqlain.csv")


def clean(s):
    return str(s).replace("<br>", "")


def load_a():
    df = pd.read_parquet(PARQUET)
    P = [clean(s) for s in df["payload"].tolist()]
    Y = [int(v) for v in df["label"].tolist()]
    pairs = [(p, y) for p, y in zip(P, Y) if len(p) > 0]
    X = np.array([p for p, _ in pairs], dtype=object)
    y = np.array([v for _, v in pairs], dtype=int)
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.125,
                                          stratify=ytv, random_state=SEED)
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)


def load_b():
    df = pd.read_csv(CSV_B)
    txt = df["Sentence"].astype(str).map(clean).values
    lab = df["Label"].astype(int).values
    keep = np.array([len(t) > 0 for t in txt])
    return txt[keep], lab[keep]


# ----------------------------------------------------------------------
# obfuscation (identical transforms to the published robustness study)
# ----------------------------------------------------------------------
import random  # noqa: E402


def t_caseswap(s):
    return "".join(c.upper() if (c.isalpha() and random.random() < 0.5) else c
                   for c in s)


def t_hex_escape(s):
    return "".join("\\x%02x" % ord(c) if (c in "<>()/'\";=" and random.random() < 0.6)
                   else c for c in s)


def t_unicode_escape(s):
    return "".join("\\u%04x" % ord(c)
                   if (c in "<>script alert" and c.isalpha() and random.random() < 0.4)
                   else c for c in s)


def t_percent(s):
    return "".join("%%%02x" % ord(c) if (c in "<>()/\"';" and random.random() < 0.7)
                   else c for c in s)


def t_split_keywords(s):
    for kw in ["alert", "script", "onerror", "eval", "cookie", "onload"]:
        if kw in s.lower():
            i = len(kw) // 2
            s = re.sub(re.escape(kw), kw[:i] + "'+'" + kw[i:], s,
                       flags=re.IGNORECASE, count=1)
    return s


def t_fromcharcode(s):
    m = re.search(r"alert", s, flags=re.IGNORECASE)
    if m:
        codes = ",".join(str(ord(c)) for c in "alert")
        s = s[:m.start()] + f"window[String.fromCharCode({codes})]" + s[m.end():]
    return s


def t_comment_ws(s):
    s = re.sub(r"(scr)(ipt)", r"\1/**/\2", s, flags=re.IGNORECASE)
    return s.replace("=", " = ") if random.random() < 0.3 else s


TRANSFORMS = [t_caseswap, t_hex_escape, t_unicode_escape, t_percent,
              t_split_keywords, t_fromcharcode, t_comment_ws]


def obfuscate(s, k=3):
    for f in random.sample(TRANSFORMS, k):
        s = f(s)
    return s


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------
class CharCNN(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, 32, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(32, 128, k, padding=k // 2)
                                    for k in (3, 5, 7)])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 3, 1)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        hs = [torch.relu(c(e)).max(dim=2)[0] for c in self.convs]
        return self.fc(self.drop(torch.cat(hs, 1))).squeeze(1)


class CharBiLSTM(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, 32, padding_idx=0)
        self.rnn = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        o, _ = self.rnn(self.emb(x))
        return self.fc(self.drop(o.max(dim=1)[0])).squeeze(1)


class CharBiLSTMAttn(nn.Module):
    """BiLSTM with additive (Bahdanau-style) attention pooling."""

    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, 32, padding_idx=0)
        self.rnn = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.att = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        o, _ = self.rnn(self.emb(x))
        mask = (x == 0)
        a = self.att(o).squeeze(-1).masked_fill(mask, -1e9).softmax(dim=1)
        ctx = torch.einsum("bt,btd->bd", a, o)
        return self.fc(self.drop(ctx)).squeeze(1)


class CharTransformer(nn.Module):
    """Char-level Transformer encoder with learned positional embeddings."""

    def __init__(self, V, d_model=128, heads=4, layers=2, maxlen=MAXLEN):
        super().__init__()
        self.emb = nn.Embedding(V, d_model, padding_idx=0)
        self.pos = nn.Embedding(maxlen, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=256, dropout=0.1,
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.emb(x) + self.pos(pos)
        mask = (x == 0)
        h = self.enc(h, src_key_padding_mask=mask)
        h = h.masked_fill(mask.unsqueeze(-1), -1e9).max(dim=1)[0]
        return self.fc(self.drop(h)).squeeze(1)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_vocab(train_texts):
    chars = set()
    for s in train_texts:
        chars.update(s[:MAXLEN])
    return {c: i + 2 for i, c in enumerate(sorted(chars))}


def encode(texts, vocab):
    out = np.zeros((len(texts), MAXLEN), dtype=np.int64)
    for i, s in enumerate(texts):
        for j, c in enumerate(s[:MAXLEN]):
            out[i, j] = vocab.get(c, 1)
    return out


def loader(X, y, bs, shuffle=False):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def train_model(model, Xtr, ytr, Xva, yva, epochs=8, patience=3, bs=256, lr=1e-3):
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    spw = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)],
                       dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=spw)
    ltr, lva = loader(Xtr, ytr, bs, True), loader(Xva, yva, 512)
    best, best_state, wait, t0 = -1, None, 0, time.time()
    for _ in range(epochs):
        model.train()
        for xb, yb in ltr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        pv = []
        with torch.no_grad():
            for xb, _ in lva:
                pv.extend(torch.sigmoid(model(xb.to(DEV))).cpu().numpy())
        ap = average_precision_score(yva, pv)
        if ap > best:
            best, wait = ap, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, time.time() - t0


def predict(model, X, bs=512):
    model.eval()
    out = []
    ld = loader(X, np.zeros(len(X)), bs)
    with torch.no_grad():
        for xb, _ in ld:
            out.extend(torch.sigmoid(model(xb.to(DEV))).cpu().numpy())
    return np.array(out)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    D = load_a()
    vocab = build_vocab(D["Xtr"])
    V = len(vocab) + 2
    Etr, Eva, Ete = (encode(D[k], vocab) for k in ("Xtr", "Xva", "Xte"))

    # obfuscated test set: malicious payloads transformed, benign untouched
    Xte_obf = np.array([obfuscate(s) if y == 1 else s
                        for s, y in zip(D["Xte"], D["yte"])], dtype=object)
    Ete_obf = encode(Xte_obf, vocab)

    # artifact-suppressing evasion (tier 3 of the R8 audit), applied to the
    # malicious payloads only, so that every model family is probed with the
    # hardest evasion setting rather than only with escape-based obfuscation
    import rev_r8_obfuscation as r8
    r8.BENIGN_POOL.extend([t for t in D["Xtr"][D["ytr"] == 0] if len(t) > 40])
    random.seed(SEED)
    Xte_t3 = np.array([r8.apply_tier(s, r8.TIER3, 3) if y == 1 else s
                       for s, y in zip(D["Xte"], D["yte"])], dtype=object)
    Ete_t3 = encode(Xte_t3, vocab)

    # cross-dataset target
    Xb, yb = load_b()
    Eb = encode(Xb, vocab)

    rows = []

    # ---- TF-IDF reference (kept identical to the published configuration) ----
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                          max_features=50000, sublinear_tf=True)
    t0 = time.time()
    Ttr = vec.fit_transform(D["Xtr"])
    lr = LogisticRegression(max_iter=2000, C=5, class_weight="balanced")
    lr.fit(Ttr, D["ytr"])
    t_tfidf = time.time() - t0
    p_va = lr.predict_proba(vec.transform(D["Xva"]))[:, 1]
    thr = best_threshold_on_val(D["yva"], p_va)
    m_clean = metrics_at(D["yte"], lr.predict_proba(vec.transform(D["Xte"]))[:, 1], thr)
    m_obf = metrics_at(D["yte"], lr.predict_proba(vec.transform(Xte_obf))[:, 1], thr)
    m_t3 = metrics_at(D["yte"], lr.predict_proba(vec.transform(Xte_t3))[:, 1], thr)
    m_cross = metrics_at(yb, lr.predict_proba(vec.transform(Xb))[:, 1], thr)
    rows.append({"model": "TF-IDF char(2-4) + LogReg", "params": Ttr.shape[1],
                 "train_s": round(t_tfidf, 1),
                 "AUPRC": round(m_clean["AUPRC"], 4), "F1": round(m_clean["F1"], 4),
                 "MCC": round(m_clean["MCC"], 4),
                 "Precision": round(m_clean["Precision"], 4),
                 "Recall": round(m_clean["Recall"], 4),
                 "Recall_clean": round(m_clean["Recall"], 4),
                 "Recall_obf": round(m_obf["Recall"], 4),
                 "F1_obf": round(m_obf["F1"], 4),
                 "Recall_tier3": round(m_t3["Recall"], 4),
                 "F1_tier3": round(m_t3["F1"], 4),
                 "MCC_crossAB": round(m_cross["MCC"], 4),
                 "F1_crossAB": round(m_cross["F1"], 4)})
    print(rows[-1])

    # ---- remaining TF-IDF variants, same protocol ----
    from sklearn.feature_extraction.text import TfidfVectorizer as _TV
    from xgboost import XGBClassifier
    extra = [
        ("TF-IDF word + LogReg",
         lambda: _TV(analyzer="word", ngram_range=(1, 2), min_df=2,
                     max_features=50000, sublinear_tf=True),
         lambda: LogisticRegression(max_iter=2000, C=5, class_weight="balanced")),
        ("TF-IDF char + XGBoost",
         lambda: _TV(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                     max_features=50000, sublinear_tf=True),
         lambda: XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.2,
                               subsample=0.9, colsample_bytree=0.6,
                               scale_pos_weight=(D["ytr"] == 0).sum() / (D["ytr"] == 1).sum(),
                               eval_metric="aucpr", tree_method="hist",
                               n_jobs=-1, random_state=SEED)),
    ]
    for name, mk_vec, mk_clf in extra:
        v = mk_vec()
        t0 = time.time()
        Z = v.fit_transform(D["Xtr"])
        c = mk_clf()
        c.fit(Z, D["ytr"])
        tt = time.time() - t0
        th = best_threshold_on_val(D["yva"], c.predict_proba(v.transform(D["Xva"]))[:, 1])
        mc = metrics_at(D["yte"], c.predict_proba(v.transform(D["Xte"]))[:, 1], th)
        mo = metrics_at(D["yte"], c.predict_proba(v.transform(Xte_obf))[:, 1], th)
        m3 = metrics_at(D["yte"], c.predict_proba(v.transform(Xte_t3))[:, 1], th)
        rows.append({"model": name, "params": Z.shape[1], "train_s": round(tt, 1),
                     "AUPRC": round(mc["AUPRC"], 4), "F1": round(mc["F1"], 4),
                     "MCC": round(mc["MCC"], 4),
                     "Precision": round(mc["Precision"], 4),
                     "Recall": round(mc["Recall"], 4),
                     "Recall_clean": round(mc["Recall"], 4),
                     "Recall_obf": round(mo["Recall"], 4),
                     "F1_obf": round(mo["F1"], 4),
                     "Recall_tier3": round(m3["Recall"], 4),
                     "F1_tier3": round(m3["F1"], 4),
                     "MCC_crossAB": float("nan"), "F1_crossAB": float("nan")})
        print(rows[-1])

    # ---- deep models, three seeds each ----
    # the neural models are re-trained under three seeds because the recall
    # retained under artifact-suppressing evasion turned out to vary far more
    # across seeds than any in-distribution metric
    zoo = [("Char-CNN", lambda: CharCNN(V)),
           ("Char-BiLSTM", lambda: CharBiLSTM(V)),
           ("Char-BiLSTM + attention", lambda: CharBiLSTMAttn(V)),
           ("Char-Transformer (2L, 4 heads)", lambda: CharTransformer(V))]
    for name, mk in zoo:
        runs = []
        npar = None
        for sd in SEEDS:
            torch.manual_seed(sd)
            torch.cuda.manual_seed_all(sd)
            np.random.seed(sd)
            model = mk()
            npar = n_params(model)
            model, tt = train_model(model, Etr, D["ytr"], Eva, D["yva"])
            thr = best_threshold_on_val(D["yva"], predict(model, Eva))
            mc = metrics_at(D["yte"], predict(model, Ete), thr)
            mo = metrics_at(D["yte"], predict(model, Ete_obf), thr)
            m3 = metrics_at(D["yte"], predict(model, Ete_t3), thr)
            mx = metrics_at(yb, predict(model, Eb), thr)
            runs.append(dict(train_s=tt, AUPRC=mc["AUPRC"], F1=mc["F1"],
                             MCC=mc["MCC"], Precision=mc["Precision"],
                             Recall=mc["Recall"], Recall_obf=mo["Recall"],
                             F1_obf=mo["F1"], Recall_tier3=m3["Recall"],
                             F1_tier3=m3["F1"], MCC_crossAB=mx["MCC"],
                             F1_crossAB=mx["F1"]))
            print(f"  {name} seed {sd}: F1={mc['F1']:.4f} "
                  f"obf_recall={mo['Recall']:.4f} tier3_recall={m3['Recall']:.4f}")

        def agg(key, nd=4):
            v = [r[key] for r in runs]
            return round(float(np.mean(v)), nd), round(float(np.std(v)), nd)

        row = {"model": name, "params": npar, "train_s": agg("train_s", 1)[0]}
        for key in ("AUPRC", "F1", "MCC", "Precision", "Recall", "Recall_obf",
                    "F1_obf", "Recall_tier3", "F1_tier3", "MCC_crossAB",
                    "F1_crossAB"):
            m, sd_ = agg(key)
            row[key] = m
            row[key + "_std"] = sd_
        row["Recall_clean"] = row["Recall"]
        rows.append(row)
        print(rows[-1])

    df = pd.DataFrame(rows)
    save(df, "r7_modern_architectures.csv")
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
