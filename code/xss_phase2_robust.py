"""
================================================================
 XSS Phase-2b: OBFUSCATION-ROBUSTNESS study (the real Q2 contribution)

 Motivation: on standard splits the raw-payload task is trivially
 separable by surface keywords (script/alert/%3c), so every model
 scores ~1.0. The meaningful, novel question is ROBUSTNESS: when
 malicious payloads are obfuscated to evade keyword features, which
 detectors still catch them?

 Design:
  * Train on clean payloads (train split).
  * Evaluate on (a) the clean test set and (b) an OBFUSCATED test set
    where each malicious test payload is transformed by realistic
    evasion techniques (case-swap, hex/unicode/percent escaping,
    keyword string-splitting, fromCharCode, comment/whitespace inject).
  * Report the recall drop (clean -> obfuscated). Keyword/TF-IDF models
    should collapse; char-level CNN/BiLSTM should degrade less.

 Run: python xss_phase2_robust.py
================================================================
"""
import os, re, time, json, random, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef, precision_recall_curve)
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
OUT = "output_p2"; os.makedirs(OUT, exist_ok=True)
MAXLEN = 400


# ----------------------------------------------------------------------
# Obfuscation transforms (textual evasion; label-preserving = still XSS)
# ----------------------------------------------------------------------
def t_caseswap(s):
    return "".join(c.upper() if (c.isalpha() and random.random() < 0.5) else c
                   for c in s)

def t_hex_escape(s):
    # escape a few key characters as JS \xHH
    out = []
    for c in s:
        if c in "<>()/'\";=" and random.random() < 0.6:
            out.append("\\x%02x" % ord(c))
        else:
            out.append(c)
    return "".join(out)

def t_unicode_escape(s):
    out = []
    for c in s:
        if c in "<>script alert" and c.isalpha() and random.random() < 0.4:
            out.append("\\u%04x" % ord(c))
        else:
            out.append(c)
    return "".join(out)

def t_percent(s):
    out = []
    for c in s:
        if c in "<>()/\"';" and random.random() < 0.7:
            out.append("%%%02x" % ord(c))
        else:
            out.append(c)
    return "".join(out)

def t_split_keywords(s):
    # break high-signal tokens with string concatenation / inert breaks
    for kw in ["alert", "script", "onerror", "eval", "cookie", "onload"]:
        if kw in s.lower():
            i = len(kw) // 2
            broken = kw[:i] + "'+'" + kw[i:]
            s = re.sub(re.escape(kw), broken, s, flags=re.IGNORECASE, count=1)
    return s

def t_fromcharcode(s):
    # replace a literal alert(...) call with a fromCharCode-encoded form
    m = re.search(r"alert", s, flags=re.IGNORECASE)
    if m:
        codes = ",".join(str(ord(c)) for c in "alert")
        s = s[:m.start()] + f"window[String.fromCharCode({codes})]" + s[m.end():]
    return s

def t_comment_ws(s):
    s = re.sub(r"(scr)(ipt)", r"\1/**/\2", s, flags=re.IGNORECASE)
    s = s.replace("=", " = ") if random.random() < 0.3 else s
    return s

TRANSFORMS = [t_caseswap, t_hex_escape, t_unicode_escape, t_percent,
              t_split_keywords, t_fromcharcode, t_comment_ws]

def obfuscate(s, k=3):
    fns = random.sample(TRANSFORMS, k)
    for f in fns:
        s = f(s)
    return s


# ----------------------------------------------------------------------
def clean(s): return str(s).replace("<br>", "")

def load():
    df = pd.read_parquet("data_raw/xss_payloads_clean.parquet")
    P = [clean(s) for s in df["payload"].tolist()]
    Y = [int(v) for v in df["label"].tolist()]
    pairs = [(p, y) for p, y in zip(P, Y) if len(p) > 0]
    X = np.array([p for p, _ in pairs], dtype=object)
    y = np.array([y for _, y in pairs], dtype=int)
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.125, stratify=ytv, random_state=SEED)
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)


def build_obfuscated_test(D):
    """Obfuscate malicious test payloads; keep benign unchanged."""
    Xob, yob = [], []
    for x, y in zip(D["Xte"], D["yte"]):
        if y == 1:
            Xob.append(obfuscate(x, k=3))
        else:
            Xob.append(x)
        yob.append(y)
    return np.array(Xob, dtype=object), np.array(yob, dtype=int)


def thr_on_val(yv, pv):
    pr, rc, th = precision_recall_curve(yv, pv)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    return float(th[int(np.nanargmax(f1[:-1]))]) if len(th) else 0.5


def evalset(y, p, thr):
    yh = (p >= thr).astype(int)
    return dict(AUPRC=round(float(average_precision_score(y, p)), 4),
                F1=round(float(f1_score(y, yh, zero_division=0)), 4),
                Recall=round(float(recall_score(y, yh, zero_division=0)), 4),
                Precision=round(float(precision_score(y, yh, zero_division=0)), 4),
                MCC=round(float(matthews_corrcoef(y, yh)), 4))


def main():
    print("=" * 66)
    print("  XSS Phase-2b — Obfuscation robustness (clean vs obfuscated test)")
    print("=" * 66)
    D = load()
    Xob, yob = build_obfuscated_test(D)
    print(f"  Train {len(D['ytr']):,} | Test {len(D['yte']):,} "
          f"(mal {int(D['yte'].sum())}) | obfuscated malicious test built")
    print("  Example obfuscation:")
    ex = D["Xte"][D["yte"] == 1][0]
    print("    clean:", ex[:90])
    print("    obf  :", obfuscate(ex, 3)[:90])
    print("-" * 66)

    results = {}

    # ---- TF-IDF char + word (keyword-reliant) + LogReg ----
    reps = {
        "TFIDF-char(2-4)+LogReg": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                                  min_df=3, max_features=50000, sublinear_tf=True),
        "TFIDF-word+LogReg": TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                             min_df=3, max_features=50000, sublinear_tf=True,
                                             token_pattern=r"(?u)\b\w+\b"),
    }
    for name, vec in reps.items():
        t0 = time.time()
        Xtr = vec.fit_transform(D["Xtr"]); Xva = vec.transform(D["Xva"])
        clf = LogisticRegression(max_iter=2000, C=5, class_weight="balanced", n_jobs=-1)
        clf.fit(Xtr, D["ytr"])
        thr = thr_on_val(D["yva"], clf.predict_proba(Xva)[:, 1])
        p_clean = clf.predict_proba(vec.transform(D["Xte"]))[:, 1]
        p_obf = clf.predict_proba(vec.transform(Xob))[:, 1]
        results[name] = {"clean": evalset(D["yte"], p_clean, thr),
                         "obfuscated": evalset(yob, p_obf, thr)}
        print(f"  {name:<26} clean R={results[name]['clean']['Recall']:.3f} "
              f"-> obf R={results[name]['obfuscated']['Recall']:.3f}  ({time.time()-t0:.1f}s)")

    # ---- char-level DL ----
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chars = set()
    for s in D["Xtr"]:
        chars.update(s[:MAXLEN])
    vocab = {c: i + 2 for i, c in enumerate(sorted(chars))}
    Vs = len(vocab) + 2
    def enc(arr):
        out = np.zeros((len(arr), MAXLEN), dtype=np.int64)
        for i, s in enumerate(arr):
            for j, c in enumerate(s[:MAXLEN]):
                out[i, j] = vocab.get(c, 1)
        return out
    Xtr_e, Xva_e = enc(D["Xtr"]), enc(D["Xva"])
    Xte_e, Xob_e = enc(D["Xte"]), enc(Xob)
    spw = torch.tensor([(D["ytr"] == 0).sum() / (D["ytr"] == 1).sum()],
                       dtype=torch.float32, device=dev)

    class CharCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(Vs, 32, padding_idx=0)
            self.convs = nn.ModuleList([nn.Conv1d(32, 128, k, padding=k//2) for k in (3, 5, 7)])
            self.drop = nn.Dropout(0.3); self.fc = nn.Linear(384, 1)
        def forward(self, x):
            e = self.emb(x).transpose(1, 2)
            h = [torch.relu(c(e)).max(2)[0] for c in self.convs]
            return self.fc(self.drop(torch.cat(h, 1))).squeeze(1)

    class CharBiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(Vs, 32, padding_idx=0)
            self.rnn = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.3); self.fc = nn.Linear(128, 1)
        def forward(self, x):
            o, _ = self.rnn(self.emb(x))
            return self.fc(self.drop(o.max(1)[0])).squeeze(1)

    def loader(X, y, bs, sh=False):
        return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.float32)),
                          batch_size=bs, shuffle=sh)
    ltr = loader(Xtr_e, D["ytr"], 256, True); lva = loader(Xva_e, D["yva"], 512)

    def probs(model, X):
        model.eval(); o = []
        ld = loader(X, np.zeros(len(X)), 512)
        with torch.no_grad():
            for xb, _ in ld:
                o.extend(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
        return np.array(o)

    for name, Net in [("Char-CNN", CharCNN), ("Char-BiLSTM", CharBiLSTM)]:
        t0 = time.time(); model = Net().to(dev)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        lossf = nn.BCEWithLogitsLoss(pos_weight=spw)
        best, bs_state, wait = -1, None, 0
        for ep in range(1, 9):
            model.train()
            for xb, yb in ltr:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ap = average_precision_score(D["yva"], probs(model, Xva_e))
            if ap > best: best, wait, bs_state = ap, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= 3: break
        model.load_state_dict(bs_state)
        thr = thr_on_val(D["yva"], probs(model, Xva_e))
        p_clean, p_obf = probs(model, Xte_e), probs(model, Xob_e)
        results[name] = {"clean": evalset(D["yte"], p_clean, thr),
                         "obfuscated": evalset(yob, p_obf, thr)}
        print(f"  {name:<26} clean R={results[name]['clean']['Recall']:.3f} "
              f"-> obf R={results[name]['obfuscated']['Recall']:.3f}  ({time.time()-t0:.0f}s)")

    json.dump(results, open(os.path.join(OUT, "robustness.json"), "w"), indent=2)

    # summary table
    print("\n" + "=" * 66)
    print("  ROBUSTNESS SUMMARY — Recall on malicious (clean -> obfuscated)")
    print("=" * 66)
    print(f"  {'Model':<26}{'Clean R':>9}{'Obf R':>8}{'Drop':>8}{'Obf F1':>8}")
    rows = []
    for n, r in results.items():
        cr, orr = r["clean"]["Recall"], r["obfuscated"]["Recall"]
        print(f"  {n:<26}{cr:>9.3f}{orr:>8.3f}{cr-orr:>8.3f}{r['obfuscated']['F1']:>8.3f}")
        rows.append({"Model": n, "Clean_Recall": cr, "Obf_Recall": orr,
                     "Recall_Drop": round(cr - orr, 4), "Obf_F1": r["obfuscated"]["F1"],
                     "Obf_MCC": r["obfuscated"]["MCC"]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "robustness_summary.csv"), index=False)
    print(f"\n  Saved: {os.path.join(OUT, 'robustness_summary.csv')}")


if __name__ == "__main__":
    main()
