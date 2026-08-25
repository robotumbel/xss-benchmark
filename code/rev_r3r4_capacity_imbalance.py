"""R3/R4 - model capacity accounting and imbalance-mechanism ablation.

R3 (Reviewer A comment 7): "matched capacity" is made precise. We report the
number of trainable parameters and the per-epoch cost of every recurrent
variant at the shared hidden size of 64, and additionally run a
parameter-matched configuration in which the hidden size of each variant is
chosen so that all four recurrent models have approximately the same number of
trainable parameters as the 64-unit LSTM.

R4 (Reviewer A comment 9): the paper applied a class-weighted loss
(pos_weight) and minority oversampling simultaneously. Here the two mechanisms
are ablated in a 2x2 design on the GRU and the MLP.
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import average_precision_score

from rev_common import (SEED, load_structural, best_threshold_on_val,
                        metrics_at, save)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 7, 2024]


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


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rnn_cell_params(kind, hidden, bidir=False):
    """Recurrent-layer parameters only (input size 1)."""
    gates = {"rnn": 1, "gru": 3, "lstm": 4}[kind]
    one = gates * (hidden * 1 + hidden * hidden + 2 * hidden)
    return one * (2 if bidir else 1)


def match_hidden(kind, target, bidir=False, lo=4, hi=512):
    """Smallest hidden size whose recurrent-layer parameter count is >= target."""
    best = lo
    for h in range(lo, hi + 1):
        if rnn_cell_params(kind, h, bidir) >= target:
            best = h
            break
    return best


def train_eval(model, D, seq, seed, use_pos_weight=True, use_oversample=True,
               epochs=30, patience=6, tag=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_neg, n_pos = int((D["y_tr"] == 0).sum()), int((D["y_tr"] == 1).sum())

    def mk(X, y, train=False):
        xt = torch.tensor(X[:, :, None] if seq else X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(xt, yt)
        if train and use_oversample:
            w = 1.0 / np.bincount(y)[y]
            s = WeightedRandomSampler(torch.tensor(w, dtype=torch.float32),
                                      len(w), replacement=True)
            return DataLoader(ds, batch_size=512, sampler=s)
        return DataLoader(ds, batch_size=512 if train else 2048, shuffle=train)

    tr, va, te = (mk(D["X_tr"], D["y_tr"], True), mk(D["X_va"], D["y_va"]),
                  mk(D["X_te"], D["y_te"]))
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pw = (torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEV)
          if use_pos_weight else None)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_ap, best_state, wait, ep_done = -1, None, 0, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        ep_done = ep + 1
        model.eval()
        pv, yv = [], []
        with torch.no_grad():
            for xb, yb in va:
                pv.extend(torch.sigmoid(model(xb.to(DEV))).cpu().numpy())
                yv.extend(yb.numpy())
        ap = average_precision_score(yv, pv)
        if ap > best_ap:
            best_ap, wait = ap, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    def probs(loader):
        model.eval()
        out = []
        with torch.no_grad():
            for xb, _ in loader:
                out.extend(torch.sigmoid(model(xb.to(DEV))).cpu().numpy())
        return np.array(out)

    # validation probabilities must be produced in a fixed order
    va_eval = DataLoader(TensorDataset(
        torch.tensor(D["X_va"][:, :, None] if seq else D["X_va"], dtype=torch.float32),
        torch.tensor(D["y_va"], dtype=torch.float32)), batch_size=2048)
    thr = best_threshold_on_val(D["y_va"], probs(va_eval))
    m = metrics_at(D["y_te"], probs(te), thr)
    m["time_s"] = round(elapsed, 1)
    m["epochs"] = ep_done
    m["s_per_epoch"] = round(elapsed / max(ep_done, 1), 2)
    return m


def main():
    D = load_structural(seed=SEED)
    d = D["X_tr"].shape[1]

    # ---------- R3a: parameter accounting at the shared hidden size ----------
    cfgs = [("Vanilla RNN", "rnn", 64, False), ("LSTM-1L", "lstm", 64, False),
            ("GRU-1L", "gru", 64, False), ("BiLSTM", "lstm", 64, True)]
    rows = []
    for name, kind, h, bi in cfgs:
        mdl = RNNNet(kind, h, bi)
        rows.append({"model": name, "hidden": h, "bidirectional": bi,
                     "recurrent params": rnn_cell_params(kind, h, bi),
                     "total trainable params": n_params(mdl)})
    mlp = MLP(d)
    rows.append({"model": "MLP", "hidden": "128-64", "bidirectional": False,
                 "recurrent params": 0, "total trainable params": n_params(mlp)})
    cap = pd.DataFrame(rows)
    print("Capacity at the shared hidden size of 64")
    print(cap.to_string(index=False))
    save(cap, "r3_capacity_hidden64.csv")

    # ---------- R3b: parameter-matched recurrent family ----------
    target = rnn_cell_params("lstm", 64, False)   # 4-gate LSTM at hidden 64
    matched = [("Vanilla RNN (matched)", "rnn", match_hidden("rnn", target), False),
               ("LSTM-1L (reference)", "lstm", 64, False),
               ("GRU-1L (matched)", "gru", match_hidden("gru", target), False),
               ("BiLSTM (matched)", "lstm", match_hidden("lstm", target, True), True)]
    rows = []
    for name, kind, h, bi in matched:
        mdl = RNNNet(kind, h, bi)
        print(f"\n[param-matched] {name}: hidden={h} "
              f"recurrent params={rnn_cell_params(kind, h, bi)} "
              f"total={n_params(mdl)}")
        accs = []
        for s in SEEDS:
            m = train_eval(RNNNet(kind, h, bi), D, True, s)
            accs.append(m)
            print(f"   seed {s}: AUPRC={m['AUPRC']:.4f} MCC={m['MCC']:.4f} "
                  f"({m['time_s']}s, {m['epochs']} epochs)")
        rows.append({
            "model": name, "hidden": h,
            "recurrent params": rnn_cell_params(kind, h, bi),
            "total params": n_params(mdl),
            "AUPRC_mean": round(np.mean([a["AUPRC"] for a in accs]), 4),
            "AUPRC_std": round(np.std([a["AUPRC"] for a in accs]), 4),
            "MCC_mean": round(np.mean([a["MCC"] for a in accs]), 4),
            "MCC_std": round(np.std([a["MCC"] for a in accs]), 4),
            "F1_mean": round(np.mean([a["F1"] for a in accs]), 4),
            "train_s_mean": round(np.mean([a["time_s"] for a in accs]), 1),
            "s_per_epoch": round(np.mean([a["s_per_epoch"] for a in accs]), 2)})
    pm = pd.DataFrame(rows)
    print("\nParameter-matched recurrent family")
    print(pm.to_string(index=False))
    save(pm, "r3_param_matched_recurrent.csv")

    # ---------- R4: pos_weight x oversampling ablation ----------
    rows = []
    for arch, mk_model, seq in (("GRU-1L", lambda: RNNNet("gru", 64), True),
                                ("MLP", lambda: MLP(d), False)):
        for pw in (True, False):
            for ov in (True, False):
                accs = []
                for s in SEEDS:
                    m = train_eval(mk_model(), D, seq, s,
                                   use_pos_weight=pw, use_oversample=ov)
                    accs.append(m)
                rows.append({
                    "model": arch,
                    "pos_weight loss": pw, "minority oversampling": ov,
                    "AUPRC_mean": round(np.mean([a["AUPRC"] for a in accs]), 4),
                    "AUPRC_std": round(np.std([a["AUPRC"] for a in accs]), 4),
                    "MCC_mean": round(np.mean([a["MCC"] for a in accs]), 4),
                    "MCC_std": round(np.std([a["MCC"] for a in accs]), 4),
                    "Precision_mean": round(np.mean([a["Precision"] for a in accs]), 3),
                    "Recall_mean": round(np.mean([a["Recall"] for a in accs]), 3)})
                print(f"  {arch} pos_weight={pw} oversample={ov} -> "
                      f"AUPRC={rows[-1]['AUPRC_mean']:.4f} "
                      f"MCC={rows[-1]['MCC_mean']:.4f}")
    ab = pd.DataFrame(rows)
    print("\nImbalance-mechanism ablation")
    print(ab.to_string(index=False))
    save(ab, "r4_imbalance_mechanism_ablation.csv")


if __name__ == "__main__":
    main()
