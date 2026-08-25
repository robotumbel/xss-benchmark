"""R2 - sensitivity of the ranking to the imbalance ratio.

Answers Reviewer C comment 2: the paper only studied 100:1, whereas real
traffic can be far more skewed. We hold the 100,000 benign samples fixed and
subsample the XSS class to obtain 100:1, 250:1, 500:1, 1000:1, 2000:1 and
5000:1, then re-run the protocol end to end at every ratio over three seeds.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import average_precision_score

from rev_common import (SEED, load_structural, tree_models, fit_eval_sklearn,
                        best_threshold_on_val, metrics_at, save)

RATIOS = [100, 250, 500, 1000, 2000, 5000]
SEEDS = [42, 7, 2024]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class GRUNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.rnn = nn.GRU(1, hidden, 1, batch_first=True)
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.BatchNorm1d(hidden // 2), nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1, :]).squeeze(1)


def torch_eval(name, model, D, seq, seed, epochs=25, patience=5):
    torch.manual_seed(seed)
    n_neg, n_pos = int((D["y_tr"] == 0).sum()), int((D["y_tr"] == 1).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEV)

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
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_ap, best_state, wait = -1, None, 0
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
        if ap > best_ap:
            best_ap, wait = ap, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)

    def probs(loader):
        model.eval()
        out = []
        with torch.no_grad():
            for xb, _ in loader:
                out.extend(torch.sigmoid(model(xb.to(DEV))).cpu().numpy())
        return np.array(out)

    thr = best_threshold_on_val(D["y_va"], probs(va))
    return metrics_at(D["y_te"], probs(te), thr)


def main():
    rows = []
    for ratio in RATIOS:
        for seed in SEEDS:
            D = load_structural(seed=seed, pos_ratio=ratio)
            n_pos_te = int(D["y_te"].sum())
            print(f"\n=== ratio {ratio}:1  seed {seed} | "
                  f"train_pos={int(D['y_tr'].sum())} test_pos={n_pos_te}")
            for name, clf in tree_models(D["y_tr"], seed=seed).items():
                m, _ = fit_eval_sklearn(clf, D)
                rows.append(dict(ratio=ratio, seed=seed, model=name,
                                 test_pos=n_pos_te, **{k: m[k] for k in
                                 ("AUPRC", "F1", "MCC", "Precision", "Recall")}))
                print(f"  {name:<20} AUPRC={m['AUPRC']:.4f} MCC={m['MCC']:.4f}")
            for name, mk_model, seq in (("MLP", lambda: MLP(D["X_tr"].shape[1]), False),
                                        ("GRU-1L", lambda: GRUNet(64), True)):
                m = torch_eval(name, mk_model(), D, seq, seed)
                rows.append(dict(ratio=ratio, seed=seed, model=name,
                                 test_pos=n_pos_te, **{k: m[k] for k in
                                 ("AUPRC", "F1", "MCC", "Precision", "Recall")}))
                print(f"  {name:<20} AUPRC={m['AUPRC']:.4f} MCC={m['MCC']:.4f}")

    df = pd.DataFrame(rows)
    save(df, "r2_imbalance_sweep_raw.csv")

    agg = (df.groupby(["ratio", "model"])[["AUPRC", "MCC", "F1", "Precision", "Recall"]]
             .agg(["mean", "std"]).round(4).reset_index())
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    save(agg, "r2_imbalance_sweep_summary.csv")
    print("\n", agg.to_string(index=False))


if __name__ == "__main__":
    main()
