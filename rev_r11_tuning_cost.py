"""R11 - development cost, not just final-fit cost.

Answers Reviewer C comment 7: the reported training times cover only the final
fit, whereas a deep model in practice also consumes a hyper-parameter search
that a tree ensemble largely avoids. We therefore run an identical random
search budget (12 configurations, same validation set, same selection
criterion) for LightGBM and for the GRU, and report

  * the total wall-clock of the search,
  * the validation AUPRC of the default configuration,
  * the validation AUPRC of the best searched configuration,
  * how much the search actually bought for each family.
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from rev_common import SEED, load_structural, best_threshold_on_val, metrics_at, save

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS = 12


class GRUNet(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.rnn = nn.GRU(1, hidden, 1, batch_first=True)
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden), nn.Dropout(dropout),
            nn.Linear(hidden, max(hidden // 2, 4)), nn.ReLU(),
            nn.BatchNorm1d(max(hidden // 2, 4)), nn.Dropout(dropout),
            nn.Linear(max(hidden // 2, 4), 1))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1, :]).squeeze(1)


def train_gru(D, hidden, dropout, lr, bs, epochs=20, patience=4):
    torch.manual_seed(SEED)
    n_neg, n_pos = int((D["y_tr"] == 0).sum()), int((D["y_tr"] == 1).sum())
    pw = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEV)

    def mk(X, y, train=False):
        xt = torch.tensor(X[:, :, None], dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(xt, yt)
        if train:
            w = 1.0 / np.bincount(y)[y]
            s = WeightedRandomSampler(torch.tensor(w, dtype=torch.float32),
                                      len(w), replacement=True)
            return DataLoader(ds, batch_size=bs, sampler=s)
        return DataLoader(ds, batch_size=2048)

    tr, va = mk(D["X_tr"], D["y_tr"], True), mk(D["X_va"], D["y_va"])
    model = GRUNet(hidden, dropout).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, wait = -1, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        pv = []
        with torch.no_grad():
            for xb, _ in va:
                pv.extend(torch.sigmoid(model(xb.to(DEV))).cpu().numpy())
        ap = average_precision_score(D["y_va"], pv)
        if ap > best:
            best, wait = ap, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best


def main():
    D = load_structural(seed=SEED)
    rng = np.random.RandomState(SEED)
    rows = []

    # ---------------- LightGBM ----------------
    t0 = time.time()
    default = LGBMClassifier(class_weight="balanced", n_jobs=-1,
                             random_state=SEED, verbose=-1)
    default.fit(D["X_tr"], D["y_tr"])
    ap_default_lgb = average_precision_score(
        D["y_va"], default.predict_proba(D["X_va"])[:, 1])
    t_default_lgb = time.time() - t0
    print(f"  LightGBM out-of-the-box: val AUPRC={ap_default_lgb:.4f} "
          f"({t_default_lgb:.1f}s)")

    t0 = time.time()
    best_lgb, best_cfg_lgb = -1, None
    for _ in range(N_TRIALS):
        cfg = dict(n_estimators=int(rng.choice([200, 400, 600, 900])),
                   num_leaves=int(rng.choice([15, 31, 63, 127])),
                   learning_rate=float(rng.choice([0.02, 0.05, 0.1, 0.2])),
                   subsample=float(rng.choice([0.7, 0.9, 1.0])),
                   colsample_bytree=float(rng.choice([0.6, 0.9, 1.0])))
        m = LGBMClassifier(class_weight="balanced", n_jobs=-1,
                           random_state=SEED, verbose=-1, **cfg)
        m.fit(D["X_tr"], D["y_tr"])
        ap = average_precision_score(D["y_va"], m.predict_proba(D["X_va"])[:, 1])
        if ap > best_lgb:
            best_lgb, best_cfg_lgb = ap, cfg
    t_search_lgb = time.time() - t0
    print(f"  LightGBM search ({N_TRIALS} configs): best val AUPRC={best_lgb:.4f} "
          f"({t_search_lgb:.1f}s)")

    # ---------------- GRU ----------------
    t0 = time.time()
    ap_default_gru = train_gru(D, 64, 0.3, 1e-3, 512)
    t_default_gru = time.time() - t0
    print(f"  GRU out-of-the-box: val AUPRC={ap_default_gru:.4f} "
          f"({t_default_gru:.1f}s)")

    t0 = time.time()
    best_gru, best_cfg_gru = -1, None
    for i in range(N_TRIALS):
        cfg = dict(hidden=int(rng.choice([32, 64, 128, 192])),
                   dropout=float(rng.choice([0.0, 0.1, 0.3, 0.5])),
                   lr=float(rng.choice([3e-4, 1e-3, 3e-3])),
                   bs=int(rng.choice([256, 512, 1024])))
        ap = train_gru(D, **cfg)
        print(f"    trial {i+1}/{N_TRIALS} {cfg} -> {ap:.4f}")
        if ap > best_gru:
            best_gru, best_cfg_gru = ap, cfg
    t_search_gru = time.time() - t0
    print(f"  GRU search ({N_TRIALS} configs): best val AUPRC={best_gru:.4f} "
          f"({t_search_gru:.1f}s)")

    rows = [
        {"family": "LightGBM (tree ensemble)",
         "default val AUPRC": round(ap_default_lgb, 4),
         "default fit (s)": round(t_default_lgb, 1),
         "search configs": N_TRIALS,
         "search wall-clock (s)": round(t_search_lgb, 1),
         "best searched val AUPRC": round(best_lgb, 4),
         "gain from search": round(best_lgb - ap_default_lgb, 4),
         "total development cost (s)": round(t_default_lgb + t_search_lgb, 1)},
        {"family": "GRU (recurrent)",
         "default val AUPRC": round(ap_default_gru, 4),
         "default fit (s)": round(t_default_gru, 1),
         "search configs": N_TRIALS,
         "search wall-clock (s)": round(t_search_gru, 1),
         "best searched val AUPRC": round(best_gru, 4),
         "gain from search": round(best_gru - ap_default_gru, 4),
         "total development cost (s)": round(t_default_gru + t_search_gru, 1)},
    ]
    df = pd.DataFrame(rows)
    save(df, "r11_development_cost.csv")
    print("\n", df.to_string(index=False))
    print("\nbest LightGBM cfg:", best_cfg_lgb)
    print("best GRU cfg:", best_cfg_gru)
    with open("../rev_out/r11_best_configs.txt", "w", encoding="utf-8") as fh:
        fh.write(f"LightGBM: {best_cfg_lgb}\nGRU: {best_cfg_gru}\n")


if __name__ == "__main__":
    main()
