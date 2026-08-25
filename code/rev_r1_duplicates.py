"""R1 - duplicate / near-duplicate audit of the structural dataset.

Answers Reviewer A comments 6 and 8: how many exact and near-duplicate feature
vectors exist, how many of them straddle the train/test boundary, and whether
removing them changes the reported ranking.
"""
import numpy as np
import pandas as pd

from rev_common import (DATA_A, SEED, load_structural, tree_models,
                        fit_eval_sklearn, save)


def main():
    df = pd.read_csv(DATA_A)
    feat = df.drop(columns=["Class"])

    exact_dup = int(df.duplicated().sum())
    feat_dup = int(feat.duplicated().sum())
    conflicting = feat_dup - exact_dup  # same features, different label

    per_class = {}
    for c in (0, 1):
        sub = df[df["Class"] == c]
        per_class[c] = int(sub.duplicated().sum())

    # how many duplicate groups are split across train/val/test in the
    # original (non-deduplicated) protocol?
    D = load_structural(seed=SEED, drop_duplicates=False)
    part = np.empty(len(df), dtype=object)
    part[D["idx_tr"]] = "train"
    part[D["idx_va"]] = "val"
    part[D["idx_te"]] = "test"

    key = feat.astype(np.float32).apply(lambda r: hash(tuple(r)), axis=1).values
    tab = pd.DataFrame({"key": key, "part": part, "y": df["Class"].values})
    grp = tab.groupby("key")["part"].nunique()
    straddling_keys = set(grp[grp > 1].index)
    n_rows_straddling = int(tab["key"].isin(straddling_keys).sum())
    test_rows_dup_with_train = int(
        tab[(tab["part"] == "test") & (tab["key"].isin(straddling_keys))].shape[0])
    test_pos_dup_with_train = int(
        tab[(tab["part"] == "test") & (tab["y"] == 1) &
            (tab["key"].isin(straddling_keys))].shape[0])

    summary = pd.DataFrame([
        {"quantity": "samples (total)", "value": len(df)},
        {"quantity": "exact duplicate rows (features+label)", "value": exact_dup},
        {"quantity": "duplicate feature vectors (any label)", "value": feat_dup},
        {"quantity": "feature vectors duplicated with conflicting labels",
         "value": conflicting},
        {"quantity": "duplicate rows within benign class", "value": per_class[0]},
        {"quantity": "duplicate rows within XSS class", "value": per_class[1]},
        {"quantity": "rows in duplicate groups spanning >1 partition",
         "value": n_rows_straddling},
        {"quantity": "test rows sharing a feature vector with train/val",
         "value": test_rows_dup_with_train},
        {"quantity": "test XSS rows sharing a feature vector with train/val",
         "value": test_pos_dup_with_train},
        {"quantity": "duplicate fraction of dataset (%)",
         "value": round(100 * exact_dup / len(df), 3)},
    ])
    print(summary.to_string(index=False))
    save(summary, "r1_duplicate_audit.csv")

    # Re-run the four strongest non-deep models on a fully deduplicated corpus
    rows = []
    for dedup in (False, True):
        D = load_structural(seed=SEED, drop_duplicates=dedup)
        print(f"\n[dedup={dedup}] n={D['info']['n_used']} "
              f"train_pos={int(D['y_tr'].sum())} test_pos={int(D['y_te'].sum())}")
        for name, clf in tree_models(D["y_tr"]).items():
            m, _ = fit_eval_sklearn(clf, D)
            rows.append({"dedup": dedup, "model": name,
                         "AUPRC": round(m["AUPRC"], 4), "F1": round(m["F1"], 4),
                         "MCC": round(m["MCC"], 4),
                         "Precision": round(m["Precision"], 3),
                         "Recall": round(m["Recall"], 3)})
            print(f"  {name:<20} AUPRC={m['AUPRC']:.4f} MCC={m['MCC']:.4f}")

    out = pd.DataFrame(rows)
    save(out, "r1_dedup_effect.csv")

    piv = out.pivot(index="model", columns="dedup",
                    values=["AUPRC", "MCC", "F1"]).round(4)
    print("\n", piv)


if __name__ == "__main__":
    main()
