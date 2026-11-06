"""R10 - a precise, testable interpretability analysis.

Answers Reviewer A comment 17 and Reviewer C comment 6.

Reviewer A asks how global importance is aggregated and on which scale the
SHAP values live, and objects that "actionable signals for WAF/IDS rule
design" is stronger than the evidence. Reviewer C objects that confirming the
usefulness of hand-designed features is not an insight.

The script therefore

  1. states the aggregation explicitly: TreeExplainer values are exact Shapley
     values of the margin output, i.e. in log-odds units; global importance is
     the mean absolute value over the test split, and the numbers are exported
     with that scale attached;
  2. separates the feature set into attack-marker features and
     benign-context features and ablates each group, which quantifies how much
     of the detector's decision does *not* come from the obvious attack
     markers;
  3. turns the interpretability claim into a measurable one: a depth-3
     decision list is distilled over the top SHAP features, printed as
     human-readable rules, and evaluated with the same protocol, so that the
     usefulness for rule design is reported as a number rather than asserted.
"""
import re

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from rev_common import (SEED, load_structural, best_threshold_on_val,
                        metrics_at, save)

ATTACK_PAT = re.compile(
    r"script|iframe|event|onload|onmouseover|onerror|alert|eval|fromcharcode|"
    r"cookie|evil|special_characters|escape|unescape|prompt|document|dom|src",
    re.IGNORECASE)


def main():
    import shap
    from lightgbm import LGBMClassifier

    D = load_structural(seed=SEED)
    feat = D["feat_cols"]
    clf = LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                         subsample=0.9, colsample_bytree=0.9,
                         class_weight="balanced", n_jobs=-1,
                         random_state=SEED, verbose=-1)
    clf.fit(D["X_tr"], D["y_tr"])
    p_va = clf.predict_proba(D["X_va"])[:, 1]
    thr = best_threshold_on_val(D["y_va"], p_va)
    base = metrics_at(D["y_te"], clf.predict_proba(D["X_te"])[:, 1], thr)
    print(f"  full model: AUPRC={base['AUPRC']:.4f} MCC={base['MCC']:.4f}")

    # ---------- 1. explicit aggregation ----------
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(D["X_te"])
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    mean_abs = np.abs(sv).mean(axis=0)
    signed_pos = sv[D["y_te"] == 1].mean(axis=0)
    imp = pd.DataFrame({
        "feature": feat,
        "mean |SHAP| (log-odds)": np.round(mean_abs, 4),
        "share of total attribution (%)": np.round(
            100 * mean_abs / mean_abs.sum(), 2),
        "mean SHAP on XSS samples (log-odds)": np.round(signed_pos, 4),
        "group": ["attack marker" if ATTACK_PAT.search(f) else "context"
                  for f in feat],
    }).sort_values("mean |SHAP| (log-odds)", ascending=False).reset_index(drop=True)
    print(imp.head(15).to_string(index=False))
    save(imp, "r10_shap_importance_scaled.csv")

    grp = imp.groupby("group")["share of total attribution (%)"].sum().round(2)
    print("\nAttribution mass by group (%):\n", grp.to_string())

    # ---------- 2. group ablation ----------
    rows = []
    groups = {
        "all 67 features": list(range(len(feat))),
        "attack-marker features only": [i for i, f in enumerate(feat)
                                        if ATTACK_PAT.search(f)],
        "context features only": [i for i, f in enumerate(feat)
                                  if not ATTACK_PAT.search(f)],
        "top-10 SHAP only": [feat.index(f) for f in imp["feature"].head(10)],
    }
    for gname, idx in groups.items():
        if not idx:
            continue
        m = LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.9,
                           class_weight="balanced", n_jobs=-1,
                           random_state=SEED, verbose=-1)
        m.fit(D["X_tr"][:, idx], D["y_tr"])
        t = best_threshold_on_val(D["y_va"], m.predict_proba(D["X_va"][:, idx])[:, 1])
        mm = metrics_at(D["y_te"], m.predict_proba(D["X_te"][:, idx])[:, 1], t)
        rows.append({"feature group": gname, "n features": len(idx),
                     "AUPRC": round(mm["AUPRC"], 4), "MCC": round(mm["MCC"], 4),
                     "F1": round(mm["F1"], 4),
                     "Precision": round(mm["Precision"], 3),
                     "Recall": round(mm["Recall"], 3)})
        print(rows[-1])
    save(pd.DataFrame(rows), "r10_feature_group_ablation.csv")

    # ---------- 3. distilled rule list ----------
    top = [feat.index(f) for f in imp["feature"].head(10)]
    rule_rows = []
    for depth in (2, 3, 4):
        dt = DecisionTreeClassifier(max_depth=depth, class_weight="balanced",
                                    random_state=SEED)
        dt.fit(D["X_tr"][:, top], D["y_tr"])
        t = best_threshold_on_val(D["y_va"], dt.predict_proba(D["X_va"][:, top])[:, 1])
        mm = metrics_at(D["y_te"], dt.predict_proba(D["X_te"][:, top])[:, 1], t)
        rule_rows.append({"max depth": depth, "leaves": dt.get_n_leaves(),
                          "AUPRC": round(mm["AUPRC"], 4),
                          "MCC": round(mm["MCC"], 4), "F1": round(mm["F1"], 4),
                          "Precision": round(mm["Precision"], 3),
                          "Recall": round(mm["Recall"], 3)})
        print(rule_rows[-1])
        if depth == 3:
            # the same rule list refit on raw (unstandardised) feature values,
            # so that the thresholds can be transcribed into a WAF/IDS rule
            dt_raw = DecisionTreeClassifier(max_depth=depth,
                                            class_weight="balanced",
                                            random_state=SEED)
            dt_raw.fit(D["X_tr_raw"][:, top], D["y_tr"])
            t_raw = best_threshold_on_val(
                D["y_va"], dt_raw.predict_proba(D["X_va_raw"][:, top])[:, 1])
            mraw = metrics_at(
                D["y_te"], dt_raw.predict_proba(D["X_te_raw"][:, top])[:, 1], t_raw)
            print("  raw-value depth-3 rule list:",
                  {k: round(mraw[k], 4) for k in
                   ("AUPRC", "MCC", "F1", "Precision", "Recall")})
            txt = export_text(dt, feature_names=[feat[i] for i in top],
                              decimals=2)
            txt_raw = export_text(dt_raw, feature_names=[feat[i] for i in top],
                                  decimals=2)
            with open("../rev_out/r10_distilled_rules_depth3.txt", "w",
                      encoding="utf-8") as fh:
                fh.write("A) Rules on standardised feature values (z-scores of "
                         "the training split)\n\n" + txt +
                         "\n\nB) The same depth-3 rule list refit on raw feature "
                         "counts, directly transcribable into a WAF rule\n"
                         f"   test AUPRC={mraw['AUPRC']:.4f} "
                         f"MCC={mraw['MCC']:.4f} F1={mraw['F1']:.4f} "
                         f"precision={mraw['Precision']:.3f} "
                         f"recall={mraw['Recall']:.3f}\n\n" + txt_raw)
            print(txt_raw)
            rule_rows.append({"max depth": "3 (raw feature values)",
                              "leaves": dt_raw.get_n_leaves(),
                              "AUPRC": round(mraw["AUPRC"], 4),
                              "MCC": round(mraw["MCC"], 4),
                              "F1": round(mraw["F1"], 4),
                              "Precision": round(mraw["Precision"], 3),
                              "Recall": round(mraw["Recall"], 3)})
    save(pd.DataFrame(rule_rows), "r10_distilled_rules.csv")


if __name__ == "__main__":
    main()
