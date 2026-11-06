"""R5/R6 - multiple-comparison-corrected significance testing and an explicit
threshold-selection comparison.

Answers Reviewer A comments 12 and 18.

R5: all-pairs McNemar tests among the top-5 models with Holm-Bonferroni
    correction, instead of one-vs-reference uncorrected tests.
R6: for every model, the operating point obtained (a) with the threshold
    calibrated on validation (our protocol), (b) with the threshold tuned on
    the test set to maximise F1, and (c) with the threshold tuned on the test
    set under a recall-biased criterion (F2), which is the common source of
    the low-precision numbers reported in leakage-prone studies.
"""
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from rev_common import OUT, mcnemar, holm_bonferroni, metrics_at, save

PRED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "output_v2", "preds")

MODELS = ["Random Forest", "LightGBM", "XGBoost", "GRU-1L", "MLP", "BiLSTM",
          "LSTM-1L", "Logistic Regression", "Vanilla RNN", "Naive Bayes"]

# validation-calibrated thresholds actually used in the paper, recovered from
# the stored results table
RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "output_v2", "results.csv")


def load():
    y = np.load(os.path.join(PRED_DIR, "_y_test.npy"))
    p = {}
    for m in MODELS:
        f = os.path.join(PRED_DIR, m.replace(" ", "_") + ".npy")
        if os.path.exists(f):
            p[m] = np.load(f)
    res = pd.read_csv(RESULTS_CSV, index_col=0)
    thr = {m: float(res.loc[m, "threshold"]) for m in p if m in res.index}
    return y, p, thr


def best_thr(y, p, beta=1.0):
    prec, rec, t = precision_recall_curve(y, p)
    b2 = beta ** 2
    f = (1 + b2) * prec * rec / (b2 * prec + rec + 1e-12)
    i = int(np.nanargmax(f[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5


def main():
    y, P, THR = load()

    # ---------- R5: all-pairs McNemar with Holm correction ----------
    top = ["Random Forest", "LightGBM", "XGBoost", "MLP", "GRU-1L"]
    yhat = {m: (P[m] >= THR[m]).astype(int) for m in P}
    pairs, stats, pvals, bs, cs = [], [], [], [], []
    for a, b in itertools.combinations(top, 2):
        s, pv, nb, nc = mcnemar(y, yhat[a], yhat[b])
        pairs.append(f"{a} vs {b}")
        stats.append(round(s, 3))
        pvals.append(pv)
        bs.append(nb)
        cs.append(nc)
    adj = holm_bonferroni(pvals)
    allpairs = pd.DataFrame({
        "comparison": pairs, "b (only A correct)": bs, "c (only B correct)": cs,
        "chi2": stats, "p_raw": np.round(pvals, 4),
        "p_Holm": np.round(adj, 4),
        "significant (Holm, 0.05)": adj < 0.05})
    print("All-pairs McNemar among the top-5 (Holm-Bonferroni corrected)")
    print(allpairs.to_string(index=False))
    save(allpairs, "r5_mcnemar_allpairs_holm.csv")

    # one-vs-reference (Random Forest), Holm corrected over the 9 comparisons
    ref = "Random Forest"
    others = [m for m in MODELS if m in P and m != ref]
    pv2, st2, b2, c2 = [], [], [], []
    for m in others:
        s, pv, nb, nc = mcnemar(y, yhat[ref], yhat[m])
        st2.append(round(s, 3)); pv2.append(pv); b2.append(nb); c2.append(nc)
    adj2 = holm_bonferroni(pv2)
    vsref = pd.DataFrame({
        "model": others, "b": b2, "c": c2, "chi2": st2,
        "p_raw": np.round(pv2, 5), "p_Holm": np.round(adj2, 5),
        "significant (Holm, 0.05)": adj2 < 0.05})
    print("\nRandom Forest vs each model (Holm-corrected over 9 tests)")
    print(vsref.to_string(index=False))
    save(vsref, "r5_mcnemar_vs_rf_holm.csv")

    # ---------- R6: threshold-selection comparison ----------
    rows = []
    for m in P:
        p = P[m]
        for label, thr in (
                ("validation-calibrated (ours)", THR[m]),
                ("test-tuned, max F1", best_thr(y, p, 1.0)),
                ("test-tuned, recall-biased (max F2)", best_thr(y, p, 2.0))):
            mm = metrics_at(y, p, thr)
            rows.append({"model": m, "threshold rule": label,
                         "threshold": round(thr, 6),
                         "Precision": round(mm["Precision"], 4),
                         "Recall": round(mm["Recall"], 4),
                         "F1": round(mm["F1"], 4), "MCC": round(mm["MCC"], 4),
                         "FP": mm["FP"], "FN": mm["FN"]})
    thr_df = pd.DataFrame(rows)
    save(thr_df, "r6_threshold_rules.csv")
    print("\nThreshold-selection comparison")
    print(thr_df.to_string(index=False))

    # compact view for the paper: three representative models
    sel = thr_df[thr_df["model"].isin(["Random Forest", "LightGBM", "BiLSTM"])]
    save(sel, "r6_threshold_rules_compact.csv")


if __name__ == "__main__":
    main()
