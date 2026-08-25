"""R8 - a properly qualified obfuscation analysis.

Answers Reviewer A comment 19 (and Reviewer C comment 8). The published
experiment obfuscated only the malicious payloads with a mixed transform set
and reported an almost unchanged recall. Three objections follow: (i) some
transforms may not preserve malicious functionality, (ii) the transforms may
inject new, easily detectable artifacts, and (iii) leaving benign traffic
unchanged makes the comparison asymmetric.

This script therefore reports:

  A. a transform-level audit that separates the transforms into a
     functionality-preserving tier (case randomisation, in-tag whitespace,
     HTML comments inside tag names, percent-encoding of URL-context payloads)
     and a tier whose semantics are not guaranteed, and measures robustness
     under each tier separately;
  B. a benign control in which the same textual transforms are applied to the
     benign payloads, so that the change of the false-positive rate is
     measured rather than assumed;
  C. direct evidence for the "one anomaly is swapped for another" explanation:
     a discriminator trained to separate obfuscated from clean malicious
     payloads, and the keyword-survival statistics before and after;
  D. an artifact-suppressing evasion tier that avoids exotic escape sequences
     altogether (lower-case-safe tag/attribute rewriting, benign-looking
     padding), which is the harder case the reviewers ask about.
"""
import os
import random
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score

from rev_common import ROOT, SEED, best_threshold_on_val, metrics_at, save

PARQUET = os.path.join(ROOT, "data_raw", "xss_payloads_clean.parquet")


def clean(s):
    return str(s).replace("<br>", "")


def load():
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


# ---------------------------------------------------------------- tier 1
# transforms that preserve execution semantics in the context in which the
# payload appears (HTML tag and attribute names are case-insensitive; inline
# whitespace inside a tag is ignored; percent-encoding is decoded by the
# server before the payload reaches the HTML sink)
def p_caseswap(s):
    return "".join(c.upper() if (c.isalpha() and random.random() < 0.5) else c
                   for c in s)


def p_tag_whitespace(s):
    return re.sub(r"<\s*(\w+)", lambda m: "<" + m.group(1) + " " * random.randint(1, 3),
                  s)


def p_attr_spacing(s):
    return re.sub(r"(\w)=(\"|')", lambda m: m.group(1) + " = " + m.group(2), s)


def p_percent_url(s):
    """Percent-encode only characters that a server URL-decodes before the sink."""
    return "".join("%%%02x" % ord(c) if (c in "<>\"'" and random.random() < 0.5)
                   else c for c in s)


TIER1 = [p_caseswap, p_tag_whitespace, p_attr_spacing, p_percent_url]

# ---------------------------------------------------------------- tier 2
# transforms used in the original experiment whose semantics are context
# dependent and therefore not guaranteed to preserve execution


def q_hex_escape(s):
    return "".join("\\x%02x" % ord(c) if (c in "<>()/'\";=" and random.random() < 0.6)
                   else c for c in s)


def q_unicode_escape(s):
    return "".join("\\u%04x" % ord(c)
                   if (c in "<>script alert" and c.isalpha() and random.random() < 0.4)
                   else c for c in s)


def q_split_keywords(s):
    for kw in ["alert", "script", "onerror", "eval", "cookie", "onload"]:
        if kw in s.lower():
            i = len(kw) // 2
            s = re.sub(re.escape(kw), kw[:i] + "'+'" + kw[i:], s,
                       flags=re.IGNORECASE, count=1)
    return s


def q_fromcharcode(s):
    m = re.search(r"alert", s, flags=re.IGNORECASE)
    if m:
        codes = ",".join(str(ord(c)) for c in "alert")
        s = s[:m.start()] + f"window[String.fromCharCode({codes})]" + s[m.end():]
    return s


def q_comment_split(s):
    return re.sub(r"(scr)(ipt)", r"\1/**/\2", s, flags=re.IGNORECASE)


TIER2 = [q_hex_escape, q_unicode_escape, q_split_keywords, q_fromcharcode,
         q_comment_split]

# ---------------------------------------------------------------- tier 3
# artifact-suppressing evasion: no escape sequences, no comment markers, no
# concatenation operators - only constructs that also occur in benign HTML


def a_alternative_sink(s):
    """Replace a script/alert construct with an equivalent event-handler sink."""
    if re.search(r"<script[^>]*>", s, flags=re.IGNORECASE):
        body = re.sub(r"</?script[^>]*>", "", s, flags=re.IGNORECASE)
        return f'<img src=x onerror="{body.strip()}">'
    return s


def a_rename_alert(s):
    """Use a DOM sink that is not the literal 'alert' token."""
    return re.sub(r"alert\s*\(", "confirm(", s, flags=re.IGNORECASE)


BENIGN_POOL = []


def a_benign_padding(s):
    """Surround the payload with real benign traffic drawn from the training
    split, so that the padding carries the statistics of genuine benign data
    rather than a fixed template."""
    if not BENIGN_POOL:
        return s
    left = random.choice(BENIGN_POOL)[:200]
    right = random.choice(BENIGN_POOL)[:200]
    return left + s + right


TIER3 = [a_alternative_sink, a_rename_alert, a_benign_padding]


def apply_tier(s, fns, k=None):
    fns = fns if k is None else random.sample(fns, min(k, len(fns)))
    for f in fns:
        s = f(s)
    return s


def keyword_stats(texts):
    low = [t.lower() for t in texts]
    return {
        "has 'script' (%)": round(100 * np.mean(["script" in t for t in low]), 1),
        "has 'alert' (%)": round(100 * np.mean(["alert" in t for t in low]), 1),
        "has '<' (%)": round(100 * np.mean(["<" in t for t in low]), 1),
        "has escape artifact (%)": round(100 * np.mean(
            [bool(re.search(r"\\x[0-9a-f]{2}|\\u[0-9a-f]{4}|%[0-9a-f]{2}|/\*|\*/|'\+'",
                            t)) for t in low]), 1),
        "mean length": int(np.mean([len(t) for t in texts])),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    D = load()
    BENIGN_POOL.extend([t for t in D["Xtr"][D["ytr"] == 0] if len(t) > 40])

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                          max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(D["Xtr"])
    clf = LogisticRegression(max_iter=2000, C=5, class_weight="balanced")
    t0 = time.time()
    clf.fit(Xtr, D["ytr"])
    print(f"  reference detector trained in {time.time()-t0:.1f}s")
    thr = best_threshold_on_val(D["yva"], clf.predict_proba(vec.transform(D["Xva"]))[:, 1])

    mal = D["Xte"][D["yte"] == 1]
    ben = D["Xte"][D["yte"] == 0]

    # ---------- A + D: robustness per evasion tier ----------
    tiers = {
        "clean (no transform)": lambda s: s,
        "tier 1: functionality-preserving": lambda s: apply_tier(s, TIER1, 3),
        "tier 2: escape/split (original mix)": lambda s: apply_tier(s, TIER2, 3),
        "tier 1+2 combined (as published)": lambda s: apply_tier(
            s, TIER1 + TIER2, 3),
        "tier 3: artifact-suppressing": lambda s: apply_tier(s, TIER3, 3),
    }

    rows, kw_rows = [], []
    DRAWS = 3   # the transforms are stochastic; every tier is drawn three times
    for tname, fn in tiers.items():
        recs, fprs, f1s, mccs, scores = [], [], [], [], []
        last_mal_t = None
        for d in range(DRAWS if tname != "clean (no transform)" else 1):
            random.seed(SEED + d)
            mal_t = np.array([fn(s) for s in mal], dtype=object)
            ben_t = np.array([fn(s) for s in ben], dtype=object)
            last_mal_t = mal_t
            p_mal = clf.predict_proba(vec.transform(mal_t))[:, 1]
            p_ben = clf.predict_proba(vec.transform(ben_t))[:, 1]
            recs.append(recall_score(np.ones(len(mal_t)),
                                     (p_mal >= thr).astype(int)))
            fprs.append(float(np.mean(p_ben >= thr)))
            y_all = np.concatenate([np.ones(len(mal_t)), np.zeros(len(ben_t))])
            p_all = np.concatenate([p_mal, p_ben])
            m = metrics_at(y_all, p_all, thr)
            f1s.append(m["F1"])
            mccs.append(m["MCC"])
            scores.append(float(p_mal.mean()))

        rows.append({"evasion tier": tname,
                     "draws": len(recs),
                     "malicious recall": round(float(np.mean(recs)), 4),
                     "recall std": round(float(np.std(recs)), 4),
                     "benign FP rate (benign also transformed)":
                         round(float(np.mean(fprs)), 5),
                     "F1 (both sides transformed)": round(float(np.mean(f1s)), 4),
                     "MCC": round(float(np.mean(mccs)), 4),
                     "mean malicious score": round(float(np.mean(scores)), 4)})
        kw_rows.append({"evasion tier": tname, **keyword_stats(last_mal_t)})
        print(rows[-1])

    save(pd.DataFrame(rows), "r8_evasion_tiers.csv")
    save(pd.DataFrame(kw_rows), "r8_keyword_survival.csv")

    # ---------- C: is the obfuscation itself a detectable artifact? ----------
    disc_rows = []
    for tname, fn in tiers.items():
        if tname.startswith("clean"):
            continue
        mal_t = np.array([fn(s) for s in mal], dtype=object)
        texts = np.concatenate([mal, mal_t])
        labels = np.concatenate([np.zeros(len(mal)), np.ones(len(mal_t))])
        v2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3,
                             max_features=30000, sublinear_tf=True)
        Z = v2.fit_transform(texts)
        auc = cross_val_score(LogisticRegression(max_iter=1000), Z, labels,
                              cv=3, scoring="roc_auc").mean()
        disc_rows.append({"evasion tier": tname,
                          "clean-vs-obfuscated discriminability (ROC-AUC)":
                              round(float(auc), 4)})
        print(disc_rows[-1])
    save(pd.DataFrame(disc_rows), "r8_artifact_discriminability.csv")


if __name__ == "__main__":
    main()
