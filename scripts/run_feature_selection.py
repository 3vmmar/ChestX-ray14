#!/usr/bin/env python
"""
run_feature_selection.py — apply the DSAI 305 L02 feature selection taxonomy
to CNN bottleneck features.

Pipeline
    1. Extract penultimate embeddings from a trained backbone.
    2. Score every feature with each technique from the lecture:
         filter (information gain, chi-square, ANOVA, Relief)
         embedded (Lasso L1, tree importance)
         unsupervised (dispersion ratio, missing value ratio, variance)
    3. Build a consensus ranking across methods (slide 23).
    4. Run the wrapper methods (RFE, forward, backward) on a pre-filtered
       subset, because wrappers refit a model per candidate and do not scale
       to a full 1024-d embedding.
    5. Evaluate a classifier before and after selection (slide 24).

Usage:
    python scripts/run_feature_selection.py --model DenseNet121
    python scripts/run_feature_selection.py --model DenseNet121 --top-k 64 --wrapper-pool 40
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src import modelzoo  # noqa: E402
from src.feature_selection import methods as M  # noqa: E402
from src.feature_selection.extractor import extract_features  # noqa: E402
from scripts.dump_predictions import CLAHETransform, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / "feature_selection"
MODELS_ROOT = PROJECT_ROOT / "outputs" / "models"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"


def _metrics(y, prob, n_features):
    pred = (prob >= 0.5).astype(int)
    return {
        "n_features": int(n_features),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
    }


def eval_subset(X, y, cols, seed=42, folds=5):
    """CV performance on a FIXED column subset.

    NOTE: if `cols` was chosen using all of y, this number is optimistically
    biased -- the selector already saw the held-out folds. Use
    eval_subset_nested for an honest estimate.
    """
    Xs = StandardScaler().fit_transform(X[:, cols])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    prob = cross_val_predict(clf, Xs, y, cv=cv, method="predict_proba")[:, 1]
    return _metrics(y, prob, len(cols))


def eval_subset_nested(X, y, score_fn, k, seed=42, folds=5):
    """Honest estimate: re-run the selector INSIDE each training fold.

    score_fn(X_train, y_train) -> per-feature scores. The held-out fold never
    influences which features are kept, so the resulting AUC is not inflated
    by selection leakage.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    prob = np.zeros(len(y), dtype=float)
    for tr, te in cv.split(X, y):
        s = score_fn(X[tr], y[tr])
        cols = np.argsort(-np.asarray(s, dtype=float))[:k]
        sc = StandardScaler().fit(X[tr][:, cols])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=seed)
        clf.fit(sc.transform(X[tr][:, cols]), y[tr])
        prob[te] = clf.predict_proba(sc.transform(X[te][:, cols]))[:, 1]
    return _metrics(y, prob, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DenseNet121")
    ap.add_argument("--split", default="val")
    ap.add_argument("--top-k", type=int, default=64, help="features kept by the ranked methods")
    ap.add_argument("--wrapper-pool", type=int, default=40,
                    help="pre-filter size handed to the wrapper methods")
    ap.add_argument("--wrapper-k", type=int, default=15, help="features wrappers select")
    ap.add_argument("--relief-iters", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", action="store_true", default=True,
                    help="reuse features.npy if present")
    args = ap.parse_args()

    name = args.model
    dest = OUT_ROOT / name
    dest.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(DATA_SPLITS / f"{args.split}.csv")
    y = df["label"].astype(int).values
    print(f"model={name}  split={args.split}  n={len(df)}  positives={int(y.sum())}  device={device}")

    # ── 1. features ─────────────────────────────────────────────────────────
    fcache = dest / "features.npy"
    if args.cache and fcache.exists():
        X = np.load(fcache)
        print(f"  [1] cached features {X.shape}")
    else:
        print("  [1] extracting bottleneck features")
        size = modelzoo.img_size(name)
        tf = T.Compose([CLAHETransform(), T.Resize((size, size)), T.ToTensor(),
                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        X = extract_features(name, df, tf, device, args.batch_size, args.workers,
                             checkpoint=MODELS_ROOT / name / "best_model.pth")
        np.save(fcache, X)
    d = X.shape[1]

    # ── 2. score with every technique ───────────────────────────────────────
    print("  [2] scoring features")
    scores = {}
    for label, fn in M.FILTER_METHODS.items():
        scores[label] = (M.relief(X, y, n_iterations=args.relief_iters)
                         if label == "relief" else fn(X, y))
        print(f"      filter/{label}")
    for label, fn in M.EMBEDDED_METHODS.items():
        scores[label] = fn(X, y)
        print(f"      embedded/{label}")
    unsup = {}
    for label, fn in M.UNSUPERVISED_METHODS.items():
        unsup[label] = fn(X)
        print(f"      unsupervised/{label}")

    # ── 3. consensus ────────────────────────────────────────────────────────
    consensus = M.consensus_ranking(scores)
    top_consensus = np.argsort(consensus)[: args.top_k]

    # ── 4/5. evaluate every selection ───────────────────────────────────────
    print("  [3] evaluating selections (5-fold CV)")
    results = {}
    baseline = eval_subset(X, y, np.arange(d))
    results["ALL_FEATURES (baseline)"] = baseline
    print(f"      baseline           d={d:<5} AUC={baseline['roc_auc']:.4f}")

    # supervised scorers must be re-fitted inside each fold to avoid leakage
    nested_fns = {
        "information_gain": lambda Xt, yt: M.information_gain(Xt, yt),
        "chi_square": lambda Xt, yt: M.chi_square(Xt, yt),
        "anova_f": lambda Xt, yt: M.anova_f(Xt, yt),
        "relief": lambda Xt, yt: M.relief(Xt, yt, n_iterations=args.relief_iters),
        "lasso_l1": lambda Xt, yt: M.lasso_importance(Xt, yt),
        "tree_importance": lambda Xt, yt: M.tree_importance(Xt, yt),
        # unsupervised scorers ignore y, so leakage is not possible, but we
        # re-fit them per fold anyway for a like-for-like comparison
        "dispersion_ratio": lambda Xt, yt: M.dispersion_ratio(Xt),
        "missing_value_ratio": lambda Xt, yt: M.missing_value_ratio(Xt),
        "variance": lambda Xt, yt: M.variance_threshold_scores(Xt),
    }

    for label, s in {**scores, **unsup}.items():
        cols = np.argsort(-s)[: args.top_k]
        leaky = eval_subset(X, y, cols)
        honest = eval_subset_nested(X, y, nested_fns[label], args.top_k)
        leaky["note"] = "selection fitted on all data - optimistically biased"
        results[label] = honest
        results[label + " (leaky)"] = leaky
        print(f"      {label:<20} d={args.top_k:<5} "
              f"AUC={honest['roc_auc']:.4f} nested  |  {leaky['roc_auc']:.4f} leaky "
              f"({leaky['roc_auc'] - honest['roc_auc']:+.4f} bias)")

    def consensus_scores(Xt, yt):
        s = {}
        for lbl, f in M.FILTER_METHODS.items():
            s[lbl] = M.relief(Xt, yt, n_iterations=args.relief_iters) if lbl == "relief" else f(Xt, yt)
        for lbl, f in M.EMBEDDED_METHODS.items():
            s[lbl] = f(Xt, yt)
        return -M.consensus_ranking(s)  # lower mean rank = better, so negate

    results["consensus"] = eval_subset_nested(X, y, consensus_scores, args.top_k)
    results["consensus (leaky)"] = eval_subset(X, y, top_consensus)
    print(f"      {'consensus':<20} d={args.top_k:<5} "
          f"AUC={results['consensus']['roc_auc']:.4f} nested  |  "
          f"{results['consensus (leaky)']['roc_auc']:.4f} leaky")

    # ── wrappers: also nested, selection re-run inside every training fold ──
    print(f"  [4] wrappers, ANOVA-prefiltered pool of {args.wrapper_pool} "
          f"(exhaustive would need 2^{d}-1 subsets)")

    def make_wrapper_scorer(fn):
        """Turn a wrapper into a score_fn so it can be nested like the others."""
        def score_fn(Xt, yt):
            pool_t = np.argsort(-M.anova_f(Xt, yt))[: args.wrapper_pool]
            Xp_t = StandardScaler().fit_transform(Xt[:, pool_t])
            idx, _ = fn(Xp_t, yt, args.wrapper_k)
            s = np.zeros(Xt.shape[1])
            s[pool_t[idx]] = 1.0  # selected features score 1, rest 0
            return s
        return score_fn

    for label, fn in (("rfe", M.rfe_select),
                      ("forward_selection", M.forward_select),
                      ("backward_selection", M.backward_select)):
        pool = np.argsort(-scores["anova_f"])[: args.wrapper_pool]
        Xp = StandardScaler().fit_transform(X[:, pool])
        idx, _ = fn(Xp, y, args.wrapper_k)
        chosen = pool[idx]
        leaky = eval_subset(X, y, chosen)
        honest = eval_subset_nested(X, y, make_wrapper_scorer(fn), args.wrapper_k)
        results[label] = honest
        results[label]["selected_global_indices"] = [int(i) for i in chosen]
        results[label + " (leaky)"] = leaky
        print(f"      {label:<20} d={args.wrapper_k:<5} "
              f"AUC={honest['roc_auc']:.4f} nested  |  {leaky['roc_auc']:.4f} leaky "
              f"({leaky['roc_auc'] - honest['roc_auc']:+.4f} bias)")

    # ── save ────────────────────────────────────────────────────────────────
    np.save(dest / "consensus_rank.npy", consensus)
    for k, v in {**scores, **unsup}.items():
        np.save(dest / f"scores_{k}.npy", v)

    payload = {
        "model": name,
        "split": args.split,
        "n_samples": int(len(df)),
        "n_positives": int(y.sum()),
        "n_features": int(d),
        "top_k": args.top_k,
        "wrapper_pool": args.wrapper_pool,
        "wrapper_k": args.wrapper_k,
        "exhaustive_subsets_required": f"2^{d}-1",
        "results": results,
    }
    (dest / "selection_results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nsaved -> {dest / 'selection_results.json'}")

    honest_keys = [k for k in results if "(leaky)" not in k and k != "ALL_FEATURES (baseline)"]
    best = max(honest_keys, key=lambda k: results[k]["roc_auc"])
    print(f"\nbest by LEAK-FREE nested CV AUC: {best}  ({results[best]['roc_auc']:.4f} "
          f"on {results[best]['n_features']} features vs baseline "
          f"{baseline['roc_auc']:.4f} on {d})")
    print("leaky columns are reported only to quantify selection bias - do not cite them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
