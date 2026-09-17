"""
methods.py — feature selection techniques from DSAI 305 L02.

Organised exactly as the lecture's taxonomy:

  Supervised
    Filter    : Information Gain, Chi-square, ANOVA F-test, Relief
    Wrapper   : Forward selection, Backward selection, RFE, (Exhaustive)
    Embedded  : Lasso (L1) regularisation, Tree-based importance
  Unsupervised
    Dispersion Ratio (AM/GM), Missing Value Ratio, Variance Threshold

Every scorer returns a 1-D array of per-feature scores where HIGHER MEANS
MORE RELEVANT, so rankings compose (lecture slide 23: rank per metric, then
look for features that rank high across several).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SequentialFeatureSelector,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

# ════════════════════════════════════════════════════════════════════════════
# FILTER METHODS
# ════════════════════════════════════════════════════════════════════════════


def information_gain(X, y, random_state=42):
    """IG(X,Y) = H(Y) - H(Y|X), estimated by sklearn's mutual information."""
    return mutual_info_classif(X, y, random_state=random_state)


def chi_square(X, y):
    """Chi-square statistic. Requires non-negative features, so we min-max scale."""
    Xs = MinMaxScaler().fit_transform(X)
    stat, _ = chi2(Xs, y)
    return np.nan_to_num(stat, nan=0.0)


def anova_f(X, y):
    """ANOVA F = MSB/MSW. High F => group means separated relative to spread."""
    stat, _ = f_classif(X, y)
    return np.nan_to_num(stat, nan=0.0)


def relief(X, y, n_iterations=200, random_state=42):
    """Original Relief (binary classification), per lecture slides 20-22.

    Repeat n_iterations times:
      pick a random instance R, find its nearest same-class neighbour
      (near-hit) and nearest different-class neighbour (near-miss), then
          W[f] -= diff(f, R, near_hit) / m
          W[f] += diff(f, R, near_miss) / m
    Features that separate the classes accumulate positive weight.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    n, d = X.shape

    rngs = X.max(axis=0) - X.min(axis=0)
    rngs[rngs == 0] = 1.0  # constant features contribute nothing

    Xn = X / rngs  # normalise so each feature's diff is on [0,1]
    W = np.zeros(d)
    idx_by_class = {c: np.flatnonzero(y == c) for c in np.unique(y)}
    m = min(n_iterations, n)
    picks = rng.choice(n, size=m, replace=False)

    for i in picks:
        same = idx_by_class[y[i]]
        other = np.concatenate([v for c, v in idx_by_class.items() if c != y[i]])

        d_same = np.abs(Xn[same] - Xn[i]).sum(axis=1)
        d_same[same == i] = np.inf              # exclude self
        hit = same[np.argmin(d_same)]

        d_other = np.abs(Xn[other] - Xn[i]).sum(axis=1)
        miss = other[np.argmin(d_other)]

        W -= np.abs(Xn[i] - Xn[hit]) / m
        W += np.abs(Xn[i] - Xn[miss]) / m
    return W


# ════════════════════════════════════════════════════════════════════════════
# EMBEDDED METHODS
# ════════════════════════════════════════════════════════════════════════════


def lasso_importance(X, y, C=0.05, random_state=42, max_iter=2000):
    """L1-penalised logistic regression; |coefficient| is the score.

    L1 drives uninformative coefficients to exactly zero, which IS the
    selection (lecture slide 32).
    """
    clf = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                             max_iter=max_iter, random_state=random_state,
                             class_weight="balanced")
    clf.fit(X, y)
    return np.abs(clf.coef_).ravel()


def tree_importance(X, y, n_estimators=300, random_state=42):
    """Random-forest impurity decrease per feature (lecture slide 32)."""
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                 class_weight="balanced", n_jobs=-1)
    clf.fit(X, y)
    return clf.feature_importances_


# ════════════════════════════════════════════════════════════════════════════
# WRAPPER METHODS
# ════════════════════════════════════════════════════════════════════════════


def _default_estimator(random_state=42):
    return LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=random_state)


def rfe_select(X, y, n_features, estimator=None, step=0.1, random_state=42):
    """Recursive Feature Elimination: rank, drop the weakest, repeat."""
    est = estimator or _default_estimator(random_state)
    sel = RFE(est, n_features_to_select=n_features, step=step).fit(X, y)
    return np.flatnonzero(sel.support_), sel


def forward_select(X, y, n_features, estimator=None, cv=3, scoring="roc_auc",
                   random_state=42, n_jobs=-1):
    """Forward selection: start empty, greedily add the best feature."""
    est = estimator or _default_estimator(random_state)
    sel = SequentialFeatureSelector(est, n_features_to_select=n_features,
                                    direction="forward", cv=cv,
                                    scoring=scoring, n_jobs=n_jobs).fit(X, y)
    return np.flatnonzero(sel.get_support()), sel


def backward_select(X, y, n_features, estimator=None, cv=3, scoring="roc_auc",
                    random_state=42, n_jobs=-1):
    """Backward selection: start full, greedily drop the least useful feature."""
    est = estimator or _default_estimator(random_state)
    sel = SequentialFeatureSelector(est, n_features_to_select=n_features,
                                    direction="backward", cv=cv,
                                    scoring=scoring, n_jobs=n_jobs).fit(X, y)
    return np.flatnonzero(sel.get_support()), sel


def exhaustive_cost(n_features):
    """Number of subsets brute force would evaluate: 2^d - 1.

    The lecture lists Exhaustive selection as an option; this function exists
    to show why it is not runnable on CNN embeddings.
    """
    return 2 ** int(n_features) - 1


# ════════════════════════════════════════════════════════════════════════════
# UNSUPERVISED METHODS
# ════════════════════════════════════════════════════════════════════════════


def dispersion_ratio(X, eps=1e-12):
    """Arithmetic mean / geometric mean, per feature (lecture slide 35).

    Higher ratio => more dispersed => more potentially informative. Defined
    for positive values, so a small epsilon shifts away from zero (ReLU
    embeddings contain exact zeros).
    """
    Xp = np.asarray(X, dtype=np.float64) + eps
    am = Xp.mean(axis=0)
    gm = np.exp(np.log(Xp).mean(axis=0))
    return am / np.maximum(gm, eps)


def missing_value_ratio(X):
    """Fraction of missing values per feature (lecture slide 36).

    Returned as a RELEVANCE score (1 - ratio) so that, like every other
    scorer here, higher means keep.
    """
    X = np.asarray(X, dtype=np.float64)
    return 1.0 - np.isnan(X).mean(axis=0)


def variance_threshold_scores(X):
    """Plain per-feature variance (the unsupervised baseline)."""
    return np.asarray(X, dtype=np.float64).var(axis=0)


# ════════════════════════════════════════════════════════════════════════════
# RANK AGGREGATION  (lecture slide 23)
# ════════════════════════════════════════════════════════════════════════════


def rank_features(scores):
    """Convert scores to ranks: rank 1 = most relevant."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def consensus_ranking(score_dict, weights=None):
    """Average the per-method rankings (optionally weighted).

    Slide 23: 'Look for features that consistently rank high across multiple
    metrics. These features are likely to be more robust and informative.'
    """
    names = list(score_dict)
    w = np.array([1.0] * len(names) if weights is None
                 else [weights.get(n, 1.0) for n in names], dtype=float)
    w = w / w.sum()
    R = np.vstack([rank_features(score_dict[n]) for n in names]).astype(float)
    return (R * w[:, None]).sum(axis=0)


FILTER_METHODS = {
    "information_gain": information_gain,
    "chi_square": chi_square,
    "anova_f": anova_f,
    "relief": relief,
}

EMBEDDED_METHODS = {
    "lasso_l1": lasso_importance,
    "tree_importance": tree_importance,
}

UNSUPERVISED_METHODS = {
    "dispersion_ratio": dispersion_ratio,
    "missing_value_ratio": missing_value_ratio,
    "variance": variance_threshold_scores,
}
