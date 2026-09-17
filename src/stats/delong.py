"""
delong.py — Fast DeLong test for comparing two correlated ROC curves.

Implements the O(N log N) algorithm of Sun & Xu (2014), "Fast Implementation
of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver
Operating Characteristic Curves", IEEE Signal Processing Letters 21(11).

The test is *paired*: both models must be scored on the same images in the
same order. That is what makes it far more sensitive than comparing two
independent confidence intervals — it accounts for the fact that the two
models agree on most cases.

Typical use:
    stats = delong_roc_test(y_true, probs_a, probs_b)
    print(stats["p_value"])
"""

import numpy as np
from scipy import stats as _st


def _compute_midrank(x):
    """Midranks of x, handling ties by averaging (needed for DeLong)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Core estimator.

    Parameters
    ----------
    predictions_sorted_transposed : (k, n) array
        k models scored on n samples, POSITIVES FIRST.
    label_1_count : int
        Number of positive samples (m).

    Returns
    -------
    aucs : (k,) array
    delongcov : (k, k) covariance matrix of the AUC estimates
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def _prepare(y_true, *score_arrays):
    y_true = np.asarray(y_true).ravel()
    if not np.array_equal(np.unique(y_true), np.array([0, 1])):
        uniq = np.unique(y_true)
        if not set(uniq.tolist()) <= {0, 1}:
            raise ValueError(f"y_true must be binary 0/1, got values {uniq}")
    order = np.argsort(-y_true, kind="mergesort")  # positives first, stable
    label_1_count = int(np.sum(y_true == 1))
    if label_1_count == 0 or label_1_count == len(y_true):
        raise ValueError("y_true must contain both classes")
    stacked = np.vstack([np.asarray(s, dtype=float).ravel()[order] for s in score_arrays])
    return stacked, label_1_count


def delong_roc_variance(y_true, y_score):
    """AUC and its DeLong variance for a single model."""
    stacked, m = _prepare(y_true, y_score)
    aucs, cov = _fast_delong(stacked, m)
    return float(aucs[0]), float(cov[0, 0])


def delong_roc_test(y_true, y_score_a, y_score_b, alpha=0.05, names=("A", "B")):
    """Paired DeLong test that AUC(A) == AUC(B) on the same samples.

    Returns a dict with both AUCs, their difference, the z statistic, the
    two-sided p-value, and a confidence interval on the difference.
    """
    stacked, m = _prepare(y_true, y_score_a, y_score_b)
    aucs, cov = _fast_delong(stacked, m)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    diff = auc_a - auc_b

    var_diff = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var_diff <= 0:
        # identical (or perfectly correlated) score vectors
        z = 0.0
        p = 1.0
        se = 0.0
    else:
        se = float(np.sqrt(var_diff))
        z = diff / se
        p = float(2 * _st.norm.sf(abs(z)))

    crit = _st.norm.ppf(1 - alpha / 2)
    return {
        "name_a": names[0],
        "name_b": names[1],
        "auc_a": auc_a,
        "auc_b": auc_b,
        "se_a": float(np.sqrt(cov[0, 0])),
        "se_b": float(np.sqrt(cov[1, 1])),
        "cov_ab": float(cov[0, 1]),
        "auc_diff": diff,
        "se_diff": se,
        "z": z,
        "p_value": p,
        "ci_low": diff - crit * se,
        "ci_high": diff + crit * se,
        "alpha": alpha,
        "n_pos": m,
        "n_neg": int(stacked.shape[1] - m),
    }


def format_report(r):
    """Human-readable summary of a delong_roc_test result."""
    verdict = (
        "REJECT H0 - the AUCs differ significantly"
        if r["p_value"] <= r["alpha"]
        else "FAIL TO REJECT H0 - no significant difference"
    )
    return "\n".join(
        [
            f"DeLong paired test  ({r['name_a']} vs {r['name_b']})",
            f"  n = {r['n_pos'] + r['n_neg']}  ({r['n_pos']} positive, {r['n_neg']} negative)",
            f"  AUC {r['name_a']:<18} = {r['auc_a']:.4f}  (SE {r['se_a']:.4f})",
            f"  AUC {r['name_b']:<18} = {r['auc_b']:.4f}  (SE {r['se_b']:.4f})",
            f"  difference           = {r['auc_diff']:+.4f}  (SE {r['se_diff']:.4f})",
            f"  {int((1 - r['alpha']) * 100)}% CI on difference = [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
            f"  z = {r['z']:.4f}",
            f"  p = {r['p_value']:.4f}",
            f"  -> {verdict} (alpha = {r['alpha']})",
        ]
    )
