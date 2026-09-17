"""Correctness tests for the DeLong implementation.

Run:  python -m pytest tests/test_delong.py -v
  or: python tests/test_delong.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.stats.delong import delong_roc_test, delong_roc_variance  # noqa: E402


def test_auc_matches_sklearn():
    """The AUC DeLong computes must equal sklearn's, or its variance is meaningless."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        n_pos, n_neg = int(rng.integers(40, 300)), int(rng.integers(400, 2500))
        y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
        s = np.r_[rng.normal(1.0, 1, n_pos), rng.normal(0, 1, n_neg)]
        idx = rng.permutation(len(y))
        y, s = y[idx], s[idx]
        assert np.isclose(delong_roc_variance(y, s)[0], roc_auc_score(y, s), atol=1e-12)


def test_ties_handled():
    """Midrank handling: heavily tied discrete scores must still match sklearn."""
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(100), np.zeros(400)]
    s = rng.integers(0, 5, 500).astype(float)
    assert np.isclose(delong_roc_variance(y, s)[0], roc_auc_score(y, s), atol=1e-12)


def test_identical_models_not_significant():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(186), np.zeros(2425)]
    s = np.r_[rng.normal(0.7, 1, 186), rng.normal(0, 1, 2425)]
    r = delong_roc_test(y, s, s.copy())
    assert r["auc_diff"] == 0.0
    assert r["p_value"] == 1.0


def test_clearly_different_models_are_significant():
    rng = np.random.default_rng(3)
    y = np.r_[np.ones(186), np.zeros(2425)]
    strong = np.r_[rng.normal(1.6, 1, 186), rng.normal(0, 1, 2425)]
    weak = np.r_[rng.normal(0.3, 1, 186), rng.normal(0, 1, 2425)]
    r = delong_roc_test(y, strong, weak)
    assert r["auc_a"] > r["auc_b"]
    assert r["p_value"] < 1e-10


def test_pairing_beats_independent_intervals():
    """The paired SE must be smaller than the naive independent-difference SE.

    This is the whole reason to use DeLong rather than overlapping CIs.
    """
    rng = np.random.default_rng(4)
    y = np.r_[np.ones(186), np.zeros(2425)]
    shared = np.r_[rng.normal(0.8, 1, 186), rng.normal(0, 1, 2425)]
    a = shared + rng.normal(0, 0.15, len(y))   # two correlated models
    b = shared + rng.normal(0, 0.15, len(y))
    r = delong_roc_test(y, a, b)
    naive = float(np.sqrt(r["se_a"] ** 2 + r["se_b"] ** 2))
    assert r["se_diff"] < naive
    assert r["cov_ab"] > 0


def test_rejects_single_class():
    y = np.ones(50)
    try:
        delong_roc_variance(y, np.linspace(0, 1, 50))
    except ValueError:
        return
    raise AssertionError("expected ValueError for single-class y")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"  PASS  {f.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
