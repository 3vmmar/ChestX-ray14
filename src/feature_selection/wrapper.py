"""
FeatureSelector — implements Recursive Feature Elimination with
Cross-Validation (RFECV) on bottleneck features.

Provides:
    - RFECV fitting with configurable estimator, CV, scoring.
    - Before/after performance comparison on train/val/test splits.
    - Plotting the RFECV performance curve.
    - Saving/loading selection results.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import joblib


class FeatureSelector:
    """
    RFECV-based feature selection with before/after evaluation.

    Parameters
    ----------
    estimator : sklearn estimator or None
        Classifier used for RFECV and evaluation.
        Default: RandomForestClassifier(100, random_state=42, class_weight="balanced").
    scoring : str
        Scoring metric for RFECV (default "f1_macro").
    cv : int
        Number of cross-validation folds (default 5).
    step : float or int
        Fraction (if < 1) or number (if >= 1) of features to remove per
        iteration (default 0.1).
    min_features_to_select : int
        Minimum number of features to retain (default 50).
    n_jobs : int
        Parallel jobs for RFECV (default -1).
    random_state : int
        Random state for reproducibility (default 42).
    """

    def __init__(
        self,
        estimator=None,
        scoring="f1_macro",
        cv=5,
        step=0.1,
        min_features_to_select=50,
        n_jobs=-1,
        random_state=42,
    ):
        if estimator is None:
            self.estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight="balanced",
            )
        else:
            self.estimator = estimator

        self.scoring = scoring
        self.cv = cv
        self.step = step
        self.min_features_to_select = min_features_to_select
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.selector_ = None
        self.selected_features_ = None
        self.ranking_ = None
        self.n_features_ = None
        self.cv_scores_ = None
        self.metrics_comparison_ = None

    def fit(self, X, y):
        """
        Run RFECV on the training data.

        Parameters
        ----------
        X : np.ndarray
            Training features (N, D).
        y : np.ndarray
            Training labels (N,).

        Returns
        -------
        self
        """
        self.selector_ = RFECV(
            estimator=self.estimator,
            step=self.step,
            cv=self.cv,
            scoring=self.scoring,
            min_features_to_select=self.min_features_to_select,
            n_jobs=self.n_jobs,
        )

        self.selector_.fit(X, y)

        self.selected_features_ = self.selector_.support_
        self.ranking_ = self.selector_.ranking_
        self.n_features_ = int(self.selector_.n_features_)
        self.cv_scores_ = self.selector_.cv_results_["mean_test_score"]

        n_total = X.shape[1]
        print(f"  RFECV complete: {self.n_features_}/{n_total} features selected")
        print(f"  Best CV {self.scoring}: {self.cv_scores_.max():.4f}")

        return self

    def transform(self, X):
        """
        Reduce feature matrix to selected features only.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        X_selected : np.ndarray
        """
        if self.selector_ is None:
            raise RuntimeError("FeatureSelector has not been fitted yet. Call .fit() first.")
        return self.selector_.transform(X)

    def _compute_metrics(self, y_true, y_pred, y_prob=None):
        """Compute a dict of metrics."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        if y_prob is not None:
            try:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
            except Exception:
                metrics["roc_auc_ovr_macro"] = None
        return metrics

    def evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Compare classifier performance on ALL features vs SELECTED features.

        Trains the same estimator on:
          1. All features (baseline)
          2. Only the selected features

        Parameters
        ----------
        X_train, y_train : training data
        X_val,   y_val   : validation data
        X_test,  y_test  : test data

        Returns
        -------
        metrics_comparison : dict
        """
        if self.selector_ is None:
            raise RuntimeError("FeatureSelector has not been fitted yet. Call .fit() first.")

        X_train_sel = self.transform(X_train)
        X_val_sel = self.transform(X_val)
        X_test_sel = self.transform(X_test)

        clf = self.estimator.__class__(**self.estimator.get_params())

        # ── Baseline: all features ──
        clf.fit(X_train, y_train)
        val_pred = clf.predict(X_val)
        val_prob = clf.predict_proba(X_val)
        test_pred = clf.predict(X_test)
        test_prob = clf.predict_proba(X_test)

        baseline_val = self._compute_metrics(y_val, val_pred, val_prob)
        baseline_test = self._compute_metrics(y_test, test_pred, test_prob)

        # ── After selection ──
        clf.fit(X_train_sel, y_train)
        val_pred_sel = clf.predict(X_val_sel)
        val_prob_sel = clf.predict_proba(X_val_sel)
        test_pred_sel = clf.predict(X_test_sel)
        test_prob_sel = clf.predict_proba(X_test_sel)

        selected_val = self._compute_metrics(y_val, val_pred_sel, val_prob_sel)
        selected_test = self._compute_metrics(y_test, test_pred_sel, test_prob_sel)

        self.metrics_comparison_ = {
            "n_features_total": int(X_train.shape[1]),
            "n_features_selected": self.n_features_,
            "feature_reduction_pct": round(
                100 * (1 - self.n_features_ / X_train.shape[1]), 2
            ),
            "baseline_all_features": {
                "val": baseline_val,
                "test": baseline_test,
            },
            "selected_features": {
                "val": selected_val,
                "test": selected_test,
            },
        }

        print()
        print("  " + "=" * 65)
        print("  |            Performance: Before vs After Selection           |")
        print("  " + "=" * 65)
        print("  | {:<22} | {:<8} | {:<8} | {:<13} |".format("Metric", "All Feat", "Selected", "Change"))
        print("  " + "-" * 65)
        for split_name, split_key in [("Validation", "val"), ("Test", "test")]:
            before = self.metrics_comparison_["baseline_all_features"][split_key]
            after = self.metrics_comparison_["selected_features"][split_key]
            print("  |--- {:<54}---|".format(split_name))
            for metric in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovr_macro"]:
                b = before.get(metric, "N/A")
                a = after.get(metric, "N/A")
                if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                    change = f"{a - b:+.4f}"
                else:
                    change = "N/A"
                b_str = f"{b:.4f}" if isinstance(b, float) else str(b)
                a_str = f"{a:.4f}" if isinstance(a, float) else str(a)
                print("  | {:<22} | {:<8} | {:<8} | {:<13} |".format(metric, b_str, a_str, change))
        print("  " + "=" * 65)

        return self.metrics_comparison_

    def plot_results(self, save_path):
        """
        Plot RFECV cross-validation score vs number of features.

        Parameters
        ----------
        save_path : str or Path
        """
        if self.cv_scores_ is None:
            raise RuntimeError("No CV results. Call .fit() first.")

        n_features_range = self.selector_.cv_results_["n_features"]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(n_features_range, self.cv_scores_, "o-", color="#3498db", linewidth=2, markersize=4)
        ax.axvline(self.n_features_, color="#e74c3c", linestyle="--", linewidth=1.5,
                   label=f"Optimal: {self.n_features_} features")
        ax.fill_between(n_features_range, self.cv_scores_, alpha=0.15, color="#3498db")

        ax.set_xlabel("Number of Features", fontsize=12)
        ax.set_ylabel(f"CV {self.scoring}", fontsize=12)
        ax.set_title("RFECV — Feature Selection Performance Curve", fontweight="bold", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVED] RFECV curve -> {save_path}")

    def save(self, output_dir):
        """
        Save selection results (mask, ranking, metrics JSON, fitted selector).

        Parameters
        ----------
        output_dir : str or Path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(str(output_dir / "selected_features.npy"), self.selected_features_)
        print(f"  [SAVED] {output_dir / 'selected_features.npy'}  shape={self.selected_features_.shape}")

        np.save(str(output_dir / "feature_ranking.npy"), self.ranking_)
        print(f"  [SAVED] {output_dir / 'feature_ranking.npy'}  shape={self.ranking_.shape}")

        if self.metrics_comparison_ is not None:
            path = output_dir / "metrics_before_after.json"
            with open(path, "w") as f:
                json.dump(self.metrics_comparison_, f, indent=2)
            print(f"  [SAVED] {path}")

        selector_path = output_dir / "selector.pkl"
        joblib.dump(self.selector_, str(selector_path))
        print(f"  [SAVED] {selector_path}")

    @staticmethod
    def load_results(input_dir):
        """
        Load saved selection mask and ranking.

        Parameters
        ----------
        input_dir : str or Path

        Returns
        -------
        dict with keys: selected_features, ranking
        """
        input_dir = Path(input_dir)
        return {
            "selected_features": np.load(str(input_dir / "selected_features.npy")),
            "ranking": np.load(str(input_dir / "feature_ranking.npy")),
        }
