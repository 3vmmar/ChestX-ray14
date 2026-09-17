#!/usr/bin/env python
"""
compare_models.py — paired DeLong test between two models' ROC curves.

Reads the per-image probability files written by scripts/dump_predictions.py
and asks whether the AUC difference is larger than sampling noise. Because
both models are scored on the SAME images, the paired DeLong test is used
rather than comparing two independent confidence intervals -- it accounts
for the fact that the models agree on most cases and is far more sensitive.

Usage:
    python scripts/compare_models.py --a ResNet101 --b EfficientNetB3
    python scripts/compare_models.py --a DenseNet121 --b DenseNet121 \
        --score-a p_tta --score-b p_single --label-a "TTA" --label-b "single-pass"
    python scripts/compare_models.py --all-pairs
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stats.delong import delong_roc_test, format_report  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "outputs" / "models"
_SUFFIX = ""
CMP_DIR = PROJECT_ROOT / "outputs" / "comparisons"


def load_probs(model, suffix=""):
    p = OUT_ROOT / model / f"val_probs{suffix}.csv"
    if not p.exists():
        raise SystemExit(
            f"missing {p}\n"
            f"run:  python scripts/dump_predictions.py --model {model}"
        )
    return pd.read_csv(p)


def aligned(df_a, df_b):
    """Ensure both frames describe the same images in the same order."""
    if len(df_a) != len(df_b):
        raise SystemExit(f"row count differs: {len(df_a)} vs {len(df_b)}")
    if not (df_a["image_path"].values == df_b["image_path"].values).all():
        key = "image_path"
        df_b = df_b.set_index(key).loc[df_a[key]].reset_index()
    if not (df_a["label"].values == df_b["label"].values).all():
        raise SystemExit("labels disagree between the two files")
    return df_a, df_b


def compare(model_a, model_b, score_a, score_b, label_a, label_b, alpha):
    da = load_probs(model_a, _SUFFIX)
    db = load_probs(model_b, _SUFFIX) if model_b != model_a else da
    da, db = aligned(da, db)
    for col, who in ((score_a, model_a), (score_b, model_b)):
        if col not in da.columns:
            raise SystemExit(f"{who}: no column {col!r}; have {list(da.columns)}")
    r = delong_roc_test(
        da["label"].values, da[score_a].values, db[score_b].values,
        alpha=alpha, names=(label_a, label_b),
    )
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="first model name")
    ap.add_argument("--b", help="second model name")
    ap.add_argument("--score-a", default="p_tta")
    ap.add_argument("--score-b", default="p_tta")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--all-pairs", action="store_true",
                    help="test every pair that has a val_probs.csv")
    ap.add_argument("--save", action="store_true", help="write JSON to outputs/comparisons/")
    ap.add_argument("--suffix", default="", help="probability-file suffix")
    args = ap.parse_args()
    global _SUFFIX
    _SUFFIX = args.suffix

    if args.all_pairs:
        avail = sorted(p.parent.name for p in OUT_ROOT.glob("*/val_probs.csv"))
        if len(avail) < 2:
            raise SystemExit(
                f"need >=2 models with val_probs.csv, found {len(avail)}: {avail}\n"
                "run scripts/dump_predictions.py first"
            )
        rows = []
        for a, b in itertools.combinations(avail, 2):
            r = compare(a, b, "p_tta", "p_tta", a, b, args.alpha)
            rows.append(r)
            print(format_report(r))
            print()
        if args.save:
            CMP_DIR.mkdir(parents=True, exist_ok=True)
            (CMP_DIR / "all_pairs_delong.json").write_text(json.dumps(rows, indent=2))
            print(f"saved -> {CMP_DIR / 'all_pairs_delong.json'}")
        return 0

    if not (args.a and args.b):
        ap.error("pass --a and --b, or --all-pairs")

    la = args.label_a or (args.a if args.a != args.b else args.score_a)
    lb = args.label_b or (args.b if args.a != args.b else args.score_b)
    r = compare(args.a, args.b, args.score_a, args.score_b, la, lb, args.alpha)
    print(format_report(r))
    if args.save:
        CMP_DIR.mkdir(parents=True, exist_ok=True)
        dest = CMP_DIR / f"delong_{args.a}_{args.score_a}_vs_{args.b}_{args.score_b}.json"
        dest.write_text(json.dumps(r, indent=2))
        print(f"\nsaved -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
