#!/usr/bin/env python
"""
build_multilabel_splits.py — patient-wise splits over the FULL NIH release.

The binary splits in data/splits cap every class at ~1,000 images, which uses
17,172 of the 112,120 available films (15%). Pneumonia positives are capped by
the dataset itself at 1,431 and all of them are already in use, so capping buys
nothing on the positive side while discarding 95,000 negatives and, more
importantly, the other 13 finding labels.

This builds uncapped, patient-wise 70/15/15 splits carrying all 14 NIH labels
as separate binary columns, plus the pneumonia column the binary task uses.
Training on all 14 at once is the CheXNet recipe: the auxiliary labels supply
representation learning that a lone binary head cannot get from 1,004
positives.

Patients, not images, are assigned to splits - the same patient's follow-up
films must never straddle an evaluation boundary. Assignment is stratified on
whether a patient has any pneumonia film, so the rare positives stay
proportionally represented.

Usage:
    python scripts/build_multilabel_splits.py
    python scripts/build_multilabel_splits.py --out-dir data/splits_multilabel
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = PROJECT_ROOT / "data" / "archive"
DEFAULT_OUT = PROJECT_ROOT / "data" / "splits_multilabel"

FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
TARGET = "Pneumonia"


def index_images(archive):
    """Map image filename -> absolute path across every images_XXX folder."""
    out = {}
    for d in sorted(archive.glob("images_*")):
        sub = d / "images"
        if not sub.exists():
            continue
        for p in sub.glob("*.png"):
            out[p.name] = str(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(ARCHIVE / "Data_Entry_2017.csv")
    meta = meta.rename(columns={
        "Image Index": "image_name", "Finding Labels": "finding_labels",
        "Patient ID": "patient_id", "Patient Age": "patient_age",
        "Patient Gender": "patient_gender", "View Position": "view_position",
    })
    print(f"Data_Entry_2017: {len(meta):,} rows, {meta.patient_id.nunique():,} patients")

    print("indexing images on disk...")
    idx = index_images(ARCHIVE)
    meta["image_path"] = meta.image_name.map(idx)
    missing = int(meta.image_path.isna().sum())
    if missing:
        print(f"  WARNING: {missing:,} rows have no image on disk; dropping them")
        meta = meta[meta.image_path.notna()].reset_index(drop=True)
    print(f"  resolved {len(meta):,} images")

    # one binary column per finding
    dummies = meta.finding_labels.str.get_dummies("|")
    for f in FINDINGS:
        meta[f] = dummies[f] if f in dummies.columns else 0
    meta["label"] = meta[TARGET]
    meta["n_findings"] = meta[FINDINGS].sum(axis=1)

    # ---- patient-wise, stratified on "patient ever has pneumonia" ----------
    rng = np.random.default_rng(args.seed)
    per_patient = meta.groupby("patient_id")[TARGET].max()
    splits = {}
    for flag in (1, 0):
        pids = per_patient[per_patient == flag].index.to_numpy()
        rng.shuffle(pids)
        n = len(pids)
        n_val = int(round(n * args.val_frac))
        n_test = int(round(n * args.test_frac))
        splits.setdefault("val", []).append(pids[:n_val])
        splits.setdefault("test", []).append(pids[n_val:n_val + n_test])
        splits.setdefault("train", []).append(pids[n_val + n_test:])
    assign = {k: set(np.concatenate(v)) for k, v in splits.items()}

    # sanity: every patient in exactly one split
    all_ids = set(per_patient.index)
    assert sum(len(v) for v in assign.values()) == len(all_ids), "patient count mismatch"
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                assert not (assign[a] & assign[b]), f"patient overlap {a}/{b}"

    cols = (["image_path", "image_name", "label", "patient_id", "view_position",
             "patient_age", "patient_gender", "finding_labels", "n_findings"] + FINDINGS)
    summary = {"seed": args.seed, "source": "Data_Entry_2017.csv (full release)",
               "findings": FINDINGS, "splits": {}}

    for name in ("train", "val", "test"):
        d = meta[meta.patient_id.isin(assign[name])][cols].reset_index(drop=True)
        d.to_csv(out / f"{name}.csv", index=False)
        pos = int(d.label.sum())
        summary["splits"][name] = {
            "images": len(d), "patients": int(d.patient_id.nunique()),
            "pneumonia_positives": pos,
            "pneumonia_rate_%": round(100 * pos / len(d), 3),
            "per_finding": {f: int(d[f].sum()) for f in FINDINGS},
        }
        print(f"  {name:<6} {len(d):>7,} images  {d.patient_id.nunique():>6,} patients  "
              f"pneumonia {pos:>5,} ({100*pos/len(d):.2f}%)")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}/train.csv, val.csv, test.csv, summary.json")

    b = PROJECT_ROOT / "data" / "splits"
    if b.exists():
        old = sum(len(pd.read_csv(b / f"{n}.csv")) for n in ("train", "val", "test"))
        new = sum(summary["splits"][n]["images"] for n in ("train", "val", "test"))
        print(f"\nbinary splits used {old:,} images; these use {new:,} "
              f"({new/old:.1f}x more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
