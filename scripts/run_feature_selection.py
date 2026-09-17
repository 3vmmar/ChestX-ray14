#!/usr/bin/env python
"""
run_feature_selection.py
========================
Wrapper-based feature selection via RFECV on bottleneck features from
any trained model in the architecture registry.

Pipeline per model:
  1. Load trained checkpoint (if available).
  2. Extract bottleneck features for train/val/test.
  3. Run RFECV on training features to find optimal subset.
  4. Evaluate classifier before and after feature selection.
  5. Save results to outputs/feature_selection/<ModelName>/.

Usage:
    # All models with checkpoints
    python scripts/run_feature_selection.py

    # Specific model
    python scripts/run_feature_selection.py --model DenseNet121

    # List available models
    python scripts/run_feature_selection.py --list-models
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ── Project paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection import (
    FeatureExtractor,
    FeatureSelector,
    ARCHITECTURES,
    list_available_models,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
RANDOM_SEED = 42
IMG_SIZE = 288
BATCH_SIZE = 64
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
TRAIN_CSV = DATA_SPLITS / "train.csv"
VAL_CSV = DATA_SPLITS / "val.csv"
TEST_CSV = DATA_SPLITS / "test.csv"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "feature_selection"
CKPT_ROOT = PROJECT_ROOT / "outputs"

NUM_CLASSES = 12

# ══════════════════════════════════════════════════════════════════════════════
# CLAHE TRANSFORM (from training notebook)
# ══════════════════════════════════════════════════════════════════════════════
class CLAHETransform:
    def __call__(self, pil_img):
        if max(pil_img.size) > 512:
            pil_img = pil_img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(pil_img.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_np)
        return Image.fromarray(enhanced).convert("RGB")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_paths = self.df["image_path"].tolist()
        self.labels = self.df["label"].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            try:
                img = self.transform(img)
            except MemoryError:
                img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return img, label, 1.0


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def find_available_models():
    available = []
    for name in list_available_models():
        ckpt_path = CKPT_ROOT / name / "best_model.pth"
        if ckpt_path.exists():
            available.append((name, ckpt_path))
    return available


def process_model(model_name, ckpt_path, df_train, df_val, df_test,
                  val_transforms, force=False):
    """Run the full feature-selection pipeline for one model."""
    OUTPUT_DIR = OUTPUT_ROOT / model_name

    # Check if already cached
    cached = all((OUTPUT_DIR / f"{k}.npy").exists()
                 for k in ["X_train", "y_train", "X_val", "y_val",
                           "X_test", "y_test"])
    if cached and not force:
        print(f"\n[LOAD] Cached features found for {model_name}, loading...")
        data = FeatureExtractor.load(OUTPUT_DIR)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val,   y_val   = data["X_val"],   data["y_val"]
        X_test,  y_test  = data["X_test"],  data["y_test"]
    else:
        print(f"\n[EXTRACT] Extracting bottleneck features for {model_name}...")
        extractor = FeatureExtractor(
            model_name=model_name,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            ckpt_path=ckpt_path,
            num_classes=NUM_CLASSES,
        )

        train_ds = ChestXrayDataset(df_train, transform=val_transforms)
        val_ds   = ChestXrayDataset(df_val,   transform=val_transforms)
        test_ds  = ChestXrayDataset(df_test,  transform=val_transforms)

        X_train, y_train = extractor.extract(train_ds, desc=f"  {model_name} train")
        X_val,   y_val   = extractor.extract(val_ds,   desc=f"  {model_name} val")
        X_test,  y_test  = extractor.extract(test_ds,  desc=f"  {model_name} test")

        FeatureExtractor.save(OUTPUT_DIR, X_train, y_train, X_val, y_val, X_test, y_test)

    feature_dim = X_train.shape[1]
    print(f"  Feature shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    print(f"  Bottleneck dimension: {feature_dim}")

    # ── Run RFECV ──
    print(f"\n[RFECV] Running RFECV on {feature_dim} features...")
    fast_rf = RandomForestClassifier(
        n_estimators=50,
        random_state=RANDOM_SEED,
        class_weight="balanced",
    )
    selector = FeatureSelector(
        estimator=fast_rf,
        scoring="f1_macro",
        cv=3,
        step=0.15,
        min_features_to_select=50,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    selector.fit(X_train, y_train)

    # ── Evaluate before/after ──
    print(f"\n[EVAL] Evaluating classifier before/after feature selection...")
    selector.evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

    # ── Save results ──
    print(f"\n[SAVE] Saving results to {OUTPUT_DIR}...")
    selector.plot_results(OUTPUT_DIR / "rfecv_curve.png")
    selector.save(OUTPUT_DIR)

    return selector.metrics_comparison_


def build_summary(all_metrics, output_path):
    """Write a cross-model comparison JSON."""
    summary_path = output_path / "comparison_all_models.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[SUMMARY] Cross-model comparison -> {summary_path}")

    # Print a simple ASCII table
    print()
    print("=" * 85)
    print("| {:<18} | {:<8} | {:<8} | {:<9} | {:<9} | {:<9} |".format(
        "Model", "Feat All", "Feat Sel", "Reduct%", "Val F1 B", "Val F1 A"))
    print("-" * 85)
    for model_name, metrics in all_metrics.items():
        n_all = metrics.get("n_features_total", "?")
        n_sel = metrics.get("n_features_selected", "?")
        reduc = metrics.get("feature_reduction_pct", "?")
        b = metrics.get("baseline_all_features", {}).get("val", {}).get("f1_macro", "?")
        a = metrics.get("selected_features", {}).get("val", {}).get("f1_macro", "?")
        if isinstance(b, float) and isinstance(a, float):
            b_s = f"{b:.4f}"
            a_s = f"{a:.4f}"
        else:
            b_s = str(b)
            a_s = str(a)
        print("| {:<18} | {:<8} | {:<8} | {:<9} | {:<9} | {:<9} |".format(
            model_name, n_all, n_sel, reduc, b_s, a_s))
    print("=" * 85)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Wrapper-based RFECV feature selection on bottleneck features")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to process (default: all available)")
    parser.add_argument("--list-models", action="store_true",
                        help="List all registered models and exit")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract features even if cached")
    args = parser.parse_args()

    if args.list_models:
        print("Registered architectures:")
        for name in list_available_models():
            cfg = ARCHITECTURES[name]
            print(f"  {name:20s}  dim={cfg['feature_dim']}")
        return

    # ── Load data splits ──
    print("=" * 70)
    print("Wrapper-Based Feature Selection via RFECV")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    print("\n[1] Loading data splits...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)
    df_test  = pd.read_csv(TEST_CSV)
    print(f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

    # ── Validation transforms (no augmentation) ──
    val_transforms = transforms.Compose([
        CLAHETransform(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ── Determine which models to process ──
    if args.model:
        # Single model mode
        model_name = args.model
        if model_name not in ARCHITECTURES:
            print(f"ERROR: Unknown model '{model_name}'. Use --list-models to see available.")
            sys.exit(1)
        ckpt_path = CKPT_ROOT / model_name / "best_model.pth"
        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        models_to_process = [(model_name, ckpt_path)]
    else:
        models_to_process = find_available_models()
        if not models_to_process:
            print("\nNo checkpoints found in outputs/<ModelName>/best_model.pth")
            print("Available models:")
            for name in list_available_models():
                print(f"  - {name}")
            sys.exit(0)

    # ── Process each model ──
    all_metrics = {}
    for model_name, ckpt_path in models_to_process:
        print(f"\n{'=' * 70}")
        print(f"Processing: {model_name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'=' * 70}")
        try:
            metrics = process_model(
                model_name, ckpt_path, df_train, df_val, df_test,
                val_transforms, force=args.force,
            )
            all_metrics[model_name] = metrics
        except Exception as e:
            print(f"  [FAIL] {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Summary ──
    if len(all_metrics) > 1:
        build_summary(all_metrics, OUTPUT_ROOT)
    elif len(all_metrics) == 1:
        # Single model: save summary too
        build_summary(all_metrics, OUTPUT_ROOT)

    print(f"\n{'=' * 70}")
    if all_metrics:
        print(f"Feature selection complete for {len(all_metrics)} model(s)!")
    else:
        print("No models processed.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
