#!/usr/bin/env python
"""
run_xai_demo.py  —  Team505 XAI Pipeline Runner

Loads each trained model checkpoint, selects a shared evaluation subset
(TP / TN / FP / FN), and generates Grad-CAM, LIME, SHAP, and Integrated
Gradients explanations for each model.

Usage:
    cd Team505_phase2
    .venv\\Scripts\\python.exe scripts/run_xai_demo.py

Outputs saved to:   outputs/<member>/xai/<method>/
"""

import sys, os, time, gc, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no Tk required

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import timm
from PIL import Image
from pathlib import Path

# ---- project paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.xai import (
    GradCAM, get_target_layer, save_gradcam,
    LIMEExplainer, save_lime,
    SHAPExplainer, save_shap,
    IGExplainer, save_ig,
)

# ==============================================================================
# CONFIG
# ==============================================================================
RANDOM_SEED = 42
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Model configs:  (member_name, model_name, build_fn, checkpoint_path)
MODELS = [
    ("Ammar_Ahmed", "DenseNet121",
     lambda: _build_densenet121(),
     PROJECT_ROOT / "outputs" / "Ammar_Ahmed" / "best_model.pth"),
    ("Hosam_Nabil", "DenseNet201",
     lambda: _build_densenet201(),
     PROJECT_ROOT / "outputs" / "Hosam_Nabil" / "best_model.pth"),
    ("Mohamed_Eslam", "Xception",
     lambda: _build_xception(),
     PROJECT_ROOT / "outputs" / "Mohamed_Eslam" / "best_model.pth"),
    ("Abdelrahman_Mostafa", "ViT-B/16",
     lambda: _build_vit(),
     PROJECT_ROOT / "outputs" / "Abdelrahman_Mostafa" / "best_model.pth"),
]

# Number of images per category for the shared evaluation subset
N_PER_CATEGORY = 2  # 2 TP, 2 TN, 2 FP, 2 FN = 8 images total


# ==============================================================================
# MODEL BUILDERS
# ==============================================================================
def _build_densenet121():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model

def _build_densenet201():
    model = models.densenet201(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model

def _build_xception():
    model = timm.create_model("legacy_xception", pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def _build_vit():
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 1)
    return model


# ==============================================================================
# HELPERS
# ==============================================================================
def load_image(path):
    """Load image, return (PIL.Image, np.ndarray [0,1] RGB, tensor)."""
    img = Image.open(path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized).astype(np.float64) / 255.0
    tensor = val_transform(img_resized).unsqueeze(0)
    return img_resized, img_np, tensor


def denormalize(tensor):
    """Denormalize a CHW tensor back to numpy HWC [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = tensor.cpu().squeeze(0) * std + mean
    t = t.clamp(0, 1).numpy().transpose(1, 2, 0)
    return t


def predict_with_model(model, tensor, device):
    """Run prediction, return (prob, pred)."""
    model.eval()
    with torch.no_grad():
        logit = model(tensor.to(device)).squeeze()
        prob = torch.sigmoid(logit).item()
        pred = int(prob >= 0.5)
    return prob, pred


def select_shared_subset(df_val, model, device, n=N_PER_CATEGORY):
    """
    Run model on validation data, find TP/TN/FP/FN cases.
    Returns dict: {category: [(image_path, label, prob), ...]}
    """
    categories = {"TP": [], "TN": [], "FP": [], "FN": []}
    needed = n * 4

    for _, row in df_val.iterrows():
        img_path = row["image_path"]
        label = int(row["target_pneumonia"])

        if not Path(img_path).exists():
            continue

        try:
            _, _, tensor = load_image(img_path)
        except Exception:
            continue

        prob, pred = predict_with_model(model, tensor, device)

        if label == 1 and pred == 1:
            cat = "TP"
        elif label == 0 and pred == 0:
            cat = "TN"
        elif label == 0 and pred == 1:
            cat = "FP"
        elif label == 1 and pred == 0:
            cat = "FN"

        if len(categories[cat]) < n:
            categories[cat].append((img_path, label, prob))

        collected = sum(len(v) for v in categories.values())
        if collected >= needed:
            break

    return categories





# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70)
    print("Team505 XAI Pipeline")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load val data (using train_dev with same 80/20 patient split as notebooks)
    np.random.seed(RANDOM_SEED)
    df_full_dev = pd.read_csv(DATA_SPLITS / "train_dev.csv")
    all_pids = df_full_dev["patient_id"].unique()
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(all_pids)
    split_idx = int(len(all_pids) * 0.8)
    val_pids = set(all_pids[split_idx:])
    df_val = df_full_dev[df_full_dev["patient_id"].isin(val_pids)].copy()
    print(f"Validation set: {len(df_val)} images")

    # Process each model
    for member, model_name, build_fn, ckpt_path in MODELS:
        print(f"\n{'='*70}")
        print(f"Processing: {member} -- {model_name}")
        print(f"{'='*70}")

        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        # Build and load model
        model = build_fn()
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE)
        model.eval()
        print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

        # Select shared evaluation subset
        print("  Selecting evaluation subset (TP/TN/FP/FN)...")
        subset = select_shared_subset(df_val, model, DEVICE)
        for cat, items in subset.items():
            print(f"    {cat}: {len(items)} images")

        # Create output dirs
        xai_base = PROJECT_ROOT / "outputs" / member / "xai"
        for method in ["gradcam", "lime", "shap", "integrated_gradients"]:
            (xai_base / method).mkdir(parents=True, exist_ok=True)

        # ------- Grad-CAM -------
        print("  Running Grad-CAM...")
        try:
            target_layer = get_target_layer(model, model_name)
            gc_obj = GradCAM(model, target_layer)
            for cat, items in subset.items():
                for i, (img_path, label, prob) in enumerate(items):
                    _, img_np, tensor = load_image(img_path)
                    tensor = tensor.to(DEVICE)
                    tensor.requires_grad_(True)
                    heatmap = gc_obj.generate(tensor)
                    title = f"Grad-CAM | {model_name} | {cat} (label={label}, prob={prob:.3f})"
                    fname = f"{cat}_{i}.png"
                    save_gradcam(
                        (img_np * 255).astype(np.uint8), heatmap,
                        xai_base / "gradcam" / fname, title=title,
                    )
            gc_obj.release()
            print("    [OK] Grad-CAM done")
        except Exception as e:
            print(f"    [FAIL] Grad-CAM failed: {e}")

        # ------- LIME -------
        print("    Running LIME...")
        try:
            lime_exp = LIMEExplainer(model, DEVICE)
            for cat, items in subset.items():
                for i, (img_path, label, prob) in enumerate(items):
                    _, img_np, tensor = load_image(img_path)
                    explanation = lime_exp.explain(img_np, num_samples=200)
                    title = f"LIME | {model_name} | {cat} (label={label}, prob={prob:.3f})"
                    fname = f"{cat}_{i}.png"
                    save_lime(
                        img_np, explanation,
                        xai_base / "lime" / fname, title=title,
                    )
            print("    [OK] LIME done")
        except Exception as e:
            print(f"    [FAIL] LIME failed: {e}")

        # ------- SHAP (PartitionExplainer) -------
        print("  Running SHAP...")
        try:
            shap_exp = SHAPExplainer(model, DEVICE, max_evals=300)
            for cat, items in subset.items():
                for i, (img_path, label, prob) in enumerate(items):
                    _, img_np, tensor = load_image(img_path)
                    shap_vals = shap_exp.explain(img_np)
                    title = f"SHAP | {model_name} | {cat} (label={label}, prob={prob:.3f})"
                    fname = f"{cat}_{i}.png"
                    save_shap(
                        img_np, shap_vals,
                        xai_base / "shap" / fname, title=title,
                    )
            print("    [OK] SHAP done")
        except Exception as e:
            print(f"    [FAIL] SHAP failed: {e}")

        # ------- Integrated Gradients -------
        print("  Running Integrated Gradients...")
        try:
            ig_exp = IGExplainer(model, DEVICE, n_steps=50)
            for cat, items in subset.items():
                for i, (img_path, label, prob) in enumerate(items):
                    _, img_np, tensor = load_image(img_path)
                    attr = ig_exp.explain(tensor)
                    title = f"IG | {model_name} | {cat} (label={label}, prob={prob:.3f})"
                    fname = f"{cat}_{i}.png"
                    save_ig(
                        img_np, attr,
                        xai_base / "integrated_gradients" / fname, title=title,
                    )
            print("    [OK] Integrated Gradients done")
        except Exception as e:
            print(f"    [FAIL] Integrated Gradients failed: {e}")

        # Cleanup GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  Model {model_name} complete and unloaded.")

    print("\n" + "=" * 70)
    print("XAI Pipeline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
