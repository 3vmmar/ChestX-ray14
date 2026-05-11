#!/usr/bin/env python
"""
run_xai_demo.py  -  Team505 XAI Pipeline Runner

Loads 12 trained model checkpoints, selects a shared evaluation subset
(TP / TN / FP / FN) from the validation dataset, and generates 
Grad-CAM (or Attention Rollout), LIME, SHAP, and Integrated
Gradients explanations for each model.

Outputs saved to:   outputs/<member>/[<model>]/xai/<method>/
"""

import sys, os, time, gc, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend - no Tk required

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
    GradCAM, AttentionRollout, get_target_layer, save_gradcam,
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

# ==============================================================================
# MODEL BUILDERS
# ==============================================================================
def _build_densenet121():
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(model.classifier.in_features, 1))
    return model

def _build_efficientnet_b3():
    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(model.classifier[1].in_features, 1))
    return model

def _build_resnet50():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(model.fc.in_features, 1))
    return model

def _build_densenet201():
    model = models.densenet201(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(model.classifier.in_features, 1))
    return model

def _build_vgg16():
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(model.classifier[6].in_features, 1))
    return model

def _build_mobilenet_v2():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(model.classifier[1].in_features, 1))
    return model

def _build_xception():
    model = timm.create_model("legacy_xception", pretrained=False)
    model.fc = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(model.fc.in_features, 1))
    return model

def _build_inception_v3():
    model = models.inception_v3(weights=None, aux_logits=True)
    model.fc = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(model.fc.in_features, 1))
    return model

def _build_resnet101():
    model = models.resnet101(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(model.fc.in_features, 1))
    return model

def _build_vit():
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(model.heads.head.in_features, 1))
    return model

def _build_swint():
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=1, drop_rate=0.6)
    return model

def _build_deits():
    model = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=1, drop_rate=0.6)
    return model

# Model configs: (model_name, build_fn, m_type, ckpt_path, xai_dir)
MODELS = [
    ("DenseNet121", _build_densenet121, "cnn",
     PROJECT_ROOT / "outputs" / "Ammar_Ahmed" / "DenseNet121" / "best_model.pth",
     PROJECT_ROOT / "outputs" / "Ammar_Ahmed" / "DenseNet121" / "xai"),
    ("EfficientNetB3", _build_efficientnet_b3, "cnn",
     PROJECT_ROOT / "outputs" / "Ammar_Ahmed" / "EfficientNetB3" / "best_model.pth",
     PROJECT_ROOT / "outputs" / "Ammar_Ahmed" / "EfficientNetB3" / "xai"),
    ("ResNet50", _build_resnet50, "cnn",
     PROJECT_ROOT / "outputs" / "Ammar_Ahmed" / "ResNet50" / "best_model.pth",
     PROJECT_ROOT / "outputs" / "Ammar_Ahmed" / "ResNet50" / "xai"),
    ("DenseNet201", _build_densenet201, "cnn",
     PROJECT_ROOT / "outputs" / "Hosam_Nabil" / "DenseNet201" / "best_model.pth",
     PROJECT_ROOT / "outputs" / "Hosam_Nabil" / "DenseNet201" / "xai"),
    ("VGG16", _build_vgg16, "cnn",
     PROJECT_ROOT / "outputs" / "Hosam_Nabil" / "VGG16" / "best_model.pth",
     PROJECT_ROOT / "outputs" / "Hosam_Nabil" / "VGG16" / "xai"),
    ("MobileNetV2", _build_mobilenet_v2, "cnn",
     PROJECT_ROOT / "outputs" / "Hosam_Nabil" / "MobileNetV2" / "best_model.pth",
     PROJECT_ROOT / "outputs" / "Hosam_Nabil" / "MobileNetV2" / "xai"),
    ("Xception", _build_xception, "cnn",
     PROJECT_ROOT / "outputs" / "Mohamed_Eslam" / "Xception" / "best_model.pth",
     PROJECT_ROOT / "outputs" / "Mohamed_Eslam" / "Xception" / "xai"),
    ("InceptionV3", _build_inception_v3, "cnn",
     PROJECT_ROOT / "outputs" / "Mohamed_Eslam" / "InceptionV3" / "best_model.pth",
     PROJECT_ROOT / "outputs" / "Mohamed_Eslam" / "InceptionV3" / "xai"),
    ("ResNet101", _build_resnet101, "cnn",
     PROJECT_ROOT / "outputs" / "Mohamed_Eslam" / "ResNet101" / "best_model.pth",
     PROJECT_ROOT / "outputs" / "Mohamed_Eslam" / "ResNet101" / "xai"),
    ("ViTB16", _build_vit, "transformer",
     PROJECT_ROOT / "outputs" / "Abdelrahman_Mostafa" / "ViTB16" / "best_model.pth",
     PROJECT_ROOT / "outputs" / "Abdelrahman_Mostafa" / "ViTB16" / "xai"),
    ("SwinT", _build_swint, "transformer",
     PROJECT_ROOT / "outputs" / "Abdelrahman_Mostafa" / "SwinT" / "best_model.pth",
     PROJECT_ROOT / "outputs" / "Abdelrahman_Mostafa" / "SwinT" / "xai"),
    ("DeiTS", _build_deits, "transformer",
     PROJECT_ROOT / "outputs" / "Abdelrahman_Mostafa" / "DeiTS" / "best_model.pth",
     PROJECT_ROOT / "outputs" / "Abdelrahman_Mostafa" / "DeiTS" / "xai"),
]

# Number of images per category
N_PER_CATEGORY = 2

# ==============================================================================
# HELPERS
# ==============================================================================
def load_image(path):
    img = Image.open(path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized).astype(np.float64) / 255.0
    tensor = val_transform(img_resized).unsqueeze(0)
    return img_resized, img_np, tensor

def predict_with_model(model, tensor, device, threshold):
    model.eval()
    with torch.no_grad():
        logit = model(tensor.to(device)).squeeze()
        prob = torch.sigmoid(logit).item()
        pred = int(prob >= threshold)
    return prob, pred

def select_shared_subset(df_val, model, device, m_type, n=N_PER_CATEGORY):
    threshold = 0.40 if m_type == "cnn" else 0.48
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

        prob, pred = predict_with_model(model, tensor, device, threshold)

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

        if sum(len(v) for v in categories.values()) >= needed:
            break

    return categories


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70)
    print("Team505 XAI Batch Pipeline")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    val_csv = DATA_SPLITS / "val.csv"
    if not val_csv.exists():
        print(f"ERROR: Cannot find {val_csv}")
        return
    
    df_val = pd.read_csv(val_csv)
    if 'target_pneumonia' not in df_val.columns and 'label' in df_val.columns:
        df_val['target_pneumonia'] = df_val['label']
    print(f"Validation set: {len(df_val)} images")

    for model_name, build_fn, m_type, ckpt_path, xai_dir in MODELS:
        print(f"\n{'='*70}")
        print(f"Processing: {model_name} ({m_type.upper()})")
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

        # Select evaluation subset based on threshold rules
        print(f"  Selecting subset (Threshold = {0.40 if m_type == 'cnn' else 0.48})...")
        subset = select_shared_subset(df_val, model, DEVICE, m_type)
        
        # Create output dirs safely
        for method in ["gradcam", "lime", "shap", "integrated_gradients"]:
            (xai_dir / method).mkdir(parents=True, exist_ok=True)

        prefix = model_name.lower().replace("-", "").replace("/", "")

        # ------- Grad-CAM / Attention Rollout -------
        cam_method = "Attention Rollout" if m_type == "transformer" else "Grad-CAM"
        print(f"  Running {cam_method}...")
        try:
            target_layer = get_target_layer(model, model_name)
            if m_type == "transformer":
                cam_obj = AttentionRollout(model, target_layer)
            else:
                cam_obj = GradCAM(model, target_layer)
                
            for cat, items in subset.items():
                for i, (img_path, label, prob) in enumerate(items, start=1):
                    _, img_np, tensor = load_image(img_path)
                    tensor = tensor.to(DEVICE)
                    tensor.requires_grad_(True)
                    heatmap = cam_obj.generate(tensor)
                    title = f"{cam_method} | {model_name} | {cat} (p={prob:.2f})"
                    fname = f"{prefix}_{cat}_{i:02d}.png"
                    save_gradcam(
                        (img_np * 255).astype(np.uint8), heatmap,
                        xai_dir / "gradcam" / fname, title=title,
                    )
            cam_obj.release()
            print(f"    [OK] {cam_method} done")
        except Exception as e:
            print(f"    [FAIL] {cam_method} failed: {e}")

        # ------- LIME -------
        print("  Running LIME...")
        try:
            lime_exp = LIMEExplainer(model, DEVICE)
            for cat, items in subset.items():
                for i, (img_path, label, prob) in enumerate(items, start=1):
                    _, img_np, tensor = load_image(img_path)
                    explanation = lime_exp.explain(img_np, num_samples=200)
                    title = f"LIME | {model_name} | {cat} (p={prob:.2f})"
                    fname = f"{prefix}_{cat}_{i:02d}.png"
                    save_lime(
                        img_np, explanation,
                        xai_dir / "lime" / fname, title=title,
                    )
            print("    [OK] LIME done")
        except Exception as e:
            print(f"    [FAIL] LIME failed: {e}")

        # ------- SHAP -------
        print("  Running SHAP...")
        try:
            shap_exp = SHAPExplainer(model, DEVICE, max_evals=300)
            for cat, items in subset.items():
                for i, (img_path, label, prob) in enumerate(items, start=1):
                    _, img_np, tensor = load_image(img_path)
                    shap_vals = shap_exp.explain(img_np)
                    title = f"SHAP | {model_name} | {cat} (p={prob:.2f})"
                    fname = f"{prefix}_{cat}_{i:02d}.png"
                    save_shap(
                        img_np, shap_vals,
                        xai_dir / "shap" / fname, title=title,
                    )
            print("    [OK] SHAP done")
        except Exception as e:
            print(f"    [FAIL] SHAP failed: {e}")

        # ------- Integrated Gradients -------
        print("  Running Integrated Gradients...")
        try:
            ig_exp = IGExplainer(model, DEVICE, n_steps=50)
            for cat, items in subset.items():
                for i, (img_path, label, prob) in enumerate(items, start=1):
                    _, img_np, tensor = load_image(img_path)
                    attr = ig_exp.explain(tensor)
                    title = f"IG | {model_name} | {cat} (p={prob:.2f})"
                    fname = f"{prefix}_{cat}_{i:02d}.png"
                    save_ig(
                        img_np, attr,
                        xai_dir / "integrated_gradients" / fname, title=title,
                    )
            print("    [OK] Integrated Gradients done")
        except Exception as e:
            print(f"    [FAIL] Integrated Gradients failed: {e}")

        # Cleanup GPU memory
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"XAI DONE: {model_name} | saved to {xai_dir}")

    print("\n" + "=" * 70)
    print("XAI Pipeline Complete for all models!")
    print("=" * 70)

if __name__ == "__main__":
    main()
