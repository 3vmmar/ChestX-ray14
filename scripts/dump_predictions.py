#!/usr/bin/env python
"""
dump_predictions.py — save PER-IMAGE probabilities for a trained model.

The aggregate tta_metrics.json files record only summary numbers, which is
not enough for a paired significance test. This script re-runs the exact
5-variant TTA evaluation used in the notebooks and writes one row per image,
so that scripts/compare_models.py can run a DeLong test on the raw scores.

Preprocessing is replicated from the training notebooks:
    CLAHE(clip=2.0, tile=8x8) -> Resize(S, S) -> ToTensor -> ImageNet norm
with S taken per-model from src/modelzoo.py (288 CNN, 299 InceptionV3, 224 ViT).

TTA variants (same order as the notebooks):
    0 plain          1 horizontal flip
    2 rotate +7 deg  3 rotate -7 deg     4 brightness jitter 0.15

Usage:
    python scripts/dump_predictions.py --model DenseNet121
    python scripts/dump_predictions.py --all
    python scripts/dump_predictions.py --list
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import modelzoo  # noqa: E402

DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
OUT_ROOT = PROJECT_ROOT / "outputs" / "models"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42


class CLAHETransform:
    """CLAHE with pre-downscale guard (verbatim from the training notebooks)."""

    def __call__(self, pil_img):
        if max(pil_img.size) > 512:
            pil_img = pil_img.resize((512, 512), Image.LANCZOS)
        img_np = np.array(pil_img.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_np)
        return Image.fromarray(enhanced).convert("RGB")


def build_tta_transforms(size):
    norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    base = [CLAHETransform(), T.Resize((size, size))]
    return [
        T.Compose(base + [T.ToTensor(), norm]),
        T.Compose(base + [T.RandomHorizontalFlip(p=1.0), T.ToTensor(), norm]),
        T.Compose(base + [T.RandomRotation(degrees=(7, 7)), T.ToTensor(), norm]),
        T.Compose(base + [T.RandomRotation(degrees=(-7, -7)), T.ToTensor(), norm]),
        T.Compose(base + [T.ColorJitter(brightness=0.15), T.ToTensor(), norm]),
    ]


VARIANT_NAMES = ["plain", "hflip", "rot+7", "rot-7", "bright"]


class ChestXrayDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["image_path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def checkpoint_out_units(path):
    """Number of output units the checkpoint was trained with."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    for k in reversed(list(sd.keys())):
        if k.endswith("weight") and getattr(sd[k], "ndim", 0) == 2:
            return int(sd[k].shape[0])
    return 1


@torch.no_grad()
def infer(model, df, transform, device, batch_size=16, workers=0, class_index=0):
    loader = DataLoader(
        ChestXrayDataset(df, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )
    out = []
    for x, _ in loader:
        logits = model(x.to(device, non_blocking=True))
        if isinstance(logits, (tuple, list)):  # InceptionV3 aux
            logits = logits[0]
        if logits.shape[-1] == 1:
            p = torch.sigmoid(logits.squeeze(-1))          # binary head
        else:
            p = torch.softmax(logits, dim=-1)[:, class_index]  # multi-class head
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def run_model(name, df, device, batch_size, workers, class_index=0, suffix=""):
    ckpt = OUT_ROOT / name / "best_model.pth"
    if not ckpt.exists():
        return {"model": name, "status": "NO_CHECKPOINT", "path": str(ckpt)}

    size = modelzoo.img_size(name)
    n_out = checkpoint_out_units(ckpt)
    if n_out != 1:
        print(f"    NOTE: this checkpoint has {n_out} output units, not 1 - it is NOT a")
        print(f"          binary pneumonia model. Scoring class index {class_index} via softmax.")
    model = modelzoo.build(name, num_classes=n_out)
    meta = modelzoo.load_checkpoint(model, ckpt, device="cpu")
    if meta["missing_keys"]:
        print(f"    warning: {len(meta['missing_keys'])} missing keys on load")
    model = model.to(device).eval()

    print(f"    checkpoint epoch={meta['epoch']} val_auc={meta['val_auc']} size={size}")

    cols = {}
    for i, tf in enumerate(build_tta_transforms(size)):
        torch.manual_seed(SEED + i)  # ColorJitter is stochastic; pin it
        np.random.seed(SEED + i)
        cols[f"p_{VARIANT_NAMES[i]}"] = infer(model, df, tf, device, batch_size, workers, class_index)
        print(f"    variant {i} ({VARIANT_NAMES[i]}) done")

    out = pd.DataFrame({"image_path": df["image_path"].values,
                        "label": df["label"].astype(int).values})
    for k, v in cols.items():
        out[k] = v
    out["p_tta"] = out[[f"p_{n}" for n in VARIANT_NAMES]].mean(axis=1)
    out["p_single"] = out["p_plain"]

    dest = OUT_ROOT / name / f"val_probs{suffix}.csv"
    out.to_csv(dest, index=False)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"model": name, "status": "OK", "rows": len(out), "out": str(dest)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", help="model name (repeatable)")
    ap.add_argument("--all", action="store_true", help="every model with a checkpoint")
    ap.add_argument("--list", action="store_true", help="list models and checkpoint status")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--class-index", type=int, default=0,
                    help="which output unit is the positive class (multi-class heads)")
    ap.add_argument("--suffix", default="", help="suffix for the output filename")
    args = ap.parse_args()

    if args.list:
        print(f"{'model':<16} {'size':<6} {'family':<12} checkpoint")
        for n in modelzoo.list_models():
            ck = OUT_ROOT / n / "best_model.pth"
            info = modelzoo.ARCHITECTURES[n]
            mark = "present" if ck.exists() else "MISSING"
            print(f"  {n:<16} {info['img_size']:<6} {info['family']:<12} {mark}")
        return 0

    targets = args.model or (modelzoo.list_models() if args.all else None)
    if not targets:
        ap.error("pass --model NAME, --all, or --list")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA_SPLITS / f"{args.split}.csv")
    print(f"device={device}  split={args.split}  rows={len(df)}  positives={int(df['label'].sum())}")

    results = []
    for name in targets:
        print(f"\n[{name}]")
        r = run_model(name, df, device, args.batch_size, args.workers,
                      args.class_index, args.suffix)
        if r["status"] != "OK":
            print(f"    SKIPPED - no checkpoint at {r['path']}")
        else:
            print(f"    wrote {r['rows']} rows -> {r['out']}")
        results.append(r)

    ok = [r for r in results if r["status"] == "OK"]
    print(f"\ndone: {len(ok)}/{len(results)} models written")
    if len(ok) < len(results):
        miss = [r["model"] for r in results if r["status"] != "OK"]
        print("no checkpoint for: " + ", ".join(miss))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
