#!/usr/bin/env python
"""
pretrain_ssl.py — SimCLR self-supervised pretraining on the full NIH release.

Why this exists: the NIH labels are NLP-mined from reports with an estimated
10-30% error rate on pneumonia, and there are only 1,431 pneumonia positives
in the entire dataset. Supervised training is therefore capped by label
quality. SimCLR never looks at a label -- it learns a representation from
112,120 images by pulling two augmented views of the same film together and
pushing different films apart -- so the noise cannot corrupt it. Labels enter
only at the fine-tuning stage.

This produces a backbone checkpoint. It is a starting point for
train_multilabel.py (--init-from), not a classifier on its own.

Honest cost note: SSL wants long schedules and large batches. On a single
8 GB laptop GPU a genuinely competitive SimCLR run is 12-24 h+. Shorter runs
give a weaker-but-nonzero benefit. Budget accordingly, and treat a 2-3 h run
as a pilot rather than the real thing.

Usage:
    python scripts/pretrain_ssl.py --smoke
    python scripts/pretrain_ssl.py --epochs 30 --batch-size 96
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import modelzoo, preprocessing as prep  # noqa: E402
from src.feature_selection.extractor import strip_head  # noqa: E402

SPLITS = PROJECT_ROOT / "data" / "splits_multilabel"
OUT_ROOT = PROJECT_ROOT / "outputs" / "ssl"


def simclr_views(img_size):
    """Two independently augmented views of the same film.

    Deliberately stronger than the supervised stack: SimCLR needs the two
    views to be hard to match, or the task is trivial and the representation
    collapses. Vertical flip stays out - an upside-down chest film is not a
    valid view of the same anatomy.
    """
    resize = prep.train_resize_for(img_size)
    return T.Compose([
        prep.CLAHETransform(),
        T.Resize(resize),
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
        T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=prep.IMAGENET_MEAN, std=prep.IMAGENET_STD),
    ])


class TwoViewCXR(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.t = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.t(img), self.t(img)


class ProjectionHead(nn.Module):
    """2-layer MLP; SimCLR contrasts in this space, not in feature space."""

    def __init__(self, in_dim, hidden=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def nt_xent(z1, z2, temperature=0.2):
    """Normalised temperature-scaled cross entropy (SimCLR loss).

    For each of the 2N views the positive is the other view of the same image;
    every other view in the batch is a negative. Bigger batches therefore give
    a harder and more informative task.
    """
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DenseNet121")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--img-size", type=int, default=224,
                    help="SSL runs at lower res for throughput; fine-tuning restores it")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    amp = not args.no_amp and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dest = OUT_ROOT / args.model
    dest.mkdir(parents=True, exist_ok=True)

    paths = pd.read_csv(SPLITS / "train.csv")["image_path"].tolist()
    if args.smoke:
        paths = paths[:512]
    print(f"SimCLR  model={args.model}  images={len(paths):,}  "
          f"img_size={args.img_size}  batch={args.batch_size}  amp={amp}")
    print("no labels are read at any point in this script")

    dl = DataLoader(TwoViewCXR(paths, simclr_views(args.img_size)),
                    batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)

    backbone = strip_head(modelzoo.build(args.model, num_classes=1), args.model).to(device)
    feat_dim = modelzoo.ARCHITECTURES[args.model]["feature_dim"]
    head = ProjectionHead(feat_dim).to(device)

    params = list(backbone.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt = dest / "ssl_backbone.pth"
    start, history = 1, []
    hist_path = dest / "ssl_history.csv"
    if args.resume and ckpt.exists():
        c = torch.load(ckpt, map_location=device, weights_only=False)
        backbone.load_state_dict(c["backbone_state_dict"])
        head.load_state_dict(c["head_state_dict"])
        start = int(c.get("epoch", 0)) + 1
        if hist_path.exists():
            history = pd.read_csv(hist_path).to_dict("records")
        print(f"resumed at epoch {start}")

    t_start = time.time()
    for epoch in range(start, args.epochs + 1):
        backbone.train(); head.train()
        t0, run, seen = time.time(), 0.0, 0
        for bi, (v1, v2) in enumerate(dl):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                loss = nt_xent(head(backbone(v1).flatten(1)),
                               head(backbone(v2).flatten(1)), args.temperature)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item() * v1.size(0); seen += v1.size(0)
            if bi % 100 == 0:
                print(f"    e{epoch} b{bi}/{len(dl)} loss={run/max(seen,1):.4f} "
                      f"({seen/max(time.time()-t0,1e-9):.0f} img/s)", flush=True)
            if args.smoke and bi >= 3:
                break
        sched.step()
        row = {"epoch": epoch, "nt_xent_loss": run / max(seen, 1),
               "lr": opt.param_groups[0]["lr"],
               "minutes": round((time.time() - t_start) / 60, 1)}
        history.append(row)
        pd.DataFrame(history).to_csv(hist_path, index=False)
        print(f"  epoch {epoch}: NT-Xent={row['nt_xent_loss']:.4f} "
              f"[{row['minutes']:.1f} min]", flush=True)
        torch.save({"epoch": epoch,
                    "backbone_state_dict": backbone.state_dict(),
                    "head_state_dict": head.state_dict(),
                    "model": args.model, "img_size": args.img_size,
                    "nt_xent_loss": row["nt_xent_loss"]}, ckpt)

    (dest / "ssl_summary.json").write_text(json.dumps({
        "model": args.model, "epochs_completed": len(history),
        "images": len(paths), "img_size": args.img_size,
        "batch_size": args.batch_size, "temperature": args.temperature,
        "final_loss": history[-1]["nt_xent_loss"] if history else None,
        "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
        "next_step": "python scripts/train_multilabel.py --init-from "
                     f"{ckpt.relative_to(PROJECT_ROOT)}",
    }, indent=2), encoding="utf-8")
    print(f"\nbackbone -> {ckpt}")
    print("fine-tune with: python scripts/train_multilabel.py --init-from "
          f"{ckpt.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
