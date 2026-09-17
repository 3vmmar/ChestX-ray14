#!/usr/bin/env python
"""
train_multilabel.py — CheXNet-style multi-label training on the full NIH set.

Trains one backbone to predict all 14 findings at once over 78,705 images,
then reads the pneumonia head. The auxiliary labels are the point: they give
the backbone a representation that 1,004 pneumonia positives alone cannot
teach it. CheXNet reports pneumonia AUC 0.7680 this way, against roughly 0.66
for the binary-only setup here.

Evaluation discipline matches the binary notebooks: the checkpoint is chosen
on val, and test.csv is scored once at the end with the threshold frozen from
val. Nothing about test influences selection.

Notes
  * Prevalence here is the true NIH rate (1.27%), not the 7% of the capped
    binary splits, so F1 is NOT comparable across the two setups. AUC is.
  * AMP is on by default; it roughly halves epoch time on this GPU.
  * A checkpoint and a history row are written every epoch, so an interrupted
    run is still usable and --resume picks up where it stopped.

Usage:
    python scripts/train_multilabel.py --epochs 20
    python scripts/train_multilabel.py --resume
    python scripts/train_multilabel.py --smoke        # 200 batches, sanity only
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
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import modelzoo, preprocessing as prep  # noqa: E402

SPLITS = PROJECT_ROOT / "data" / "splits_multilabel"
OUT_ROOT = PROJECT_ROOT / "outputs" / "multilabel"

FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
PNEU = FINDINGS.index("Pneumonia")


class MultiLabelCXR(Dataset):
    def __init__(self, df, transform):
        self.paths = df["image_path"].tolist()
        self.y = df[FINDINGS].values.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), torch.from_numpy(self.y[i])


def loaders(args, img_size):
    tr = pd.read_csv(SPLITS / "train.csv")
    va = pd.read_csv(SPLITS / "val.csv")
    te = pd.read_csv(SPLITS / "test.csv")
    if args.smoke:
        tr, va, te = tr.head(2000), va.head(600), te.head(600)
    mk = lambda df, tf, sh: DataLoader(
        MultiLabelCXR(df, tf), batch_size=args.batch_size, shuffle=sh,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0, drop_last=sh)
    return (mk(tr, prep.build_train_transforms(img_size), True),
            mk(va, prep.build_val_transforms(img_size), False),
            mk(te, prep.build_val_transforms(img_size), False), tr, va, te)


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=amp):
            out = model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
        P.append(torch.sigmoid(out.float()).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def per_class_auc(probs, y):
    aucs = {}
    for i, f in enumerate(FINDINGS):
        if y[:, i].sum() > 0 and y[:, i].sum() < len(y):
            aucs[f] = float(roc_auc_score(y[:, i], probs[:, i]))
    return aucs


def tune_threshold(p, y):
    best = (0.0, 0.5)
    for t in np.arange(0.01, 0.99, 0.01):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best[0]:
            best = (f, float(t))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DenseNet121")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--unfreeze-epoch", type=int, default=2,
                    help="epoch at which the whole backbone unfreezes")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--init-from", default=None,
                    help="SimCLR backbone from scripts/pretrain_ssl.py")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    amp = not args.no_amp and torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dest = OUT_ROOT / args.model
    dest.mkdir(parents=True, exist_ok=True)
    img_size = modelzoo.img_size(args.model)

    train_dl, val_dl, test_dl, tr, va, te = loaders(args, img_size)
    print(f"model={args.model} size={img_size} device={device} amp={amp}")
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} images")
    print(f"pneumonia positives: train {int(tr.Pneumonia.sum())}, "
          f"val {int(va.Pneumonia.sum())}, test {int(te.Pneumonia.sum())}")

    model = modelzoo.build(args.model, num_classes=len(FINDINGS))
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = ck["backbone_state_dict"]
        own = model.state_dict()
        # the SSL backbone has its head stripped, so head tensors are absent
        usable = {k: v for k, v in sd.items()
                  if k in own and own[k].shape == v.shape}
        model.load_state_dict(usable, strict=False)
        print(f"initialised {len(usable)}/{len(own)} tensors from SimCLR backbone "
              f"{args.init_from} (epoch {ck.get('epoch')}, "
              f"NT-Xent {ck.get('nt_xent_loss'):.4f})")
    model = model.to(device)

    # rare findings get upweighted so they are not ignored by the loss
    pos = tr[FINDINGS].sum().values.astype(np.float64)
    pw = np.clip((len(tr) - pos) / np.maximum(pos, 1), 1.0, 50.0)
    print("pos_weight: " + ", ".join(f"{f}={w:.1f}" for f, w in zip(FINDINGS, pw)))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pw, dtype=torch.float32, device=device))

    def set_trainable(full):
        for p in model.parameters():
            p.requires_grad = full
        head = getattr(model, "classifier", None) or getattr(model, "fc", None)
        for p in head.parameters():
            p.requires_grad = True

    set_trainable(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    hist_path = dest / "history.csv"
    ckpt_path = dest / "best_model.pth"
    start_epoch, best_auc, no_improve = 1, -1.0, 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_auc = float(ck.get("pneumonia_auc", -1.0))
        print(f"resumed from epoch {start_epoch-1} (pneumonia AUC {best_auc:.4f})")

    history = []
    if hist_path.exists() and args.resume:
        history = pd.read_csv(hist_path).to_dict("records")

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.unfreeze_epoch:
            set_trainable(True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.3,
                                    weight_decay=args.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(args.epochs - epoch + 1, 1), eta_min=1e-6)
            print(f"  [epoch {epoch}] backbone unfrozen")

        model.train()
        t0, run, seen = time.time(), 0.0, 0
        for bi, (x, y) in enumerate(train_dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                out = model(x)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(x)
            seen += len(x)
            if bi % 200 == 0:
                print(f"    e{epoch} b{bi}/{len(train_dl)} loss={run/max(seen,1):.4f} "
                      f"({seen/max(time.time()-t0,1e-9):.0f} img/s)", flush=True)
            if args.smoke and bi >= 30:
                break
        train_loss = run / max(seen, 1)
        sched.step()

        probs, y = evaluate(model, val_dl, device, amp)
        aucs = per_class_auc(probs, y)
        pa = aucs.get("Pneumonia", float("nan"))
        mean_auc = float(np.mean(list(aucs.values()))) if aucs else float("nan")
        f1p, thr = tune_threshold(probs[:, PNEU], y[:, PNEU])
        pr = float(average_precision_score(y[:, PNEU], probs[:, PNEU]))

        row = {"epoch": epoch, "train_loss": train_loss, "pneumonia_auc": pa,
               "pneumonia_pr_auc": pr, "pneumonia_f1": f1p, "threshold": thr,
               "mean_auc_14": mean_auc, "lr": opt.param_groups[0]["lr"],
               "minutes": round((time.time() - t_start) / 60, 1)}
        history.append(row)
        pd.DataFrame(history).to_csv(hist_path, index=False)
        print(f"  epoch {epoch}: loss={train_loss:.4f}  PNEUMONIA auc={pa:.4f} "
              f"pr={pr:.4f} f1={f1p:.4f}@{thr:.2f}  mean14={mean_auc:.4f}  "
              f"[{row['minutes']:.1f} min]", flush=True)

        if pa > best_auc:
            best_auc, no_improve = pa, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "pneumonia_auc": pa, "threshold": thr,
                        "findings": FINDINGS}, ckpt_path)
            (dest / "val_metrics.json").write_text(json.dumps(
                {"epoch": epoch, "pneumonia_auc": pa, "pneumonia_pr_auc": pr,
                 "pneumonia_f1": f1p, "threshold": thr, "mean_auc_14": mean_auc,
                 "per_class_auc": aucs}, indent=2), encoding="utf-8")
            print(f"    [BEST] pneumonia AUC {pa:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early stop at epoch {epoch}")
                break

    # ---- locked test evaluation ------------------------------------------
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        thr = float(ck["threshold"])
        probs, y = evaluate(model, test_dl, device, amp)
        aucs = per_class_auc(probs, y)
        pred = (probs[:, PNEU] >= thr).astype(int)
        res = {
            "model": args.model, "evaluation": "LOCKED_TEST_multilabel",
            "threshold_source": "validation (frozen)", "threshold": round(thr, 3),
            "best_epoch": int(ck["epoch"]),
            "test_pneumonia_auc": round(float(roc_auc_score(y[:, PNEU], probs[:, PNEU])), 4),
            "test_pneumonia_pr_auc": round(float(average_precision_score(y[:, PNEU], probs[:, PNEU])), 4),
            "test_pneumonia_f1": round(float(f1_score(y[:, PNEU], pred, zero_division=0)), 4),
            "test_mean_auc_14": round(float(np.mean(list(aucs.values()))), 4),
            "test_per_class_auc": {k: round(v, 4) for k, v in aucs.items()},
            "val_pneumonia_auc": round(float(ck["pneumonia_auc"]), 4),
            "n_test": int(len(y)), "n_test_pneumonia": int(y[:, PNEU].sum()),
            "chexnet_reference_pneumonia_auc": 0.7680,
        }
        (dest / "test_metrics_locked.json").write_text(json.dumps(res, indent=2),
                                                       encoding="utf-8")
        print("\n" + "=" * 62)
        print(f"LOCKED TEST  pneumonia AUC = {res['test_pneumonia_auc']}  "
              f"(CheXNet reference 0.7680)")
        print(f"             pneumonia F1  = {res['test_pneumonia_f1']}  "
              f"at prevalence {100*res['n_test_pneumonia']/res['n_test']:.2f}%")
        print(f"             mean AUC over 14 findings = {res['test_mean_auc_14']}")
        print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
