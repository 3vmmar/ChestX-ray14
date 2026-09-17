#!/usr/bin/env python
"""
patch_notebooks_preprocessing.py — point all 12 notebooks at src/preprocessing.py.

Each notebook defined its own CLAHE class and transform stack, duplicated 12
times and drifting from the copies in scripts/. This replaces the definitions
with imports from the shared module, which fixes (see that module for the
measurements):

  * train/val scale mismatch    train saw anatomy 11% larger than val
  * black wedges from rotation  9.1% of every rotated training image, now 0.0%
  * RandomVerticalFlip(p=0.1)   anatomically impossible chest X-rays
  * RandomGrayscale(p=0.05)     verified no-op after CLAHE
  * redundant 512 downscale     CLAHE now at native res, tiles held at ~64px

Names the rest of each notebook depends on (IMAGENET_MEAN/STD, CLAHETransform,
TRAIN_RESIZE, train_transforms, val_transforms) are re-exported so no other
cell needs changing.

Usage:
    python scripts/patch_notebooks_preprocessing.py --dry-run
    python scripts/patch_notebooks_preprocessing.py
"""

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"

NOTEBOOKS = [
    "DenseNet121", "DenseNet201", "EfficientNetB3", "ResNet50", "ResNet101",
    "VGG16", "MobileNetV2", "InceptionV3", "Xception", "ViTB16", "SwinT", "DeiTS",
]

MARKER = "# [PREPROC-PATCH]"

NEW_PREPROC = f'''{MARKER} transforms centralised in src/preprocessing.py
# Fixes applied there (each measured on real NIH data before changing):
#   - train/val scale mismatch: train saw anatomy 11% larger than validation
#   - rotation ran after the crop, filling 9.1% of each rotated image with
#     black; it now runs before an oversized crop, measured 0.0%
#   - RandomVerticalFlip(0.1) produced anatomically impossible radiographs
#   - RandomGrayscale(0.05) was a verified no-op after CLAHE
#   - CLAHE's "MemoryError guard" downscale to 512 was unnecessary (0.93 ms,
#     1 MB at full res) and cost an extra resample; tiles are now held at
#     ~64px regardless of input size instead of a fixed 8x8 grid
import sys as _sys
if str(PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT))
from src import preprocessing as _prep

IMAGENET_MEAN  = _prep.IMAGENET_MEAN
IMAGENET_STD   = _prep.IMAGENET_STD
CLAHETransform = _prep.CLAHETransform
TRAIN_RESIZE   = _prep.train_resize_for(IMG_SIZE)

train_transforms = _prep.build_train_transforms(IMG_SIZE)
val_transforms   = _prep.build_val_transforms(IMG_SIZE)

print(f'Train: CLAHE->Resize({{TRAIN_RESIZE}})->Rot({{_prep.ROTATION_DEG}})->'
      f'SafeCrop({{_prep.safe_size_for(IMG_SIZE)}})->RandCrop({{IMG_SIZE}})->HFlip->Jitter->Norm')
print(f'Val  : CLAHE->Resize({{TRAIN_RESIZE}})->CenterCrop({{IMG_SIZE}})->Norm   '
      f'(same field of view as training)')
'''

NEW_TTA = (f"{MARKER} TTA variants share the evaluation geometry exactly\n"
           "tta_transforms_list = _prep.build_tta_transforms(IMG_SIZE)\n")


def patch_one(path, dry_run=False):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if MARKER in "\n".join("".join(c["source"]) for c in cells):
        return "already patched"

    changes = []

    # ---- preprocessing cell -------------------------------------------------
    pc = next((c for c in cells if "class CLAHETransform" in "".join(c["source"])
               and "train_transforms" in "".join(c["source"])), None)
    if pc is None:
        return "FAILED: preprocessing cell not found"
    pc["source"] = NEW_PREPROC.splitlines(keepends=True)
    changes.append("transforms -> src/preprocessing")

    # ---- TTA list -----------------------------------------------------------
    tc = next((c for c in cells if "tta_transforms_list = [" in "".join(c["source"])), None)
    if tc is None:
        return "FAILED: tta_transforms_list not found"
    s = "".join(tc["source"])
    s2, n = re.subn(r"tta_transforms_list = \[.*?\n\]\n", NEW_TTA, s, count=1, flags=re.S)
    if not n:
        return "FAILED: could not replace tta_transforms_list"
    tc["source"] = s2.splitlines(keepends=True)
    changes.append("TTA -> build_tta_transforms")

    if not dry_run:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return ", ".join(changes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ok = fail = skip = 0
    for name in NOTEBOOKS:
        p = NB_DIR / f"{name}.ipynb"
        res = patch_one(p, args.dry_run) if p.exists() else "FAILED: missing"
        if res.startswith("FAILED"):
            fail += 1
        elif res == "already patched":
            skip += 1
        else:
            ok += 1
        print(f"  {name:<16} {res}")
    print(f"\npatched={ok} skipped={skip} failed={fail}"
          + ("  (dry run)" if args.dry_run else ""))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
