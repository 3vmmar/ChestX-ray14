#!/usr/bin/env python
"""
patch_notebooks_schedule_fix.py — repair the training schedule after the
curriculum patch.

Three problems, found by diagnosing the DenseNet121 retrain (val AUC fell
0.6572 -> 0.6325):

1. BROKEN STAGE-A LEARNING RATE  (regression introduced by the curriculum patch)
   The stage-A scheduler is CosineAnnealingLR(T_max=WARMUP_EPOCHS). The
   curriculum patch moved PARTIAL_UNFREEZE_EPOCH from 3 to 9, so stage A now
   spans 8 epochs, but WARMUP_EPOCHS stayed at 2. The cosine finished its
   cycle in 2 epochs and then sawtoothed:
       1e-4 -> 5e-5 -> 1e-6 -> 5e-5 -> 1e-4 -> 5e-5 -> 1e-6 -> 5e-5
   visible in training_history.csv as lr collapsing to ~0 at epochs 3 and 7.
   Fix: WARMUP_EPOCHS is derived from the actual stage-A length.

2. PATIENCE CALIBRATED ON THE WRONG MODEL
   Patience was cut 20 -> 8 based on ViT-B/16, which peaks at epoch 13 and
   decays. DenseNet121 peaked at epoch 40. Patience 8 stopped the retrain at
   epoch 28 with its best at 20, giving ~10 epochs of full fine-tuning versus
   ~36 in the original run.
   Fix: patience is not counted while the backbone is frozen. Those epochs
   cannot improve much by construction, so they should not consume the budget.
   The counter starts at FULL_UNFREEZE_EPOCH.

3. SHRUNKEN FULL-FINE-TUNING BUDGET
   The curriculum spends 9 epochs frozen, so NUM_EPOCHS=50 left 41 epochs of
   full fine-tuning against 47 before.
   Fix: NUM_EPOCHS is raised by the curriculum offset so the post-unfreeze
   budget matches the original.

Usage:
    python scripts/patch_notebooks_schedule_fix.py --dry-run
    python scripts/patch_notebooks_schedule_fix.py
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

MARKER = "# [SCHEDULE-FIX]"

CONST_BLOCK = f"""
{MARKER} schedule repair - see scripts/patch_notebooks_schedule_fix.py
# WARMUP_EPOCHS must span the actual stage-A length, or the cosine schedule
# restarts mid-warmup and the LR sawtooths between 1e-4 and 1e-6.
WARMUP_EPOCHS = PARTIAL_UNFREEZE_EPOCH - 1

# Frozen-backbone epochs cannot improve much by construction, so they should
# not burn the early-stopping budget. Counting starts once the backbone is
# fully unfrozen.
PATIENCE_STARTS_AT = FULL_UNFREEZE_EPOCH

# Keep the post-unfreeze budget equal to the pre-curriculum run.
NUM_EPOCHS = NUM_EPOCHS + (PHASE1_EPOCHS + PHASE2_EPOCHS)
"""

PATIENCE_GUARD = """    if epoch < PATIENCE_STARTS_AT:
        epochs_no_improve = 0   # frozen phases do not consume patience
"""


def patch_one(path, dry_run=False):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    joined = "\n".join("".join(c["source"]) for c in cells)
    if MARKER in joined:
        return "already patched"
    if "[CURRICULUM-PATCH]" not in joined:
        return "FAILED: curriculum patch not applied first"

    changes = []

    # ---- constants ---------------------------------------------------------
    hp = next((c for c in cells if "LOCKED_TEST_EVAL" in "".join(c["source"])), None)
    if hp is None:
        return "FAILED: hyperparameter cell not found"
    s = "".join(hp["source"])
    anchor = re.search(r"LOCKED_TEST_EVAL\s*=\s*\w+.*?\n", s)
    if not anchor:
        return "FAILED: LOCKED_TEST_EVAL anchor missing"
    s = s[:anchor.end()] + CONST_BLOCK + s[anchor.end():]
    hp["source"] = s.splitlines(keepends=True)
    changes.append("WARMUP/patience/epochs")

    # ---- patience guard in the loop ---------------------------------------
    tl = next((c for c in cells
               if "epochs_no_improve += 1" in "".join(c["source"])), None)
    if tl is None:
        return "FAILED: training loop not found"
    s = "".join(tl["source"])
    s, n = re.subn(
        r"(\n)(    if EARLY_STOP_ENABLED and epochs_no_improve >= EARLY_STOP_PATIENCE:)",
        r"\1" + PATIENCE_GUARD + r"\2", s, count=1)
    if not n:
        return "FAILED: could not insert patience guard"
    tl["source"] = s.splitlines(keepends=True)
    changes.append("patience guard")

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
