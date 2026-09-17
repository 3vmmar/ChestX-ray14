#!/usr/bin/env python
"""
patch_notebooks_curriculum.py — apply the agreed training/evaluation changes
to all 12 model notebooks.

Changes applied to each notebook:

  1. EVALUATION LEAK FIX (the important one)
     Today the checkpoint is chosen by val AUC AND the decision threshold is
     tuned on the same val TTA probabilities that are then reported. Every
     headline number is therefore a maximum taken over the set it is reported
     on. This appends a LOCKED TEST EVALUATION cell: the val-tuned threshold
     is frozen, and test.csv -- never used for checkpoint or threshold
     selection -- is scored once with it.

  2. EARLY STOPPING  patience 20 -> 8
     Val AUC peaks around epoch 13 and decays for the remaining 20 epochs.

  3. DATA CURRICULUM  phases 1/2/3 by measured negative difficulty
     Phase 1 Pneumonia vs No Finding, phase 2 adds moderate negatives,
     phase 3 adds Edema/Consolidation/Infiltration. The sampler is rebuilt
     per phase because the ratio moves from 1:1 to 1:11.

  4. UNFREEZE SCHEDULE shifted into phase 3
     Each notebook's relative A->B->C cadence is preserved, just offset so
     that progressive unfreezing begins when the full data does.

Idempotent: a notebook already carrying the marker is skipped.

Usage:
    python scripts/patch_notebooks_curriculum.py --dry-run
    python scripts/patch_notebooks_curriculum.py
    python scripts/patch_notebooks_curriculum.py --revert
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"

NOTEBOOKS = [
    "DenseNet121", "DenseNet201", "EfficientNetB3", "ResNet50", "ResNet101",
    "VGG16", "MobileNetV2", "InceptionV3", "Xception", "ViTB16", "SwinT", "DeiTS",
]

MARKER = "# [CURRICULUM-PATCH]"
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 3

# ── 1. hyperparameters ──────────────────────────────────────────────────────
CONST_BLOCK = f"""
{MARKER} data curriculum + leak-free evaluation
CURRICULUM_ENABLED = True
PHASE1_EPOCHS      = {PHASE1_EPOCHS}   # Pneumonia vs No Finding
PHASE2_EPOCHS      = {PHASE2_EPOCHS}   # + moderate negatives
PHASE3_START       = PHASE1_EPOCHS + PHASE2_EPOCHS + 1   # + hard negatives; unfreezing starts here
LOCKED_TEST_EVAL   = True    # score test.csv once, with the threshold frozen from val
"""

# ── 2. phase-aware loader (appended to the loader cell) ─────────────────────
LOADER_BLOCK = f'''

{MARKER} rebuild the training loader for a given data phase
import sys as _sys
if str(PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT))
from src.training import curriculum as _curr

def build_phase_loader(phase):
    """Training loader restricted to the classes visible in `phase`.

    The sampler is recomputed every phase: the positive:negative ratio moves
    from about 1:1 in phase 1 to 1:11 in phase 3, so weights carried over
    from an earlier phase would oversample positives badly.
    """
    df_ph = _curr.phase_frame(df_train, phase)
    w = _curr.sample_weights(df_ph['target_pneumonia'].values, TARGET_POS_FRAC)
    smp = WeightedRandomSampler(
        weights=torch.tensor(w, dtype=torch.float64),
        num_samples=len(w), replacement=True
    )
    dl = DataLoader(
        ChestXrayDataset(df_ph, transform=train_transforms),
        batch_size=BATCH_SIZE, sampler=smp, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    print('  ' + _curr.describe_phase(df_ph, phase))
    return dl
'''

# ── 3. phase switch inside the epoch loop ───────────────────────────────────
PHASE_SWITCH = f"""    {MARKER} switch training data when the phase changes
    _phase_now = _curr.phase_for_epoch(epoch, PHASE1_EPOCHS, PHASE2_EPOCHS) if CURRICULUM_ENABLED else 3
    if _phase_now != _current_phase:
        _current_phase = _phase_now
        train_loader = build_phase_loader(_current_phase)
"""

# ── 4. locked test evaluation (new final cell) ──────────────────────────────
TEST_CELL = f'''{MARKER} LOCKED TEST EVALUATION
# ---------------------------------------------------------------------------
# The numbers above are VALIDATION numbers, and both the checkpoint and the
# decision threshold were selected on that same validation set - so they are
# optimistic by construction.
#
# Here test.csv is scored ONCE. The threshold is frozen at the value tuned on
# validation; it is NOT re-tuned. Nothing about the test set influenced the
# checkpoint, the threshold, or any hyperparameter. These are the numbers to
# report.
# ---------------------------------------------------------------------------
if LOCKED_TEST_EVAL:
    LOCKED_THRESHOLD = float(tta_metrics['tta_threshold'])   # frozen, from val
    print('=' * 70)
    print(f'LOCKED TEST EVALUATION | threshold frozen at {{LOCKED_THRESHOLD:.3f}} (tuned on val)')
    print('=' * 70)

    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_probs, test_labels = run_tta_inference(
        model, df_test, tta_transforms_list, DEVICE, BATCH_SIZE
    )

    test_roc = roc_auc_score(test_labels, test_probs)
    _p, _r, _ = precision_recall_curve(test_labels, test_probs)
    test_pr = sk_auc(_r, _p)

    test_preds = (test_probs >= LOCKED_THRESHOLD).astype(int)
    test_f1   = f1_score(test_labels, test_preds, zero_division=0)
    test_prec = precision_score(test_labels, test_preds, zero_division=0)
    test_rec  = recall_score(test_labels, test_preds, zero_division=0)
    t_tn, t_fp, t_fn, t_tp = confusion_matrix(test_labels, test_preds).ravel()

    print(f'TEST ROC-AUC : {{test_roc:.4f}}   (val {{tta_metrics["tta_roc_auc"]:.4f}})')
    print(f'TEST PR-AUC  : {{test_pr:.4f}}   (val {{tta_metrics["tta_pr_auc"]:.4f}})')
    print(f'TEST F1      : {{test_f1:.4f}}   (val {{tta_metrics["tta_f1"]:.4f}})')
    print(f'TEST P / R   : {{test_prec:.4f}} / {{test_rec:.4f}}')
    print(f'TEST CONF    : TP={{t_tp}} FP={{t_fp}} TN={{t_tn}} FN={{t_fn}}')
    print(f'OPTIMISM     : val - test ROC-AUC = {{tta_metrics["tta_roc_auc"] - test_roc:+.4f}}')
    print('=' * 70)

    test_metrics = {{
        'model': tta_metrics['model'],
        'evaluation': 'LOCKED_TEST_TTA_5variants',
        'threshold_source': 'validation (frozen, not re-tuned on test)',
        'threshold': round(LOCKED_THRESHOLD, 3),
        'best_epoch': int(best_epoch),
        'test_roc_auc': round(float(test_roc), 4),
        'test_pr_auc': round(float(test_pr), 4),
        'test_f1': round(float(test_f1), 4),
        'test_precision': round(float(test_prec), 4),
        'test_recall': round(float(test_rec), 4),
        'test_tp': int(t_tp), 'test_fp': int(t_fp),
        'test_tn': int(t_tn), 'test_fn': int(t_fn),
        'val_roc_auc': tta_metrics['tta_roc_auc'],
        'val_pr_auc': tta_metrics['tta_pr_auc'],
        'val_f1': tta_metrics['tta_f1'],
        'optimism_roc_auc': round(float(tta_metrics['tta_roc_auc'] - test_roc), 4),
    }}
    with open(OUTPUT_DIR / 'test_metrics_locked.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f'[SAVED] {{OUTPUT_DIR / "test_metrics_locked.json"}}')
'''


def code_cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def patch_one(path, dry_run=False):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = code_cells(nb)
    joined = "\n".join("".join(c["source"]) for c in cells)
    if MARKER in joined:
        return "already patched"

    changes = []

    # ---- 1. hyperparameters ------------------------------------------------
    hp = next((c for c in cells if "EARLY_STOP_PATIENCE" in "".join(c["source"])), None)
    if hp is None:
        return "FAILED: no hyperparameter cell"
    s = "".join(hp["source"])

    s, n = re.subn(r"EARLY_STOP_PATIENCE\s*=\s*\d+",
                   "EARLY_STOP_PATIENCE = 8", s)
    if not n:
        return "FAILED: EARLY_STOP_PATIENCE not matched"
    changes.append("patience->8")

    m_p = re.search(r"PARTIAL_UNFREEZE_EPOCH\s*=\s*(\d+)", s)
    m_f = re.search(r"FULL_UNFREEZE_EPOCH\s*=\s*(\d+)", s)
    if not (m_p and m_f):
        return "FAILED: unfreeze epochs not matched"
    shift = PHASE1_EPOCHS + PHASE2_EPOCHS
    new_p, new_f = int(m_p.group(1)) + shift, int(m_f.group(1)) + shift
    s = re.sub(r"PARTIAL_UNFREEZE_EPOCH\s*=\s*\d+",
               f"PARTIAL_UNFREEZE_EPOCH = {new_p}", s)
    s = re.sub(r"FULL_UNFREEZE_EPOCH\s*=\s*\d+",
               f"FULL_UNFREEZE_EPOCH    = {new_f}", s)
    changes.append(f"unfreeze {m_p.group(1)}/{m_f.group(1)}->{new_p}/{new_f}")

    anchor = re.search(r"EARLY_STOP_ENABLED\s*=\s*\w+\n", s)
    if not anchor:
        return "FAILED: EARLY_STOP_ENABLED anchor missing"
    s = s[:anchor.end()] + CONST_BLOCK + s[anchor.end():]
    hp["source"] = s.splitlines(keepends=True)
    changes.append("curriculum consts")

    # ---- 2. phase-aware loader --------------------------------------------
    ld = next((c for c in cells if "WeightedRandomSampler(" in "".join(c["source"])
               and "train_loader" in "".join(c["source"])), None)
    if ld is None:
        return "FAILED: loader cell not found"
    ld["source"] = ("".join(ld["source"]) + LOADER_BLOCK).splitlines(keepends=True)
    changes.append("phase loader")

    # ---- 3. phase switch in the epoch loop --------------------------------
    tl = next((c for c in cells if re.search(r"for epoch in range\(1,\s*NUM_EPOCHS",
                                             "".join(c["source"]))), None)
    if tl is None:
        return "FAILED: training loop not found"
    s = "".join(tl["source"])
    s, n = re.subn(r"(\n)(for epoch in range\(1,\s*NUM_EPOCHS \+ 1\):\n)",
                   r"\1_current_phase = None\n\2", s, count=1)
    if not n:
        return "FAILED: could not insert _current_phase"
    s, n = re.subn(r"(for epoch in range\(1,\s*NUM_EPOCHS \+ 1\):\n\s*epoch_start = time\.time\(\)\n)",
                   r"\1" + PHASE_SWITCH, s, count=1)
    if not n:
        return "FAILED: could not insert phase switch"
    tl["source"] = s.splitlines(keepends=True)
    changes.append("phase switch")

    # ---- 4. locked test evaluation ----------------------------------------
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": TEST_CELL.splitlines(keepends=True),
    })
    changes.append("locked test cell")

    if not dry_run:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return ", ".join(changes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", action="append")
    args = ap.parse_args()

    targets = args.only or NOTEBOOKS
    ok = fail = skip = 0
    for name in targets:
        p = NB_DIR / f"{name}.ipynb"
        if not p.exists():
            print(f"  {name:<16} MISSING")
            fail += 1
            continue
        res = patch_one(p, args.dry_run)
        if res.startswith("FAILED"):
            fail += 1
        elif res == "already patched":
            skip += 1
        else:
            ok += 1
        print(f"  {name:<16} {res}")
    print(f"\npatched={ok} skipped={skip} failed={fail}"
          + ("  (dry run - nothing written)" if args.dry_run else ""))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
