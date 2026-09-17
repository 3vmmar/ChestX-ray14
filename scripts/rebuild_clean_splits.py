"""
rebuild_clean_splits.py
=======================
NIH ChestX-ray14 — multi-class lung disease classification pipeline.

Strategy : NIH-only, keep all 14 disease classes (excl. Hernia),
           capped at 1431 samples each, patient-wise 70 / 15 / 15 split.

Run once before training any model:
    python scripts/rebuild_clean_splits.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# ------- PATHS -------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / 'data' / 'archive'
DATA_SPLITS  = PROJECT_ROOT / 'data' / 'splits'
DATA_META    = PROJECT_ROOT / 'data' / 'metadata'
NIH_META     = DATA_RAW / 'Data_Entry_2017.csv'

SEED          = 42
CAP_PER_CLASS = 1431   # cap each class at this many samples
DROP_CLASSES  = ['Hernia']   # always dropped; classes < CAP_PER_CLASS are added dynamically below
drop_classes  = set(DROP_CLASSES)  # filled in Section C with any extra classes
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
TEST_FRAC     = 0.15
# ═══════════════════════════════════════════════════════════════
# SECTION A: LOAD NIH METADATA
# ═══════════════════════════════════════════════════════════════
print("=== SECTION A: Loading NIH Metadata ===")
df_nih = pd.read_csv(NIH_META)
df_nih['pid_str']    = df_nih['Patient ID'].astype(str)
print("Building image path mapping...")
image_paths = {f.name: str(f) for f in DATA_RAW.rglob('*.png')}
df_nih['image_path'] = df_nih['Image Index'].map(image_paths)

missing = df_nih['image_path'].isna().sum()
if missing > 0:
    print(f"WARNING: {missing} images could not be found in data/archive!")

print(f"Loaded {len(df_nih):,} rows from {NIH_META.name}")

# ═══════════════════════════════════════════════════════════════
# SECTION B: HANDLE MULTI-LABEL ROWS
# ═══════════════════════════════════════════════════════════════
print("\n=== SECTION B: Assigning labels ===")

def assign_label(finding_labels_str):
    """
    Rule:
      - 'Pneumonia' anywhere  → label=1, class_name='Pneumonia'
      - 'No Finding' exactly  → label=0, class_name='No Finding'
      - otherwise             → label=0, class_name=first label before '|'
    """
    s = str(finding_labels_str)
    if 'Pneumonia' in s:
        return 1, 'Pneumonia'
    elif s == 'No Finding':
        return 0, 'No Finding'
    else:
        primary = s.split('|')[0].strip()
        return 0, primary

labels_and_classes = df_nih['Finding Labels'].apply(assign_label)
df_nih['label']      = labels_and_classes.apply(lambda x: x[0])
df_nih['class_name'] = labels_and_classes.apply(lambda x: x[1])

print(f"Label distribution:\n{df_nih['label'].value_counts().to_string()}")

# ═══════════════════════════════════════════════════════════════
# SECTION C: REMOVE DROP CLASSES
# ═══════════════════════════════════════════════════════════════
print("\n=== SECTION C: Removing drop classes ===")

class_counts = df_nih['class_name'].value_counts()
print("Samples per class (full dataset):")
for cls, cnt in class_counts.items():
    print(f"  {cls:<25} {cnt:>6,}")

# Dynamically drop any class with fewer than CAP_PER_CLASS samples
insufficient = [c for c in class_counts.index if class_counts[c] < CAP_PER_CLASS and c not in drop_classes]
for cls in insufficient:
    drop_classes.add(cls)
    print(f"  {cls:<25} {class_counts[cls]:>6,}  <-- INSUFFICIENT SAMPLES, DROPPED")

keep_classes = [c for c in class_counts.index if c not in drop_classes]
print(f"\nFinal dropped classes:")
for cls in sorted(drop_classes):
    print(f"  {cls:<25} {class_counts[cls]:>6,}")

df_filtered = df_nih[df_nih['class_name'].isin(keep_classes)].copy()
print(f"\nAfter filtering: {len(df_filtered):,} rows, {len(keep_classes)} classes kept")

# ═══════════════════════════════════════════════════════════════
# SECTION D: CAP EACH CLASS AT CAP_PER_CLASS SAMPLES
# ═══════════════════════════════════════════════════════════════
print(f"\n=== SECTION D: Capping classes at {CAP_PER_CLASS} samples ===")

rng = np.random.default_rng(SEED)
capped_frames = []

for cls in keep_classes:
    df_cls = df_filtered[df_filtered['class_name'] == cls]
    if len(df_cls) > CAP_PER_CLASS:
        df_cls = df_cls.sample(n=CAP_PER_CLASS, random_state=SEED)
    capped_frames.append(df_cls)

df_capped = pd.concat(capped_frames, ignore_index=True)

# Multi-class label encoding: sort alphabetically for deterministic mapping
class_names = sorted(df_capped['class_name'].unique().tolist())
class_to_label = {name: idx for idx, name in enumerate(class_names)}
df_capped['label'] = df_capped['class_name'].map(class_to_label)

print("\nClass-to-label mapping (alphabetical):")
for name, label in class_to_label.items():
    print(f"  label={label:<2}  {name}")

print("\nCounts after capping:")
for cls in sorted(keep_classes):
    n = (df_capped['class_name'] == cls).sum()
    lbl = class_to_label[cls]
    print(f"  label={lbl:<2}  {cls:<25} {n:>6,}")

print(f"\nTotal capped: {len(df_capped):,}  |  Classes: {len(class_names)}")

# ═══════════════════════════════════════════════════════════════
# SECTION E: PATIENT-WISE 70 / 15 / 15 SPLIT
# ═══════════════════════════════════════════════════════════════
print("\n=== SECTION E: Patient-wise 70/15/15 split ===")

# E1 — unique patient IDs
unique_pids = df_capped['pid_str'].unique()
print(f"Unique patients in capped dataset: {len(unique_pids):,}")

# E2 — shuffle
rng_split = np.random.default_rng(SEED)
shuffled_pids = rng_split.permutation(unique_pids)

# E3 — split patient IDs
n_total   = len(shuffled_pids)
n_train   = int(np.floor(TRAIN_FRAC * n_total))
n_val     = int(np.floor(VAL_FRAC   * n_total))
# test gets the remainder to ensure every patient is assigned
train_pids = set(shuffled_pids[:n_train])
val_pids   = set(shuffled_pids[n_train : n_train + n_val])
test_pids  = set(shuffled_pids[n_train + n_val :])

print(f"Patient split  -> Train: {len(train_pids):,}  Val: {len(val_pids):,}  Test: {len(test_pids):,}")

# E5 - Assert zero patient overlap
assert len(train_pids & val_pids)  == 0, \
    f"LEAKAGE: {len(train_pids & val_pids)} patients overlap between train and val!"
assert len(train_pids & test_pids) == 0, \
    f"LEAKAGE: {len(train_pids & test_pids)} patients overlap between train and test!"
assert len(val_pids   & test_pids) == 0, \
    f"LEAKAGE: {len(val_pids & test_pids)} patients overlap between val and test!"
print("Patient leakage assertions: ALL PASSED OK")

# E4 - Assign rows to splits
df_capped['patient_split'] = 'unassigned'
df_capped.loc[df_capped['pid_str'].isin(train_pids), 'patient_split'] = 'train'
df_capped.loc[df_capped['pid_str'].isin(val_pids),   'patient_split'] = 'val'
df_capped.loc[df_capped['pid_str'].isin(test_pids),  'patient_split'] = 'test'

unassigned = (df_capped['patient_split'] == 'unassigned').sum()
if unassigned > 0:
    raise ValueError(f"BUG: {unassigned} rows were not assigned to any split!")

# ═══════════════════════════════════════════════════════════════
# SECTION F: BUILD FINAL SPLIT DATAFRAMES
# ═══════════════════════════════════════════════════════════════
print("\n=== SECTION F: Building final split dataframes ===")

FINAL_COLS = ['image_path', 'label', 'class_name', 'patient_id',
              'view_position', 'source_weight', 'split_tag']

def build_split_df(df_src, split_name):
    df_out = df_src.copy()
    df_out['patient_id']    = df_out['pid_str']
    df_out['view_position'] = df_out['View Position']
    df_out['source_weight'] = 1.0
    df_out['split_tag']     = split_name
    return df_out[FINAL_COLS].reset_index(drop=True)

df_train_raw = df_capped[df_capped['patient_split'] == 'train']
df_val_raw   = df_capped[df_capped['patient_split'] == 'val']
df_test_raw  = df_capped[df_capped['patient_split'] == 'test']

df_train = build_split_df(df_train_raw, 'train')
df_val   = build_split_df(df_val_raw,   'val')
df_test  = build_split_df(df_test_raw,  'test')

# ═══════════════════════════════════════════════════════════════
# SECTION G: SAVE SPLITS
# ═══════════════════════════════════════════════════════════════
print("\n=== SECTION G: Saving split CSVs ===")

DATA_SPLITS.mkdir(parents=True, exist_ok=True)
df_train.to_csv(DATA_SPLITS / 'train.csv', index=False)
df_val.to_csv(DATA_SPLITS   / 'val.csv',   index=False)
df_test.to_csv(DATA_SPLITS  / 'test.csv',  index=False)

# ═══════════════════════════════════════════════════════════════
# SECTION H: SAVE METADATA SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n=== SECTION H: Saving dataset_summary.json ===")

DATA_META.mkdir(parents=True, exist_ok=True)

def split_per_class_counts(df, class_names):
    counts = {}
    for name in class_names:
        counts[name] = int((df['class_name'] == name).sum())
    return counts

class_names_sorted = sorted(keep_classes)
summary = {
    "strategy":          "NIH-only, capped per class, patient-wise 70/15/15",
    "task":              "multi-class lung disease classification",
    "cap_per_class":     CAP_PER_CLASS,
    "classes_kept":      class_names_sorted,
    "classes_dropped":   sorted(drop_classes),
    "class_to_label":    {name: idx for idx, name in enumerate(class_names_sorted)},
    "seed":              SEED,
    "splits": {
        "train": {
            "total": int(len(df_train)),
            "per_class": split_per_class_counts(df_train, class_names_sorted),
        },
        "val": {
            "total": int(len(df_val)),
            "per_class": split_per_class_counts(df_val, class_names_sorted),
        },
        "test": {
            "total": int(len(df_test)),
            "per_class": split_per_class_counts(df_test, class_names_sorted),
        },
    },
}

with open(DATA_META / 'dataset_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("Class mapping:")
for name, label in class_to_label.items():
    print(f"  label={label:<2}  {name}")

# ═══════════════════════════════════════════════════════════════
# SECTION I: FINAL AUDIT PRINT
# ═══════════════════════════════════════════════════════════════
print("\n================================================")
print("FINAL SPLIT AUDIT")
print("================================================")
print(f"Total samples: {len(df_capped):,} across {len(class_names)} classes")
print(f"Classes kept: {len(class_names)} | Dropped: {sorted(drop_classes)}")
print()
for split_name, df_sp in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    print(f"  {split_name:<6}: {len(df_sp):>6} samples")
    for name in class_names_sorted:
        cnt = int((df_sp['class_name'] == name).sum())
        pct = cnt / len(df_sp) * 100 if len(df_sp) > 0 else 0
        print(f"         {name:<25} {cnt:>6,}  ({pct:>5.2f}%)")
    print()
print("Patient leakage: ZERO (verified)")
print(f"Saved: data/splits/train.csv")
print(f"Saved: data/splits/val.csv")
print(f"Saved: data/splits/test.csv")
print(f"Saved: data/metadata/dataset_summary.json")
print("================================================")
