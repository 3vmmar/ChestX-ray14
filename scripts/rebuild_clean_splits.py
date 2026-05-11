"""
rebuild_clean_splits.py
=======================
NIH ChestX-ray14 — binary pneumonia detection data pipeline.

Strategy : NIH-only, 14 classes capped at 1431 samples each,
           patient-wise 70 / 15 / 15 split.

Run once before training any model:
    python scripts/rebuild_clean_splits.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# ------- PATHS -------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / 'data' / 'raw'
DATA_SPLITS  = PROJECT_ROOT / 'data' / 'splits'
DATA_META    = PROJECT_ROOT / 'data' / 'metadata'
NIH_META     = DATA_RAW / 'Data_Entry_2017.csv'

SEED          = 42
CAP_PER_CLASS = 1431   # pneumonia class size — cap all classes here
MIN_PER_CLASS = 1431   # drop any class below this
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
TEST_FRAC     = 0.15
IMAGE_DIR     = DATA_RAW / 'images'

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
    print(f"WARNING: {missing} images could not be found in data/raw!")

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
# SECTION C: COUNT CLASSES AND FILTER
# ═══════════════════════════════════════════════════════════════
print("\n=== SECTION C: Class counts and filtering ===")

class_counts = df_nih['class_name'].value_counts()
print("Samples per class (full dataset):")
for cls, cnt in class_counts.items():
    print(f"  {cls:<25} {cnt:>6,}")

# Keep only classes meeting minimum threshold
keep_classes   = class_counts[class_counts >= MIN_PER_CLASS].index.tolist()
drop_classes   = class_counts[class_counts <  MIN_PER_CLASS].index.tolist()

print(f"\nDropped classes (count < {MIN_PER_CLASS}):")
for cls in drop_classes:
    print(f"  {cls:<25} {class_counts[cls]:>6,}  <-- DROPPED")

df_filtered = df_nih[df_nih['class_name'].isin(keep_classes)].copy()
print(f"\nAfter filtering: {len(df_filtered):,} rows, {len(keep_classes)} classes kept")

# ═══════════════════════════════════════════════════════════════
# SECTION D: CAP EACH CLASS AT CAP_PER_CLASS SAMPLES
# ═══════════════════════════════════════════════════════════════
print("\n=== SECTION D: Capping classes at {CAP_PER_CLASS} samples ===")

rng = np.random.default_rng(SEED)
capped_frames = []

for cls in keep_classes:
    df_cls = df_filtered[df_filtered['class_name'] == cls]
    if len(df_cls) > CAP_PER_CLASS:
        df_cls = df_cls.sample(n=CAP_PER_CLASS, random_state=SEED)
    capped_frames.append(df_cls)

df_capped = pd.concat(capped_frames, ignore_index=True)

# After capping: Pneumonia=1, everything else=0
df_capped['label'] = (df_capped['class_name'] == 'Pneumonia').astype(int)

print("Counts after capping:")
for cls in keep_classes:
    n   = (df_capped['class_name'] == cls).sum()
    lbl = 1 if cls == 'Pneumonia' else 0
    print(f"  {cls:<25} {n:>6,}  label={lbl}")

n_pos = df_capped['label'].sum()
n_neg = len(df_capped) - n_pos
print(f"\nTotal capped: {len(df_capped):,}  |  Pos={n_pos:,}  |  Neg={n_neg:,}")
print(f"Pos rate: {n_pos/len(df_capped)*100:.2f}%")

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

def split_stats(df):
    total = len(df)
    pos   = int(df['label'].sum())
    neg   = total - pos
    rate  = round(pos / total, 4) if total > 0 else 0.0
    return total, pos, neg, rate

tr_total, tr_pos, tr_neg, tr_rate = split_stats(df_train)
vl_total, vl_pos, vl_neg, vl_rate = split_stats(df_val)
te_total, te_pos, te_neg, te_rate = split_stats(df_test)

summary = {
    "strategy":         "NIH-only, capped per class, patient-wise 70/15/15",
    "cap_per_class":    CAP_PER_CLASS,
    "classes_kept":     sorted(keep_classes),
    "classes_dropped":  sorted(drop_classes),
    "train_total":      tr_total,
    "train_pos":        tr_pos,
    "train_neg":        tr_neg,
    "train_pos_rate":   tr_rate,
    "val_total":        vl_total,
    "val_pos":          vl_pos,
    "val_neg":          vl_neg,
    "val_pos_rate":     vl_rate,
    "test_total":       te_total,
    "test_pos":         te_pos,
    "test_neg":         te_neg,
    "test_pos_rate":    te_rate,
    "seed":             SEED,
}

with open(DATA_META / 'dataset_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

# ═══════════════════════════════════════════════════════════════
# SECTION I: FINAL AUDIT PRINT
# ═══════════════════════════════════════════════════════════════
print("\n================================================")
print("FINAL SPLIT AUDIT")
print("================================================")
print(f"{'Split':<8} | {'Total':>6} | {'Pos':>5} | {'Neg':>5} | {'Pos Rate':>8}")
print(f"{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}-+-{'-'*9}")
print(f"{'Train':<8} | {tr_total:>6} | {tr_pos:>5} | {tr_neg:>5} | {tr_rate*100:>7.2f}%")
print(f"{'Val':<8} | {vl_total:>6} | {vl_pos:>5} | {vl_neg:>5} | {vl_rate*100:>7.2f}%")
print(f"{'Test':<8} | {te_total:>6} | {te_pos:>5} | {te_neg:>5} | {te_rate*100:>7.2f}%")
print("================================================")
print("Patient leakage: ZERO (verified)")
print(f"Saved: data/splits/train.csv")
print(f"Saved: data/splits/val.csv")
print(f"Saved: data/splits/test.csv")
print(f"Saved: data/metadata/dataset_summary.json")
print("================================================")
