"""
patch_notebooks.py
==================
Applies the complete data-pipeline rebuild to all 14 notebooks.

Changes applied to each of the 12 model notebooks:
  A  Replace old CSV loading block → new train/val/test.csv block
  B  Remove val_clean_dataset / val_clean_loader
  C  Remove CLEAN AUC print from training loop
  D  FocalLoss alpha = 0.75
  E  TARGET_POS_FRAC = 0.20

Changes applied to Team505_Preprocessing.ipynb:
  Replace all manifest-generation cells with one markdown notice cell.

Changes applied to Team505_EDA.ipynb:
  Replace old split-loading cell with new train/val/test.csv loading.

Run:
    python scripts/patch_notebooks.py
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = PROJECT_ROOT / "notebooks"

# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────

def load_nb(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_nb(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"  [SAVED] {path.name}")

def cell_source(cell):
    """Return the cell source as a single string."""
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src

def set_cell_source(cell, text):
    """Set the cell source from a single string (split into lines for JSON)."""
    lines = text.splitlines(keepends=True)
    cell["source"] = lines

def source_contains(cell, pattern):
    return pattern in cell_source(cell)

def source_matches(cell, regex_pattern):
    return bool(re.search(regex_pattern, cell_source(cell)))

# ─────────────────────────────────────────────────────────────
# NEW DATA-LOADING BLOCK (Change A)
# ─────────────────────────────────────────────────────────────
NEW_DATA_LOADING = """\
if RUN_MODE == "full":
    df_train = pd.read_csv(DATA_SPLITS / 'train.csv')
    df_val   = pd.read_csv(DATA_SPLITS / 'val.csv')
    df_test  = pd.read_csv(DATA_SPLITS / 'test.csv')
elif RUN_MODE == "dev":
    df_train = pd.read_csv(DATA_SPLITS / 'train.csv')
    df_val   = pd.read_csv(DATA_SPLITS / 'val.csv')
    df_test  = pd.read_csv(DATA_SPLITS / 'test.csv')
    df_train = df_train.sample(frac=0.2, random_state=42).reset_index(drop=True)
else:
    raise ValueError(f'Unknown RUN_MODE: {RUN_MODE}')

# Normalize label column
for _df in [df_train, df_val, df_test]:
    if 'label' in _df.columns and 'target_pneumonia' not in _df.columns:
        _df['target_pneumonia'] = _df['label']
    if 'source_weight' not in _df.columns:
        _df['source_weight'] = 1.0

_pos = 'target_pneumonia'
print(f'Train: {len(df_train):,} | Pos: {df_train[_pos].sum():.0f} | Rate: {df_train[_pos].mean()*100:.1f}%')
print(f'Val  : {len(df_val):,}   | Pos: {df_val[_pos].sum():.0f}   | Rate: {df_val[_pos].mean()*100:.1f}%')
print(f'Test : {len(df_test):,}  | Pos: {df_test[_pos].sum():.0f}  | Rate: {df_test[_pos].mean()*100:.1f}%')
"""

# ─────────────────────────────────────────────────────────────
# patch helpers
# ─────────────────────────────────────────────────────────────

def patch_data_loading(cell):
    """Change A: Replace the old CSV loading block."""
    src = cell_source(cell)
    if "train_clean_balanced.csv" not in src and "train.csv" not in src:
        return False
    # Only replace cells that still reference the old file
    if "train_clean_balanced.csv" not in src:
        return False
    set_cell_source(cell, NEW_DATA_LOADING)
    cell["outputs"] = []
    cell["execution_count"] = None
    return True


def patch_sampler_loader_cell(cell):
    """Changes B + D + E: Remove val_clean loader, fix alpha and TARGET_POS_FRAC."""
    src = cell_source(cell)
    changed = False

    # Change D: FocalLoss alpha
    if "criterion = FocalLoss(alpha=0.60" in src:
        src = src.replace(
            "criterion = FocalLoss(alpha=0.60, gamma=2.0)",
            "criterion = FocalLoss(alpha=0.75, gamma=2.0)"
        )
        changed = True

    # Change E: TARGET_POS_FRAC
    for old_frac in ("TARGET_POS_FRAC = 0.50", "TARGET_POS_FRAC = 0.40", "TARGET_POS_FRAC = 0.25"):
        if old_frac in src:
            src = src.replace(old_frac, "TARGET_POS_FRAC = 0.20")
            changed = True

    # Change B: Remove val_clean_dataset / val_clean_loader lines
    lines_to_drop = [
        "val_clean_dataset = ChestXrayDataset",
        "val_clean_loader = DataLoader(",
        "    val_clean_dataset,",
        "val_clean_dataset, batch_size=BATCH_SIZE",
        "print(f'Val clean loader:",
    ]
    new_lines = []
    skip_block = False
    for line in src.splitlines(keepends=True):
        # Detect start of val_clean_loader DataLoader block
        if "val_clean_loader = DataLoader(" in line:
            skip_block = True
        if skip_block:
            # end of block when we hit the closing paren line
            if line.strip() == ")":
                skip_block = False
                continue  # drop closing paren too
            continue
        # Drop standalone val_clean lines
        if any(marker in line for marker in lines_to_drop):
            changed = True
            continue
        new_lines.append(line)

    new_src = "".join(new_lines)
    if new_src != src:
        changed = True
        src = new_src

    # Also remove comment lines about old dataset strategy near TARGET_POS_FRAC
    old_comments = [
        "# Train ratio is 1:3 - target 25% positives per batch\n",
        "# Keeps batch distribution close to training prevalence (33%)\n",
        "# while slightly oversampling positives vs val/test prevalence\n",
        "# Dataset is now 1:1 balanced — target 50% positives per batch\n",
        "# Dataset is now 1:1 balanced \u2014 target 50% positives per batch\n",
        "# Slight positive oversampling to improve recall\n",
    ]
    for oc in old_comments:
        if oc in src:
            src = src.replace(oc, "")
            changed = True

    # Add the correct comment after TARGET_POS_FRAC line if not present
    new_comment = "# Val/test are now balanced (~7% pos rate after capping)\n# 20% positive per batch is a slight oversample \u2014 appropriate\n"
    if "TARGET_POS_FRAC = 0.20" in src and new_comment not in src:
        src = src.replace(
            "TARGET_POS_FRAC = 0.20\n",
            "TARGET_POS_FRAC = 0.20\n" + new_comment
        )
        changed = True

    if changed:
        set_cell_source(cell, src)
        cell["outputs"] = []
        cell["execution_count"] = None
    return changed


def patch_training_loop(cell):
    """Change C: Remove CLEAN AUC block from training loop."""
    src = cell_source(cell)
    if "val_clean_loader" not in src and "CLEAN" not in src:
        return False

    # Remove the clean-val try/except block
    clean_block_pattern = re.compile(
        r"    # Clean val ---.*?print\(f'  CLEAN : skipped \(\{_e\}\)'\)\n",
        re.DOTALL
    )
    new_src, n_subs = re.subn(clean_block_pattern, "", src)

    if n_subs == 0:
        # Try broader pattern
        clean_block_pattern2 = re.compile(
            r"    # Clean val ---[^\n]*\n.*?except Exception as _e:\n.*?print\(f'  CLEAN.*?'\)\n",
            re.DOTALL
        )
        new_src, n_subs = re.subn(clean_block_pattern2, "", src)

    if n_subs > 0 and new_src != src:
        set_cell_source(cell, new_src)
        cell["outputs"] = []
        cell["execution_count"] = None
        return True
    return False


def patch_model_notebook(nb_path):
    """Apply changes A–E to a model notebook."""
    nb = load_nb(nb_path)
    cells = nb["cells"]
    changes = {"A": False, "B": False, "C": False, "D": False, "E": False}

    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)

        # Change A — data loading
        if "train_clean_balanced.csv" in src or (
            "train.csv" in src and "DATA_SPLITS" in src and "RUN_MODE" in src
        ):
            if patch_data_loading(cell):
                changes["A"] = True

        # Changes B + D + E — sampler/loader cell
        if "FocalLoss" in src or "TARGET_POS_FRAC" in src or "val_clean_dataset" in src:
            if patch_sampler_loader_cell(cell):
                changes["B"] = True
                changes["D"] = True
                changes["E"] = True

        # Change C — training loop
        if "val_clean_loader" in src or ("CLEAN" in src and "validate" in src):
            if patch_training_loop(cell):
                changes["C"] = True

    print(f"\n  {nb_path.name}")
    for k, v in changes.items():
        status = "OK" if v else "— (already OK)"
        print(f"    Change {k}: {status}")

    save_nb(nb, nb_path)


# ─────────────────────────────────────────────────────────────
# STEP 4 — Team505_Preprocessing.ipynb
# ─────────────────────────────────────────────────────────────

PREPROCESSING_NOTICE_MD = (
    "# Data splits are generated by scripts/rebuild_clean_splits.py\n"
    "# Run that script once before training any models.\n"
    "# Splits: data/splits/train.csv, val.csv, test.csv\n"
    "# Strategy: NIH-only, 14 classes capped at 1431, patient-wise 70/15/15"
)

# Keywords that identify cells that generate old manifests
OLD_MANIFEST_KEYWORDS = [
    "train_d121_metadata_v3.csv",
    "train_d121_metadata_relaxed.csv",
    "train_clean_balanced.csv",
    "pneumonia_1",
    "pneumonia_2",
    "USE_EXTERNAL_POS",
    "external positives",
    # The section that saves train.csv / val.csv / test.csv in old format
    "df_train_dev",
    "GroupShuffleSplit",
]

# Cells to KEEP (by partial source match)
KEEP_KEYWORDS = [
    "# Team505",         # title markdown
    "## 0 - Imports",    # imports cell — keep imports
    "## 1 - Load Master",
    "Load official NIH metadata",
    "## 7 - Final Summary",  # summary stats — keep
    "PREPROCESSING SUMMARY",
    "kernelspec",
]


def patch_preprocessing_notebook(nb_path):
    nb = load_nb(nb_path)
    cells = nb["cells"]

    new_cells = []
    notice_inserted = False

    for cell in cells:
        src = cell_source(cell)
        # Check if this cell references old manifest logic
        is_old_manifest = any(kw in src for kw in OLD_MANIFEST_KEYWORDS)

        if is_old_manifest:
            # Replace with notice cell (only once)
            if not notice_inserted:
                notice_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": PREPROCESSING_NOTICE_MD.splitlines(keepends=True)
                }
                new_cells.append(notice_cell)
                notice_inserted = True
            # Skip / drop the old cell
            continue

        new_cells.append(cell)

    nb["cells"] = new_cells
    print(f"\n  {nb_path.name}")
    print(f"    Replaced old manifest cells with notice cell: OK")
    save_nb(nb, nb_path)


# ─────────────────────────────────────────────────────────────
# STEP 5 — Team505_EDA.ipynb
# ─────────────────────────────────────────────────────────────

NEW_EDA_LOADING = """\
df_train = pd.read_csv(DATA_SPLITS / 'train.csv')
df_val   = pd.read_csv(DATA_SPLITS / 'val.csv')
df_test  = pd.read_csv(DATA_SPLITS / 'test.csv')
df_all   = pd.concat([df_train, df_val, df_test], ignore_index=True)
print(f'Total dataset: {len(df_all):,} images')
print(f'Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}')
"""

OLD_SPLIT_KEYWORDS_EDA = [
    "train_clean_balanced.csv",
    "val_clean.csv",
    "test_clean.csv",
    "train_d121",
    "pneumonia_1_train",
    "pneumonia_2_train",
]


def patch_eda_notebook(nb_path):
    nb = load_nb(nb_path)
    cells = nb["cells"]
    changed = False

    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        src = cell_source(cell)
        if any(kw in src for kw in OLD_SPLIT_KEYWORDS_EDA):
            set_cell_source(cell, NEW_EDA_LOADING)
            cell["outputs"] = []
            cell["execution_count"] = None
            changed = True

    print(f"\n  {nb_path.name}")
    print(f"    Updated split loading cell: {'OK' if changed else '— (no old refs found)'}")
    save_nb(nb, nb_path)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

MODEL_NOTEBOOKS = [
    "Ammar_Ahmed_DenseNet121.ipynb",
    "Ammar_Ahmed_EfficientNetB3.ipynb",
    "Ammar_Ahmed_ResNet50.ipynb",
    "Hosam_Nabil_DenseNet201.ipynb",
    "Hosam_Nabil_VGG16.ipynb",
    "Hosam_Nabil_MobileNetV2.ipynb",
    "Mohamed_Eslam_Xception.ipynb",
    "Mohamed_Eslam_InceptionV3.ipynb",
    "Mohamed_Eslam_ResNet101.ipynb",
    "Abdelrahman_Mostafa_ViTB16.ipynb",
    "Abdelrahman_Mostafa_SwinT.ipynb",
    "Abdelrahman_Mostafa_DeiTS.ipynb",
]

print("=" * 60)
print("NOTEBOOK PATCH — Data Pipeline Rebuild")
print("=" * 60)

print("\n[STEP 3] Patching 12 model notebooks ...")
for nb_name in MODEL_NOTEBOOKS:
    nb_path = NB_DIR / nb_name
    if not nb_path.exists():
        print(f"  [SKIP] {nb_name} not found")
        continue
    patch_model_notebook(nb_path)

print("\n[STEP 4] Patching Team505_Preprocessing.ipynb ...")
patch_preprocessing_notebook(NB_DIR / "Team505_Preprocessing.ipynb")

print("\n[STEP 5] Patching Team505_EDA.ipynb ...")
patch_eda_notebook(NB_DIR / "Team505_EDA.ipynb")

print("\n" + "=" * 60)
print("ALL DONE.  Run scripts/rebuild_clean_splits.py next.")
print("=" * 60)
