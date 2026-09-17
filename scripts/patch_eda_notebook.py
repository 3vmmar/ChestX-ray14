#!/usr/bin/env python
"""
patch_eda_notebook.py — append the advanced analyses to notebooks/EDA.ipynb.

Adds six sections that the original EDA did not cover, all backed by
src/eda.py so the notebook and scripts/run_eda_report.py cannot diverge:

  12  Split integrity audit          tests the patient-wise claim
  13  Multi-label structure          why the hard negatives are hard
  14  Distribution shift             are the splits drawn alike
  15  Confounds and shortcut risk    view position, sex, age
  16  Image-level quality control    border waste, intensity, duplicates
  17  Negative-class difficulty      the curriculum's evidence base

Also repairs DATA_RAW, which still pointed at data/raw after the dataset moved
to data/archive.

Usage:
    python scripts/patch_eda_notebook.py --dry-run
    python scripts/patch_eda_notebook.py
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB = PROJECT_ROOT / "notebooks" / "EDA.ipynb"
MARKER = "# [ADV-EDA]"


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


SECTIONS = [
    md("""
---
# Advanced EDA

The sections below go past describing the data and test the assumptions the
rest of the project rests on. They are backed by `src/eda.py`, which also
powers `scripts/run_eda_report.py`, so the notebook and the generated report
can never disagree.
"""),
    code(f"""
{MARKER} shared analysis routines
import sys as _sys
if str(PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(PROJECT_ROOT))
from src import eda as E
import ntpath

SPLITS = {{'train': df_train, 'val': df_val, 'test': df_test}}
for _n, _d in SPLITS.items():
    if 'image_name' not in _d.columns:
        _d['image_name'] = _d['image_path'].apply(ntpath.basename)

registry = pd.read_csv(PROJECT_ROOT / 'data' / 'metadata' / 'master_registry.csv',
                       low_memory=False)
print(f'registry: {{registry.shape[0]:,}} rows x {{registry.shape[1]}} cols')
"""),

    md("""
## 12 — Split Integrity Audit

The patient-wise split is this project's central methodological claim: no
patient may appear on both sides of an evaluation boundary, or the model is
partly scored on anatomy it trained on. Here it is **tested**, not assumed.
"""),
    code(f"""
{MARKER}
integrity = E.split_integrity(SPLITS)
sizes     = E.split_sizes(SPLITS)
coverage  = E.registry_coverage(SPLITS, registry)

display(integrity); display(sizes); display(coverage)

if (integrity.verdict == 'clean').all():
    print('PASS: no patient and no image is shared between any two splits.')
else:
    print('FAIL: leakage detected - see the table above.')

print('\\nNote: metadata coverage above is the share of each split present in the')
print('registry. Every statistic drawn from registry columns (findings,')
print('demographics) describes only that share, not the whole split.')
"""),

    md("""
## 13 — Multi-Label Structure: Why the Hard Negatives Are Hard

NIH labels are multi-label, but the binary task forces one class per image.
This section asks what *else* is on the films tagged Pneumonia — and gives a
data-level explanation for why no model in this project separates Edema,
Consolidation or Infiltration from pneumonia.
"""),
    code(f"""
{MARKER}
co, co_meta = E.cooccurrence_with(registry, 'Pneumonia')
print(f"Films tagged Pneumonia: {{co_meta['images_with_target']}}")
print(f"  Pneumonia alone      : {{co_meta['target_alone']}} ({{co_meta['target_alone_%']}}%)")
print(f"  with a co-finding    : {{co_meta['with_cofinding_%']}}%")
display(co.head(10))

fig, ax = plt.subplots(figsize=(9, 4.5))
top = co.head(10).iloc[::-1]
ax.barh(top.co_finding, top['%_of_pneumonia'], color='#2E5A88')
ax.set_xlabel('% of Pneumonia-tagged films also carrying this finding')
ax.set_title('What else is on a pneumonia film')
plt.tight_layout(); plt.savefig(EDA_OUTPUT / 'adv_cooccurrence.png', dpi=140); plt.show()

joined = E.attach_metadata(df_train, registry)
display(E.findings_per_image(joined))
display(E.negatives_carrying_finding(
    joined, ['Infiltration', 'Effusion', 'Edema', 'Atelectasis', 'Consolidation']))

print('A film tagged Infiltration|Pneumonia is a POSITIVE; one tagged Infiltration')
print('alone is a NEGATIVE. They can be radiographically identical. That is the')
print('mechanism behind the difficulty ordering the training curriculum uses.')
"""),

    md("""
## 14 — Distribution Shift Across Splits

A patient-wise split is clean but not necessarily *balanced*. If prevalence
differs between validation and test, a decision threshold tuned on one will
sit at the wrong operating point on the other — which matters directly for the
locked test evaluation in the model notebooks.
"""),
    code(f"""
{MARKER}
share, drift = E.class_share_across_splits(SPLITS)
display(share)
print(f"chi2={{drift['chi2']}}  p={{drift['p_value']:.4g}}  Cramer's V={{drift['cramers_v']}}")
print('significant drift' if drift['p_value'] < 0.05 else 'no detectable drift',
      '- but read Cramer\\'s V for the effect size, not the p-value.')

fig, ax = plt.subplots(figsize=(11, 4.5))
share.plot(kind='bar', ax=ax, width=0.8)
ax.set_ylabel('% of split'); ax.set_title('Class share by split')
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig(EDA_OUTPUT / 'adv_class_share_by_split.png', dpi=140); plt.show()

rates = {{n: 100 * d['label'].mean() for n, d in SPLITS.items()}}
print('\\nPneumonia prevalence: ' + '  '.join(f'{{k}}={{v:.2f}}%' for k, v in rates.items()))
print(f"val -> test relative change: {{100*(rates['test']-rates['val'])/rates['val']:+.1f}}%")
print('A threshold frozen on val therefore lands on a different operating point')
print('on test. Expect precision and recall to move even if ranking is unchanged.')
"""),

    md("""
## 15 — Confounds and Shortcut Risk

A variable that predicts the label without being the pathology is something a
model can learn *instead of* the disease. AP films in particular come from
sicker, bedridden patients imaged portably, so "AP view" partly encodes
"unwell" — a shortcut that inflates apparent performance and fails on transfer.

Chi-square gives significance; Cramér's V gives effect size, which is what
actually matters, because chi-square grows with sample size on its own.
"""),
    code(f"""
{MARKER}
for factor in ['view_position', 'patient_gender']:
    r = E.association_test(joined, factor)
    if r is None:
        print(f'{{factor}}: unavailable'); continue
    ct = r.pop('contingency')
    print(f"\\n{{factor}}  n={{r['n']}}  chi2={{r['chi2']}}  p={{r['p_value']:.4g}}  "
          f"Cramer's V={{r['cramers_v']}}")
    print(f"  pneumonia rate by level: {{r['pos_rate_by_level_%']}}")
    display(ct)

age = E.numeric_group_test(joined, 'patient_age')
if age:
    print(f"\\npatient_age  median {{age['median_pos']}} (pos) vs {{age['median_neg']}} (neg)  "
          f"p={{age['p_value']:.4g}}  Cohen's d={{age['cohens_d']}}")

vr = E.association_test(joined, 'view_position')
if vr:
    lv = list(vr['pos_rate_by_level_%'])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(lv, [vr['pos_rate_by_level_%'][k] for k in lv], color='#C44E52')
    ax.set_ylabel('pneumonia rate (%)'); ax.set_title('Pneumonia rate by view position')
    plt.tight_layout(); plt.savefig(EDA_OUTPUT / 'adv_view_position_confound.png', dpi=140); plt.show()
"""),

    md("""
## 16 — Image-Level Quality Control

Everything above treats an image as a row in a table. This section opens the
files: how much of each frame is empty border (wasted resolution once the
image is resized to 288), how brightness varies, and whether any image is
byte-identical to another.
"""),
    code(f"""
{MARKER}
q = E.image_quality_sample(df_train['image_path'], n=250)
display(q[['mean_intensity', 'std_intensity', 'border_frac']].describe().round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(q.border_frac * 100, bins=30, color='#55A868', edgecolor='white')
axes[0].set_xlabel('border fraction (%)'); axes[0].set_title('Wasted frame area')
axes[1].hist(q.mean_intensity, bins=30, color='#8172B2', edgecolor='white')
axes[1].set_xlabel('mean intensity'); axes[1].set_title('Global brightness')
plt.tight_layout(); plt.savefig(EDA_OUTPUT / 'adv_image_quality.png', dpi=140); plt.show()

print(f'border fraction: mean {{100*q.border_frac.mean():.1f}}%  '
      f'max {{100*q.border_frac.max():.1f}}%  '
      f'above 5%: {{100*(q.border_frac>0.05).mean():.0f}}% of images')

dups = E.duplicate_report(registry)
print(f'\\nduplicate image hashes in the registry: {{len(dups)}}')
if len(dups): display(dups.head(10))
"""),

    md("""
## 17 — Negative-Class Difficulty

This is the evidence the training curriculum is built on. Each negative class
is scored by the pneumonia probability a trained model assigns it; classes
scoring close to the true positives are the ones the model cannot separate.

Requires per-image scores from `scripts/dump_predictions.py`; the cell skips
cleanly if none exist.
"""),
    code(f"""
{MARKER}
prob_files = sorted((PROJECT_ROOT / 'outputs' / 'models').glob('*/val_probs*.csv'))
if not prob_files:
    print('No outputs/models/*/val_probs*.csv found.')
    print('Run: python scripts/dump_predictions.py --model <name>')
else:
    pf = prob_files[0]
    p = pd.read_csv(pf)
    merged = df_val[['image_path', 'class_name', 'label']].merge(
        p[['image_path', 'p_tta']], on='image_path', how='inner')
    tbl, s = E.negative_difficulty(merged, merged['p_tta'])
    print(f'scores from {{pf.name}}  (n={{len(merged)}})')
    display(tbl)
    print(f"positives average {{s['positive_mean_score']}}")
    print(f"  AUC vs all negatives : {{s['auc_vs_all_negatives']}}")
    print(f"  AUC vs hardest 3     : {{s['auc_vs_hardest_3']}}  ({{', '.join(s['hardest_3'])}})")
    print(f"  AUC vs the rest      : {{s['auc_vs_rest']}}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = tbl.iloc[::-1]
    ax.barh(t['class_name'], t['mean'], color='#C44E52')
    ax.axvline(s['positive_mean_score'], color='#2E5A88', ls='--',
               label=f"Pneumonia mean = {{s['positive_mean_score']}}")
    ax.set_xlabel('mean predicted pneumonia score')
    ax.set_title('Negative-class confusability'); ax.legend()
    plt.tight_layout(); plt.savefig(EDA_OUTPUT / 'adv_negative_difficulty.png', dpi=140); plt.show()
"""),

    md("""
## 18 — Regenerate the Standalone Report

`scripts/run_eda_report.py` runs every analysis above headless and writes
`outputs/eda/advanced_eda_report.md` plus a JSON summary, so the findings can
be cited without re-running the notebook.

```bash
python scripts/run_eda_report.py
```
"""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    nb = json.loads(NB.read_text(encoding="utf-8"))
    existing = "\n".join("".join(c["source"]) for c in nb["cells"])
    if MARKER in existing:
        print("already patched")
        return 0

    fixed_path = False
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        if "DATA_RAW" in s and "'data' / 'raw'" in s:
            c["source"] = s.replace("'data' / 'raw'", "'data' / 'archive'").splitlines(keepends=True)
            fixed_path = True

    nb["cells"].extend(SECTIONS)

    if not args.dry_run:
        NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"appended {len(SECTIONS)} cells"
          + ("; DATA_RAW -> data/archive" if fixed_path else "")
          + ("  (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
