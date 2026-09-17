"""
eda.py — analysis routines for the advanced EDA report.

Each function returns data (DataFrame / dict) rather than printing, so the
same code backs both the notebook and scripts/run_eda_report.py.

The analyses here were chosen because they answer questions this project
actually depends on:

  * Is the patient-wise split real? The paper's central integrity claim.
  * Why can no model separate Edema/Consolidation/Infiltration from
    Pneumonia? The multi-label structure gives a data-level answer.
  * Are train/val/test drawn from the same distribution?
  * Is view position a shortcut? AP films come from sicker, bedridden
    patients, so a model can learn "portable film" instead of "pneumonia".
  * How much of the metadata actually covers the splits?
"""

import ntpath

import numpy as np
import pandas as pd
from scipy import stats

FINDING_SEP = "|"


# ════════════════════════════════════════════════════════════════════════════
# INTEGRITY
# ════════════════════════════════════════════════════════════════════════════

def split_integrity(splits, id_col="patient_id", path_col="image_path"):
    """Patient- and image-level overlap between every pair of splits.

    `splits` is a dict name -> DataFrame. A patient-wise split must show zero
    shared patients; a shared patient means the same anatomy appears on both
    sides of the evaluation boundary.
    """
    names = list(splits)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = set(splits[a][id_col]), set(splits[b][id_col])
            ia, ib = set(splits[a][path_col]), set(splits[b][path_col])
            rows.append({
                "pair": f"{a} n {b}",
                "shared_patients": len(pa & pb),
                "shared_images": len(ia & ib),
                "verdict": "LEAK" if (pa & pb) or (ia & ib) else "clean",
            })
    return pd.DataFrame(rows)


def split_sizes(splits, label_col="label", id_col="patient_id"):
    rows = []
    for name, df in splits.items():
        pos = int(df[label_col].sum())
        rows.append({
            "split": name,
            "images": len(df),
            "patients": df[id_col].nunique(),
            "images_per_patient": round(len(df) / max(df[id_col].nunique(), 1), 2),
            "positives": pos,
            "negatives": len(df) - pos,
            "pos_rate_%": round(100 * pos / len(df), 2),
            "ratio": f"1:{(len(df) - pos) / max(pos, 1):.1f}",
        })
    return pd.DataFrame(rows)


def registry_coverage(splits, registry, name_col="image_name"):
    """How much of each split the metadata registry actually covers.

    Anything computed from registry columns (demographics, view position,
    finding_labels) describes only this fraction of the split, not all of it.
    """
    have = set(registry[name_col])
    labelled = set(registry.loc[registry["finding_labels"].notna(), name_col])
    rows = []
    for name, df in splits.items():
        n = df[name_col] if name_col in df else df["image_path"].apply(ntpath.basename)
        rows.append({
            "split": name,
            "images": len(df),
            "in_registry_%": round(100 * n.isin(have).mean(), 1),
            "with_findings_%": round(100 * n.isin(labelled).mean(), 1),
        })
    return pd.DataFrame(rows)


def attach_metadata(df, registry, cols=("finding_labels", "view_position",
                                        "patient_age", "patient_gender")):
    """Left-join registry metadata onto a split by image basename.

    Columns the split already carries are NOT re-joined: the splits hold
    view_position for every row, whereas the registry covers under 60% of
    them, so the split's own column is both complete and authoritative.
    """
    d = df.copy()
    if "image_name" not in d.columns:
        d["image_name"] = d["image_path"].apply(ntpath.basename)
    keep = ["image_name"] + [c for c in cols
                             if c in registry.columns and c not in d.columns]
    return d.merge(registry[keep], on="image_name", how="left")


# ════════════════════════════════════════════════════════════════════════════
# MULTI-LABEL STRUCTURE
# ════════════════════════════════════════════════════════════════════════════

def cooccurrence_with(registry, target="Pneumonia", col="finding_labels"):
    """What else is on the films labelled `target`.

    This is the data-level explanation for why the hard negatives are hard:
    the findings a model must separate from pneumonia are frequently printed
    on the very same image.
    """
    d = registry[registry[col].notna()]
    hit = d[d[col].str.contains(target, na=False)]
    counts = {}
    for s in hit[col]:
        for f in s.split(FINDING_SEP):
            if f and f != target:
                counts[f] = counts.get(f, 0) + 1
    out = (pd.DataFrame({"co_finding": list(counts), "images": list(counts.values())})
           .sort_values("images", ascending=False).reset_index(drop=True))
    out[f"%_of_{target.lower()}"] = (100 * out["images"] / max(len(hit), 1)).round(1)
    alone = int((hit[col] == target).sum())
    meta = {
        "target": target,
        "images_with_target": int(len(hit)),
        "target_alone": alone,
        "target_alone_%": round(100 * alone / max(len(hit), 1), 1),
        "with_cofinding_%": round(100 * (len(hit) - alone) / max(len(hit), 1), 1),
    }
    return out, meta


def findings_per_image(joined, class_col="class_name", col="finding_labels"):
    d = joined.copy()
    d["n_findings"] = d[col].fillna("").apply(
        lambda s: len([x for x in s.split(FINDING_SEP) if x]))
    g = (d[d[col].notna()].groupby(class_col)["n_findings"]
         .agg(["mean", "max", "count"]).sort_values("mean", ascending=False))
    return g.round(2).reset_index()


def negatives_carrying_finding(joined, findings, label_col="label",
                               col="finding_labels"):
    """How many NEGATIVE images carry each pneumonia-adjacent finding."""
    neg = joined[(joined[label_col] == 0) & (joined[col].notna())]
    rows = []
    for f in findings:
        n = int(neg[col].str.contains(f, na=False).sum())
        rows.append({"finding": f, "negatives_with_it": n,
                     "%_of_labelled_negatives": round(100 * n / max(len(neg), 1), 1)})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION SHIFT AND CONFOUNDS
# ════════════════════════════════════════════════════════════════════════════

def cramers_v(contingency):
    chi2 = stats.chi2_contingency(contingency)[0]
    n = contingency.values.sum()
    r, k = contingency.shape
    return float(np.sqrt((chi2 / n) / max(min(r - 1, k - 1), 1)))


def association_test(df, factor, label_col="label"):
    """Chi-square test of independence between a categorical factor and label.

    Same test the lecture describes for filter-based feature selection, used
    here to ask whether a factor is a usable shortcut. Cramer's V gives the
    effect size, because chi-square alone grows with n.
    """
    d = df[[factor, label_col]].dropna()
    if d.empty or d[factor].nunique() < 2:
        return None
    ct = pd.crosstab(d[factor], d[label_col])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    rate = (d.groupby(factor)[label_col].mean() * 100).round(2)
    return {
        "factor": factor, "n": int(len(d)), "chi2": round(float(chi2), 2),
        "p_value": float(p), "dof": int(dof), "cramers_v": round(cramers_v(ct), 4),
        "pos_rate_by_level_%": rate.to_dict(),
        "contingency": ct,
    }


def numeric_group_test(df, numeric, label_col="label"):
    """Mann-Whitney U on a numeric column split by label (age, for example)."""
    d = df[[numeric, label_col]].dropna()
    a = d.loc[d[label_col] == 1, numeric]
    b = d.loc[d[label_col] == 0, numeric]
    if len(a) < 2 or len(b) < 2:
        return None
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {
        "variable": numeric, "n_pos": int(len(a)), "n_neg": int(len(b)),
        "median_pos": float(a.median()), "median_neg": float(b.median()),
        "mean_pos": round(float(a.mean()), 2), "mean_neg": round(float(b.mean()), 2),
        "u_stat": float(u), "p_value": float(p),
        "cohens_d": round(float((a.mean() - b.mean()) / pooled), 4) if pooled else 0.0,
    }


def class_share_across_splits(splits, class_col="class_name"):
    """Per-class share of each split, plus a chi-square for drift between them."""
    frames = {}
    for name, df in splits.items():
        frames[name] = df[class_col].value_counts(normalize=True).mul(100).round(2)
    share = pd.DataFrame(frames).fillna(0)
    counts = pd.DataFrame({n: d[class_col].value_counts() for n, d in splits.items()}).fillna(0)
    chi2, p, dof, _ = stats.chi2_contingency(counts.values)
    return share, {"chi2": round(float(chi2), 2), "p_value": float(p), "dof": int(dof),
                   "cramers_v": round(cramers_v(counts), 4)}


# ════════════════════════════════════════════════════════════════════════════
# IMAGE-LEVEL QUALITY
# ════════════════════════════════════════════════════════════════════════════

def image_quality_sample(paths, n=200, seed=0, threshold=10):
    """Per-image geometry and intensity stats on a random sample.

    border_frac is the share of the frame outside the bounding box of
    non-black content -- large values mean the lung field occupies only part
    of the image, which wastes resolution after resizing.
    """
    from PIL import Image
    rng = np.random.default_rng(seed)
    paths = list(paths)
    pick = rng.choice(len(paths), size=min(n, len(paths)), replace=False)
    rows = []
    for i in pick:
        p = paths[i]
        try:
            g = np.asarray(Image.open(p).convert("L"))
        except Exception:
            continue
        mask = g > threshold
        if not mask.any():
            continue
        r = np.where(mask.any(1))[0]
        c = np.where(mask.any(0))[0]
        h = r[-1] - r[0] + 1
        w = c[-1] - c[0] + 1
        rows.append({
            "width": g.shape[1], "height": g.shape[0],
            "mean_intensity": float(g.mean()), "std_intensity": float(g.std()),
            "p01": float(np.percentile(g, 1)), "p99": float(np.percentile(g, 99)),
            "border_frac": float(1 - (h * w) / g.size),
        })
    return pd.DataFrame(rows)


def duplicate_report(registry, hash_col="image_hash", name_col="image_name"):
    if hash_col not in registry.columns:
        return pd.DataFrame()
    g = registry.groupby(hash_col)[name_col].agg(["count", lambda s: list(s)[:4]])
    g.columns = ["n_images", "examples"]
    return g[g.n_images > 1].sort_values("n_images", ascending=False).reset_index()


# ════════════════════════════════════════════════════════════════════════════
# NEGATIVE DIFFICULTY  (needs model scores; optional)
# ════════════════════════════════════════════════════════════════════════════

def negative_difficulty(df, probs, class_col="class_name", label_col="label"):
    """Rank negative classes by the positive-class score a model assigns them.

    Classes scoring close to the true positives are the ones the model cannot
    separate; they are what the training curriculum defers.
    """
    from sklearn.metrics import roc_auc_score
    d = df.copy()
    d["_p"] = np.asarray(probs, dtype=float)
    pos = d[d[label_col] == 1]
    neg = d[d[label_col] == 0]
    tbl = (neg.groupby(class_col)["_p"].agg(["mean", "median", "count"])
           .sort_values("mean", ascending=False).round(4).reset_index())
    tbl["gap_to_positive_mean"] = (pos["_p"].mean() - tbl["mean"]).round(4)

    def auc_against(classes):
        sub = pd.concat([pos, neg[neg[class_col].isin(classes)]])
        if sub[label_col].nunique() < 2:
            return float("nan")
        return round(float(roc_auc_score(sub[label_col], sub["_p"])), 4)

    order = tbl[class_col].tolist()
    summary = {
        "positive_mean_score": round(float(pos["_p"].mean()), 4),
        "auc_vs_all_negatives": auc_against(order),
        "auc_vs_hardest_3": auc_against(order[:3]),
        "auc_vs_rest": auc_against(order[3:]),
        "hardest_3": order[:3],
    }
    return tbl, summary
