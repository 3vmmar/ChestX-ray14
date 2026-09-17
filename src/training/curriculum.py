"""
curriculum.py — negative-difficulty curriculum for binary pneumonia training.

Motivation (measured, not assumed)
----------------------------------
Scoring every negative class by the pneumonia probability a trained model
assigns it gives a clear difficulty ordering:

    Edema          0.1521   <- actual Pneumonia scores 0.1528
    Consolidation  0.1350
    Infiltration   0.1162
    Effusion       0.1067
    Atelectasis    0.1049
    Cardiomegaly   0.1017
    No Finding     0.0915
    Nodule         0.0869
    Pneumothorax   0.0720
    Mass           0.0706
    Emphysema      0.0644

Split by negative type, the same model scores AUC 0.7456 against the easy
negatives but 0.5616 against {Edema, Consolidation, Infiltration} -- chance.
Those three are 27% of the training negatives and 3x the positive count, and
they are also where NIH's NLP-derived labels are noisiest, because report
language for pneumonia / consolidation / infiltration overlaps heavily.

Feeding all of that in from epoch 1 means a quarter of the gradient signal
comes from pairs the model cannot separate. The curriculum defers them:
establish a boundary on separable negatives first, then introduce the hard
ones once the representation is stable.

Phases
------
    1  Pneumonia vs No Finding            (clean clinical contrast)
    2  + moderate negatives               (other findings, still separable)
    3  + hard negatives                   (full training set)

Progressive unfreezing runs INSIDE phase 3 -- phases 1 and 2 train the head
on a frozen backbone, which is the LP-FT pattern (Kumar et al., ICLR 2022).

CAVEAT: the ordering above was derived from the 12-class checkpoint, the only
one available when this was written. The ranking reflects radiographic
similarity and should be stable, but re-derive it from a binary model before
citing the numbers. Use rank_negatives_by_difficulty() to do so.
"""

import numpy as np

POSITIVE_CLASS = "Pneumonia"

# measured difficulty tiers (hardest last)
EASY_NEGATIVES = ["No Finding"]
MODERATE_NEGATIVES = [
    "Effusion", "Atelectasis", "Cardiomegaly",
    "Nodule", "Pneumothorax", "Mass", "Emphysema",
]
HARD_NEGATIVES = ["Infiltration", "Consolidation", "Edema"]

PHASES = {
    1: EASY_NEGATIVES,
    2: EASY_NEGATIVES + MODERATE_NEGATIVES,
    3: EASY_NEGATIVES + MODERATE_NEGATIVES + HARD_NEGATIVES,
}


def phase_for_epoch(epoch, phase1_epochs, phase2_epochs):
    """Which data phase a given 1-indexed epoch belongs to."""
    if epoch <= phase1_epochs:
        return 1
    if epoch <= phase1_epochs + phase2_epochs:
        return 2
    return 3


def phase_classes(phase):
    """Class names visible in this phase (positives always included)."""
    if phase not in PHASES:
        raise ValueError(f"phase must be 1, 2 or 3; got {phase}")
    return [POSITIVE_CLASS] + PHASES[phase]


def phase_frame(df_train, phase, class_col="class_name"):
    """Subset of the training frame visible in this phase.

    Falls back to the full frame (with a warning) if the class column is
    absent, so a notebook without class_name still runs rather than crashing.
    """
    if class_col not in df_train.columns:
        print(f"    WARNING: no {class_col!r} column - curriculum disabled, using full set")
        return df_train
    keep = phase_classes(phase)
    sub = df_train[df_train[class_col].isin(keep)].reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(f"phase {phase} selected 0 rows; classes present: "
                         f"{sorted(df_train[class_col].unique())}")
    return sub


def sample_weights(labels, target_pos_frac=0.20):
    """WeightedRandomSampler weights giving each batch ~target_pos_frac positives.

    Recomputed per phase: the positive:negative ratio changes from roughly
    1:1 in phase 1 to 1:11 in phase 3, so weights from one phase are wrong
    for another.
    """
    labels = np.asarray(labels, dtype=np.float32)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.ones(len(labels), dtype=np.float64)
    return np.where(labels == 1,
                    target_pos_frac / n_pos,
                    (1.0 - target_pos_frac) / n_neg).astype(np.float64)


def describe_phase(df_phase, phase, label_col="target_pneumonia"):
    n = len(df_phase)
    pos = int(df_phase[label_col].sum())
    neg = n - pos
    ratio = (neg / pos) if pos else float("inf")
    added = PHASES[phase][len(PHASES[phase - 1]):] if phase > 1 else PHASES[1]
    return (f"DATA PHASE {phase}: {n:,} images | pos={pos} neg={neg} | 1:{ratio:.1f}"
            f" | negatives: {', '.join(PHASES[phase])}"
            + (f" | newly added: {', '.join(added)}" if phase > 1 else ""))


def rank_negatives_by_difficulty(df, probs, class_col="class_name",
                                 label_col="label"):
    """Re-derive the difficulty ordering from a model's own predictions.

    Returns (ordering hardest-first, per-class mean score). Use this to
    regenerate the tier lists from a binary checkpoint.
    """
    d = df.copy()
    d["_p"] = np.asarray(probs, dtype=float)
    neg = d[d[label_col] == 0]
    means = neg.groupby(class_col)["_p"].mean().sort_values(ascending=False)
    return list(means.index), means.to_dict()


def suggest_tiers(means, n_hard=3, n_easy=1):
    """Split a per-class mean-score dict into hard / moderate / easy tiers."""
    order = sorted(means, key=lambda k: means[k], reverse=True)
    return {
        "hard": order[:n_hard],
        "moderate": order[n_hard:len(order) - n_easy],
        "easy": order[len(order) - n_easy:],
    }
