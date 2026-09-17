#!/usr/bin/env python
"""
fix_schedule_block_position.py — move the [SCHEDULE-FIX] constants below the
unfreeze definitions they depend on.

patch_notebooks_schedule_fix.py anchored the block on LOCKED_TEST_EVAL, which
sits ABOVE the unfreeze constants. The result referenced
PARTIAL_UNFREEZE_EPOCH before assignment (NameError at run time), and the
later literal `WARMUP_EPOCHS = 2` would have overwritten the derived value
anyway.

This relocates the block to just after FULL_UNFREEZE_EPOCH and drops the now
redundant literal WARMUP_EPOCHS assignment. Cell outputs are left untouched,
so a notebook that has already been executed keeps its results.
"""

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

START = "# [SCHEDULE-FIX]"
END_RE = re.compile(r"^NUM_EPOCHS = NUM_EPOCHS \+ \(PHASE1_EPOCHS \+ PHASE2_EPOCHS\)\s*$")
FULL_RE = re.compile(r"^FULL_UNFREEZE_EPOCH\s*=\s*\d+\s*$")
WARM_RE = re.compile(r"^WARMUP_EPOCHS\s*=\s*\d+\s*$")


def fix(path, dry_run=False):
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    hp = next((c for c in cells if START in "".join(c["source"])), None)
    if hp is None:
        return "no SCHEDULE-FIX block"

    lines = "".join(hp["source"]).splitlines()
    try:
        s = next(i for i, l in enumerate(lines) if l.startswith(START))
        e = next(i for i, l in enumerate(lines) if END_RE.match(l))
    except StopIteration:
        return "FAILED: block boundaries not found"
    if s > e:
        return "FAILED: malformed block"

    block = lines[s:e + 1]
    rest = lines[:s] + lines[e + 1:]

    # already correctly placed?
    try:
        full_idx_before = next(i for i, l in enumerate(lines) if FULL_RE.match(l))
    except StopIteration:
        return "FAILED: FULL_UNFREEZE_EPOCH not found"
    if full_idx_before < s:
        return "already correctly placed"

    # drop the redundant literal WARMUP_EPOCHS and any blank run left behind
    rest = [l for l in rest if not WARM_RE.match(l)]
    while s < len(rest) and rest[s].strip() == "" and s > 0 and rest[s - 1].strip() == "":
        rest.pop(s)

    full_idx = next(i for i, l in enumerate(rest) if FULL_RE.match(l))
    new = rest[:full_idx + 1] + [""] + block + rest[full_idx + 1:]

    hp["source"] = [l + "\n" for l in new]
    if not dry_run:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return "relocated below FULL_UNFREEZE_EPOCH"


def main():
    dry = "--dry-run" in sys.argv
    bad = 0
    for n in NOTEBOOKS:
        p = NB_DIR / f"{n}.ipynb"
        r = fix(p, dry) if p.exists() else "FAILED: missing"
        if r.startswith("FAILED"):
            bad += 1
        print(f"  {n:<16} {r}")
    print(f"\n{'dry run - nothing written' if dry else 'done'}; failures={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
