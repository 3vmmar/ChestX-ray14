#!/usr/bin/env python
"""
build_image_cache.py — precompute the deterministic prefix of the pipeline.

Runs CLAHE at native resolution and a single resize to --size for every image,
writing a uint8 memmap plus an index. Training then reads a row instead of
decoding a 1024x1024 PNG and recomputing CLAHE every epoch.

Measured justification: the GPU manages 193 img/s but training only sustained
113, so the card idled 42% of the time. Per image, 17.6 ms goes to PNG decode
and ~10 ms to CLAHE and resize, all of it identical on every epoch.

Cost: about 15.4 GB at the default 384, and roughly 10 minutes to build with
6 workers. Resumable -- rerunning fills only the rows still missing.

Usage:
    python scripts/build_image_cache.py
    python scripts/build_image_cache.py --size 384 --workers 6
    python scripts/build_image_cache.py --verify
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import imagecache as IC  # noqa: E402
from src.preprocessing import CLAHETransform  # noqa: E402

ARCHIVE = PROJECT_ROOT / "data" / "archive"


class _PrepDataset(Dataset):
    """Yields (row, CLAHE'd + resized uint8 array) for the worker pool."""

    def __init__(self, rows, paths, size):
        self.rows = rows
        self.paths = paths
        self.size = size
        self.clahe = CLAHETransform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i])
            g = self.clahe(img).convert("L")
            g = g.resize((self.size, self.size), Image.LANCZOS)
            return self.rows[i], torch.from_numpy(np.asarray(g, dtype=np.uint8)), 1
        except Exception:
            return self.rows[i], torch.zeros(self.size, self.size, dtype=torch.uint8), 0


def _identity_collate(batch):
    """Module-level so Windows spawn can pickle it (a lambda cannot be)."""
    return batch


def index_images(archive):
    out = {}
    for d in sorted(archive.glob("images_*")):
        sub = d / "images"
        if sub.exists():
            for p in sub.glob("*.png"):
                out[p.name] = str(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=IC.CACHE_SIZE)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="cache only N images (testing)")
    ap.add_argument("--verify", action="store_true", help="check an existing cache and exit")
    args = ap.parse_args()

    arr_path, idx_path = IC.cache_paths(args.size)
    arr_path.parent.mkdir(parents=True, exist_ok=True)

    if args.verify:
        if not IC.cache_exists(args.size):
            print(f"no cache at {arr_path}")
            return 1
        c = IC.ImageCache(args.size)
        arr = c.arr
        n = len(c)
        blank = sum(1 for r in np.random.default_rng(0).choice(n, min(400, n), replace=False)
                    if not np.asarray(arr[r]).any())
        print(f"  cache: {arr_path.name}  shape={arr.shape}  "
              f"{arr_path.stat().st_size/2**30:.1f} GB")
        print(f"  index entries: {n:,}")
        print(f"  blank rows in a 400-row sample: {blank}")
        return 0

    print("indexing images...")
    idx = index_images(ARCHIVE)
    names = sorted(idx)
    if args.limit:
        names = names[:args.limit]
    n = len(names)
    gb = IC.expected_bytes(n, args.size) / 2 ** 30
    print(f"  {n:,} images -> {args.size}x{args.size} uint8 = {gb:.1f} GB")

    # allocate / open the memmap
    if arr_path.exists() and idx_path.exists():
        existing = pd.read_csv(idx_path)
        if len(existing) == n:
            print("  existing cache found, resuming (only blank rows are filled)")
            arr = np.load(arr_path, mmap_mode="r+")
        else:
            print("  size changed, rebuilding from scratch")
            arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8,
                                            shape=(n, args.size, args.size))
    else:
        arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8,
                                        shape=(n, args.size, args.size))

    pd.DataFrame({"image_name": names, "row": np.arange(n)}).to_csv(idx_path, index=False)

    todo = [i for i in range(n) if not np.asarray(arr[i]).any()]
    print(f"  {len(todo):,} rows to fill ({n - len(todo):,} already present)")
    if not todo:
        print("cache already complete")
        return 0

    ds = _PrepDataset(todo, [idx[names[i]] for i in todo], args.size)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                    shuffle=False, collate_fn=_identity_collate)

    t0 = time.time()
    done = failed = 0
    for batch in dl:
        for row, a, ok in batch:
            arr[int(row)] = a.numpy()
            done += 1
            failed += (ok == 0)
        if done % (args.batch_size * 20) < args.batch_size:
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            eta = (len(todo) - done) / max(rate, 1e-9) / 60
            print(f"    {done:,}/{len(todo):,}  {rate:.0f} img/s  ETA {eta:.1f} min",
                  flush=True)
    arr.flush()
    el = (time.time() - t0) / 60
    print(f"\n  wrote {done:,} rows in {el:.1f} min ({done/max(el*60,1e-9):.0f} img/s)")
    if failed:
        print(f"  WARNING: {failed} images failed to decode and were left blank")
    print(f"  cache: {arr_path}  ({arr_path.stat().st_size/2**30:.1f} GB)")
    print("\nuse it with:  python scripts/train_multilabel.py --use-cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
