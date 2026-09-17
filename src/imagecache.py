"""
imagecache.py — memory-mapped cache of CLAHE-processed chest films.

Why this exists (measured, not assumed):

    pure GPU throughput, DenseNet121 @288, batch 24 : 193 img/s
    actual end-to-end throughput during training    : 113 img/s
    peak VRAM                                       : 2.6 GB of 8.0

The GPU sits idle 42% of the time waiting for data. Per image the pipeline
spends 28.1 ms, of which 17.6 ms is decoding a 1024x1024 PNG and ~10 ms is
CLAHE plus the resize -- repeated every single epoch for the same unchanging
result. Caching that work once turns per-image cost into a ~1-2 ms memmap
read, which moves the bottleneck onto the GPU where it belongs.

What is cached: CLAHE at native resolution, then a single resize to
CACHE_SIZE, stored as uint8 grayscale. Everything after that (rotation,
crops, flip, jitter, normalisation) stays random per epoch, so augmentation
is unaffected -- the cache holds only the deterministic prefix of the
pipeline.

CACHE_SIZE must be at least the largest train_resize_for() any model needs
(365 for InceptionV3 at 299px), which is why the default is 384.

Layout:
    data/cache/clahe_<size>.npy        uint8 memmap, shape (N, size, size)
    data/cache/clahe_<size>_index.csv  image_name -> row
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_SIZE = 384


def cache_paths(size=CACHE_SIZE, cache_dir=CACHE_DIR):
    return (Path(cache_dir) / f"clahe_{size}.npy",
            Path(cache_dir) / f"clahe_{size}_index.csv")


def cache_exists(size=CACHE_SIZE, cache_dir=CACHE_DIR):
    a, i = cache_paths(size, cache_dir)
    return a.exists() and i.exists()


def expected_bytes(n_images, size=CACHE_SIZE):
    return n_images * size * size


class ImageCache:
    """Read-only view over the cached array."""

    def __init__(self, size=CACHE_SIZE, cache_dir=CACHE_DIR):
        arr_path, idx_path = cache_paths(size, cache_dir)
        if not (arr_path.exists() and idx_path.exists()):
            raise FileNotFoundError(
                f"no cache at {arr_path}\n"
                f"build it with: python scripts/build_image_cache.py --size {size}")
        idx = pd.read_csv(idx_path)
        self.size = size
        self.row_of = dict(zip(idx["image_name"], idx["row"]))
        self.arr = np.load(arr_path, mmap_mode="r")
        if self.arr.shape[0] != len(idx):
            raise ValueError(f"cache/index mismatch: {self.arr.shape[0]} vs {len(idx)}")

    def __len__(self):
        return self.arr.shape[0]

    def __contains__(self, image_name):
        return image_name in self.row_of

    def get(self, image_name):
        """Cached film as a 3-channel PIL image, or None if absent."""
        r = self.row_of.get(image_name)
        if r is None:
            return None
        return Image.fromarray(np.asarray(self.arr[r])).convert("RGB")


class CachedCXRDataset(Dataset):
    """Dataset that prefers the cache and falls back to the original PNG.

    `transform` must be built with clahe=False: CLAHE is already baked in.
    A film missing from the cache is read from disk and CLAHE'd on the fly,
    so a partial cache degrades in speed rather than correctness.
    """

    def __init__(self, df, transform, cache, label_cols=None,
                 path_col="image_path", name_col="image_name"):
        self.paths = df[path_col].tolist()
        if name_col in df.columns:
            self.names = df[name_col].tolist()
        else:
            import ntpath
            self.names = [ntpath.basename(p) for p in self.paths]
        self.transform = transform
        self.cache = cache
        self.multi = label_cols is not None
        if self.multi:
            self.y = df[label_cols].values.astype(np.float32)
        else:
            self.y = df["label"].astype(np.float32).values
        self._fallbacks = 0

    def __len__(self):
        return len(self.paths)

    def fallback_count(self):
        return self._fallbacks

    def __getitem__(self, i):
        img = self.cache.get(self.names[i]) if self.cache is not None else None
        if img is None:
            from src.preprocessing import CLAHETransform
            self._fallbacks += 1
            img = CLAHETransform()(Image.open(self.paths[i]).convert("RGB"))
        x = self.transform(img)
        y = torch.from_numpy(self.y[i]) if self.multi else torch.tensor(self.y[i])
        return x, y
