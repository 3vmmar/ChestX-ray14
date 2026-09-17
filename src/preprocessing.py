"""
preprocessing.py — single source of truth for image transforms.

Previously each notebook defined its own CLAHE + transform stack, and
scripts/dump_predictions.py, run_feature_selection.py and run_xai_demo.py
each defined another. They had already drifted (run_xai_demo evaluates at 224
regardless of the model's training resolution). Everything now builds its
transforms here.

Problems fixed relative to the original stack, each measured on real NIH data
before changing:

1. TRAIN/VAL SCALE MISMATCH
   train was Resize(320) -> RandomCrop(288)  = 90% field of view
   val   was Resize((288,288))               = 100% field of view
   Anatomy therefore appeared 11% larger during training than at inference.
   Both paths now share Resize(TRAIN_RESIZE) and a 288 crop (random for
   train, centre for val/TTA), so the scales agree.

2. BLACK WEDGES FROM ROTATION
   RandomRotation(15) ran AFTER RandomCrop, filling the corners with zeros:
   measured 10.1% of every rotated training image became pure black, which
   normalises to a strong constant signal and is a free shortcut feature.
   Rotation now runs BEFORE the crop on a 336px source. A 288 crop needs a
   source of 288*(cos t + sin t) to stay inside real content: 334px at 10
   degrees, so 336 clears it with no black at all.

3. ANATOMICALLY IMPOSSIBLE AUGMENTATION
   RandomVerticalFlip(p=0.1) put the diaphragm above the apices in 10% of
   training images. No such radiograph exists, and none appears at test time.
   Removed.

4. DEAD TRANSFORM
   RandomGrayscale(p=0.05) followed CLAHE, which already emits three
   identical channels. Verified a no-op on real images. Removed.

5. REDUNDANT RESAMPLE
   CLAHE carried a "prevents MemoryError" downscale to 512. A 1024x1024 uint8
   CLAHE takes 0.93 ms and 1 MB, so there was no memory issue; the guard just
   added a resample (1024 -> 512 -> 320 -> 288). CLAHE now runs at native
   resolution with tileGridSize derived from TILE_PX, which keeps the tile
   size in pixels equal to the old behaviour (~64px) rather than silently
   changing how aggressive the enhancement is.

Kept deliberately: RandomHorizontalFlip. It mirrors left/right anatomy, which
is not strictly label-preserving, but it is standard in the CXR literature and
the TTA stack already relies on it.
"""

import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLIP_LIMIT = 2.0
TILE_PX = 64          # target CLAHE tile size in pixels, at native resolution
ROTATION_DEG = 10     # training rotation, matched to the +-7 deg TTA variants
JITTER_PX = 16        # random-crop translation range (replaces RandomAffine)

# Rotating an SxS image by t leaves an inscribed black-free square of
# S / (cos t + sin t). A RandomCrop can land anywhere, so the safe region must
# be centre-cropped out FIRST; only then is every crop position clean.
_ROT_FACTOR = 1.1585  # cos(10 deg) + sin(10 deg)


class CLAHETransform:
    """Contrast Limited Adaptive Histogram Equalisation at native resolution.

    tileGridSize is computed from the image so tiles stay ~TILE_PX pixels
    regardless of input size. A fixed 8x8 grid means 64px tiles on a 512px
    image but 128px tiles on a 1024px image -- a different operation.
    """

    def __init__(self, clip_limit=CLIP_LIMIT, tile_px=TILE_PX):
        self.clip_limit = clip_limit
        self.tile_px = tile_px

    def __call__(self, pil_img):
        arr = np.array(pil_img.convert("L"))
        n = int(max(1, round(min(arr.shape) / self.tile_px)))
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(n, n))
        return Image.fromarray(clahe.apply(arr)).convert("RGB")

    def __repr__(self):
        return f"CLAHETransform(clip={self.clip_limit}, tile_px={self.tile_px})"


def safe_size_for(img_size):
    """Largest centre square guaranteed black-free, sized to allow crop jitter."""
    return img_size + JITTER_PX


def train_resize_for(img_size):
    """Source size whose rotated inscribed square is at least safe_size_for()."""
    return int(np.ceil(safe_size_for(img_size) * _ROT_FACTOR))


def _norm():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def build_train_transforms(img_size, rotation_deg=ROTATION_DEG, jitter=True):
    """Training stack.

    Order is load-bearing: rotate on the oversized source, centre-crop away
    every rotated corner, and only then take the random crop. That gives
    translation augmentation with no black fill anywhere -- which is why
    RandomAffine is gone, since both its translate and its scale<1 reintroduce
    black borders that the crop can no longer remove.
    """
    ops = [
        CLAHETransform(),
        T.Resize(train_resize_for(img_size)),
        T.RandomRotation(degrees=rotation_deg),
        T.CenterCrop(safe_size_for(img_size)),   # discard rotated corners
        T.RandomCrop(img_size),                  # +-JITTER_PX/2 translation
        T.RandomHorizontalFlip(p=0.5),
    ]
    if jitter:
        # contrast jitter is mild: CLAHE has already normalised local contrast,
        # and re-randomising it aggressively undoes that work
        ops.append(T.ColorJitter(brightness=0.15, contrast=0.10))
    ops += [T.ToTensor(), _norm()]
    return T.Compose(ops)


def build_val_transforms(img_size):
    """Deterministic evaluation stack at the SAME scale as training."""
    return T.Compose([
        CLAHETransform(),
        T.Resize(train_resize_for(img_size)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        _norm(),
    ])


def build_tta_transforms(img_size):
    """The 5 TTA variants, all sharing the evaluation geometry.

    Order matches the reported protocol: plain, hflip, +7, -7, brightness.
    """
    resize = train_resize_for(img_size)
    base = [CLAHETransform(), T.Resize(resize)]
    tail = [T.CenterCrop(img_size), T.ToTensor(), _norm()]
    return [
        T.Compose(base + tail),
        T.Compose(base + [T.RandomHorizontalFlip(p=1.0)] + tail),
        T.Compose(base + [T.RandomRotation(degrees=(7, 7))] + tail),
        T.Compose(base + [T.RandomRotation(degrees=(-7, -7))] + tail),
        T.Compose(base + [T.ColorJitter(brightness=0.15)] + tail),
    ]


TTA_VARIANT_NAMES = ["plain", "hflip", "rot+7", "rot-7", "bright"]
