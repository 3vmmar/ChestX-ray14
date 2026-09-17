"""
extractor.py — pull bottleneck (penultimate) embeddings out of a trained CNN.

Feature selection in the lecture operates on a tabular feature matrix. A CNN
does not hand you one directly, so we strip the classification head and take
the pooled embedding that feeds it: DenseNet121 -> 1024-d, ResNet -> 2048-d,
and so on (see src/modelzoo.ARCHITECTURES["<name>"]["feature_dim"]).

Because the head is removed, a checkpoint trained with a different number of
output classes can still be used as a feature extractor -- only the discarded
head depended on the class count.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import modelzoo  # noqa: E402

# classifier attribute to replace with Identity, per architecture
_HEAD_ATTR = {
    "DenseNet121": "classifier",
    "DenseNet201": "classifier",
    "EfficientNetB3": "classifier",
    "MobileNetV2": "classifier",
    "ResNet50": "fc",
    "ResNet101": "fc",
    "InceptionV3": "fc",
    "Xception": "fc",
    "VGG16": None,      # head is a Sequential; handled below
    "ViTB16": None,     # heads.head
    "SwinT": None,      # timm reset_classifier
    "DeiTS": None,
}


def strip_head(model, name):
    """Replace the classification head with Identity so forward() returns features."""
    attr = _HEAD_ATTR.get(name)
    if attr is not None:
        setattr(model, attr, nn.Identity())
        return model
    if name == "VGG16":
        model.classifier[6] = nn.Identity()
    elif name == "ViTB16":
        model.heads.head = nn.Identity()
    else:  # timm models
        if hasattr(model, "reset_classifier"):
            model.reset_classifier(0)
        else:
            raise ValueError(f"don't know how to strip the head of {name}")
    return model


class _ImageDS(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), 0


@torch.no_grad()
def extract_features(model_name, df, transform, device, batch_size=32, workers=0,
                     checkpoint=None, verbose=True):
    """Return an (n_samples, feature_dim) float32 array of bottleneck features."""
    n_out = 1
    if checkpoint is not None and Path(checkpoint).exists():
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
        for k in reversed(list(sd.keys())):
            if k.endswith("weight") and getattr(sd[k], "ndim", 0) == 2:
                n_out = int(sd[k].shape[0])
                break

    model = modelzoo.build(model_name, num_classes=n_out)
    if checkpoint is not None and Path(checkpoint).exists():
        info = modelzoo.load_checkpoint(model, checkpoint, allow_head_mismatch=True)
        if verbose:
            print(f"    loaded checkpoint (epoch={info['epoch']}, {n_out} output units)")
            if info["skipped_shape_mismatch"]:
                print(f"    skipped {len(info['skipped_shape_mismatch'])} head tensor(s) - "
                      f"backbone only, which is what feature extraction needs")
    elif verbose:
        print("    WARNING: no checkpoint - using randomly initialised weights")

    model = strip_head(model, model_name).to(device).eval()

    loader = DataLoader(_ImageDS(df["image_path"].tolist(), transform),
                        batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=(device.type == "cuda"))
    feats = []
    for i, (x, _) in enumerate(loader):
        out = model(x.to(device, non_blocking=True))
        if isinstance(out, (tuple, list)):
            out = out[0]
        feats.append(out.flatten(1).float().cpu().numpy())
        if verbose and (i + 1) % 20 == 0:
            print(f"    batch {i + 1}/{len(loader)}")
    X = np.concatenate(feats).astype(np.float32)
    if verbose:
        print(f"    features: {X.shape}")
    return X
