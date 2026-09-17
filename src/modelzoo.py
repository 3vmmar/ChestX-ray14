"""
modelzoo.py — single source of truth for model construction and preprocessing.

Each entry records how a model was BUILT and at what resolution it was
TRAINED. Getting the resolution right matters: the CNNs were trained at 288
(InceptionV3 at 299) and the transformers at 224, so evaluating everything at
one size would not reproduce the published metrics.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

try:
    import timm
except ImportError:  # only needed for Xception / Swin / DeiT
    timm = None


# ── builders (must match the training notebooks) ────────────────────────────
def _densenet121(n=1):
    m = tvm.densenet121(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(m.classifier.in_features, n))
    return m


def _densenet201(n=1):
    m = tvm.densenet201(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(m.classifier.in_features, n))
    return m


def _efficientnet_b3(n=1):
    m = tvm.efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(m.classifier[1].in_features, n))
    return m


def _resnet50(n=1):
    m = tvm.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(m.fc.in_features, n))
    return m


def _resnet101(n=1):
    m = tvm.resnet101(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(m.fc.in_features, n))
    return m


def _vgg16(n=1):
    m = tvm.vgg16(weights=None)
    m.classifier[6] = nn.Sequential(nn.Dropout(0.6), nn.Linear(m.classifier[6].in_features, n))
    return m


def _mobilenet_v2(n=1):
    m = tvm.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(m.classifier[1].in_features, n))
    return m


def _inception_v3(n=1):
    m = tvm.inception_v3(weights=None, aux_logits=True, init_weights=False)
    m.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(m.fc.in_features, n))
    return m


def _xception(n=1):
    m = timm.create_model("legacy_xception", pretrained=False)
    m.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(m.fc.in_features, n))
    return m


def _vit_b16(n=1):
    m = tvm.vit_b_16(weights=None)
    m.heads.head = nn.Sequential(nn.Dropout(0.6), nn.Linear(m.heads.head.in_features, n))
    return m


def _swin_t(n=1):
    return timm.create_model("swin_tiny_patch4_window7_224", pretrained=False,
                             num_classes=n, drop_rate=0.6)


def _deit_s(n=1):
    return timm.create_model("deit_small_patch16_224", pretrained=False,
                             num_classes=n, drop_rate=0.6)


ARCHITECTURES = {
    "DenseNet121":    {"builder": _densenet121,     "img_size": 288, "family": "cnn",         "feature_dim": 1024},
    "DenseNet201":    {"builder": _densenet201,     "img_size": 288, "family": "cnn",         "feature_dim": 1920},
    "EfficientNetB3": {"builder": _efficientnet_b3, "img_size": 288, "family": "cnn",         "feature_dim": 1536},
    "ResNet50":       {"builder": _resnet50,        "img_size": 288, "family": "cnn",         "feature_dim": 2048},
    "ResNet101":      {"builder": _resnet101,       "img_size": 288, "family": "cnn",         "feature_dim": 2048},
    "VGG16":          {"builder": _vgg16,           "img_size": 288, "family": "cnn",         "feature_dim": 4096},
    "MobileNetV2":    {"builder": _mobilenet_v2,    "img_size": 288, "family": "cnn",         "feature_dim": 1280},
    "InceptionV3":    {"builder": _inception_v3,    "img_size": 299, "family": "cnn",         "feature_dim": 2048},
    "Xception":       {"builder": _xception,        "img_size": 288, "family": "cnn",         "feature_dim": 2048},
    "ViTB16":         {"builder": _vit_b16,         "img_size": 224, "family": "transformer", "feature_dim": 768},
    "SwinT":          {"builder": _swin_t,          "img_size": 224, "family": "transformer", "feature_dim": 768},
    "DeiTS":          {"builder": _deit_s,          "img_size": 224, "family": "transformer", "feature_dim": 384},
}


def list_models():
    return sorted(ARCHITECTURES)


def build(name, num_classes=1):
    if name not in ARCHITECTURES:
        raise KeyError(f"unknown model {name!r}; known: {list_models()}")
    return ARCHITECTURES[name]["builder"](num_classes)


def img_size(name):
    return ARCHITECTURES[name]["img_size"]


def load_checkpoint(model, path, device="cpu", allow_head_mismatch=False):
    """Load a checkpoint saved either raw or under model_state_dict/state_dict.

    allow_head_mismatch=True skips tensors whose shape disagrees with the
    model (typically the classifier head, when a checkpoint was trained with a
    different number of output classes). The backbone still loads, which is
    all that is needed for feature extraction. Skipped keys are reported so
    the caller can never mistake a partial load for a full one.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ck, dict):
        sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    else:
        sd = ck

    skipped = []
    if allow_head_mismatch:
        own = model.state_dict()
        filtered = {}
        for k, v in sd.items():
            if k in own and hasattr(v, "shape") and own[k].shape != v.shape:
                skipped.append((k, tuple(v.shape), tuple(own[k].shape)))
            else:
                filtered[k] = v
        sd = filtered

    missing, unexpected = model.load_state_dict(sd, strict=False)
    return {
        "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
        "val_auc": ck.get("val_auc") if isinstance(ck, dict) else None,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "skipped_shape_mismatch": skipped,
    }
