"""
FeatureExtractor — extracts bottleneck feature embeddings from trained
CNN/Transformer models by stripping the classification head.

Architecture Registry
---------------------
ARCHITECTURES dict defines per-model:
    - builder: callable(num_classes=12) -> nn.Module
    - feature_dim: int (bottleneck dimension)
    - classifier_attr: str (model attribute name for the classifier head)

Add new architectures by registering with register_architecture().
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
ARCHITECTURES = {}


def register_architecture(name, builder, feature_dim, classifier_attr):
    ARCHITECTURES[name] = {
        "builder": builder,
        "feature_dim": feature_dim,
        "classifier_attr": classifier_attr,
    }


def list_available_models():
    return sorted(ARCHITECTURES.keys())


# ── Model Builders ──────────────────────────────────────────────────────────

def _build_densenet121(num_classes=12):
    import torchvision.models as models
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier.in_features, num_classes),
    )
    return model


def _build_densenet201(num_classes=12):
    import torchvision.models as models
    model = models.densenet201(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier.in_features, num_classes),
    )
    return model


def _build_efficientnet_b3(num_classes=12):
    import torchvision.models as models
    model = models.efficientnet_b3(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    return model


def _build_resnet50(num_classes=12):
    import torchvision.models as models
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def _build_resnet101(num_classes=12):
    import torchvision.models as models
    model = models.resnet101(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def _build_vgg16(num_classes=12):
    import torchvision.models as models
    model = models.vgg16(weights=None)
    orig_in = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(orig_in, num_classes),
    )
    return model


def _build_mobilenet_v2(num_classes=12):
    import torchvision.models as models
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    return model


def _build_xception(num_classes=12):
    import timm
    model = timm.create_model("legacy_xception", pretrained=False,
                              num_classes=num_classes)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def _build_inception_v3(num_classes=12):
    import torchvision.models as models
    model = models.inception_v3(weights=None, aux_logits=True)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def _build_vit_b16(num_classes=12):
    import torchvision.models as models
    model = models.vit_b_16(weights=None)
    model.heads = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(768, num_classes),
    )
    return model


def _build_swin_t(num_classes=12):
    import timm
    model = timm.create_model("swin_tiny_patch4_window7_224",
                              pretrained=False, num_classes=num_classes)
    model.head = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(model.head.in_features, num_classes),
    )
    return model


def _build_deit_s(num_classes=12):
    import timm
    model = timm.create_model("deit_small_patch16_224",
                              pretrained=False, num_classes=num_classes)
    model.head = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(model.head.in_features, num_classes),
    )
    return model


# ── Register all 12 architectures ──

register_architecture("DenseNet121",    _build_densenet121,    1024, "classifier")
register_architecture("DenseNet201",    _build_densenet201,    1920, "classifier")
register_architecture("EfficientNetB3", _build_efficientnet_b3, 1536, "classifier")
register_architecture("ResNet50",       _build_resnet50,       2048, "fc")
register_architecture("ResNet101",      _build_resnet101,      2048, "fc")
register_architecture("VGG16",          _build_vgg16,          25088, "classifier")
register_architecture("MobileNetV2",    _build_mobilenet_v2,   1280, "classifier")
register_architecture("Xception",       _build_xception,       2048, "fc")
register_architecture("InceptionV3",    _build_inception_v3,   2048, "fc")
register_architecture("ViTB16",         _build_vit_b16,        768, "heads")
register_architecture("SwinT",          _build_swin_t,         768, "head")
register_architecture("DeiTS",          _build_deit_s,         384, "head")


def build_model(model_name, num_classes=12):
    cfg = ARCHITECTURES.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown architecture: {model_name}. "
                         f"Available: {list_available_models()}")
    return cfg["builder"](num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """
    Extracts bottleneck feature vectors from a trained model.

    The classifier head is temporarily replaced with Identity so the
    model's native forward() returns bottleneck features directly.

    Parameters
    ----------
    model_name : str
        Name of the architecture (must be in ARCHITECTURES).
    device : torch.device
    batch_size : int
    num_workers : int
    ckpt_path : str or Path, optional
        Path to checkpoint. If None, model must be passed to replace_model().
    num_classes : int
        Number of classes the model was trained with.
    """

    def __init__(self, model_name, device, batch_size=64, num_workers=0,
                 ckpt_path=None, num_classes=12):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        cfg = ARCHITECTURES.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown architecture: {model_name}. "
                             f"Available: {list_available_models()}")
        self.feature_dim = cfg["feature_dim"]
        self.classifier_attr = cfg["classifier_attr"]

        if ckpt_path is not None:
            self._load_model(ckpt_path, num_classes)

    def _load_model(self, ckpt_path, num_classes=12):
        """Build and load model from checkpoint, strip classifier head."""
        cfg = ARCHITECTURES[self.model_name]
        model = cfg["builder"](num_classes=num_classes)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval().to(self.device)
        setattr(model, self.classifier_attr, nn.Identity())
        self.model = model

    def replace_model(self, model):
        """
        Replace with an externally-loaded model (strips its classifier).
        """
        model.eval()
        setattr(model, self.classifier_attr, nn.Identity())
        self.model = model.to(self.device)

    @torch.no_grad()
    def extract(self, dataset, desc="Extracting features"):
        """
        Run the dataset through the backbone and return bottleneck features.

        Returns
        -------
        features : np.ndarray of shape (N, feature_dim)
        labels : np.ndarray of shape (N,)
        """
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        all_features = []
        all_labels = []

        for images, labels, _ in tqdm(loader, desc=desc):
            images = images.to(self.device, non_blocking=True)
            out = self.model(images)
            all_features.append(out.cpu().numpy())
            all_labels.append(labels.numpy())

        return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)

    @staticmethod
    def save(output_dir, X_train, y_train, X_val, y_val, X_test, y_test):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs = [
            ("X_train.npy", X_train),
            ("y_train.npy", y_train),
            ("X_val.npy", X_val),
            ("y_val.npy", y_val),
            ("X_test.npy", X_test),
            ("y_test.npy", y_test),
        ]
        for name, arr in pairs:
            path = output_dir / name
            np.save(str(path), arr)
            print(f"  [SAVED] {path}  shape={arr.shape}")

    @staticmethod
    def load(input_dir):
        input_dir = Path(input_dir)
        keys = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
        data = {}
        for k in keys:
            path = input_dir / f"{k}.npy"
            data[k] = np.load(str(path))
        return data
