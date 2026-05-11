"""
SHAP -- SHapley Additive exPlanations for image classification.

Uses PartitionExplainer with an Image masker, which is the recommended
SHAP approach for image models. This is model-agnostic (only needs a
predict function) and avoids the dimension-mismatch issues that
GradientExplainer encounters with DenseNet shared storage, timm Xception,
and ViT token representations.

Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model
           Predictions", NeurIPS 2017.

Note on method change (Phase 2):
    The original implementation used shap.GradientExplainer, which failed
    on all 4 architectures due to internal tensor dimension mismatches.
    PartitionExplainer is the officially recommended SHAP method for images
    (see https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/).
    It produces valid Shapley-value attributions via hierarchical masking
    and is compatible with any PyTorch model.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class SHAPExplainer:
    """
    SHAP PartitionExplainer for chest X-ray binary classification.

    Uses shap.maskers.Image for hierarchical superpixel partitioning
    and a model-agnostic predict wrapper.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model in eval mode.
    device : torch.device
    max_evals : int
        Maximum number of model evaluations per explanation.
        Higher = more accurate but slower. 200-500 is practical.
    """

    def __init__(self, model, device, max_evals=300):
        self.model = model
        self.model.eval()
        self.device = device
        self.max_evals = max_evals

        # Image masker: replaces masked regions with blurred version
        # Input shape for masker: (224, 224, 3)
        self._masker = shap.maskers.Image("blur(64,64)", (224, 224, 3))

        # Build the explainer with our predict wrapper
        self._explainer = shap.PartitionExplainer(
            self._predict_fn, self._masker
        )

    def _predict_fn(self, images_np):
        """
        Predict function for SHAP.

        Parameters
        ----------
        images_np : np.ndarray
            Batch of images (N, H, W, 3), float in [0, 1].

        Returns
        -------
        probs : np.ndarray
            Shape (N, 2) with [P(neg), P(pos)] per image.
        """
        batch = []
        for img in images_np:
            img_norm = (img - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
            batch.append(tensor)

        batch_tensor = torch.stack(batch).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tensor).squeeze(-1)
            pos_probs = torch.sigmoid(logits).cpu().numpy()

        neg_probs = 1.0 - pos_probs
        return np.column_stack([neg_probs, pos_probs])

    def explain(self, image_np):
        """
        Compute SHAP values for a single image.

        Parameters
        ----------
        image_np : np.ndarray
            Image (H, W, 3) in [0, 1] float, RGB, resized to 224x224.

        Returns
        -------
        shap_values : np.ndarray
            SHAP values array (H, W, C) for the positive (pneumonia) class.
        """
        # PartitionExplainer expects (N, H, W, C)
        input_batch = image_np[np.newaxis, ...]

        explanation = self._explainer(
            input_batch,
            max_evals=self.max_evals,
        )

        # explanation.values shape: (1, H, W, C, num_classes)
        # We want class 1 (pneumonia)
        sv = explanation.values[0]  # (H, W, C, num_classes)
        if sv.ndim == 4:
            sv = sv[:, :, :, 1]  # positive class -> (H, W, C)
        elif sv.ndim == 3:
            pass  # already (H, W, C)

        return sv


def save_shap(image_np, shap_values, save_path, title="SHAP"):
    """
    Save SHAP visualization.

    Parameters
    ----------
    image_np : np.ndarray
        Original image (H, W, 3), float [0, 1].
    shap_values : np.ndarray
        SHAP values (H, W, C) or (H, W).
    save_path : str or Path
    title : str
    """
    # Aggregate across channels for a single attribution map
    if shap_values.ndim == 3:
        shap_agg = np.abs(shap_values).sum(axis=-1)
    else:
        shap_agg = np.abs(shap_values)

    # Normalize
    if shap_agg.max() > shap_agg.min():
        shap_norm = (shap_agg - shap_agg.min()) / (shap_agg.max() - shap_agg.min())
    else:
        shap_norm = np.zeros_like(shap_agg)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(shap_norm, cmap="hot")
    axes[1].set_title("|SHAP| Attribution")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(image_np)
    axes[2].imshow(shap_norm, cmap="hot", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
