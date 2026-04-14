"""
LIME — Local Interpretable Model-agnostic Explanations.

Provides superpixel-based perturbation explanations for chest X-ray models.
Reference: Ribeiro et al., "Why Should I Trust You?: Explaining the Predictions
           of Any Classifier", KDD 2016.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from pathlib import Path


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class LIMEExplainer:
    """
    LIME explainer for binary classification chest X-ray models.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model in eval mode.
    device : torch.device
        Device to run inference on.
    img_size : int
        Expected input image size (default 224).
    """

    def __init__(self, model, device, img_size=224):
        self.model = model
        self.model.eval()
        self.device = device
        self.img_size = img_size
        self.explainer = lime_image.LimeImageExplainer(random_state=42)

    def _batch_predict(self, images_np):
        """
        Batch prediction function for LIME.

        Parameters
        ----------
        images_np : np.ndarray
            Array of images (N, H, W, 3), float64, in [0, 1].

        Returns
        -------
        probs : np.ndarray
            Array of shape (N, 2) with [P(neg), P(pos)] for each image.
        """
        batch = []
        for img in images_np:
            # Normalize
            img_norm = (img - IMAGENET_MEAN) / IMAGENET_STD
            # HWC -> CHW
            tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
            batch.append(tensor)

        batch_tensor = torch.stack(batch).to(self.device)

        with torch.no_grad():
            logits = self.model(batch_tensor).squeeze(-1)
            pos_probs = torch.sigmoid(logits).cpu().numpy()

        # LIME expects (N, num_classes) — binary: [P(neg), P(pos)]
        neg_probs = 1.0 - pos_probs
        return np.column_stack([neg_probs, pos_probs])

    def explain(self, image_np, num_samples=300, num_features=10):
        """
        Generate LIME explanation.

        Parameters
        ----------
        image_np : np.ndarray
            Image (H, W, 3) in [0, 1] float, RGB, already resized to img_size.
        num_samples : int
            Number of perturbed samples for LIME.
        num_features : int
            Number of superpixel features to select.

        Returns
        -------
        explanation : lime_image.ImageExplanation
            LIME explanation object.
        """
        explanation = self.explainer.explain_instance(
            image_np.astype(np.double),
            self._batch_predict,
            top_labels=2,
            hide_color=0,
            num_samples=num_samples,
            num_features=num_features,
            random_seed=42,
        )
        return explanation


def save_lime(image_np, explanation, save_path, title="LIME", label=1):
    """
    Save LIME visualization.

    Parameters
    ----------
    image_np : np.ndarray
        Original image (H, W, 3), float [0, 1].
    explanation : lime_image.ImageExplanation
    save_path : str or Path
    title : str
    label : int
        Class label to explain (1=Pneumonia).
    """
    # Positive-only mask
    temp_pos, mask_pos = explanation.get_image_and_mask(
        label, positive_only=True, num_features=5, hide_rest=False
    )
    # Positive and negative regions
    temp_both, mask_both = explanation.get_image_and_mask(
        label, positive_only=False, num_features=10, hide_rest=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mark_boundaries(temp_pos, mask_pos))
    axes[1].set_title("Positive Regions")
    axes[1].axis("off")

    axes[2].imshow(mark_boundaries(temp_both, mask_both))
    axes[2].set_title("Pos + Neg Regions")
    axes[2].axis("off")

    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
