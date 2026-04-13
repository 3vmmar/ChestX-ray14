"""
Integrated Gradients — attribution by integrating gradients along a path
from a baseline to the input.

Uses Captum (PyTorch attribution library).
Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks",
           ICML 2017.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients as CaptumIG
from pathlib import Path


class IGExplainer:
    """
    Integrated Gradients explainer for binary classification.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model in eval mode. Must output raw logits (single neuron).
    device : torch.device
    n_steps : int
        Number of interpolation steps (default 50).
    """

    def __init__(self, model, device, n_steps=50):
        self.model = model
        self.model.eval()
        self.device = device
        self.n_steps = n_steps
        self._ig = CaptumIG(model)

    def explain(self, input_tensor, baseline=None):
        """
        Compute Integrated Gradients attribution.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input image (1, C, H, W).
        baseline : torch.Tensor or None
            Baseline image (1, C, H, W). None uses a zero (black) image.

        Returns
        -------
        attributions : np.ndarray
            Attribution map (H, W, C).
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)

        if baseline is None:
            baseline = torch.zeros_like(input_tensor).to(self.device)
        else:
            baseline = baseline.to(self.device)

        # target=0 for single-output binary model (the only logit)
        attr = self._ig.attribute(
            input_tensor,
            baselines=baseline,
            n_steps=self.n_steps,
            target=0,
        )

        # (1, C, H, W) -> (H, W, C)
        attr_np = attr.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        return attr_np


def save_ig(image_np, attributions, save_path, title="Integrated Gradients"):
    """
    Save Integrated Gradients visualization.

    Parameters
    ----------
    image_np : np.ndarray
        Original image (H, W, 3), float [0, 1].
    attributions : np.ndarray
        IG attributions (H, W, C).
    save_path : str or Path
    title : str
    """
    attr_agg = np.abs(attributions).sum(axis=-1)

    if attr_agg.max() > attr_agg.min():
        attr_norm = (attr_agg - attr_agg.min()) / (attr_agg.max() - attr_agg.min())
    else:
        attr_norm = np.zeros_like(attr_agg)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    im = axes[1].imshow(attr_norm, cmap="inferno")
    axes[1].set_title("|IG| Attribution")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(image_np)
    axes[2].imshow(attr_norm, cmap="inferno", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
