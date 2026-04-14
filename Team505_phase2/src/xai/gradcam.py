"""
Grad-CAM — Gradient-weighted Class Activation Mapping.

Supports: DenseNet121, DenseNet201, Xception (timm), ViT-B/16 (torchvision).
Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
           via Gradient-based Localization", ICCV 2017.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


class GradCAM:
    """
    Hook-based Grad-CAM that works with any model by specifying the target
    convolutional layer.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model in eval mode.
    target_layer : torch.nn.Module
        The layer whose activations/gradients will be captured.
        Typically the last conv/feature-extraction layer.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self._activations = None
        self._gradients = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input image tensor of shape (1, C, H, W).
        class_idx : int or None
            Class index for the gradient. None uses predicted class logit.

        Returns
        -------
        heatmap : np.ndarray
            Normalized heatmap of shape (H, W) in [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            target = output.squeeze()
        else:
            target = output[0, class_idx]

        target.backward(retain_graph=False)

        # Global average pooling of gradients
        gradients = self._gradients
        activations = self._activations

        if gradients.dim() == 4:
            # CNN: (B, C, H, W)
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            cam = F.relu(cam)
            cam = cam.squeeze().cpu().numpy()
        elif gradients.dim() == 3:
            # Transformer: (B, N, D) — patch tokens
            weights = gradients.mean(dim=1, keepdim=True)  # (B, 1, D)
            cam = (weights * activations).sum(dim=-1)  # (B, N)
            cam = cam.squeeze().cpu().numpy()
            # Remove CLS token if present, reshape to spatial grid
            num_patches = cam.shape[0]
            if num_patches == 197:  # 14*14 + 1 CLS
                cam = cam[1:]  # drop CLS
                side = 14
            else:
                side = int(np.sqrt(num_patches))
                # If there's an extra CLS token
                if side * side != num_patches and (num_patches - 1) == side * side:
                    cam = cam[1:]
                    # recalculate
                elif side * side != num_patches:
                    side = int(np.sqrt(num_patches - 1))
                    cam = cam[1:]
            cam = cam.reshape(side, side)
            cam = np.maximum(cam, 0)
        else:
            raise ValueError(f"Unexpected gradient dimensions: {gradients.dim()}")

        # Normalize to [0, 1]
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def release(self):
        """Remove hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model, model_name):
    """
    Return the appropriate target layer for Grad-CAM given the model name.

    Parameters
    ----------
    model : torch.nn.Module
    model_name : str
        One of 'DenseNet121', 'DenseNet201', 'Xception', 'ViT-B/16'.

    Returns
    -------
    target_layer : torch.nn.Module
    """
    name_lower = model_name.lower().replace("-", "").replace("/", "").replace(" ", "")

    if "densenet121" in name_lower:
        return model.features.denseblock4.denselayer16.conv2
    elif "densenet201" in name_lower:
        return model.features.denseblock4.denselayer32.conv2
    elif "xception" in name_lower:
        return model.conv4  # timm legacy_xception last conv
    elif "vit" in name_lower:
        return model.encoder.layers[-1].ln_1  # last transformer block
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def overlay_heatmap(image_np, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on image.

    Parameters
    ----------
    image_np : np.ndarray
        Original image (H, W, 3), uint8, RGB.
    heatmap : np.ndarray
        Heatmap (any size), float [0, 1].
    alpha : float
        Blending factor.

    Returns
    -------
    overlay : np.ndarray
        Blended image (H, W, 3), uint8, RGB.
    """
    h, w = image_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (alpha * colored_rgb + (1 - alpha) * image_np).astype(np.uint8)
    return overlay


def save_gradcam(original_image_np, heatmap, save_path, title="Grad-CAM"):
    """Save Grad-CAM visualization as a figure."""
    overlay = overlay_heatmap(original_image_np, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
