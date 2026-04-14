"""
XAI module for Team505 — Explainability methods for chest X-ray models.

Available explainers:
    - GradCAM: Gradient-weighted Class Activation Mapping
    - LIMEExplainer: Local Interpretable Model-agnostic Explanations
    - SHAPExplainer: SHapley Additive exPlanations (GradientExplainer)
    - IGExplainer: Integrated Gradients (via Captum)
"""

from .gradcam import GradCAM, get_target_layer, overlay_heatmap, save_gradcam
from .lime_explainer import LIMEExplainer, save_lime
from .shap_explainer import SHAPExplainer, save_shap
from .integrated_gradients import IGExplainer, save_ig

__all__ = [
    "GradCAM", "get_target_layer", "overlay_heatmap", "save_gradcam",
    "LIMEExplainer", "save_lime",
    "SHAPExplainer", "save_shap",
    "IGExplainer", "save_ig",
]
