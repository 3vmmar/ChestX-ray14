# XAI Method Comparison — Team505 Phase 2

## Overview

Four explainability methods were applied to all trained models on a shared
evaluation subset (2 TP, 2 TN, 2 FP, 2 FN per model from the validation set):

| Method | Type | Library | Output |
|--------|------|---------|--------|
| Grad-CAM | Gradient-based | Custom (PyTorch hooks) | Heatmap overlay |
| LIME | Perturbation-based | `lime` | Superpixel regions |
| SHAP | Shapley-value-based | `shap` (PartitionExplainer) | Pixel attribution map |
| Integrated Gradients | Path-based attribution | `captum` | Pixel attribution map |

---

## Method Details

### 1. Grad-CAM (Gradient-weighted Class Activation Mapping)

- **What it explains:** Which spatial regions of the image most influenced the model's prediction, based on the gradients flowing into the last convolutional (or attention) layer.
- **Strengths:**
  - Fast (single forward + backward pass)
  - Produces intuitive, coarse-grained localization maps
  - Well-suited for clinical visualization ("where is the model looking?")
  - Works with both CNNs and adapted for ViT
- **Weaknesses:**
  - Resolution limited by the spatial dimensions of the target layer
  - Can miss fine-grained or multi-region patterns
  - For ViT, requires adaptation (attention rollout or token-level gradients)
- **Computational cost:** Very low (~0.5s per image)
- **Practical usefulness:** High — the most commonly used XAI method in medical imaging
- **Observed behavior:** On DenseNet models, heatmaps consistently highlighted lung fields. On TP cases, activation concentrated around lower lung zones where infiltrates are common. On TN cases, activations were more diffuse. On ViT-B/16, the patch-based attention produced blockier heatmaps reflecting the 16x16 patch grid.
- **Clinical plausibility:** Good — highlights anatomically relevant regions for pneumonia

### 2. LIME (Local Interpretable Model-agnostic Explanations)

- **What it explains:** Which superpixel regions of the image are most important for the prediction, based on local perturbation experiments.
- **Strengths:**
  - Model-agnostic — works identically for all architectures
  - Produces human-interpretable segmented explanations
  - Shows both positive (supporting) and negative (opposing) evidence
- **Weaknesses:**
  - Stochastic — results vary slightly between runs despite fixed seeds
  - Computationally expensive (200 perturbations per image)
  - Superpixel granularity may not match medical feature boundaries
  - Sensitive to segmentation parameters
- **Computational cost:** Moderate (~1-2s per image with 200 samples)
- **Practical usefulness:** Medium-High — useful for individual case review
- **Observed behavior:** LIME highlighted irregular regions in positive cases, often corresponding to opacified lung areas. In TP cases, supporting superpixels tended to cluster around the cardiac silhouette and lower lobes. In TN cases, fewer and more scattered positive regions appeared. Results were consistent across all 4 models.
- **Clinical plausibility:** Moderate — superpixel boundaries don't always align with anatomical structures

### 3. SHAP (SHapley Additive exPlanations)

- **What it explains:** Pixel-level importance based on Shapley values from cooperative game theory, computed via PartitionExplainer.
- **Strengths:**
  - Theoretically grounded in axiomatic fairness properties
  - Provides fine-grained pixel-level attributions
  - Model-agnostic with PartitionExplainer (works seamlessly for CNNs and Transformers)
- **Weaknesses:**
  - Computationally expensive (requires hundreds of model evaluations per image)
  - Raw SHAP maps can be noisy without aggregation
- **Computational cost:** High (~5-10s per image with 300 evaluations)
- **Practical usefulness:** Medium — valuable for research but less intuitive for clinicians
- **Observed behavior:** SHAP was applied successfully to all 4 models using hierarchical image partitioning. Attribution maps showed fine-grained patterns, typically highlighting opacities in positive cases and structural context in negative cases.
- **Clinical plausibility:** Moderate — produces detailed maps but may include artifacts from the partitioning pattern

### 4. Integrated Gradients

- **What it explains:** Attribution of each pixel relative to a baseline (black image), computed by integrating gradients along the straight-line path from baseline to input.
- **Strengths:**
  - Axiomatically sound (satisfies Sensitivity and Implementation Invariance)
  - Deterministic — same input always produces same output
  - Works with any differentiable model including Transformers
  - Produces fine-grained pixel-level attributions
- **Weaknesses:**
  - Requires choosing a baseline (black image may not be ideal for X-rays)
  - Computationally intensive (50 interpolation steps by default)
  - Raw maps can be noisy; requires absolute-value aggregation
- **Computational cost:** High (~2-4s per image with 50 steps)
- **Practical usefulness:** Medium-High — strong theoretical foundation
- **Observed behavior:** IG worked on all 4 models including ViT-B/16. Attribution maps on TP cases showed high attribution in lung regions. On FP cases, attributions revealed which non-pneumonia features the model incorrectly relied on. Consistent patterns emerged across DenseNet variants.
- **Clinical plausibility:** Good — particularly useful for understanding model errors

---

## Compatibility Matrix

| Method | DenseNet-121 | DenseNet-201 | Xception | ViT-B/16 |
|--------|:---:|:---:|:---:|:---:|
| Grad-CAM | OK | OK | OK | OK |
| LIME | OK | OK | OK | OK |
| SHAP (PartitionExplainer) | OK | OK | OK | OK |
| Integrated Gradients | OK | OK | OK | OK |

---

## Summary Ranking

| Criterion | Best Method |
|-----------|-------------|
| Speed | Grad-CAM |
| Interpretability for clinicians | Grad-CAM / LIME |
| Theoretical soundness | SHAP / Integrated Gradients |
| Model compatibility | LIME (model-agnostic) |
| Fine-grained attribution | Integrated Gradients |
| Overall recommendation | Grad-CAM (primary) + IG (verification) |
