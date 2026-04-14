# Presentation Outline — Team505 Phase 2

## Slide-Ready Content Structure

---

### Slide 1: Title

**Explainable Multi-Architecture Pneumonia Detection on ChestX-ray14**

Team 505 — DSAI 305
Ammar Ahmed · Hosam Nabil · Mohamed Eslam · Abdelrahman Mostafa

---

### Slide 2: Problem & Motivation

- Pneumonia: 2.5M deaths/year globally
- Chest X-ray: most common first-line diagnostic
- Challenge: inter-observer variability, shortage of radiologists
- **Goal:** Compare 4 deep learning architectures with 4 explainability methods
- **Key question:** Can we build a transparent, reproducible AI pipeline for pneumonia detection?

---

### Slide 3: Project Positioning

- **Unified pipeline** — same dataset, preprocessing, splits, metrics for all models
- **Patient-wise splitting** — no data leakage
- **Multi-architecture comparison** — CNNs vs. Transformer
- **Four XAI methods** — comprehensive explanability
- **Grounded in published research** — each model tied to a peer-reviewed paper

---

### Slide 4: Dataset — NIH ChestX-ray14

- 112,120 frontal CXR images, 30,805 patients
- 14 pathology labels (NLP-mined from reports)
- Binary target: Pneumonia (~1.3% positive rate)
- Patient-wise splits: Train / Val / Test
- Dev subset for rapid iteration (~8K images)

---

### Slide 5: Preprocessing & EDA

- Resize to 224×224, grayscale → RGB
- ImageNet normalization
- Training augmentation: flip, rotation (±10°), color jitter
- **EDA findings:** severe class imbalance, age/gender distributions, multi-label co-occurrence patterns
- Split verification: consistent positive rates across splits

---

### Slide 6: The 4 Models

| Model | Member | Params | Key Feature |
|-------|--------|--------|-------------|
| DenseNet-121 | Ammar | 7M | Dense connections, CheXNet baseline |
| DenseNet-201 | Hosam | 18M | Deeper dense blocks |
| Xception | Mohamed | 21M | Depthwise separable convolutions |
| ViT-B/16 | Abdelrahman | 86M | Self-attention on image patches |

- All share: same loss (weighted BCE), optimizer (Adam), scheduler, early stopping
- All trained on identical dev subset

---

### Slide 7: Results Comparison

| Metric | DenseNet-121 | DenseNet-201 | Xception | ViT-B/16 |
|--------|:---:|:---:|:---:|:---:|
| ROC-AUC | 0.608 | 0.605 | **0.616** | 0.577 |
| F1 | **0.208** | 0.192 | 0.205 | 0.198 |
| Precision | **0.135** | 0.110 | 0.128 | 0.122 |
| Recall | 0.455 | **0.773** | 0.526 | 0.526 |

- **Xception:** best AUC
- **DenseNet-121:** best F1 and precision — most balanced
- **ViT-B/16:** still learning at epoch 5 — needs more data/epochs

---

### Slide 8: XAI Methods Overview

| Method | Type | Speed | Output |
|--------|------|-------|--------|
| Grad-CAM | Gradient | Fast | Heatmap |
| LIME | Perturbation | Medium | Superpixels |
| SHAP | Shapley | Slow | Pixel map |
| Integrated Gradients | Path | Slow | Pixel map |

- Applied to shared evaluation subset (TP/TN/FP/FN cases)
- 8 images × 4 methods × 4 models = 128 visualizations

---

### Slide 9: XAI Results — Grad-CAM

*(Include sample Grad-CAM visualizations from outputs)*

- CNN models: focused heatmaps on lung fields
- ViT: blockier attention patterns (16×16 patch grid)
- TP cases: activation in lower lobes (consistent with pneumonia)
- FP cases: sometimes activated on non-pulmonary features

---

### Slide 10: XAI Results — LIME & Integrated Gradients

*(Include sample LIME and IG visualizations)*

- LIME: model-agnostic superpixel explanations
- IG: finest-grained attribution maps
- Key insight: models sometimes rely on image artifacts, not true pathology
- Multi-method consensus strengthens confidence in explanations

---

### Slide 11: XAI Comparison Summary

| Best for | Method |
|----------|--------|
| Speed | Grad-CAM |
| Clinical interpretability | Grad-CAM + LIME |
| Error analysis | Integrated Gradients |
| Model compatibility | LIME (model-agnostic) |

- **Recommendation:** Grad-CAM for screening + IG for deep analysis

---

### Slide 12: Ethical & Legal Considerations

- **Privacy:** De-identified dataset (HIPAA compliant)
- **Bias:** Single institution, NLP-mined labels (~10% noise), no subgroup analysis yet
- **Misuse risk:** Dev-mode models — NOT for clinical use
- **Explainability for trust:** XAI reveals what the model sees — essential for responsible AI

---

### Slide 13: Conclusion

1. DenseNet-121 remains the gold-standard baseline (CheXNet confirmed)
2. Xception offers slight AUC improvement with depthwise separable convolutions
3. ViT-B/16 underperforms in low-data regime — needs more training data
4. Class imbalance is the dominant challenge, not architecture
5. Multi-method XAI provides complementary insights
6. **Grad-CAM + IG** is the strongest explanatory combination

---

### Slide 14: Future Work

- Full-data training on complete train/val splits
- Official test-set evaluation
- Demographic subgroup fairness analysis
- External validation (CheXpert, MIMIC-CXR)
- Clinician user study for XAI utility
- Expand to 9-12 model variants per course requirements

---

### Slide 15: Thank You / Q&A

Team 505 — DSAI 305
Repository: `Team505_phase2/`
