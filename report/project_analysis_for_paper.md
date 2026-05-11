# Comprehensive Project Analysis Report — Team505 Phase 3
## Explainable Pneumonia Detection on ChestX-ray14

This document synthesizes the empirical results, training methodologies, and analytical findings from Phase 3 of the Team505 Pneumonia Detection project. It is intended to serve as the master technical reference document for drafting the final academic research paper.

---

## 1. Dataset Characteristics & Preprocessing

### 1.1 Data Source and Structure
The project utilizes the **NIH ChestX-ray14** dataset. To ensure clinical validity and prevent data leakage, a strict **patient-wise split** was enforced across train, validation, and test subsets (70/15/15 ratio), guaranteeing no patient appears in more than one subset.

**Final Dataset Composition (Clean, Balanced):**
- **Training Set:** 11,943 total images (1,004 Positives, 10,939 Negatives) — Positive Rate: 8.41%
- **Validation Set:** 2,611 total images (186 Positives, 2,425 Negatives) — Positive Rate: 7.12%
- **Test Set:** 2,618 total images (241 Positives, 2,377 Negatives) — Positive Rate: 9.21%

### 1.2 Class Selection Strategy
To construct a realistic but computationally tractable dataset:
- We retained 12 key NIH pathology classes, deliberately dropping "Fibrosis," "Hernia," and "Pleural_Thickening".
- The representation of remaining classes was capped at 1,431 images per class to prevent common findings from dominating the negative distribution.
- The task was formulated as a strictly binary classification: **Pneumonia vs. No Finding / Other Abnormalities**.

---

## 2. Training Strategy & Optimization

The Phase 3 training pipeline was heavily hardened to overcome the dataset's inherent challenges (severe 13:1 class imbalance and estimated 30–40% label noise on the Pneumonia class).

### 2.1 Handling Class Imbalance
Initial experiments utilizing overlapping imbalance treatments (WeightedRandomSampler combined with positive class weighting) resulted in severe over-correction and decision threshold collapse. The finalized pipeline employed a **single-method approach** using **Focal Loss**. By down-weighting the loss contribution from easily classified negative examples, Focal Loss successfully calibrated the models without collapsing the classification threshold.

### 2.2 Advanced Regularisation Techniques
All 12 architectures were trained using a unified, standardized training regime:
- **Label Smoothing (0.1):** Prevented the models from becoming overconfident on noisy NIH labels, reducing overfitting to spurious artifacts.
- **Mixup Augmentation (alpha=0.2):** Enhanced generalisation by encouraging the models to learn linear interpolations of the feature space, stabilizing training on the limited positive samples.
- **Learning Rate Schedule:** Cosine Annealing Warm Restarts (CAWR) was used to systematically escape local minima.
- **Data Augmentation:** Strong geometric and photometric augmentations (rotation, flip, brightness, contrast adjustments) were applied to prevent memorization.

---

## 3. Evaluated Architectures

A diverse suite of 12 model architectures was evaluated to benchmark performance ceilings and architectural inductive biases.

**CNN Family (9 Models):**
1. ResNet-101 (44.5M)
2. EfficientNet-B3 (12.2M)
3. MobileNet-V2 (3.4M)
4. InceptionV3 (23.8M)
5. Xception (20.8M)
6. DenseNet-201 (18.1M)
7. DenseNet-121 (6.95M)
8. ResNet-50 (25.6M)
9. VGG-16 (138M)

**Vision Transformer Family (3 Models):**
1. DeiT-S (22M)
2. Swin-T (28M)
3. ViT-B/16 (85.8M)

---

## 4. Empirical Performance Results

Models were evaluated on the 2,611-image validation set using **Test Time Augmentation (TTA)** (5 variants: original, horizontal flip, ±7° rotation, +0.15 brightness).

### 4.1 Global Performance Ceiling
- **Highest AUC:** ResNet-101 (0.6747) followed closely by EfficientNet-B3 (0.6740).
- **Highest F1-Score:** EfficientNet-B3 (0.2198).
- **Highest PR-AUC:** InceptionV3 (0.1344).
- **Highest Recall:** MobileNet-V2 (0.6398, catching 119/186 positives).

The spread between the best CNN (0.6747) and the worst CNN (0.6517) is extremely narrow (0.023 AUC). This suggests that the **dataset's inherent label noise dictates the performance ceiling**, rather than model depth or capacity.

### 4.2 CNNs vs. Transformers
The CNN family uniformly outperformed the Transformer models. The best Transformer (DeiT-S, AUC 0.6463) underperformed the worst CNN (VGG-16, AUC 0.6517). The ViT-B/16 model, despite being the largest parameter-wise (85.8M), was the worst performer overall (AUC 0.6246). This confirms that Transformers lack the spatial inductive biases of CNNs and require vastly more positive training samples (far exceeding the 1,004 available in our dataset) to converge effectively on localized medical pathologies.

### 4.3 Efficiency Breakdown
MobileNet-V2 proved to be the most efficient architecture. At only 3.4M parameters, it achieved the 3rd highest AUC (0.6676) and the highest recall, making it an ideal candidate for mobile or low-resource clinical screening deployments where sensitivity is critical.

### 4.4 Test Time Augmentation (TTA) Impact
TTA provided the most substantial improvements to models with higher single-inference variance (DenseNet-121 gained +0.019 AUC). Conversely, highly stable, well-converged models (EfficientNet-B3) gained almost nothing (+0.0001 AUC), demonstrating that TTA acts as a variance-reduction technique rather than an absolute performance amplifier.

---

## 5. Explainable AI (XAI) Analysis

To ensure transparency and clinical safety, all architectures were audited using four distinct XAI methods.

### 5.1 CNN Interpretability (Grad-CAM)
DenseNet-121, DenseNet-201, and ResNet-101 produced the most anatomically coherent Grad-CAM heatmaps. True Positives consistently highlighted lower-lobe opacifications corresponding to clinical pneumonia presentations. Feature reuse in dense blocks and residual connections heavily promoted tight, spatially focused attribution.

### 5.2 Transformer Interpretability (Attention Rollout)
As Grad-CAM is incompatible with flat token sequences, Attention Rollout was applied to the Transformers. ViT-B/16 produced blurry, coarse mappings (16x16 patch resolution) that provided low clinical utility. Swin-T provided marginally better, more localized maps due to its hierarchical windowed attention design, but still failed to match the spatial precision of CNN Grad-CAMs.

### 5.3 Identifying Confounds (LIME & Integrated Gradients)
LIME (perturbation-based) and Integrated Gradients (path-attribution) successfully audited False Positive cases across all models. Both methods consistently revealed that models were learning a systematic confound: **cardiac silhouette borders and rib-vertebra overlaps**. These dense anatomical borders were repeatedly mistaken for the texture and spatial frequency of early-stage pneumonia consolidation.

### 5.4 Global Consistency (SHAP)
SHAP confirmed that models were learning globally consistent features (opacity textures) across different patients rather than memorizing individual image artifacts. However, due to its high computational cost (2-5 minutes per image), SHAP is relegated to offline post-hoc analysis rather than real-time clinical screening.

---

## 6. Key Conclusions for Publication

1. **Label Noise is the Limiting Factor:** Standard architecture scaling does not improve pneumonia detection on ChestX-ray14. The ~30–40% noise in the automatic NLP-derived labels creates a hard upper bound (~0.67 AUC) on patient-wise generalized performance.
2. **CNNs Remain Superior for Small-Data Medical Imaging:** Transformers fail to generalize efficiently on datasets with ~1,000 positive samples. CNNs, leveraging spatial inductive biases, are vastly more data-efficient for localized anomaly detection tasks.
3. **Focal Loss is Essential for Imbalance:** Rather than complex sampling strategies, Focal Loss provides the cleanest solution to severe class imbalance by preventing classification threshold collapse and ensuring mathematically sound probability calibration.
4. **XAI is Crucial for Safe Deployment:** XAI successfully exposed non-pulmonary anatomical confounds (rib overlaps) driving False Positives. Without these insights, the models would present a silent risk in clinical environments. Any deployment of these models must be strictly positioned as a **triage support tool**, requiring mandatory human review for all positive flags to prevent the dangerous consequences of False Negatives (36% missed cases at best).
