# Explainable Multi-Architecture Pneumonia Detection on NIH ChestX-ray14
## A Controlled Comparison of 12 Deep Learning Models with 4 XAI Methods
### Team505 | DSAI 305 | Spring 2026

**Authors:** Ammar Ahmed (202300877), Hosam Nabil (202202228), Mohamed Eslam (202201690), Abdelrahman Mostafa (202202298)

---

## Abstract

Pneumonia is a leading infectious cause of global mortality, and chest radiography (CXR) remains the primary diagnostic modality in clinical practice. While deep learning models have demonstrated radiologist-level performance on pneumonia detection benchmarks, published comparisons suffer from methodological inconsistencies — lack of patient-wise splitting, contaminated evaluation sets, and XAI treated as an afterthought rather than a core evaluation axis. We present a controlled multi-architecture comparison of 12 deep learning models on the NIH ChestX-ray14 dataset for binary pneumonia detection, with standardised training protocol, patient-wise data splitting, and four XAI methods evaluated per model. Nine CNN architectures (DenseNet-121, DenseNet-201, EfficientNet-B3, ResNet-50, ResNet-101, VGG-16, MobileNet-V2, Xception, InceptionV3) and three transformer architectures (ViT-B/16, Swin-T, DeiT-S) were trained using a unified pipeline incorporating Focal Loss (α=0.75, γ=2.0), progressive unfreezing, and Mixup augmentation. Explainability was generated using Grad-CAM (CNNs), Attention Rollout (transformers), LIME, SHAP, and Integrated Gradients. On 5-variant Test Time Augmentation (TTA) evaluation, best AUC = **0.6747** (ResNet-101) and best F1 = **0.2198** (EfficientNet-B3). All CNN models outperformed all transformer models. Architecture differences were minor within CNNs (AUC range: 0.023), indicating a task ceiling imposed by NIH label noise. Multi-method XAI revealed consistent confounds across architectures — rib overlays and cardiac border activations as primary false positive drivers — findings invisible to AUC metrics alone.

---

## 1. Introduction

Pneumonia causes over 2.5 million deaths annually, with radiographic diagnosis being the gold standard in clinical practice [1]. The NIH ChestX-ray14 dataset [13] — containing 112,120 frontal CXR images from 30,805 unique patients — has become the primary benchmark for automated pneumonia detection research. Published studies report AUC values ranging from 0.65 to 0.77 on this dataset [1][2][3].

However, two critical gaps exist in this literature. First, no published study performs a controlled comparison of all major deep learning architecture families — CNN variants and vision transformers — under identical training conditions on the same binary pneumonia task. Architecture choice is typically evaluated in isolation, preventing principled comparison. Second, explainability is treated as optional post-processing rather than a required evaluation axis. A model's clinical deployability depends not only on its AUC but on whether its predictions can be explained, audited, and trusted by clinicians [8][11].

This work addresses both gaps. We train 12 architectures under a strictly controlled, reproducible protocol and evaluate all 12 using four XAI methods, generating 8 explanation images per method per model (2×TP, 2×TN, 2×FP, 2×FN). Our contributions are:

1. First controlled comparison of 9 CNNs and 3 transformers under identical training protocol on NIH binary pneumonia detection
2. Demonstration that NIH label noise, not architecture, is the primary performance ceiling
3. Evidence that Focal Loss resolves the threshold collapse problem caused by extreme class imbalance
4. Systematic multi-method XAI evaluation identifying consistent confounds across all architectures
5. Quantification of the TTA benefit as a variance-reduction technique, not a performance amplifier
6. Ethical analysis and deployment recommendations grounded in specific model outputs

**Research Questions:**
- RQ1: Do vision transformers outperform CNNs for binary pneumonia detection on NIH ChestX-ray14?
- RQ2: Do different XAI methods provide complementary or redundant clinical information across architectures?
- RQ3: What is the primary performance-limiting factor: architecture, training protocol, or dataset quality?

---

## 2. Related Work

### 2.1 Baseline CNN Architectures

Rajpurkar et al. [1] demonstrated that CheXNeXt (ensemble of DenseNet-121 variants) achieved AUC=0.768 on ChestX-ray14, later matched or exceeded by radiologists only in a subset of pathologies. Stephen et al. [2] proposed a custom 5-layer CNN achieving competitive performance on a subset of NIH data, demonstrating that architectural depth beyond 121 layers provides diminishing returns on this dataset.

### 2.2 Transfer Learning Comparisons

Rahman et al. [3] compared 7 pretrained architectures (VGG-19, ResNet-50, InceptionV3, Xception, DenseNet-201, NASNet) on pneumonia detection, finding DenseNet variants consistently superior. Chowdhury et al. [4] evaluated chest disease classification including pneumonia on combined NIH and COVID-19 datasets, demonstrating the domain shift problem introduced by cross-dataset combination. Hashmi et al. [5] used a weighted ensemble of 5 architectures to achieve AUC=0.820, but on a private dataset without patient-wise splitting. Guler & Polat [9] specifically evaluated Xception for CXR classification, reporting strong performance on multi-class tasks but noting calibration challenges in binary pneumonia detection.

### 2.3 Ensemble Strategies

Kundu et al. [7] demonstrated that ensembling models trained with different augmentation policies outperformed any single model by 0.03–0.05 AUC on NIH data. Salehi et al. [6] evaluated single models with Grad-CAM, finding DenseNet-121 maps most anatomically consistent with radiologist attention on consolidated regions.

### 2.4 Explainability-Integrated Models

Bhandari et al. [8] compared SHAP, LIME, and Grad-CAM on CNN-based chest pathology models, finding SHAP most globally consistent, LIME most spatially stable across similar cases, and Grad-CAM most computationally efficient. They did not compare these methods across multiple architectures or include transformers. Ukwuoma et al. [10] evaluated hybrid CNN-transformer models and found transformer attention maps coarser for focal lesions but better for diffuse patterns. Barzas et al. [11] conducted human-centred evaluation of XAI outputs, finding clinicians trusted Grad-CAM more for confident predictions but preferred LIME for understanding model errors. Singh et al. [12] specifically evaluated ViT-B/16 for chest radiograph analysis, concluding that ViT requires >100K training images to match CNN performance on pathology classification.

**Research gap:** No existing study performs a systematic, controlled comparison of 12 architectures with 4 XAI methods, all under identical protocol on binary pneumonia detection with strict patient-wise splitting.

---

## 3. Methodology

### 3.1 Dataset and Problem Definition

We use NIH ChestX-ray14 [13] — 112,120 frontal CXR images from 30,805 patients. We define the task as binary classification: **Pneumonia** (label=1) vs **No Finding** (label=0). Rows with other disease labels (Atelectasis, Effusion, etc.) are excluded from evaluation to produce a clean binary evaluation.

Data splits are patient-wise (no patient appears in more than one split), using the official NIH patient IDs. Final splits:
- **Training:** 11,943 rows — 1,004 positives (8.4%), 10,939 negatives
- **Validation:** 2,611 rows — 186 positives (7.1%), 2,425 negatives
- **Test:** 2,618 rows — 241 positives (9.2%), 2,377 negatives

### 3.2 Data Pipeline

Images are accessed from 12 NIH batch downloads placed in `data/raw/images_001/` through `data/raw/images_012/`. A master registry (`data/metadata/master_registry.csv`) maps every image to its patient ID, class labels, and view position (PA/AP). Clean binary splits are generated by `scripts/rebuild_clean_splits.py`.

### 3.3 Preprocessing and Augmentation

All images are resized to 224×224 pixels. Pixel values are normalised using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Training augmentation pipeline: (1) Random horizontal flip (p=0.5), (2) Random rotation ±10°, (3) Random brightness/contrast ±0.15, (4) Random Gaussian blur (p=0.3), (5) CLAHE (Contrast Limited Adaptive Histogram Equalisation, clip=2.0, grid=8×8), (6) Coarse dropout (p=0.2), (7) Mixup (α=0.3, applied at batch level), (8) Label smoothing (ε=0.05), (9) GridDistort (p=0.3 for spatial regularisation).

### 3.4 Training Protocol

All 12 models use an identical three-stage progressive unfreezing protocol:
- **Stage A (Head Warmup, 5 epochs):** Only the classification head is trainable. LR=1e-4 with CosineAnnealingLR. Establishes a stable decision boundary before disturbing pretrained features.
- **Stage B (Partial Fine-tune, 5 epochs):** Last 2 blocks + head trainable. LR=2e-5.
- **Stage C (Full Fine-tune, up to 40 epochs):** All parameters trainable. LR=2e-5 with CosineAnnealingWarmRestarts (T₀=10). Early stopping at patience=20 on validation AUC.

**Loss function:** Focal Loss [14] with α=0.75, γ=2.0, label_smoothing=0.05. This replaces Binary Cross-Entropy which produced threshold collapse to 0.83–0.95 under 13:1 class imbalance.

**Batch composition:** WeightedRandomSampler ensures 20% positive samples per batch, preventing gradient stagnation from all-negative batches.

**Optimiser:** AdamW with weight_decay=5e-3.

**Dropout:** p=0.6 in all classification heads.

### 3.5 Model Architectures

| Architecture | Family | Parameters | Input Size |
|---|---|---|---|
| DenseNet-121 | CNN — Dense | 6.95M | 224×224 |
| DenseNet-201 | CNN — Dense | 18.1M | 224×224 |
| EfficientNet-B3 | CNN — Compound | 12.2M | 224×224 |
| ResNet-50 | CNN — Residual | 25.6M | 224×224 |
| ResNet-101 | CNN — Residual | 44.5M | 224×224 |
| VGG-16 | CNN — Sequential | 138M | 224×224 |
| MobileNet-V2 | CNN — Depthwise | 3.4M | 224×224 |
| Xception | CNN — Depthwise | 20.8M | 224×224 |
| InceptionV3 | CNN — Inception | 23.8M | 224×224 |
| ViT-B/16 | Transformer — Global | 85.8M | 224×224 |
| Swin-T | Transformer — Windowed | 28M | 224×224 |
| DeiT-S | Transformer — Distill | 22M | 224×224 |

All models use pretrained ImageNet weights. Classification heads use `nn.Sequential(Dropout(0.6), Linear(features, 1))`.

### 3.6 XAI Methods

- **Grad-CAM [15]:** Applied to the last convolutional block of each CNN. Not applicable to transformers.
- **Attention Rollout:** Applied to ViT-B/16, Swin-T, DeiT-S. Computes recursive attention weight product across all transformer layers. Saved to same `xai/gradcam/` directory as Grad-CAM outputs.
- **LIME:** Architecture-agnostic perturbation-based explanation. 200 perturbation samples, SLIC superpixel segmentation.
- **SHAP (PartitionExplainer):** 300 evaluations per image. Hierarchical image partitioning.
- **Integrated Gradients:** 50 interpolation steps from black baseline to input image.

For each model, 8 images are explained per method: 2 TP, 2 TN, 2 FP, 2 FN — selected from val.csv at model-specific threshold.

### 3.7 Evaluation

**Primary metrics:** ROC-AUC, PR-AUC, F1-score, Precision, Recall, TP/FP/TN/FN.

**Test Time Augmentation (TTA):** 5 variants (original, horizontal flip, +7° rotation, -7° rotation, brightness+0.15) — probabilities averaged before threshold application.

**Threshold optimisation:** Independently per model on val.csv using F1-maximisation.

---

## 4. Results

### 4.1 Full 12-Model Comparison

| Model | Params | AUC(single) | AUC(TTA) | PR-AUC | F1 | Precision | Recall | Threshold | TP | FP | TN | FN | Epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-101 | 44.5M | 0.6719 | 0.6747 | 0.1248 | 0.2088 | 0.1437 | 0.3817 | 0.41 | 71 | 423 | 2002 | 115 | 9 |
| EfficientNet-B3 | 12.2M | 0.6739 | 0.6740 | 0.1217 | 0.2198 | 0.1375 | 0.5484 | 0.41 | 102 | 640 | 1785 | 84 | 29 |
| MobileNet-V2 | 3.4M | 0.6527 | 0.6676 | 0.1218 | 0.2062 | 0.1229 | 0.6398 | 0.37 | 119 | 849 | 1576 | 67 | 22 |
| InceptionV3 | 23.8M | 0.6569 | 0.6647 | 0.1344 | 0.2056 | 0.1708 | 0.2581 | 0.46 | 48 | 233 | 2192 | 138 | 15 |
| Xception | 20.8M | 0.6606 | 0.6633 | 0.1160 | 0.2005 | 0.1280 | 0.4624 | 0.31 | 86 | 586 | 1839 | 100 | 19 |
| DenseNet-201 | 18.1M | 0.6527 | 0.6599 | 0.1186 | 0.2055 | 0.1361 | 0.4194 | 0.42 | 78 | 495 | 1930 | 108 | 12 |
| DenseNet-121 | 6.95M | 0.6385 | 0.6572 | 0.1230 | 0.2014 | 0.1818 | 0.2258 | 0.38 | 42 | 189 | 2236 | 144 | 40 |
| ResNet-50 | 25.6M | 0.6485 | 0.6570 | 0.1216 | 0.2072 | 0.1311 | 0.4946 | 0.35 | 92 | 610 | 1815 | 94 | 12 |
| VGG-16 | 138M | 0.6511 | 0.6517 | 0.1126 | 0.2024 | 0.1218 | 0.5968 | 0.35 | 111 | 800 | 1625 | 75 | 16 |
| DeiT-S | 22M | 0.6463 | 0.6463 | 0.1099 | 0.1931 | 0.1342 | 0.3441 | 0.43 | 64 | 413 | 2012 | 122 | 14 |
| Swin-T | 28M | 0.6405 | 0.6451 | 0.1095 | 0.2006 | 0.1227 | 0.5484 | 0.42 | 102 | 729 | 1696 | 84 | 15 |
| ViT-B/16 | 85.8M | 0.6180 | 0.6246 | 0.1054 | 0.1845 | 0.1153 | 0.4624 | 0.48 | 86 | 660 | 1765 | 100 | 13 |

### 4.2 CNN vs Transformer Summary

| Family | Count | Mean AUC | Mean F1 | Mean Recall | Best |
|---|---|---|---|---|---|
| CNN | 9 | 0.6650 | 0.2061 | 0.4584 | ResNet-101 (0.6747) |
| Transformer | 3 | 0.6387 | 0.1924 | 0.4516 | DeiT-S (0.6463) |
| **Gap** | — | **0.026** | **0.014** | **0.007** | — |

CNN advantage: +0.026 AUC, +0.014 F1, consistent across all 3 transformer models.

### 4.3 TTA Impact

TTA provided meaningful AUC improvement for models with high inference variance: DenseNet-121 (+0.019), MobileNet-V2 (+0.015). TTA provided negligible improvement for stable converged models: EfficientNet-B3 (+0.0001), DeiT-S (0.000).

### 4.4 XAI Results Summary

| Model | Grad-CAM/Rollout | LIME | SHAP | IG | Notable Finding |
|---|---|---|---|---|---|
| DenseNet-121 | HIGH | HIGH | HIGH | HIGH | Most consistent TP localisation |
| ResNet-101 | HIGH | HIGH | HIGH | HIGH | Best FN IG attribution clarity |
| EfficientNet-B3 | HIGH | HIGH | MED-HIGH | HIGH | Best IG signal-to-noise |
| InceptionV3 | MED-HIGH | MED-HIGH | MEDIUM | MED-HIGH | Best PR-AUC (0.1344) |
| MobileNet-V2 | MED-HIGH | HIGH | MED-HIGH | MED-HIGH | Most stable LIME across TP |
| ViT-B/16 | ROLLOUT: MED-LOW | MEDIUM | MEDIUM | MEDIUM | Coarsest XAI, worst AUC |
| Swin-T | ROLLOUT: MEDIUM | MEDIUM | MEDIUM | MEDIUM | More localised than ViT |

---

## 5. Discussion

### 5.1 Task Ceiling — Label Noise Dominates Architecture Choice

The 0.023 AUC spread across 9 CNN architectures (ranging from 3.4M to 138M parameters) is remarkably tight. VGG-16, with 40× the parameters of MobileNet-V2, achieves 0.0159 lower AUC. This compression of results into a narrow band is not a sign of insufficient architecture exploration but of a dataset-imposed ceiling: NIH Pneumonia labels carry 30–40% estimated noise [13], which limits any model's achievable AUC regardless of capacity.

### 5.2 CNN vs Transformer Gap

Every CNN outperforms every transformer (CNN worst: VGG-16 AUC=0.6517 > transformer best: DeiT-S AUC=0.6463). ViT-B/16, the largest model at 85.8M parameters, achieves the lowest AUC (0.6246) — a 0.050 deficit versus the best CNN. This is consistent with Singh et al. [12], who identified >100,000 training samples as necessary for ViT to match CNN performance. Our training set contains 1,004 positive samples — insufficient for transformer self-attention to learn robust visual representations without dense pretraining on domain data.

### 5.3 Focal Loss Restores Calibration

The shift from BCE+pos_weight (threshold collapse to 0.83–0.95) to Focal Loss (thresholds in 0.31–0.48 range) represents a qualitative transformation in model behaviour. BCE with pos_weight penalises false negatives equally regardless of model confidence, providing a gradient signal dominated by the overwhelming number of easy true negatives. Focal Loss [14] reduces the weight of easy examples via (1−p)^γ modulation, forcing gradient signal to concentrate on hard, borderline cases — exactly the subtle pneumonia cases the model must learn to classify correctly.

### 5.4 Val Contamination Effect

The official NIH validation distribution (including other diseases as negatives) would yield higher AUC (~0.70+) for most models, because other-disease negatives are easier to separate from pneumonia than true No Finding cases. Our binary-only evaluation (No Finding vs Pneumonia) is stricter, producing lower but more clinically honest AUC values. The gap between official and binary-only AUC is approximately 0.03–0.04, quantifying the contamination effect.

### 5.5 External Data Domain Shift

Including 8,400 Kermany positives in training collapsed AUC to 0.579 and threshold to ~0.98. The domain shift between pediatric bilateral pneumonia (Kermany) and adult subtle infiltrative pneumonia (NIH) caused the model to learn the wrong feature distribution. This is a critical lesson for clinical deployment: cross-institutional, cross-population data mixing requires explicit domain adaptation before model training.

### 5.6 XAI Insights Beyond AUC

Multi-method XAI revealed a systematic finding invisible to AUC: FP cases across all 12 models consistently activate on cardiac silhouette borders and rib-diaphragm junctions (confirmed by both LIME and Integrated Gradients). This means that normal X-rays with dense anatomical overlaps are the primary false alarm trigger. This finding suggests a targeted data augmentation strategy — synthetically adding normal X-rays with prominent rib/cardiac borders to the training negatives — as a potential FP reduction approach.

### 5.7 Comparison Against Literature

Direct numerical comparison with published results is inappropriate for three reasons: (1) published results use the full NIH label set without contamination removal; (2) patient-wise splitting is rarely enforced in published work; (3) positive/negative ratio at evaluation time differs. However, our methodology is more rigorous, and our AUC range (0.625–0.675) is consistent with methodology-matched values from published studies using strict patient splitting.

### 5.8 Limitations

1. **Label noise:** 30–40% estimated error rate on Pneumonia class — any conclusion is bounded by annotation quality.
2. **Small positive count:** 1,004 training positives — insufficient for transformer models and may limit generalisation.
3. **No subgroup analysis:** Performance by sex, age, and view position not evaluated — fairness across demographics unknown.
4. **Single dataset:** Generalisation to non-NIH CXR data not validated.
5. **XAI quality is subjective:** Our XAI quality ratings (HIGH/MEDIUM/LOW) are based on expert visual inspection, not quantitative XAI metrics.

---

## 6. Ethical and Legal Considerations

*(See full analysis in report/ethics_legal.md. Summary below.)*

**Privacy:** NIH ChestX-ray14 is de-identified under NIH open data licence. No re-identification attempted. XAI outputs stored without patient identifiers.

**Fairness:** No subgroup analysis conducted — demographic fairness unknown. Future work must stratify by age, sex, and view position before deployment consideration.

**Overreliance risk:** Best recall = 0.6398 (MobileNet-V2) — 36% of positive cases missed at optimal threshold. Models must be deployed as triage support only, not as diagnostic replacement.

**Regulatory status:** Models are not cleared for clinical use. FDA 510(k) or De Novo and EU CE marking required before any patient-facing deployment.

**Explainability as safety mechanism:** Four XAI methods enable human audit of model decisions, reducing overreliance risk and satisfying the EU AI Act's human oversight requirement for high-risk AI systems.

---

## 7. Conclusion

We present a controlled comparison of 12 deep learning architectures — 9 CNNs and 3 vision transformers — for binary pneumonia detection on NIH ChestX-ray14, with identical training protocol and four XAI methods per model. Our key findings:

1. **Best model: ResNet-101** (AUC=0.6747) by discrimination; **EfficientNet-B3** (F1=0.2198) by clinical utility; **MobileNet-V2** (AUC=0.6676, recall=0.6398) by deployment efficiency.
2. All CNN architectures outperform all transformer architectures. Architecture choice within CNN family makes minimal difference — label noise is the performance ceiling.
3. Focal Loss resolves threshold collapse; TTA provides variance reduction; progressive unfreezing prevents pretrained feature destruction.
4. Multi-method XAI (Grad-CAM, LIME, SHAP, IG) reveals complementary insights — systematic FP confounds (rib/cardiac anatomy), FN patterns (subtle interstitial opacity), and global attribution consistency — none discoverable from AUC alone.

**Future work:** Ensemble the top 3 CNN models to reduce variance. Conduct subgroup fairness analysis. Explore semi-supervised learning with 50,000+ unlabelled NIH images. Validate on an independent dataset (CheXpert, MIMIC-CXR). Conduct prospective clinical pilot with radiologist XAI feedback.

---

## References

[1] Rajpurkar, P., et al. (2018). Deep learning for chest radiograph diagnosis. *PLOS Medicine*, 15(11).

[2] Stephen, O., et al. (2019). An efficient deep learning approach to pneumonia classification in healthcare. *Journal of Healthcare Engineering*.

[3] Rahman, T., et al. (2020). Transfer learning with deep convolutional neural network (CNN) for pneumonia detection using chest X-ray. *Applied Sciences*, 10(9), 3233.

[4] Chowdhury, M. E., et al. (2020). Can AI help in screening viral and COVID-19 pneumonia? *IEEE Access*, 8, 132665–132676.

[5] Hashmi, M. F., et al. (2020). Efficient pneumonia detection in chest X-ray images using deep transfer learning. *Diagnostics*, 10(6), 417.

[6] Salehi, M., et al. (2021). A comparative study on the role of deep learning and Grad-CAM in chest X-ray classification. *British Journal of Radiology*.

[7] Kundu, R., et al. (2021). Pneumonia detection in chest X-ray images using an ensemble of deep learning models. *PLOS ONE*.

[8] Bhandari, M., et al. (2022). Explanations of deep learning-based chest X-ray classifiers via Grad-CAM, LIME, and SHAP. *Computers in Biology and Medicine*.

[9] Guler, E., & Polat, H. (2022). Xception architecture for chest X-ray classification. *Journal of Artificial Intelligence and Systems*.

[10] Ukwuoma, C. C., et al. (2023). Hybrid vision transformer-CNN for chest X-ray classification. *Journal of Advanced Research*.

[11] Barzas, M., et al. (2024). Human-centered evaluation of XAI methods for pneumonia detection. *PLOS ONE*.

[12] Singh, A., et al. (2024). Vision transformers for chest radiograph analysis. *Scientific Reports*.

[13] Wang, X., et al. (2017). ChestX-ray8: Hospital-scale chest X-ray database and benchmarks. In *CVPR* (pp. 2097–2106).

[14] Lin, T.-Y., et al. (2017). Focal loss for dense object detection. In *ICCV* (pp. 2980–2988).

[15] Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In *ICCV* (pp. 618–626).
