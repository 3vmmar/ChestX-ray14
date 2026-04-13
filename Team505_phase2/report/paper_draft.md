# Explainable Multi-Architecture Pneumonia Detection on ChestX-ray14

**Team 505 — DSAI 305 Final Project**

---

## 1. Abstract

We present a unified pipeline for binary pneumonia detection on the NIH ChestX-ray14 dataset, comparing four deep learning architectures — DenseNet-121, DenseNet-201, Xception, and ViT-B/16 — under identical preprocessing, patient-wise splitting, and evaluation conditions. Each model represents a distinct architectural paradigm grounded in published chest X-ray classification research. We further apply four explainability (XAI) methods — Grad-CAM, LIME, SHAP, and Integrated Gradients — to generate visual explanations of model predictions. Preliminary results on a development subset show ROC-AUC scores ranging from 0.577 (ViT-B/16) to 0.616 (Xception), with all models challenged by severe class imbalance (~1.3% positive rate). Explainability analysis reveals that CNN models consistently attend to anatomically relevant lung regions, while the transformer model produces coarser spatial attention. We discuss the ethical implications of AI-assisted medical diagnosis, including label noise from NLP-mined annotations, demographic bias, and the critical role of explainability in clinical trust.

---

## 2. Introduction

Pneumonia is a leading cause of mortality worldwide, responsible for an estimated 2.5 million deaths annually. Chest X-ray (CXR) is the most common first-line imaging modality for pneumonia detection, but interpretation requires trained radiologists and is subject to inter-observer variability.

Deep learning has shown promise in automating CXR interpretation. Rajpurkar et al. (2017) demonstrated radiologist-level pneumonia detection using DenseNet-121 (CheXNet), establishing a benchmark on the NIH ChestX-ray14 dataset. Subsequent work has explored deeper DenseNet variants (Rahman et al., 2021), alternative CNN architectures such as Xception (Guler & Polat, 2021), and Vision Transformers (Singh et al., 2022).

However, most studies evaluate single architectures in isolation, making cross-architecture comparison difficult. Additionally, the "black box" nature of deep learning models raises concerns for clinical deployment, motivating the need for explainability methods.

This project addresses both gaps by:
1. Implementing a **unified comparison pipeline** for four architectures under identical conditions
2. Applying **four XAI methods** to provide complementary explanations of model behavior
3. Analyzing the **ethical and practical implications** of deploying such systems

---

## 3. Related Work

### 3.1 Deep Learning for Chest X-ray Classification

**CheXNet (Rajpurkar et al., 2017):** Demonstrated that a DenseNet-121 fine-tuned on ChestX-ray14 achieves F1 scores exceeding the average of four radiologists for pneumonia detection. This work established DenseNet-121 as the de facto baseline for CXR classification.

**CheXNeXt (Rajpurkar et al., 2018):** Extended CheXNet using an ensemble approach, improving multi-label classification performance across all 14 pathologies.

**Rahman et al. (2021):** Evaluated DenseNet-201 as part of a multi-architecture study for respiratory disease detection on chest X-rays, demonstrating competitive performance.

**Guler & Polat (2021):** Applied Xception with depthwise separable convolutions to thorax disease classification, showing that architectures beyond DenseNet can achieve strong results.

**Singh et al. (2022):** Assessed Vision Transformers for CXR pathology detection, demonstrating that ViT-based models can match or exceed CNN baselines on larger datasets.

### 3.2 Explainability in Medical AI

Grad-CAM (Selvaraju et al., 2017), LIME (Ribeiro et al., 2016), SHAP (Lundberg & Lee, 2017), and Integrated Gradients (Sundararajan et al., 2017) represent the major families of model explanation: gradient-based, perturbation-based, Shapley-value-based, and path-integration methods. Each provides a different lens on model behavior.

---

## 4. Dataset and Preprocessing

### 4.1 NIH ChestX-ray14

The dataset (Wang et al., 2017) contains 112,120 frontal-view chest X-ray images from 30,805 unique patients. Disease labels for 14 pathologies were mined from radiology reports using NLP, with an estimated accuracy of ~90%.

### 4.2 Binary Target

We define a binary pneumonia target: positive if "Pneumonia" appears in the Finding Labels, negative otherwise. The full dataset has a positive rate of approximately 1.3%.

### 4.3 Patient-Wise Splitting

To prevent data leakage, splits are performed at the patient level:
- **Train:** ~66% of patients
- **Validation:** ~17% of patients  
- **Test:** ~17% of patients (locked, unused during development)

A development subset (`train_dev.csv`) with ~10% of training patients is used for rapid iteration.

### 4.4 Preprocessing Pipeline

- Images are resized to 224×224 pixels
- Grayscale PNG → 3-channel RGB (channel replication)
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Training augmentation: random horizontal flip, rotation (±10°), color jitter (brightness/contrast ±0.1)
- Validation/test: resize and normalize only

---

## 5. Methodology

### 5.1 Unified Training Protocol

All models are trained with:
- **Loss:** BCEWithLogitsLoss with pos_weight (neg/pos ratio ≈ 9.82)
- **Optimizer:** Adam (lr=3×10⁻⁵, weight_decay=1×10⁻⁵)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=2)
- **Early stopping:** patience=3 on validation AUC
- **Batch size:** 32 (16 for ViT-B/16 due to memory)
- **Epochs:** Up to 5 (dev mode)

### 5.2 Evaluation Metrics

- ROC-AUC (primary — threshold-independent)
- F1-score (with threshold tuning)
- Precision, Recall, Accuracy
- Confusion matrix

### 5.3 Threshold Tuning

For each model, we search for the threshold (0.05–0.95, step 0.05) that maximizes F1-score on the validation set. This is critical given the severe class imbalance.

---

## 6. Model Baselines

### 6.1 DenseNet-121 (Ammar Ahmed)
- **Reference:** Rajpurkar et al. (CheXNet/CheXNeXt)
- **Parameters:** 6.95M
- **Key feature:** Dense connections enable feature reuse across layers

### 6.2 DenseNet-201 (Hosam Nabil)
- **Reference:** Rahman et al.
- **Parameters:** 18.09M
- **Key feature:** Deeper dense blocks for richer feature hierarchies

### 6.3 Xception (Mohamed Eslam)
- **Reference:** Guler & Polat
- **Parameters:** 20.81M
- **Key feature:** Depthwise separable convolutions for parameter-efficient feature extraction

### 6.4 ViT-B/16 (Abdelrahman Mostafa)
- **Reference:** Singh et al.
- **Parameters:** 85.80M
- **Key feature:** Self-attention on 16×16 image patches — no convolutional layers

---

## 7. Explainability Methods

Four XAI methods were applied to each model on a shared evaluation subset (8 images: 2 TP, 2 TN, 2 FP, 2 FN):

1. **Grad-CAM:** Gradient-weighted Class Activation Mapping using last feature layer activations
2. **LIME:** Local Interpretable Model-agnostic Explanations using superpixel perturbation (200 samples)
3. **SHAP:** Shapley value estimation via PartitionExplainer with hierarchical image masking
4. **Integrated Gradients:** Path-based attribution from black baseline (50 interpolation steps)

XAI outputs are saved as PNG visualizations for each model × method × case category.

---

## 8. Results

### 8.1 Classification Performance

| Model | ROC-AUC | F1 | Precision | Recall | Accuracy |
|-------|:-------:|:--:|:---------:|:------:|:--------:|
| DenseNet-121 | 0.608 | **0.208** | **0.135** | 0.455 | **0.690** |
| DenseNet-201 | 0.605 | 0.192 | 0.110 | **0.773** | 0.417 |
| Xception | **0.616** | 0.205 | 0.128 | 0.526 | 0.636 |
| ViT-B/16 | 0.577 | 0.198 | 0.122 | 0.526 | 0.619 |

### 8.2 Key Observations

- Xception achieves the highest AUC; DenseNet-121 the best F1/precision
- DenseNet-201's high recall (0.77) comes at severe precision cost
- ViT-B/16 lags in the low-data regime but was still improving at epoch 5
- All models struggle with sub-10% positive rate

### 8.3 XAI Results Summary

- **Grad-CAM, LIME, SHAP, and IG** successfully generated explanations for all 4 models (32 visualizations each, 128 total)
- **SHAP** via PartitionExplainer worked seamlessly across both CNNs and the Vision Transformer
- CNN models produce more focused, anatomically aligned attention maps
- ViT-B/16 produces blockier attention reflecting its patch grid

---

## 9. Comparative Analysis

Architecture comparison reveals that **model choice has limited impact at this data scale** (AUC range: 0.577–0.616). The dominant factor is class imbalance handling. CNN models with inductive biases (locality, equivariance) outperform the ViT in the low-data regime.

XAI comparison shows that **Grad-CAM + Integrated Gradients** provide the strongest complementary framework: Grad-CAM for fast, coarse localization and IG for fine-grained attribution analysis. LIME adds value for individual case review. Multi-method explanability is more informative than any single approach.

---

## 10. Ethical and Legal Considerations

1. **Privacy:** NIH ChestX-ray14 is de-identified under HIPAA Safe Harbor. No re-identification attempted.
2. **Bias:** Single-institution data, NLP-mined labels (~10% noise), no subgroup fairness analysis yet.
3. **Misuse risk:** Dev-mode models are not clinically deployable. No FDA/CE regulatory approval.
4. **Explainability for trust:** XAI methods revealed model attention patterns, identified failure modes (FP attending to non-pulmonary features), and support future clinical validation.
5. **Label quality:** NLP-mined labels set a performance ceiling. Expert-annotated validation subsets would strengthen future work.

---

## 11. Conclusion

This project demonstrates a unified, reproducible pipeline for multi-architecture chest X-ray classification with integrated explainability. Key contributions:

1. **Controlled comparison** of 4 architectures (3 CNNs + 1 Transformer) under identical conditions
2. **Four XAI methods** applied systematically across all models
3. **Patient-wise splitting** preventing data leakage — critical for medical AI reproducibility
4. **Honest reporting** of limitations including class imbalance, NLP label noise, and dev-mode constraints

Future work includes full-data training, official test-set evaluation, subgroup fairness analysis, external validation, and clinician user studies for XAI utility assessment.

---

## References

1. Rajpurkar, P., et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.
2. Rajpurkar, P., et al. (2018). CheXNeXt: Deep-Learning-Powered Radiograph Diagnosis.
3. Rahman, T., et al. (2021). Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection.
4. Guler, O. & Polat, K. (2021). Xception Networks for Thorax Disease Classification.
5. Singh, S., et al. (2022). Assessment of CXR Pathology Detection Using Vision Transformers.
6. Wang, X., et al. (2017). ChestX-ray8: Hospital-scale CXR Database and Benchmarks.
7. Selvaraju, R.R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks.
8. Ribeiro, M.T., et al. (2016). Why Should I Trust You? Explaining Predictions of Any Classifier.
9. Lundberg, S.M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions.
10. Sundararajan, M., et al. (2017). Axiomatic Attribution for Deep Networks.
