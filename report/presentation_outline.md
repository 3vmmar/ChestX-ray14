# Presentation Outline — Team505 Phase 3
## DSAI 305 | Spring 2026 | 15–20 Minute Presentation

---

## SLIDE 1 — Title Slide

**Display:**
> # Explainable Pneumonia Detection
> ### A Comparison of 12 Deep Learning Models on NIH ChestX-ray14
>
> **Team 505 | DSAI 305 | Spring 2026**
>
> | Member | ID |
> |---|---|
> | Ammar Ahmed | 202300877 |
> | Hosam Nabil | 202202228 |
> | Mohamed Eslam | 202201690 |
> | Abdelrahman Mostafa | 202202298 |

**Speaker notes:** Good [morning/afternoon]. We are Team 505. Our project is a controlled comparison of 12 deep learning architectures for pneumonia detection on chest X-rays, with explainability analysis using four different methods. I will walk you through our problem, what we built, what we found, and what it means clinically in the next 15–18 minutes.

---

## SLIDE 2 — Problem Statement and Motivation

**Display:**
- Pneumonia: >2.5M deaths/year globally (WHO 2022)
- CXR: Primary diagnostic modality in clinical practice
- AI models: AUC 0.65–0.77 on NIH benchmark [Rajpurkar 2018]
- **Gap 1:** No controlled multi-architecture comparison under identical conditions
- **Gap 2:** Explainability treated as optional afterthought, not safety requirement
- **Our question:** Which architectures work best, and can we trust them?

**Speaker notes:** Pneumonia remains one of the leading infectious killers globally. Chest X-ray is the front line. AI models now approach radiologist performance on benchmarks — but published papers compare architectures under different conditions, making comparisons meaningless. More critically, none systematically evaluate explainability alongside performance. Our work addresses both gaps.

---

## SLIDE 3 — Research Questions and Contributions

**Display:**
**Research Questions:**
1. Do vision transformers outperform CNNs for NIH binary pneumonia detection?
2. Do different XAI methods give complementary or redundant clinical information?
3. What is the primary performance-limiting factor: architecture, training, or data?

**6 Contributions:**
1. First controlled comparison of 9 CNNs + 3 transformers under identical protocol
2. Evidence that NIH label noise, not architecture, is the performance ceiling
3. Focal Loss shown to resolve threshold collapse under extreme imbalance
4. Systematic 4-method XAI evaluation revealing consistent FP confounds across all models
5. TTA quantified as variance reduction, not amplifier
6. Ethical analysis grounded in model-specific recall numbers

**Speaker notes:** Three clear research questions. Six concrete contributions — I want to highlight especially #4: our XAI analysis found something that AUC metrics would never show you — a consistent systematic error across all 12 models, which I will explain in the XAI slides.

---

## SLIDE 4 — Dataset Overview

**Display:**
**NIH ChestX-ray14 (Wang et al., 2017)**
- 112,120 frontal CXR images
- 30,805 unique patients
- Labels: NLP-extracted from radiology reports (~30-40% estimated noise on Pneumonia)

**Our Task: Binary Pneumonia Detection**

| Split | Rows | Positives | Negatives | Ratio |
|---|---|---|---|---|
| Training | 11,943 | 1,004 | 10,939 | 1:11 |
| Validation | 2,611 | 186 | 2,425 | 1:13 |
| Test | 2,618 | 241 | 2,377 | 1:10 |

⚠️ **Key constraint:** Patient-wise splitting — zero data leakage

**Speaker notes:** We use only No Finding and Pneumonia images — clean binary task. Important to note: the ratio is about 1 positive for every 11 negatives in training. This extreme imbalance is one of our core challenges. Patient-wise splitting means no patient appears in more than one split — preventing the artificial AUC inflation that many published studies suffer from.

---

## SLIDE 5 — Data Pipeline

**Display:**
```
NIH Raw Images (12 batches)
         ↓
Master Registry Construction
  • All 112K images catalogued
  • Patient IDs assigned
  • No-Finding + Pneumonia filtered
         ↓
Patient-wise 70/15/15 Split (seed=42)
         ↓
train.csv | val.csv | test.csv
         ↓
Reproducible via: python scripts/rebuild_clean_splits.py
```

**Key decisions:**
- Excluded other diseases from evaluation (contamination removal)
- No external Kermany data — domain shift failure (AUC dropped to 0.579)
- NIH-only positives — radiologically consistent with adult pneumonia

**Speaker notes:** Two critical decisions here. First, we excluded other diseases from evaluation — an atelectasis labeled as 0 is not a true negative for pneumonia detection, and including it inflates AUC while suppressing F1. Second — and this is important — we tried adding 8,400 external Kermany positives. AUC collapsed to 0.58 because those are pediatric obvious bilateral pneumonias, completely different from NIH adult subtle infiltrates. NIH-only training recovered performance.

---

## SLIDE 6 — Preprocessing and Augmentation

**Display:**
**Input:** Raw CXR PNG → 224×224 pixels, ImageNet normalisation

**Augmentation Pipeline (9 steps):**
| Step | Transform | Parameter |
|---|---|---|
| 1 | Random horizontal flip | p=0.5 |
| 2 | Random rotation | ±10° |
| 3 | Brightness/contrast | ±0.15 |
| 4 | Gaussian blur | p=0.3 |
| 5 | CLAHE | clip=2.0, grid=8×8 |
| 6 | Coarse dropout | p=0.2 |
| 7 | GridDistort | p=0.3 |
| 8 | Mixup | α=0.3 (batch) |
| 9 | Label smoothing | ε=0.05 |

**Speaker notes:** CLAHE — Contrast Limited Adaptive Histogram Equalisation — is particularly important for CXR. It enhances local contrast in low-density lung regions where subtle pneumonia infiltrates are visible, making them more detectable during training. Mixup combines two training images linearly and mixes their labels — prevents overconfident predictions on our small 1,004-positive training set. Label smoothing (0.05) prevents the model from becoming pathologically overconfident on noisy NIH labels.

---

## SLIDE 7 — Training Pipeline

**Display:**

**Problem:** BCE loss + 13:1 imbalance → threshold = 0.90 (model says "no pneumonia" for everything)

**Solution: Focal Loss [Lin 2017]**
- α=0.75 (upweights positive class)
- γ=2.0 (downweights easy examples)
- Result: Thresholds move to 0.31–0.48 range ✅

**Progressive Unfreezing — 3 Stages:**
```
Stage A (5 epochs):   [FROZEN backbone] → [HEAD only]       LR=1e-4
Stage B (5 epochs):   [last 2 blocks]   → [partial]         LR=2e-5
Stage C (40 epochs):  [ALL layers]      → [full fine-tune]  LR=2e-5, WarmRestarts
```

**Additional:** WeightedRandomSampler (20% positive per batch) | AdamW WD=5e-3

**Speaker notes:** The threshold collapse with BCE was the biggest failure we encountered. When the loss is dominated by 10,939 negatives, the model learns to predict zero for everything and still achieves 91% accuracy. Focal Loss solved this by making easy-to-classify negatives contribute almost nothing to the gradient. Progressive unfreezing prevents destroying ImageNet-pretrained features — we first train only the head, then slowly release the backbone layer by layer.

---

## SLIDE 8 — All 12 Models Overview

**Display:**
**CNN Family (9 models):**

| Model | Architecture Type | Params |
|---|---|---|
| DenseNet-121 | Dense connectivity | 6.95M |
| DenseNet-201 | Dense connectivity | 18.1M |
| EfficientNet-B3 | Compound scaling | 12.2M |
| ResNet-50 | Residual | 25.6M |
| ResNet-101 | Residual | 44.5M |
| VGG-16 | Sequential | 138M |
| MobileNet-V2 | Depthwise separable | 3.4M |
| Xception | Extreme depthwise | 20.8M |
| InceptionV3 | Multi-scale inception | 23.8M |

**Transformer Family (3 models):**

| Model | Architecture Type | Params |
|---|---|---|
| ViT-B/16 | Global self-attention | 85.8M |
| Swin-T | Windowed hierarchical | 28M |
| DeiT-S | Distilled knowledge | 22M |

**Speaker notes:** 9 CNNs + 3 transformers = 12 total. All trained under identical protocol — same loss, same augmentation, same split. This controlled setup is what makes our comparison valid.

---

## SLIDE 9 — Results — Full Comparison Table

**Display:**

| Rank | Model | AUC | F1 | Precision | Recall | Type |
|---|---|---|---|---|---|---|
| 1 | ResNet-101 | **0.6747** | 0.2088 | 0.1437 | 0.3817 | CNN |
| 2 | EfficientNet-B3 | 0.6740 | **0.2198** | 0.1375 | 0.5484 | CNN |
| 3 | MobileNet-V2 | 0.6676 | 0.2062 | 0.1229 | **0.6398** | CNN |
| 4 | InceptionV3 | 0.6647 | 0.2056 | **0.1708** | 0.2581 | CNN |
| 5 | Xception | 0.6633 | 0.2005 | 0.1280 | 0.4624 | CNN |
| 6 | DenseNet-201 | 0.6599 | 0.2055 | 0.1361 | 0.4194 | CNN |
| 7 | DenseNet-121 | 0.6572 | 0.2014 | 0.1818 | 0.2258 | CNN |
| 8 | ResNet-50 | 0.6570 | 0.2072 | 0.1311 | 0.4946 | CNN |
| 9 | VGG-16 | 0.6517 | 0.2024 | 0.1218 | 0.5968 | CNN |
| 10 | DeiT-S | 0.6463 | 0.1931 | 0.1342 | 0.3441 | Transformer |
| 11 | Swin-T | 0.6451 | 0.2006 | 0.1227 | 0.5484 | Transformer |
| 12 | ViT-B/16 | 0.6246 | 0.1845 | 0.1153 | 0.4624 | Transformer |

*All values from TTA (5 variants) on val.csv*

**Speaker notes:** Two things stand out immediately. First: every CNN outperforms every transformer. Second: the CNN spread is tiny — 0.023 AUC from rank 1 to rank 9. VGG-16 with 138 million parameters ranks 9th, MobileNet-V2 with 3.4 million ranks 3rd. This tells us something important about the data, not the models.

---

## SLIDE 10 — Results — CNN vs Transformer

**Display:**
[Conceptual bar chart: CNN bars 0.65–0.675, Transformer bars 0.62–0.646]

| Family | Count | Mean AUC | Mean F1 |
|---|---|---|---|
| CNN | 9 | 0.6650 | 0.2061 |
| Transformer | 3 | 0.6387 | 0.1924 |
| Gap | — | **+0.026** | **+0.014** |

**Why transformers underperform:**
- ViT needs >100K images to surpass CNN [Singh 2024]
- Our training: only **1,004 positive samples**
- Transformer self-attention requires more data to learn spatial visual representations

**Cost:** ViT-B/16 — 85.8M params, 742 min training → WORST AUC (0.6246)

**Speaker notes:** The CNN vs transformer gap is 0.026 AUC. Not huge, but consistent — every single CNN beats every single transformer. The reason is data efficiency. ViT learns spatial representations from scratch through self-attention — it needs massive datasets. We have 1,004 positive training samples. CNN architectures, with their spatial inductive bias from convolutional kernels, can learn meaningful local features from this limited data far more efficiently.

---

## SLIDE 11 — Best Model Deep-Dive

**Display:**

**ResNet-101 (Best AUC = 0.6747):**
- Epoch 9/50 — early convergence
- TP=71 | FP=423 | TN=2002 | FN=115
- Threshold=0.41 (clinically usable)

**EfficientNet-B3 (Best F1 = 0.2198):**
- Epoch 29/50 — later, more stable convergence
- TP=102 | FP=640 | TN=1785 | FN=84
- Catches 102/186 positives (55% recall) — strongest for clinical screening

**MobileNet-V2 (Best Recall = 0.6398, Best Efficiency):**
- Only 3.4M parameters
- AUC=0.6676 — 3rd best overall
- Catches 119/186 positives — highest of all 12 models

**Speaker notes:** Three models worth highlighting for different reasons. ResNet-101 has the best AUC but moderate recall. EfficientNet-B3 has the best F1 — the best balance of precision and recall — making it the best for clinical screening where you want to catch positives while controlling false alarms. MobileNet-V2 is the dark horse — 3.4 million parameters, ranks 3rd overall in AUC, highest recall of all models. For a deployment where you want maximum sensitivity on minimal compute, MobileNet-V2 is the recommendation.

---

## SLIDE 12 — XAI Method Overview

**Display:**

| Method | Type | Cost | CNN | Transformer |
|---|---|---|---|---|
| Grad-CAM | Gradient-based heatmap | ~1 sec | ✅ | ❌ |
| Attention Rollout | Attention accumulation | ~1 sec | ❌ | ✅ |
| LIME | Perturbation superpixels | ~30 sec | ✅ | ✅ |
| SHAP | Shapley attribution | ~3 min | ✅ | ✅ |
| Integrated Gradients | Path-integrated gradient | ~2 min | ✅ | ✅ |

**Output per model:** 8 images per method × 4 methods = 32 explanation images
Composition: 2 TP + 2 TN + 2 FP + 2 FN per method

**Speaker notes:** We used four XAI methods. For CNN models, Grad-CAM produces a heatmap showing which spatial regions drove the prediction — very fast. For transformer models, Grad-CAM is inapplicable (no spatial feature maps), so we use Attention Rollout instead, which propagates attention weights across all transformer layers. LIME perturbs the image 200 times. SHAP runs 300 evaluations. Integrated Gradients interpolates 50 steps from black to the actual image. Each method reveals something different — this is the key message of our XAI section.

---

## SLIDE 13 — XAI Cross-Model Comparison

**Display:**

**Grad-CAM — TP case consistency (CNN models):**
- DenseNet-121, ResNet-101: Bilateral lower-lobe activation (clinically correct)
- VGG-16: Diffuse whole-lung activation (less specific)
- EfficientNet-B3: Broader activation — captures diffuse bilateral infiltrates

**Attention Rollout — TP comparison (Transformer models):**
- ViT-B/16: Coarse 14×14 patch-level attention — anatomically imprecise
- Swin-T: Finer windowed attention — more localised to lung zones
- DeiT-S: Similar to ViT but slightly more focused

**XAI Quality Summary:**
- DenseNet-121, ResNet-101, DenseNet-201: ⭐⭐⭐⭐⭐ (all 4 methods HIGH)
- EfficientNet-B3, MobileNet-V2: ⭐⭐⭐⭐ (HIGH–MED-HIGH)
- VGG-16, ViT-B/16: ⭐⭐–⭐⭐½ (worst XAI quality)

**Speaker notes:** CNN Grad-CAM maps are consistently more anatomically precise than transformer Attention Rollout maps. Swin-T's windowed attention produces finer-grained maps than ViT's global attention. The DenseNet family consistently produces the most anatomically aligned XAI across all four methods — which is part of why DenseNet-121 remains the gold standard in CXR AI [Rajpurkar 2018, Salehi 2021].

---

## SLIDE 14 — XAI Failure Analysis

**Display:**

**🔴 False Positive Analysis (all 12 models — LIME + IG agree):**
- Consistent activation on: **cardiac silhouette borders**, **rib-diaphragm junctions**, **rib overlays**
- Interpretation: Normal dense anatomical structures mimic pneumonia opacity texture
- Finding: ALL 12 models share this failure mode — it is dataset-level, not architecture-level

**🟡 False Negative Analysis (IG most informative):**
- Model "sees" the pathology region (positive IG values present)
- But assigns insufficient confidence (below threshold)
- Interpretation: Subtle interstitial pneumonias are present but under-confident
- Fix: Add more subtle-opacity training cases, not architectural changes

**Speaker notes:** This is the most important XAI finding. Across ALL 12 models — every architecture, every training run — false positives share the same cause: cardiac borders and rib overlays. This means the problem is not in the architecture; it's in the training data. Normal X-rays with dense anatomical overlaps look like pneumonia to every model. The fix is data-level, not model-level. For false negatives, Integrated Gradients shows the model does see the pathology — it just doesn't have enough confidence. More subtle-opacity training cases would fix this.

---

## SLIDE 15 — Ethical Considerations

**Display:**

**Issue 1 — Patient Privacy**
- NIH data de-identified at metadata level (not pixel level)
- Our pipeline: no PHI in outputs, no cloud transfer, no re-identification

**Issue 2 — Algorithmic Fairness**
- NIH dataset: US adult hospital population — not representative globally
- We did NOT conduct subgroup analysis (age, sex, view position) → limitation

**Issue 3 — Overreliance Risk**
- Best recall = 0.6398 (MobileNet-V2) → 36% of positives MISSED
- Clinical role: **Triage support only — mandatory radiologist review**

**Issue 4 — Regulatory Compliance**
- Not cleared for clinical use (no FDA 510(k), no CE marking)
- EU AI Act: High-risk category → conformity assessment required

**Role of XAI:** Enables human audit → reduces overreliance → satisfies EU AI Act oversight requirement

**Speaker notes:** Four clear ethical issues. The most critical for clinical safety is Issue 3: even our best recall model misses 36% of pneumonia cases. This is not acceptable as a standalone diagnostic tool. It works as a screening tool — flagging suspicious cases for priority radiologist review — but the radiologist must always have the final word. Our XAI outputs support this by showing the radiologist WHY the model made its prediction, enabling informed collaboration rather than blind trust.

---

## SLIDE 16 — Discussion and Limitations

**Display:**

**Key findings recap:**
1. NIH label noise (~30-40%) is the performance ceiling, not architecture
2. CNN-transformer gap = 0.026 AUC — consistent and data-driven
3. Focal Loss moved thresholds from 0.90 → 0.31–0.48 (clinically usable)
4. XAI revealed a cross-architecture dataset-level confound (rib/cardiac FPs)
5. TTA: highest benefit for high-variance models (DenseNet-121 +0.019 AUC)

**Limitations we acknowledge:**
- NIH pneumonia labels have 30-40% estimated noise
- Only 1,004 positive training samples
- No subgroup fairness analysis (age, sex, view position)
- Single dataset — external generalisation not validated
- XAI quality ratings are qualitative, not quantitative

**Speaker notes:** We acknowledge all limitations openly. The most fundamental is the data quality problem — NIH labels are NLP-extracted and noisy. This is not our methodological failure; it's a dataset limitation that affects every published study on this benchmark. Our strict patient-wise splitting and clean binary evaluation makes our numbers lower but more honest than most published comparisons.

---

## SLIDE 17 — Conclusion

**Display:**

**3 Key Takeaways:**

1. **Architecture choice matters less than data quality.**
   CNN spread: 0.023 AUC across 9 architectures. Label noise is the ceiling.

2. **CNNs outperform transformers on limited CXR data.**
   All 9 CNNs > all 3 transformers. ViT-B/16 (85.8M, 742 min) = worst performer.

3. **Multi-method XAI is essential, not optional.**
   Grad-CAM + LIME + SHAP + IG together revealed systematic FP confounds
   invisible to AUC alone. Explainability is a safety mechanism, not a nice-to-have.

**Future work:**
- Ensemble top 3 CNN models (expected +0.01–0.02 AUC)
- Subgroup fairness analysis (sex, age, view position)
- Semi-supervised learning with 50K+ unlabelled NIH images
- Prospective clinical pilot with radiologist XAI feedback
- Validate on CheXpert or MIMIC-CXR for cross-dataset generalisation

**Speaker notes:** Three takeaways to remember. The data limits the model — not the architecture. CNNs win on limited medical data — transformers need more. And XAI is not decoration — it revealed a finding that AUC would never show us. Thank you. We are happy to take questions.

---

## SLIDE 18 — Q&A Preparation

**Anticipated questions with prepared answers:**

---

**Q1: Why is F1 so low (0.18–0.22)?**

Answer: Two reasons. First, NIH Pneumonia labels have 30–40% estimated noise — the annotation ceiling constrains maximum achievable F1. Second, the positive rate in validation is 7.1% (186/2611), creating a precision-recall tradeoff where both are bounded by imbalance. No published study with strict patient-wise splitting on NIH binary pneumonia exceeds F1=0.25. Our F1=0.2198 (EfficientNet-B3) is competitive with the state of the art under equivalent methodology.

---

**Q2: Why not use external Kermany/RSNA data?**

Answer: We tried. Adding 8,400 Kermany positives caused AUC to drop from 0.65 to 0.579 and threshold to collapse to 0.98. The domain shift is the cause: Kermany positives are pediatric bilateral obvious consolidations; NIH positives are adult subtle interstitial infiltrates. The model learned the Kermany visual pattern, which doesn't match NIH validation. NIH-only training recovered all performance.

---

**Q3: Why is patient-wise splitting important?**

Answer: One patient can have 10–50 scans in the NIH dataset. Without patient-wise splitting, the same patient's scans appear in both training and validation. The model memorises patient-specific anatomical features (rib cage shape, implants, body habitus) rather than learning pneumonia-specific features. AUC inflates to 0.72–0.75 with random splitting but collapses to 0.62–0.67 with patient-wise splitting. Our lower numbers are more representative of true generalisation.

---

**Q4: Which XAI method would you recommend for clinical use?**

Answer: Depends on the use case. For real-time clinical deployment where a radiologist needs a quick visual check: **Grad-CAM** — 1 second per image, spatially intuitive. For error investigation and audit: **Integrated Gradients** — most informative for understanding why the model failed. For non-specialist review and bias detection: **LIME** — understandable without deep AI knowledge. For global model validation: **SHAP** — confirms the model is using clinically meaningful features consistently.

---

**Q5: Why do CNNs outperform transformers?**

Answer: Vision transformers learn global self-attention patterns without the spatial inductive bias of convolutional kernels. They require large amounts of data — published research (Singh et al. 2024) suggests >100,000 training images are needed for ViT to match CNN performance on chest pathology. We have 1,004 positive training samples. Under this data scarcity, CNN architectures with their local feature extraction inductive bias generalise far better. ViT-B/16 used 85.8M parameters, 742 minutes of training, and achieved the worst AUC of all 12 models.

---

**Q6: How would this be deployed clinically?**

Answer: As a **triage support tool only**. The model would run overnight on the queue of CXR images not yet reviewed and flag the top N% most likely positive cases for priority radiologist review the next morning. The radiologist reviews every case — they are not replaced. The AI provides Grad-CAM overlays alongside each flagged case so the radiologist can see where the model is looking. If the Grad-CAM shows activation on a rib rather than lung parenchyma, the radiologist can appropriately discount the flag. This is the human-AI collaboration model recommended by EU AI Act for high-risk medical AI systems.
