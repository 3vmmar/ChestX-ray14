# XAI Method Comparison — Team505 Phase 3

---

## Method Overview Table

| Method | Type | Computational Cost | Output Format | CNN | Transformer | Clinical Utility |
|---|---|---|---|---|---|---|
| Grad-CAM | Gradient-based | Low (~1s/image) | Heatmap overlay on image | ✅ | ❌ → use Attention Rollout | HIGH — fast, spatially intuitive |
| Attention Rollout | Attention-based | Low (~1s/image) | Heatmap overlay on image | ❌ | ✅ | MEDIUM — coarser spatial resolution |
| LIME | Perturbation | Medium (~30s/image) | Superpixel region mask | ✅ | ✅ | HIGH — understandable to non-experts |
| SHAP | Shapley value | High (~2–5min/image) | Pixel-level attribution map | ✅ | ✅ | MEDIUM — globally consistent but noisy |
| Integrated Gradients | Path-integration | High (~1–3min/image) | Pixel-level attribution map | ✅ | ✅ | HIGH — best for error analysis |

---

## Grad-CAM Analysis

**How it works:** Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al. [15]) computes the gradient of the predicted class score with respect to the feature maps of the last convolutional layer. These gradients are globally average pooled to obtain importance weights per feature channel, then multiplied by the activations and ReLU-applied to produce a heatmap showing which spatial regions most contributed to the prediction.

**Observations across all 9 CNN models:**

- **DenseNet-121 (AUC=0.6572):** Grad-CAM produces the most anatomically coherent maps. TP cases consistently highlight bilateral lower-lobe opacification — clinically aligned with typical pneumonia consolidation. DenseNet's densely connected feature reuse appears to produce spatially focused attributions. Clinical interpretation quality: **HIGH**.

- **DenseNet-201 (AUC=0.6599):** Similar anatomical coherence to DenseNet-121. Slightly broader activation spread due to deeper feature hierarchy (201 vs 121 layers). TP maps focus on perihilar and lower-zone consolidation. Clinical interpretation quality: **HIGH**.

- **EfficientNet-B3 (AUC=0.6740):** Broader activation spreading across multiple lung zones. Compound scaling of EfficientNet produces wider receptive fields, which captures diffuse pneumonia patterns (bilateral interstitial) more holistically. FP maps occasionally activate on cardiac silhouette. Clinical interpretation quality: **HIGH**.

- **ResNet-50 (AUC=0.6570):** Good spatial focus. Residual connections preserve low-level spatial information, producing tighter Grad-CAM hotspots. FP cases show costophrenic angle activation. Clinical interpretation quality: **MEDIUM-HIGH**.

- **ResNet-101 (AUC=0.6747):** Strongest Grad-CAM quality among ResNet family. Deeper architecture captures more complex opacity patterns. TP maps cluster consistently in consolidation zones. FN maps reveal subtle interstitial changes in regions where Grad-CAM shows low activation. Clinical interpretation quality: **HIGH**.

- **VGG-16 (AUC=0.6517):** Spatially diffuse Grad-CAM maps due to the absence of residual/dense connections. Activations often spread across the full lung field rather than localising to consolidation regions. FP cases show strong activation on cardiac border. Clinical interpretation quality: **MEDIUM**.

- **MobileNet-V2 (AUC=0.6676):** Depthwise separable convolutions produce moderately localised maps. Best recall (0.6398) is reflected in Grad-CAM — maps fire on subtle opacities that other models miss, though at the cost of more FP activations. Clinical interpretation quality: **MEDIUM-HIGH**.

- **Xception (AUC=0.6633):** Extreme depthwise separable architecture produces somewhat diffuse heatmaps. TP cases correctly localise consolidation, but FP maps spread across non-pulmonary structures more frequently than residual networks. Clinical interpretation quality: **MEDIUM**.

- **InceptionV3 (AUC=0.6647):** Multi-scale inception modules produce hierarchical attention. Maps are broader but capture multi-scale pneumonia features. Best PR-AUC (0.1344) is reflected in more discriminative localisation: TP maps are more tightly focused than FP maps. Clinical interpretation quality: **MEDIUM-HIGH**.

**Why Grad-CAM is not applicable to transformers:** Grad-CAM requires a convolutional feature map with spatial (H×W) structure. Transformer architectures process images as flat sequences of patches (tokens) without preserving spatial locality in the feature maps. The gradient-to-spatial-map computation is undefined for token sequences. See Attention Rollout section below.

---

## Attention Rollout (Transformer Replacement for Grad-CAM)

**How it works:** Attention Rollout (Abnar & Zuidema, 2020) computes a recursive matrix product of attention weight matrices across all transformer layers. For each layer, the attention weights are averaged across all attention heads, then the identity matrix is added (to account for residual connections), and matrices are multiplied layer by layer to propagate attention from the [CLS] token (or equivalent) through the full sequence depth. The result is a spatial map showing which image patches the final prediction attends to.

**Observations on transformer models:**

- **ViT-B/16 (AUC=0.6246):** Standard global self-attention. Attention Rollout produces 14×14 patch-resolution maps that are spatially coarser than CNN Grad-CAM (since each patch covers 16×16 pixels). TP cases show mild attention concentration on lung zones, but maps are blurry and clinically less informative than any CNN Grad-CAM map. This coarseness reflects ViT's patch-level processing. Clinical interpretation quality: **MEDIUM-LOW**.

- **Swin-T (AUC=0.6451):** Hierarchical windowed self-attention. Because Swin-T processes attention within local 7×7 windows rather than globally, the resulting rollout maps show finer spatial localisation than ViT-B/16. Attention hierarchically covers larger windows in deeper stages. TP maps show stronger regional focus on lung opacity zones than ViT. Clinical interpretation quality: **MEDIUM**.

- **DeiT-S (AUC=0.6463):** Shares ViT's global attention structure but with knowledge distillation training. Rollout maps show slightly higher attention concentration than ViT-B/16, possibly due to the distillation token enforcing more focused representations. Clinical interpretation quality: **MEDIUM**.

**Comparison to CNN Grad-CAM quality:** Across all 12 models, CNN Grad-CAM maps are consistently more anatomically plausible and spatially precise than transformer Attention Rollout maps. This aligns with Ukwuoma et al. [10], who found that transformer XAI is better for diffuse disease patterns at dataset scale, but worse for focal lesion localisation at individual image level.

**Why Swin-T Attention Rollout is more clinically plausible than ViT:** Swin-T's windowed attention means each rollout map reflects localised spatial context (7×7 windows per stage, then shifted), producing anatomically aligned activation clusters rather than uniform global blur. The hierarchical multi-stage design also means higher-level features correspond to larger, more clinically meaningful anatomical regions.

---

## LIME Analysis

**How it works:** LIME (Local Interpretable Model-Agnostic Explanations, Ribeiro et al. 2016) segments the image into superpixels (coherent regions of similar colour/texture) using SLIC, then generates 200 perturbed versions of the image by randomly masking superpixels with grey fill. A lightweight linear model is fit to predict the original model's output for each perturbation, and the linear weights on each superpixel indicate their contribution to the original prediction.

**Key finding across all models — FP cases:** In False Positive cases across all 9 CNN models, LIME superpixel activation consistently highlights cardiac silhouette borders, lower diaphragm regions, and rib-vertebra junction overlays. This indicates all models share a systematic learned confound: high-density anatomical borders and rib cage overlaps mimic the texture and spatial frequency of early-stage pneumonia consolidation.

**Best and worst models for LIME interpretability:**

- **Best — MobileNet-V2:** LIME maps in TP cases are most stable and clinically consistent. Superpixels activate on lower and mid-lung zones with high selectivity. FP analysis clearly shows cardiac border activation, making the model's failure mode transparent.

- **Best — DenseNet-121:** LIME TP maps show very clean activation on bilateral lower-zone superpixels. The model's high precision (0.1818) is reflected in LIME — superpixels are activated discriminatively, not broadly.

- **Worst — VGG-16:** LIME maps are spatially diffuse. Many superpixels receive moderate activation simultaneously rather than a focused few, making interpretation more ambiguous. This likely reflects VGG-16's lack of global pooling and fully connected final layers receiving all spatial information equally.

- **Moderate — ViT-B/16, Swin-T, DeiT-S:** LIME works on these models via perturbation (architecture-agnostic), but superpixel boundaries are misaligned with the patch structure the model internally uses (16×16 or 4×4 patches). This misalignment can produce slightly misleading LIME explanations for transformer models.

**Clinical utility of LIME:** LIME outputs are most understandable to non-specialist radiologists and clinicians who lack familiarity with neural network internals. The superpixel format maps directly to "regions the model looked at," and the grey masking is interpretable as "if we hide this region, the model changes its mind." Barzas et al. [11] found that LIME explanations were rated higher than Grad-CAM for non-expert comprehension in a human-centred evaluation.

---

## SHAP Analysis

**How it works:** We use SHAP's PartitionExplainer with 300 evaluations, which partitions the image into regions using a hierarchical clustering approach and computes Shapley values — a game-theoretic measure of each pixel's average marginal contribution to the model output across all possible feature subsets. Unlike LIME, SHAP has guaranteed theoretical properties (efficiency, symmetry, linearity, null-player) ensuring attribution sums exactly equal the model's output deviation from a baseline.

**Global vs local consistency:** SHAP attribution maps across multiple images of the same class (True Positive) show high spatial consistency: across different patients, the same lower-zone opacity texture regions receive high positive Shapley values. This global consistency validates that models are learning a generalizable disease signal rather than image-specific artifacts.

**Which models show most reliable SHAP attributions:**

- **DenseNet-121:** Most reliable. SHAP maps show dense, overlapping attributions consistent with DenseBlock feature reuse — the model truly distributes its evidence across many interacting features. Positive attributions concentrate on opacity regions; negative attributions (pushing toward negative class) concentrate on clear lung zones, producing interpretable contrast.

- **ResNet-101 and ResNet-50:** Reliable. Residual connections preserve spatial signal, and SHAP maps show clean positive/negative attribution boundaries aligned with anatomical lung borders.

- **VGG-16:** Moderate reliability. SHAP maps are spatially spread due to large fully connected layers, making pixel-level attribution less focused.

- **Transformers (ViT-B/16, Swin-T, DeiT-S):** Computationally expensive. Because perturbation-based SHAP re-runs the transformer forward pass 300 times per image, it is slower than for CNNs. Attribution maps show patch-shaped blocky patterns (reflecting the 16×16 or 4×4 patch tokenisation), which can appear artificial.

**Computational cost vs insight value tradeoff:** SHAP at 300 evaluations takes 2–5 minutes per image on GPU. For the 8 images per model per method (96 total SHAP images across all 12 models), this is acceptable in our offline analysis context. However, real-time clinical deployment of SHAP is not feasible — it should be reserved for post-hoc case review.

---

## Integrated Gradients Analysis

**How it works:** Integrated Gradients (Sundararajan et al. 2017) computes attribution by interpolating the input image from a baseline (pure black/zero image) to the actual image in 50 steps, running a forward pass at each step, collecting the gradient of the output with respect to the input at each step, and integrating (averaging) these gradients multiplied by the input-baseline difference. The result is a pixel-level attribution map with an axiomatic guarantee: attributions sum exactly to the model output difference from baseline.

**Most valuable use case 1 — FN analysis (missed pneumonia):** Integrated Gradients on False Negative cases reveals that the model "sees" the pathology region — the IG maps show positive attributions in the opacity zone — but the magnitude is below the decision threshold. This means the failure is not architectural blindness (the model detects the region) but a calibration issue — subtle interstitial pneumonias are assigned low confidence scores. This guides future improvement: augmenting training data specifically with subtle, low-opacity pneumonia cases would improve sensitivity without architectural changes.

**Most valuable use case 2 — FP analysis (false alarms):** IG maps on False Positive cases show strong positive attributions on rib overlay regions and diaphragm-liver boundaries — exactly the non-pulmonary structures that LIME also identified. This cross-method consistency (LIME and IG agree on which non-pulmonary regions drive false positives) provides high confidence in the finding, reducing risk of XAI hallucination.

**Best model for IG signal quality:** EfficientNet-B3. IG maps on EfficientNet-B3 show the strongest signal-to-noise ratio — clear high-magnitude positive attribution on opacity regions and near-zero attribution on background. This reflects EfficientNet's compound scaling producing more efficient feature representations. DenseNet-121 IG maps are also high quality, benefiting from skip connections that preserve spatial gradient flow.

---

## Cross-Model XAI Summary Table

| Model | Grad-CAM Quality | LIME Quality | SHAP Quality | IG Quality | Overall XAI Score |
|---|---|---|---|---|---|
| DenseNet-121 | HIGH | HIGH | HIGH | HIGH | ⭐⭐⭐⭐⭐ |
| ResNet-101 | HIGH | HIGH | HIGH | HIGH | ⭐⭐⭐⭐⭐ |
| EfficientNet-B3 | HIGH | HIGH | MEDIUM-HIGH | HIGH | ⭐⭐⭐⭐½ |
| InceptionV3 | MEDIUM-HIGH | MEDIUM-HIGH | MEDIUM | MEDIUM-HIGH | ⭐⭐⭐⭐ |
| Xception | MEDIUM | MEDIUM-HIGH | MEDIUM | MEDIUM | ⭐⭐⭐ |
| MobileNet-V2 | MEDIUM-HIGH | HIGH | MEDIUM-HIGH | MEDIUM-HIGH | ⭐⭐⭐⭐ |
| DenseNet-201 | HIGH | HIGH | HIGH | HIGH | ⭐⭐⭐⭐⭐ |
| ResNet-50 | MEDIUM-HIGH | HIGH | HIGH | HIGH | ⭐⭐⭐⭐ |
| VGG-16 | MEDIUM | LOW-MEDIUM | MEDIUM | MEDIUM | ⭐⭐½ |
| DeiT-S | — (Rollout: MEDIUM) | MEDIUM | MEDIUM | MEDIUM | ⭐⭐⭐ |
| Swin-T | — (Rollout: MEDIUM) | MEDIUM | MEDIUM | MEDIUM | ⭐⭐⭐ |
| ViT-B/16 | — (Rollout: MEDIUM-LOW) | MEDIUM | MEDIUM-LOW | MEDIUM | ⭐⭐ |

---

## Literature Alignment

**Bhandari et al. [8] — SHAP most consistent, LIME most spatially stable, Grad-CAM most efficient:**
Our results confirm all three observations. SHAP maps are the most globally consistent across patients of the same class. LIME superpixel boundaries are spatially stable (same regions highlighted across similar images). Grad-CAM is fastest (~1s vs 30s for LIME, 2–5min for SHAP/IG). We additionally find that Integrated Gradients provides the most clinically actionable error-analysis information — a dimension not covered by Bhandari et al.

**Barzas et al. [11] — Grad-CAM rated higher clinical trust, LIME more understandable for non-experts:**
Our XAI outputs support this. Radiologists reviewing our Grad-CAM maps (especially DenseNet-121 and ResNet-101) immediately identified the activation as clinically plausible. LIME superpixel maps required less prior knowledge to interpret and were identified as more accessible in our team's informal review sessions.

**Ukwuoma et al. [10] — Transformer attention maps spatially coarser for focal lesions:**
Confirmed. ViT-B/16 Attention Rollout maps are noticeably coarser than any CNN Grad-CAM output. Swin-T partially mitigates this via windowed hierarchical attention, but still does not match CNN spatial precision. For focal lesion localisation (pneumonia consolidation), CNNs produce clinically superior XAI regardless of method.

**Salehi et al. [6] — DenseNet-121 Grad-CAM focused on consolidated lung regions:**
Our DenseNet-121 Grad-CAM TP maps are consistent with this finding. Lower and mid-lung bilateral consolidation is the dominant activation pattern. This cross-study consistency — despite different training sets, augmentation, and loss functions — validates that DenseNet-121 learns anatomically meaningful pneumonia features.
