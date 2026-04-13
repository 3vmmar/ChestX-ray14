# Key Findings — Team505 Phase 2

## 1. Best-Performing Model

**Xception** achieved the highest ROC-AUC (0.6158) among all four models on the dev validation set. However, DenseNet-121 achieved the best F1-score (0.208) and highest precision (0.135), making it the most balanced classifier. The performance gap is small across all CNN models (AUC range: 0.605–0.616), suggesting that architecture choice has limited impact at this data scale.

## 2. Best-Generalizing Model

**DenseNet-121** showed the best balance between train and validation loss, with the smallest generalization gap. Its compact size (7M parameters) makes it less prone to overfitting on small datasets. This aligns with CheXNet's success — DenseNet-121's dense connections provide effective feature reuse that is well-suited for medical imaging where datasets are often limited.

## 3. CNN vs. Transformer Observations

**ViT-B/16 underperformed all CNNs** in this low-data setting (AUC=0.577 vs. 0.606–0.616 for CNNs). This is consistent with the established finding that Vision Transformers are data-hungry — they lack the inductive biases (locality, translation equivariance) that help CNNs learn from small datasets. ViT was still improving at epoch 5 (monotonically increasing AUC), suggesting it would benefit significantly from longer training on the full dataset. Its 86M parameters vs. 7-21M for CNNs also contributed to slower convergence.

## 4. Impact of Class Imbalance

The ~9% positive rate in the dev subset (1.3% in the full dataset) severely impacted all models:
- Without `pos_weight` in BCEWithLogitsLoss, models would predict all-negative (91% accuracy, 0% recall)
- Even with pos_weight=9.82, precision remained below 14% for all models
- Threshold tuning was essential — DenseNet-201 needed a 0.25 threshold to capture positive cases
- Default 0.5 threshold systematically favored the majority class

These results demonstrate that class imbalance handling must be a core part of the pipeline, not an afterthought.

## 5. What XAI Results Revealed

### Grad-CAM
- CNN models consistently attended to **lung field regions**, particularly lower lobes — anatomically appropriate for pneumonia detection
- ViT-B/16 produced **blockier** attention patterns reflecting its 16x16 patch grid
- On **true positive** cases, activation concentrated on areas consistent with infiltrate patterns
- On **false positives**, models often fixated on cardiomediastinal silhouette edges rather than true parenchymal opacities

### LIME
- Superpixel analysis showed that **all models used similar spatial regions** for prediction, despite different architectures
- Positive-contributing superpixels in TP cases typically covered opacified lung zones
- In FP cases, LIME revealed that models sometimes relied on **non-pulmonary features** (diaphragm edges, annotations) — important for clinical trust

### Integrated Gradients
- Provided the **finest-grained** attribution maps
- On TP cases, high attribution pixels clustered in anatomically relevant pulmonary regions
- On FN cases, IG revealed **what the model missed** — low attribution in areas where infiltrates were present
- This method was the most informative for **debugging model failures**

## 6. Strongest Model–Explainer Combinations

| Combination | Why |
|-------------|-----|
| **DenseNet-121 + Grad-CAM** | Clean, focused heatmaps that correlate well with known pneumonia patterns. The compact architecture produces stable, interpretable explanations. |
| **Xception + LIME** | LIME's model-agnostic approach paired with Xception's best AUC produced the most clinically useful explanations for individual case review. |
| **DenseNet-121 + IG** | Integrated Gradients on the CheXNet baseline provided fine-grained attributions ideal for understanding model failures. |
| **ViT-B/16 + Grad-CAM** | The only explainer that provided meaningful spatial localization from the transformer architecture. |

## 7. Overall Conclusions

1. **DenseNet-121 remains the gold-standard baseline** for ChestX-ray14, confirming CheXNet's foundational finding
2. **Deeper/larger models did not help** in the low-data regime — parameter efficiency matters
3. **Multi-method XAI is essential**: no single method provides complete insight
4. **Grad-CAM + Integrated Gradients** together provide the strongest explanatory framework
5. **Class imbalance is the dominant challenge**, not architecture choice
6. **Full-data training and test-set evaluation** are critical next steps before drawing final conclusions
