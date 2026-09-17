# Explainable Pneumonia Detection on ChestX-ray14

---

## Dataset
- **Source:** NIH ChestX-ray14 (Wang et al., 2017)
- **Task:** Binary pneumonia detection (Pneumonia vs No Finding)
- **Total images:** 112,120
- **Patient-wise split:** enforced — zero leakage

| Split | File | Rows | Positives | Negatives | Ratio |
|---|---|---|---|---|---|
| Training | train.csv | 11,943 | 1,004 | 10,939 | ~1:11 |
| Validation | val.csv | 2,611 | 186 | 2,425 | ~1:13 |
| Test | test.csv | 2,618 | 241 | 2,377 | ~1:10 |

---

## Repository Structure
```
├── data/
│   ├── splits/          ← train/val/test CSVs
│   ├── metadata/        ← dataset_summary.json, master_registry.csv
│   └── archive/         ← NIH images + Data_Entry_2017.csv
├── notebooks/           ← 12 model notebooks + EDA + Preprocessing
├── outputs/
│   ├── models/          ← best_model.pth, metrics, XAI images per model
│   ├── eda/             ← EDA visualisations
│   └── preprocessing/   ← preprocessing outputs
├── report/              ← all documentation files
├── scripts/
│   ├── rebuild_clean_splits.py
│   ├── patch_notebooks.py
│   └── run_xai_demo.py
├── src/
│   └── xai/             ← gradcam.py, lime_explainer.py, shap_explainer.py, integrated_gradients.py
└── requirements.txt     ← project dependencies
```

---

## How to Reproduce

### 1. Rebuild data splits
```bash
python scripts/rebuild_clean_splits.py
```

### 2. Train a model
Open any notebook in `notebooks/`.
Set `RUN_MODE = "full"` and run all cells.

### 3. Run XAI pipeline
```bash
python scripts/run_xai_demo.py
```

---

## Training Pipeline — Key Decisions
| Decision | Choice | Reason |
|---|---|---|
| Loss function | Focal Loss (α=0.75, γ=2.0, smoothing=0.05) | Fixes calibration collapse from extreme imbalance |
| Batch sampling | WeightedRandomSampler (20% pos/batch) | Stabilises gradient updates |
| Training data | NIH-only positives, No Finding negatives | External data caused domain shift failure |
| Unfreeze strategy | Progressive: head → partial → full | Prevents destroying pretrained features |
| Overfitting defence | Dropout(0.6) + Mixup(α=0.3) + WD=5e-3 | Small positive count (1,004) vs 7–140M params |
| Evaluation | TTA (5 variants) on val.csv | Robust evaluation on held-out patient split |

---

## Final Results — All 12 Models
**Evaluation:** TTA with 5 variants on val.csv (2,611 rows, 186 positives)
TTA variants: original, horizontal flip, +7° rotation, -7° rotation, brightness+0.15

| Rank | Model | Params | AUC | PR-AUC | F1 | Precision | Recall | Threshold | Best Epoch |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ResNet-101 | 44.5M | 0.6747 | 0.1248 | 0.2088 | 0.1437 | 0.3817 | 0.41 | 9 |
| 2 | EfficientNet-B3 | 12.2M | 0.6740 | 0.1217 | 0.2198 | 0.1375 | 0.5484 | 0.41 | 29 |
| 3 | MobileNet-V2 | 3.4M | 0.6676 | 0.1218 | 0.2062 | 0.1229 | 0.6398 | 0.37 | 22 |
| 4 | InceptionV3 | 23.8M | 0.6647 | 0.1344 | 0.2056 | 0.1708 | 0.2581 | 0.46 | 15 |
| 5 | Xception | 20.8M | 0.6633 | 0.1160 | 0.2005 | 0.1280 | 0.4624 | 0.31 | 19 |
| 6 | DenseNet-201 | 18.1M | 0.6599 | 0.1186 | 0.2055 | 0.1361 | 0.4194 | 0.42 | 12 |
| 7 | DenseNet-121 | 6.95M | 0.6572 | 0.1230 | 0.2014 | 0.1818 | 0.2258 | 0.38 | 40 |
| 8 | ResNet-50 | 25.6M | 0.6570 | 0.1216 | 0.2072 | 0.1311 | 0.4946 | 0.35 | 12 |
| 9 | VGG-16 | 138M | 0.6517 | 0.1126 | 0.2024 | 0.1218 | 0.5968 | 0.35 | 16 |
| 10 | DeiT-S | 22M | 0.6463 | 0.1099 | 0.1931 | 0.1342 | 0.3441 | 0.43 | 14 |
| 11 | Swin-T | 28M | 0.6451 | 0.1095 | 0.2006 | 0.1227 | 0.5484 | 0.42 | 15 |
| 12 | ViT-B/16 | 85.8M | 0.6246 | 0.1054 | 0.1845 | 0.1153 | 0.4624 | 0.48 | 13 |

---

## XAI Methods
| Method | Type | Cost | CNN | Transformer |
|---|---|---|---|---|
| Grad-CAM | Gradient-based heatmap | Low | ✅ | ❌ → Attention Rollout |
| LIME | Perturbation superpixels | Medium | ✅ | ✅ |
| SHAP | Shapley value attribution | High | ✅ | ✅ |
| Integrated Gradients | Path-integration | High | ✅ | ✅ |

XAI outputs: `outputs/models/<model>/xai/<method>/`
8 images per method per model (2×TP, 2×TN, 2×FP, 2×FN)

---

## References
[1] Rajpurkar et al. (2018) — CheXNeXt, PLOS Medicine
[2] Stephen et al. (2019) — Custom CNN, J. Healthcare Eng.
[3] Rahman et al. (2020) — Transfer learning, Applied Sciences
[4] Chowdhury et al. (2020) — COVID+Pneumonia, IEEE Access
[5] Hashmi et al. (2020) — Weighted ensemble, Diagnostics
[6] Salehi et al. (2021) — Grad-CAM comparison, Br. J. Radiol.
[7] Kundu et al. (2021) — Ensemble, PLOS ONE
[8] Bhandari et al. (2022) — SHAP+LIME+GradCAM, Comput. Biol. Med.
[9] Guler & Polat (2022) — Xception, J. Artif. Intell. Syst.
[10] Ukwuoma et al. (2023) — Hybrid transformer, J. Adv. Res.
[11] Barzas et al. (2024) — Human-centered XAI, PLOS ONE
[12] Singh et al. (2024) — ViT for CXR, Sci. Reports
[13] Wang et al. (2017) — NIH ChestX-ray14 dataset, CVPR
[14] Lin et al. (2017) — Focal Loss, ICCV
[15] Selvaraju et al. (2017) — Grad-CAM, ICCV
