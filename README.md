# Team505 — Phase 2: Pneumonia Detection from ChestX-ray14

## DSAI 305 — Healthcare XAI Project

This repository contains the Phase 2 deliverables for our pneumonia detection project using the [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) dataset. The goal is to build a unified, controlled pipeline with patient-wise data splitting, multi-architecture comparison, and multi-method explainability (XAI).

---

## Phase 2 Deliverable Status

| Deliverable | Status |
|---|---|
| Shared preprocessing notebook | ✅ Complete |
| Shared EDA notebook | ✅ Complete |
| Individual model notebooks (×4) | ✅ Complete (DEV mode) |
| Preliminary training results | ✅ Complete |
| XAI implementation (4 methods) | ✅ Complete |
| XAI output generation | ✅ Complete (Grad-CAM, LIME, SHAP, IG for all 4 models) |
| Model comparison | ✅ Complete |
| XAI comparison | ✅ Complete |
| Ethics / legal analysis | ✅ Complete |
| Paper draft structure | ✅ Complete |
| Presentation outline | ✅ Complete |
| Progress report (PDF) | ✅ Present |

---

## Repository Layout

```
Team505_phase2/
├── README.md                  ← You are here
├── requirements.txt           ← Python dependencies
├── .gitignore
│
├── data/
│   ├── metadata/              ← ChestX-ray14 CSV files (not tracked)
│   └── splits/                ← Train / val / test patient-wise splits
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv           ← Locked — not used in dev mode
│       └── train_dev.csv      ← ~10% subset for rapid iteration
│
├── notebooks/
│   ├── Team505_Preprocessing.ipynb            ← Shared preprocessing pipeline
│   ├── Team505_EDA.ipynb                      ← Shared exploratory data analysis
│   ├── Ammar_Ahmed_DenseNet121.ipynb          ← Ammar — DenseNet-121
│   ├── Hosam_Nabil_DenseNet201.ipynb          ← Hosam — DenseNet-201
│   ├── Mohamed_Eslam_Xception.ipynb           ← Mohamed — Xception
│   └── Abdelrahman_Mostafa_ViTB16.ipynb       ← Abdelrahman — ViT-B/16
│
├── outputs/
│   ├── eda/                   ← EDA charts (8 PNGs)
│   ├── preprocessing/         ← Preprocessing artifacts
│   ├── Ammar_Ahmed/           ← DenseNet-121 outputs
│   │   ├── best_model.pth
│   │   ├── training_curves.png
│   │   ├── confusion_matrix.png
│   │   ├── training_history.csv
│   │   ├── validation_metrics.csv
│   │   └── xai/               ← XAI visualizations
│   │       ├── gradcam/       ← 8 images (TP/TN/FP/FN × 2)
│   │       ├── lime/          ← 8 images
│   │       └── integrated_gradients/  ← 8 images
│   ├── Hosam_Nabil/           ← DenseNet-201 (same structure)
│   ├── Mohamed_Eslam/         ← Xception (same structure)
│   └── Abdelrahman_Mostafa/   ← ViT-B/16 (same structure)
│
├── figures/                   ← Publication-quality figures
│
├── report/
│   ├── Team505_Phase2_Report.pdf  ← Progress report
│   ├── model_comparison.md        ← 4-model metric comparison table
│   ├── xai_comparison.md          ← XAI method comparison
│   ├── key_findings.md            ← Summary of project findings
│   ├── ethics_legal.md            ← Ethical and legal considerations
│   ├── paper_draft.md             ← Full paper structure with results
│   └── presentation_outline.md   ← Slide-ready presentation structure
│
├── scripts/
│   └── run_xai_demo.py       ← XAI pipeline runner (GPU)
│
└── src/                       ← Shared reusable Python modules
    ├── data/                  ← Metadata loading, dataset utilities
    ├── features/              ← Image feature engineering
    ├── models/                ← Model factory functions
    ├── xai/                   ← Explainability methods
    │   ├── __init__.py
    │   ├── gradcam.py         ← Grad-CAM implementation
    │   ├── lime_explainer.py  ← LIME implementation
    │   ├── shap_explainer.py  ← SHAP (PartitionExplainer) implementation
    │   └── integrated_gradients.py  ← Integrated Gradients (Captum)
    └── utils/                 ← Config, metrics, plotting, seeding
```

---

## Team Members — Approved Paper / Model Mapping

| Member | Model | Params | Approved Paper | Notebook |
|---|---|---|---|---|
| Ammar Ahmed | DenseNet-121 | 6.95M | Rajpurkar et al. / CheXNet | `Ammar_Ahmed_DenseNet121.ipynb` |
| Hosam Nabil | DenseNet-201 | 18.09M | Rahman et al. | `Hosam_Nabil_DenseNet201.ipynb` |
| Mohamed Eslam | Xception | 20.81M | Güler & Polat | `Mohamed_Eslam_Xception.ipynb` |
| Abdelrahman Mostafa | ViT-B/16 | 85.80M | Singh et al. | `Abdelrahman_Mostafa_ViTB16.ipynb` |

---

## Preliminary Results (DEV Mode)

| Model | ROC-AUC | F1 | Precision | Recall | Accuracy |
|-------|:-------:|:--:|:---------:|:------:|:--------:|
| DenseNet-121 | 0.608 | **0.208** | **0.135** | 0.455 | **0.690** |
| DenseNet-201 | 0.605 | 0.192 | 0.110 | **0.773** | 0.417 |
| Xception | **0.616** | 0.205 | 0.128 | 0.526 | 0.636 |
| ViT-B/16 | 0.577 | 0.198 | 0.122 | 0.526 | 0.619 |

All models trained on `train_dev.csv` (~6K images). See `report/model_comparison.md` for details.

---

## XAI Methods

Four explainability methods are implemented in `src/xai/`:

| Method | Type | Output |
|--------|------|--------|
| **Grad-CAM** | Gradient-based | Heatmap overlays |
| **LIME** | Perturbation-based | Superpixel regions |
| **SHAP** | Shapley-value | Pixel attribution maps |
| **Integrated Gradients** | Path-integration | Pixel attribution maps |

XAI outputs are organized under `outputs/<member>/xai/<method>/` with 8 images per method (2 TP, 2 TN, 2 FP, 2 FN).

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training (already done)
Open any model notebook in `notebooks/` with Jupyter. Set `RUN_MODE = "dev"` for quick training or `"full"` for final results.

### XAI Pipeline
```bash
cd Team505_phase2
.venv\Scripts\python.exe scripts\run_xai_demo.py
```

This script:
1. Loads each trained model checkpoint sequentially
2. Selects a shared evaluation subset (TP/TN/FP/FN from validation set)
3. Runs Grad-CAM, LIME, SHAP, and Integrated Gradients
4. Saves visualizations to `outputs/<member>/xai/<method>/`

**Resource notes:**
- Requires GPU for reasonable execution time (~15 min total)
- Processes one model at a time to manage memory
- SHAP uses PartitionExplainer which requires multiple forward passes per image

---

## Notes

- **Patient-wise splitting** is enforced — no patient appears in more than one split
- Raw images are **not** stored in this repo — see `data/metadata/` for CSV references
- Large model checkpoints (`.pth` files) should be listed in `.gitignore`
- All models use `RANDOM_SEED = 42` for reproducibility
- The official test split is **locked** and not used during development
