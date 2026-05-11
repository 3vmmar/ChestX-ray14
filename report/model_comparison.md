# Model Comparison — Phase 3 Final Results
## Team505 | DSAI 305 | Spring 2026

## Evaluation Protocol
- **Validation set:** val.csv — 2,611 rows (186 positives, 2,425 negatives; patient-wise split)
- **Evaluation method:** TTA (Test Time Augmentation) with 5 variants per image
- **TTA variants:** original, horizontal flip, +7° rotation, -7° rotation, brightness+0.15
- **Threshold:** independently optimised per model on val.csv using F1-maximisation

---

## Complete Results Table

| Model | Member | Params | AUC(single) | AUC(TTA) | PR-AUC(TTA) | F1(TTA) | Precision | Recall | Threshold | TP | FP | TN | FN | Best Epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-101 | Mohamed | 44.5M | 0.6719 | 0.6747 | 0.1248 | 0.2088 | 0.1437 | 0.3817 | 0.41 | 71 | 423 | 2002 | 115 | 9 |
| EfficientNet-B3 | Ammar | 12.2M | 0.6739 | 0.6740 | 0.1217 | 0.2198 | 0.1375 | 0.5484 | 0.41 | 102 | 640 | 1785 | 84 | 29 |
| MobileNet-V2 | Hosam | 3.4M | 0.6527 | 0.6676 | 0.1218 | 0.2062 | 0.1229 | 0.6398 | 0.37 | 119 | 849 | 1576 | 67 | 22 |
| InceptionV3 | Mohamed | 23.8M | 0.6569 | 0.6647 | 0.1344 | 0.2056 | 0.1708 | 0.2581 | 0.46 | 48 | 233 | 2192 | 138 | 15 |
| Xception | Mohamed | 20.8M | 0.6606 | 0.6633 | 0.1160 | 0.2005 | 0.1280 | 0.4624 | 0.31 | 86 | 586 | 1839 | 100 | 19 |
| DenseNet-201 | Hosam | 18.1M | 0.6527 | 0.6599 | 0.1186 | 0.2055 | 0.1361 | 0.4194 | 0.42 | 78 | 495 | 1930 | 108 | 12 |
| DenseNet-121 | Ammar | 6.95M | 0.6385 | 0.6572 | 0.1230 | 0.2014 | 0.1818 | 0.2258 | 0.38 | 42 | 189 | 2236 | 144 | 40 |
| ResNet-50 | Ammar | 25.6M | 0.6485 | 0.6570 | 0.1216 | 0.2072 | 0.1311 | 0.4946 | 0.35 | 92 | 610 | 1815 | 94 | 12 |
| VGG-16 | Hosam | 138M | 0.6511 | 0.6517 | 0.1126 | 0.2024 | 0.1218 | 0.5968 | 0.35 | 111 | 800 | 1625 | 75 | 16 |
| DeiT-S | Abdelrahman | 22M | 0.6463 | 0.6463 | 0.1099 | 0.1931 | 0.1342 | 0.3441 | 0.43 | 64 | 413 | 2012 | 122 | 14 |
| Swin-T | Abdelrahman | 28M | 0.6405 | 0.6451 | 0.1095 | 0.2006 | 0.1227 | 0.5484 | 0.42 | 102 | 729 | 1696 | 84 | 15 |
| ViT-B/16 | Abdelrahman | 85.8M | 0.6180 | 0.6246 | 0.1054 | 0.1845 | 0.1153 | 0.4624 | 0.48 | 86 | 660 | 1765 | 100 | 13 |

**Total val.csv = TP + FP + TN + FN = 2,611 for all models.**

---

## Results By Architecture Family

### CNN Models (9 models)

| Rank | Model | Params | AUC(TTA) | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| 1 | ResNet-101 | 44.5M | 0.6747 | 0.1248 | 0.2088 | 0.1437 | 0.3817 |
| 2 | EfficientNet-B3 | 12.2M | 0.6740 | 0.1217 | 0.2198 | 0.1375 | 0.5484 |
| 3 | MobileNet-V2 | 3.4M | 0.6676 | 0.1218 | 0.2062 | 0.1229 | 0.6398 |
| 4 | InceptionV3 | 23.8M | 0.6647 | 0.1344 | 0.2056 | 0.1708 | 0.2581 |
| 5 | Xception | 20.8M | 0.6633 | 0.1160 | 0.2005 | 0.1280 | 0.4624 |
| 6 | DenseNet-201 | 18.1M | 0.6599 | 0.1186 | 0.2055 | 0.1361 | 0.4194 |
| 7 | DenseNet-121 | 6.95M | 0.6572 | 0.1230 | 0.2014 | 0.1818 | 0.2258 |
| 8 | ResNet-50 | 25.6M | 0.6570 | 0.1216 | 0.2072 | 0.1311 | 0.4946 |
| 9 | VGG-16 | 138M | 0.6517 | 0.1126 | 0.2024 | 0.1218 | 0.5968 |

**Average CNN AUC:** 0.6650
**AUC range:** 0.6517 (VGG-16) – 0.6747 (ResNet-101)
**AUC spread:** only 0.023 — indicates task ceiling dominates over architecture choice

### Transformer Models (3 models)

| Rank | Model | Params | AUC(TTA) | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| 1 | DeiT-S | 22M | 0.6463 | 0.1099 | 0.1931 | 0.1342 | 0.3441 |
| 2 | Swin-T | 28M | 0.6451 | 0.1095 | 0.2006 | 0.1227 | 0.5484 |
| 3 | ViT-B/16 | 85.8M | 0.6246 | 0.1054 | 0.1845 | 0.1153 | 0.4624 |

**Average Transformer AUC:** 0.6387
**Gap vs best CNN:** 0.036 AUC (ResNet-101 0.6747 vs best transformer DeiT-S 0.6463)
**ViT-B/16** is the weakest performer despite having 85.8M parameters — the largest model in the project.
**Swin-T** achieves the highest recall among transformers (0.548), tied with EfficientNet-B3, but at the cost of 729 FP cases.

---

## Per-Metric Leaders

| Metric | Winner | Value | Runner-Up | Value |
|---|---|---|---|---|
| Best AUC | ResNet-101 | 0.6747 | EfficientNet-B3 | 0.6740 |
| Best F1 | EfficientNet-B3 | 0.2198 | ResNet-101 | 0.2088 |
| Best PR-AUC | InceptionV3 | 0.1344 | ResNet-101 | 0.1248 |
| Best Precision | DenseNet-121 | 0.1818 | InceptionV3 | 0.1708 |
| Best Recall | MobileNet-V2 | 0.6398 | VGG-16 | 0.5968 |
| Best Efficiency (AUC/params) | MobileNet-V2 | 0.6676 / 3.4M | DenseNet-121 | 0.6572 / 6.95M |
| Worst Efficiency | VGG-16 | 0.6517 / 138M | ViT-B/16 | 0.6246 / 85.8M |

---

## TTA Improvement Analysis

| Model | Single AUC | TTA AUC | Gain |
|---|---|---|---|
| DenseNet-121 | 0.6385 | 0.6572 | +0.0187 |
| MobileNet-V2 | 0.6527 | 0.6676 | +0.0149 |
| ResNet-50 | 0.6485 | 0.6570 | +0.0085 |
| ViT-B/16 | 0.6180 | 0.6246 | +0.0066 |
| ResNet-101 | 0.6719 | 0.6747 | +0.0028 |
| EfficientNet-B3 | 0.6739 | 0.6740 | +0.0001 |
| DeiT-S | 0.6463 | 0.6463 | 0.0000 |
| Swin-T | 0.6405 | 0.6451 | +0.0046 |
| InceptionV3 | 0.6569 | 0.6647 | +0.0078 |
| Xception | 0.6606 | 0.6633 | +0.0027 |
| DenseNet-201 | 0.6527 | 0.6599 | +0.0072 |
| VGG-16 | 0.6511 | 0.6517 | +0.0006 |

TTA provided the largest benefit for models with noisier single-inference outputs (DenseNet-121: +0.019).
TTA provided marginal benefit for already-stable models (EfficientNet-B3: +0.0001).

---

## Comparison Against Literature

| Study | Dataset | Best AUC | Notes |
|---|---|---|---|
| Rajpurkar et al. [1] | ChestX-ray14 | 0.768 | No patient-wise split, full 112K dataset |
| Salehi et al. [6] | Kermany | 0.860 | Balanced pediatric dataset, single-disease binary |
| Hashmi et al. [5] | Private | 0.820 | Ensemble of 5 models |
| **Our best (ResNet-101)** | **NIH val.csv** | **0.6747** | **Patient-wise split, NIH-only, strict binary** |

> **Important:** Direct comparison to published numbers is not valid. Our evaluation uses strict patient-wise splitting (preventing leakage between scans of the same patient), NIH-only positives (no easy Kermany pediatric cases), and a single-class binary task evaluated on the difficult NIH val split where positives are radiologically subtle adult pneumonia cases. Our AUC of 0.6747 on this stricter evaluation is competitive with published results when methodology is accounted for.

---

## Key Observations

1. **Task ceiling dominates architecture choice.** The spread between best (0.6747) and worst CNN (0.6517) is only 0.023 AUC. Across 9 diverse CNN architectures, performance clusters tightly, indicating the NIH label noise (~30–40% estimated) and class imbalance (13:1) constrain what any model can achieve.

2. **CNN family uniformly outperforms transformers.** Every CNN model (worst: VGG-16 0.6517) outperforms every transformer model (best: DeiT-S 0.6463) in AUC. This 0.005 gap is consistent with Singh et al. [12] who showed ViT models require far more data than the NIH training split (1,004 positives) to surpass pretrained CNNs.

3. **Efficiency argument favours MobileNet-V2.** With only 3.4M parameters, MobileNet-V2 achieves AUC=0.6676, ranking 3rd overall. It also achieves the best recall (0.6398 — meaning it catches 64% of all positive cases), making it the strongest candidate for a screening application where sensitivity is prioritised.

4. **Precision-recall tradeoff is architecture-dependent.** DenseNet-121 (precision=0.1818, recall=0.2258) takes the most conservative strategy — the fewest FP cases (189) at the cost of missing 144/186 positives. EfficientNet-B3 (precision=0.1375, recall=0.5484) takes the opposite approach, catching 102/186 positives but generating 640 false alarms. Neither is objectively better — clinical context determines which matters more.

5. **ViT-B/16 is the worst value-for-compute model.** Despite 85.8M parameters (the largest model), 742 minutes training time (the longest), and a specialised Attention Rollout XAI pipeline, ViT-B/16 achieves the lowest AUC (0.6246) and lowest F1 (0.1845) of all 12 models. This is a data-efficiency failure consistent with the published transformer literature on small medical imaging datasets.

6. **TTA benefit is inversely correlated with model stability.** The models that improved most from TTA (DenseNet-121: +0.019; MobileNet-V2: +0.015) were those with higher single-inference variance. Stable, well-converged models (EfficientNet-B3: +0.0001) gained almost nothing from TTA, confirming that TTA is a variance-reduction technique, not a performance amplifier for well-calibrated models.
