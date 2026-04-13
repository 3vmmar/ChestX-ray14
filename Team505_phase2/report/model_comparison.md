# Model Comparison — Team505 Phase 2

## Preliminary Baseline Results (DEV Mode)

All models were trained on the same `train_dev.csv` subset (6,365 train / 1,720 val images)
with identical hyperparameters and patient-wise splitting (seed=42).

| Metric | DenseNet-121 | DenseNet-201 | Xception | ViT-B/16 |
|--------|:------------:|:------------:|:--------:|:--------:|
| **ROC-AUC** | 0.6076 | 0.6051 | **0.6158** | 0.5770 |
| **F1-Score** | **0.2080** | 0.1919 | 0.2053 | 0.1980 |
| **Precision** | **0.1349** | 0.1096 | 0.1276 | 0.1220 |
| **Recall** | 0.4545 | **0.7727** | 0.5260 | 0.5260 |
| **Accuracy** | **0.6901** | 0.4174 | 0.6355 | 0.6186 |
| Threshold | 0.55 | 0.25 | 0.50 | 0.55 |
| Best Epoch | 3 | 3 | 3 | 5 |
| Params | 6.95M | 18.09M | 20.81M | 85.80M |
| Train Time | 16.9 min | 81.1 min | 61.8 min | 26.7 min |

### Paper Basis

| Model | Member | Approved Paper |
|-------|--------|----------------|
| DenseNet-121 | Ammar Ahmed | Rajpurkar et al. (CheXNet/CheXNeXt, 2017/2018) |
| DenseNet-201 | Hosam Nabil | Rahman et al. (2021) |
| Xception | Mohamed Eslam | Guler & Polat (2021) |
| ViT-B/16 | Abdelrahman Mostafa | Singh et al. (2022) |

### Training Behavior Notes

- **DenseNet-121:** Stable training; best AUC at epoch 3. Mild overfitting visible (val loss increases while train loss decreases after epoch 3). Compact model (7M params), fastest convergence. Threshold tuned from 0.50 to 0.55 for slight F1 improvement.

- **DenseNet-201:** Deeper variant shows similar AUC (0.605 vs 0.608) but with much higher recall (0.77) at the cost of precision and accuracy. The model adopted a very low threshold (0.25), indicating it learned a weaker decision boundary. Training took 5x longer than DenseNet-121 due to larger parameter count. Scheduler triggered LR reduction at epoch 4.

- **Xception:** Best ROC-AUC (0.616) among all models. Stable training curve with consistent improvement through epoch 3. Depthwise separable convolutions provide a different feature extraction strategy than dense connections. F1 comparable to DenseNet-121.

- **ViT-B/16:** Slowest to converge (still improving at epoch 5). AUC steadily increased across all epochs, suggesting the model was still learning. With 86M parameters, ViT-B/16 is data-hungry and the small dev subset may be insufficient. Batch size was reduced to 16 due to memory constraints. Performance is expected to improve significantly with full training data.

### Key Takeaways

1. **Best AUC:** Xception (0.6158) slightly outperforms both DenseNet variants
2. **Best F1/Precision:** DenseNet-121, despite being the simplest model
3. **Best Recall:** DenseNet-201 (0.77), but at severe cost to precision
4. **All models struggle** with the ~9% positive rate; this is consistent with the literature on ChestX-ray14 binary pneumonia detection with limited training data
5. **ViT-B/16 underperforms** in the low-data regime, consistent with known behavior of Transformers requiring more training data than CNNs
6. **Full training** on the complete dataset is expected to improve all models substantially
