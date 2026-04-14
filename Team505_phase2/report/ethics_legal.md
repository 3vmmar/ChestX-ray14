# Ethical and Legal Considerations — Team505 Phase 2

## 1. Patient Data Privacy and Confidentiality

### Dataset Status
The NIH ChestX-ray14 dataset is a **de-identified, publicly available** research dataset released by the National Institutes of Health (NIH) Clinical Center. All images were stripped of protected health information (PHI) before public release in compliance with HIPAA Safe Harbor guidelines.

### Our Responsibilities
- We use the dataset **as-is** without attempting to re-identify patients
- No external data linkage was performed
- Patient IDs in our split CSVs are anonymized integer identifiers (not real medical record numbers)
- Model checkpoints and outputs do not embed patient-identifiable information
- The patient-wise splitting strategy ensures no data leakage across train/val/test sets

### Limitations
- De-identification does not guarantee absolute anonymity — face-adjacent images or rare conditions could theoretically enable re-identification
- The dataset was collected from a single institution (NIH Clinical Center), limiting demographic representativeness

---

## 2. Algorithmic Bias and Fairness

### Known Dataset Biases
- **Institutional bias:** All images come from a single US academic medical center, which may not represent global populations, imaging equipment diversity, or disease prevalence distributions
- **Label bias:** Disease labels were mined from radiology reports using Natural Language Processing (NLP), not manually annotated by radiologists for this dataset. Wang et al. (2017) reported label accuracy of approximately 90%, meaning ~10% of labels may be incorrect
- **Demographic bias:** While the dataset includes basic demographic information (age, gender), subgroup-specific performance was not separately validated in this project
- **Class imbalance:** Pneumonia represents only ~1.3% of the full dataset, making the classification task inherently challenging and potentially biasing models toward predicting the majority class

### Fairness Considerations
- We did **not** perform subgroup analysis (by age, sex, or view position) in this phase — this is a known limitation
- Model thresholds were tuned globally; optimal thresholds may vary across demographic subgroups
- Pos_weight in the loss function partially addresses class imbalance but does not eliminate it

### Mitigation Steps
- We report **ROC-AUC** as the primary metric rather than accuracy, which would be misleadingly high due to class imbalance
- We use **threshold tuning** to balance precision and recall
- We include **multiple evaluation metrics** (accuracy, precision, recall, F1, AUC) to provide a comprehensive performance picture
- We plan subgroup analysis for the final project phase

---

## 3. Risk of Misuse and Overreliance in Clinical Settings

### Current System Limitations
- These are **preliminary dev-mode baselines**, trained on a small subset (~6K images). They are **not suitable for clinical use**
- ROC-AUC scores of 0.58–0.62 are well below the clinical deployment threshold
- No model has been evaluated on the official test set
- No external validation on out-of-distribution data has been performed

### Risks of Clinical Deployment Without Proper Validation
1. **False negatives** could lead to missed pneumonia diagnoses, delaying treatment with potentially deadly consequences
2. **False positives** could lead to unnecessary follow-up tests, patient anxiety, and healthcare resource waste
3. **Overreliance** by clinicians on AI predictions could reduce diagnostic vigilance ("automation bias")
4. **NLP-mined labels** mean the model is learning from imperfect ground truth, which sets a ceiling on achievable performance

### Responsible Use Guidelines
- AI-assisted diagnosis should always be **supplementary**, not primary
- Clinical decisions must remain with qualified medical professionals
- Any deployed system must undergo **prospective clinical trials** and regulatory approval (FDA/CE marking)
- Continuous monitoring for performance degradation and bias is essential

---

## 4. Why This Project Is Ethically Responsible

1. **We use a public, de-identified dataset** with appropriate institutional permissions
2. **We do not claim clinical utility** — this is a research and educational project
3. **We implement and compare explainability methods** (Grad-CAM, LIME, SHAP, IG) to make model decisions transparent
4. **We honestly report limitations** including class imbalance effects, NLP label noise, and dev-mode constraints
5. **We use patient-wise splitting** to prevent data leakage — a common ethical lapse in medical AI papers
6. **We compare multiple architectures** under controlled conditions rather than cherry-picking results

---

## 5. How Explainability Contributes to Trust and Safety

### Why XAI Matters for Medical AI
- **Clinical trust:** Physicians are more likely to trust and appropriately use AI systems whose reasoning they can understand
- **Error detection:** XAI visualizations revealed that models sometimes attend to non-diagnostic regions (e.g., image borders, annotations) — this insight is only possible with explainability
- **Regulatory compliance:** Emerging AI regulations (EU AI Act, FDA guidance) increasingly require explainability for high-risk AI applications including medical devices
- **Patient rights:** Patients have a right to understand how AI-assisted decisions about their healthcare are made

### Project-Specific XAI Insights
- Grad-CAM confirmed that models generally attend to **lung fields** — the anatomically correct region for pneumonia detection
- LIME revealed that **false positive** predictions sometimes relied on non-pulmonary features, highlighting a failure mode
- Integrated Gradients provided the most fine-grained view of model attention, useful for **debugging** individual cases
- Multi-method comparison showed that **different XAI methods highlight different aspects** of model behavior — no single method is sufficient

---

## 6. Limitations and Future Steps

### Current Limitations
1. No subgroup fairness analysis (by age, sex, ethnicity)
2. No external validation dataset
3. NLP-mined labels introduce irreducible noise (~10%)
4. Dev-mode training only — full-data results may differ
5. Single-institution data limits generalizability

### Recommended Future Steps
1. Perform demographic subgroup analysis after full training
2. Validate on external chest X-ray datasets (e.g., CheXpert, MIMIC-CXR)
3. Compare NLP-mined labels against expert radiologist annotations on sample subset
4. Apply fairness metrics (equalized odds, demographic parity)
5. Conduct user studies with clinicians to evaluate XAI utility
