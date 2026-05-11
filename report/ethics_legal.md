# Ethical and Legal Considerations — Team505 Phase 3
## DSAI 305 | Spring 2026

---

## Issue 1 — Patient Data Privacy and Confidentiality

**Description of the risk:** Medical imaging data contains highly sensitive protected health information (PHI). Even de-identified datasets can pose re-identification risks through combinations of clinical metadata (age, view position, scan date), radiological features unique to an individual's anatomy, or linkage with external records.

**Why significant in healthcare AI:** In traditional software, a data breach exposes names, emails, or financial data — bad but recoverable. In medical AI, a breach can expose a patient's diagnosis (e.g., pneumonia as a marker of immunosuppression or HIV), their clinical timeline, and their anatomical characteristics. Under HIPAA (Health Insurance Portability and Accountability Act) in the US, and equivalent GDPR Article 9 protections in the EU, processing medical imaging data without explicit consent or an appropriate legal basis carries severe regulatory penalties and creates a fundamental violation of patient trust.

**Specific relation to NIH ChestX-ray14:** The NIH dataset was de-identified using automated pipelines that removed 18 HIPAA direct identifiers from metadata. However, the images themselves are not pixel-level de-identified — patient anatomy, implants, and radiological features unique to an individual remain. Published research (Jonathon Winkel et al., 2020) demonstrated that CXR de-identification at the metadata level is insufficient to prevent re-identification in linked datasets. Our project uses this data strictly for academic research under the NIH open data licence.

**Specific relation to our model outputs:** Our XAI pipeline produces outputs (Grad-CAM heatmaps, SHAP maps, IG maps) that are derived from individual patient images. These outputs should not be stored with patient identifiers or shared publicly. Our current outputs directory does not retain metadata linking XAI images to patient IDs — image filenames use model names and case types (TP/FP/TN/FN) without patient identifiers.

**Mitigation in our project:** (1) Data used strictly within local filesystem — no cloud upload of patient images. (2) No attempt to re-identify patients. (3) Results reported at population level (aggregate metrics), not individual patient level. (4) Dataset access follows NIH open data licence.

---

## Issue 2 — Algorithmic Bias and Demographic Fairness

**Description of the risk:** An AI model trained on a non-representative population may perform differently across demographic subgroups — race, sex, age, body habitus, or equipment type. If the model performs worse on underrepresented groups, clinical deployment could exacerbate health disparities rather than reduce them.

**Why significant in healthcare AI:** A model with AUC=0.67 on average could have AUC=0.75 for one demographic and AUC=0.58 for another. If deployed as a triage tool, the underserved group would receive lower-quality screening. This is not a theoretical concern — published studies on dermoscopy AI (Kinyanjui et al., 2020) and chest pathology AI (Larrazabal et al., 2020) both documented significant sex- and race-based performance gaps.

**Specific relation to our dataset:** NIH ChestX-ray14 is skewed toward adult male patients from the NIH Clinical Center in Bethesda, Maryland — a US research hospital with non-representative demographic distribution. The dataset does not include sufficient representation of pediatric patients, non-Western populations, or patients imaged on non-Siemens CT/X-ray equipment. Additionally, we excluded all external Kermany/RSNA data — which contained pediatric patients — specifically because of domain shift failure (Finding 3). This exclusion, while technically correct, means our models are particularly untested on younger patients.

**Specific relation to our results:** We did not conduct subgroup analysis by age, sex, or view position (PA vs AP). We cannot claim fairness across demographic groups. The NIH metadata includes sex and age fields which could enable subgroup evaluation — this is a concrete future work recommendation.

**What should be done in future work:** (1) Stratify AUC, F1, and recall by demographic subgroup before any deployment consideration. (2) Apply algorithmic fairness constraints (equalized odds, calibrated probabilities per group) if disparities are found. (3) Collect prospective data from underrepresented populations.

---

## Issue 3 — Risk of Model Misuse and Overreliance

**Description of the risk:** Clinicians or administrators may treat AI model outputs as definitive diagnoses rather than decision-support signals, leading to over-reliance. Conversely, systematic under-reliance (ignoring AI outputs entirely) would waste any clinical benefit the model provides. Both extremes are risks.

**Why significant — False Negatives in pneumonia detection:** Our best-recall model (MobileNet-V2) catches 119/186 positives (recall=0.6398), meaning it misses 36% of all pneumonia cases on the validation set. If a clinician relies solely on the model's negative output and dismisses the case without review, those missed cases (FN) represent patients with untreated pneumonia — which can progress to sepsis, respiratory failure, and death. This is not a hypothetical: pneumonia is the leading infectious cause of death globally (WHO, 2022).

**Specific recall numbers for all models:**
- Highest recall: MobileNet-V2 = 0.6398 (67/186 missed)
- Lowest recall: DenseNet-121 = 0.2258 (144/186 missed — extremely conservative)
- Best for clinical screening: EfficientNet-B3 (recall=0.5484, catches 102/186 positives)

**Role of XAI in mitigating overreliance:** Our 4 XAI methods produce visual explanations alongside predictions. A radiologist reviewing a Grad-CAM map that shows activation on a rib rather than lung parenchyma can appropriately distrust the positive prediction. An IG map showing the model activated on a subtle opacity in a false negative case can prompt re-review. XAI converts opaque binary predictions into interpretable evidence, enabling human-AI collaboration rather than human-AI replacement.

**Clinical deployment constraint:** Our models should be deployed as **triage support only** — flagging cases for prioritised radiologist review, not replacing radiologist sign-off. Any deployment must include a mandatory human review step for positive predictions and for cases near the decision threshold.

---

## Issue 4 — Regulatory Compliance (HIPAA / GDPR / EU AI Act)

**Description of the risk:** Medical AI systems that inform clinical decisions are classified as Software as a Medical Device (SaMD) in the US (FDA), as Class IIa medical devices in the EU (MDR 2017/745), and are subject to healthcare data regulations under HIPAA (US) and GDPR (EU). Deploying an AI model without regulatory clearance or proper data governance exposes the deploying institution to legal liability.

**Why significant in medical AI deployment:** An uncleared AI system producing clinical recommendations could cause patient harm, expose the hospital to liability, violate FDA and CE marking requirements, and trigger HIPAA audit penalties. The EU AI Act (2024) specifically classifies medical AI as high-risk, requiring conformity assessment, technical documentation, and human oversight mechanisms.

**Specific relation to our pipeline:**
- **Data handling:** We store model outputs (tta_metrics.json, XAI images) in local filesystem. No cloud transfer of patient data. No PHI in filenames or outputs.
- **Model distribution:** Our model checkpoints (best_model.pth) trained on NIH data should not be distributed as a clinical product without FDA 510(k) clearance or EU CE marking.
- **Output storage:** XAI images derived from patient data (even de-identified) are secondary processing outputs of PHI and should be treated accordingly.
- **Audit trail:** Training history CSVs provide partial audit trail. Full deployment would require logging of every inference with metadata, justification, and review outcome.

**What compliance frameworks would apply to deployment:**
- **FDA:** Pre-submission to CDER/CDRH as AI/ML-based SaMD; De Novo or 510(k) pathway depending on risk classification.
- **EU:** CE marking under MDR for Class IIa; conformity assessment by notified body.
- **HIPAA:** Covered entity or Business Associate Agreement with any institution storing model outputs linked to PHI.
- **EU AI Act:** High-risk category requiring technical documentation, conformity assessment, human oversight, and post-market monitoring.

---

## Why Our Solution is Ethically Responsible

**Patient data:** We use strictly de-identified NIH ChestX-ray14 data under its open research licence. No re-identification was attempted. No patient-level data is included in any report or public output.

**Transparency:** Our pipeline generates XAI explanations for every prediction. No black-box outputs without supporting visual evidence. All training decisions (loss function, sampling, architecture) are documented in this report suite with rationale.

**Limitations disclosed:** We explicitly document label noise (~30–40% on Pneumonia class), class imbalance (13:1), domain shift failure with external data, absence of subgroup fairness analysis, and the fact that our best model misses 34–78% of positive cases depending on threshold choice. We do not overstate performance.

**Research framing:** Our system is clearly framed as an academic research project under DSAI 305. It is not presented as a clinical product. All claims are qualified with the evaluation dataset and methodology.

---

## Role of Explainability in Safe Deployment

Explainability is not an optional add-on in medical AI — it is a safety mechanism. Our four XAI methods each address a distinct aspect of clinical trust and safety:

**Grad-CAM / Attention Rollout** provides spatial justification — a radiologist can verify that the model's activation aligns with the anatomically plausible location of pneumonia (lower lobes, perihilar regions). If Grad-CAM activates on non-pulmonary structures (cardiac border, rib overlay), the radiologist can appropriately discount the model's positive prediction, reducing false alarm acceptance.

**LIME** enables non-expert audit. Hospital administrators, clinical informaticists, and patient advocates who are not radiology-trained can examine LIME outputs and identify if the model is systematically firing on implausible image regions (e.g., patient labels, equipment artifacts, diaphragm borders). This broadens the scope of quality assurance beyond specialists.

**SHAP** provides global validation. By examining SHAP attributions across multiple patients, model developers and clinical informatics teams can confirm that the model is not learning spurious correlations (e.g., AP vs PA view position, age-related anatomical features) and is genuinely detecting pathology-related image features.

**Integrated Gradients** enables clinical failure audit. When the model makes an error (FP or FN), IG reveals which pixels drove that error, enabling targeted improvement — more training data on cases with similar characteristics, or flag cases near the IG-identified confound zone for mandatory secondary review.

Together, these four methods provide the multi-layer transparency required by the EU AI Act's human oversight mandate and the FDA's principles for transparency in AI/ML-based SaMD, making our system substantially more deployable than a black-box model with equivalent AUC would be.
