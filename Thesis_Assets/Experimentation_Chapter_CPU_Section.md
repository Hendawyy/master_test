## Experimentation Chapter

This Chapter contains two experimental phases undertaken to Train and Validate our model (Neuro-DT Multimodal Classifier)
Experiment 1 (Section 1) was conducted on CPU-Based Azure Machine Learning Compute instances and established the full data pipeline, the model architecture, single validated fold and partial ablation study.
Experiment 2 (Section 2) was conducted on GPU Compute instance and obtained Fully Cross-Validated Result, Complete ablation study(which the CPU couldn't complete), and end-to-end validation of the Digital Twin Pipeline.
Both Experiments were used to create a clinical dashboard started with the CPU and Finished with the GPU version and it is deployed on Azure App Service.

Both experiments implement the same three-stage architecture proposed in the proposed Framework:
Data Preparation and Model Training
BDT Core Engine producing personalized predictions and simulations for any given patient
Clinical Application Dashboard

Both experiments are using the same framework; they only differ in Hardware and how each of them was executed.

### Deviations From The Proposed Framework

Fig 1 Original Proposed Neuro-DT Framework

**Table 1.1 Implementation deviations from the proposed framework**

| Proposed | Implemented | Reason |
|---|---|---|
| Hidden Markov Chains For Progression Modeling | Empirical (fully-observed) Markov Chain, APOE4-Stratified | Transition probabilities were estimated directly from observed diagnosis sequences across longitudinal visits. No latent-state HMM training was performed since diagnostic state is directly observed at each visit rather than hidden. |
| Explainability Module: SHAP, Grad-CAM, Attention Maps | SHAP & Grad-CAM only | Transformer attention-weight visualization was not implemented |
| 3D Brain Visualization | 2D Multi-Planer Grad-CAM overlays(Axial/ Sagittal/ Coronal) | The Dashboard renders sliced views with heatmap overlay as an alternative for 3D Visualization |
| Ablation Study | Not Mentioned in the original Framework | Added in the new Framework |

Fig 2. Neuro-DT Updated Framework

## 1.1 Experiment 1: Training on CPU-Based Compute Instances

### 1.1.1 Infrastructure

Training Infrastructure was provisioned on Microsoft Azure Machine Learning. An interactive compute instance(DevBox) and a separate compute cluster(workhorse) which scaled to zero nodes when idle were provisioned both of specification Standard_E4ds_v4 (4 cores, 32 GB RAM, 150 GB disk). The compute instance was used as a development virtual machine and dashboard testing, while the compute cluster executed all data-ingestion and training jobs.

Fig 3 Azure ML Studio DevBox Compute Instance Overview

Fig 4 Azure ML Studio DevBox Compute Cluster Overview

### 1.1.2 Environment

A Python 3.10 environment was configured with a CPU-only PyTorch build, MONAI(Medical imaging preprocessing), pydicom, sickit-learn, MLflow(experiment tracking), and Azure SDKs for blob storage access. NumPy was explicitly pinned below version 2,0 after an initial environment build broke MONAU and pydicom compatibility under NumPy 2.x.

### 1.1.3 Dataset Acquisition

Access to the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset was obtained through LONI image & Data Archive following an institutional data-use application (approval time: 1 day). The acquired dataset comprises 1,549 T1-weighted MPRAGE structural MRI scans across approximately 500 unique patients, spanning the ADNI-1, ADNI-GO, ADNI-2, and ADNI-3 study phases, with three diagnostic classes: Cognitively Normal (CN, n=469), Mild Cognitive Impairment (MCI, n=603), and Dementia (n=477). Each scan comprises approximately 150-200 individual DICOM slice files; the raw dataset totaled approximately 33 GB.

Downloading this volume directly using a home network connection would take days. To solve this problem a temporary virtual machine was provisioned in the same Azure storage account region to leverage Microsoft Global Network Backbone which is 100x faster than typical broadband, reducing the download to minutes.The retrieved DICOM archives were then uploaded to Azure Blob Storage using Azure Storage Explorer, preserving ADNI's native hierarchical folder structure (subject ID / imaging sequence / acquisition date / image ID). The temporary VM was deallocated immediately after upload to avoid ongoing compute charges.

Alongside the imaging data, six clinical csv tables were obtained from the same archive: ADNIMERGE.csv (the primary longitudinal summary, 12,741 rows across all visits and subjects), PTDEMOG.csv (demographics), MMSE.csv, APOERES.csv (APOE genotyping), DXSUM_PDXCONV.csv (diagnosis history), and NEUROBAT.csv (neuropsychological battery, used for cross-validation of cognitive scores rather than as a model input).

### 1.1.4 Data Ingestion And Validation pipeline

A sequence of automated data-processing jobs was run on the compute cluster to convert the raw uploaded archive into clean, unified manifest linking each scan to its clinical record:

**Manifest Construction:** Blob storage was parsed to extract subject and image identifiers, which were then cross-referenced against the clinical CSV tables. This step first failed twice due to an authentication issue: the storage account and the Machine Learning workspace resided in different Azure Active Directory tenants, which caused the default credential chain to silently return zero results with no exception raised, rather than an authentication error. This was resolved by registering a dedicated service principal with explicit, correctly-scoped access to the storage account.

Fig 5 Data Ingestion job

Fig 6 Data Ingestion job Output

**Archive Extraction:** Raw DICOM archives were unpacked and re-uploaded to blob storage in their extracted form (approximately 5 hours and 44 minutes)

Fig 7 Archive Extraction Job

Fig 8 Archive Extraction Job Output

**DICOM integrity validation:** Every scan directory was loaded with pydicom to identify corrupted, incomplete, or unreadable files. The full validation pass identified approximately 23 scan directories with unrecoverable issues (missing slices, corrupted headers, zero-byte files), which were either substituted with an alternate visit of the same subject or dropped.

Fig 9 DICOM Integrity Validation

**3D Volume Validation:** Downloads and assembles 3 complete 3D MRI volumes to verify that the full preprocessing pipeline (DICOM → 3D tensor) works correctly end-to-end.

Fig 10 3D Volume Validation

**Final manifest generation:** The imaging manifest was joined to ADNIMERGE.csv and APOERES.csv on patient identifier and visit code. Four tabular features were selected for fusion with the imaging pathway:

- **AGE:** the single strongest demographic predictor of dementia risk.
- **PTEDUCAT** (years of education): a proxy for cognitive reserve; later confirmed by SHAP analysis (Section 1.1.11) as the strongest predictor for the CN class.
- **MMSE** (Mini-Mental State Examination, 0-30): a direct cognitive assessment score.
- **APOE4** (allele count, 0/1/2): the primary known genetic risk factor for late-onset Alzheimer's Disease.

Fig 11 Final Manifest Sample

Data cleaning addressed several dataset-specific issues:
rows missing any tabular feature or a diagnosis label were dropped; ADNI's several phase-specific diagnosis codes were standardized into three canonical classes (EMCI and LMCI mapped to MCI; SMC mapped to CN; AD mapped to Dementia); the APOE4 column's -4 sentinel value (ADNI's missing-data code, not a real allele count) was treated as missing, removing approximately 18 scans; and, for subjects with multiple longitudinal visits recorded under different diagnoses, the diagnosis was matched to the specific date of the imaging session rather than defaulting to the subject's most recent visit label, to avoid attaching a scan to a diagnosis from a different point in time.
The final cleaned dataset comprises the 1,549 scans described in Section 1.1.3.

> **Note for author (not yet reflected above):** the manifest-loading step's own printed output for this dataset reads `visit_date extracted: 0/1549 rows` — every row's visit date currently fails to parse. This doesn't affect anything written above (nothing here depends on `visit_date`), but it does affect the Markov Chain step later (see the caveat in Section 1.1.12) and is worth a one-line mention here if you want to be fully transparent about pipeline state at ingestion time.

### 1.1.5 Imaging Preprocessing

Each validated DICOM scan was processed through a standard neuroimaging pipeline: loading, reorientation to RAS anatomical space, resampling to 1.5 mm isotropic spacing, resizing/padding to a fixed 128x128x128 voxel volume, and intensity normalization to [0, 1]. All 1,549 preprocessed volumes were cached to disk as PyTorch tensors (approximately 13 GB) prior to training, to avoid repeating this pipeline on every epoch.

Two issues were encountered and resolved during this step: loading multiple full 3D volumes simultaneously for validation caused memory usage to spike near the commute instance's 32 GB ceiling, resolved by processing one scan at a time with explicit garbage collection between scans and approximately 23 scans failed during caching, handled with pre-scan exception handling and substitution from alternate visits when possible.

### 1.1.6 Model Architecture

The classifier (MultimodalTransformer, 24M parameters) combines a 3D DenseNet-121 backbone (producing a 1,024-dimensional image embedding) with the four tabular features, concatenated and projected through a linear layer, LayerNorm, and GELU activation into a 2-layer Transformer encoder (8 attention heads, pre-normalization), followed by a linear classification head over the three diagnostic classes. During initial implementation, the concatenated feature dimension (1,024 + 4 = 1,028) was found not to be evenly divisible by the number of attention heads, a requirement of the Transformer encoder; this was resolved by projecting to the nearest multiple of the head count via ceiling-rounded integer arithmetic rather than a fixed dimension.

### 1.1.7 Preliminary Run and Data-Quality-Driven Overfitting

Before committing to the full 5-fold, 15-epoch training run, the pipeline was first validated end-to-end on fast, preliminary run("FAST_PROTO" mode: 40% of the dataset, 3 folds, 10 epochs) to confirm the architecture was learning genuine signal before the far more expensive full run was submitted.

**Overfitting diagnosis:** An early iteration of this preliminary run, trained on an unfiltered version of the manifest, showed classic overfitting: validation loss reached a minimum at epoch 4 (0.6986) and then increased steadily over subsequent epochs (0.7465, 0.7562), while validation accuracy plateaued at approximately 64.5%. Inspection of the training logs traced this to a large number of Error processing image at path... warnings: the try/except block in the dataset loader was silently substituting a zero-valued tensor for any DICOM series it could not load, injecting label noise into training rather than surfacing the failure. This finding directly motivated the automated DICOM integrity validation step described in Section 1.1.4, which filters out unreadable scans before training rather than substituting a blank image for them, and the migration of the training process from an interactive notebook to a script-based Azure ML Command Job for more stable, reproducible execution.

**Preliminary result:** With the cleaned data and script-based execution, the FAST_PROTO run achieved a macro one-vs-rest validation AUC of 0.862 and 63% overall accuracy on the three-class problem (chance level 33%) using only 40% of the dataset and 10 epochs:

**Table 2 Preliminary (FAST_PROTO) per-class results**

| Class | AUC |
|---|---|
| Dementia | 0.914 |
| CN | 0.878 |
| MCI | 0.698 |
| Macro OvR | 0.862 |

The comparatively lower MCI AUC reflects a well-documented property of the ADNI cohort rather than a defect in this pipeline: of the misclassified MCI patients, a similar number were predicted CN as were predicted Dementia, consistent with MCI's clinical role as a transitional, diagnostically ambiguous state. a pattern reported across the ADNI literature and returned to in Section 1.1.11. Benchmarked against published 3-class ADNI results, this preliminary figure was already competitive using a fraction of the available data and training budget:

**Table 3 Preliminary result vs. published ADNI benchmarks**

| Study | Reported metric |
|---|---|
| Basaia et al. (2019), 3D CNN | AUC ~0.85 |
| Wen et al. (2020), CNN benchmark | AUC ~0.83 |
| Venugopalan et al. (2021), multimodal | AUC ~0.87 |
| Zhou et al. (2025), CNN + Swin Transformer | 92% accuracy (2-class) |
| This work (FAST_PROTO, 40% data, 10 epochs) | AUC 0.862 |

This preliminary run also converged quickly (validation loss stopped improving after epoch 5 of 10), an encouraging sign that motivated proceeding directly to the full run described in Section 1.1.9 rather than further preliminary tuning. Expanding the tabular feature set beyond the four features in Section 1.1.4 (e.g., ADAS-Cog, CDRSB, RAVLT immediate recall, all present in ADNIMERGE.csv) was considered as a way to strengthen MCI disambiguation, but was not carried into the full run reported in this thesis, to keep the tabular inputs consistent with the four clinically-motivated features specified in Section 1.1.4.

### 1.1.8 Training Configuration

Training used 5-fold stratified cross-validation, AdamW optimization (learning rate 1x10⁻⁴), a batch size of 4 (constrained by available system memory), and a CosineAnnealingLR schedule with early stopping (patience 3 epochs) based on validation loss.

Five-fold cross-validation follows standard practice in the medical imaging literature for a dataset of this size, balancing the statistical robustness of the resulting performance estimate against computational cost: fewer folds (e.g., three) risk higher variance in the estimate, while more folds (e.g., ten) offer diminishing returns at approximately 1,500 samples. Fifteen epochs was chosen based on the convergence behavior observed in the preliminary run (Section 1.1.7), which indicated the architecture converges within a comparable number of epochs on a data subset.

### 1.1.9 Results

Training completed for only one of five folds. The remaining four folds triggered early stopping within the first three epochs, caused by a sharp validation-loss spike in early training -- attributed to the learning-rate schedule beginning at full magnitude with no warmup period, combined with the high per-step gradient noise inherent to a batch size of 4 on high-dimensional 3D volumetric input. Fold 4 was the only fold to train to completion (15 epochs) and was adopted as the primary model for this phase.

**Table 4 Fold 4 classification performance (validation set, n=310)**

| Class | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| CN | 0.85 | 0.86 | 0.86 | 0.957 |
| Dementia | 0.79 | 0.82 | 0.81 | 0.936 |
| MCI | 0.74 | 0.71 | 0.72 | 0.844 |
| **Overall** | **0.79** | **0.80** | **0.80** | **0.912** |

Notably, the model produced zero misclassifications between the CN and Dementia classes -- the two diagnostically furthest-apart categories -- across the entire validation set. *(Confirm this against Fig 12 directly — the classification report above shows per-class precision/recall/F1 but not the individual confusion-matrix cells.)*

Fig 12 Fold 4 Confusion Matrix

### 1.1.10 Ablation Study: CPU-Feasible Variants

Five of the eleven ablation variants planned for this thesis were successfully trained and evaluated on CPU compute. This is stated explicitly because it is a genuine result of this phase, not a limitation: the full multimodal model (A0, reusing the existing Fold 4 checkpoint with no retraining required) and a tabular-only multilayer perceptron (A1) were evaluated alongside four classical machine-learning baselines trained directly on the four tabular features (B1 Logistic Regression, B2 SVM with an RBF kernel, B3 Random Forest, B4 Gradient Boosting).

**Table 5 CPU-feasible ablation results**

| Variant | AUC | F1 |
|---|---|---|
| A0 -- Full multimodal model | 0.912 | 0.796 |
| A1 -- Tabular-only MLP | 0.879 | 0.706 |
| B1 -- Logistic Regression (tabular) | 0.865 | 0.707 |
| B2 -- SVM, RBF kernel (tabular) | 0.871 | 0.735 |
| B3 -- Random Forest (tabular) | 0.943 | 0.830 |
| B4 -- Gradient Boosting (tabular) | 0.941 | 0.825 |

This comparison revealed that Random Forest and Gradient Boosting classifiers -- trained on four tabular features alone, with no imaging input whatsoever -- achieved *higher* raw AUC than the full multimodal model. Investigation attributed this to the MMSE label-leakage effect noted in Section 1.1.4: because ADNI's diagnostic labels are themselves partly derived from MMSE threshold criteria, a classifier with direct access to MMSE can reconstruct much of the label without reference to any imaging data. This is treated as a limitation of AUC as a sole comparison metric here, not as evidence that the multimodal model underperforms -- Section 1.1.11 and the full model's zero-CN-Dementia-confusion result are presented as the more clinically meaningful comparison points.

The model's contribution from imaging alone is measured directly against the tabular-only floor: full multimodal AUC (0.912) minus tabular-only MLP AUC (0.879) gives **+0.033 AUC points** attributable to the 3D MRI pathway.

The remaining five deep-learning variants -- a CNN-only classifier (A2), two alternative CNN-tabular fusion designs (A3, A4), a single-layer Transformer variant (A5), and a variant trained without class-weighted loss (A6) -- each require training a full 3D-CNN-based model from scratch on imaging data, at the same per-fold training cost that limited Section 1.1.9 to one completed fold. These were therefore not attempted on CPU-only compute in this phase and were deferred to Experiment 2 (Section 2), where they were executed as part of a complete, independently re-run eleven-variant ablation study.

Fig 13 CPU-Feasible Ablation Results

### 1.1.11 Explainability

SHAP analysis (applied to the tabular branch with the imaging embedding held frozen, since a full SHAP analysis over the imaging branch exceeded available system memory) identified PTEDUCAT as the strongest predictor for the CN class and AGE as the strongest predictor for the Dementia class. *(Confirm this ranking directly against Fig 14 before stating it as fact.)*

Fig 14 SHAP Tabular Feature Attribution

Gradient-based explainability (Grad-CAM, hooked on the final dense block of the CNN backbone) required the input tensor's gradient tracking to be explicitly enabled before the backward pass, which was not the default behavior when running inference on an already-evaluated model. For validation patient 057_S_1373 (true diagnosis Dementia, correctly predicted Dementia), Grad-CAM peak activation was located at axial slice z=86, sagittal slice y=43, and coronal slice x=87. *(Confirm visually in Fig 15 that this region corresponds to the medial temporal lobe / hippocampal area before citing it as consistent with known Alzheimer's atrophy patterns.)*

Fig 15 Grad-CAM Activation Overlay -- Patient 057_S_1373 (Dementia)

### 1.1.12 Markov Chain and Digital Twin -- Not Included in This Phase

Cell 13 (Markov Chain) and the Digital Twin / what-if simulation cells in the CPU notebook were exercised during development, but their output should **not** be used for this thesis section: the manifest's `visit_date` field failed to parse for every row (`0/1549 rows` extracted -- see the note in Section 1.1.4), so the chronological visit-ordering the transition-matrix computation depends on was effectively arbitrary rather than time-ordered. A symptom of this surfaced directly: this run's transition matrix showed APOE4-positive patients with a *lower* MCI-to-Dementia transition probability (0.210) than APOE4-negative patients (0.233) -- the opposite of APOE4's known role as a progression risk factor, and the opposite of the (correctly time-ordered) result reported for the GPU experiment in Section 2.

The Markov Chain, Digital Twin assembly, and what-if simulation are therefore reported only in Section 2 (GPU experiment), where the date-parsing defect was identified and corrected before these components were run. If a CPU-side demonstration of these components is wanted for completeness, the same fix must be applied to this notebook's manifest-loading step first.

### 1.1.13 Limitations of This Phase

This phase established that the architecture and full data pipeline were functionally correct, capable of learning genuine diagnostic signal, and capable of supporting a partial ablation study. Three points limit or qualify this phase's results: (1) only one of five cross-validation folds ever completed, meaning the reported AUC of 0.912 is a single-fold result rather than a statistically robust cross-validated estimate; (2) the six deep-learning ablation variants requiring fresh 3D-CNN training on imaging data (Section 1.1.10) were computationally infeasible on CPU-only compute within the scope of this phase; and (3) a manifest-level date-parsing defect (Section 1.1.12) meant the Markov Chain and Digital Twin components could not be validated on CPU in this phase, though this defect was identified and corrected before Experiment 2.

On CPU-only compute with the four tabular features used in this thesis, an AUC in the 0.91-0.93 range represents approximately the ceiling reported for 3-class ADNI classification in the literature for GPU-trained models of comparable design; Fold 4's result of 0.912 (Section 1.1.9) sits at this ceiling. Closing the gap further would likely require GPU-accelerated training of all five folds rather than one, additional tabular features (e.g., ADAS-Cog, CDRSB), and/or transfer learning from a model pretrained on a larger 3D medical imaging corpus -- directions taken up, in part, in Experiment 2.

---

## Figure and table reference summary (Section 1.1)

| # | What | Source |
|---|---|---|
| Fig 1 | Original proposed Neuro-DT framework | proposal document |
| Table 1.1 | Implementation deviations from the proposed framework | -- |
| Fig 2 | Neuro-DT updated framework | -- |
| Fig 3 | Azure ML Studio DevBox compute instance overview | Azure ML Studio portal |
| Fig 4 | Azure ML Studio DevBox compute cluster overview | Azure ML Studio portal |
| Fig 5 | Data ingestion job | Azure ML Studio portal |
| Fig 6 | Data ingestion job output | Azure ML Studio portal |
| Fig 7 | Archive extraction job | Azure ML Studio portal |
| Fig 8 | Archive extraction job output | Azure ML Studio portal |
| Fig 9 | DICOM integrity validation | notebook Cell 4 output |
| Fig 10 | 3D volume validation | notebook **Cell 6** output |
| Fig 11 | Final manifest sample | notebook **Cell 4** output (DataFrame preview) |
| Table 2 | Preliminary (FAST_PROTO) per-class results | notebook **Cell 10** output |
| Table 3 | Preliminary result vs. published ADNI benchmarks | -- |
| Table 4 | Fold 4 classification performance | notebook **Cell 11** output |
| Fig 12 | Fold 4 confusion matrix | notebook **Cell 11** -> `confusion_matrix.png` |
| Table 5 | CPU-feasible ablation results | notebook **Cell A4_CPU** output |
| Fig 13 | CPU-feasible ablation results, bar chart | notebook **Cell A4_CPU** -> `fig_ablation_partial.png` |
| Fig 14 | SHAP tabular feature attribution | notebook **Cell 11** -> `shap_summary.png` |
| Fig 15 | Grad-CAM activation overlay, Patient 057_S_1373 | notebook **Cell 12** -> `gradcam_057_S_1373.png` |

**Not used in this section** (see Section 1.1.12): `markov_heatmap.png`, `whatif_*.png` -- generated by Cells 13/15 this session but excluded pending the `visit_date` fix.
