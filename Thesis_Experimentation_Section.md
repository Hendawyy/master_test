# Chapter 4: Experimentation

This chapter describes the two experimental phases undertaken to train and
validate the Neuro-DT multimodal classifier. Experiment 1 (Section 4.1) was
conducted on CPU-based Azure Machine Learning compute and established the
baseline pipeline, architecture, and a single validated fold. Experiment 2
(Section 4.2) migrated the same pipeline to GPU compute to obtain a complete,
cross-validated result, a full ablation study, and end-to-end validation of
the Digital Twin prognostic pipeline, and concludes with deployment of the
resulting model to a production clinical dashboard.

---

## 4.1 Experiment 1: Training on CPU-Based Compute

### 4.1.1 Experimental Setup

Training infrastructure was provisioned on Microsoft Azure Machine Learning.
A compute cluster (`Standard_E4ds_v4`: 4 vCPUs, 32 GB RAM, CPU-only) was used
for all data ingestion, preprocessing, and training jobs, with a separate
compute instance of the same specification used for interactive development.
No GPU acceleration was available in this phase.

The dataset comprised 1,549 T1-weighted MPRAGE MRI scans drawn from the ADNI-1,
ADNI-GO, ADNI-2, and ADNI-3 cohorts (obtained via the LONI Image & Data
Archive), covering approximately 500 unique patients across three diagnostic
classes: Cognitively Normal (CN, n=469), Mild Cognitive Impairment (MCI,
n=603), and Dementia (n=477). Each scan was paired with four clinical/tabular
features drawn from the ADNIMERGE, PTDEMOG, MMSE, and APOERES tables: patient
age, years of education, Mini-Mental State Examination (MMSE) score, and
APOE4 allele count.

Data preprocessing followed a standard neuroimaging pipeline: DICOM loading
(with an explicit `force=True` reader flag required to parse ADNI-1-era scans
that predate strict DICOM header enforcement), reorientation to RAS space,
resampling to 1.5 mm isotropic spacing, resizing/padding to a fixed
128×128×128 voxel volume, and intensity normalization to [0, 1]. All 1,549
preprocessed volumes were cached to disk as PyTorch tensors to avoid
repeating this pipeline on every training epoch.

### 4.1.2 Model Architecture and Training Configuration

The classifier (`MultimodalTransformer`, 24M parameters) combines a 3D
DenseNet-121 backbone (producing a 1,024-dimensional image embedding) with
the four tabular features, concatenated and projected through a linear
layer, LayerNorm, and GELU activation into a 2-layer Transformer encoder
(8 attention heads, pre-normalization), followed by a linear classification
head over the three diagnostic classes.

Training used 5-fold stratified cross-validation, AdamW optimization
(learning rate 1×10⁻⁴), a batch size of 4 (constrained by available system
memory), and a `CosineAnnealingLR` schedule with early stopping (patience 3
epochs) based on validation loss.

### 4.1.3 Results

Training completed for only one of five folds. The remaining four folds
triggered early stopping within the first three epochs, caused by a sharp
validation-loss spike in early training — attributed to the learning-rate
schedule beginning at full magnitude with no warmup period, combined with
the high per-step gradient noise inherent to a batch size of 4 on
high-dimensional 3D volumetric input. Fold 4 was the only fold to train to
completion (15 epochs) and was adopted as the primary model for this phase.

**Table 4.1 — Fold 4 classification performance (validation set, n=310)**

| Class | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| CN | 0.85 | 0.86 | 0.86 | 0.957 |
| Dementia | 0.79 | 0.82 | 0.81 | 0.936 |
| MCI | 0.74 | 0.71 | 0.72 | 0.844 |
| **Overall** | **0.79** | **0.80** | **0.80** | **0.912** |

Notably, the model produced zero misclassifications between the CN and
Dementia classes — the two diagnostically furthest-apart categories — across
the entire validation set.

An ablation comparison against classical machine learning baselines trained
on the four tabular features alone (Table 4.2) revealed that Random Forest
and Gradient Boosting classifiers achieved higher raw AUC than the full
multimodal model. Investigation attributed this to a label-leakage effect:
ADNI's diagnostic labels are, to a degree, clinically derived from MMSE
threshold criteria, so a classifier with direct access to MMSE can partially
reconstruct the label itself without reference to any imaging data.

**Table 4.2 — CPU-feasible ablation results**

| Variant | AUC | F1 |
|---|---|---|
| A0 — Full multimodal model | 0.912 | 0.796 |
| A1 — Tabular-only MLP | 0.877 | 0.692 |
| B1 — Logistic Regression (tabular) | 0.865 | 0.707 |
| B2 — SVM, RBF kernel (tabular) | 0.871 | 0.735 |
| B3 — Random Forest (tabular) | 0.943 | 0.830 |
| B4 — Gradient Boosting (tabular) | 0.941 | 0.825 |

Gradient-based explainability (Grad-CAM, applied to the final dense block of
the CNN backbone) showed activation concentrated in the medial temporal lobe
and hippocampal/entorhinal regions for Dementia predictions, consistent with
established patterns of atrophy in Alzheimer's Disease.

### 4.1.4 Limitations of This Phase

This phase established that the architecture and pipeline were functionally
correct and capable of learning genuine diagnostic signal, but two structural
limitations motivated a second experimental phase: (1) only one of five
cross-validation folds ever completed, meaning the reported AUC of 0.912 is a
single-fold result rather than a statistically robust cross-validated
estimate; and (2) the deep-learning ablation variants beyond the tabular-only
baseline (CNN-only and fusion-architecture comparisons) were computationally
infeasible on CPU-only compute and were not attempted.

### Screenshots to include in Section 4.1
- Azure ML Studio compute overview (cluster/instance specification page)
- Azure ML job run page showing the training experiment's fold-by-fold status (completed vs. early-stopped)
- A validation loss/accuracy curve plot for Fold 4 (the completed fold), ideally alongside one early-stopped fold for visual contrast
- The confusion matrix for Fold 4
- The Grad-CAM activation overlay figure (axial/sagittal/coronal, one Dementia example)
- The ablation results bar chart (Table 4.2, visualized)

---

## 4.2 Experiment 2: GPU-Based Training and Full Pipeline Evaluation

### 4.2.1 Motivation and Experimental Setup

To obtain a statistically valid cross-validated result and complete the
planned ablation and prognostic-modeling work, the identical codebase was
migrated to a GPU-equipped workstation (NVIDIA RTX 5070 Ti, 17.1 GB VRAM).
All Azure ML Workspace-specific dependencies were removed from the pipeline
so it could run as a self-contained local process, with data transferred via
Azure Blob Storage.

### 4.2.2 Training Configuration Changes

Two configuration changes were made in response to the instability observed
in Experiment 1: the learning-rate schedule was changed from
`CosineAnnealingLR` to `OneCycleLR` with a 10% warmup period, and the batch
size was increased from 4 to 16 (enabled by GPU memory capacity), together
with mixed-precision (AMP) training. The training loop was additionally
redesigned to be resumable across interruptions — checkpoints store
optimizer, scheduler, and AMP-scaler state alongside model weights, allowing
a fold to resume from its last completed epoch rather than restarting.

### 4.2.3 Cross-Validated Results

With the revised schedule, all five folds completed training successfully.

**Table 4.3 — 5-fold cross-validated results (GPU-trained model)**

| Fold | Validation AUC |
|---|---|
| 1 | 0.8821 |
| 2 | 0.8591 |
| 3 | 0.8125 |
| 4 | 0.9511 |
| 5 | 0.8481 |
| **Mean ± SD** | **0.8706 ± 0.0461** |

This is reported as the primary result of this thesis: a genuine 5-fold
cross-validated estimate, in contrast to Experiment 1's single-fold result.
A direct comparison is possible on Fold 4, since both experiments used an
identical stratified split (same random seed) and therefore evaluated on
the same held-out patients:

**Table 4.4 — Direct comparison on the shared Fold 4 split**

| | CPU (Experiment 1) | GPU (Experiment 2) |
|---|---|---|
| Fold 4 AUC | 0.9120 | **0.9511** |
| Folds completed | 1 of 5 | 5 of 5 |
| Reported metric | Single fold | 5-fold mean: **0.8706 ± 0.0461** |

On this shared fold, the GPU-trained model outperforms the CPU-trained model
by 0.039 AUC, attributable to the scheduler and batch-size changes described
above, independent of the cross-validation completeness improvement.

**Per-class evaluation of the GPU model (Fold 4, n=310):** CN AUC 0.982,
Dementia AUC 0.958, MCI AUC 0.913; overall accuracy 85%, macro F1 0.85.

### 4.2.4 Checkpoint Selection Methodology Experiment

A secondary experiment tested whether checkpoint selection based on maximum
validation AUC (rather than minimum validation loss) would yield a more
favorable — or simply different — result. An identical training run was
performed using AUC as both the selection and early-stopping criterion.

**Table 4.5 — Loss-selected vs. AUC-selected checkpoint criteria**

| Fold | Loss-selected AUC | AUC-selected AUC |
|---|---|---|
| 1 | 0.8821 | 0.9439 |
| 2 | 0.8591 | 0.9398 |
| 3 | 0.8125 | 0.9265 |
| 4 | 0.9511 | 0.9451 |
| 5 | 0.8481 | 0.9342 |
| **Mean ± SD** | **0.8706 ± 0.0461** | **0.9379 ± 0.0069** |

Although the AUC-selected criterion produced a higher mean score, analysis
of the per-epoch training logs showed that every fold trained through all 20
available epochs without triggering early stopping, reaching training
accuracies of 97–99% by the final epochs — a clear indicator of overfitting
that a loss-based criterion would have halted earlier. Because the AUC
criterion also selects the checkpoint that is the argmax of a noisy metric
evaluated on a validation set of only ~310 samples, this method is a biased
estimator of true generalization performance. The loss-selected result
(0.8706 ± 0.0461) was therefore retained as the primary reported result;
this comparison is presented as a methodological contribution regarding
checkpoint-selection bias rather than as a competing headline figure.

### 4.2.5 Ablation Study

The complete ablation study — infeasible in Experiment 1 — was executed on
the GPU platform, comprising seven deep-learning architecture variants and
four classical machine-learning baselines, each evaluated on the Fold 4
data split.

**Table 4.6 — Full ablation study results**

| Model | AUC | Accuracy | Macro F1 |
|---|---|---|---|
| Transformer, 1 layer | 0.9563 | 0.8516 | 0.8552 |
| CNN + Linear fusion | 0.9487 | 0.8419 | 0.8454 |
| **Full multimodal model** | **0.9486** | 0.8516 | 0.8550 |
| CNN-only (no tabular branch) | 0.9474 | 0.8452 | 0.8497 |
| Random Forest (tabular) | 0.9428 | 0.8258 | 0.8298 |
| Gradient Boosting (tabular) | 0.9407 | 0.8226 | 0.8253 |
| CNN + MLP fusion | 0.9389 | 0.8387 | 0.8436 |
| Full model, no class weighting | 0.9294 | 0.7839 | 0.7906 |
| Tabular-only MLP | 0.8716 | 0.6839 | 0.6750 |
| SVM, RBF kernel (tabular) | 0.8711 | 0.7355 | 0.7348 |
| Logistic Regression (tabular) | 0.8649 | 0.7097 | 0.7073 |

As these results are single-fold (n=310) rather than cross-validated, the
close scores among the top five variants (within approximately 0.01 AUC of
one another) should be interpreted as within normal fold-level variance
rather than a definitive architectural ranking.

Two findings are of particular note. First, removing the tabular branch
entirely (CNN-only) produced performance statistically indistinguishable
from the full multimodal model (0.9474 vs. 0.9486), indicating that the
imaging pathway carries the substantial majority of the model's predictive
signal in this architecture. Second, and consistent with the label-leakage
finding of Experiment 1 (Section 4.1.3), classical tree-ensemble methods
outperformed the deep tabular-only branch by a wide margin (~0.94 vs. 0.87
AUC) on identical input features — attributed to an architecture-capacity
mismatch between a high-capacity neural network and a four-dimensional
input, compounded by the same MMSE label-leakage effect.

### 4.2.6 Explainability and Prognostic Modeling

**Grad-CAM.** Explainability analysis was extended across multiple patients
in this phase. Activation patterns were found to be comparatively diffuse
— spanning a broad hemispheric region rather than sharply localizing to the
hippocampal/medial-temporal-lobe region associated with Alzheimer's
pathology. This is presented as an open limitation requiring further
investigation across a larger patient sample, rather than a confirmed
finding.

**Markov chain prognostic model.** Disease-state transition probabilities
were estimated from longitudinal ADNI visit sequences, stratified by APOE4
genotype. The resulting model exhibited two forms of external clinical
validity without being explicitly constrained to do so: Dementia was
correctly modeled as an effectively absorbing state (self-transition
probability 0.997–1.00), and APOE4-positive patients showed a higher
MCI→Dementia transition probability (0.184) than APOE4-negative patients
(0.133) — the direction consistent with APOE4's known status as a
progression risk factor.

**Age-sensitivity analysis.** An initial single-patient simulation, varying
patient age while holding the MRI scan constant, produced a
counter-intuitive result — predicted dementia risk decreasing with
increasing age. Repeating this analysis across five independent patients
showed the direction was patient-dependent (two of five patients showed the
clinically expected increasing-risk direction), consistent with the AGE
tabular feature carrying a comparatively weak and noisy signal relative to
the imaging pathway (Section 4.2.5), rather than indicating a systematic
error in the simulation methodology.

**What-if intervention simulation.** Simulated pharmacological intervention
(reducing the MCI→Dementia transition probability by a fixed percentage, as
a proxy for anti-amyloid therapies such as Lecanemab and Donanemab) produced
plausible directional effects; these simulated effect sizes are explicitly
not calibrated against real clinical-trial outcome data and are presented
as an illustrative capability of the framework rather than a clinical
efficacy claim.

### 4.2.7 Clinical Dashboard Deployment

The GPU-trained model and associated Markov transition matrices were
integrated into a Streamlit-based clinical dashboard, containerized and
deployed to Azure App Service. During this integration, three functional
defects were identified and resolved prior to production release:

1. A checkpoint-loading incompatibility with current PyTorch versions
   (missing `weights_only=False`, required because the checkpoint embeds a
   fitted scikit-learn scaler object).
2. A label-mapping orientation mismatch between the two model generations'
   saved checkpoint metadata, which caused inference to fail on every
   request against the newer checkpoint format.
3. A silent-failure mode in which a patient with no available cached MRI
   scan received a full-confidence diagnostic prediction generated from a
   blank image tensor, with no indication to the clinician that the imaging
   pathway had not been used. This was resolved by explicitly tracking scan
   availability and surfacing an unmissable warning wherever such a
   prediction occurs.

Following these fixes, the deployed model was validated against a genuine
held-out ADNI patient record not used during any training or validation
step in this thesis, correctly predicting the patient's true diagnosis
(Dementia) with near-full confidence, with the reported risk-trajectory and
genotype-effect figures matching hand-computed values derived independently
from the underlying transition matrices.

### Screenshots to include in Section 4.2
- GPU specification confirmation (e.g. `nvidia-smi` output or the notebook's own GPU-detection print) — establishes the experimental hardware
- Training-loop console output showing all 5 folds completing (fold summary table/log)
- A bar chart or line plot of Table 4.3 (per-fold AUC) and Table 4.4 (CPU vs. GPU Fold 4 comparison)
- Training curves (train/validation loss and AUC per epoch) for at least one loss-selected fold and its AUC-selected counterpart, illustrating the overfitting pattern described in Section 4.2.4
- The full ablation results table/bar chart (Table 4.6), ideally sorted by AUC as shown
- Grad-CAM overlays for at least two patients (to support the "diffuse activation" discussion)
- The Markov transition-matrix heatmap, ideally shown for the APOE4-positive and APOE4-negative strata side by side
- The age-sensitivity multi-patient result (the per-patient direction table/plot)
- A what-if simulation trajectory plot (5-year risk curves under baseline vs. treatment scenarios)
- A screenshot of the deployed dashboard's prediction output for the held-out verification patient described in Section 4.2.7 (diagnosis probabilities panel and risk-trajectory panel)
- The dashboard's "About"/model-info panel showing the reported performance metrics as displayed to end users

---

## Notes for Integration

- Infrastructure minutiae (specific Azure resource names, job IDs, credential
  handling, container-build debugging) have been deliberately omitted from
  this chapter as they belong in a Methods/Implementation appendix rather
  than the Experimentation results narrative — happy to draft that appendix
  separately if your advisor wants it.
- All numerical results above are drawn directly from the two project
  journey documents provided; none have been rounded or adjusted beyond
  what was already reported there.
- Table/section numbering (4.1, 4.2, Table 4.1, etc.) assumes this becomes
  Chapter 4 of the thesis — renumber to match your actual chapter sequence.
