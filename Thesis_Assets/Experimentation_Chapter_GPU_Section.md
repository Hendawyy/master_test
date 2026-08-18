## 2.1 Experiment 2: Training and Full Pipeline Evaluation on GPU-Based Compute

### 2.1.1 Motivation and Experimental Setup

To obtain a statistically valid cross-validated result and complete the planned ablation and prognostic-modeling work, the identical codebase was migrated to a university GPU lab workstation (NVIDIA GeForce RTX 5070 Ti, 17.1 GB VRAM), running locally rather than on Azure ML Compute.

Fig 26 GPU Detection / Specification Confirmation

### 2.1.2 Migration: Environment and Data Transfer

Every Azure ML Workspace-specific dependency was removed from the pipeline so it could run as a self-contained local process, reading only from local files (the lab PC's own disk) rather than a live Azure ML workspace connection. Data (the cached tensors, model checkpoints, and clinical manifest) was transferred via Azure Blob Storage using a download utility hardened specifically for this migration: downloads write to a temporary file and are renamed to their final path only on success, so an interrupted transfer can never be mistaken for a complete one on a later run; both individual file downloads and the storage-listing operation itself retry with exponential backoff. A companion verification step then loaded every transferred tensor and checkpoint file directly (rather than only checking for file existence) to catch any corruption that completed writing but was still invalid -- confirming all 1,549 cached tensors and all checkpoints transferred correctly.

During this migration, the visit\_date extraction defect described in Section 1.1.12 was identified and corrected -- the fix applied here is the same one later ported back into the CPU notebook.

### 2.1.3 Training Configuration Changes

Three configuration changes were made in response to the instability observed in Experiment 1: the learning-rate schedule was changed from `CosineAnnealingLR` to `OneCycleLR` with a 10% warmup period, the batch size was increased from 4 to 16 (enabled by GPU memory capacity), and mixed-precision (AMP) training was enabled. Early stopping patience was also relaxed from 3 to 5 epochs.

### 2.1.4 Resumable Training Design

Because lab PC access was not continuous, the training loop was designed to tolerate interruption at any point: each checkpoint stores the model, optimizer, learning-rate scheduler, and mixed-precision scaler state together with a completion flag, so a run can be stopped (kernel death, closing the notebook, Ctrl+C) and resumed later without restarting completed folds from scratch -- a fold cut off mid-training resumes from its last saved epoch rather than epoch 1.

### 2.1.5 Cross-Validated Results

With the revised schedule, all five folds completed training successfully -- a first for this thesis, since only one of five folds completed on CPU (Section 1.1.9).

**Table 10 -- 5-fold cross-validated results (GPU-trained model, loss-selected checkpoints)**

| Fold | Validation AUC |
| :---- | :---- |
| 1 | 0.8821 |
| 2 | 0.8591 |
| 3 | 0.8125 |
| 4 | 0.9511 |
| 5 | 0.8481 |
| **Mean ± SD** | **0.8706 ± 0.0461** |

This is the primary result of this thesis: a genuine 5-fold cross-validated estimate, in contrast to Experiment 1's single-fold result. A direct comparison is possible on Fold 4, since both experiments used an identical stratified split (same random seed, `random_state=42`) and therefore evaluated on the same held-out 310 patients:

**Table 11 -- Direct comparison on the shared Fold 4 split**

| | CPU (Experiment 1) | GPU (Experiment 2) |
| :---- | :---- | :---- |
| Fold 4 AUC | 0.9120 | **0.9511** |
| Folds completed | 1 of 5 | 5 of 5 |
| Reported metric | Single fold | 5-fold mean: **0.8706 ± 0.0461** |

On this shared fold, the GPU-trained model outperforms the CPU-trained model by 0.039 AUC, attributable to the scheduler and batch-size changes above, independent of the cross-validation completeness improvement.

Fig 27 5-Fold Training Log (All Folds Completing)

### 2.1.6 Checkpoint Selection Methodology Experiment

A secondary experiment tested whether checkpoint selection based on maximum validation AUC (rather than minimum validation loss) would yield a more favorable -- or simply different -- result. An identical training run was performed using AUC as both the selection and early-stopping criterion, writing to separate checkpoint files so the original loss-selected run remained untouched for comparison.

**Table 12 -- Loss-selected vs. AUC-selected checkpoint criteria**

| Fold | Loss-selected AUC | AUC-selected AUC |
| :---- | :---- | :---- |
| 1 | 0.8821 | 0.9439 |
| 2 | 0.8591 | 0.9398 |
| 3 | 0.8125 | 0.9265 |
| 4 | 0.9511 | 0.9451 |
| 5 | 0.8481 | 0.9342 |
| **Mean ± SD** | **0.8706 ± 0.0461** | **0.9379 ± 0.0069** |

Although the AUC-selected criterion produced a higher mean score, analysis of the per-epoch training logs showed that every fold trained through all 20 available epochs without triggering early stopping, reaching training accuracies of 97-99% by the final epochs -- a clear indicator of overfitting that a loss-based criterion would have halted earlier. Because the AUC criterion also selects the checkpoint that is the argmax of a noisy metric evaluated on a validation set of only ~310 samples, this method is a biased estimator of true generalization performance. The loss-selected result (0.8706 ± 0.0461) was therefore retained as the primary reported result; this comparison is presented as a methodological contribution regarding checkpoint-selection bias rather than as a competing headline figure.

### 2.1.7 Full Ablation Study

The complete eleven-variant ablation study was executed on the GPU platform as a fresh, independent run -- not a continuation of Experiment 1's partial results (Section 1.1.10) -- re-training all seven deep-learning variants (including the five that were CPU-infeasible: CNN-only, two fusion designs, the single-layer Transformer, and the no-class-weighting variant) and re-fitting all four classical baselines, each evaluated on the same Fold 4 data split used throughout this thesis.

**Table 13 -- Full ablation study results**

| Model | AUC | Accuracy | Macro F1 |
| :---- | :---- | :---- | :---- |
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

As these results are single-fold (n=310) rather than cross-validated, the close scores among the top five variants (within approximately 0.01 AUC of one another) should be interpreted as within normal fold-level variance rather than a definitive architectural ranking.

Two findings are of particular note. First, removing the tabular branch entirely (CNN-only, 0.9474) produced performance statistically indistinguishable from the full multimodal model (0.9486), indicating that the imaging pathway carries the substantial majority of the model's predictive signal in this architecture. Second, consistent with the label-leakage finding of Experiment 1 (Section 1.1.10), classical tree-ensemble methods outperformed the deep tabular-only branch by a wide margin (~0.94 vs. 0.87 AUC) on identical input features.

Fig 28 Ablation Study, AUC Comparison (All 11 Variants)
Fig 29 Ablation Study, Per-Class AUC Grouped Chart

### 2.1.8 Post-Training Evaluation

**Table 14 -- Fold 4 classification performance (validation set, n=310, GPU-trained model)**

| Class | Precision | Recall | F1 | AUC |
| :---- | :---- | :---- | :---- | :---- |
| CN | 0.88 | 0.94 | 0.91 | 0.982 |
| Dementia | 0.85 | 0.84 | 0.85 | 0.958 |
| MCI | 0.82 | 0.78 | 0.80 | 0.913 |
| **Overall** | **0.85** | **0.85** | **0.85** | **0.951** |

Every class improved over the CPU-trained model's Fold 4 result (Table 4): CN AUC rose from 0.957 to 0.982, Dementia from 0.936 to 0.958, and MCI -- the hardest class throughout this thesis -- from 0.844 to 0.913.

Fig 30 Fold 4 Confusion Matrix (GPU-Trained Model)

### 2.1.9 Explainability

SHAP analysis (tabular branch, frozen image embedding, same method as Section 1.1.11) was re-run against the GPU-trained checkpoint.

Fig 31 SHAP Tabular Feature Attribution (GPU-Trained Model)

Grad-CAM was run on the same validation patient used in the CPU section, 057\_S\_1373 (true diagnosis Dementia, correctly predicted Dementia), for direct before/after comparison of the same patient under the CPU- and GPU-trained models. Peak activation moved from (axial z=86, sagittal y=43, coronal x=87) under the CPU model to (axial z=94, sagittal y=75, coronal x=108) under the GPU model -- a visibly different activation region between the two model generations for the identical patient, which is itself worth noting in the write-up rather than treating Grad-CAM output as a fixed property of the patient. As with the CPU section, the actual localization (focal vs. diffuse) should be read directly off the figure before making any claim about anatomical correspondence.

Fig 32 Grad-CAM Activation Overlay, Patient 057\_S\_1373 (Dementia) -- GPU-Trained Model

### 2.1.10 Markov Chain and Digital Twin

The Markov Chain prognostic engine (Section 1.1.12's method) was re-run on this platform after the same visit\_date fix, against the identical 1,549-scan dataset. The resulting matrices are numerically consistent with the CPU section's corrected result, since the Markov chain is computed directly from ADNI visit history and does not depend on which model (CPU- or GPU-trained) is used for classification:

**Table 15 -- Markov transition matrix (full cohort)**

| From \\ To | CN | Dementia | MCI |
| :---- | :---- | :---- | :---- |
| CN | 0.967 | 0.003 | 0.030 |
| Dementia | 0.000 | 0.997 | 0.003 |
| MCI | 0.002 | 0.157 | 0.841 |

**Table 16 -- Markov transition matrix, APOE4-positive**

| From \\ To | CN | Dementia | MCI |
| :---- | :---- | :---- | :---- |
| CN | 0.955 | 0.000 | 0.045 |
| Dementia | 0.000 | 0.995 | 0.006 |
| MCI | 0.004 | 0.184 | 0.811 |

**Table 17 -- Markov transition matrix, APOE4-negative**

| From \\ To | CN | Dementia | MCI |
| :---- | :---- | :---- | :---- |
| CN | 0.972 | 0.004 | 0.024 |
| Dementia | 0.000 | 1.000 | 0.000 |
| MCI | 0.000 | 0.133 | 0.867 |

Dementia behaves as an effectively absorbing state (self-transition probability 0.995--1.00 across all three strata), and APOE4-positive patients show a higher MCI-to-Dementia transition probability (0.184) than APOE4-negative patients (0.133) -- the clinically expected direction. These figures match the CPU section's corrected Tables 6-8 exactly, which is itself a useful internal consistency check: two independent runs of the same fixed pipeline, on two different machines, produced identical transition matrices.

Fig 33 Markov Transition Matrix Heatmap (Full Cohort, GPU Run)

### 2.1.11 What-If Simulation

The Digital Twin's what-if simulation (Section 7.7 of the proposal) was run for validation patient 016\_S\_1149 (diagnosed MCI, APOE4-negative) -- the same patient examined in the CPU section's What-if result (Table 9), enabling a direct before/after comparison under the two model generations.

**Table 18 -- What-if simulation, Patient 016\_S\_1149 (5-year Dementia probability, GPU-trained model)**

| Scenario | P(Dementia) at Year 5 |
| :---- | :---- |
| Baseline (APOE4-negative) | 14.5% |
| APOE4-positive (no treatment) | 25.2% |
| APOE4-positive \+ Lecanemab (30%) | 12.1% |
| APOE4-positive \+ Donanemab (35%) | 11.9% |

Every simulated effect remains in the clinically expected direction (APOE4 increases risk, both treatments decrease it), matching the CPU result's pattern. The absolute risk estimates differ substantially between the two model generations, however -- the CPU-trained model estimated this same patient's baseline 5-year Dementia risk at 54.9% (Table 9), while the GPU-trained model estimates 14.5%. This is a large enough gap to discuss explicitly rather than gloss over: it reflects the underlying classifier's diagnosis probabilities for this patient at Year 0 (which seed the Monte Carlo simulation), and the GPU-trained model is the more accurate of the two per Table 11 -- but a five-fold difference in a single patient's headline risk number between two "versions" of the same pipeline is worth a sentence of honest caveat in the thesis about single-patient case studies not being a robustness demonstration on their own.

Fig 34 What-If Simulation, Patient 016\_S\_1149 (Baseline / APOE4 / Lecanemab / Donanemab trajectories) -- GPU-Trained Model

### 2.1.12 Additional Digital Twin Validation

Three further checks were run to probe the Digital Twin's behavior beyond a single patient, none of which were feasible within the CPU phase's scope.

**Age sensitivity.** An initial single-patient age sweep (65-85 years) produced a counter-intuitive result: predicted Dementia risk *decreasing* with increasing age. Rather than accept this at face value, the check was repeated across five independent MCI patients:

**Table 19 -- Age-sensitivity check, 5-year Dementia probability by simulated age (65-85)**

| Patient | 65 | 70 | 75 | 80 | 85 | Direction |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 016\_S\_1149 | 71.3% | 51.3% | 29.6% | 19.0% | 15.3% | decreases |
| 099\_S\_0880 | 88.8% | 82.5% | 77.3% | 74.7% | 70.1% | decreases |
| 057\_S\_1007 | 77.5% | 74.6% | 79.7% | 75.2% | 78.7% | increases |
| 109\_S\_1114 | 75.7% | 75.0% | 74.5% | 76.9% | 79.6% | increases |
| 128\_S\_0167 | 100.0% | 100.0% | 99.1% | 97.8% | 95.0% | decreases |

Two of five patients showed the clinically expected increasing-risk direction. This is consistent with the AGE tabular feature carrying a comparatively weak and noisy signal relative to the dominant imaging pathway -- directly supported by Section 2.1.7's finding that removing the tabular branch entirely (CNN-only, 0.9474) barely changes performance versus the full model (0.9486) -- rather than indicating a systematic error in the simulation methodology. The direction varying by patient, rather than being uniformly wrong, is what distinguishes a noisy weak feature from a code defect.

Fig 35 Age-Sensitivity Analysis, Patient 016\_S\_1149

**Population subgroup comparison.** The Digital Twin was run on 10 representative patients from each diagnostic class to validate population-level behavior:

**Table 20 -- Mean 5-year Dementia probability by diagnostic subgroup (n=10 per group)**

| Diagnostic group | Mean P(Dementia) at Year 5 |
| :---- | :---- |
| CN | 26.3% ± 18.7% |
| MCI | 64.4% ± 23.3% |
| Dementia | 95.1% ± 5.0% |

The monotonic ordering (CN \< MCI \< Dementia) is the expected population-level pattern and was not explicitly enforced by the model or the simulation -- it emerges from the combination of the classifier's diagnosis probabilities and the Markov transition matrix.

Fig 36 Population Subgroup Trajectory Comparison (CN / MCI / Dementia)

**Early vs. late intervention timing.** Using time-varying transition matrices, the same APOE4-positive MCI patient (128\_S\_0167) was simulated receiving Lecanemab starting at Year 0 versus Year 2:

**Table 21 -- Early vs. late intervention timing, Patient 128\_S\_0167 (5-year Dementia probability)**

| Scenario | P(Dementia) at Year 5 |
| :---- | :---- |
| No treatment | 97.3% |
| Lecanemab starting Year 2 | 96.5% |
| Lecanemab starting Year 0 | 95.7% |

Earlier intervention produces a lower simulated risk, the clinically expected direction, though the absolute effect is modest for this particular patient because their baseline risk was already very high (97.3%) by the time of assessment -- a ceiling effect worth noting rather than presenting the 1.6-percentage-point gap as a strong treatment-timing effect. This simulation is highlighted in the notebook as the most novel capability in this thesis: no published static ADNI classifier can produce a time-varying intervention-timing projection, since that requires the combination of a trained classifier and a Markov progression model that neither component provides alone.

Fig 37 Early vs. Late Intervention Timing, Patient 128\_S\_0167

### 2.1.13 Clinical Dashboard Deployment

The GPU-trained model and associated Markov transition matrices were integrated into a Streamlit-based clinical dashboard, containerized and deployed to Azure App Service. During this integration, three functional defects were identified and resolved prior to production release:

1. A checkpoint-loading incompatibility with current PyTorch versions (missing an explicit flag required because the checkpoint embeds a fitted scikit-learn scaler object, which newer PyTorch versions reject loading by default) -- the same class of defect independently re-discovered and fixed in the CPU notebook.
2. A label-mapping orientation mismatch between the two model generations' saved checkpoint metadata, which caused inference to fail on every request against the newer checkpoint format.
3. A silent-failure mode in which a patient with no available cached MRI scan received a full-confidence diagnostic prediction generated from a blank image tensor, with no indication to the clinician that the imaging pathway had not been used. This was resolved by explicitly tracking scan availability and surfacing an unmissable warning wherever such a prediction occurs -- the same safeguard is visible directly in the notebook's Cell 33 output (Section 2.1.12's custom-patient demo), which correctly triggers this exact warning when no cached scan is found.

Following these fixes, the deployed model was validated against a genuine held-out ADNI patient record not used during any training or validation step in this thesis, correctly predicting the patient's true diagnosis with near-full confidence, with the reported risk-trajectory and genotype-effect figures matching hand-computed values derived independently from the underlying transition matrices.

Fig 38 Deployed Dashboard, Prediction Output for the Held-Out Verification Patient
Fig 39 Deployed Dashboard, "About" / Model-Info Panel

### 2.1.14 Limitations of This Phase

Three points limit or qualify this phase's results. First, Table 13's full ablation study is single-fold (n=310) rather than cross-validated, so the close ranking among the top five variants should not be read as a definitive architectural conclusion. Second, the AGE tabular feature's weak and direction-inconsistent signal (Section 2.1.12) limits how much clinical weight the age-sensitivity capability can be given until a larger patient sample or an additional cognitive-score feature (e.g., ADAS-Cog, CDRSB) strengthens that pathway. Third, the What-if simulation's large absolute divergence between the CPU- and GPU-trained models for the same patient (Section 2.1.11) is a reminder that a single-patient case study demonstrates the Digital Twin's *capability*, not its *robustness* -- a population-level validation (as in Section 2.1.12's subgroup comparison) is the more defensible evidence for the latter.
