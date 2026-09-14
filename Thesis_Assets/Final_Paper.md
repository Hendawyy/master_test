# Brain Digital Twin (BDT): A Hybrid 3D CNN-Transformer Framework with Markov-Based Prognostic Simulation for Personalized Alzheimer's Disease Progression Modeling

**Seif Hendawy**

Arab Academy for Science, Technology and Maritime Transport (AASTMT), Sheraton Branch, Cairo, Egypt

**Supervised by:** Prof. Fahima Maghraby, Assoc. Prof. Ahmed Salem

## Abstract

Alzheimer's Disease (AD) affects over 55 million people worldwide, a figure projected to reach 139 million by 2050, yet current AI-based diagnostic tools remain static classifiers that cannot answer the patient-specific "what-if" questions the new generation of disease-modifying therapies (DMTs) demands. This work designs, implements, and validates a Brain Digital Twin (BDT): a hybrid deep-learning and simulation framework that combines a 3D DenseNet-121 convolutional backbone with a Transformer encoder for multimodal Alzheimer's classification, and an empirically estimated Markov Chain progression model for personalized, simulative "what-if" forecasting. The BDT was developed and validated across two sequential experimental phases sharing an identical codebase and architecture. Experiment 1, on CPU-only Azure Machine Learning compute, established the full data pipeline end to end, achieved a single-fold validation AUC of 0.912, completed five of eleven planned ablation-study variants, and validated proof-of-concept explainability and progression-modeling components, while also surfacing two hard constraints of CPU-only compute: an out-of-memory failure after twelve hours on the full ablation study, and only one of five cross-validation folds completing due to training instability. Experiment 2, migrated to a GPU workstation with a revised training schedule, removed both constraints. All five folds completed in approximately one hour, yielding the thesis's primary result of AUC 0.8706 ± 0.0461; the complete eleven-variant ablation study finished in under four hours; and the framework was extended to population-level Digital Twin validation, multi-patient age-sensitivity analysis, a novel early-versus-late intervention-timing simulation, and deployment to a production clinical dashboard on Azure App Service, validated against a genuine held-out patient. Across both experiments, per-class validation AUC reached up to 0.982 (Cognitively Normal), 0.958 (Dementia), and 0.913 (Mild Cognitive Impairment). SHAP and Grad-CAM explainability, together with an APOE4-stratified Markov transition matrix consistent with the known clinical direction of APOE4 as a progression risk factor, support the framework's clinical interpretability and validity. These results establish the BDT as a working, end-to-end, and clinically interpretable simulative alternative to static AD classifiers, ready for the interactive clinical deployment phase.

**Keywords:** Brain Digital Twin · Alzheimer's Disease · Multimodal deep learning · 3D Convolutional Neural Network · Transformer · Markov Chain · Explainable AI · SHAP · Grad-CAM · ADNI

---

## 1 Introduction

Alzheimer's Disease (AD) is a progressive neurodegenerative disorder and one of the most significant public health crises of the 21st century. Defined by the pathological accumulation of amyloid-beta plaques and neurofibrillary tangles, the disease leads to a relentless decline in cognitive function, memory, and the ability to perform daily activities. Despite the availability of large-scale, longitudinal research datasets like the Alzheimer's Disease Neuroimaging Initiative (ADNI), AD remains incurable, so there is a continuing need for advanced computational tools to improve diagnosis, monitoring, and treatment planning.

A primary obstacle in managing AD is its profound heterogeneity. The disease does not follow a uniform path; it manifests with considerable variability in age of onset, rate of cognitive decline, and individual biomarker expression. This complexity challenges traditional "one-size-fits-all" approaches to care. In response, the concept of the Digital Twin (DT), a dynamic, virtual, and personalized model of a patient, has emerged from engineering as a transformative paradigm for personalized medicine. A Health Digital Twin integrates multimodal data (imaging, genetics, clinical scores) and simulates time-dependent biological processes and disease progression, enabling virtual experimentation with hypothetical interventions.

### 1.1 Motivation

The motivation for developing a Brain Digital Twin for Alzheimer's is rooted in a global health crisis that has reached a critical point, driven by large societal costs and a recent shift in therapeutic possibilities.

**The scale of the Alzheimer's crisis.** According to the World Health Organization, over 55 million people worldwide are living with dementia, with AD being the most common cause [21]. This number is projected to reach an estimated 139 million by 2050 [22]. The economic impact is equally severe: the total global cost of dementia is expected to surpass $2.8 trillion USD by 2030. This escalating crisis places an immense burden on patients, families, and healthcare systems, and it demands solutions that can change how the disease is monitored and managed.

**The new therapeutic landscape.** For decades, treatment for AD was limited to symptomatic drugs, such as cholinesterase inhibitors (e.g., Donepezil), which do not halt the underlying neurodegenerative process [23]. The field has since changed with the arrival of disease-modifying therapies (DMTs): drugs like Lecanemab (Leqembi), designed to clear amyloid plaques, have shown in clinical trials a statistically significant slowing of cognitive decline by approximately 27% over 18 months compared to placebo [24]. The arrival of these powerful but complex DMTs is the central motivation for this work. Their varied efficacy, potential side effects, and high cost mean a generic diagnosis is no longer sufficient. Clinicians now need a way to combine a patient's own data to simulate and compare potential futures under different intervention scenarios, a capability current systems lack.

### 1.2 Problem Statement

The fundamental problem addressed here is that current artificial intelligence tools for Alzheimer's are static classifiers, while the clinical need in the era of disease-modifying therapies is for dynamic, personalized simulators [10, 11, 25]. This disconnect creates a gap between existing technology and the requirements of modern clinical practice [26, 27], which breaks down into several specific limitations of current approaches:

- **Static predictions.** Most AI systems provide a one-time diagnosis or risk score and cannot model the continuous, longitudinal disease trajectories that define a patient's journey.
- **Lack of personalization.** Models are typically trained on population-level data and cannot dynamically adapt to an individual's unique combination of neuroimaging features, biomarkers, genetic risk factors (e.g., APOE4 status), or treatment history.
- **Fragmented data integration.** Many models use multimodal data without a cohesive framework for integrating these streams into a single, dynamic model that evolves over time.
- **Limited simulation capabilities.** Existing models are predictive, not simulative; they cannot answer "what-if" questions essential for personalized treatment planning, such as forecasting a patient's progression with versus without a specific medication.
- **"Black-box" nature.** Many deep learning models are opaque, making it difficult for clinicians to trust their outputs without clear, visual explanations linking a prediction back to the underlying patient data.
- **No feedback loop.** Models do not adapt to evolving patient data.

### 1.3 Objectives and Contributions

The primary objective of this work is to design, implement, and validate a novel Brain Digital Twin (BDT) framework for the personalized simulation of Alzheimer's Disease progression. This overarching goal breaks down into five specific objectives:

1. **Develop a unified multimodal framework**: an end-to-end computational pipeline that ingests, harmonizes, and integrates multimodal data from ADNI, including 3D neuroimaging, clinical assessments, cognitive scores, and genetic biomarkers.
2. **Implement a state-of-the-art diagnostic model**: a hybrid deep learning model using a 3D Convolutional Neural Network for spatial feature extraction from MRI scans and a Transformer-based architecture for robust, multi-class disease-stage classification.
3. **Create a data-driven prognostic engine**: a probabilistic progression model, based on Markov Chains, to simulate longitudinal disease-state transitions from real patient data, forming the temporal core of the BDT.
4. **Ensure clinical interpretability**: multi-level explainability using SHAP to interpret tabular feature contributions and Grad-CAM to generate visual explanations of the neuroimaging features driving diagnostic decisions.
5. **Demonstrate clinical utility through simulation**: "what-if" simulations for clinically relevant use cases, including the impact of genetic risk factors (APOE4) and the potential effects of pharmacological interventions on individual patient trajectories.

Beyond the originally proposed scope, this work makes two further contributions realized during implementation: a systematic **ablation study** (eleven architectural and classical-baseline variants) quantifying the contribution of each component of the BDT architecture, which was not present in the original framework design; and a direct, controlled **CPU-versus-GPU experimental comparison** using the identical codebase and data splits, characterizing exactly what compute-platform migration changes and does not change about the resulting model. Table 1 summarizes where the as-built system departs from the originally proposed framework, for full transparency in the results that follow.

**Table 1. Implementation deviations from the proposed framework**

| Proposed | Implemented | Reason |
|---|---|---|
| Dataset sources: MRIs, ADNIMERGE, RECCMEDS | MRIs + ADNIMERGE only | The medications log (RECCMEDS.csv) was not integrated; simulated drug intervention uses fixed literature-informed effect sizes (Lecanemab 30%, Donanemab 35% reduction in MCI→Dementia transition probability) rather than per-patient medication history. |
| Hidden Markov Chains for progression modeling | Empirical (fully-observed) Markov Chain, APOE4-stratified | Transition probabilities were estimated directly from observed diagnosis sequences across longitudinal visits. No latent-state HMM training was performed, since diagnostic state is directly observed at each visit rather than hidden. |
| Explainability Module: SHAP, Grad-CAM, Attention Maps | SHAP and Grad-CAM only | Transformer attention-weight visualization was not implemented; given the architecture fuses image and tabular features into a single token before the Transformer encoder (Section 3.4), attention-map visualization would be degenerate for a sequence length of one. |
| 3D Brain Visualization (atrophy/activation maps) | 2D multi-planar Grad-CAM overlays (axial/sagittal/coronal) | The dashboard renders orthogonal slice views with heatmap overlay rather than an interactive volumetric 3D render. |
| Ablation study | Not in the original framework diagram | Added as a substantial methodological contribution, executed in two stages across both experiments (Sections 5.1.10 and 5.2.7). |

The remainder of this paper is organized as follows: Section 2 covers Related Work, Section 3 covers Methodology (including the Digital Twin and end-to-end training algorithms), Section 4 covers Dataset and Description, Section 5 covers Experiments and Result Analysis (both the CPU and GPU experimental phases in full), Section 6 covers Discussion (synthesizing findings across both experiments, comparing them with prior work, and consolidating limitations), and Section 7 covers Conclusion and Future Work.

---

## 2 Related Work

This section reviews the literature forming the foundation for the proposed Brain Digital Twin. Following the format used in the original project proposal, each study is summarized in a structured table listing its objectives, methodology and findings, and its specific contribution to this thesis. The review follows the research trajectory that motivated this work: starting with broader applications of predictive AI in medicine, narrowing to the Digital Twin paradigm, and finally focusing on state-of-the-art deep learning and explainability techniques in Alzheimer's Disease research specifically.

### 2.1 Predictive Modeling and Interpretability in General Medicine

The foundation of the BDT lies in successful applications of machine learning for clinical decision support in other high-stakes domains.

| Authors / Year | Title | Objectives | Methodology / Findings | Contribution to This Thesis |
|---|---|---|---|---|
| Salaün et al., 2024 [1] | Predicting graft and patient outcomes following kidney transplantation using interpretable ML models | Develop interpretable ML models for long-term kidney-transplant outcomes | Clinical and demographic data; applied SHAP for explainability; identified key predictors | Directly motivates this work's own use of SHAP (Section 3.6) |
| Yoo et al., 2024 [2] | A machine learning-driven virtual biopsy system for kidney transplant patients | Build a non-invasive "virtual biopsy" predictor of lesion scores | Ensemble model (XGBoost, GBM, neural network); high accuracy for biopsy prediction | Validates gradient-boosted tree ensembles as a strong tabular baseline (Section 3.8 ablation) |
| Zhang et al., 2022 [3] | AdaDiag: adversarial domain adaptation of diagnostic prediction with clinical event sequences | Adapt predictive models across heterogeneous clinical datasets | Domain-adversarial training; improved cross-dataset generalization | Supports the broader case for transfer learning and domain adaptation in clinical AI |
| Awuah et al., 2023 [4] | Recent outcomes and challenges of AI, ML, and DL in neurosurgery | Review AI/ML/DL applications in neurosurgery | Identified bias, explainability, and data limitations as recurring barriers | Reinforces the need for explainability (SHAP, Grad-CAM) and robust validation built into the BDT from the outset |

### 2.2 The Digital Twin Paradigm: From General Concepts to Healthcare

Predictive models are powerful, but Digital Twins extend the paradigm by adding dynamic simulation. The following works establish the DT concept and its adaptation in healthcare.

| Authors / Year | Title | Objectives | Methodology / Findings | Contribution to This Thesis |
|---|---|---|---|---|
| VanDerHorn & Mahadevan, 2021 [5] | Digital Twin: generalization, characterization and implementation | Define the DT concept across engineering domains | DT as a dynamic virtual representation with continuous physical-to-virtual data flow | Provides the foundational DT definition adapted for the BDT (Section 3.2) |
| Venkatesh et al., 2022 [6] | Health digital twins as tools for precision medicine | Explore Health Digital Twins for precision medicine | Integrated multimodal data; highlighted computational and regulatory challenges | Connects the DT concept to personalized medicine, a key motivation for the BDT |
| Gaebel et al., 2021 [7] | The Digital Twin: modular model-based approach to personalized medicine | Propose a modular DT architecture | RDF-based EHR integration; explainable and extensible | Informs a modular, stream-based design philosophy |
| Coorey et al., 2022 [8] | The health digital twin to tackle cardiovascular disease | Apply the DT concept to cardiovascular disease management | Simulated systemic circulation; optimized interventions | Domain-specific precedent for a simulative DT, analogous to the brain |
| Chakshu et al., 2021 [9] | Towards enabling a cardiovascular digital twin for human systemic circulation using inverse analysis | Build a patient-specific circulation DT | Inverse modeling from pressure signals; predicted aneurysms | Demonstrates the feasibility of a data-driven, simulative DT |

### 2.3 Focusing on Alzheimer's Disease: AI Classification and Early Digital Twins

This subsection reviews AI-based AD classification and early AD-focused Digital Twin approaches directly, including the DT concept's own AD-specific precedents.

| Authors / Year | Title | Objectives | Methodology / Findings | Contribution to This Thesis |
|---|---|---|---|---|
| Bertolini et al., 2020 [10] | Modeling disease progression in mild cognitive impairment and Alzheimer's Disease with digital twins | Generate synthetic "Digital Twin" clinical records predicting standard-of-care disease progression | Conditional Restricted Boltzmann Machines modeling MMSE/ADAS-Cog trajectories over time (ADNI and CODR-AD datasets) | Direct validation of the AD-focused, simulative DT concept this work adopts |
| Ravi et al., 2022 [29] | Degenerative adversarial neuroimage nets for brain scan simulations: application in ageing and dementia | Generate high-resolution, personalized synthetic MRI sequences simulating patient-specific brain atrophy | GAN-based "4D-DANI-Net" generating synthetic future MRI from a baseline scan (ADNI training/testing, OASIS-3 external validation); synthetic follow-up images were visually indistinguishable from real ones; uni-modal (MRI only), does not simulate clinical/cognitive scores alongside imaging | Demonstrates that visual disease-progression simulation is feasible, but, being uni-modal and disconnected from clinical scores, illustrates precisely the fragmentation this work's integrated, multimodal DT addresses (Section 2.4) |
| Basaia et al., 2019 [12] | Automated classification of Alzheimer's Disease and MCI using a single MRI and deep neural networks | Develop a fully automated AD/MCI classifier | 3D CNN trained on ADNI MRI; high accuracy | Validates 3D CNNs for imaging feature extraction (Section 3.4) |
| Wen et al., 2020 [13] | Convolutional neural networks for classification of Alzheimer's Disease: overview and reproducible evaluation | Benchmark multiple CNN architectures on ADNI | Compared architectures via the Clinica platform; established reproducibility benchmarks | Informs this work's model-selection and cross-validation practices |
| Qiu et al., 2020 [17] | Development and validation of an interpretable deep learning framework for AD classification | Improve interpretability of CNN-based AD classifiers | 3D CNN combined with Grad-CAM; resulting heatmaps confirmed clinical relevance of the highlighted brain regions | Directly justifies this work's use of Grad-CAM (Section 3.6) |
| Venugopalan et al., 2021 [16] | Multimodal deep learning models for early detection of Alzheimer's Disease stage | Fuse multimodal patient data | Combined MRI, PET, and cognitive data; multimodal fusion outperformed any single modality | Validates the multimodal integration central to the BDT's design |
| Kushol et al., 2022 [14] | ADDformer: Alzheimer's Disease detection from structural MRI using fusion Transformer | Design a Transformer-based AD detection model | Captured global neuroimaging relationships | Supports the Transformer component of this work's hybrid CNN-Transformer classifier |
| Hu et al., 2023 [15] | Conv-Swinformer: integration of CNN and shift window attention for Alzheimer's Disease classification | Combine CNN local features with Transformer global context | CNN (VGG16) texture features fed into a Swin Transformer; 93.56% accuracy (AD vs. CN) on ADNI, externally validated at 92.31% on OASIS | Directly supports this work's hybrid CNN-Transformer design, and demonstrates the value of external validation |
| Zhou et al., 2025 [28] | A deep learning model for early diagnosis of Alzheimer's Disease combined with 3D CNN and video Swin Transformer | Design a hybrid 3D-CNN-VideoSwinFormer model for early AD diagnosis | 3D CNN + CBAM attention + Video Swin Transformer; 92.92% accuracy, 0.96 AUC (AD vs. CN); emphasized a fully 3D approach to avoid data leakage | Strong contemporary validation of the hybrid CNN-Transformer, 3D-first direction this work adopts; direct benchmark comparison point (Section 5.1.7) |
| Lozupone et al., 2024 [11] | AXIAL: attention-based explainability for interpretable Alzheimer's localized diagnosis using 2D CNNs on 3D MRI brain scans | Develop an explainable CNN model for MRI-based AD diagnosis | 2D CNNs over sliced 3D MRI (axial/coronal/sagittal), fused via soft attention; 85.6% accuracy on ADNI, validated with Grad-CAM and 3D attention maps; limited to MRI only, no multimodal integration | Reinforces the explainability and attention-based design elements adopted here |
| Sarkar, 2025 [25] | Integrating machine learning and deep learning techniques for advanced Alzheimer's Disease detection through gait analysis | Explore gait analysis as a non-invasive AD biomarker | Hybrid CNN-RNN (LSTM) on wearable-sensor gait data; 93% accuracy, 95% AUC, outperforming traditional ML; limited dataset, no neuroimaging validation | Broadens the BDT's context to functional biomarkers beyond imaging |
| Hechkel & Helali, 2025 [26] | Unveiling Alzheimer's Disease early: a comprehensive review of machine learning and imaging techniques | Provide a comprehensive review of ML and neuroimaging techniques for AD | Surveys a wide range of models; highlights CNN-Transformer hybrids as state-of-the-art | Validates this work's overall architectural and multimodal direction |
| Aghdam et al., 2025 [27] | Machine-learning models for Alzheimer's Disease diagnosis using neuroimaging data: survey, reproducibility, and generalizability evaluation | Examine the reproducibility and generalizability of ML models for AD | Stress-tested open-source models (2D CNN, 3D ConvLSTM) trained on ADNI and tested on OASIS; internal accuracy of 97.9% dropped to approximately 67% externally, revealing a severe generalizability gap | Supports this work's methodological emphasis on stratified K-fold cross-validation and on honestly reporting single-fold versus cross-validated results (Section 6.3) rather than a headline internal accuracy alone |
| Falahati et al., 2014 [18] | Multivariate data analysis and machine learning in Alzheimer's Disease with a focus on structural MRI | Review pre-deep-learning ML approaches in AD | Highlighted SVMs and handcrafted features | Provides a historical baseline for the evolution to deep learning |
| Dosovitskiy et al., 2020 [19] | An image is worth 16x16 words: Transformers for image recognition at scale | Introduce the Vision Transformer | Self-attention rivals CNNs for image modeling | Theoretical foundation for the Transformer backbone used here (Section 3.4) |
| Selvaraju et al., 2017 [20] | Grad-CAM: visual explanations from deep networks via gradient-based localization | Improve CNN interpretability | Introduced class-discriminative, gradient-based heatmaps | Foundational explainability method integrated directly into the BDT (Section 3.6) |

### 2.4 Summary and Research Gap

The reviewed literature demonstrates a clear progression: (1) interpretable ML provides actionable, personalized clinical support across medical domains; (2) the Digital Twin paradigm, originating in engineering, enables dynamic simulation beyond static prediction; and (3) Alzheimer's Disease research has evolved from SVM-based classifiers to powerful deep learning models, with hybrid CNN-Transformer architectures representing the current state of the art, alongside explainability tools such as SHAP and Grad-CAM.

**Research gap.** Current AD Digital Twin implementations are fragmented and rely on comparatively simple progression models. Some, like Bertolini et al. [10], simulate clinical/cognitive scores but not imaging; others, like Ravi et al. [29] (table above), simulate imaging but not clinical scores. State-of-the-art multimodal deep learning classification methods have not, in prior work, been embedded into a single dynamic, simulative Digital Twin capable of personalized "what-if" intervention analysis across both modalities at once. This work addresses that gap by integrating a state-of-the-art hybrid CNN-Transformer diagnostic engine, trained jointly on imaging and clinical/genetic tabular data, with a robust, data-driven Markov simulation model for prognostic and simulative capabilities. Together these components form the Brain Digital Twin: an end-to-end, multimodal system intended to be more clinically useful than a diagnostic classifier alone.

---

## 3 Methodology

This section presents the unified, end-to-end framework for the Brain Digital Twin. It first establishes the theoretical foundations for each component (Sections 3.1-3.3), then describes the as-built architecture, training procedure, and explainability pipeline exactly as implemented (Sections 3.4-3.7), and finally formalizes the framework as two algorithms with a complexity analysis (Section 3.8). The BDT was implemented once, as a single codebase, and executed identically (Section 5) on two compute platforms. The only differences between the two experimental runs are in compute hardware and, where explicitly noted, training hyperparameters changed in direct response to instability observed in the first run.

### 3.1 Problem Formulation

Given a patient's 3D T1-weighted structural MRI volume $I \in \mathbb{R}^{D \times H \times W}$ and a vector of tabular clinical features $x \in \mathbb{R}^{4}$ (age, years of education, MMSE score, APOE4 allele count), the diagnostic task is a three-class classification problem over the label space $\mathcal{Y} = \{\text{CN}, \text{MCI}, \text{Dementia}\}$, learning a function $f_\theta(I, x) \rightarrow p(y \mid I, x)$. The prognostic task, given the classifier's current-state probability distribution $p_0 = f_\theta(I, x)$ and an empirically estimated state-transition matrix $P \in \mathbb{R}^{3 \times 3}$, is to forecast a distribution over health states at future time horizons $t = 1, \ldots, T$, optionally under a counterfactual intervention that modifies $P$.

### 3.2 The Digital Twin Concept

The Digital Twin (DT) originates in industrial engineering, where it is defined as a dynamic, virtual representation of a physical asset or system that is continuously updated with real-world data. The concept relies on a bidirectional flow of information: the physical object provides data to the virtual model, and the virtual model simulates, predicts, and optimizes the behavior of the physical counterpart. In healthcare, a Health Digital Twin (HDT) is a personalized computational model of an individual patient that integrates heterogeneous data sources, such as medical imaging, biomarkers, genetics, and longitudinal clinical records, into a holistic, in-silico replica. Unlike a static predictive model, a DT emphasizes simulation: forecasting future health states and testing hypothetical interventions (e.g., a new medication) prior to real-world application. The Brain Digital Twin (BDT) instantiates this concept for Alzheimer's Disease. Each patient's data generates a personalized, explainable model capable of forecasting trajectories, testing "what-if" interventions, and supporting clinical decision-making.

### 3.3 Theoretical Foundations of the Core Components

**3.3.1 3D Convolutional Neural Networks.** CNNs are the standard approach for computer vision and medical image analysis, automatically learning spatial hierarchies of features from grid-like data. For a 3D input volume $I$ and kernel $K$, the convolution at voxel $(x,y,z)$ is

$$(I * K)(x,y,z) = \sum_i \sum_j \sum_k I(x-i,\, y-j,\, z-k)\, K(i,j,k)$$

followed by a non-linear activation (this work uses the standard ReLU, $f(x) = \max(0, x)$, throughout the backbone) and pooling for downsampling and shift-robustness. For this work, a 3D CNN is applied to volumetric MRI data so that the model captures complex anatomical structures and spatial relationships associated with Alzheimer's progression, rather than processing 2D slices independently.

**3.3.2 The Transformer Architecture and Self-Attention.** The Transformer architecture, originally designed for natural language processing, has since been adapted to computer vision. Unlike CNNs, which capture local features, Transformers use self-attention to model global dependencies. For a sequence of input vectors, each element is projected into Query ($Q$), Key ($K$), and Value ($V$) representations, and the attention scores are computed as

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right) V$$

where $d_k$ is the dimension of the key vectors. Within the BDT, the Transformer encoder provides the mechanism for multimodal fusion between the imaging embedding and the tabular clinical features (Section 3.4).

**3.3.3 Explainable AI (XAI).** To avoid a "black-box" system, two complementary explainability methods are used. SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory: it explains a prediction by assigning each input feature a Shapley value representing its contribution to shifting the model's output from a baseline (average) prediction to the actual prediction, providing feature-level interpretability for the tabular branch. Grad-CAM (Gradient-weighted Class Activation Mapping) is designed for CNN-based models: for feature map $A^k$ of the target convolutional layer, the class-specific importance weight is

$$\alpha_k^{c} = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^{c}}{\partial A_{ij}^{k}}$$

and the resulting localization map is

$$L^{c}_{\mathrm{Grad\text{-}CAM}} = \mathrm{ReLU}\!\left(\sum_k \alpha_k^{c} A^{k}\right)$$

which is then upsampled to the input resolution and overlaid on the original volume, producing a visual explanation of which anatomical regions drove a given prediction.

**3.3.4 Markov Chains for Disease Progression.** A Markov Chain provides a stochastic framework for modeling disease progression across a discrete set of states $S = \{s_1, \ldots, s_r\}$, where transitions are governed by a probability matrix $P$, with $P_{ij}$ the probability of transitioning from $s_i$ to $s_j$, subject to $\sum_j P_{ij} = 1\;\forall i$. Using longitudinal ADNI visit data, transition probabilities between diagnostic categories (CN, MCI, Dementia) are empirically estimated directly from observed diagnosis sequences, rather than fitted as a Hidden Markov Model with latent states, since diagnostic state is directly observed at each visit (see Table 1). This enables simulation of disease trajectories that complements the deep-learning classifier with an interpretable, probabilistic model of progression.

### 3.4 As-Built Model Architecture

The classifier, `MultimodalTransformer` (24,012,643 parameters), combines a 3D DenseNet-121 backbone with the four tabular clinical features described in Section 3.1. The backbone is used purely as a feature extractor (`out_channels=1024`, no classification head of its own), producing a 1024-dimensional image embedding per scan. This embedding is concatenated with the four (standardized) tabular features and projected through a linear layer, LayerNorm, and GELU activation into a fixed-width representation whose dimension is the nearest multiple of the Transformer's head count (computed via ceiling-rounded integer arithmetic, which resolves an initial dimension mismatch between the 1,028-dimensional concatenated vector and the 8-head requirement). The projected, fused representation, a single token, is passed through a 2-layer Transformer encoder (8 attention heads, pre-normalization) and a final linear classification head over the three diagnostic classes.

Two implementation-level defects were identified and fixed relative to an initial draft of the architecture. First, the DenseNet-121 backbone was originally invoked in a way that both extracted features and implicitly forced a classification output, conflating feature extraction with classification. Second, a batch-squeeze operation used in average pooling collapsed the batch dimension entirely when batch size was 1, since PyTorch's `.squeeze()` removes all singleton dimensions. Both were corrected: the backbone is now a pure 1024-d feature extractor, and pooling uses `.flatten(1)` instead of an unconditional `.squeeze()`.

Because the fused image-plus-tabular representation reaches the Transformer encoder as a single token (sequence length one), the Transformer's self-attention operates over a degenerate one-element sequence. This is the architectural reason, noted in Table 1, that Transformer attention-map visualization was not pursued as an explainability method: attention weights over a sequence of length one carry no information to visualize.

### 3.5 Training Configuration

Training uses stratified $K$-fold cross-validation ($K=5$, fixed random seed, identical splits across both experimental phases so that any given fold's held-out set is the same physical patients on CPU and GPU) with AdamW optimization and class-weighted cross-entropy loss, the class weights computed via `compute_class_weight('balanced')` over the full training-fold label distribution to counteract ADNI's typical MCI-heavy imbalance. The `StandardScaler` fitted to the tabular features is fitted independently inside each fold, on the training partition only, to avoid the data leakage that would result from fitting on the full dataset before splitting. The specific hyperparameters (batch size, learning-rate schedule, epoch budget, early-stopping patience, mixed precision) differ between the CPU and GPU experimental phases and are given in full, with the rationale for each change, in Section 5.

### 3.6 Explainability Pipeline

SHAP is applied to the tabular branch with the imaging embedding held frozen (a full SHAP analysis over the imaging branch was found to exceed available system memory), producing per-feature attribution values for each diagnostic class. Grad-CAM is hooked on the final dense block of the DenseNet-121 backbone (`denseblock4`). Computing it requires the input tensor's gradient tracking to be explicitly re-enabled before the backward pass, since this is not the default state when running inference with an already-evaluated model. The resulting class-activation map is upsampled to the full $128^3$ volume resolution and displayed as axial, sagittal, and coronal overlays.

### 3.7 Markov Chain Estimation and Digital Twin Assembly

For each patient, longitudinal visits are grouped by patient identifier and sorted chronologically by visit date. Observed state-to-state transitions (e.g., $\mathrm{CN} \rightarrow \mathrm{MCI}$) are counted across all patients and all consecutive visit pairs, and each row of the resulting count matrix is normalized to a probability distribution, yielding the full-cohort transition matrix $P$. The same procedure is repeated on the APOE4-positive and APOE4-negative subgroups independently, yielding $P^{+}$ and $P^{-}$, which enable the APOE4 "what-if" simulation (Objective 5).

The Digital Twin itself (`NeuroDT`) combines the trained classifier $f_\theta$ with the Markov transition matrices to produce, for a given patient, a current-state diagnosis distribution and a Monte Carlo-simulated multi-year trajectory with 95% confidence intervals, optionally under a simulated pharmacological intervention that modifies the relevant transition-matrix entries before simulation. The full procedure is given as Algorithm 2 (Section 3.8).

### 3.8 Algorithm and Complexity

**Algorithm 1** summarizes the end-to-end BDT training and evaluation pipeline exactly as implemented, unifying the ten-step framework originally proposed (data ingestion through Digital Twin assembly) with the cross-validation and checkpoint-selection procedure actually used.

```
Algorithm 1: End-to-End BDT Training and Evaluation Pipeline
──────────────────────────────────────────────────────────────────────
Input:  Raw ADNI archive (DICOM scans + ADNIMERGE.csv), fold count K,
        epoch budget E, early-stopping patience p
Output: Trained classifier f_θ, cross-validated AUC (mean ± SD),
        ablation results, explainability artifacts, transition
        matrices {P, P⁺, P⁻}, assembled Digital Twin

 1: manifest ← INGEST(raw_archive)                    ▷ join scans ↔ ADNIMERGE, drop
                                                          incomplete rows, standardize
                                                          ADNI diagnosis codes to {CN,MCI,Dementia}
 2: manifest ← VALIDATE(manifest)                      ▷ pydicom integrity check; drop or
                                                          substitute unreadable scans
 3: for each scan in manifest do
 4:     v ← PREPROCESS(scan)                            ▷ DICOM→RAS orient→resample 1.5mm→
                                                          resize/pad/crop 128³→normalize [0,1]
 5:     CACHE(v)                                        ▷ persist as .pt tensor; avoids
                                                          re-preprocessing every epoch
 6: end for
 7: folds ← STRATIFIED_K_FOLD(manifest, K, seed)        ▷ identical seed/splits reused for
                                                          every downstream experiment
 8: fold_results ← ∅
 9: for k = 1 to K do
10:     scaler_k ← FIT_SCALER(train_k.tabular)          ▷ fold-local only; no leakage
11:     θ_k ← INIT(MultimodalTransformer)
12:     best_loss ← ∞;  patience_left ← p
13:     for e = 1 to E do
14:         θ_k ← TRAIN_EPOCH(θ_k, train_k, scaler_k)    ▷ weighted cross-entropy, AdamW,
                                                          LR schedule (Section 3.5)
15:         (val_loss, val_AUC) ← EVALUATE(θ_k, val_k, scaler_k)
16:         if val_loss < best_loss then
17:             best_loss ← val_loss;  patience_left ← p
18:             SAVE_CHECKPOINT(θ_k, e, val_loss, val_AUC)
19:         else
20:             patience_left ← patience_left − 1
21:             if patience_left = 0 then break            ▷ early stopping
22:         end if
23:     end for
24:     fold_results[k] ← BEST_CHECKPOINT(k)
25: end for
26: (μ_AUC, σ_AUC) ← MEAN_STD({fold_results[k].val_AUC : k = 1..K})
27: θ* ← CHECKPOINT of arg max_k fold_results[k].val_AUC  ▷ selected fold for
                                                             downstream analysis
28: ablation_results ← RUN_ABLATION_VARIANTS(θ*, folds)    ▷ A0-A6, B1-B4 (Section 5.1.10)
29: RUN_SHAP(θ*);  RUN_GRADCAM(θ*)                          ▷ Section 3.6
30: {P, P⁺, P⁻} ← ESTIMATE_MARKOV(manifest)                 ▷ Section 3.7
31: neuro_dt ← ASSEMBLE_DIGITAL_TWIN(θ*, scaler, {P, P⁺, P⁻})  ▷ Algorithm 2
32: return θ*, (μ_AUC, σ_AUC), ablation_results, neuro_dt
──────────────────────────────────────────────────────────────────────
```

**Algorithm 2** formalizes the Digital Twin's per-patient inference and simulation procedure (`predict_patient` / `simulate_intervention`), including the counterfactual intervention path used for the APOE4 and pharmacological "what-if" analyses (Section 5.2.11) and its time-varying extension used for the early-versus-late intervention-timing study (Section 5.2.12).

```
Algorithm 2: Digital Twin Patient Simulation
──────────────────────────────────────────────────────────────────────
Input:  Trained classifier f_θ, fitted scaler, patient scan and
        tabular vector x, transition matrix P (or time-varying
        list [P_1, ..., P_T]), number of years n_years,
        visits per year v, number of Monte Carlo simulations n_sims,
        optional intervention effect ε ∈ [0,1) applied from step s
Output: Current diagnosis distribution p₀, mean trajectory over
        n_years·v steps, 95% confidence interval, image_available flag

 1: x' ← scaler.transform(x)
 2: if cached preprocessed tensor exists for this scan then
 3:     v_img ← LOAD_CACHED_TENSOR(scan);  image_available ← true
 4: else
 5:     v_img ← ZERO_TENSOR();  image_available ← false      ▷ caller MUST surface a
                                                                warning; never present
                                                                as image-based (Section 5.2.13)
 6: end if
 7: p₀ ← softmax(f_θ(v_img, x'))                              ▷ current-state distribution
 8: if intervention requested then
 9:     for each step t with t ≥ s do
10:         P_t[MCI, Dementia] ← P_t[MCI, Dementia] · (1 − ε)  ▷ e.g. ε=0.30 (Lecanemab),
                                                                  ε=0.35 (Donanemab)
11:         RENORMALIZE_ROW(P_t, MCI)                          ▷ row must still sum to 1
12:     end for
13: end if
14: trajectories ← zeros(n_sims, n_years·v + 1, |states|)
15: for sim = 1 to n_sims do
16:     state ← SAMPLE_CATEGORICAL(p₀)
17:     trajectories[sim, 0] ← ONE_HOT(state)
18:     for t = 1 to n_years·v do
19:         P_t ← P if P is static else P[t]                  ▷ time-varying matrices support
                                                                  the early-vs-late study
20:         state ← SAMPLE_CATEGORICAL(P_t[state, :])
21:         trajectories[sim, t] ← ONE_HOT(state)
22:     end for
23: end for
24: mean_traj ← MEAN(trajectories, axis=sim)
25: ci_95 ← PERCENTILE(trajectories, [2.5, 97.5], axis=sim)
26: return p₀, mean_traj, ci_95, image_available
──────────────────────────────────────────────────────────────────────
```

**Formal mathematical formulation.** Algorithm 2 can equivalently be expressed in closed form. The current-state diagnosis distribution is the classifier's softmax output over the three-class logits,

$$p_0 = \mathrm{softmax}\big(f_\theta(v_{\mathrm{img}}, x')\big), \qquad p_0 \in \Delta^2$$

where $\Delta^2$ denotes the probability simplex over $\mathcal{Y} = \{\mathrm{CN}, \mathrm{MCI}, \mathrm{Dementia}\}$. The baseline transition matrix $P$ (Section 3.7) is estimated by maximum likelihood directly from observed longitudinal ADNI visit sequences,

$$P_{ij} = \frac{\displaystyle\sum_{n=1}^{N}\sum_{t} \mathbb{1}\big[s_t^{(n)} = i,\; s_{t+1}^{(n)} = j\big]}{\displaystyle\sum_{n=1}^{N}\sum_{t} \mathbb{1}\big[s_t^{(n)} = i\big]}, \qquad i, j \in \mathcal{Y}$$

the ratio of observed $i \to j$ transitions to all observed transitions out of state $i$, across all $N$ patients and all consecutive visit pairs $(s_t^{(n)}, s_{t+1}^{(n)})$. A simulated intervention with effect size $\varepsilon \in [0,1)$, applied from simulation step $s$ onward, first scales down the target transition and then renormalizes the affected row so it remains a valid probability distribution:

$$\tilde{P}_{i,j} = \begin{cases} P_{i,j}\,(1-\varepsilon) & \text{if } i = \mathrm{MCI},\ j = \mathrm{Dementia} \\[2pt] P_{i,j} & \text{otherwise} \end{cases}, \qquad P'_{i,j} = \frac{\tilde{P}_{i,j}}{\sum_{k \in \mathcal{Y}} \tilde{P}_{i,k}}$$

For each Monte Carlo simulation $n = 1, \ldots, N_{\mathrm{sims}}$, the patient's state trajectory is a discrete-time Markov process seeded by the classifier's own diagnosis distribution rather than a fixed initial state,

$$s_0^{(n)} \sim \mathrm{Categorical}(p_0), \qquad s_t^{(n)} \sim \mathrm{Categorical}\big(P'^{(t)}_{s_{t-1}^{(n)},\,\cdot}\big), \qquad t = 1, \ldots, n_{\mathrm{years}} \cdot v$$

where $P'^{(t)} = P'$ for a static intervention and $P'^{(t)} = P'_t$ for the time-varying matrices used in the early-versus-late intervention-timing study (Section 5.2.12). The Digital Twin's reported output, the mean trajectory and its 95% confidence interval at each future step, is then the Monte Carlo estimate of the probability of occupying a given state $c \in \mathcal{Y}$ at time $t$, together with its empirical percentile interval:

$$\hat{p}_t(c) = \frac{1}{N_{\mathrm{sims}}} \sum_{n=1}^{N_{\mathrm{sims}}} \mathbb{1}\big[s_t^{(n)} = c\big], \qquad \mathrm{CI}_{95\%}\big(\hat{p}_t(c)\big) = \Big[\, Q_{0.025}\big(\{\mathbb{1}[s_t^{(n)}=c]\}_{n=1}^{N_{\mathrm{sims}}}\big),\ Q_{0.975}\big(\{\mathbb{1}[s_t^{(n)}=c]\}_{n=1}^{N_{\mathrm{sims}}}\big) \,\Big]$$

with $Q_\alpha$ denoting the empirical $\alpha$-quantile. As $N_{\mathrm{sims}} \to \infty$, $\hat{p}_t(c)$ converges to the exact $t$-step state-occupation probability $\big(p_0 (P')^t\big)_c$ that would result from repeated multiplication of the initial distribution by the transition matrix. The Monte Carlo formulation is used in place of this closed-form matrix power because it composes naturally with a time-varying $P'_t$ (Algorithm 2, line 19) and produces the empirical confidence interval reported alongside the mean trajectory in every what-if result in Section 5.

**Complexity.** Let $B$ denote batch size, $d=1024$ the image embedding dimension, and $p$ the projected fusion dimension. Per training batch, the DenseNet-121 forward/backward pass dominates cost at $O(B \cdot D \cdot H \cdot W \cdot d)$ for a volume of size $D \times H \times W = 128^3$, several orders of magnitude larger than the Transformer encoder's cost of $O(B \cdot p^2)$ over its single-token sequence. The imaging branch, not the fusion mechanism, is the computational bottleneck, consistent with the empirical training-time results reported in Section 5. For Algorithm 2, each Monte Carlo simulation is $O(n_{\mathrm{years}} \cdot v)$ (a fixed number of categorical draws over a $3\times3$ matrix), so total simulation cost is $O(n_{\mathrm{sims}} \cdot n_{\mathrm{years}} \cdot v)$, independent of the classifier's cost since $p_0$ is computed once per patient rather than once per simulation. With $n_{\mathrm{sims}}=1{,}000$ and a 5-year, semi-annual-visit horizon ($n_{\mathrm{years}} \cdot v = 10$ steps), this cost is negligible next to the single forward pass through $f_\theta$ that dominates per-patient latency in practice.

---

## 4 Dataset and Description

### 4.1 The Alzheimer's Disease Neuroimaging Initiative (ADNI)

This work is built entirely on data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), a multi-site, longitudinal research study that began in 2004 with the primary goal of standardizing and validating neuroimaging and biofluid biomarkers for the early detection and tracking of Alzheimer's Disease. The study has enrolled over 2,000 unique participants across four sequential phases to date: ADNI-1 (2004-2009, ~800 new enrollees; CN, MCI, and early AD, foundational biomarker validation), ADNI-GO (2009-2011, ~200; early MCI, a bridging study), ADNI-2 (2011-2016, ~700; CN, MCI, AD, and a new Significant Memory Concern group, focused on the earliest disease stages), and ADNI-3 (2016-2022, hundreds of participants; continued all groups and added advanced Tau PET imaging), with ADNI-4 ongoing. This work draws on the ADNI-1, ADNI-GO, ADNI-2, and ADNI-3 phases.

The dataset is fundamentally multimodal, providing raw neuroimaging files and structured tabular data for each participant. The primary imaging data is raw, 3D T1-weighted MPRAGE structural MRI, sourced from the ADNI LONI Image Data Archive; each scan is provided as a directory of individual 2D DICOM slice files organized hierarchically by subject ID, imaging sequence, and acquisition date. The tabular data is provided as a series of CSV files, of which two were used in this work: `ADNIMERGE.csv`, the primary longitudinal summary file (often considered the "Rosetta Stone" of ADNI, merging key data points from dozens of source tables into one row per patient per visit), and `APOERES.csv` (APOE genotyping). As noted in Table 1, `RECCMEDS.csv` (the concurrent medications log) was in the original proposal's data-source scope but was not integrated in the as-built system.

### 4.2 As-Built Cohort

Access to ADNI was obtained through the LONI Image & Data Archive following an institutional data-use application. The acquired dataset comprises **1,549** T1-weighted MPRAGE structural MRI scans across approximately 500 unique patients, spanning the ADNI-1, ADNI-GO, ADNI-2, and ADNI-3 phases, with three diagnostic classes: Cognitively Normal (CN, $n=469$), Mild Cognitive Impairment (MCI, $n=603$), and Dementia ($n=477$). Each scan comprises approximately 150-200 individual DICOM slice files; the raw dataset totaled approximately 33 GB.

Because downloading this volume directly over a standard network connection would take days, a temporary virtual machine was provisioned in the same Azure storage-account region to make use of the intra-datacenter network backbone (roughly 100x faster than typical broadband), reducing the transfer to minutes. The retrieved DICOM archives were then uploaded to Azure Blob Storage, preserving ADNI's native hierarchical folder structure, and the temporary VM was deallocated immediately after upload.

A sequence of automated data-processing jobs converted the raw uploaded archive into a clean, unified manifest linking each scan to its clinical record. **(1) Manifest construction**: parsing blob storage to extract subject and image identifiers and cross-referencing them against the clinical CSV tables. **(2) Archive extraction**: unpacking and re-uploading the raw DICOM archives in extracted form (approximately 5 hours 44 minutes). **(3) DICOM integrity validation**: loading every scan directory with `pydicom` to identify corrupted, incomplete, or unreadable files, which identified approximately 23 scan directories with unrecoverable issues (missing slices, corrupted headers, zero-byte files); each was either substituted with an alternate visit of the same subject or dropped. **(4) 3D volume validation**: downloading and assembling complete 3D volumes for a sample of scans to verify the full DICOM-to-tensor pipeline end to end. **(5) Final manifest generation**: joining the validated imaging manifest to `ADNIMERGE.csv` and `APOERES.csv` on patient identifier and visit code.

Four tabular features were selected for fusion with the imaging pathway: **AGE**, the single strongest demographic predictor of dementia risk; **PTEDUCAT** (years of education), a proxy for cognitive reserve, later confirmed by SHAP analysis (Section 5.1.9) as the strongest predictor for the CN class; **MMSE** (Mini-Mental State Examination, 0-30), a direct cognitive assessment score; and **APOE4** (allele count, 0/1/2), the primary known genetic risk factor for late-onset Alzheimer's Disease.

Data cleaning addressed several dataset-specific issues. Rows missing any tabular feature or a diagnosis label were dropped. ADNI's several phase-specific diagnosis codes were standardized into the three canonical classes (EMCI and LMCI mapped to MCI; SMC mapped to CN; AD mapped to Dementia). The APOE4 column's `-4` sentinel value (ADNI's missing-data code, not a real allele count) was treated as missing, removing approximately 18 scans. For subjects with multiple longitudinal visits recorded under different diagnoses, the diagnosis was matched to the specific date of the imaging session rather than defaulting to the subject's most recent visit label, to avoid attaching a scan to a diagnosis from a different point in time. The final cleaned dataset comprises the 1,549 scans described above.

One feature-selection caveat, returned to in Section 5.1.10, is that MMSE functions, to a meaningful degree, as a proxy for ADNI's own diagnostic threshold criteria, since clinical diagnosis in the source data is itself partly informed by MMSE score. This is a label-leakage effect relevant to interpreting the classical-baseline ablation results.

### 4.3 Imaging Preprocessing

Each validated DICOM scan was processed through a standard neuroimaging pipeline: loading, reorientation to RAS anatomical space, resampling to 1.5 mm isotropic spacing, resizing/padding to a fixed $128 \times 128 \times 128$ voxel volume, and intensity normalization to $[0,1]$ (clipped at $[0, 1500]$ Hounsfield units), with random affine augmentation applied at training time only. All 1,549 preprocessed volumes were cached to disk as PyTorch tensors (approximately 13 GB total) prior to training, avoiding repetition of this pipeline on every epoch. This preprocessing stage is purely deterministic, classical image processing; it involves no learned components. The CNN and Transformer are applied only afterward, at the modeling stage (Section 3.4), operating on the resulting normalized tensor rather than on raw DICOM data.

---

## 5 Experiments and Result Analysis

The BDT (Section 3) was trained and evaluated across two sequential experimental phases sharing an identical codebase, architecture, dataset, and cross-validation splits (Section 3.5). The only differences between them are compute hardware and, where explicitly noted, training hyperparameters changed in direct response to instability observed in the first phase. Experiment 1 (Section 5.1) was conducted on CPU-based Azure Machine Learning compute and established the full data pipeline, the model architecture, one validated fold, and a partial ablation study. Experiment 2 (Section 5.2) was conducted on a GPU workstation and obtained a fully cross-validated result, the complete ablation study, and end-to-end validation of the Digital Twin pipeline, culminating in deployment to a production clinical dashboard.

### 5.1 Experiment 1: Training on CPU-Based Compute

#### 5.1.1 Infrastructure

Training infrastructure was provisioned on Microsoft Azure Machine Learning. An interactive compute instance ("DevBox") and a separate compute cluster ("Workhorse"), which scaled to zero nodes when idle, were provisioned, both of specification `Standard_E4ds_v4` (4 cores, 32 GB RAM, 150 GB disk). The compute instance was used for interactive development and dashboard testing, while the compute cluster executed all data-ingestion and training jobs.

#### 5.1.2 Environment

A Python 3.10 environment was configured with a CPU-only PyTorch build, MONAI (medical imaging preprocessing), pydicom, scikit-learn, MLflow (experiment tracking), and the Azure SDKs for blob storage access. NumPy was explicitly pinned below version 2.0 after an initial environment build broke MONAI and pydicom compatibility under NumPy 2.x.

#### 5.1.3 Dataset Acquisition and Ingestion

Covered in full in Section 4.2.

#### 5.1.4 Model Architecture

The classifier is exactly the `MultimodalTransformer` described in Section 3.4 (24,012,643 parameters).

#### 5.1.5 Preliminary Run and Data-Quality-Driven Overfitting

Before committing to the full 5-fold, 15-epoch training run, the pipeline was first validated end to end on a fast, low-cost preliminary run ("FAST_PROTO" mode: 40% of the dataset, 3 folds, 10 epochs) to confirm the architecture was learning genuine signal before the far more expensive full run was submitted.

An early iteration of this preliminary run, trained on an unfiltered version of the manifest, showed classic overfitting: validation loss reached a minimum at epoch 4 (0.6986) and then increased steadily over subsequent epochs (0.7465, 0.7562), while validation accuracy plateaued at approximately 64.5%. Inspection of the training logs traced this to a large number of `Error processing image at path...` warnings: the `try`/`except` block in the dataset loader was silently substituting a zero-valued tensor for any DICOM series it could not load, injecting label noise into training rather than surfacing the failure. This finding directly motivated the automated DICOM integrity validation step described in Section 4.2, which filters out unreadable scans before training rather than substituting a blank image for them, along with the migration of the training process from an interactive notebook to a script-based Azure ML Command Job for more stable, reproducible execution.

With the cleaned data and script-based execution, the FAST_PROTO run achieved a macro one-vs-rest validation AUC of 0.862 and 63% overall accuracy on the three-class problem (chance level 33%) using only 40% of the dataset and 10 epochs (Table 2). The comparatively lower MCI AUC reflects a well-documented property of the ADNI cohort rather than a defect in the pipeline: of the misclassified MCI patients, a similar number were predicted CN as were predicted Dementia, consistent with MCI's clinical role as a transitional, diagnostically ambiguous state. Benchmarked against published 3-class ADNI results, this preliminary figure was already competitive using a fraction of the available data and training budget (Table 3).

**Table 2. Preliminary (FAST_PROTO) per-class results**

| Class | AUC |
|---|---|
| Dementia | 0.914 |
| CN | 0.878 |
| MCI | 0.698 |
| **Macro OvR** | **0.862** |

**Table 3. Preliminary result vs. published ADNI benchmarks**

| Study | Reported metric |
|---|---|
| Basaia et al. (2019) [12], 3D CNN | AUC ~0.85 |
| Wen et al. (2020) [13], CNN benchmark | AUC ~0.83 |
| Venugopalan et al. (2021) [16], multimodal | AUC ~0.87 |
| Zhou et al. (2025) [28], CNN + Swin Transformer | 92% accuracy (2-class) |
| This work (FAST_PROTO, 40% data, 10 epochs) | AUC 0.862 |

This preliminary run also converged quickly (validation loss stopped improving after epoch 5 of 10), an encouraging sign that motivated proceeding directly to the full run rather than further preliminary tuning. Expanding the tabular feature set beyond the four features in Section 4.2 (e.g., ADAS-Cog, CDRSB, RAVLT immediate recall, all present in `ADNIMERGE.csv`) was considered as a way to strengthen MCI disambiguation, but was not carried into the full run, to keep the tabular inputs consistent with the four clinically motivated features specified there.

#### 5.1.6 Training Configuration

Training used 5-fold stratified cross-validation, AdamW optimization (learning rate $1 \times 10^{-4}$), a batch size of 4 (constrained by available system memory), and a `CosineAnnealingLR` schedule with early stopping (patience 3 epochs) based on validation loss. Five-fold cross-validation follows standard practice in the medical-imaging literature for a dataset of this size, balancing statistical robustness against computational cost: fewer folds (e.g., three) risk higher variance in the resulting estimate, while more folds (e.g., ten) offer diminishing returns at approximately 1,500 samples. Fifteen epochs was chosen based on the convergence behavior observed in the preliminary run, which indicated the architecture converges within a comparable number of epochs on a data subset.

#### 5.1.7 Results

Training completed for only one of five folds. The remaining four folds triggered early stopping within a few epochs of their best result, caused by a sharp validation-loss spike in early training, attributed to the learning-rate schedule beginning at full magnitude with no warmup period, combined with the high per-step gradient noise inherent to a batch size of 4 on high-dimensional 3D volumetric input. Fold 4 was the only fold to train to completion (15 epochs) and was adopted as the primary model for this phase. The full training job (all five fold attempts, run as a single Azure ML Command Job, `BDT-Hybrid-Model-Training-CleanData`) ran for **1 day, 10 hours** of compute time, the great majority of which was consumed by Fold 4's complete 15-epoch run; the four early-stopped folds each contributed comparatively little wall-clock time by comparison.

**Table 4. Fold 4 classification performance (validation set, $n=310$)**

| Class | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| CN | 0.85 | 0.86 | 0.86 | 0.957 |
| Dementia | 0.79 | 0.82 | 0.81 | 0.936 |
| MCI | 0.74 | 0.71 | 0.72 | 0.844 |
| **Overall** | **0.79** | **0.80** | **0.80** | **0.912** |

The model produced zero misclassifications between the CN and Dementia classes, the two diagnostically furthest-apart categories, across the entire validation set (Figure 1).

*Figure 1. Fold 4 confusion matrix (CPU-trained model).*

#### 5.1.8 Ablation Study: CPU-Feasible Variants

Five of the eleven ablation variants planned for this work were successfully trained and evaluated on CPU compute, a genuine result of this phase rather than just a limitation: the full multimodal model (A0, reusing the existing Fold 4 checkpoint with no retraining required) and a tabular-only multilayer perceptron (A1) were evaluated alongside four classical machine-learning baselines trained directly on the four tabular features (B1 Logistic Regression, B2 SVM with an RBF kernel, B3 Random Forest, B4 Gradient Boosting).

**Table 5. CPU-feasible ablation results**

| Variant | AUC | F1 |
|---|---|---|
| A0, Full multimodal model | 0.912 | 0.796 |
| A1, Tabular-only MLP | 0.879 | 0.706 |
| B1, Logistic Regression (tabular) | 0.865 | 0.707 |
| B2, SVM, RBF kernel (tabular) | 0.871 | 0.735 |
| B3, Random Forest (tabular) | 0.943 | 0.830 |
| B4, Gradient Boosting (tabular) | 0.941 | 0.825 |

This comparison revealed that Random Forest and Gradient Boosting classifiers, trained on the four tabular features alone with no imaging input whatsoever, achieved higher raw AUC than the full multimodal model. Investigation attributed this to the MMSE label-leakage effect noted in Section 4.2: because ADNI's diagnostic labels are themselves partly derived from MMSE threshold criteria, a classifier with direct access to MMSE can reconstruct much of the label without reference to any imaging data. This is treated as a limitation of AUC as a sole comparison metric here, not as evidence that the multimodal model underperforms. Section 5.1.9 and the full model's zero-CN-Dementia-confusion result above are the more clinically meaningful comparison points. The model's contribution from imaging alone can be measured directly against the tabular-only floor: full multimodal AUC (0.912) minus tabular-only MLP AUC (0.879) gives **+0.033 AUC points** attributable to the 3D MRI pathway.

The remaining five deep-learning variants (a CNN-only classifier, A2; two alternative CNN-tabular fusion designs, A3 and A4; a single-layer Transformer variant, A5; and a variant trained without class-weighted loss, A6) each require training a full 3D-CNN-based model from scratch on imaging data, at the same per-fold training cost that limited Section 5.1.7 to one completed fold. A single continuous run covering these variants was attempted on CPU-only compute. It ran for approximately **12 hours** before failing with an out-of-memory error on the 32 GB compute instance, without completing. These variants were therefore deferred to Experiment 2 (Section 5.2), where they were executed as part of a complete, independently re-run eleven-variant ablation study on GPU hardware with substantially more available memory.

*Figure 2. CPU-feasible ablation results, bar chart.*

#### 5.1.9 Explainability

SHAP analysis (applied to the tabular branch with the imaging embedding held frozen) identified PTEDUCAT as the strongest predictor for the CN class and AGE as the strongest predictor for the Dementia class. Gradient-based explainability (Grad-CAM, hooked on the final dense block of the CNN backbone, Section 3.6) required the input tensor's gradient tracking to be explicitly enabled before the backward pass, which was not the default behavior when running inference on an already-evaluated model. For validation patient 057_S_1373 (true diagnosis Dementia, correctly predicted Dementia), Grad-CAM peak activation was located at axial slice $z=86$, sagittal slice $y=43$, and coronal slice $x=87$. As shown in the corresponding figure, activation spans a broad region of the affected hemisphere rather than sharply localizing to the hippocampus. The actual localization should be read directly from the figure rather than assumed; this diffuse pattern is returned to in Section 5.2.9.

*Figure 3. SHAP tabular feature attribution (CPU-trained model).*
*Figure 4. Grad-CAM activation overlay, Patient 057_S_1373 (Dementia), CPU-trained model.*

#### 5.1.10 Markov Chain and Digital Twin

An earlier run of the Markov Chain estimation procedure (Section 3.7) surfaced a manifest-level defect: the `visit_date` field, extracted from each scan's file path, failed to parse for every row (0/1,549 rows extracted), because the extraction function required an exact 10-character path segment while real segments concatenate date, time, and image identifier into a longer string. Since the chronological visit-ordering the transition-matrix computation depends on was therefore effectively arbitrary rather than time-ordered, a symptom surfaced directly: the resulting matrix showed APOE4-positive patients with a lower MCI-to-Dementia transition probability than APOE4-negative patients, backwards from APOE4's known role as a progression risk factor. The date-extraction function was corrected (matching on the leading 10 characters of a sufficiently long path segment rather than requiring an exact-length match) during the GPU migration (Section 5.2.2), and the fix was subsequently ported back into the CPU pipeline. The Markov Chain and Digital Twin results reported in this work (Section 5.2.10) reflect the corrected pipeline in both experimental phases.

This phase's results are limited in several respects (a single completed fold, a compute-infeasible portion of the ablation study, and a manifest-level defect that blocked Markov Chain validation), discussed together with the GPU phase's own limitations in Section 6.3.

### 5.2 Experiment 2: Training and Full Pipeline Evaluation on GPU-Based Compute

#### 5.2.1 Motivation and Experimental Setup

To obtain a statistically valid cross-validated result and complete the planned ablation and prognostic-modeling work left incomplete by Experiment 1's CPU-only compute constraints (Section 5.1), the identical codebase (same architecture, same data splits, same preprocessing pipeline) was migrated to a university GPU lab workstation (NVIDIA GeForce RTX 5070 Ti, 17.1 GB VRAM), running locally rather than on Azure ML Compute. This experiment isolates the effect of the compute platform: nothing about the model, the data, or the task changes between Experiments 1 and 2, so any difference in outcome is attributable to the compute environment and the training-configuration adjustments it enabled, not to a different problem being solved.

**Figure 5. GPU detection / specification confirmation.**

#### 5.2.2 Migration: Environment and Data Transfer

Every Azure ML Workspace-specific dependency was removed from the pipeline so it could run as a self-contained local process, reading only from local files (the lab PC's own disk) rather than a live Azure ML workspace connection. Data (the cached tensors, model checkpoints, and clinical manifest) was transferred via Azure Blob Storage using a download utility hardened specifically for this migration: downloads write to a temporary file and are renamed to their final path only on success, so an interrupted transfer can never be mistaken for a complete one on a later run; both individual file downloads and the storage-listing operation itself retry with exponential backoff. A companion verification step then loaded every transferred tensor and checkpoint file directly, rather than only checking for file existence, to catch any corruption that completed writing but was still invalid. This confirmed all 1,549 cached tensors and all checkpoints transferred correctly.

During this migration, the `visit_date` extraction defect described in Section 5.1.10 was identified and corrected. The fix applied here is the same one later ported back into the CPU notebook.

#### 5.2.3 Training Configuration Changes

Three configuration changes were made in response to the training instability observed in Experiment 1: the learning-rate schedule was changed from `CosineAnnealingLR` to `OneCycleLR` with a 10% warmup period, the batch size was increased from 4 to 16 (enabled by GPU memory capacity), and mixed-precision (AMP) training was enabled. Early stopping patience was also relaxed from 3 to 5 epochs.

#### 5.2.4 Resumable Training Design

Because lab PC access was not continuous, the training loop was designed to tolerate interruption at any point: each checkpoint stores the model, optimizer, learning-rate scheduler, and mixed-precision scaler state together with a completion flag, so a run can be stopped (kernel death, closing the notebook, Ctrl+C) and resumed later without restarting completed folds from scratch. A fold cut off mid-training resumes from its last saved epoch rather than epoch 1.

#### 5.2.5 Cross-Validated Results

With the revised schedule, all five folds completed training successfully, a first for this thesis, since only one of five folds completed on CPU (Section 5.1.7).

**Table 6. 5-fold cross-validated results (GPU-trained model, loss-selected checkpoints)**

| Fold | Validation AUC |
|---|---|
| 1 | 0.8821 |
| 2 | 0.8591 |
| 3 | 0.8125 |
| 4 | 0.9511 |
| 5 | 0.8481 |
| **Mean ± SD** | **0.8706 ± 0.0461** |

This is the primary result of this thesis: a genuine 5-fold cross-validated estimate, in contrast to Experiment 1's single-fold result. A direct comparison is possible on Fold 4, since both experiments used an identical stratified split (same random seed, `random_state=42`) and therefore evaluated on the same held-out 310 patients:

**Table 7. Direct comparison on the shared Fold 4 split**

| | CPU (Experiment 1) | GPU (Experiment 2) |
|---|---|---|
| Fold 4 AUC | 0.9120 | **0.9511** |
| Folds completed | 1 of 5 | 5 of 5 |
| Reported metric | Single fold | 5-fold mean: **0.8706 ± 0.0461** |

On this shared fold, the GPU-trained model outperforms the CPU-trained model by 0.039 AUC, attributable to the scheduler and batch-size changes above, independent of the cross-validation completeness improvement.

The full 5-fold training run completed in approximately **1 hour**, a dramatic reduction from the CPU run's 1 day 10 hours (Section 5.1.7), despite the GPU run completing all five folds to their full epoch budget versus the CPU run completing only one. This reflects the combined effect of GPU-accelerated 3D convolution, the larger batch size (16 vs. 4), and mixed-precision training, rather than any change to the amount of data or number of epochs per fold.

**Figure 6. 5-fold training log, all folds completing.**

#### 5.2.6 Checkpoint Selection Methodology Experiment

A secondary experiment tested whether checkpoint selection based on maximum validation AUC (rather than minimum validation loss) would yield a more favorable, or simply different, result. An identical training run was performed using AUC as both the selection and early-stopping criterion, writing to separate checkpoint files so the original loss-selected run remained untouched for comparison.

**Table 8. Loss-selected vs. AUC-selected checkpoint criteria**

| Fold | Loss-selected AUC | AUC-selected AUC |
|---|---|---|
| 1 | 0.8821 | 0.9439 |
| 2 | 0.8591 | 0.9398 |
| 3 | 0.8125 | 0.9265 |
| 4 | 0.9511 | 0.9451 |
| 5 | 0.8481 | 0.9342 |
| **Mean ± SD** | **0.8706 ± 0.0461** | **0.9379 ± 0.0069** |

Although the AUC-selected criterion produced a higher mean score, analysis of the per-epoch training logs showed that every fold trained through all 20 available epochs without triggering early stopping, reaching training accuracies of 97-99% by the final epochs, a clear indicator of overfitting that a loss-based criterion would have halted earlier. Because the AUC criterion also selects the checkpoint that is the argmax of a noisy metric evaluated on a validation set of only ~310 samples, this method is a biased estimator of true generalization performance. The loss-selected result (0.8706 ± 0.0461) was therefore retained as the primary reported result of this thesis; this comparison is presented as a methodological contribution regarding checkpoint-selection bias rather than as a competing headline figure.

This second run completed in approximately **90 minutes**, slightly longer than the loss-selected run's 1 hour, consistent with every fold training through its full 20-epoch budget here (this section's overfitting observation) rather than being cut short by early stopping as several folds were in the loss-selected run.

#### 5.2.7 Full Ablation Study

The complete eleven-variant ablation study was executed on the GPU platform as a fresh, independent run, not a continuation of Experiment 1's partial results (Section 5.1.8). It re-trained all seven deep-learning variants (including the five that were CPU-infeasible: CNN-only, two fusion designs, the single-layer Transformer, and the no-class-weighting variant) and re-fit all four classical baselines, each evaluated on the same Fold 4 data split used throughout this thesis. The full study completed in **3 hours 58 minutes**, feasible on GPU within an afternoon. The same deep-learning variants had previously been attempted on CPU-only compute (Section 5.1.8), where the run failed with an out-of-memory error after approximately 12 hours without completing. The GPU platform's larger memory and faster per-batch throughput together turned a run that had failed outright into one that finished comfortably within a single working session.

**Table 9. Full ablation study results**

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

As these results are single-fold (n=310) rather than cross-validated, the close scores among the top five variants (within approximately 0.01 AUC of one another) should be interpreted as within normal fold-level variance rather than a definitive architectural ranking.

Two results stand out here: the near-identical scores of the CNN-only variant (0.9474) and the full multimodal model (0.9486), and the continued outperformance of classical tree-ensemble methods over the deep tabular-only branch (~0.94 vs. 0.87 AUC) on identical input features. Both are examined in Section 6.1.

**Figure 7. Ablation study, AUC comparison across all 11 variants.**
**Figure 8. Ablation study, per-class AUC grouped chart.**

#### 5.2.8 Post-Training Evaluation

**Table 10. Fold 4 classification performance (validation set, n=310, GPU-trained model)**

| Class | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| CN | 0.88 | 0.94 | 0.91 | 0.982 |
| Dementia | 0.85 | 0.84 | 0.85 | 0.958 |
| MCI | 0.82 | 0.78 | 0.80 | 0.913 |
| **Overall** | **0.85** | **0.85** | **0.85** | **0.951** |

Every class improved over the CPU-trained model's Fold 4 result (Table 4): CN AUC rose from 0.957 to 0.982, Dementia from 0.936 to 0.958, and MCI, the hardest class throughout this thesis, from 0.844 to 0.913.

**Figure 9. Fold 4 confusion matrix, GPU-trained model.**

#### 5.2.9 Explainability

SHAP analysis (tabular branch, frozen image embedding, same method as Section 5.1.9) was re-run against the GPU-trained checkpoint.

**Figure 10. SHAP tabular feature attribution, GPU-trained model.**

Grad-CAM was run on the same validation patient used in the CPU section, 057_S_1373 (true diagnosis Dementia, correctly predicted Dementia), for direct before/after comparison of the same patient under the CPU- and GPU-trained models. Peak activation moved from (axial z=86, sagittal y=43, coronal x=87) under the CPU model to (axial z=94, sagittal y=75, coronal x=108) under the GPU model, a visibly different activation region between the two model generations for the identical patient. Grad-CAM output should not be treated as a fixed property of the patient. As with the CPU section, the actual localization (focal vs. diffuse) should be read directly off the figure before making any claim about anatomical correspondence.

**Figure 11. Grad-CAM activation overlay, Patient 057_S_1373 (Dementia), GPU-trained model.**

#### 5.2.10 Markov Chain and Digital Twin

The Markov Chain prognostic engine (Section 5.1.10's method) was re-run on this platform after the same `visit_date` fix, against the identical 1,549-scan dataset. The resulting matrices are numerically consistent with the CPU section's corrected result, since the Markov chain is computed directly from ADNI visit history and does not depend on which model (CPU- or GPU-trained) is used for classification:

**Table 11. Markov transition matrix, full cohort**

| From \ To | CN | Dementia | MCI |
|---|---|---|---|
| CN | 0.967 | 0.003 | 0.030 |
| Dementia | 0.000 | 0.997 | 0.003 |
| MCI | 0.002 | 0.157 | 0.841 |

**Table 12. Markov transition matrix, APOE4-positive**

| From \ To | CN | Dementia | MCI |
|---|---|---|---|
| CN | 0.955 | 0.000 | 0.045 |
| Dementia | 0.000 | 0.995 | 0.006 |
| MCI | 0.004 | 0.184 | 0.811 |

**Table 13. Markov transition matrix, APOE4-negative**

| From \ To | CN | Dementia | MCI |
|---|---|---|---|
| CN | 0.972 | 0.004 | 0.024 |
| Dementia | 0.000 | 1.000 | 0.000 |
| MCI | 0.000 | 0.133 | 0.867 |

Dementia behaves as an effectively absorbing state (self-transition probability 0.995-1.00 across all three strata), and APOE4-positive patients show a higher MCI-to-Dementia transition probability (0.184) than APOE4-negative patients (0.133), the clinically expected direction. These figures match the CPU section's corrected transition matrices (Section 5.1.10) exactly, which is itself a useful internal consistency check: two independent runs of the same fixed pipeline, on two different machines, produced identical transition matrices.

**Figure 12. Markov transition matrix heatmap, full cohort, GPU run.**

#### 5.2.11 What-If Simulation

The Digital Twin's what-if simulation (Section 3.7, Algorithm 2) was run for validation patient 016_S_1149 (diagnosed MCI, APOE4-negative), the same patient examined in the CPU section's what-if result, enabling a direct before/after comparison under the two model generations.

**Table 14. What-if simulation, Patient 016_S_1149, 5-year Dementia probability, GPU-trained model**

| Scenario | P(Dementia) at Year 5 |
|---|---|
| Baseline (APOE4-negative) | 14.5% |
| APOE4-positive (no treatment) | 25.2% |
| APOE4-positive + Lecanemab (30%) | 12.1% |
| APOE4-positive + Donanemab (35%) | 11.9% |

Every simulated effect remains in the clinically expected direction (APOE4 increases risk, both treatments decrease it), matching the CPU result's pattern. The absolute risk estimates differ substantially between the two model generations, however: the CPU-trained model estimated this same patient's baseline 5-year Dementia risk at 54.9% (Section 5.1.10), while the GPU-trained model estimates 14.5%. This divergence is examined in Section 6.1.

**Figure 13. What-if simulation for Patient 016_S_1149: baseline, APOE4, Lecanemab, and Donanemab trajectories under the GPU-trained model.**

#### 5.2.12 Additional Digital Twin Validation

Three further checks were run to probe the Digital Twin's behavior beyond a single patient, none of which were feasible within the CPU phase's scope.

**Age sensitivity.** An initial single-patient age sweep (65-85 years) produced a counter-intuitive result: predicted Dementia risk decreasing with increasing age. Rather than accept this at face value, the check was repeated across five independent MCI patients:

**Table 15. Age-sensitivity check, 5-year Dementia probability by simulated age (65-85)**

| Patient | 65 | 70 | 75 | 80 | 85 | Direction |
|---|---|---|---|---|---|---|
| 016_S_1149 | 71.3% | 51.3% | 29.6% | 19.0% | 15.3% | decreases |
| 099_S_0880 | 88.8% | 82.5% | 77.3% | 74.7% | 70.1% | decreases |
| 057_S_1007 | 77.5% | 74.6% | 79.7% | 75.2% | 78.7% | increases |
| 109_S_1114 | 75.7% | 75.0% | 74.5% | 76.9% | 79.6% | increases |
| 128_S_0167 | 100.0% | 100.0% | 99.1% | 97.8% | 95.0% | decreases |

Two of five patients showed the clinically expected increasing-risk direction. This is consistent with the AGE tabular feature carrying a comparatively weak and noisy signal relative to the dominant imaging pathway, directly supported by Section 5.2.7's finding that removing the tabular branch entirely (CNN-only, 0.9474) barely changes performance versus the full model (0.9486), rather than indicating a systematic error in the simulation methodology. The direction varying by patient, rather than being uniformly wrong, is what distinguishes a noisy weak feature from a code defect.

**Figure 14. Age-sensitivity analysis, Patient 016_S_1149.**

**Population subgroup comparison.** The Digital Twin was run on 10 representative patients from each diagnostic class to validate population-level behavior:

**Table 16. Mean 5-year Dementia probability by diagnostic subgroup (n=10 per group)**

| Diagnostic group | Mean P(Dementia) at Year 5 |
|---|---|
| CN | 26.3% ± 18.7% |
| MCI | 64.4% ± 23.3% |
| Dementia | 95.1% ± 5.0% |

The monotonic ordering (CN < MCI < Dementia) is the expected population-level pattern and was not explicitly enforced by the model or the simulation. It emerges from the combination of the classifier's diagnosis probabilities and the Markov transition matrix.

**Figure 15. Population subgroup trajectory comparison across CN, MCI, and Dementia.**

**Early vs. late intervention timing.** Using time-varying transition matrices (Algorithm 2, Section 3.8), the same APOE4-positive MCI patient (128_S_0167) was simulated receiving Lecanemab starting at Year 0 versus Year 2:

**Table 17. Early vs. late intervention timing, Patient 128_S_0167, 5-year Dementia probability**

| Scenario | P(Dementia) at Year 5 |
|---|---|
| No treatment | 97.3% |
| Lecanemab starting Year 2 | 96.5% |
| Lecanemab starting Year 0 | 95.7% |

Earlier intervention produces a lower simulated risk, the clinically expected direction, though the absolute effect is modest for this particular patient because their baseline risk was already very high (97.3%) by the time of assessment. This ceiling effect should be kept in mind rather than presenting the 1.6-percentage-point gap as a strong treatment-timing effect. This simulation is the most novel capability demonstrated in this thesis: no published static ADNI classifier can produce a time-varying intervention-timing projection, since that requires the combination of a trained classifier and a Markov progression model that neither component provides alone.

**Figure 16. Early vs. late intervention timing, Patient 128_S_0167.**

#### 5.2.13 Clinical Dashboard Deployment

The GPU-trained model and associated Markov transition matrices were integrated into a Streamlit-based clinical dashboard, containerized and deployed to Azure App Service. During this integration, three functional defects were identified and resolved prior to production release:

1. A checkpoint-loading incompatibility with current PyTorch versions (missing an explicit flag required because the checkpoint embeds a fitted scikit-learn scaler object, which newer PyTorch versions reject loading by default). This was the same class of defect independently re-discovered and fixed in the CPU notebook.
2. A label-mapping orientation mismatch between the two model generations' saved checkpoint metadata, which caused inference to fail on every request against the newer checkpoint format.
3. A silent-failure mode in which a patient with no available cached MRI scan received a full-confidence diagnostic prediction generated from a blank image tensor, with no indication to the clinician that the imaging pathway had not been used. This was resolved by explicitly tracking scan availability and surfacing an unmissable warning wherever such a prediction occurs.

Following these fixes, the deployed model was validated against a genuine held-out ADNI patient record not used during any training or validation step in this thesis, correctly predicting the patient's true diagnosis with near-full confidence, with the reported risk-trajectory and genotype-effect figures matching hand-computed values derived independently from the underlying transition matrices.

**Figure 17. Deployed dashboard, prediction output for the held-out verification patient.**
**Figure 18. Deployed dashboard, "About" / model-info panel.**

This phase's results also carry limitations (single-fold ablation, a weak age signal, and a large single-patient divergence from the CPU phase), discussed together with the CPU phase's own limitations in Section 6.3.

---

## 6 Discussion

### 6.1 Synthesis of Findings Across Experiments

The two-stage ablation study (Sections 5.1.8, 5.2.7) quantified the marginal contribution of each architectural component. Its clearest finding is that the imaging pathway alone (CNN-only, AUC 0.9474) performs almost identically to the full multimodal model (AUC 0.9486), indicating that the 3D MRI pathway carries the substantial majority of the model's predictive signal in this architecture, while the tabular branch contributes comparatively little on top of it. The same pattern reappears in the age-sensitivity check (Section 5.2.12), where the AGE feature's weak, direction-inconsistent effect is consistent with a tabular pathway that plays a secondary role. A second, related finding is that classical tree-ensemble methods (Random Forest, Gradient Boosting) consistently outperformed the deep tabular-only branch on identical features, in both the CPU ablation (Section 5.1.8) and the full GPU ablation (Section 5.2.7). This is not read as an argument that classical methods should replace the multimodal deep model. It traces instead to the MMSE label-leakage effect noted in Section 4.2: because ADNI's diagnostic labels are themselves partly derived from MMSE threshold criteria, any classifier with direct access to MMSE can reconstruct much of the label from that single feature, which inflates AUC for tabular-only models without indicating better real-world clinical value. This is also why the full multimodal model's zero-CN-Dementia-confusion result (Section 5.1.7) is treated as a more clinically meaningful comparison point than AUC alone.

The controlled CPU-versus-GPU comparison (Section 5.1 versus Section 5.2), run on the identical codebase and data splits, isolates what compute-platform migration does and does not change. It removed two hard constraints of CPU-only compute: a 12-hour out-of-memory failure on the full ablation study, and only one of five cross-validation folds completing due to training instability. On the shared Fold 4 split, the GPU-trained model also scored 0.039 AUC higher (0.9511 vs. 0.9120), attributable to the revised learning-rate schedule and larger batch size rather than to the platform itself. At the same time, the Markov Chain component, computed directly from ADNI visit history rather than from either trained model, produced numerically identical transition matrices on both platforms (Tables 6-8 vs. Tables 11-13). This is a useful internal consistency check: two independent runs of the same fixed pipeline, on two different machines, agree exactly where the underlying computation does not depend on the classifier.

Where the two platforms diverge sharply is in the single-patient what-if result for patient 016_S_1149 (Section 5.2.11): the CPU-trained model estimated a 54.9% baseline five-year Dementia risk, while the GPU-trained model estimated 14.5% for the identical patient. Because the GPU-trained model is the more accurate of the two by every cross-validated metric in this thesis (Table 7), its estimate is the more trustworthy one, but a five-fold difference in a single patient's headline number between two versions of the same pipeline is a useful reminder that single-patient case studies demonstrate the Digital Twin's capability, not its robustness. The population-level subgroup comparison (Table 16), which shows the expected monotonic CN < MCI < Dementia ordering, is the more defensible evidence for robustness.

### 6.2 Comparison with Prior Work

The GPU-trained model's per-class AUC (0.982 CN, 0.958 Dementia, 0.913 MCI; Table 10) and its 5-fold cross-validated mean of 0.8706 ± 0.0461 (Table 6) can be set against the published 3-class ADNI benchmarks introduced in Section 2.3 and Table 3. Basaia et al. [12] reported approximately 0.85 AUC with a single-modality 3D CNN; Wen et al. [13] reported approximately 0.83 with a benchmarked CNN; and Venugopalan et al. [16] reported approximately 0.87 with a multimodal model. The BDT's cross-validated mean sits within this range, and its best fold (0.9511) and per-class AUCs exceed it, consistent with the hybrid CNN-Transformer direction that Hu et al. [15] and Zhou et al. [28] also report improving on CNN-only baselines (93.56% and 92.92% 2-class accuracy respectively). This comparison should be read with two caveats already established in this thesis. First, most published benchmarks report a single train/test split or a single fold rather than a 5-fold cross-validated mean, so the comparison in Table 3 is necessarily approximate. Second, per Aghdam et al. [27]'s finding of a severe internal-to-external generalizability gap (97.9% dropping to approximately 67%) in comparable AD models, none of these figures, including this thesis's own, should be read as evidence of generalization beyond the ADNI cohort itself without external validation (Section 6.3).

On the Digital Twin side, this thesis's early-versus-late intervention-timing simulation (Section 5.2.12) and its time-varying, intervention-aware Monte Carlo simulation (Algorithm 2) extend beyond the progression-modeling approaches surveyed in Sections 2.2 and 2.3. Bertolini et al. [10] simulate clinical and cognitive score trajectories but not imaging, while Ravi et al. [29] simulate MRI trajectories but not clinical scores. The BDT's classifier is trained jointly on both modalities and feeds a single Monte Carlo simulation, closing the specific fragmentation gap identified in Section 2.4.

### 6.3 Limitations

Several limitations qualify the results in this thesis, spanning both experimental phases and the framework as a whole.

**Cross-validation completeness.** Only one of five CPU cross-validation folds completed (Section 5.1.7), so the CPU-phase AUC of 0.912 is a single-fold result rather than a statistically robust estimate. The GPU phase resolves this with a genuine 5-fold mean of 0.8706 ± 0.0461 (Table 6), the primary reported result of this thesis, but the full ablation study (Table 9) remains single-fold (n=310) even on GPU, so the close ranking among its top five variants should not be read as a definitive architectural conclusion.

**Compute-constrained scope on CPU.** The six deep-learning ablation variants requiring fresh 3D-CNN training (Section 5.1.8) were computationally infeasible on CPU-only compute, failing with an out-of-memory error after approximately 12 hours rather than completing slowly. The manifest-level `visit_date` parsing defect (Section 5.1.10) also meant the Markov Chain and Digital Twin components could not be validated on CPU in that phase, although the defect was identified and corrected before Experiment 2, and both phases' corrected results agree exactly (Section 6.1).

**Label leakage in the tabular baseline comparison.** The MMSE feature carries a partial label-leakage relationship with the ADNI diagnostic labels themselves (Section 4.2), since clinical diagnosis in the source data is itself partly informed by MMSE score. This should temper any claim that the classical-baseline ablation comparison (Section 6.1) reflects a fair, leakage-free benchmark, and is a property of the source data rather than of this work's modeling choices.

**Weak and inconsistent age signal.** The AGE tabular feature showed a weak, direction-inconsistent signal in the age-sensitivity check (Section 5.2.12; two of five patients showed the clinically expected increasing-risk direction). This is consistent with the imaging pathway dominating the model's predictive signal (Section 6.1) rather than indicating a code defect, but it limits how much clinical weight the age-sensitivity capability can currently be given, until a larger patient sample or an additional cognitive-score feature (e.g., ADAS-Cog, CDRSB) strengthens that pathway.

**Single-patient case studies are not a robustness demonstration.** The what-if simulation's large absolute divergence between the CPU- and GPU-trained models for the same patient (54.9% vs. 14.5% five-year Dementia probability, Section 5.2.11) is a reminder that a single-patient case study demonstrates the Digital Twin's capability, not its robustness. The population-level subgroup validation (Table 16) is the more defensible evidence for the latter.

**No external validation.** All results in this thesis are derived from ADNI, a single, if large and well-established, research cohort. Aghdam et al. [27] report that comparable AD models can lose most of their accuracy on an external dataset (Section 6.2); this thesis has not yet tested the BDT against an independent cohort, so its generalization beyond ADNI's population and acquisition protocols is untested (Section 7.2).

**Deviations from the proposed framework.** Table 1 documents five specific deviations between the originally proposed framework and the as-built system. Most significantly, a fully-observed empirical Markov Chain was substituted for the originally proposed Hidden Markov Model, justified by the fact that diagnostic state is directly observed at each ADNI visit rather than hidden, and Transformer attention-map visualization was omitted, since it would be architecturally degenerate given the single-token fusion design (Section 3.4).

---

## 7 Conclusion and Future Work

### 7.1 Summary of Contributions

This work set out to design, implement, and validate a Brain Digital Twin (BDT) for Alzheimer's Disease: a system that moves beyond static diagnostic classification to personalized, simulative "what-if" forecasting, motivated directly by the clinical demands of the new generation of disease-modifying therapies. All five objectives stated in Section 1.3 were addressed. A unified multimodal ingestion pipeline was built and validated end to end against 1,549 ADNI T1-weighted MRI scans and four tabular clinical features (Section 4). A hybrid 3D DenseNet-121 CNN and Transformer classifier was implemented, trained, and cross-validated, reaching a primary 5-fold cross-validated AUC of 0.8706 ± 0.0461, with per-class AUC reaching 0.982 (CN), 0.958 (Dementia), and 0.913 (MCI) on the shared Fold 4 evaluation split (Section 5.2.8). An empirically estimated, APOE4-stratified Markov Chain progression model was built directly from observed ADNI visit histories and assembled with the classifier into a working Digital Twin capable of Monte Carlo trajectory simulation, intervention modeling, and time-varying intervention-timing analysis (Sections 3.7-3.8, 5.2.11-5.2.12). Multi-level explainability, SHAP for the tabular pathway and Grad-CAM for the imaging pathway, was integrated throughout and supports the framework's clinical interpretability (Sections 5.1.9, 5.2.9). Finally, "what-if" simulation was demonstrated at the individual-patient level (genetic risk and pharmacological intervention scenarios), the population level (diagnostic subgroup comparison), and, uniquely, at the intervention-timing level: an early-versus-late treatment-timing simulation that, to this thesis's knowledge, no published static ADNI classifier is capable of producing, since it requires the combination of a trained diagnostic model and a temporal progression model that neither component provides alone (Section 5.2.12).

Beyond the originally proposed scope (Section 1.3), the eleven-variant ablation study and the controlled CPU-versus-GPU comparison, discussed in full in Section 6.1, further strengthen these results: they quantify each architectural component's contribution and isolate exactly what compute-platform migration changes and does not change about the resulting model. Section 6.3 consolidates this thesis's limitations, and the honest divergences and caveats surfaced along the way, into a single account.

### 7.2 Future Work

Several directions extend directly from the as-built system's current scope, drawing on both this work's own findings and the interactive-deployment and integration phases envisioned in the original project proposal (Sections 7.8-7.9 of the proposal) but not yet realized.

**Interactive clinical visualization.** The current dashboard (Section 5.2.13) renders 2D multi-planar Grad-CAM overlays. The originally proposed interactive 3D brain visualization of atrophy and activation maps, allowing a clinician to rotate and explore a volumetric render rather than fixed orthogonal slices, remains future work.

**Richer medication and intervention modeling.** RECCMEDS.csv, ADNI's concurrent medications log, was not integrated in the as-built system (Table 1); the current intervention simulation instead applies fixed, literature-informed effect sizes (Lecanemab 30%, Donanemab 35% reduction in MCI-to-Dementia transition probability) uniformly across patients. Integrating per-patient medication history would allow the Digital Twin to model a patient's actual treatment record rather than a hypothetical uniform intervention, and would strengthen the intervention-timing simulation demonstrated in Section 5.2.12 into a validated, data-grounded capability.

**Strengthening the age and demographic pathway.** Given the weak, direction-inconsistent age-sensitivity signal observed in Section 5.2.12, incorporating an additional cognitive-trajectory feature (e.g., ADAS-Cog or CDRSB, both available in ADNIMERGE.csv but not used in the as-built four-feature tabular branch) may sharpen the model's sensitivity to demographic and cognitive-trend risk factors beyond the diagnostic-imaging signal that currently dominates.

**EHR integration and continuous learning.** The original proposal's integration and automation phase (Section 7.9 of the proposal) envisioned connecting the BDT to live Electronic Health Record systems via a standardized interface such as HL7 FHIR, and establishing a continuous-learning feedback loop in which new patient visits automatically refine the Markov transition estimates and, on a periodic retraining cycle, the classifier itself. Realizing this would require production MLOps practices not yet built for this system: versioned model registries, a shadow-mode deployment stage for validating a retrained model against live data before promoting it to serve predictions, and drift monitoring on both the imaging and tabular input distributions.

**Broader external validation.** All results in this thesis are derived from ADNI, a single, if large and well-established, research cohort. Validating the trained classifier and the Markov progression model against an independent cohort, and against ADNI's own newer ADNI-4 phase as it becomes available, would test generalization beyond the specific population and acquisition protocols represented in the ADNI-1 through ADNI-3 data used here.

Taken together, the Brain Digital Twin developed and validated across these two experimental phases establishes a working, end-to-end, and clinically interpretable simulative alternative to static Alzheimer's Disease classifiers, with a clear and concrete path from its current research-validated state toward the interactive, continuously learning clinical deployment envisioned at the outset of this project.

## Acknowledgements

The author gratefully acknowledges the Arab Academy for Science, Technology and Maritime Transport (AASTMT), Sheraton Branch, Cairo, Egypt, for providing the laboratory facilities and the GPU-equipped workstation used to conduct the GPU-based training and validation experiments (Section 5.2) reported in this thesis. The author also thanks his supervisors, Prof. Fahima Maghraby and Assoc. Prof. Ahmed Salem, for their guidance and supervision throughout the design, implementation, and evaluation of this work. Data used in the preparation of this work were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database.

## References

[1] A. Salaün, S. Knight, L. Wingfield, and T. Zhu, "Predicting graft and patient outcomes following kidney transplantation using interpretable machine learning models," *Scientific Reports*, vol. 14, 17356, 2024. Available: https://doi.org/10.1038/s41598-024-66976-0

[2] D. Yoo, G. Divard, M. Raynaud, et al., "A machine learning-driven virtual biopsy system for kidney transplant patients," *Nature Communications*, vol. 15, 554, 2024. Available: https://doi.org/10.1038/s41467-023-44595-z

[3] T. Zhang, M. Chen, and A. A. T. Bui, "AdaDiag: Adversarial domain adaptation of diagnostic prediction with clinical event sequences," *Journal of Biomedical Informatics*, vol. 134, Oct. 2022. Available: https://doi.org/10.1016/j.jbi.2022.104168

[4] W. A. Awuah, F. T. Adebusoye, J. Wellington, L. David, A. Salam, A. L. W. Yee, E. Lansiaux, R. Yarlagadda, T. Garg, T. Abdul-Rahman, J. Kalmanovich, G. D. Miteu, M. Kundu, and I. N. Mykolaivna, "Recent outcomes and challenges of artificial intelligence, machine learning, and deep learning in neurosurgery," *Frontiers in Neuroscience*, vol. 17, no. 1023456, 2023. Available: https://doi.org/10.1016/j.wnsx.2024.100301

[5] E. VanDerHorn and S. Mahadevan, "Digital Twin: Generalization, characterization and implementation," *Computers in Industry*, vol. 123, 2021. Available: https://doi.org/10.1016/j.dss.2021.113524

[6] K. P. Venkatesh, M. M. Raza, and J. C. Kvedar, "Health digital twins as tools for precision medicine: Considerations for computation, implementation, and regulation," *npj Digital Medicine*, vol. 5, 150, 2022. Available: https://doi.org/10.1038/s41746-022-00694-7

[7] J. Gaebel, J. Keller, D. Schneider, A. Lindenmeyer, T. Neumuth, and S. Franke, "The Digital Twin: Modular Model-Based Approach to Personalized Medicine," *Current Directions in Biomedical Engineering*, vol. 7, no. 2, pp. 223-226, 2021. Available: https://doi.org/10.1515/cdbme-2021-2057

[8] G. Coorey, G. A. Figtree, D. F. Fletcher, et al., "The health digital twin to tackle cardiovascular disease: A review of an emerging interdisciplinary field," *npj Digital Medicine*, vol. 5, no. 126, 2022. Available: https://doi.org/10.1038/s41746-022-00640-7

[9] N. K. Chakshu, I. Sazonov, and P. Nithiarasu, "Towards enabling a cardiovascular digital twin for human systemic circulation using inverse analysis," *Biomechanics and Modeling in Mechanobiology*, vol. 20, pp. 449-465, 2021. Available: https://doi.org/10.1007/s10237-020-01393-6

[10] D. Bertolini, A. D. Loukianov, A. M. Smith, D. Li-Bland, Y. Pouliot, J. R. Walsh, and C. K. Fisher, "Modeling disease progression in mild cognitive impairment and Alzheimer's disease with digital twins," *arXiv preprint arXiv:2012.13455*, 2020. Available: https://doi.org/10.48550/arXiv.2012.13455

[11] G. Lozupone, A. Bria, F. Fontanella, F. J. A. Meijer, and C. De Stefano, "AXIAL: Attention-based eXplainability for Interpretable Alzheimer's Localized Diagnosis using 2D CNNs on 3D MRI brain scans," *arXiv preprint arXiv:2407.02418*, 2024. Available: https://doi.org/10.48550/arXiv.2407.02418

[12] S. Basaia, F. Agosta, L. Wagner, E. Canu, G. Magnani, R. Santangelo, and M. Filippi, "Automated classification of Alzheimer's disease and mild cognitive impairment using a single MRI and deep neural networks," *NeuroImage: Clinical*, vol. 21, 101645, 2019. Available: https://doi.org/10.1016/j.nicl.2018.101645

[13] J. Wen, E. Thibeau-Sutre, M. Diaz-Melo, J. Samper-González, A. Routier, S. Bottani, D. Dormont, S. Durrleman, N. Burgos, and O. Colliot, "Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation," *Medical Image Analysis*, vol. 63, 101694, 2020. Available: https://doi.org/10.1016/j.media.2020.101694

[14] R. Kushol, A. Masoumzadeh, D. Huo, S. Kalra, and Y.-H. Yang, "ADDformer: Alzheimer's Disease Detection from Structural MRI Using Fusion Transformer," in *2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI)*, Kolkata, India, 2022, pp. 1-5. doi: 10.1109/ISBI52829.2022.9761421

[15] Z. Hu, Y. Li, Z. Wang, S. Zhang, W. Hou, and the Alzheimer's Disease Neuroimaging Initiative, "Conv-Swinformer: Integration of CNN and shift window attention for Alzheimer's disease classification," *Computers in Biology and Medicine*, vol. 159, 107304, 2023. Available: https://doi.org/10.1016/j.compbiomed.2023.107304

[16] J. Venugopalan, L. Tong, H. R. Hassanzadeh, et al., "Multimodal deep learning models for early detection of Alzheimer's disease stage," *Scientific Reports*, vol. 11, 3254, 2021. Available: https://doi.org/10.1038/s41598-020-74399-w

[17] S. Qiu, P. S. Joshi, M. I. Miller, C. Xue, X. Zhou, C. Karjadi, G. H. Chang, A. S. Joshi, B. Dwyer, S. Zhu, M. Kaku, Y. Zhou, Y. J. Alderazi, A. Swaminathan, S. Kedar, M.-H. Saint-Hilaire, S. H. Auerbach, J. Yuan, E. A. Sartor, R. Au, and V. B. Kolachalama, "Development and validation of an interpretable deep learning framework for Alzheimer's disease classification," *Brain*, vol. 143, no. 6, pp. 1920-1933, Jun. 2020. Available: https://doi.org/10.1093/brain/awaa137

[18] F. Falahati, E. Westman, and A. Simmons, "Multivariate data analysis and machine learning in Alzheimer's disease with a focus on structural magnetic resonance imaging," *Journal of Alzheimer's Disease*, vol. 41, no. 3, pp. 685-708, 2014. Available: https://doi.org/10.3233/JAD-131928

[19] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, "An image is worth 16x16 words: Transformers for image recognition at scale," *arXiv preprint arXiv:2010.11929*, 2020. Available: https://arxiv.org/abs/2010.11929

[20] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *2017 IEEE International Conference on Computer Vision (ICCV)*, Venice, Italy, 2017, pp. 618-626. Available: https://doi.org/10.1109/ICCV.2017.74

[21] World Health Organization, "Dementia: Key Facts," 2023. Available: https://www.who.int/news-room/fact-sheets/detail/dementia

[22] GBD 2019 Dementia Forecasting Collaborators, "Estimation of the global prevalence of dementia in 2019 and forecasted prevalence in 2050: an analysis for the Global Burden of Disease Study 2019," *The Lancet Public Health*, vol. 7, no. 2, pp. e105-e125, 2022. Available: https://doi.org/10.1016/S2468-2667(21)00249-8

[23] National Institute on Aging, "How Is Alzheimer's Disease Treated?," 2021. Available: https://www.nia.nih.gov/health/alzheimers-disease/how-alzheimers-disease-treated

[24] C. H. van Dyck, S. Swanson, R. A. Aisen, et al., "Lecanemab in Early Alzheimer's Disease," *The New England Journal of Medicine*, vol. 388, no. 1, pp. 9-21, 2023. Available: https://doi.org/10.1056/NEJMoa2212948

[25] M. Sarkar, "Integrating Machine Learning and Deep Learning Techniques for Advanced Alzheimer's Disease Detection through Gait Analysis," *Journal of Business and Management Studies*, vol. 7, no. 1, pp. 140-147, 2025. Available: https://doi.org/10.32996/jbms.2025.7.1.8

[26] W. Hechkel and A. Helali, "Unveiling Alzheimer's Disease Early: A Comprehensive Review of Machine Learning and Imaging Techniques," *Archives of Computational Methods in Engineering*, vol. 32, pp. 471-484, 2025. Available: https://doi.org/10.1007/s11831-024-10179-3

[27] M. A. Aghdam, S. Bozdag, F. Saeed, et al., "Machine-learning Models for Alzheimer's Disease Diagnosis Using Neuroimaging Data: Survey, Reproducibility, and Generalizability Evaluation," *Brain Informatics*, vol. 12, no. 8, 2025. Available: https://doi.org/10.1186/s40708-025-00252-3

[28] J. Zhou, Y. Wei, X. Li, et al., "A deep learning model for early diagnosis of Alzheimer's disease combined with 3D CNN and video Swin transformer," *Scientific Reports*, vol. 15, no. 23311, 2025. Available: https://doi.org/10.1038/s41598-025-05568-y

[29] D. Ravi, S. B. Blumberg, S. Ingala, F. Barkhof, D. C. Alexander, N. P. Oxtoby, and the Alzheimer's Disease Neuroimaging Initiative, "Degenerative adversarial neuroimage nets for brain scan simulations: Application in ageing and dementia," *Medical Image Analysis*, vol. 75, 102257, 2022. Available: https://doi.org/10.1016/j.media.2021.102257
