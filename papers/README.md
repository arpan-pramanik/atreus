# Research Papers & Theoretical Foundations

This directory contains foundational and advanced research papers directly guiding the implementation, drift detectors, online feature selection, dual-memory systems, and adaptive learning strategies in **Atreus**.

---

## Complete Research Bibliography

### 1. Learning under Concept Drift: A Review
- **Authors**: Jie Lu, Anjin Liu, Fan Dong, Feng Gu, Joao Gama, Guangquan Zhang
- **Venue**: *IEEE Transactions on Knowledge and Data Engineering (TKDE)*
- **File**: `Learning_under_Concept_Drift_A_Review.pdf`
- **Contributions to Framework**:
  - Taxonomy of concept drift (abrupt, gradual, incremental, recurring).
  - Prequential (Test-Then-Train) evaluation methodology.
  - Statistical drift detection thresholds (Warning vs Drift).

---

### 2. Self-Adjusting Memory for Concept Drift (SAM-kNN / SAM Ensembles)
- **Authors**: Viktor Losing, Barbara Hammer, Heiko Wersing
- **Venue**: *IEEE International Conference on Data Mining (ICDM)*
- **File**: `Self_Adjusting_Memory_for_Streaming_Concept_Drift.pdf`
- **Contributions to Framework**:
  - **Dual-Memory Architecture**: Short-Term Fast Adaptation Memory (STM) + Long-Term Consolidated Memory (LTM).
  - Preserves recurring historical concepts without catastrophic forgetting.
  - Zero-Shot Concept Recall via statistical distribution distance matching.

---

### 3. Ensemble Learning for Data Stream Analysis: A Survey
- **Authors**: Bartosz Krawczyk, Leandro L. Minku, João Gama, Jerzy Stefanowski, Michał Woźniak
- **Venue**: *Information Fusion*
- **File**: `Adaptive_Learning_Streaming_Concept_Drift_Survey.pdf`
- **Contributions to Framework**:
  - Dynamic Ensemble Selection (DES) and selective component update.
  - Performance-weighted combination of diverse base learners.
  - Trade-offs between full retraining cost and selective adaptation.

---

### 4. Adaptive Random Forests for Evolving Data Stream Classification
- **Authors**: Heitor M. Gomes, Albert Bifet, Jesse Read, Jean Paul Barddal, Fabricio Enembreck, Bernhard Pfharinger, Geoff Holmes, Talel Abdessalem
- **Venue**: *ACM Transactions on Knowledge Discovery from Data (TKDD) / ACM CSUR*
- **File**: `Ensemble_Methods_for_Classifying_Streaming_Data_with_Concept_Drift.pdf`
- **Contributions to Framework**:
  - **Component-Level Micro-Detectors**: Independent drift trackers per tree component.
  - **Background Shadow Trees**: Pre-warming candidate learners in warning zones to eliminate cold-start lag.

---

### 5. Dynamic Heterogeneous Ensembles & Meta-Learning for Concept Drift
- **Authors**: J. Read, B. Pfahringer, G. Holmes, E. Frank
- **File**: `Heterogeneous_Ensembles_Concept_Drift.pdf` & `Meta_Learning_Concept_Drift_Streams.pdf`
- **Contributions to Framework**:
  - Heterogeneous model diversity: Decision Trees + Linear Margins (Online SGD/SVM) + Naive Bayes.
  - Meta-learning concept fingerprinting for fast recovery.

---

### 6. DriftMoE: Mixture of Experts for Streaming Classification with Concept Drift
- **Authors**: Miguel Aspis, Sebastián A. Cajas Ordóñez, Andrés L. Suárez-Cetrulo, Ricardo Simón Carbajo
- **Venue**: *Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD 2025 / 2026)*
- **Contributions to Framework**:
  - **Dynamic Gating Router**: Online Softmax router selecting Top-K experts per streaming instance with symbiotic multi-hot correctness feedback.
  - **Heterogeneous Hardware-Accelerated Expert Pool**: Co-locates NVIDIA CUDA GPU-resident Gradient Boosted Decision Trees (XGBoost `hist`), Subspace Decision Trees, and Online Incremental Margin classifiers.
  - **Micro-Adaptation**: Component-level drift tracking replacing solely the degraded experts to prevent catastrophic forgetting.

