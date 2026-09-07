# Handling Concept Drift in Streaming Data: An Adaptive Framework for Dynamic Data

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/next.js-16.3-black.svg)](https://nextjs.org/)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-RTX%205070-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end research framework and real-time streaming platform designed to detect statistical concept drift in dynamic data streams and perform **selective, localized model adaptation** rather than costly full retraining from scratch.

Developed in accordance with the project specification (`plan.md`) and research presentation (`Concept_Drift_Framework.pptx`).

---

## Table of Contents

- [1. Research Motivation & Problem Statement](#1-research-motivation--problem-statement)
- [2. System Architecture & Adaptation Loop](#2-system-architecture--adaptation-loop)
- [3. Algorithms & Methodology](#3-algorithms--methodology)
  - [3.1 Drift Detection Algorithms](#31-drift-detection-algorithms)
  - [3.2 Predictive Model Architectures](#32-predictive-model-architectures)
  - [3.3 Selective Adaptation vs. Baselines](#33-selective-adaptation-vs-baselines)
- [4. Benchmark Datasets](#4-benchmark-datasets)
- [5. Hardware-Accelerated Compute Engine](#5-hardware-accelerated-compute-engine)
- [6. Interactive Streaming Control Panel & Visualizer](#6-interactive-streaming-control-panel--visualizer)
- [7. Output & Telemetry Export System](#7-output--telemetry-export-system)
- [8. Quick Start Guide](#8-quick-start-guide)
- [9. Empirical Benchmark Results](#9-empirical-benchmark-results)
- [10. References & Research Papers](#10-references--research-papers)

---

## 1. Research Motivation & Problem Statement

In real-world data streaming applications (finance, sensor networks, e-commerce, weather modeling), the statistical properties of the target variable change over time in an unforeseen way. This phenomenon—**Concept Drift**—causes machine learning models deployed under stationary assumptions to silently degrade in accuracy.

Traditional machine learning fails in production because:
1. It assumes a static, stationary data distribution.
2. Models are trained once and deployed without continuous validation.
3. Systems cannot detect when predictions grow stale.
4. Naive full retraining from scratch on every incoming batch is computationally wasteful, slow, and discards valuable historic concept knowledge.

### Project Objectives

1. **Real-Time Detection**: Continuously monitor streaming prediction errors and statistical properties to detect drift as soon as it emerges.
2. **Objective Detector Comparison**: Empirically evaluate and compare ADWIN, DDM, EDDM, and Page-Hinkley across diverse drift manifestations (abrupt, gradual, recurring, and subspace).
3. **Selective Adaptation**: Design a self-healing ensemble that updates only the sub-estimators affected by the shift, preserving stable sub-concepts.
4. **Trade-Off Optimization**: Evaluate empirical Pareto frontiers between prequential accuracy, detection delay, adaptation count, and wall-clock compute cost.
5. **Real-World Benchmarking**: Validate on established real-world streams (NSW Electricity Market, Airlines Flight Delays, Forest Covertype, Phishing) and synthetic generators (SEA, Agrawal).
6. **Deployable Production Pipeline**: Provide a unified, one-click interactive streaming engine with hardware acceleration (NVIDIA CUDA + Ryzen multi-threading) and multi-format output exports.

---

## 2. System Architecture & Adaptation Loop

The framework implements the closed-loop prequential (test-then-train) streaming lifecycle specified in `plan.md` and Slide 3 of `Concept_Drift_Framework.pptx`:

```
 Incoming Streaming Instance (x_t, y_t)
                   │
                   ▼
       [ 1. Model Prediction ]
       Compute y_hat_t = f(x_t)
       Evaluate error e_t = I(y_hat_t != y_t)
                   │
                   ▼
       [ 2. Error Monitoring ]
       Feed e_t into online detector
       Track rolling window & cumulative metrics
                   │
                   ▼
       [ 3. Drift Detection ]
       Does statistical test signal a distribution shift?
            ├── NO  ──> Incremental Online Maintenance / Buffer Ingest
            │
            └── YES ──> [ 4. Performance Degradation Analysis ]
                             │
                             ▼
                        [ 5. Selective Adaptation Loop ]
                        • Dynamic feature relevance re-ranking
                        • Evaluate individual sub-estimator errors
                        • Replace worst-performing subset (e.g., 30%)
                        • Retain stable knowledge estimators
                        • Exponential decay re-weighting
                             │
                             ▼
                        [ 6. Updated Self-Healing Model ]
```

---

## 3. Algorithms & Methodology

### 3.1 Drift Detection Algorithms

| Detector | Method Type | Mechanism | Best Suited For |
|---|---|---|---|
| **ADWIN** (*Bifet & Gavalda, 2007*) | Adaptive Windowing | Dynamically resizes observation window using Hoeffding bound cutoffs when sub-window means diverge | Abrupt & gradual drift with rigorous theoretical false-positive guarantees |
| **DDM** (*Gama et al., 2004*) | Error Rate Modeling | Tracks binomial error probability $p_t$ and standard deviation $s_t$; triggers warning at $p_t + s_t \ge p_{min} + 2s_{min}$ and drift at $+ 3s_{min}$ | Abrupt concept changes with stationary pre-drift baselines |
| **EDDM** (*Baena-García et al., 2006*) | Inter-Error Distance | Tracks distance between consecutive errors; sensitive to slowing error intervals | Gradual and subtle concept shifts |
| **Page-Hinkley** (*Page, 1954*) | Sequential Analysis | Cumulative sum of deviations from running mean with configurable threshold margin | Fast change detection with minimal computational overhead |

### 3.2 Predictive Model Architectures

1. **Proposed Self-Healing Adaptive Framework** (`SelfHealingConceptDriftFramework`):
   - Multi-layer ensemble combining randomized subspace decision trees, online linear support vector machines, and dynamic feature selection.
   - Re-evaluates component weights upon drift signal and selectively adapts degraded sub-models.
2. **Adaptive Online SVM** (`AdaptiveOnlineSVM`):
   - Online stochastic gradient margin classifier (`SGDClassifier`) with modified Huber / hinge loss.
   - Incorporates sliding buffer maintenance and drift-triggered accelerated margin recalibration.
3. **Hardware-Accelerated Adaptive XGBoost** (`GPUAcceleratedAdaptiveXGBoost`):
   - Leverages NVIDIA GeForce RTX 5070 CUDA execution with tree-based warm-start boosting and pruning upon detected shifts.
4. **Boosted Adaptive Random Forest** (`BoostedAdaptiveForest`):
   - Implements Adaptive Random Forest (ARF) with background shadow trees and Dynamic Weighted Majority (DWM) pruning.
5. **Self-Healing Dual-Memory Meta-Ensemble** (`SelfHealingMetaEnsemble`):
   - Implements short-term reactive memory and long-term proactive memory banks to seamlessly counter both novel abrupt drift and recurring concepts.
6. **Selective Adaptive Ensemble** (`SelectiveAdaptiveEnsemble`):
   - Standard benchmark implementation replacing a configurable ratio (e.g., 30%) of the poorest-performing trees upon drift.
7. **Baselines**:
   - **Static Streaming Model** (`StaticStreamingModel`): Trained once on initial warmup; serves as a stationary baseline.
   - **Full Retraining Baseline** (`FullRetrainingModel`): Discards the entire model and retrains from scratch on a sliding window whenever drift triggers.

---

## 4. Benchmark Datasets

The repository includes real-world, synthetic, and Hugging Face streaming datasets:

- **NSW Electricity Market Stream** (`electricity_real.csv`): 45,312 instances, 8 features. Real-world electricity market prices in New South Wales, Australia, featuring seasonal, demand, and price fluctuations.
- **Airlines Flight Delay Stream** (`airlines_real.csv`): 15,000 instances, 7 features. Predicting flight delays with strong seasonal, routing, and weather shifts.
- **Forest Covertype Stream** (`covertype_real.csv`): 30,000 instances, 54 cartographic features. Highly dimensional spatial shift.
- **Phishing Websites Stream** (`phishing_real.csv`): 11,055 instances, 30 features. Evolving adversarial web techniques.
- **SEA Concepts Generator** (`synthetic.py`): Controlled abrupt concept drift at predefined intervals across 3 numerical features with varying hyperplane thresholds.
- **Agrawal Stream Generator** (`synthetic.py`): Subspace drift with 9 mixed numerical and categorical attributes simulating loan approval rules.
- **Rotating Hyperplane Generator** (`synthetic.py`): Continuous, gradual multidimensional drift.
- **Hugging Face Streaming Hub**: Direct streaming support for `scikit-learn/adult-census-income`, `inria-soda/tabular-benchmark:bank-marketing`, and `credit-default-risk`.

---

## 5. Hardware-Accelerated Compute Engine

Atreus is optimized to take full advantage of modern workstation hardware:

- **GPU Acceleration**: NVIDIA GeForce RTX 5070 Laptop GPU 8GB GDDR7 (Blackwell architecture). Accelerates gradient boosted tree updates, tensor operations, and CUDA memory residency.
- **CPU Parallelism**: AMD Ryzen 9 9955HX (16 Cores / 32 Execution Threads). Parallelizes ensemble subtree fits, prequential scoring batches, and dataset streaming I/O.
- **Active Telemetry**: Real-time polling of GPU utilization, VRAM usage/total, CPU thread activity, and system RAM.

---

## 6. Interactive Streaming Control Panel & Visualizer

A zero-animation, high-density black and white interface built with Next.js 16 (Turbopack) and Tailwind CSS:

- **Live SVG Trajectory Visualizer**: Zero-dependency vector rendering of Cumulative Accuracy (solid line), Windowed Accuracy (dotted line), and Drift Markers (vertical dashed lines).
- **Component Modification Log**: Complete audit trail showing exact sample indices where drift was triggered, sub-components modified, adaptation latency (ms), and compute binding.
- **Online Feature Selection Matrix**: Real-time visualization of retained vs. pruned features across drift boundaries.
- **Live Hardware Telemetry**: Live metrics for Ryzen 9 9955HX and RTX 5070 CUDA binding.
- **Custom Dataset Ingest**: Upload any local CSV or import any Hugging Face tabular dataset on the fly.

---

## 7. Output & Telemetry Export System

Both the web interface and the REST API support multi-format exports for empirical verification and publication:

| Format | File Extension | Content Description |
|---|---|---|
| **Summary Metrics** | `.csv` | Overall Cumulative Accuracy, Window Accuracy, F1-Score, Processed Samples, Drift Counts, Adaptations, Latency (ms), Throughput (samples/s), Hardware Binding, and Timestamp |
| **Stream Trajectory** | `.csv` | Sample-by-sample trajectory (`sample,cumulative_accuracy_pct,window_accuracy_pct,f1_score_pct,drift_signal`) |
| **Component Audit Log** | `.csv` | Chronological adaptation log (`sample_index,event,components_affected,adaptation_latency_ms,device`) |
| **Execution Logs** | `.txt` | Full raw console trace with timestamps and logging levels |
| **Full Run Package** | `.json` | Un-truncated run snapshot containing hardware specs, configs, evaluation metrics, trajectory time-series, modifications, and logs |

REST API Endpoints:
```
GET /api/export?format=summary_csv
GET /api/export?format=trajectory_csv
GET /api/export?format=modifications_csv
GET /api/export?format=logs
GET /api/export?format=json
```

---

## 8. Quick Start Guide

### Prerequisites
- Linux / macOS / Windows
- Python 3.10+ (tested on Python 3.14)
- Node.js v18+ (tested on Node v24) and npm / pnpm / bun

### Single-Command Start (Backend + Frontend)
```bash
# Clone the repository
git clone https://github.com/arpan-pramanik/atreus.git
cd atreus

# Install dependencies
pip install -r requirements.txt
npm --prefix frontend install

# Launch both Python engine and Next.js frontend
npm run dev
# or
./start.sh
```

- Web Interface: [http://localhost:3000](http://localhost:3000)
- Backend Telemetry API: [http://localhost:8000](http://localhost:8000)

### Single-Command Stop
```bash
npm run stop
# or
./stop.sh
```

### Running Automated Test Suite
```bash
PYTHONPATH=. .venv/bin/pytest tests/
```

### Running Offline Benchmarks
```bash
# Synthetic streams benchmark (SEA, Agrawal, Hyperplane)
PYTHONPATH=. .venv/bin/python experiments/run_synthetic_benchmarks.py

# Real-world streams benchmark (Electricity, Airlines, Covertype)
PYTHONPATH=. .venv/bin/python experiments/run_realworld_benchmarks.py

# Ablation study on replacement ratio
PYTHONPATH=. .venv/bin/python experiments/run_ablation_study.py

# Comprehensive presentation suite
PYTHONPATH=. .venv/bin/python experiments/run_comprehensive_presentation_benchmark.py
```

---

## 9. Empirical Benchmark Results

Comprehensive evaluation over benchmark streams demonstrates the superiority of selective adaptation over static baselines and full retraining:

| Dataset | Stream Type | Samples | Static Model | Full Retrain | Proposed Selective | Compute Savings vs Full Retrain |
|---|---|---|---|---|---|---|
| **SEA Concepts** | Abrupt Drift | 10,000 | 79.4% | 88.2% | **88.9%** | **78.4% less compute** |
| **Agrawal Stream** | Subspace Shift | 10,000 | 67.2% | 85.1% | **87.4%** | **81.2% less compute** |
| **NSW Electricity** | Real Recurring | 45,312 | 68.3% | 80.5% | **82.6%** | **84.0% less compute** |
| **Airlines Delay** | Real Seasonal | 15,000 | 59.1% | 66.8% | **68.7%** | **76.9% less compute** |
| **Covertype 30k** | High Dimensional | 30,000 | 72.4% | 86.3% | **88.1%** | **83.5% less compute** |

All raw numerical summaries are preserved in `./results/` and publication plots in `./figures/`.

---

## 10. References & Research Papers

The project is grounded in literature on adaptive stream learning. Copies of key papers are provided in `./papers/`:

1. **Bifet, A., & Gavalda, R.** (2007). *Learning from Time-Changing Data with Adaptive Windowing*. Proceedings of the 2007 SIAM International Conference on Data Mining (SDM).
2. **Gama, J., Medas, P., Castillo, G., & Rodrigues, P.** (2004). *Learning with Drift Detection*. SBIA 2004: Advances in Artificial Intelligence.
3. **Baena-García, M., del Campo-Ávila, J., Fidalgo, R., Bifet, A., Gavaldà, R., & Morales-Bueno, R.** (2006). *Early Drift Detection Method*. Fourth International Workshop on Knowledge Discovery from Data Streams.
4. **Gomes, H. M., et al.** (2017). *Adaptive Random Forests for Evolving Data Stream Classification*. Machine Learning, 106(9-10), 1469-1495.
5. **Kolter, J. Z., & Maloof, M. A.** (2007). *Dynamic Weighted Majority: An Ensemble Method for Drifting Concepts*. Journal of Machine Learning Research (JMLR), 8, 2337-2364.
6. **Lu, J., et al.** (2018). *Learning under Concept Drift: A Review*. IEEE Transactions on Knowledge and Data Engineering (TKDE).
7. **Losing, V., Hammer, B., & Wersing, H.** (2018). *Self-Adjusting Memory (SAM) for Streaming Concept Drift*. IEEE TKDE.
8. **Krawczyk, B., et al.** (2017). *Ensemble Learning for Data Stream Analysis: A Survey*. Information Fusion.
