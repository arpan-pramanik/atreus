"""Comprehensive Benchmark Suite Exactly Matching Concept_Drift_Framework.pptx & plan.md.

Directly benchmarks the requirements from Slide 4 & Slide 5:
- Datasets: SEA, Agrawal, Electricity, Airlines.
- Models: Static Baseline, Full Retrain, Proposed Selective Adaptation Pipeline, Boosted Ensemble.
- Drift Detectors Compared: ADWIN, DDM, EDDM, Page-Hinkley.
- Metrics: Accuracy, F1-Score, Detection Delay, Retraining Count, Computational Cost (Wall-Clock Time).
"""

import os
import pandas as pd
from src.datasets.synthetic import generate_sea_stream, generate_agrawal_stream
from src.datasets.loader import load_electricity_stream, load_airlines_stream
from src.detectors.adwin import ADWINDetector
from src.detectors.ddm import DDMDetector
from src.detectors.eddm import EDDMDetector
from src.detectors.page_hinkley import PageHinkleyDetector
from src.models.static_model import StaticStreamingModel
from src.models.full_retrain import FullRetrainingModel
from src.models.framework_pipeline import SelfHealingConceptDriftFramework
from src.models.boosted_adaptive_forest import BoostedAdaptiveForest
from src.models.self_healing_meta_ensemble import SelfHealingMetaEnsemble
from src.evaluation.evaluator import StreamEvaluator
from src.visualizer import plot_drift_trajectories, plot_performance_and_cost_comparison


def run_benchmark_matrix(
    dataset_fn,
    dataset_name: str,
    true_drifts=None,
    warmup_samples: int = 1000,
    rolling_window: int = 250,
):
    print("\n" + "=" * 80)
    print(f"   EVALUATION MATRIX: {dataset_name.upper()}")
    print("=" * 80)

    output_dir = "results"
    fig_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    X, y, name = dataset_fn()
    print(f"Scale: {len(X):,} instances | {X.shape[1]} features")

    evaluator = StreamEvaluator(warmup_samples=warmup_samples, rolling_window=rolling_window)

    # All model and detector permutations from the presentation slides
    models = [
        StaticStreamingModel(name="Static (No Adapt)"),
        FullRetrainingModel(drift_detector=ADWINDetector(delta=0.005), name="Full Retraining (ADWIN)"),
        FullRetrainingModel(drift_detector=DDMDetector(warm_start=30), name="Full Retraining (DDM)"),
        
        # Proposed Selective Adaptation Framework (Slide 3 & 5)
        SelfHealingConceptDriftFramework(drift_detector=ADWINDetector(delta=0.005), name="Proposed Selective Adapt (ADWIN)"),
        SelfHealingConceptDriftFramework(drift_detector=DDMDetector(warm_start=30), name="Proposed Selective Adapt (DDM)"),
        SelfHealingConceptDriftFramework(drift_detector=EDDMDetector(warm_start=30), name="Proposed Selective Adapt (EDDM)"),
        SelfHealingConceptDriftFramework(drift_detector=PageHinkleyDetector(threshold=25.0), name="Proposed Selective Adapt (Page-Hinkley)"),
        
        # Boosted Research Ensembles
        BoostedAdaptiveForest(n_estimators=30, detector_type="ddm", name="Boosted ARF-DWM Forest"),
        SelfHealingMetaEnsemble(n_tree_components=20, name="Self-Healing Dual-Memory Meta-Ensemble"),
    ]

    results = []
    summaries = []

    print("-" * 80)
    print(f"{'Model Architecture & Detector':<46} | {'Accuracy':<9} | {'F1-Score':<9} | {'Adapt Count':<11} | {'Time (s)':<8}")
    print("-" * 80)

    for model in models:
        res = evaluator.evaluate(model, X, y, true_drifts=true_drifts)
        results.append(res)
        s = res["summary"]
        summaries.append(s)
        print(
            f"{model.name:<46} | {s['accuracy']*100:>7.2f}% | {s['f1_score']*100:>7.2f}% | "
            f"{s['adaptation_count']:>11} | {s['adaptation_time_sec']:>7.3f}s"
        )

    print("-" * 80)

    # Save summary dataframe
    clean_name = dataset_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    df_summary = pd.DataFrame(summaries)
    csv_path = os.path.join(output_dir, f"{clean_name}_comprehensive_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    # Generate Figures
    traj_fig = os.path.join(fig_dir, f"{clean_name}_comprehensive_trajectories.png")
    plot_drift_trajectories(results, true_drifts=true_drifts, dataset_name=dataset_name, save_path=traj_fig)

    comp_fig = os.path.join(fig_dir, f"{clean_name}_comprehensive_comparison.png")
    plot_performance_and_cost_comparison(summaries, dataset_name=dataset_name, save_path=comp_fig)

    return df_summary


def main():
    print("=" * 80)
    print("   STARTING COMPREHENSIVE PRESENTATION BENCHMARKS SUITE")
    print("=" * 80)

    # 1. SEA Concepts Benchmark (Abrupt Concept Drift)
    run_benchmark_matrix(
        lambda: generate_sea_stream(n_samples=10000, drift_points=[2500, 5000, 7500], noise_ratio=0.05, seed=42),
        "SEA Concepts Benchmark (Abrupt Drift)",
        true_drifts=[2500, 5000, 7500],
        warmup_samples=800,
    )

    # 2. Agrawal Stream Generator Benchmark (Categorical / Subspace Drift)
    run_benchmark_matrix(
        lambda: generate_agrawal_stream(n_samples=10000, drift_points=[3000, 6000], noise_ratio=0.05, seed=42),
        "Agrawal Stream Benchmark (Subspace Drift)",
        true_drifts=[3000, 6000],
        warmup_samples=800,
    )

    # 3. Real-World NSW Electricity Market Stream (NSW Price UP/DOWN)
    run_benchmark_matrix(
        lambda: load_electricity_stream(max_samples=20000),
        "NSW Electricity Market (Real-World Stream)",
        warmup_samples=1000,
    )

    # 4. Real-World Airlines Flight Delay Stream
    run_benchmark_matrix(
        lambda: load_airlines_stream(max_samples=15000),
        "Airlines Flight Delay (Real-World Stream)",
        warmup_samples=1000,
    )


if __name__ == "__main__":
    main()
