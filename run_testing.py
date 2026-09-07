"""Full-Scale Rigorous Testing Script on Actual Real-World Streaming Datasets."""

import os
import sys
import pandas as pd
from src.datasets.loader import (
    load_electricity_stream,
    load_covertype_stream,
    load_phishing_stream,
    load_airlines_stream,
)
from src.detectors.adwin import ADWINDetector
from src.detectors.ddm import DDMDetector
from src.detectors.page_hinkley import PageHinkleyDetector
from src.models.static_model import StaticStreamingModel
from src.models.full_retrain import FullRetrainingModel
from src.models.selective_adaptive import SelectiveAdaptiveEnsemble
from src.models.xgboost_adaptive import GPUAcceleratedAdaptiveXGBoost
from src.models.boosted_adaptive_forest import BoostedAdaptiveForest
from src.models.adaptive_svm import AdaptiveOnlineSVM
from src.evaluation.evaluator import StreamEvaluator
from src.visualizer import plot_drift_trajectories, plot_performance_and_cost_comparison


def run_actual_dataset_test(dataset_loader, dataset_name: str, warmup_samples: int = 1000, rolling_window: int = 300):
    print("\n" + "=" * 70)
    print(f"  RIGOROUS STREAMING TEST: {dataset_name.upper()}")
    print("=" * 70)

    output_dir = "results"
    fig_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    X, y, name = dataset_loader()
    print(f"Stream Scale: {len(X):,} actual instances | {X.shape[1]} input features")
    class_dist = {int(k): int(v) for k, v in dict(pd.Series(y).value_counts()).items()}
    print(f"Class Distribution: {class_dist}")

    evaluator = StreamEvaluator(warmup_samples=warmup_samples, rolling_window=rolling_window)

    models = [
        StaticStreamingModel(name="Static (No Adapt)"),
        FullRetrainingModel(drift_detector=ADWINDetector(delta=0.005), name="Full Retrain (ADWIN)"),
        FullRetrainingModel(drift_detector=DDMDetector(warm_start=30), name="Full Retrain (DDM)"),
        SelectiveAdaptiveEnsemble(
            n_estimators=30,
            replacement_ratio=0.3,
            drift_detector=ADWINDetector(delta=0.005),
            name="Selective Adapt (ADWIN)",
        ),
        SelectiveAdaptiveEnsemble(
            n_estimators=30,
            replacement_ratio=0.3,
            drift_detector=DDMDetector(warm_start=30),
            name="Selective Adapt (DDM)",
        ),
        SelectiveAdaptiveEnsemble(
            n_estimators=30,
            replacement_ratio=0.3,
            drift_detector=PageHinkleyDetector(threshold=30.0),
            name="Selective Adapt (Page-Hinkley)",
        ),
        BoostedAdaptiveForest(
            n_estimators=30,
            detector_type="ddm",
            name="Boosted ARF-DWM Forest (Micro-DDM)",
        ),
        BoostedAdaptiveForest(
            n_estimators=30,
            detector_type="adwin",
            name="Boosted ARF-DWM Forest (Micro-ADWIN)",
        ),
        GPUAcceleratedAdaptiveXGBoost(
            n_estimators=40,
            drift_detector=ADWINDetector(delta=0.005),
            device="cuda",
            n_jobs=32,
            name="Adaptive XGBoost (GPU RTX 5070)",
        ),
        AdaptiveOnlineSVM(
            drift_detector=ADWINDetector(delta=0.005),
            buffer_size=300,
            name="Adaptive Online SVM (Incremental)",
        ),
    ]

    results = []
    summaries = []

    print("-" * 70)
    print(f"{'Model Name':<42} | {'Accuracy':<9} | {'F1-Score':<9} | {'Adapt Count':<11} | {'Time (s)':<8}")
    print("-" * 70)

    for model in models:
        res = evaluator.evaluate(model, X, y)
        results.append(res)
        s = res["summary"]
        summaries.append(s)
        print(
            f"{model.name:<42} | {s['accuracy']*100:>7.2f}% | {s['f1_score']*100:>7.2f}% | "
            f"{s['adaptation_count']:>11} | {s['adaptation_time_sec']:>7.3f}s"
        )

    print("-" * 70)

    # Save summary dataframe
    clean_name = dataset_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    df_summary = pd.DataFrame(summaries)
    csv_path = os.path.join(output_dir, f"{clean_name}_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    # Generate Figures
    traj_fig = os.path.join(fig_dir, f"{clean_name}_trajectories.png")
    plot_drift_trajectories(results, dataset_name=dataset_name, save_path=traj_fig)

    comp_fig = os.path.join(fig_dir, f"{clean_name}_comparison.png")
    plot_performance_and_cost_comparison(summaries, dataset_name=dataset_name, save_path=comp_fig)

    return df_summary


def main():
    print("=" * 70)
    print("  LAUNCHING FULL ACTUAL DATASET RIGOROUS TESTING (BOOSTED ARF-DWM)")
    print("=" * 70)

    # 1. Complete Australian NSW Electricity Stream (45,312 Samples)
    run_actual_dataset_test(
        lambda: load_electricity_stream(max_samples=45312),
        "NSW Electricity Market (Full 45k Stream)",
        warmup_samples=1500,
    )

    # 2. Authentic Forest Covertype Stream (30,000 Samples)
    run_actual_dataset_test(
        lambda: load_covertype_stream(max_samples=30000),
        "Forest Covertype Stream (30k Stream)",
        warmup_samples=1500,
    )


if __name__ == "__main__":
    main()
