"""Comprehensive Real-World Streaming Benchmarks Runner on Actual Datasets.

Evaluates on:
1. Full Authentic NSW Electricity Market (45,312 samples)
2. Authentic Phishing Websites (11,055 samples)
3. Authentic Airlines Flight Delay Stream (30,000 samples)
"""

import os
import pandas as pd
from src.datasets.loader import (
    load_electricity_stream,
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
from src.evaluation.evaluator import StreamEvaluator
from src.visualizer import plot_drift_trajectories, plot_performance_and_cost_comparison


def run_benchmark(dataset_fn, dataset_name: str, warmup_samples: int = 1000, rolling_window: int = 300):
    print(f"\n==================================================")
    print(f"   Running Real-World Benchmark on: {dataset_name}")
    print(f"==================================================")

    output_dir = "results"
    fig_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    X, y, name = dataset_fn()
    print(f"Loaded {len(X)} actual streaming records with {X.shape[1]} features.")
    print(f"Class distribution: {dict(pd.Series(y).value_counts())}")

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
        GPUAcceleratedAdaptiveXGBoost(
            n_estimators=40,
            drift_detector=ADWINDetector(delta=0.005),
            device="cuda",
            n_jobs=32,
            name="Adaptive XGBoost (GPU/RTX 5070)",
        ),
    ]

    results = []
    summaries = []

    for model in models:
        print(f"-> Evaluating {model.name}...")
        res = evaluator.evaluate(model, X, y)
        results.append(res)
        summaries.append(res["summary"])
        s = res["summary"]
        print(
            f"   [Done] Acc: {s['accuracy']*100:.2f}%, F1: {s['f1_score']*100:.2f}%, "
            f"Adapt Count: {s['adaptation_count']}, Time: {s['adaptation_time_sec']:.3f}s"
        )

    # Save summary dataframe
    clean_name = dataset_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    df_summary = pd.DataFrame(summaries)
    csv_path = os.path.join(output_dir, f"{clean_name}_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary to: {csv_path}")

    # Generate Figures
    traj_fig = os.path.join(fig_dir, f"{clean_name}_trajectories.png")
    plot_drift_trajectories(results, dataset_name=dataset_name, save_path=traj_fig)
    print(f"Saved Trajectory Plot to: {traj_fig}")

    comp_fig = os.path.join(fig_dir, f"{clean_name}_comparison.png")
    plot_performance_and_cost_comparison(summaries, dataset_name=dataset_name, save_path=comp_fig)
    print(f"Saved Comparison Plot to: {comp_fig}")

    return df_summary


def main():
    # 1. Full Real NSW Electricity Dataset (45,312 rows)
    run_benchmark(lambda: load_electricity_stream(max_samples=25000), "NSW Electricity Market (Real)", warmup_samples=1000)

    # 2. Authentic Phishing Websites Stream (11,055 rows)
    run_benchmark(load_phishing_stream, "Phishing Websites (Real)", warmup_samples=800)

    # 3. Authentic Airlines Flight Delays (25,000 rows)
    run_benchmark(lambda: load_airlines_stream(max_samples=25000), "Airlines Flight Delay (Real)", warmup_samples=1000)


if __name__ == "__main__":
    main()
