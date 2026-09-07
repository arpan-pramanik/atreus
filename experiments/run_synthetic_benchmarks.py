"""Synthetic Streaming Benchmarks Runner (SEA, Agrawal, Hyperplane)."""

import os
import pandas as pd
from src.datasets.synthetic import (
    generate_sea_stream,
    generate_agrawal_stream,
    generate_rotating_hyperplane_stream,
)
from src.detectors.adwin import ADWINDetector
from src.detectors.ddm import DDMDetector
from src.detectors.eddm import EDDMDetector
from src.detectors.page_hinkley import PageHinkleyDetector
from src.models.static_model import StaticStreamingModel
from src.models.full_retrain import FullRetrainingModel
from src.models.selective_adaptive import SelectiveAdaptiveEnsemble
from src.evaluation.evaluator import StreamEvaluator
from src.visualizer import plot_drift_trajectories, plot_performance_and_cost_comparison


def run_benchmark_on_stream(
    stream_name: str,
    X,
    y,
    true_drifts,
    output_dir: str = "results",
    fig_dir: str = "figures",
):
    print(f"\n==================================================")
    print(f"   Running Benchmark on: {stream_name}")
    print(f"   Samples: {len(X)}, Features: {X.shape[1]}, Drift Points: {true_drifts}")
    print(f"==================================================")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    evaluator = StreamEvaluator(warmup_samples=500, rolling_window=200)

    # Instantiate model configurations
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
    ]

    results = []
    summaries = []

    for model in models:
        print(f"-> Evaluating {model.name}...")
        res = evaluator.evaluate(model, X, y, true_drifts=true_drifts)
        results.append(res)
        summaries.append(res["summary"])
        s = res["summary"]
        print(
            f"   [Done] Acc: {s['accuracy']*100:.2f}%, F1: {s['f1_score']*100:.2f}%, "
            f"Adapt Count: {s['adaptation_count']}, Time: {s['adaptation_time_sec']:.3f}s"
        )

    # Save summary dataframe
    df_summary = pd.DataFrame(summaries)
    csv_path = os.path.join(output_dir, f"{stream_name.lower().replace(' ', '_')}_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary to: {csv_path}")

    # Generate Figures
    traj_fig = os.path.join(fig_dir, f"{stream_name.lower().replace(' ', '_')}_trajectories.png")
    plot_drift_trajectories(results, true_drifts=true_drifts, dataset_name=stream_name, save_path=traj_fig)
    print(f"Saved Trajectory Plot to: {traj_fig}")

    comp_fig = os.path.join(fig_dir, f"{stream_name.lower().replace(' ', '_')}_comparison.png")
    plot_performance_and_cost_comparison(summaries, dataset_name=stream_name, save_path=comp_fig)
    print(f"Saved Comparison Plot to: {comp_fig}")

    return df_summary


def main():
    # 1. SEA Benchmark (Abrupt Concept Drift)
    X_sea, y_sea, drifts_sea = generate_sea_stream(
        n_samples=8000,
        drift_points=[2000, 4000, 6000],
        noise_ratio=0.08,
        seed=42,
    )
    run_benchmark_on_stream("SEA Benchmark", X_sea, y_sea, drifts_sea)

    # 2. Agrawal Benchmark (Categorical & Numeric Drift)
    X_agr, y_agr, drifts_agr = generate_agrawal_stream(
        n_samples=8000,
        drift_points=[2500, 5000],
        noise_ratio=0.05,
        seed=42,
    )
    run_benchmark_on_stream("Agrawal Benchmark", X_agr, y_agr, drifts_agr)

    # 3. Rotating Hyperplane (Continuous/Gradual Drift)
    X_hyp, y_hyp, drifts_hyp = generate_rotating_hyperplane_stream(
        n_samples=8000,
        drift_speed=0.003,
        noise_ratio=0.05,
        seed=42,
    )
    run_benchmark_on_stream("Hyperplane Drift Benchmark", X_hyp, y_hyp, drifts_hyp)


if __name__ == "__main__":
    main()
