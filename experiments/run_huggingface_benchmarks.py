"""Comprehensive Benchmark Suite on Hugging Face Streaming Datasets."""

import os
import pandas as pd
from src.datasets.hf_loader import (
    load_hf_adult_income_stream,
    load_hf_bank_marketing_stream,
    load_hf_credit_default_stream,
)
from src.detectors.adwin import ADWINDetector
from src.detectors.ddm import DDMDetector
from src.detectors.page_hinkley import PageHinkleyDetector
from src.models.static_model import StaticStreamingModel
from src.models.full_retrain import FullRetrainingModel
from src.models.selective_adaptive import SelectiveAdaptiveEnsemble
from src.models.boosted_adaptive_forest import BoostedAdaptiveForest
from src.models.self_healing_meta_ensemble import SelfHealingMetaEnsemble
from src.models.xgboost_adaptive import GPUAcceleratedAdaptiveXGBoost
from src.evaluation.evaluator import StreamEvaluator
from src.visualizer import plot_drift_trajectories, plot_performance_and_cost_comparison


def run_hf_dataset_test(dataset_loader, dataset_name: str, warmup_samples: int = 1000, rolling_window: int = 300):
    print("\n" + "=" * 75)
    print(f"  HUGGING FACE STREAMING BENCHMARK: {dataset_name.upper()}")
    print("=" * 75)

    output_dir = "results"
    fig_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    X, y, name = dataset_loader()
    print(f"HF Stream Scale: {len(X):,} actual instances | {X.shape[1]} input features")
    class_dist = {int(k): int(v) for k, v in dict(pd.Series(y).value_counts()).items()}
    print(f"Class Distribution: {class_dist}")

    evaluator = StreamEvaluator(warmup_samples=warmup_samples, rolling_window=rolling_window)

    models = [
        StaticStreamingModel(name="Static (No Adapt)"),
        FullRetrainingModel(drift_detector=ADWINDetector(delta=0.005), name="Full Retrain (ADWIN)"),
        FullRetrainingModel(drift_detector=DDMDetector(warm_start=30), name="Full Retrain (DDM)"),
        SelectiveAdaptiveEnsemble(
            n_estimators=25,
            replacement_ratio=0.3,
            drift_detector=ADWINDetector(delta=0.005),
            name="Selective Adapt (ADWIN)",
        ),
        SelectiveAdaptiveEnsemble(
            n_estimators=25,
            replacement_ratio=0.3,
            drift_detector=DDMDetector(warm_start=30),
            name="Selective Adapt (DDM)",
        ),
        BoostedAdaptiveForest(
            n_estimators=25,
            detector_type="ddm",
            name="Boosted ARF-DWM Forest (Micro-DDM)",
        ),
        SelfHealingMetaEnsemble(
            n_tree_components=20,
            enable_feature_selection=True,
            name="Self-Healing Dual-Memory Meta-Ensemble",
        ),
        GPUAcceleratedAdaptiveXGBoost(
            n_estimators=30,
            drift_detector=ADWINDetector(delta=0.005),
            device="cuda",
            n_jobs=32,
            name="Adaptive XGBoost (GPU RTX 5070)",
        ),
    ]

    results = []
    summaries = []

    print("-" * 75)
    print(f"{'Model Name':<42} | {'Accuracy':<9} | {'F1-Score':<9} | {'Adapt Count':<11} | {'Time (s)':<8}")
    print("-" * 75)

    for model in models:
        res = evaluator.evaluate(model, X, y)
        results.append(res)
        s = res["summary"]
        summaries.append(s)
        print(
            f"{model.name:<42} | {s['accuracy']*100:>7.2f}% | {s['f1_score']*100:>7.2f}% | "
            f"{s['adaptation_count']:>11} | {s['adaptation_time_sec']:>7.3f}s"
        )

    print("-" * 75)

    # Save summary dataframe
    clean_name = dataset_name.lower().replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
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
    print("=" * 75)
    print("  EXECUTING HUGGING FACE STREAMING BENCHMARKS")
    print("=" * 75)

    # 1. Hugging Face Adult Census Income (32,561 records)
    run_hf_dataset_test(
        load_hf_adult_income_stream,
        "HuggingFace Adult Census Income (32k Stream)",
        warmup_samples=1500,
    )

    # 2. Hugging Face Bank Marketing Stream (10,578 records)
    run_hf_dataset_test(
        load_hf_bank_marketing_stream,
        "HuggingFace Bank Marketing Stream (10k Stream)",
        warmup_samples=800,
    )

    # 3. Hugging Face Credit Card Default Stream (13,272 records)
    run_hf_dataset_test(
        load_hf_credit_default_stream,
        "HuggingFace Credit Default Risk (13k Stream)",
        warmup_samples=1000,
    )


if __name__ == "__main__":
    main()
