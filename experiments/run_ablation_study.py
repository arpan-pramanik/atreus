"""Ablation Study: Replacement ratio and ensemble size sensitivity."""

import os
import matplotlib.pyplot as plt
import pandas as pd
from src.datasets.synthetic import generate_sea_stream
from src.detectors.adwin import ADWINDetector
from src.models.selective_adaptive import SelectiveAdaptiveEnsemble
from src.evaluation.evaluator import StreamEvaluator


def main():
    print(f"\n==================================================")
    print(f"   Running Ablation Study on Selective Adaptation")
    print(f"==================================================")

    output_dir = "results"
    fig_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    X, y, drifts = generate_sea_stream(
        n_samples=8000,
        drift_points=[2000, 4000, 6000],
        noise_ratio=0.08,
        seed=42,
    )
    evaluator = StreamEvaluator(warmup_samples=500, rolling_window=200)

    # 1. Ablation on Replacement Ratio
    replacement_ratios = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    ratio_results = []

    print("Evaluating Replacement Ratio Sensitivity:")
    for r in replacement_ratios:
        model = SelectiveAdaptiveEnsemble(
            n_estimators=30,
            replacement_ratio=r,
            drift_detector=ADWINDetector(delta=0.005),
            name=f"Selective (r={r})",
        )
        res = evaluator.evaluate(model, X, y, true_drifts=drifts)
        s = res["summary"]
        ratio_results.append({
            "replacement_ratio": r,
            "accuracy": s["accuracy"],
            "f1_score": s["f1_score"],
            "adaptation_time_sec": s["adaptation_time_sec"],
            "adaptation_count": s["adaptation_count"],
        })
        print(f"   r={r:.1f} -> Acc: {s['accuracy']*100:.2f}%, Time: {s['adaptation_time_sec']:.3f}s")

    df_ratio = pd.DataFrame(ratio_results)
    df_ratio.to_csv(os.path.join(output_dir, "ablation_replacement_ratio.csv"), index=False)

    # Plot Replacement Ratio Ablation Curve
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    p1 = ax1.plot(df_ratio["replacement_ratio"] * 100, df_ratio["accuracy"] * 100, "b-o", linewidth=2, label="Accuracy (%)")
    p2 = ax2.plot(df_ratio["replacement_ratio"] * 100, df_ratio["adaptation_time_sec"], "r--s", linewidth=2, label="Adaptation Time (s)")

    ax1.set_xlabel("Component Replacement Ratio (%)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", color="b", fontsize=11)
    ax2.set_ylabel("Adaptation Time (seconds)", color="r", fontsize=11)
    plt.title("Ablation: Accuracy vs Compute Cost under Selective Replacement", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    plots = p1 + p2
    labels = [p.get_label() for p in plots]
    ax1.legend(plots, labels, loc="center right")

    plt.tight_layout()
    ablation_fig = os.path.join(fig_dir, "ablation_replacement_ratio.png")
    plt.savefig(ablation_fig, dpi=300)
    plt.close()
    print(f"Saved Ablation Plot to: {ablation_fig}")


if __name__ == "__main__":
    main()
