"""Visualization module for concept drift benchmarks and adaptive models."""

import os
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_drift_trajectories(
    results: List[Dict[str, Any]],
    true_drifts: Optional[List[int]] = None,
    dataset_name: str = "Synthetic Stream",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot rolling accuracy curves comparing models over time with drift markers.
    """
    plt.figure(figsize=(13, 6))
    
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
    
    for i, res in enumerate(results):
        model_name = res["model_name"]
        rolling_acc = res["rolling_accuracies"]
        warmup = res["warmup_samples"]
        x_axis = np.arange(warmup, warmup + len(rolling_acc))
        c = colors[i % len(colors)]
        
        plt.plot(x_axis, rolling_acc, label=model_name, color=c, alpha=0.85, linewidth=2.0)
        
        # Plot detected drift markers
        drifts = res["detected_drifts"]
        if drifts:
            plt.scatter(
                drifts,
                [rolling_acc[min(d - warmup, len(rolling_acc) - 1)] for d in drifts],
                color=c,
                marker="o",
                s=40,
                zorder=5,
            )

    # Plot true drift vertical lines
    if true_drifts:
        for idx, td in enumerate(true_drifts):
            plt.axvline(
                x=td,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label="Ground Truth Drift" if idx == 0 else "",
            )

    plt.title(f"Prequential Accuracy Under Concept Drift: {dataset_name}", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Streaming Sample Index (Time Step)", fontsize=12)
    plt.ylabel("Prequential Rolling Accuracy", fontsize=12)
    plt.ylim(0.4, 1.02)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=10, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_performance_and_cost_comparison(
    summaries: List[Dict[str, Any]],
    dataset_name: str = "Benchmark",
    save_path: Optional[str] = None,
) -> None:
    """
    Generate side-by-side bar chart comparison:
    1. Accuracy & F1-Score
    2. Retraining Count & Wall-clock Time (Cost Reduction)
    """
    df = pd.DataFrame(summaries)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Accuracy and F1
    x = np.arange(len(df))
    width = 0.35
    
    axes[0].bar(x - width/2, df["accuracy"] * 100, width, label="Accuracy (%)", color="#1f77b4", alpha=0.85)
    axes[0].bar(x + width/2, df["f1_score"] * 100, width, label="F1-Score (%)", color="#2ca02c", alpha=0.85)
    axes[0].set_ylabel("Percentage (%)", fontsize=11)
    axes[0].set_title(f"Predictive Performance ({dataset_name})", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["model"], rotation=20, ha="right", fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right")

    # 2. Adaptation Compute Time
    axes[1].bar(x, df["adaptation_time_sec"], width=0.5, color="#d62728", alpha=0.85)
    axes[1].set_ylabel("Adaptation CPU Time (seconds)", fontsize=11)
    axes[1].set_title("Computational Cost (Lower is Better)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["model"], rotation=20, ha="right", fontsize=9)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
