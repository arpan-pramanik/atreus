"""Prequential (Test-Then-Train) Streaming Evaluator."""

import time
from typing import Dict, Any, List, Optional
import numpy as np
from src.evaluation.metrics import compute_stream_summary


class StreamEvaluator:
    """
    Prequential (Test-Then-Train) Streaming Evaluation Engine.
    """

    def __init__(
        self,
        warmup_samples: int = 500,
        rolling_window: int = 200,
    ):
        self.warmup_samples = warmup_samples
        self.rolling_window = rolling_window

    def evaluate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        true_drifts: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the prequential evaluation loop on a stream.
        
        Args:
            model: Model instance (Static, FullRetraining, or SelectiveAdaptive).
            X: Stream feature matrix.
            y: Stream ground truth labels.
            true_drifts: Optional ground truth drift positions for synthetic streams.
            verbose: Whether to print progress milestones.
            
        Returns:
            Dict containing predictions, rolling accuracies, detected drifts, and summary metrics.
        """
        n_samples = len(X)
        assert n_samples > self.warmup_samples, "Stream length must exceed warmup samples."

        # 1. Warmup training
        X_warmup = X[: self.warmup_samples]
        y_warmup = y[: self.warmup_samples]
        t_start = time.perf_counter()
        model.fit_initial(X_warmup, y_warmup)
        warmup_time = time.perf_counter() - t_start

        # 2. Prequential stream loop
        test_samples = n_samples - self.warmup_samples
        y_preds = np.zeros(test_samples, dtype=int)
        y_trues = y[self.warmup_samples :]
        is_correct = np.zeros(test_samples, dtype=int)
        rolling_accuracies = np.zeros(test_samples, dtype=float)

        recent_window = []
        adaptation_times = 0.0

        for idx in range(test_samples):
            global_idx = self.warmup_samples + idx
            xi = X[global_idx]
            yi = y_trues[idx]

            # Test step
            pred = model.predict_one(xi)
            y_preds[idx] = pred
            correct = 1 if pred == yi else 0
            is_correct[idx] = correct

            # Rolling accuracy tracking
            recent_window.append(correct)
            if len(recent_window) > self.rolling_window:
                recent_window.pop(0)
            rolling_accuracies[idx] = sum(recent_window) / len(recent_window)

            # Train / Adapt step
            t0 = time.perf_counter()
            model.update_sample(xi, yi, sample_idx=global_idx)
            adaptation_times += (time.perf_counter() - t0)

            if verbose and (idx + 1) % 2000 == 0:
                print(f"[{model.name}] Processed {idx + 1}/{test_samples} samples. Rolling Acc: {rolling_accuracies[idx]:.3f}")

        # Extract drift and adaptation statistics
        detected_drifts = getattr(model, "drift_events", [])
        adaptation_count = getattr(model, "adaptation_count", getattr(model, "retrain_count", 0))
        total_adapt_time = getattr(model, "total_adaptation_time", getattr(model, "total_retrain_time", 0.0))

        # Adjust ground truth drifts relative to stream indices
        summary = compute_stream_summary(
            y_true=y_trues,
            y_pred=y_preds,
            model_name=model.name,
            adaptation_count=adaptation_count,
            adaptation_time=total_adapt_time + warmup_time,
            detected_drifts=detected_drifts,
            true_drifts=true_drifts,
        )

        return {
            "summary": summary,
            "rolling_accuracies": rolling_accuracies,
            "y_preds": y_preds,
            "y_trues": y_trues,
            "detected_drifts": detected_drifts,
            "warmup_samples": self.warmup_samples,
            "model_name": model.name,
        }
