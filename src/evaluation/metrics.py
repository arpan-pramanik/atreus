"""Evaluation metrics for concept drift and streaming ML."""

from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_drift_metrics(
    detected_drifts: List[int],
    true_drifts: List[int],
    grace_window: int = 500,
) -> Dict[str, Any]:
    """
    Compute drift detection quality metrics:
    - Detection Delay: Average delay (in samples) after a ground truth drift until first alarm.
    - True Positives: True drift points successfully caught within grace_window.
    - False Alarms: Detections that occurred outside any grace_window after a true drift.
    """
    if len(true_drifts) == 0:
        return {
            "detection_delay_mean": 0.0,
            "detection_rate": 1.0,
            "false_alarms": len(detected_drifts),
            "total_detected": len(detected_drifts),
        }

    delays = []
    caught_true = set()
    matched_detections = set()

    for td in true_drifts:
        # Find first detection in [td, td + grace_window]
        candidates = [d for d in detected_drifts if td <= d <= td + grace_window]
        if candidates:
            first_alarm = min(candidates)
            delays.append(first_alarm - td)
            caught_true.add(td)
            matched_detections.add(first_alarm)

    detection_rate = len(caught_true) / len(true_drifts)
    mean_delay = float(np.mean(delays)) if delays else float("nan")
    false_alarms = len(detected_drifts) - len(matched_detections)

    return {
        "detection_delay_mean": mean_delay,
        "detection_rate": detection_rate,
        "caught_drifts": len(caught_true),
        "total_true_drifts": len(true_drifts),
        "false_alarms": max(0, false_alarms),
        "total_detected": len(detected_drifts),
    }


def compute_stream_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    adaptation_count: int,
    adaptation_time: float,
    detected_drifts: List[int],
    true_drifts: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compile comprehensive streaming evaluation summary dictionary."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    summary = {
        "model": model_name,
        "accuracy": float(acc),
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "adaptation_count": adaptation_count,
        "adaptation_time_sec": float(adaptation_time),
        "total_drifts_detected": len(detected_drifts),
    }

    if true_drifts is not None:
        drift_metrics = compute_drift_metrics(detected_drifts, true_drifts)
        summary.update(drift_metrics)

    return summary
