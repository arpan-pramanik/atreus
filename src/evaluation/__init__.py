"""Evaluation package."""

from src.evaluation.metrics import compute_drift_metrics, compute_stream_summary
from src.evaluation.evaluator import StreamEvaluator

__all__ = ["compute_drift_metrics", "compute_stream_summary", "StreamEvaluator"]
