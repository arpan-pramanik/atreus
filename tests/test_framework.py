"""Integration tests for the drift-adaptive ML framework."""

import numpy as np
from src.datasets.synthetic import generate_sea_stream
from src.detectors.adwin import ADWINDetector
from src.detectors.ddm import DDMDetector
from src.models.static_model import StaticStreamingModel
from src.models.full_retrain import FullRetrainingModel
from src.models.selective_adaptive import SelectiveAdaptiveEnsemble
from src.models.boosted_adaptive_forest import BoostedAdaptiveForest
from src.models.self_healing_meta_ensemble import SelfHealingMetaEnsemble
from src.evaluation.evaluator import StreamEvaluator


def test_framework_pipeline():
    """Verify end-to-end prequential evaluation across all model architectures."""
    X, y, drift_points = generate_sea_stream(n_samples=1500, drift_points=[750], seed=42)
    evaluator = StreamEvaluator(warmup_samples=200, rolling_window=50)

    # 1. Static
    static_model = StaticStreamingModel()
    res_static = evaluator.evaluate(static_model, X, y, true_drifts=drift_points)
    assert res_static["summary"]["accuracy"] > 0.0

    # 2. Full Retrain
    full_model = FullRetrainingModel(drift_detector=ADWINDetector(delta=0.01))
    res_full = evaluator.evaluate(full_model, X, y, true_drifts=drift_points)
    assert res_full["summary"]["accuracy"] > 0.0

    # 3. Selective Adaptive
    adaptive_model = SelectiveAdaptiveEnsemble(
        n_estimators=10,
        replacement_ratio=0.3,
        drift_detector=ADWINDetector(delta=0.01),
    )
    res_adapt = evaluator.evaluate(adaptive_model, X, y, true_drifts=drift_points)
    assert res_adapt["summary"]["accuracy"] > 0.0
    assert res_adapt["summary"]["adaptation_time_sec"] >= 0.0

    # 4. Boosted Adaptive Forest (ARF-DWM)
    boosted_model = BoostedAdaptiveForest(
        n_estimators=10,
        detector_type="ddm",
    )
    res_boosted = evaluator.evaluate(boosted_model, X, y, true_drifts=drift_points)
    assert res_boosted["summary"]["accuracy"] > 0.0
    assert res_boosted["summary"]["adaptation_time_sec"] >= 0.0

    # 5. Advanced Self-Healing Meta-Ensemble
    meta_model = SelfHealingMetaEnsemble(
        n_tree_components=10,
        enable_feature_selection=True,
    )
    res_meta = evaluator.evaluate(meta_model, X, y, true_drifts=drift_points)
    assert res_meta["summary"]["accuracy"] > 0.0
    assert res_meta["summary"]["adaptation_time_sec"] >= 0.0

    # 6. Adaptive Online SVM
    from src.models.adaptive_svm import AdaptiveOnlineSVM
    svm_model = AdaptiveOnlineSVM(
        drift_detector=ADWINDetector(delta=0.01),
        buffer_size=150,
    )
    res_svm = evaluator.evaluate(svm_model, X, y, true_drifts=drift_points)
    assert res_svm["summary"]["accuracy"] > 0.0
    assert res_svm["summary"]["adaptation_time_sec"] >= 0.0

