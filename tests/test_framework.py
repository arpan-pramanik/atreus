"""Integration tests and regression suites for the drift-adaptive ML framework."""

import numpy as np
from src.datasets.synthetic import generate_sea_stream, generate_agrawal_stream
from src.detectors.adwin import ADWINDetector
from src.detectors.eddm import EDDMDetector
from src.detectors.ddm import DDMDetector
from src.models.static_model import StaticStreamingModel
from src.models.full_retrain import FullRetrainingModel
from src.models.selective_adaptive import SelectiveAdaptiveEnsemble
from src.models.boosted_adaptive_forest import BoostedAdaptiveForest
from src.models.self_healing_meta_ensemble import SelfHealingMetaEnsemble
from src.models.framework_pipeline import SelfHealingConceptDriftFramework
from src.models.adaptive_svm import AdaptiveOnlineSVM
from src.evaluation.evaluator import StreamEvaluator
from app import build_detector


def test_framework_pipeline():
    """Verify end-to-end prequential evaluation across all model architectures."""
    X, y, drift_points = generate_sea_stream(n_samples=1500, drift_points=[750], seed=42)
    evaluator = StreamEvaluator(warmup_samples=200, rolling_window=50)

    # 1. Static
    static_model = StaticStreamingModel()
    res_static = evaluator.evaluate(static_model, X, y, true_drifts=drift_points)
    assert res_static["summary"]["accuracy"] >= 0.65

    # 2. Full Retrain
    full_model = FullRetrainingModel(drift_detector=ADWINDetector(delta=0.01))
    res_full = evaluator.evaluate(full_model, X, y, true_drifts=drift_points)
    assert res_full["summary"]["accuracy"] >= 0.65

    # 3. Selective Adaptive
    adaptive_model = SelectiveAdaptiveEnsemble(
        n_estimators=10,
        replacement_ratio=0.3,
        drift_detector=ADWINDetector(delta=0.01),
    )
    res_adapt = evaluator.evaluate(adaptive_model, X, y, true_drifts=drift_points)
    assert res_adapt["summary"]["accuracy"] >= 0.65
    assert res_adapt["summary"]["adaptation_time_sec"] >= 0.0

    # 4. Boosted Adaptive Forest (ARF-DWM)
    boosted_model = BoostedAdaptiveForest(
        n_estimators=10,
        detector_type="ddm",
    )
    res_boosted = evaluator.evaluate(boosted_model, X, y, true_drifts=drift_points)
    assert res_boosted["summary"]["accuracy"] >= 0.65
    assert res_boosted["summary"]["adaptation_time_sec"] >= 0.0

    # 5. Advanced Self-Healing Meta-Ensemble
    meta_model = SelfHealingMetaEnsemble(
        n_tree_components=10,
        enable_feature_selection=True,
    )
    res_meta = evaluator.evaluate(meta_model, X, y, true_drifts=drift_points)
    assert res_meta["summary"]["accuracy"] >= 0.65
    assert res_meta["summary"]["adaptation_time_sec"] >= 0.0

    # 6. Adaptive Online SVM
    svm_model = AdaptiveOnlineSVM(
        drift_detector=ADWINDetector(delta=0.01),
        buffer_size=150,
    )
    res_svm = evaluator.evaluate(svm_model, X, y, true_drifts=drift_points)
    assert res_svm["summary"]["accuracy"] >= 0.60
    assert res_svm["summary"]["adaptation_time_sec"] >= 0.0


def test_regression_bug1_feature_misalignment_after_drift():
    """
    Regression Test for Bug 1:
    When drift occurs and online feature selection selects a different subset,
    all trees must remain synchronized with the new mapping rather than receiving shifted features.
    """
    X, y, _ = generate_agrawal_stream(n_samples=2500, seed=42)
    framework = SelfHealingConceptDriftFramework(
        n_trees=10,
        replacement_ratio=0.3,
        drift_detector=ADWINDetector(delta=0.005),
    )
    framework.fit_initial(X[:300], y[:300])

    initial_features = np.copy(framework.selected_feature_indices)

    # Force artificial feature reallocation buffer to trigger adaptation
    for idx in range(300, 1000):
        framework.update_sample(X[idx], y[idx], sample_idx=idx)

    # All trees must successfully predict on new samples without shape error or value error
    test_x = X[1005]
    proba = framework.predict_proba_one(test_x)
    assert len(proba) == 2
    assert np.isclose(np.sum(proba), 1.0)
    pred = framework.predict_one(test_x)
    assert pred in (0, 1)


def test_regression_bug5_eddm_routing():
    """
    Regression Test for Bug 5:
    'build_detector("eddm")' must return an EDDMDetector instance,
    and must not prematurely match 'ddm'.
    """
    eddm_det = build_detector("eddm")
    assert isinstance(eddm_det, EDDMDetector), f"Expected EDDMDetector, got {type(eddm_det)}"
    assert eddm_det.name == "EDDM"

    ddm_det = build_detector("ddm")
    assert isinstance(ddm_det, DDMDetector), f"Expected DDMDetector, got {type(ddm_det)}"
    assert ddm_det.name == "DDM"

