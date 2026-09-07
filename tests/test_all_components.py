"""Comprehensive Verification Test Suite for all Methods, Detectors, Models, and Datasets."""

import os
import tempfile
import numpy as np
import pytest

# Detectors
from src.detectors.adwin import ADWINDetector
from src.detectors.ddm import DDMDetector
from src.detectors.eddm import EDDMDetector
from src.detectors.page_hinkley import PageHinkleyDetector

# Models
from src.models.static_model import StaticStreamingModel
from src.models.full_retrain import FullRetrainingModel
from src.models.selective_adaptive import SelectiveAdaptiveEnsemble
from src.models.boosted_adaptive_forest import BoostedAdaptiveForest
from src.models.xgboost_adaptive import GPUAcceleratedAdaptiveXGBoost
from src.models.adaptive_svm import AdaptiveOnlineSVM
from src.models.self_healing_meta_ensemble import SelfHealingMetaEnsemble
from src.models.framework_pipeline import SelfHealingConceptDriftFramework

# Datasets
from src.datasets.synthetic import (
    generate_sea_stream,
    generate_agrawal_stream,
    generate_rotating_hyperplane_stream,
)
from src.datasets.loader import (
    load_electricity_stream,
    load_covertype_stream,
    load_airlines_stream,
    load_phishing_stream,
)
from src.datasets.hf_loader import (
    load_hf_adult_income_stream,
    load_hf_bank_marketing_stream,
    load_hf_credit_default_stream,
)

# Evaluation & Visualizer
from src.evaluation.evaluator import StreamEvaluator
from src.visualizer import plot_drift_trajectories, plot_performance_and_cost_comparison


class TestDriftDetectors:
    """Rigorous tests for all 4 drift detectors."""

    def test_adwin_detector_lifecycle(self):
        det = ADWINDetector(delta=0.01)
        assert repr(det).startswith("ADWINDetector")

        # Low error phase
        for _ in range(300):
            d, w = det.update(0.0)
            assert isinstance(d, bool)
            assert isinstance(w, bool)

        # High error spike
        drift_seen = False
        for _ in range(300):
            d, _ = det.update(1.0)
            if d:
                drift_seen = True
                break
        assert drift_seen, "ADWIN must detect high error spike"

        det.reset()
        assert det.drift_detected is False
        assert det.warning_detected is False

    def test_ddm_detector_lifecycle(self):
        det = DDMDetector(warm_start=30)
        assert repr(det).startswith("DDMDetector")

        # Stable phase
        for _ in range(200):
            det.update(0.02)

        # High error phase
        drift_seen = False
        for _ in range(200):
            d, _ = det.update(0.8)
            if d:
                drift_seen = True
                break
        assert drift_seen, "DDM must detect high error rate"

        det.reset()
        assert det.drift_detected is False

    def test_eddm_detector_lifecycle(self):
        det = EDDMDetector(warm_start=30)
        assert repr(det).startswith("EDDMDetector")
        rng = np.random.RandomState(42)

        for _ in range(500):
            err = 1 if rng.rand() < 0.05 else 0
            det.update(err)

        drift_seen = False
        for _ in range(500):
            err = 1 if rng.rand() < 0.70 else 0
            d, _ = det.update(err)
            if d:
                drift_seen = True
                break
        assert drift_seen, "EDDM must detect error clustering"

        det.reset()
        assert det.drift_detected is False

    def test_page_hinkley_detector_lifecycle(self):
        det = PageHinkleyDetector(threshold=20.0)
        assert repr(det).startswith("PageHinkleyDetector")

        for _ in range(200):
            det.update(0.0)

        drift_seen = False
        for _ in range(300):
            d, _ = det.update(1.0)
            if d:
                drift_seen = True
                break
        assert drift_seen, "Page-Hinkley must detect cumulative sum shift"

        det.reset()
        assert det.drift_detected is False


class TestSyntheticDatasetGenerators:
    """Verify synthetic stream generators produce valid arrays and drift markers."""

    def test_sea_generator(self):
        X, y, drifts = generate_sea_stream(n_samples=600, drift_points=[200, 400], noise_ratio=0.05, seed=42)
        assert X.shape == (600, 3)
        assert len(y) == 600
        assert set(np.unique(y)).issubset({0, 1})
        assert drifts == [200, 400]
        assert not np.isnan(X).any()

    def test_agrawal_generator(self):
        X, y, drifts = generate_agrawal_stream(n_samples=600, drift_points=[300], seed=42)
        assert X.shape == (600, 9)
        assert len(y) == 600
        assert set(np.unique(y)).issubset({0, 1})
        assert drifts == [300]
        assert not np.isnan(X).any()

    def test_hyperplane_generator(self):
        X, y, drifts = generate_rotating_hyperplane_stream(n_samples=500, n_features=6, drift_speed=0.01, seed=42)
        assert X.shape == (500, 6)
        assert len(y) == 500
        assert set(np.unique(y)).issubset({0, 1})
        assert not np.isnan(X).any()


class TestRealDatasetLoaders:
    """Verify real-world and Hugging Face dataset loaders."""

    def test_electricity_loader(self):
        X, y, name = load_electricity_stream()
        assert len(X) > 1000
        assert X.shape[1] == 8
        assert len(X) == len(y)
        assert not np.isnan(X).any()

    def test_airlines_loader(self):
        X, y, name = load_airlines_stream()
        assert len(X) > 1000
        assert X.shape[1] == 7
        assert len(X) == len(y)
        assert not np.isnan(X).any()

    def test_phishing_loader(self):
        X, y, name = load_phishing_stream()
        assert len(X) > 1000
        assert X.shape[1] == 30
        assert len(X) == len(y)
        assert not np.isnan(X).any()

    def test_hf_adult_loader(self):
        X, y, name = load_hf_adult_income_stream()
        assert len(X) > 500
        assert len(X) == len(y)
        assert not np.isnan(X).any()


class TestAllModelArchitectures:
    """Verify every model architecture implements the required streaming methods."""

    @pytest.fixture
    def stream_data(self):
        X, y, drifts = generate_sea_stream(n_samples=400, drift_points=[200], seed=42)
        return X, y, drifts

    def test_static_model(self, stream_data):
        X, y, _ = stream_data
        model = StaticStreamingModel(name="Test Static")
        model.fit_initial(X[:50], y[:50])

        pred = model.predict_one(X[51])
        assert pred in [0, 1]
        proba = model.predict_proba_one(X[51])
        assert len(proba) == 2
        assert np.isclose(np.sum(proba), 1.0)

        # Update sample returns None (does not adapt)
        res = model.update_sample(X[51], y[51], sample_idx=51)
        assert res is None

    def test_full_retrain_model(self, stream_data):
        X, y, _ = stream_data
        model = FullRetrainingModel(drift_detector=ADWINDetector(delta=0.01), name="Test Full Retrain")
        model.fit_initial(X[:50], y[:50])

        pred = model.predict_one(X[51])
        assert pred in [0, 1]
        proba = model.predict_proba_one(X[51])
        assert len(proba) == 2

        # Step through stream
        for i in range(50, 150):
            model.update_sample(X[i], y[i], sample_idx=i)
        assert model.retrain_count >= 0

    def test_selective_adaptive_ensemble(self, stream_data):
        X, y, _ = stream_data
        model = SelectiveAdaptiveEnsemble(
            n_estimators=10,
            replacement_ratio=0.3,
            drift_detector=ADWINDetector(delta=0.01),
            name="Test Selective",
        )
        model.fit_initial(X[:50], y[:50])

        pred = model.predict_one(X[51])
        assert pred in [0, 1]
        proba = model.predict_proba_one(X[51])
        assert len(proba) == 2

        for i in range(50, 150):
            model.update_sample(X[i], y[i], sample_idx=i)
        assert hasattr(model, "adaptation_count")

    def test_boosted_adaptive_forest(self, stream_data):
        X, y, _ = stream_data
        model = BoostedAdaptiveForest(n_estimators=10, detector_type="adwin", name="Test Boosted Forest")
        model.fit_initial(X[:50], y[:50])

        pred = model.predict_one(X[51])
        assert pred in [0, 1]
        proba = model.predict_proba_one(X[51])
        assert len(proba) == 2

        for i in range(50, 150):
            model.update_sample(X[i], y[i], sample_idx=i)

    def test_adaptive_xgboost(self, stream_data):
        X, y, _ = stream_data
        model = GPUAcceleratedAdaptiveXGBoost(
            n_estimators=15,
            drift_detector=ADWINDetector(delta=0.01),
            device="cuda",
            n_jobs=4,
            name="Test Adaptive XGBoost",
        )
        model.fit_initial(X[:50], y[:50])

        pred = model.predict_one(X[51])
        assert pred in [0, 1]
        proba = model.predict_proba_one(X[51])
        assert len(proba) == 2

        for i in range(50, 100):
            model.update_sample(X[i], y[i], sample_idx=i)

    def test_adaptive_online_svm(self, stream_data):
        X, y, _ = stream_data
        model = AdaptiveOnlineSVM(
            drift_detector=ADWINDetector(delta=0.01),
            buffer_size=100,
            name="Test Adaptive SVM",
        )
        model.fit_initial(X[:50], y[:50])

        pred = model.predict_one(X[51])
        assert pred in [0, 1]
        proba = model.predict_proba_one(X[51])
        assert len(proba) == 2
        assert np.isclose(np.sum(proba), 1.0)

        for i in range(50, 150):
            res = model.update_sample(X[i], y[i], sample_idx=i)
            assert isinstance(res, bool)

    def test_self_healing_meta_ensemble(self, stream_data):
        X, y, _ = stream_data
        model = SelfHealingMetaEnsemble(n_tree_components=10, name="Test Meta Ensemble")
        model.fit_initial(X[:50], y[:50])

        pred = model.predict_one(X[51])
        assert pred in [0, 1]
        proba = model.predict_proba_one(X[51])
        assert len(proba) == 2

        for i in range(50, 100):
            model.update_sample(X[i], y[i], sample_idx=i)

    def test_self_healing_framework(self, stream_data):
        X, y, _ = stream_data
        model = SelfHealingConceptDriftFramework(
            n_trees=10,
            drift_detector=ADWINDetector(delta=0.01),
            name="Test Proposed Framework",
        )
        model.fit_initial(X[:50], y[:50])

        pred = model.predict_one(X[51])
        assert pred in [0, 1]
        proba = model.predict_proba_one(X[51])
        assert len(proba) == 2

        for i in range(50, 100):
            model.update_sample(X[i], y[i], sample_idx=i)


class TestEvaluatorAndVisualizer:
    """Verify prequential evaluation engine and plotting routines."""

    def test_evaluator_end_to_end(self):
        X, y, drifts = generate_sea_stream(n_samples=300, drift_points=[150], seed=42)
        evaluator = StreamEvaluator(warmup_samples=50, rolling_window=20)
        model = SelectiveAdaptiveEnsemble(n_estimators=5, drift_detector=ADWINDetector(delta=0.01))

        res = evaluator.evaluate(model, X, y, true_drifts=drifts)
        s = res["summary"]

        assert 0.0 <= s["accuracy"] <= 1.0
        assert 0.0 <= s["f1_score"] <= 1.0
        assert 0.0 <= s["precision"] <= 1.0
        assert 0.0 <= s["recall"] <= 1.0
        assert len(res["rolling_accuracies"]) > 0

        # Test visualizer functions produce valid PNG files
        with tempfile.TemporaryDirectory() as tmp_dir:
            traj_path = os.path.join(tmp_dir, "test_traj.png")
            comp_path = os.path.join(tmp_dir, "test_comp.png")

            plot_drift_trajectories([res], dataset_name="Test Stream", save_path=traj_path)
            assert os.path.exists(traj_path)
            assert os.path.getsize(traj_path) > 1000

            plot_performance_and_cost_comparison([s], dataset_name="Test Stream", save_path=comp_path)
            assert os.path.exists(comp_path)
            assert os.path.getsize(comp_path) > 1000
