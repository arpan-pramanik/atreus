"""Selective Adaptive Ensemble Model.

Core proposed innovation: Selectively adapts only the affected components (sub-estimators / trees)
upon detected concept drift, preserving stable knowledge while cutting computational cost.
"""

import time
from typing import Optional, List, Tuple
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from src.detectors.base import BaseDriftDetector
from src.detectors.adwin import ADWINDetector


class SelectiveAdaptiveEnsemble:
    """
    Selective Adaptive Ensemble Framework for Streaming Data with Concept Drift.
    
    Key Features:
    1. Modular ensemble of diverse base learners (e.g., decision trees).
    2. Component-level health and error tracking.
    3. Selective replacement: when drift is detected, only the worst-performing k components
       are retrained/replaced on the recent window, retaining stable historical knowledge.
    4. Dynamic performance-based weighting (exponential decay on component error).
    """

    def __init__(
        self,
        n_estimators: int = 30,
        replacement_ratio: float = 0.3,
        drift_detector: Optional[BaseDriftDetector] = None,
        buffer_size: int = 500,
        max_depth: Optional[int] = 8,
        random_state: int = 42,
        name: str = "Selective Adaptive",
    ):
        self.n_estimators = n_estimators
        self.replacement_ratio = replacement_ratio
        self.buffer_size = buffer_size
        self.max_depth = max_depth
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        self.drift_detector = (
            drift_detector if drift_detector is not None else ADWINDetector()
        )
        self.name = f"{name} ({self.drift_detector.name})"

        # Ensemble state
        self.estimators: List[DecisionTreeClassifier] = []
        self.estimator_errors: np.ndarray = np.zeros(n_estimators)
        self.estimator_weights: np.ndarray = np.ones(n_estimators) / n_estimators
        self.estimator_ages: np.ndarray = np.zeros(n_estimators, dtype=int)
        
        # Buffers
        self.buffer_X: List[np.ndarray] = []
        self.buffer_y: List[int] = []
        self.warning_buffer_X: List[np.ndarray] = []
        self.warning_buffer_y: List[int] = []
        self.in_warning: bool = False

        # Profiling & statistics
        self.is_fitted: bool = False
        self.adaptation_count: int = 0
        self.components_updated_count: int = 0
        self.total_adaptation_time: float = 0.0
        self.drift_events: List[int] = []
        self.warning_events: List[int] = []

    def fit_initial(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initial bootstrap training of the ensemble."""
        n_samples = len(X)
        self.estimators = []
        
        for i in range(self.n_estimators):
            # Bootstrap sample for diversity
            boot_idx = self.rng.choice(n_samples, size=n_samples, replace=True)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.rng.randint(0, 100000),
                max_features="sqrt",
            )
            tree.fit(X[boot_idx], y[boot_idx])
            self.estimators.append(tree)

        self.estimator_errors = np.zeros(self.n_estimators)
        self.estimator_weights = np.ones(self.n_estimators) / self.n_estimators
        self.estimator_ages = np.zeros(self.n_estimators, dtype=int)
        self.is_fitted = True

        for xi, yi in zip(X[-self.buffer_size :], y[-self.buffer_size :]):
            self.buffer_X.append(xi)
            self.buffer_y.append(yi)

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        """Weighted probability prediction from active ensemble."""
        if not self.is_fitted or len(self.estimators) == 0:
            return np.array([0.5, 0.5])
        
        x_2d = x.reshape(1, -1)
        probas = np.zeros(2)
        total_weight = 0.0

        for est, w in zip(self.estimators, self.estimator_weights):
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(x_2d)[0]
                if len(p) == 1:
                    # Single class tree
                    c = est.classes_[0]
                    p = np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
                probas += w * p
                total_weight += w

        if total_weight > 0:
            probas /= total_weight
        else:
            probas = np.array([0.5, 0.5])
        return probas

    def predict_one(self, x: np.ndarray) -> int:
        """Predict binary class for single instance."""
        proba = self.predict_proba_one(x)
        return int(np.argmax(proba))

    def _evaluate_components_on_buffer(self, X_eval: np.ndarray, y_eval: np.ndarray) -> np.ndarray:
        """Compute individual error rate for each component on recent buffer."""
        errors = np.zeros(self.n_estimators)
        n_eval = len(y_eval)
        if n_eval == 0:
            return errors

        for i, est in enumerate(self.estimators):
            try:
                preds = est.predict(X_eval)
                errors[i] = np.mean(preds != y_eval)
            except Exception:
                errors[i] = 1.0

        return errors

    def _adapt_selective(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Selective component adaptation:
        1. Identify the worst performing subset of estimators.
        2. Replace only those with new trees trained on recent concept data.
        3. Recalibrate ensemble weights.
        """
        t0 = time.perf_counter()
        k_replace = max(1, int(np.ceil(self.n_estimators * self.replacement_ratio)))
        
        # Evaluate component errors on recent window
        comp_errors = self._evaluate_components_on_buffer(X_train, y_train)
        
        # Identify worst components (highest error)
        worst_indices = np.argsort(comp_errors)[-k_replace:]
        
        n_samples = len(X_train)
        for idx in worst_indices:
            boot_idx = self.rng.choice(n_samples, size=n_samples, replace=True)
            new_tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.rng.randint(0, 100000),
                max_features="sqrt",
            )
            new_tree.fit(X_train[boot_idx], y_train[boot_idx])
            self.estimators[idx] = new_tree
            self.estimator_ages[idx] = 0

        # Recalculate weights for the entire ensemble
        updated_errors = self._evaluate_components_on_buffer(X_train, y_train)
        self.estimator_errors = updated_errors
        # Softmax / exponential weighting based on accuracy (1 - error)
        accuracies = np.clip(1.0 - updated_errors, 1e-4, 1.0)
        exp_weights = np.exp(3.0 * (accuracies - np.mean(accuracies)))
        self.estimator_weights = exp_weights / np.sum(exp_weights)

        self.adaptation_count += 1
        self.components_updated_count += k_replace
        t1 = time.perf_counter()
        self.total_adaptation_time += (t1 - t0)

    def update_sample(self, x: np.ndarray, y: int, sample_idx: int = 0) -> bool:
        """
        Process new sample, update online detector, and selectively adapt on drift.
        
        Returns:
            bool: True if adaptation was triggered.
        """
        # Step 1: Compute prediction and ensemble error
        y_pred = self.predict_one(x)
        error = 1 if y_pred != y else 0

        # Step 2: Buffer management
        self.buffer_X.append(x)
        self.buffer_y.append(y)
        if len(self.buffer_X) > self.buffer_size:
            self.buffer_X.pop(0)
            self.buffer_y.pop(0)

        # Step 3: Drift detector update
        drift_detected, warning_detected = self.drift_detector.update(error)

        # Age increment
        self.estimator_ages += 1

        if warning_detected:
            self.in_warning = True
            self.warning_events.append(sample_idx)
            self.warning_buffer_X.append(x)
            self.warning_buffer_y.append(y)
            return False

        if drift_detected:
            self.drift_events.append(sample_idx)
            # Use warning buffer if populated, else recent sliding buffer
            if len(self.warning_buffer_X) >= 30:
                X_adapt = np.array(self.warning_buffer_X)
                y_adapt = np.array(self.warning_buffer_y)
            else:
                X_adapt = np.array(self.buffer_X)
                y_adapt = np.array(self.buffer_y)

            # Clear warning state
            self.warning_buffer_X = []
            self.warning_buffer_y = []
            self.in_warning = False

            if len(np.unique(y_adapt)) > 1:
                self._adapt_selective(X_adapt, y_adapt)
                return True

        return False
