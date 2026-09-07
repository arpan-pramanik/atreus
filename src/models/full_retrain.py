"""Full Retraining Baseline Model."""

import time
from typing import Optional, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from src.detectors.base import BaseDriftDetector
from src.detectors.adwin import ADWINDetector


class FullRetrainingModel:
    """
    Blind Full-Retraining Model.
    Retrains the entire model from scratch on a sliding memory buffer whenever drift is detected.
    """

    def __init__(
        self,
        base_estimator: Optional[Any] = None,
        drift_detector: Optional[BaseDriftDetector] = None,
        buffer_size: int = 500,
        name: str = "Full Retrain",
    ):
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else RandomForestClassifier(n_estimators=30, random_state=42)
        )
        self.model = clone(self.base_estimator)
        self.drift_detector = (
            drift_detector if drift_detector is not None else ADWINDetector()
        )
        self.buffer_size = buffer_size
        self.name = f"{name} ({self.drift_detector.name})"
        
        self.buffer_X = []
        self.buffer_y = []
        self.is_fitted = False
        self.retrain_count = 0
        self.total_retrain_time = 0.0
        self.drift_events = []

    def fit_initial(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initial training on warmup batch."""
        self.model.fit(X, y)
        self.is_fitted = True
        for xi, yi in zip(X[-self.buffer_size :], y[-self.buffer_size :]):
            self.buffer_X.append(xi)
            self.buffer_y.append(yi)

    def predict_one(self, x: np.ndarray) -> int:
        """Predict a single sample."""
        if not self.is_fitted:
            return 0
        return int(self.model.predict(x.reshape(1, -1))[0])

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            return np.array([0.5, 0.5])
        return self.model.predict_proba(x.reshape(1, -1))[0]

    def update_sample(self, x: np.ndarray, y: int, sample_idx: int = 0) -> bool:
        """
        Process incoming true label, update detector, and retrain full model if drift detected.
        
        Returns:
            bool: True if drift was detected and model was retrained.
        """
        # Calculate error
        y_pred = self.predict_one(x)
        error = 1 if y_pred != y else 0

        # Maintain buffer
        self.buffer_X.append(x)
        self.buffer_y.append(y)
        if len(self.buffer_X) > self.buffer_size:
            self.buffer_X.pop(0)
            self.buffer_y.pop(0)

        # Update drift detector
        drift_detected, _ = self.drift_detector.update(error)

        if drift_detected:
            self.drift_events.append(sample_idx)
            # Full retrain from scratch on buffer
            t0 = time.perf_counter()
            X_train = np.array(self.buffer_X)
            y_train = np.array(self.buffer_y)
            if len(np.unique(y_train)) > 1:
                self.model = clone(self.base_estimator)
                self.model.fit(X_train, y_train)
                self.retrain_count += 1
            t1 = time.perf_counter()
            self.total_retrain_time += (t1 - t0)
            return True

        return False
