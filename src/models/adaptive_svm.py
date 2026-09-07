"""Adaptive Online Support Vector Machine (SVM) for Streaming Concept Drift.

Implements online incremental SVM learning with dynamic drift-triggered
margin adaptation and learning rate boosting as specified in plan.md and PPT.
"""

import time
from typing import Optional, Any, List
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from src.detectors.adwin import ADWINDetector
from src.detectors.base import BaseDriftDetector


class AdaptiveOnlineSVM:
    """
    Online Support Vector Machine (SGD margin classifier) with concept drift adaptation.
    Performs continuous prequential learning and triggers accelerated margin recalibration
    upon detected distribution shift.
    """

    def __init__(
        self,
        alpha: float = 1e-4,
        loss: str = "modified_huber",
        drift_detector: Optional[BaseDriftDetector] = None,
        buffer_size: int = 300,
        name: str = "Adaptive Online SVM",
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.loss = loss
        self.drift_detector = drift_detector if drift_detector is not None else ADWINDetector(delta=0.005)
        self.buffer_size = buffer_size
        self.name = name
        self.random_state = random_state

        self.classes_ = np.array([0, 1])
        self.scaler = StandardScaler()
        self.model = SGDClassifier(
            loss=self.loss,
            penalty="l2",
            alpha=self.alpha,
            learning_rate="adaptive",
            eta0=0.05,
            random_state=self.random_state,
        )

        self.buffer_X: List[np.ndarray] = []
        self.buffer_y: List[int] = []
        self.is_fitted = False

        # Telemetry & Benchmark stats
        self.adaptation_count = 0
        self.components_updated = 1
        self.total_adaptation_time = 0.0
        self.drift_events: List[int] = []

    def fit_initial(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit scaler and initialize online SVM on warmup batch."""
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            self.classes_ = np.array([0, 1])

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        for xi, yi in zip(X[-self.buffer_size :], y[-self.buffer_size :]):
            self.buffer_X.append(xi)
            self.buffer_y.append(int(yi))

    def predict_one(self, x: np.ndarray) -> int:
        """Predict class label for a single streaming sample."""
        if not self.is_fitted:
            return 0
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        return int(self.model.predict(x_scaled)[0])

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        """Predict calibrated posterior probabilities."""
        if not self.is_fitted:
            return np.array([0.5, 0.5])
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x_scaled)[0]
        # Distance to margin fallback
        dec = float(self.model.decision_function(x_scaled)[0])
        p1 = 1.0 / (1.0 + np.exp(-np.clip(dec, -15, 15)))
        return np.array([1.0 - p1, p1])

    def update_sample(self, x: np.ndarray, y: int, sample_idx: int = 0) -> bool:
        """
        Evaluate prediction error on detector and incrementally update model.
        Returns True if concept drift was detected and adaptation triggered.
        """
        if not self.is_fitted:
            self.fit_initial(x.reshape(1, -1), np.array([y]))
            return False

        pred = self.predict_one(x)
        err = 1 if pred != y else 0

        # Incremental online learning step
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        self.model.partial_fit(x_scaled, [y], classes=self.classes_)

        # Buffer maintenance
        self.buffer_X.append(x)
        self.buffer_y.append(y)
        if len(self.buffer_X) > self.buffer_size:
            self.buffer_X.pop(0)
            self.buffer_y.pop(0)

        # Feed error to drift detector
        res = self.drift_detector.update(err)
        drift_detected = res[0] if isinstance(res, tuple) else bool(res)

        if drift_detected:
            t0 = time.perf_counter()
            self._adapt_on_drift(sample_idx)
            dur = time.perf_counter() - t0
            self.total_adaptation_time += dur
            self.drift_events.append(sample_idx)
            return True

        return False

    def _adapt_on_drift(self, sample_idx: int) -> None:
        """Recalibrate decision boundary and retrain on sliding buffer."""
        self.adaptation_count += 1
        if len(self.buffer_X) >= 20:
            X_buf = np.array(self.buffer_X)
            y_buf = np.array(self.buffer_y)

            # Recalibrate scaler with recent distribution
            self.scaler.partial_fit(X_buf)
            X_scaled = self.scaler.transform(X_buf)

            # Accelerated partial fit with boosted updates
            for _ in range(3):
                self.model.partial_fit(X_scaled, y_buf, classes=self.classes_)

        self.drift_detector.reset()
