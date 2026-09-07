"""Hardware-Accelerated Adaptive XGBoost Model with NVIDIA GPU & Multicore Support."""

import time
from typing import Optional, List
import numpy as np
import xgboost as xgb
from src.detectors.base import BaseDriftDetector
from src.detectors.adwin import ADWINDetector


class GPUAcceleratedAdaptiveXGBoost:
    """
    Adaptive XGBoost streaming model leveraging NVIDIA CUDA GPU (e.g. RTX 5070)
    or multicore parallel CPU (e.g. AMD Ryzen 9 9955HX 32 threads).
    """

    def __init__(
        self,
        n_estimators: int = 40,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        drift_detector: Optional[BaseDriftDetector] = None,
        buffer_size: int = 600,
        device: str = "cuda",
        n_jobs: int = 32,
        name: str = "Adaptive XGBoost (GPU)",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.drift_detector = drift_detector if drift_detector is not None else ADWINDetector()
        self.buffer_size = buffer_size
        self.device = device
        self.n_jobs = n_jobs
        self.name = f"{name} ({self.drift_detector.name})"

        self.model: Optional[xgb.XGBClassifier] = None
        self.buffer_X: List[np.ndarray] = []
        self.buffer_y: List[int] = []
        self.is_fitted: bool = False
        self.adaptation_count: int = 0
        self.total_adaptation_time: float = 0.0
        self.drift_events: List[int] = []

    def _build_model(self) -> xgb.XGBClassifier:
        """Instantiate XGBoost classifier with hardware acceleration."""
        try:
            return xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                tree_method="hist",
                device=self.device,
                random_state=42,
                eval_metric="logloss",
            )
        except Exception:
            # Fallback to high-performance multicore CPU
            return xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                tree_method="hist",
                n_jobs=self.n_jobs,
                random_state=42,
                eval_metric="logloss",
            )

    def fit_initial(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initial training."""
        self.model = self._build_model()
        self.model.fit(X, y)
        self.is_fitted = True
        for xi, yi in zip(X[-self.buffer_size :], y[-self.buffer_size :]):
            self.buffer_X.append(xi)
            self.buffer_y.append(yi)

    def predict_one(self, x: np.ndarray) -> int:
        """Predict single instance."""
        if not self.is_fitted or self.model is None:
            return 0
        x_2d = x.reshape(1, -1).astype(np.float32)
        return int(self.model.predict(x_2d)[0])

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        """Predict class probability."""
        if not self.is_fitted or self.model is None:
            return np.array([0.5, 0.5])
        x_2d = x.reshape(1, -1).astype(np.float32)
        return self.model.predict_proba(x_2d)[0]

    def update_sample(self, x: np.ndarray, y: int, sample_idx: int = 0) -> bool:
        """Update stream buffer, check drift, adapt model."""
        y_pred = self.predict_one(x)
        error = 1 if y_pred != y else 0

        self.buffer_X.append(x)
        self.buffer_y.append(y)
        if len(self.buffer_X) > self.buffer_size:
            self.buffer_X.pop(0)
            self.buffer_y.pop(0)

        drift_detected, _ = self.drift_detector.update(error)

        if drift_detected:
            self.drift_events.append(sample_idx)
            t0 = time.perf_counter()
            X_train = np.array(self.buffer_X, dtype=np.float32)
            y_train = np.array(self.buffer_y, dtype=int)
            if len(np.unique(y_train)) > 1:
                self.model = self._build_model()
                self.model.fit(X_train, y_train)
                self.adaptation_count += 1
            t1 = time.perf_counter()
            self.total_adaptation_time += (t1 - t0)
            return True

        return False
