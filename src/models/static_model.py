"""Static ML Model Baseline (No adaptation)."""

from typing import Optional, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone


class StaticStreamingModel:
    """
    Static baseline model trained once on initial batch and never updated.
    Illustrates performance collapse under concept drift.
    """

    def __init__(self, base_estimator: Optional[Any] = None, name: str = "Static Model"):
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else RandomForestClassifier(n_estimators=30, random_state=42)
        )
        self.model = clone(self.base_estimator)
        self.name = name
        self.is_fitted = False
        self.retrain_count = 0
        self.total_retrain_time = 0.0
        self.drift_events = []

    def fit_initial(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initial training."""
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_one(self, x: np.ndarray) -> int:
        """Predict a single streaming instance."""
        if not self.is_fitted:
            return 0
        x_2d = x.reshape(1, -1)
        return int(self.model.predict(x_2d)[0])

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        """Predict probability for a single instance."""
        if not self.is_fitted:
            return np.array([0.5, 0.5])
        x_2d = x.reshape(1, -1)
        return self.model.predict_proba(x_2d)[0]

    def update_sample(self, x: np.ndarray, y: int, sample_idx: int = 0) -> None:
        """Static model ignores new streaming data."""
        pass
