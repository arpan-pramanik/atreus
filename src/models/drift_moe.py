"""DriftMoE: Mixture of Experts for Streaming Classification with Concept Drift (ECML PKDD 2025 / 2026).

Paper Reference:
"DriftMoE: Mixture of Experts for Streaming Classification with Concept Drift"
Authors: Miguel Aspis, Sebastián A. Cajas Ordóñez, Andrés L. Suárez-Cetrulo, Ricardo Simón Carbajo
Proceedings of the European Conference on Machine Learning and Principles and Practice
of Knowledge Discovery in Databases (ECML PKDD 2025 / 2026)

Core Innovations Implemented:
1. Dynamic Gating Router: Online Softmax Router G(x) selecting Top-K experts per instance.
2. Heterogeneous Expert Pool with Native NVIDIA CUDA Acceleration:
   - Expert 0: GPU-Accelerated Gradient Boosted Decision Trees (XGBoost CUDA:0)
   - Expert 1..K-2: Fast Subspace Random Decision Trees (Bootstrap Resampling)
   - Expert K-1: Incremental Online Margin Classifier (SGD Log-Loss)
3. Multi-Hot Correctness Masking: Reinforces accurate experts and updates router weights online.
4. Drift-Triggered Micro-Adaptation: Component drift detectors monitor individual experts;
   only degraded experts undergo selective replacement, preventing catastrophic forgetting of recurring concepts.
"""

import time
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from src.detectors.base import BaseDriftDetector
from src.detectors.adwin import ADWINDetector
from src.detectors.ddm import DDMDetector
from src.models.framework_pipeline import OnlineStandardScaler


class StreamingMoERouter:
    """Online Softmax Router with Multi-Hot Correctness Feedback."""

    def __init__(self, n_features: int, n_experts: int, lr: float = 0.05, top_k: int = 2):
        self.n_features = n_features
        self.n_experts = n_experts
        self.lr = lr
        self.top_k = min(top_k, n_experts)
        self.weights = np.zeros((n_experts, n_features), dtype=np.float32)
        self.bias = np.zeros(n_experts, dtype=np.float32)

    def route(self, x: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """Compute Softmax routing distribution and Top-K expert indices."""
        logits = np.dot(self.weights, x) + self.bias
        # Numerical stability
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-9)

        # Select Top-K experts
        top_k_indices = np.argsort(probs)[-self.top_k:][::-1]
        active_weights = np.zeros_like(probs)
        active_weights[top_k_indices] = probs[top_k_indices]
        sum_active = np.sum(active_weights)
        if sum_active > 0:
            active_weights /= sum_active
        else:
            active_weights[top_k_indices] = 1.0 / self.top_k

        return active_weights, list(top_k_indices)

    def update_feedback(self, x: np.ndarray, correctness_mask: np.ndarray) -> None:
        """
        Symbiotic co-training update:
        Multi-hot gradient step reinforcing experts that yielded accurate predictions.
        """
        logits = np.dot(self.weights, x) + self.bias
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-9)

        # Multi-hot target normalized
        pos_sum = np.sum(correctness_mask)
        if pos_sum > 0:
            target = correctness_mask / pos_sum
            grad = probs - target
            for i in range(self.n_experts):
                self.weights[i] -= self.lr * grad[i] * x
                self.bias[i] -= self.lr * grad[i]


class DriftMoEClassifier:
    """
    Modern DriftMoE (2025/2026) Streaming Classifier with Native CUDA Acceleration.
    """

    def __init__(
        self,
        n_experts: int = 6,
        top_k: int = 2,
        buffer_size: int = 400,
        cuda_device: str = "cuda:0",
        random_state: int = 42,
        name: str = "DriftMoE (2026 Streaming MoE + CUDA)",
    ):
        self.n_experts = max(3, n_experts)
        self.top_k = top_k
        self.buffer_size = buffer_size
        self.cuda_device = cuda_device
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.name = name

        # Streaming router and scaler
        self.router: Optional[StreamingMoERouter] = None
        self.scaler: Optional[OnlineStandardScaler] = None

        # Expert pool components
        self.xgb_expert: Optional[xgb.XGBClassifier] = None
        self.tree_experts: List[DecisionTreeClassifier] = []
        self.sgd_expert: Optional[SGDClassifier] = None

        # Component-level micro drift detectors (ADWIN / DDM per expert)
        self.expert_detectors: List[BaseDriftDetector] = []

        # Buffer & health telemetry
        self.buffer_X: List[np.ndarray] = []
        self.buffer_y: List[int] = []
        self.is_fitted: bool = False
        self.adaptation_count: int = 0
        self.last_components_updated: int = 0
        self.total_adaptation_time: float = 0.0
        self.drift_events: List[int] = []
        self.cuda_active: bool = False

    def _init_xgb_expert(self) -> xgb.XGBClassifier:
        """Create CUDA-accelerated XGBoost expert."""
        try:
            clf = xgb.XGBClassifier(
                n_estimators=25,
                max_depth=5,
                learning_rate=0.1,
                tree_method="hist",
                device=self.cuda_device,
                random_state=self.random_state,
                eval_metric="logloss",
            )
            self.cuda_active = True
            return clf
        except Exception:
            self.cuda_active = False
            return xgb.XGBClassifier(
                n_estimators=25,
                max_depth=5,
                learning_rate=0.1,
                tree_method="hist",
                n_jobs=4,
                random_state=self.random_state,
                eval_metric="logloss",
            )

    def fit_initial(self, X: np.ndarray, y: np.ndarray) -> None:
        """Warmup initialization of DriftMoE router and expert pool."""
        n_features = X.shape[1]
        self.scaler = OnlineStandardScaler(n_features)
        for xi in X:
            self.scaler.update(xi)
        X_scaled = np.array([self.scaler.transform(xi) for xi in X], dtype=np.float32)

        self.router = StreamingMoERouter(
            n_features=n_features,
            n_experts=self.n_experts,
            lr=0.05,
            top_k=self.top_k,
        )

        # 1. Expert 0: GPU XGBoost
        self.xgb_expert = self._init_xgb_expert()
        self.xgb_expert.fit(X_scaled, y)

        # 2. Experts 1..n-2: Subspace Random Trees
        self.tree_experts = []
        n_samples = len(X)
        for i in range(self.n_experts - 2):
            boot_idx = self.rng.choice(n_samples, size=n_samples, replace=True)
            tree = DecisionTreeClassifier(
                max_depth=10,
                max_features="sqrt",
                random_state=self.rng.randint(0, 100000),
            )
            tree.fit(X_scaled[boot_idx], y[boot_idx])
            self.tree_experts.append(tree)

        # 3. Expert n-1: Online Linear SGD Margin
        self.sgd_expert = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            random_state=self.random_state,
        )
        self.sgd_expert.fit(X_scaled, y)

        # 4. Attach micro drift detectors
        self.expert_detectors = [
            ADWINDetector(delta=0.005) for _ in range(self.n_experts)
        ]

        for xi, yi in zip(X[-self.buffer_size :], y[-self.buffer_size :]):
            self.buffer_X.append(xi)
            self.buffer_y.append(yi)

        self.is_fitted = True

    def _predict_expert(self, expert_idx: int, x_scaled: np.ndarray) -> np.ndarray:
        """Query individual expert probability distribution."""
        x_2d = x_scaled.reshape(1, -1)
        if expert_idx == 0:
            # XGBoost CUDA prediction
            if self.xgb_expert is not None:
                try:
                    booster = self.xgb_expert.get_booster()
                    p1 = float(booster.predict(xgb.DMatrix(x_2d))[0])
                    return np.array([1.0 - p1, p1])
                except Exception:
                    return self.xgb_expert.predict_proba(x_2d)[0]
            return np.array([0.5, 0.5])
        elif expert_idx < self.n_experts - 1:
            tree_idx = expert_idx - 1
            if tree_idx < len(self.tree_experts):
                try:
                    p = self.tree_experts[tree_idx].predict_proba(x_2d)[0]
                    if len(p) == 1:
                        c = self.tree_experts[tree_idx].classes_[0]
                        return np.array([1.0, 0.0]) if c == 0 else np.array([0.0, 1.0])
                    return p
                except Exception:
                    return np.array([0.5, 0.5])
            return np.array([0.5, 0.5])
        else:
            # SGD expert
            if self.sgd_expert is not None:
                try:
                    return self.sgd_expert.predict_proba(x_2d)[0]
                except Exception:
                    return np.array([0.5, 0.5])
            return np.array([0.5, 0.5])

    def predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        """Route input through Top-K experts with Softmax Gating."""
        if not self.is_fitted or self.router is None:
            return np.array([0.5, 0.5])

        x_scaled = self.scaler.transform(x).astype(np.float32) if self.scaler is not None else x
        routing_weights, top_k_indices = self.router.route(x_scaled)

        aggregated_prob = np.zeros(2, dtype=np.float64)
        for idx in top_k_indices:
            w = routing_weights[idx]
            if w > 0:
                p = self._predict_expert(idx, x_scaled)
                aggregated_prob += w * p

        sum_p = np.sum(aggregated_prob)
        if sum_p > 0:
            return aggregated_prob / sum_p
        return np.array([0.5, 0.5])

    def predict_one(self, x: np.ndarray) -> int:
        """Predict binary class label."""
        proba = self.predict_proba_one(x)
        return int(np.argmax(proba))

    def update_sample(self, x: np.ndarray, y: int, sample_idx: int = 0) -> bool:
        """
        Prequential streaming step:
        1. Query predictions from all active experts.
        2. Feed multi-hot correctness mask back to neural router.
        3. Monitor component micro-detectors for localized drift.
        4. Selectively retrain affected experts on drift.
        """
        if self.scaler is not None:
            self.scaler.update(x)
            x_scaled = self.scaler.transform(x).astype(np.float32)
        else:
            x_scaled = x

        self.buffer_X.append(x)
        self.buffer_y.append(y)
        if len(self.buffer_X) > self.buffer_size:
            self.buffer_X.pop(0)
            self.buffer_y.pop(0)

        # SGD incremental step
        if self.sgd_expert is not None:
            try:
                self.sgd_expert.partial_fit(x_scaled.reshape(1, -1), [y], classes=[0, 1])
            except Exception:
                pass

        # Evaluate correctness across all experts
        correctness_mask = np.zeros(self.n_experts, dtype=np.float32)
        drift_signals = []

        for i in range(self.n_experts):
            p = self._predict_expert(i, x_scaled)
            pred = int(np.argmax(p))
            err = 1 if pred != y else 0
            if err == 0:
                correctness_mask[i] = 1.0

            d_drift, _ = self.expert_detectors[i].update(err)
            if d_drift:
                drift_signals.append(i)

        # Symbiotic router feedback
        if self.router is not None:
            self.router.update_feedback(x_scaled, correctness_mask)

        # Selective Drift Adaptation
        if len(drift_signals) > 0:
            t0 = time.perf_counter()
            self.drift_events.append(sample_idx)
            X_buf = np.array(self.buffer_X, dtype=np.float32)
            y_buf = np.array(self.buffer_y, dtype=int)
            X_scaled_buf = np.array([self.scaler.transform(xi) for xi in X_buf], dtype=np.float32) if self.scaler is not None else X_buf

            if len(np.unique(y_buf)) > 1:
                n_b = len(X_buf)
                for exp_idx in drift_signals:
                    if exp_idx == 0:
                        # Retrain GPU XGBoost expert on fresh buffer
                        self.xgb_expert = self._init_xgb_expert()
                        self.xgb_expert.fit(X_scaled_buf, y_buf)
                    elif exp_idx < self.n_experts - 1:
                        # Retrain individual decision tree expert
                        tree_idx = exp_idx - 1
                        boot_idx = self.rng.choice(n_b, size=n_b, replace=True)
                        new_tree = DecisionTreeClassifier(
                            max_depth=10,
                            max_features="sqrt",
                            random_state=self.rng.randint(0, 100000),
                        )
                        new_tree.fit(X_scaled_buf[boot_idx], y_buf[boot_idx])
                        if tree_idx < len(self.tree_experts):
                            self.tree_experts[tree_idx] = new_tree
                    else:
                        # Reset SGD expert with initial partial fit on recent buffer
                        self.sgd_expert = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, random_state=self.random_state)
                        self.sgd_expert.fit(X_scaled_buf, y_buf)

                    self.expert_detectors[exp_idx].reset()

                self.last_components_updated = len(drift_signals)
                self.adaptation_count += len(drift_signals)

            t1 = time.perf_counter()
            self.total_adaptation_time += (t1 - t0)
            return True

        return False
