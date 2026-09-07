"""Synthetic Streaming Datasets with Controlled Concept Drift."""

from typing import Tuple, List, Optional
import numpy as np


def generate_sea_stream(
    n_samples: int = 10000,
    noise_ratio: float = 0.1,
    drift_points: Optional[List[int]] = None,
    concepts: Optional[List[float]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate the classic SEA Concepts streaming benchmark (Street & Kim, 2001).
    
    Features: 3 continuous features x1, x2, x3 in [0, 10].
    Classification rule: (x1 + x2) <= threshold -> 1 else 0.
    Feature x3 is an irrelevant noise feature.
    
    Args:
        n_samples: Total number of streaming samples.
        noise_ratio: Fraction of labels randomly perturbed.
        drift_points: Sample indices where concept changes.
        concepts: List of thresholds for each concept block.
        seed: Random seed for reproducibility.
        
    Returns:
        X: (n_samples, 3) features.
        y: (n_samples,) binary labels (0 or 1).
        drift_points: List of indices where drift occurred.
    """
    rng = np.random.RandomState(seed)
    
    if drift_points is None:
        drift_points = [2500, 5000, 7500]
        
    n_segments = len(drift_points) + 1
    if concepts is None:
        # Default pool of concept thresholds
        pool = [8.0, 9.0, 7.0, 9.5, 6.5, 8.5]
        concepts = [pool[i % len(pool)] for i in range(n_segments)]

    assert len(concepts) == n_segments, f"Number of concepts ({len(concepts)}) must match segments ({n_segments})"

    X = rng.uniform(0.0, 10.0, size=(n_samples, 3))
    y = np.zeros(n_samples, dtype=int)

    # Assign boundaries
    boundaries = [0] + drift_points + [n_samples]
    for i in range(len(concepts)):
        start, end = boundaries[i], boundaries[i + 1]
        threshold = concepts[i]
        # Rule: x1 + x2 <= threshold
        y[start:end] = (X[start:end, 0] + X[start:end, 1] <= threshold).astype(int)

    # Add label noise
    if noise_ratio > 0.0:
        noise_mask = rng.rand(n_samples) < noise_ratio
        y[noise_mask] = 1 - y[noise_mask]

    return X, y, drift_points


def generate_agrawal_stream(
    n_samples: int = 10000,
    noise_ratio: float = 0.05,
    drift_points: Optional[List[int]] = None,
    function_ids: Optional[List[int]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate the Agrawal Stream Generator benchmark (Agrawal et al., 1993).
    9 attributes:
    0: salary (20000 to 150000)
    1: commission (0 to 75000)
    2: age (20 to 80)
    3: elevel (0 to 4)
    4: car (1 to 20)
    5: zipcode (0 to 8)
    6: hvalue (50000 to 150000)
    7: hyears (1 to 30)
    8: loan (0 to 500000)
    """
    rng = np.random.RandomState(seed)
    
    if drift_points is None:
        drift_points = [3000, 6000]
        
    n_segments = len(drift_points) + 1
    if function_ids is None:
        pool = [1, 2, 3, 4]
        function_ids = [pool[i % len(pool)] for i in range(n_segments)]

    salary = rng.uniform(20000, 150000, n_samples)
    commission = rng.uniform(0, 75000, n_samples)
    age = rng.uniform(20, 80, n_samples)
    elevel = rng.randint(0, 5, n_samples)
    car = rng.randint(1, 21, n_samples)
    zipcode = rng.randint(0, 9, n_samples)
    hvalue = rng.uniform(50000, 150000, n_samples)
    hyears = rng.uniform(1, 30, n_samples)
    loan = rng.uniform(0, 500000, n_samples)

    X = np.column_stack([salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan])
    y = np.zeros(n_samples, dtype=int)

    boundaries = [0] + drift_points + [n_samples]
    for i, fn_id in enumerate(function_ids):
        start, end = boundaries[i], boundaries[i + 1]
        s = salary[start:end]
        c = commission[start:end]
        a = age[start:end]
        e = elevel[start:end]
        l = loan[start:end]

        if fn_id == 1:
            cond = (a < 40) | (a >= 60)
        elif fn_id == 2:
            cond = (a < 40) & (s >= 50000) & (s <= 100000)
        elif fn_id == 3:
            cond = ((a < 40) & (e >= 3)) | ((a >= 40) & (a < 60) & (s >= 75000) & (s <= 125000))
        else:
            cond = (s + c > 100000)

        y[start:end] = cond.astype(int)

    if noise_ratio > 0.0:
        noise_mask = rng.rand(n_samples) < noise_ratio
        y[noise_mask] = 1 - y[noise_mask]

    return X, y, drift_points


def generate_rotating_hyperplane_stream(
    n_samples: int = 10000,
    n_features: int = 10,
    n_drift_features: int = 4,
    drift_speed: float = 0.002,
    noise_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate continuous rotating hyperplane stream with gradual/continuous concept drift.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_features))
    weights = rng.uniform(-1.0, 1.0, size=n_features)
    weights /= np.linalg.norm(weights)

    y = np.zeros(n_samples, dtype=int)
    drift_points = []
    
    for t in range(n_samples):
        for d in range(n_drift_features):
            weights[d] += drift_speed * (1.0 if t % 2000 < 1000 else -1.0)
        weights /= np.linalg.norm(weights)

        score = np.dot(X[t], weights)
        y[t] = 1 if score > 0 else 0
        
        if (t + 1) % 2500 == 0 and t + 1 < n_samples:
            drift_points.append(t + 1)

    if noise_ratio > 0.0:
        noise_mask = rng.rand(n_samples) < noise_ratio
        y[noise_mask] = 1 - y[noise_mask]

    return X, y, drift_points
