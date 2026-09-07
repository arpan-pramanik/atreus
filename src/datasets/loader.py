"""Real-world streaming datasets loader with actual benchmark streams."""

import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd


def load_electricity_stream(
    data_dir: str = "data",
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load the authentic New South Wales Electricity Market dataset (45,312 instances).
    Target: price change UP (1) or DOWN (0).
    Features: date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "electricity_real.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        from sklearn.datasets import fetch_openml
        print("Fetching authentic NSW Electricity dataset from OpenML...")
        elec = fetch_openml("electricity", version=1, as_frame=True, parser="auto")
        df = elec.frame
        df.to_csv(csv_path, index=False)

    target_col = "class" if "class" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    # Preprocess features
    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        if X_df[col].dtype == object or str(X_df[col].dtype).startswith("category"):
            X_df[col] = pd.factorize(X_df[col])[0]
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0.0)

    X = X_df.values.astype(np.float32)
    y_raw = df[target_col].astype(str).str.upper().str.strip().values
    y = np.where(y_raw == "UP", 1, np.where(y_raw == "1", 1, 0)).astype(int)

    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    return X, y, "NSW Electricity Market (Real)"


def load_covertype_stream(
    data_dir: str = "data",
    max_samples: Optional[int] = 50000,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load the authentic Forest Covertype streaming benchmark (up to 581,012 instances).
    Target: 1 (Spruce-Fir / Type 1) vs 0 (All other forest types).
    Features: 54 cartographic and soil attributes.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "covertype_real.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        from sklearn.datasets import fetch_openml
        print("Fetching authentic Covertype dataset from OpenML...")
        cov = fetch_openml("covertype", version=3, as_frame=True, parser="auto")
        df = cov.frame
        if max_samples is not None and len(df) > max_samples:
            df = df.iloc[:max_samples]
        df.to_csv(csv_path, index=False)

    target_col = "class" if "class" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0.0)

    X = X_df.values.astype(np.float32)
    y_raw = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values
    # Binary classification: Type 1 (Spruce-Fir) vs Other
    y = (y_raw == 1).astype(int)

    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    return X, y, "Forest Covertype Stream (Real)"


def load_phishing_stream(
    data_dir: str = "data",
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load the authentic Phishing Websites dataset (11,055 instances).
    Target: 1 (phishing) vs 0 (legitimate).
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "phishing_real.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        from sklearn.datasets import fetch_openml
        print("Fetching authentic Phishing dataset from OpenML...")
        phish = fetch_openml("PhishingWebsites", version=1, as_frame=True, parser="auto")
        df = phish.frame
        df.to_csv(csv_path, index=False)

    target_col = "Result" if "Result" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0.0)

    X = X_df.values.astype(np.float32)
    y_raw = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values
    y = (y_raw > 0).astype(int)

    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    return X, y, "Phishing Websites (Real)"


def load_airlines_stream(
    data_dir: str = "data",
    max_samples: Optional[int] = 50000,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load the authentic Airlines Flight Delay dataset (up to 539,383 instances).
    Target: Delay > 15 mins (1 vs 0).
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "airlines_real.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        from sklearn.datasets import fetch_openml
        print("Fetching authentic Airlines dataset from OpenML...")
        air = fetch_openml("airlines", version=1, as_frame=True, parser="auto")
        df = air.frame
        if max_samples is not None and len(df) > max_samples:
            df = df.iloc[:max_samples]
        df.to_csv(csv_path, index=False)

    target_col = "Delay" if "Delay" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        if X_df[col].dtype == object or str(X_df[col].dtype).startswith("category"):
            X_df[col] = pd.factorize(X_df[col])[0]
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0.0)

    X = X_df.values.astype(np.float32)
    y_raw = df[target_col].astype(str).str.strip().values
    y = np.where((y_raw == "1") | (y_raw == "True") | (y_raw == "Y"), 1, 0).astype(int)

    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    return X, y, "Airlines Flight Delay (Real)"
