"""Hugging Face Datasets Loader for Streaming & Concept Drift Evaluation."""

import os
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from datasets import load_dataset


def load_hf_adult_income_stream(
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load Adult Census Income dataset directly from Hugging Face Hub (`scikit-learn/adult-census-income`).
    32,561 instances predicting income >50K (1) vs <=50K (0).
    """
    ds = load_dataset("scikit-learn/adult-census-income", split="train")
    df = ds.to_pandas()

    target_col = "income"
    feature_cols = [c for c in df.columns if c != target_col]

    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        if X_df[col].dtype == object or str(X_df[col].dtype).startswith("category"):
            X_df[col] = pd.factorize(X_df[col])[0]
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0.0)

    X = X_df.values.astype(np.float32)
    y_raw = df[target_col].astype(str).str.strip().values
    y = np.where((y_raw == ">50K") | (y_raw == ">50K.") | (y_raw == "1"), 1, 0).astype(int)

    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    return X, y, "HuggingFace: Adult Census Income (32k Stream)"


def load_hf_bank_marketing_stream(
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load Bank Marketing dataset from Hugging Face (`inria-soda/tabular-benchmark`, config `clf_num_bank-marketing`).
    10,578 instances.
    """
    ds = load_dataset("inria-soda/tabular-benchmark", "clf_num_bank-marketing", split="train")
    df = ds.to_pandas()

    target_col = "Class" if "Class" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        if X_df[col].dtype == object or str(X_df[col].dtype).startswith("category"):
            X_df[col] = pd.factorize(X_df[col])[0]
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0.0)

    X = X_df.values.astype(np.float32)
    y_raw = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values
    # Binary 0 vs 1
    y = (y_raw > np.median(y_raw)).astype(int) if len(np.unique(y_raw)) > 2 else (y_raw == y_raw.max()).astype(int)

    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    return X, y, "HuggingFace: Bank Marketing Stream (10k Stream)"


def load_hf_credit_default_stream(
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Load Credit Card Default dataset from Hugging Face (`inria-soda/tabular-benchmark`, config `clf_cat_default-of-credit-card-clients`).
    13,272 instances.
    """
    ds = load_dataset("inria-soda/tabular-benchmark", "clf_cat_default-of-credit-card-clients", split="train")
    df = ds.to_pandas()

    target_col = "y" if "y" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]

    X_df = df[feature_cols].copy()
    for col in X_df.columns:
        if X_df[col].dtype == object or str(X_df[col].dtype).startswith("category"):
            X_df[col] = pd.factorize(X_df[col])[0]
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0.0)

    X = X_df.values.astype(np.float32)
    y_raw = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values
    y = (y_raw > 0).astype(int)

    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    return X, y, "HuggingFace: Credit Default Risk (13k Stream)"
