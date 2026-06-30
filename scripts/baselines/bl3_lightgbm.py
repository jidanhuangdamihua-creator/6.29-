"""BL3: direct multi-step LightGBM baseline."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


def _load_lgbm_regressor():
    try:
        module = importlib.import_module("lightgbm")
    except ImportError as exc:
        raise ImportError(
            "BL3 requires LightGBM. Please install it manually with: pip install lightgbm"
        ) from exc
    return module.LGBMRegressor


def _usable_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    numeric = feature_df.select_dtypes(include=[np.number])
    columns = []
    for column in numeric.columns:
        lowered = str(column).lower()
        if lowered == "sales":
            continue
        if lowered == "customers":
            continue
        if lowered.endswith("_leakage_risk"):
            continue
        series = numeric[column]
        if series.isna().all() or series.nunique(dropna=True) <= 1:
            continue
        columns.append(column)
    return columns


def predict_bl3(feature_df, test_feature_df, random_state=42):
    """Fit a target-only LightGBM regressor and predict each test row."""
    if not isinstance(feature_df, pd.DataFrame):
        raise TypeError("feature_df must be a pandas DataFrame")
    if not isinstance(test_feature_df, pd.DataFrame):
        raise TypeError("test_feature_df must be a pandas DataFrame")
    if feature_df.empty or test_feature_df.empty:
        raise ValueError("feature_df and test_feature_df must not be empty")
    if "sales" not in feature_df.columns:
        raise ValueError("feature_df must contain the sales target column")

    target = pd.to_numeric(feature_df["sales"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("feature_df sales must contain only finite values")

    columns = _usable_feature_columns(feature_df)
    if not columns:
        raise ValueError("no usable BL3 feature columns remain after filtering")
    missing = [column for column in columns if column not in test_feature_df.columns]
    if missing:
        raise ValueError(f"test_feature_df is missing BL3 feature columns: {missing}")

    train_x = feature_df.loc[:, columns]
    test_x = test_feature_df.loc[:, columns]
    regressor_class = _load_lgbm_regressor()
    model = regressor_class(
        random_state=int(random_state),
        min_data_in_leaf=1,
        min_data_in_bin=1,
        verbose=-1,
    )
    model.fit(train_x, target)
    prediction = np.asarray(model.predict(test_x), dtype=float).reshape(-1)
    if prediction.shape != (len(test_x),):
        raise ValueError(
            f"BL3 prediction shape mismatch: expected {(len(test_x),)}, got {prediction.shape}"
        )
    if not np.isfinite(prediction).all():
        raise ValueError("BL3 produced non-finite predictions")
    return np.clip(prediction, 0.0, None)
