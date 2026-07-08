from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing.data_preprocessing import build_tabular_sequence, normalize_features
from src.utils.finite_diagnostics import NonFiniteArrayError, validate_feature_frame_finite


def _sequence_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "entity_id": ["store"] * 4,
            "item_id": ["item"] * 4,
            "sales": [10.0, 11.0, 12.0, 13.0],
            "promo": [0.0, 1.0, 0.0, 1.0],
            "silent_extra": [100.0, 101.0, 102.0, 103.0],
        }
    )


def test_normalize_features_uses_explicit_columns_and_ignores_extra_numeric() -> None:
    train_df = pd.DataFrame({"sales": [0.0, 10.0], "promo": [1.0, 3.0], "silent_extra": [100.0, 200.0]})
    val_df = pd.DataFrame({"sales": [5.0], "promo": [2.0], "silent_extra": [150.0]})
    test_df = pd.DataFrame({"sales": [20.0], "promo": [5.0], "silent_extra": [300.0]})

    train_scaled, val_scaled, test_scaled, _, feature_columns = normalize_features(
        train_df,
        val_df,
        test_df,
        feature_columns=["sales", "promo"],
    )

    assert feature_columns == ["sales", "promo"]
    assert train_scaled["silent_extra"].tolist() == [100.0, 200.0]
    assert val_scaled["silent_extra"].tolist() == [150.0]
    assert test_scaled["silent_extra"].tolist() == [300.0]


def test_build_tabular_sequence_uses_explicit_order_and_width() -> None:
    df = _sequence_frame()

    x, y = build_tabular_sequence(
        df,
        horizon=1,
        window_size=2,
        feature_columns=["sales", "promo"],
    )

    assert x.shape == (2, 2, 2)
    np.testing.assert_allclose(x[0], np.array([[10.0, 0.0], [11.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(y, np.array([12.0, 13.0], dtype=np.float32))


def test_build_tabular_sequence_requires_sales_in_explicit_columns() -> None:
    with pytest.raises(ValueError, match="sales"):
        build_tabular_sequence(
            _sequence_frame(),
            horizon=1,
            window_size=2,
            feature_columns=["promo"],
        )


def test_validate_feature_frame_finite_reports_bad_columns_and_context() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=2),
            "sales": [1.0, np.nan],
            "promo": [np.inf, 0.0],
        }
    )

    with pytest.raises(NonFiniteArrayError) as captured:
        validate_feature_frame_finite(
            df,
            ["sales", "promo"],
            context="pre_normalize",
            dataset_id=5,
            method="SS-TL",
            role="source",
            entity_id="11_848765",
            stage="pre_normalize",
        )

    diagnostics = captured.value.diagnostics
    assert diagnostics["dataset_id"] == 5
    assert diagnostics["method"] == "SS-TL"
    assert diagnostics["role"] == "source"
    assert diagnostics["entity_id"] == "11_848765"
    assert diagnostics["stage"] == "pre_normalize"
    assert diagnostics["bad_columns"]["sales"]["nan_count"] == 1
    assert diagnostics["bad_columns"]["promo"]["posinf_count"] == 1
