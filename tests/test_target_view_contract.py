from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from src.data_processing.sealed_daily import (
    TargetViewContractError,
    build_target_views,
    validate_target_view_frame,
)
from src.utils.parquet_data_loader import load_sealed_target_views
from src.protocols.feature_schema import get_predictor_schema
from src.protocols.sealing_protocol import get_target_window


def _target_frame(dataset_id: str = "D1") -> pd.DataFrame:
    window = get_target_window(dataset_id)
    dates = pd.date_range(window.target_start, window.blind_end, freq="D")
    return pd.DataFrame(
        {
            "store_id": 1,
            "item_id": 10,
            "entity_id": "1",
            "date": dates,
            "sales": 1.0,
            "year": dates.year,
            "month": dates.month,
            "week": dates.isocalendar().week.astype("int64"),
            "day": dates.day,
        }
    )


def test_four_target_views_have_exact_windows_and_no_blind_truth() -> None:
    views = build_target_views(_target_frame(), "D1")
    target = get_target_window("D1")

    assert set(views) == {
        "knn_observed_frame",
        "observed_model_frame",
        "blind_covariate_frame",
        "evaluator_truth_frame",
    }
    assert views.knn_observed_frame["date"].nunique() == 30
    assert views.observed_model_frame["date"].nunique() == 30
    assert views.blind_covariate_frame["date"].nunique() == 180
    assert views.evaluator_truth_frame["date"].nunique() == 180
    assert views.knn_observed_frame["date"].min().date() == target.observed_start
    assert views.knn_observed_frame["date"].max().date() == target.observed_end
    assert views.blind_covariate_frame["date"].min().date() == target.blind_start
    assert views.blind_covariate_frame["date"].max().date() == target.blind_end

    predictor_names = set(get_predictor_schema("D1").ordered_names)
    assert set(views.observed_model_frame.columns) == {
        "target_entity_key",
        "date",
        *predictor_names,
    }
    assert "sales" in views.observed_model_frame.columns
    assert "sales" not in views.blind_covariate_frame.columns
    assert set(views.evaluator_truth_frame.columns) == {
        "target_entity_key",
        "date",
        "y_true",
        "is_synthetic_date",
        "truth_key",
    }
    assert not {"y_true", "truth_key", "is_synthetic_date"}.intersection(
        views.blind_covariate_frame.columns
    )


def test_blind_view_rejects_sales_and_unknown_columns() -> None:
    frame = _target_frame()
    frame["unexpected_future_field"] = 1.0

    views = build_target_views(frame, "D1")
    assert "unexpected_future_field" not in views.blind_covariate_frame.columns
    assert "sales" not in views.blind_covariate_frame.columns


def test_view_rejects_missing_calendar_date_without_filling_target_truth() -> None:
    frame = _target_frame()
    frame = frame[frame["date"] != pd.Timestamp("2017-12-01")]

    with pytest.raises(TargetViewContractError, match="date|calendar|180"):
        build_target_views(frame, "D1")


def test_materialized_blind_view_rejects_truth_and_unknown_fields() -> None:
    views = build_target_views(_target_frame(), "D1")
    invalid = views.blind_covariate_frame.assign(sales=0.0)

    with pytest.raises(TargetViewContractError, match="exact|unknown|blind"):
        validate_target_view_frame(invalid, "D1", "blind_covariate_frame")


def test_sealed_loader_reads_only_versioned_dataset_target(tmp_path) -> None:
    dataset_dir = tmp_path / "d1_d6_sealed_v1" / "dataset1"
    dataset_dir.mkdir(parents=True)
    _target_frame().to_parquet(dataset_dir / "target.parquet", index=False)

    views = load_sealed_target_views(1, dataset_dir)

    assert views.dataset_id == "D1"
    assert views.blind_covariate_frame["date"].nunique() == 180
