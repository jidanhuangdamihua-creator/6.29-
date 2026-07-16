from __future__ import annotations

import numpy as np
import pandas as pd

from src.protocols.feature_schema import get_knn_schema
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector


OBSERVED = pd.date_range("2018-06-01", periods=30, freq="D")
SOURCE_DATES = pd.date_range(OBSERVED[0] - pd.Timedelta(days=150), periods=180, freq="D")
TARGET_DATES = pd.date_range(OBSERVED[0], periods=35, freq="D")


def _source_item(item: int) -> pd.DataFrame:
    sales = float(item)
    promo = 0.0
    if item == 2:
        sales, promo = 1.0, 1.0
    return pd.DataFrame(
        {
            "brand_id": "1",
            "item_id": str(item),
            "date": SOURCE_DATES,
            "sales": sales,
            "promo": promo,
            "predictor_only": float(item * 100),
        }
    )


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.concat([_source_item(item) for item in range(1, 10)], ignore_index=True)
    target = pd.DataFrame(
        {
            "brand_id": "1",
            "item_id": "10",
            "date": TARGET_DATES,
            "sales": np.r_[np.zeros(30), np.full(5, 9999.0)],
            "promo": np.r_[np.zeros(30), np.full(5, 9999.0)],
        }
    )
    return source, target


def _select(source: pd.DataFrame, target: pd.DataFrame):
    source, target = configure_protocol_frames(
        source,
        target,
        dataset_id="D2",
        scenario="without",
        group_cols=("brand_id", "item_id"),
        observed_start=OBSERVED[0],
    )
    return SourceSelector().select_top_k_sources(
        target,
        source,
        feature_cols=("sales", "predictor_only"),
        k=3,
        group_cols=("brand_id", "item_id"),
    )


def test_d2_paper_knn_fingerprint_uses_only_historical_sales_then_promo() -> None:
    source, target = _frames()
    baseline = _select(source, target)

    assert get_knn_schema("D2").ordered_names == ("sales", "promo")
    assert baseline["meta"]["feature_cols"] == ["sales", "promo"]
    assert baseline["meta"]["source_window_start"] == SOURCE_DATES[0].strftime("%Y-%m-%d")
    assert baseline["meta"]["knn_observed_start"] == OBSERVED[0].strftime("%Y-%m-%d")
    assert [tuple(row["source_key"]) for row in baseline["sources"]] == [
        ("1", "1"),
        ("1", "2"),
        ("1", "3"),
    ]

    future_changed = target.copy()
    future_changed.loc[future_changed["date"] > OBSERVED[-1], ["sales", "promo"]] = -1e12
    predictor_changed = source.assign(predictor_only=-1e12)
    invariant = _select(predictor_changed, future_changed)
    assert invariant["meta"]["selection_result_digest"] == baseline["meta"]["selection_result_digest"]
    assert invariant["sources"] == baseline["sources"]

    historical_promo_changed = source.copy()
    historical_promo_changed.loc[
        (historical_promo_changed["item_id"] == "1")
        & historical_promo_changed["date"].isin(OBSERVED),
        "promo",
    ] = 9.0
    sensitive = _select(historical_promo_changed, target)
    assert sensitive["meta"]["selection_result_digest"] != baseline["meta"]["selection_result_digest"]
    assert tuple(sensitive["sources"][0]["source_key"]) != ("1", "1")
