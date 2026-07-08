from __future__ import annotations

import pandas as pd

from src.data_processing.data_preprocessing import infer_source_selection_feature_columns
from src.source_selection.source_selector import SourceSelector


def _source_target_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    source_df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "entity_id": ["s1"] * 3 + ["s2"] * 3,
            "item_id": ["i1"] * 6,
            "store_nbr": [1] * 6,
            "item_nbr": [101] * 6,
            "store_id": [10] * 6,
            "product_id": [20] * 6,
            "category_id": [30] * 6,
            "custom_nbr": [40] * 6,
            "sales": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            "promo": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    target_df = pd.DataFrame(
        {
            "date": dates,
            "entity_id": ["t1"] * 3,
            "item_id": ["i1"] * 3,
            "store_nbr": [1] * 3,
            "item_nbr": [101] * 3,
            "store_id": [10] * 3,
            "product_id": [20] * 3,
            "category_id": [30] * 3,
            "custom_nbr": [40] * 3,
            "sales": [1.0, 2.0, 3.0],
            "promo": [0.0, 1.0, 0.0],
        }
    )
    return source_df, target_df


def test_infer_source_selection_excludes_identifier_like_columns() -> None:
    source_df, target_df = _source_target_frames()

    info = infer_source_selection_feature_columns(source_df, target_df)

    assert info["knn_feature_mode"] == "paper_available_features_no_ids_v2"
    assert info["selected_features"] == ["sales", "promo"]
    for col in ("store_nbr", "item_nbr", "store_id", "product_id", "category_id", "custom_nbr"):
        assert col in info["excluded_by_rule"]


def test_source_selector_keeps_explicit_feature_columns_order_without_infer_widening() -> None:
    source_df, target_df = _source_target_frames()

    result = SourceSelector().select_top_k_sources(
        target_df=target_df,
        source_df=source_df,
        feature_cols=["promo", "sales"],
        k=1,
    )

    assert result["meta"]["feature_cols"] == ["promo", "sales"]
    assert result["meta"]["knn_feature_mode"] == "explicit_feature_cols"
