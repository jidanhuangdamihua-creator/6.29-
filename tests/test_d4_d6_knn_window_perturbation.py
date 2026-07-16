from __future__ import annotations

import numpy as np
import pandas as pd

from src.protocols.candidate_pool import build_candidate_pool_digest
from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector


def test_d4_d6_audit_uses_production_digest_and_daily_representation() -> None:
    dates = pd.date_range("2020-01-01", periods=35, freq="D")
    source_dates = pd.date_range(dates[0] - pd.Timedelta(days=150), periods=180, freq="D")
    target = pd.DataFrame(
        {
            "store_nbr": "S1",
            "item_nbr": "I1",
            "family": "F1",
            "date": dates,
            "sales": np.r_[np.zeros(30), np.ones(5)],
            "onpromotion": 0.0,
            "oil_price": 40.0,
        }
    )
    source = pd.concat(
        [
            pd.DataFrame(
                {
                    "store_nbr": store,
                    "item_nbr": item,
                    "family": "F1",
                    "date": source_dates,
                    "sales": value,
                    "onpromotion": 0.0,
                    "oil_price": 40.0,
                }
            )
            for store, item, value in (("S1", "I2", 1.0), ("S2", "I2", 2.0))
        ]
    )
    configured_source, configured_target = configure_protocol_frames(
        source,
        target,
        dataset_id="D5",
        scenario="with",
        group_cols=("store_nbr", "item_nbr"),
        grouping_col="family",
        observed_start="2020-01-01",
    )
    result = SourceSelector().select_top_k_sources(
        configured_target,
        configured_source,
        feature_cols=("sales",),
        k=2,
        group_cols=("store_nbr", "item_nbr"),
    )
    digest_input = result["meta"]["candidate_pool_digest_input"]

    assert result["meta"]["representation"] == "daily_sales_flattened_30d"
    assert result["meta"]["feature_cols"] == ["sales", "onpromotion", "oil_price"]
    assert result["meta"]["candidate_pool_digest"] == build_candidate_pool_digest(**digest_input)
    assert result["meta"]["selected_sources_runtime"] == result["sources"]
