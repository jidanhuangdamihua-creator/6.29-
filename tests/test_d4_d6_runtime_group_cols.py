from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.protocols.runner_adapter import configure_protocol_frames, source_key_mask
from src.source_selection.source_selector import SourceSelector


@pytest.mark.parametrize(
    "dataset_id,group_cols,grouping_col",
    (
        (4, ("store_id", "product_id"), "second_category_id"),
        (5, ("store_nbr", "item_nbr"), "family"),
        (6, ("store_id", "item_id"), "dept_id"),
    ),
)
def test_shared_selector_records_physical_group_keys(
    dataset_id: int,
    group_cols: tuple[str, str],
    grouping_col: str,
) -> None:
    dates = pd.date_range("2020-01-01", periods=35, freq="D")
    source_dates = pd.date_range(dates[0] - pd.Timedelta(days=150), periods=180, freq="D")
    target = pd.DataFrame(
        {
            group_cols[0]: "T1",
            group_cols[1]: "I0",
            grouping_col: "G1",
            "date": dates,
            "sales": np.r_[np.zeros(30), np.ones(5)],
        }
    )
    source = pd.concat(
        [
            pd.DataFrame(
                {
                    group_cols[0]: store,
                    group_cols[1]: item,
                    grouping_col: "G1",
                    "date": source_dates,
                    "sales": value,
                }
            )
            for store, item, value in (("T1", "I1", 1.0), ("S2", "I2", 2.0))
        ],
        ignore_index=True,
    )
    if dataset_id == 5:
        source["onpromotion"] = 0.0
        source["oil_price"] = 40.0
        target["onpromotion"] = 0.0
        target["oil_price"] = 40.0
    source, target = configure_protocol_frames(
        source,
        target,
        dataset_id=dataset_id,
        scenario="with",
        group_cols=group_cols,
        grouping_col=grouping_col,
        observed_start="2020-01-01",
    )
    result = SourceSelector().select_top_k_sources(
        target,
        source,
        feature_cols=("sales",),
        k=2,
        group_cols=group_cols,
    )

    assert result["meta"]["group_cols"] == list(group_cols)
    assert {tuple(row["source_key"]) for row in result["sources"]} == {
        ("T1", "I1"),
        ("S2", "I2"),
    }
    for row in result["sources"]:
        assert source_key_mask(source, group_cols, row["source_key"]).any()
