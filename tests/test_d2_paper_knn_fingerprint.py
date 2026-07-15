from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from src.protocols.runner_adapter import configure_protocol_frames
from src.source_selection.source_selector import SourceSelector


SEALED_D2 = (
    Path(__file__).resolve().parents[1]
    / "数据集"
    / "固化数据"
    / "d1_d6_sealed_v1"
    / "dataset2"
)


def test_d2_paper_knn_fingerprint_uses_sales_then_promo() -> None:
    source = pd.read_parquet(SEALED_D2 / "source.parquet")
    target = pd.read_parquet(SEALED_D2 / "target.parquet")
    source, target = configure_protocol_frames(
        source,
        target,
        dataset_id="D2",
        scenario="without",
        group_cols=("brand_id", "item_id"),
        observed_start="2018-06-01",
    )

    result = SourceSelector().select_top_k_sources(
        target,
        source,
        feature_cols=("sales", "year", "month", "week", "day"),
        k=3,
        group_cols=("brand_id", "item_id"),
    )

    assert result["meta"]["feature_cols"] == ["sales", "promo"]
    assert [tuple(row["source_key"]) for row in result["sources"]] == [
        ("1", "4"),
        ("1", "6"),
        ("1", "8"),
    ]
    assert [row["distance"] for row in result["sources"]] == pytest.approx(
        [24.98, 26.85, 26.85], abs=0.02
    )
