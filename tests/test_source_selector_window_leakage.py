from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.source_selection.source_selector import SourceSelector
from src.utils.parquet_data_loader import load_parquet_source_target


def test_d4_d6_loader_materializes_runtime_windows_and_bounds_source_history(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "dataset4-source.parquet"
    target_path = tmp_path / "dataset4-target.parquet"
    source_path.touch()
    target_path.touch()

    source_frame = pd.DataFrame(
        {
            "entity_id": ["source-a"] * 4,
            "item_id": ["item-a"] * 4,
            "date": pd.to_datetime(
                ["2023-04-05", "2023-04-06", "2024-01-30", "2024-01-31"]
            ),
            "sales": [1.0, 2.0, 3.0, 4.0],
        }
    )
    target_frame = pd.DataFrame(
        {
            "entity_id": ["target-a"] * 40,
            "item_id": ["item-a"] * 40,
            "date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "sales": range(40),
        }
    )

    def fake_read_parquet(path: Path) -> pd.DataFrame:
        if Path(path) == source_path:
            return source_frame.copy()
        if Path(path) == target_path:
            return target_frame.copy()
        raise AssertionError(f"unexpected parquet path: {path}")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    source_df, target_df = load_parquet_source_target(
        dataset_id=4,
        parquet_dir=tmp_path,
        windows={
            "dataset_id": 4,
            "train_start": "2024-01-01",
            "test_end": "2024-02-09",
            "target_train_window": {"start": "2024-01-01", "end": "2024-01-15"},
        },
        source_history_days=300,
    )

    assert target_df["date"].min() == pd.Timestamp("2024-01-01")
    assert target_df["date"].max() == pd.Timestamp("2024-02-09")
    assert target_df.attrs["target_observed_start"] == pd.Timestamp("2024-01-01")
    assert target_df.attrs["target_observed_end"] == pd.Timestamp("2024-01-30")
    assert source_df.attrs["source_history_start"] == pd.Timestamp("2023-04-06")
    assert source_df.attrs["source_history_end"] == pd.Timestamp("2024-01-30")
    assert source_df["date"].tolist() == [
        pd.Timestamp("2023-04-06"),
        pd.Timestamp("2024-01-30"),
    ]
    for frame in (source_df, target_df):
        assert frame.attrs["selection_authority"] == "runtime"
        assert frame.attrs["protocol_version"] == "runtime_knn_windowed_stats_v1"
        assert frame.attrs["target_test_excluded"] is True
        assert frame.attrs["source_future_excluded"] is True
        assert frame.attrs["source_alignment_mode"] == "exact_target_observed_dates"
        assert frame.attrs["representation"] == "mean_std_min_max_last"
        assert frame.attrs["scaling"] == "none"
        assert frame.attrs["scaler_fit_scope"] == "not_applicable"


def _runtime_selector_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_dates = pd.date_range("2024-01-01", periods=30, freq="D")
    test_dates = pd.date_range("2024-01-31", periods=10, freq="D")
    target_df = pd.DataFrame(
        {
            "entity_id": ["target-a"] * 40,
            "item_id": ["target-item"] * 40,
            "date": observed_dates.append(test_dates),
            "sales": [0.0] * 40,
        }
    )

    source_rows: list[dict[str, object]] = []
    for entity_id, sales in (("source-a", 0.0), ("source-b", 10.0)):
        for date in observed_dates:
            source_rows.append(
                {"entity_id": entity_id, "item_id": "item", "date": date, "sales": sales}
            )
        source_rows.append(
            {
                "entity_id": entity_id,
                "item_id": "item",
                "date": pd.Timestamp("2024-01-31"),
                "sales": sales,
            }
        )
    for date in observed_dates[:-1]:
        source_rows.append(
            {
                "entity_id": "source-incomplete",
                "item_id": "item",
                "date": date,
                "sales": 50.0,
            }
        )
    source_df = pd.DataFrame(source_rows)

    attrs = {
        "dataset_name": "Dataset4",
        "selection_authority": "runtime",
        "protocol_version": "runtime_knn_windowed_stats_v1",
        "target_observed_start": pd.Timestamp("2024-01-01"),
        "target_observed_end": pd.Timestamp("2024-01-30"),
        "source_history_start": pd.Timestamp("2023-04-06"),
        "source_history_end": pd.Timestamp("2024-01-30"),
        "target_test_excluded": True,
        "source_future_excluded": True,
        "source_alignment_mode": "exact_target_observed_dates",
        "representation": "mean_std_min_max_last",
        "scaling": "none",
        "scaler_fit_scope": "not_applicable",
    }
    target_df.attrs.update(attrs)
    target_df.attrs["role"] = "target"
    source_df.attrs.update(attrs)
    source_df.attrs["role"] = "source"
    return source_df, target_df


def _select(source_df: pd.DataFrame, target_df: pd.DataFrame) -> dict[str, object]:
    return SourceSelector().select_top_k_sources(
        target_df=target_df,
        source_df=source_df,
        feature_cols=["sales"],
        k=1,
        group_cols=("entity_id", "item_id"),
    )


def _top_source(result: dict[str, object]) -> tuple[str, str]:
    sources = result["sources"]
    assert isinstance(sources, list)
    return tuple(sources[0]["source_key"])


def test_target_test_sales_do_not_change_runtime_top_k() -> None:
    source_df, target_df = _runtime_selector_frames()
    baseline = _select(source_df, target_df)

    perturbed = target_df.copy()
    perturbed.loc[perturbed["date"] > pd.Timestamp("2024-01-30"), "sales"] = 1000.0
    changed = _select(source_df, perturbed)

    assert _top_source(baseline) == ("source-a", "item")
    assert _top_source(changed) == _top_source(baseline)


def test_source_future_sales_do_not_change_runtime_top_k() -> None:
    source_df, target_df = _runtime_selector_frames()
    baseline = _select(source_df, target_df)

    perturbed = source_df.copy()
    future = perturbed["date"] > pd.Timestamp("2024-01-30")
    perturbed.loc[future & perturbed["entity_id"].eq("source-a"), "sales"] = 1000.0
    changed = _select(perturbed, target_df)

    assert _top_source(changed) == _top_source(baseline)


def test_observed_sales_can_change_runtime_top_k() -> None:
    source_df, target_df = _runtime_selector_frames()
    baseline = _select(source_df, target_df)

    perturbed = target_df.copy()
    observed = perturbed["date"] <= pd.Timestamp("2024-01-30")
    perturbed.loc[observed, "sales"] = 10.0
    changed = _select(source_df, perturbed)

    assert _top_source(baseline) == ("source-a", "item")
    assert _top_source(changed) == ("source-b", "item")


def test_runtime_selection_skips_source_missing_observed_date_and_records_reason() -> None:
    source_df, target_df = _runtime_selector_frames()

    result = _select(source_df, target_df)
    meta = result["meta"]
    assert isinstance(meta, dict)

    assert meta["selection_authority"] == "runtime"
    assert meta["protocol_version"] == "runtime_knn_windowed_stats_v1"
    assert meta["source_alignment_mode"] == "exact_target_observed_dates"
    assert meta["representation"] == "mean_std_min_max_last"
    assert meta["scaling"] == "none"
    assert meta["scaler_fit_scope"] == "not_applicable"
    assert meta["selected_sources_runtime"] == result["sources"]
    assert len(meta["candidate_pool_digest"]) == 64
    assert len(meta["selection_result_digest"]) == 64
    assert meta["source_skip_diagnostics"] == [
        {
            "source_key": ("source-incomplete", "item"),
            "reason": "missing_target_observed_dates",
            "missing_dates": ["2024-01-30"],
        }
    ]


def test_d4_d6_runtime_selection_missing_observed_metadata_fails_fast() -> None:
    source_df, target_df = _runtime_selector_frames()
    target_df.attrs.pop("target_observed_end")

    with pytest.raises(ValueError, match="Missing D4-D6 runtime KNN metadata"):
        _select(source_df, target_df)


def test_unmarked_legacy_selection_does_not_require_runtime_window_metadata() -> None:
    source_df, target_df = _runtime_selector_frames()
    source_df.attrs.clear()
    target_df.attrs.clear()

    result = _select(source_df, target_df)

    assert result["sources"]
    assert "selection_authority" not in result["meta"]
