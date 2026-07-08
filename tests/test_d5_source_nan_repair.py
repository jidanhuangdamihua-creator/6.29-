from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
)
from src.utils import entity_experiment
from src.utils.source_fillna import fill_source_numeric_na
from scripts import run_d5_experiment


D5_MODEL_FEATURE_COLS = [
    "sales",
    "year",
    "month",
    "week",
    "day",
    "class",
    "perishable",
    "cluster",
    "transactions",
    "oil_price",
    "is_holiday",
]


def _d5_like_frame(entity_id: str = "48_938574", item_id: int = 938574) -> pd.DataFrame:
    dates = pd.date_range("2017-01-01", periods=24, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "entity_id": [entity_id] * len(dates),
            "item_id": [item_id] * len(dates),
            "sales": np.linspace(1.0, 24.0, len(dates)),
            "year": [2017] * len(dates),
            "month": [1] * len(dates),
            "week": [1] * len(dates),
            "day": list(range(1, len(dates) + 1)),
            "class": [101] * len(dates),
            "perishable": [0] * len(dates),
            "cluster": [12] * len(dates),
            "transactions": np.linspace(100.0, 123.0, len(dates)),
            "oil_price": np.linspace(50.0, 55.0, len(dates)),
            "is_holiday": [0] * len(dates),
            "onpromotion": [np.nan] + [0.0] * (len(dates) - 1),
        }
    )
    frame.loc[0, "oil_price"] = np.nan
    frame.loc[1, "transactions"] = np.nan
    frame.attrs["split_role"] = "source"
    frame.attrs["split_mode"] = "ratio"
    frame.attrs["split_config"] = {"train_ratio": 0.8, "val_ratio": 0.1, "test_ratio": 0.1}
    return frame


def test_d5_source_nan_repair_produces_finite_sequence_without_runtime_extra_features():
    source_df = _d5_like_frame()

    repaired = fill_source_numeric_na(source_df, feature_columns=D5_MODEL_FEATURE_COLS)
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(repaired)
    train_df, val_df, test_df, _, feature_columns = normalize_features(
        train_df,
        val_df,
        test_df,
        feature_columns=D5_MODEL_FEATURE_COLS,
    )
    X, y = build_tabular_sequence(
        train_df,
        horizon=1,
        window_size=3,
        feature_columns=feature_columns,
    )

    assert "onpromotion" not in feature_columns
    assert X.shape[2] == len(D5_MODEL_FEATURE_COLS)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_entity_loop_passes_same_solidified_model_features_to_all_tl_methods(monkeypatch):
    source_df = pd.concat(
        [
            _d5_like_frame("48_938574", 938574),
            _d5_like_frame("48_1146785", 1146785),
        ],
        ignore_index=True,
    )
    source_df["class"] = source_df["class"].astype(float)
    source_df.loc[source_df["entity_id"].eq("48_938574"), "oil_price"] = np.nan
    source_df.loc[source_df["entity_id"].eq("48_1146785"), "transactions"] = np.inf
    source_df.loc[source_df["entity_id"].eq("48_1146785"), "class"] = -np.inf
    target_df = _d5_like_frame("48_1159415", 1159415)
    target_df.loc[:, ["transactions", "oil_price"]] = target_df[["transactions", "oil_price"]].fillna(0)
    target_df.attrs["split_role"] = "target"
    original_target_df = target_df.copy(deep=True)

    captured: dict[str, list[str]] = {}
    source_validation_calls: list[dict[str, object]] = []

    original_validate_feature_frame_finite = entity_experiment.validate_feature_frame_finite

    def fake_no_tl_runner(**kwargs):
        captured["No-TL"] = list(kwargs["feature_cols"])
        return {"rmse": 1.0, "accuracy": 1.0, "smape": 1.0, "error": ""}

    def fake_tl_runner(**kwargs):
        method = kwargs.pop("_method")
        captured[method] = list(kwargs["feature_cols"])
        source_values = kwargs["source_df"][D5_MODEL_FEATURE_COLS].to_numpy(dtype=float)
        assert np.isfinite(source_values).all()
        return {
            "rmse": 1.0,
            "accuracy": 1.0,
            "smape": 1.0,
            "error": "",
            "target_store_id": "48",
            "target_item_id": "1159415",
            "meta": {
                "source_key": ("48", "938574"),
                "selected_sources": [{"source_key": ("48", "938574"), "distance": 0.1}],
            },
        }

    def fake_method_runner(method):
        if method == "No-TL":
            return fake_no_tl_runner

        def runner(**kwargs):
            return fake_tl_runner(_method=method, **kwargs)

        return runner

    def recording_validate_feature_frame_finite(df, feature_columns, **kwargs):
        result = original_validate_feature_frame_finite(df, feature_columns, **kwargs)
        if kwargs.get("role") == "source":
            values = df[list(feature_columns)].to_numpy(dtype=float)
            assert np.isfinite(values).all()
            source_validation_calls.append(dict(kwargs))
        return result

    monkeypatch.setattr(entity_experiment, "_method_runner", fake_method_runner)
    monkeypatch.setattr(
        entity_experiment,
        "validate_feature_frame_finite",
        recording_validate_feature_frame_finite,
    )

    rows = entity_experiment.run_single_entity_experiment(
        entity_key="48_1159415",
        source_df=source_df,
        target_entity_df=target_df,
        feature_cols=D5_MODEL_FEATURE_COLS,
        config={
            "dataset_id": 5,
            "dataset_name": "Dataset5",
            "info_sharing": "without",
            "source_count": 2,
            "horizon": 1,
            "window_size": 3,
            "learning_rate": 0.001,
            "source_epochs": 1,
            "target_epochs": 1,
            "batch_size": 1,
        },
        enabled_methods=["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"],
    )

    assert len(rows) == 6
    assert set(captured) == {"No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"}
    assert source_validation_calls
    assert source_validation_calls[0]["entity_id"] == "source_pool"
    assert source_validation_calls[0]["stage"] == "post_build_model_dataframe"
    pd.testing.assert_frame_equal(target_df, original_target_df)
    for feature_cols in captured.values():
        assert feature_cols == D5_MODEL_FEATURE_COLS
        assert "onpromotion" not in feature_cols
    tl_rows = [row for row in rows if row["method"] != "No-TL"]
    assert tl_rows
    for row in tl_rows:
        assert row["target_entity_key"] == "48_1159415"
        assert row["target_store_id"] == "48"
        assert row["target_item_id"] == "1159415"
        assert row["source_identifier"]
        assert row["selected_sources"]


@pytest.mark.parametrize("dataset_id", [4, 5, 6])
def test_source_sanitize_path_is_shared_by_d4_d5_d6(monkeypatch, dataset_id):
    source_df = _d5_like_frame("source", 1)
    source_df["class"] = source_df["class"].astype(float)
    source_df.loc[0, "oil_price"] = np.nan
    source_df.loc[1, "transactions"] = np.inf
    source_df.loc[2, "class"] = -np.inf
    target_df = _d5_like_frame("target", 2)
    target_df.loc[:, ["transactions", "oil_price"]] = target_df[["transactions", "oil_price"]].fillna(0)
    target_df.attrs["split_role"] = "target"

    def fake_method_runner(method):
        def runner(**kwargs):
            source_values = kwargs["source_df"][D5_MODEL_FEATURE_COLS].to_numpy(dtype=float)
            assert np.isfinite(source_values).all()
            return {"rmse": 1.0, "accuracy": 1.0, "smape": 1.0, "error": ""}

        return runner

    monkeypatch.setattr(entity_experiment, "_method_runner", fake_method_runner)

    rows = entity_experiment.run_single_entity_experiment(
        entity_key="target",
        source_df=source_df,
        target_entity_df=target_df,
        feature_cols=D5_MODEL_FEATURE_COLS,
        config={
            "dataset_id": dataset_id,
            "dataset_name": f"Dataset{dataset_id}",
            "info_sharing": "without",
            "source_count": 1,
            "horizon": 1,
            "window_size": 3,
            "learning_rate": 0.001,
            "source_epochs": 1,
            "target_epochs": 1,
            "batch_size": 1,
        },
        enabled_methods=["MSWA-TL"],
    )

    assert rows[0]["dataset_id"] == dataset_id
    assert rows[0]["target_entity_key"] == "target"


def test_source_level_failure_is_recorded_but_target_schema_errors_still_raise(monkeypatch):
    dates = pd.date_range("2017-01-01", periods=4, freq="D")
    source_df = pd.DataFrame(
        {
            "date": dates,
            "entity_id": ["source"] * 4,
            "item_id": [1] * 4,
            "sales": [1.0, 2.0, 3.0, 4.0],
        }
    )
    target_df = pd.DataFrame(
        {
            "date": dates,
            "entity_id": ["target"] * 4,
            "item_id": [2] * 4,
            "sales": [1.0, 2.0, 3.0, 4.0],
        }
    )

    failed_sources = [
        {
            "failed_source_key": ("source", 1),
            "exception_type": "NonFiniteArrayError",
            "exception_message": "X contains non-finite values: nan_count=1 inf_count=0",
        }
    ]
    selected_sources = [{"source_key": ("source", 1), "distance": 1.0, "weight": 1.0}]

    def fake_all_sources_failed(**kwargs):
        from src.transfer_methods.source_failure_tolerance import AllSourcesFailedError

        raise AllSourcesFailedError("MSWA-TL", failed_sources, selected_sources=selected_sources)

    monkeypatch.setattr(entity_experiment, "run_mswa_experiment", fake_all_sources_failed)

    rows = entity_experiment.run_single_entity_experiment(
        entity_key="target",
        source_df=source_df,
        target_entity_df=target_df,
        feature_cols=["sales"],
        config={
            "dataset_id": 5,
            "dataset_name": "Dataset5",
            "info_sharing": "without",
            "source_count": 1,
            "horizon": 1,
            "window_size": 1,
            "learning_rate": 0.001,
            "source_epochs": 1,
            "target_epochs": 1,
            "batch_size": 1,
        },
        enabled_methods=["MSWA-TL"],
    )

    assert rows[0]["skipped_source_count"] == 1
    assert "all selected sources failed" in rows[0]["error"].lower()

    with pytest.raises(ValueError, match="sales must remain"):
        entity_experiment.run_single_entity_experiment(
            entity_key="target",
            source_df=source_df,
            target_entity_df=target_df.drop(columns=["sales"]),
            feature_cols=["sales"],
            config={
                "dataset_id": 5,
                "dataset_name": "Dataset5",
                "info_sharing": "without",
                "source_count": 1,
                "horizon": 1,
                "window_size": 1,
                "learning_rate": 0.001,
                "source_epochs": 1,
                "target_epochs": 1,
                "batch_size": 1,
            },
            enabled_methods=["MSWA-TL"],
        )


def test_d5_target_keys_argument_is_optional_and_parses_narrow_keys(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_d5_experiment.py"])
    default_args = run_d5_experiment._parse_args()
    assert default_args.target_keys is None

    monkeypatch.setattr(
        "sys.argv",
        ["run_d5_experiment.py", "--target-keys", "48_1159415", "48_320682"],
    )
    narrow_args = run_d5_experiment._parse_args()
    assert narrow_args.target_keys == ["48_1159415", "48_320682"]
