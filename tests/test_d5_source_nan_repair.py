from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
)
from src.evaluation.metrics import compute_metrics_with_protocol
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


def _feature_frame() -> pd.DataFrame:
    """Return the shared numeric model payload, not a dataset protocol fixture."""
    dates = pd.date_range("2017-01-01", periods=40, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates.to_numpy(),
            "sales": np.linspace(1.0, 40.0, len(dates)),
            "year": dates.year.to_numpy(),
            "month": dates.month.to_numpy(),
            "week": dates.isocalendar().week.to_numpy(dtype=int),
            "day": dates.day.to_numpy(),
            "class": [101] * len(dates),
            "perishable": [0] * len(dates),
            "cluster": [12] * len(dates),
            "transactions": np.linspace(100.0, 139.0, len(dates)),
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


def _d5_protocol_frame(store_nbr: int, item_nbr: int) -> pd.DataFrame:
    """D5 fixture using the solidified runner's (store_nbr, item_nbr) keys."""
    frame = _feature_frame()
    frame["entity_id"] = f"{store_nbr}_{item_nbr}"
    frame["store_nbr"] = store_nbr
    frame["item_nbr"] = item_nbr
    frame["item_id"] = item_nbr
    frame["family"] = "F1"
    return frame


def _d4_protocol_frame(store_id: int, product_id: int) -> pd.DataFrame:
    """D4 fixture with its runtime identity and target-domain fields."""
    frame = _feature_frame()
    frame["entity_id"] = f"{store_id}_{product_id}"
    frame["store_id"] = store_id
    frame["product_id"] = product_id
    frame["first_category_id"] = 15
    frame["second_category_id"] = 20
    frame["category"] = "20"
    return frame


def _d6_protocol_frame(store_id: str, item_id: str) -> pd.DataFrame:
    """D6 fixture with its actual keys, department, and store hierarchy fields."""
    frame = _feature_frame()
    frame["entity_id"] = f"{store_id}_{item_id}"
    frame["store_id"] = store_id
    frame["item_id"] = item_id
    frame["dept_id"] = "FOODS_3"
    frame["cat_id"] = "FOODS"
    frame["state_id"] = "CA"
    frame["id"] = f"{item_id}_{store_id}_evaluation"
    return frame


def _strict_metric_result(
    *,
    expected_metric_identity: dict[str, object] | None = None,
    source_key: tuple[object, object] | None = None,
) -> dict[str, object]:
    """Build a real strict original-sales metric payload for fake TL runners."""
    count = int((expected_metric_identity or {}).get("metric_sample_count", 2))
    y_true = np.linspace(10.0, 10.0 + count - 1, count)
    result = compute_metrics_with_protocol(
        y_true=y_true,
        y_pred=y_true + 0.5,
        metric_protocol={
            "current_metric_space": "original_sales_space",
            "paper_metric_space": "original_sales_space",
            "strict_paper_metrics": True,
        },
    )
    result["error"] = ""
    result["prediction_shape"] = (count, 1)
    if expected_metric_identity is not None:
        result.update(expected_metric_identity)
    if source_key is not None:
        result["meta"] = {
            "source_key": source_key,
            "selected_sources": [
                {"source_key": source_key, "distance": 0.1, "weight": 1.0}
            ],
        }
    return result


def test_d5_source_nan_repair_produces_finite_sequence_without_runtime_extra_features():
    source_df = _d5_protocol_frame(48, 938574)

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
            _d5_protocol_frame(48, 938574),
            _d5_protocol_frame(48, 1146785),
        ],
        ignore_index=True,
    )
    source_df["class"] = source_df["class"].astype(float)
    source_df.loc[source_df["entity_id"].eq("48_938574"), "oil_price"] = np.nan
    source_df.loc[source_df["entity_id"].eq("48_1146785"), "transactions"] = np.inf
    source_df.loc[source_df["entity_id"].eq("48_1146785"), "class"] = -np.inf
    target_df = _d5_protocol_frame(48, 1159415)
    target_df.loc[:, ["transactions", "oil_price"]] = target_df[["transactions", "oil_price"]].fillna(0)
    target_df.attrs["split_role"] = "target"
    original_target_df = target_df.copy(deep=True)

    captured: dict[str, list[str]] = {}
    source_validation_calls: list[dict[str, object]] = []

    original_validate_feature_frame_finite = entity_experiment.validate_feature_frame_finite

    def fake_no_tl_runner(**kwargs):
        captured["No-TL"] = list(kwargs["feature_cols"])
        assert kwargs["expected_metric_identity"]
        return _strict_metric_result(
            expected_metric_identity=kwargs["expected_metric_identity"]
        )

    def fake_tl_runner(**kwargs):
        method = kwargs.pop("_method")
        captured[method] = list(kwargs["feature_cols"])
        source_values = kwargs["source_df"][D5_MODEL_FEATURE_COLS].to_numpy(dtype=float)
        assert np.isfinite(source_values).all()
        expected_metric_identity = kwargs.get("expected_metric_identity")
        if expected_metric_identity is None:
            expected_metric_identity = entity_experiment._metric_identity_from_manifest(
                kwargs["target_df"].attrs["protocol_sample_manifest"],
                horizon=kwargs["horizon"],
            )
        result = _strict_metric_result(
            expected_metric_identity=expected_metric_identity,
            source_key=(48, 938574),
        )
        result.update(target_store_id="48", target_item_id="1159415")
        return result

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
            "group_cols": ("store_nbr", "item_nbr"),
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
        assert row["target_entity_key"] == "48/1159415"
        assert row["target_store_id"] == "48"
        assert row["target_item_id"] == "1159415"
        assert row["source_identifier"]
        assert row["selected_sources"]


@pytest.mark.parametrize("dataset_id", [4, 5, 6])
def test_source_sanitize_path_is_shared_by_d4_d5_d6(monkeypatch, dataset_id):
    if dataset_id == 4:
        source_df = _d4_protocol_frame(166, 259)
        target_df = _d4_protocol_frame(166, 258)
        entity_key = "166_258"
        group_cols = ("store_id", "product_id")
        source_key = (166, 259)
    elif dataset_id == 5:
        source_df = _d5_protocol_frame(48, 938574)
        target_df = _d5_protocol_frame(48, 1159415)
        entity_key = "48_1159415"
        group_cols = ("store_nbr", "item_nbr")
        source_key = (48, 938574)
    else:
        source_df = _d6_protocol_frame("CA_1", "FOODS_3_226")
        target_df = _d6_protocol_frame("CA_1", "FOODS_3_586")
        entity_key = "CA_1_FOODS_3_586"
        group_cols = ("store_id", "item_id")
        source_key = ("CA_1", "FOODS_3_226")
    source_df["class"] = source_df["class"].astype(float)
    source_df.loc[0, "oil_price"] = np.nan
    source_df.loc[1, "transactions"] = np.inf
    source_df.loc[2, "class"] = -np.inf
    target_df.loc[:, ["transactions", "oil_price"]] = target_df[["transactions", "oil_price"]].fillna(0)
    target_df.attrs["split_role"] = "target"

    def fake_method_runner(method):
        def runner(**kwargs):
            source_values = kwargs["source_df"][D5_MODEL_FEATURE_COLS].to_numpy(dtype=float)
            assert np.isfinite(source_values).all()
            return _strict_metric_result(
                expected_metric_identity=kwargs["expected_metric_identity"],
                source_key=source_key,
            )

        return runner

    monkeypatch.setattr(entity_experiment, "_method_runner", fake_method_runner)

    rows = entity_experiment.run_single_entity_experiment(
        entity_key=entity_key,
        source_df=source_df,
        target_entity_df=target_df,
        feature_cols=D5_MODEL_FEATURE_COLS,
        config={
            "dataset_id": dataset_id,
            "dataset_name": f"Dataset{dataset_id}",
            "info_sharing": "without",
            "group_cols": group_cols,
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
    assert rows[0]["target_entity_key"] == {
        4: "166/258",
        5: "48/1159415",
        6: "CA_1/FOODS_3_586",
    }[dataset_id]


def test_source_level_failure_is_recorded_but_target_schema_errors_still_raise(monkeypatch):
    source_df = _d5_protocol_frame(48, 938574)
    target_df = _d5_protocol_frame(48, 1159415)
    target_df.attrs["split_role"] = "target"

    failed_sources = [
        {
            "failed_source_key": ("48", "938574"),
            "exception_type": "NonFiniteArrayError",
            "exception_message": "X contains non-finite values: nan_count=1 inf_count=0",
        }
    ]
    selected_sources = [{"source_key": ("48", "938574"), "distance": 1.0, "weight": 1.0}]

    def fake_all_sources_failed(**kwargs):
        from src.transfer_methods.source_failure_tolerance import AllSourcesFailedError

        raise AllSourcesFailedError("MSWA-TL", failed_sources, selected_sources=selected_sources)

    monkeypatch.setattr(entity_experiment, "run_mswa_experiment", fake_all_sources_failed)

    rows = entity_experiment.run_single_entity_experiment(
        entity_key="48_1159415",
        source_df=source_df,
        target_entity_df=target_df,
        feature_cols=["sales"],
        config={
            "dataset_id": 5,
            "dataset_name": "Dataset5",
            "info_sharing": "without",
            "group_cols": ("store_nbr", "item_nbr"),
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

    with pytest.raises(ValueError, match="sales"):
        entity_experiment.run_single_entity_experiment(
            entity_key="48_1159415",
            source_df=source_df,
            target_entity_df=target_df.drop(columns=["sales"]),
            feature_cols=["sales"],
            config={
                "dataset_id": 5,
                "dataset_name": "Dataset5",
                "info_sharing": "without",
                "group_cols": ("store_nbr", "item_nbr"),
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
