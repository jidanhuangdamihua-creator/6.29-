from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from src.data_processing.data_preprocessing import (
    build_tabular_sequence,
    normalize_features,
    temporal_split_by_ratio_or_dates,
)
from src.protocols.provenance import bind_actual_cnn_source_frame
from src.utils.dataframe_attrs import (
    PROTOCOL_CONTEXT_ATTR,
    ProtocolFrameContext,
    get_protocol_frame_context,
)
from src.utils.source_fillna import fill_source_numeric_na


class DeepcopyBomb:
    def __deepcopy__(self, memo):
        raise AssertionError("heavy runtime metadata must not be deep-copied")


HEAVY_KEYS = {
    "protocol_candidate_keys",
    "protocol_knn_observed_frame",
    "forecast_consumer_frame",
    "prepared_daily_sequence_pool",
    "protocol_sample_manifest",
    "protocol_actual_cnn_audit",
    "protocol_raw_partition",
    "protocol_fitted_scaler",
    "audit_metadata",
}


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    frame = pd.DataFrame(
        {
            "entity_id": "S1",
            "item_id": "I1",
            "date": dates,
            "sales": np.arange(40, dtype=float),
            "planned_price": np.arange(40, dtype=float) + 100.0,
        }
    )
    frame.loc[3, "planned_price"] = np.nan
    return frame


def test_working_pipeline_uses_allowlist_and_never_deepcopies_heavy_attrs() -> None:
    frame = _frame()
    canonical_observed = frame.iloc[:30].copy()
    forecast = frame.iloc[30:].copy()
    canonical_before = canonical_observed.copy(deep=True)
    context = ProtocolFrameContext(
        observed_frames={"source": canonical_observed},
        forecast_frame=forecast,
        candidate_keys=(("S1", "I1"),),
        prepared_pool=DeepcopyBomb(),
        protocol_report=DeepcopyBomb(),
    )
    frame.attrs.update(
        {
            PROTOCOL_CONTEXT_ATTR: context,
            "split_role": "source",
            "split_mode": "ratio",
            "split_config": {"train_ratio": 0.8, "val_ratio": 0.1},
            "protocol_version": "d1_d6_protocol_v1",
            "protocol_dataset_id": "D1",
            "protocol_scenario": "with_information_sharing",
            "protocol_target_key": ("T1", "I1"),
            "protocol_group_cols": ("entity_id", "item_id"),
            "protocol_cell_identity": (
                "D1",
                "with_information_sharing",
                1,
                42,
                ("T1", "I1"),
            ),
            "protocol_candidate_keys": [DeepcopyBomb()],
            "protocol_knn_observed_frame": canonical_observed,
            "forecast_consumer_frame": forecast,
            "prepared_daily_sequence_pool": DeepcopyBomb(),
            "protocol_sample_manifest": DeepcopyBomb(),
            "protocol_actual_cnn_audit": {"bomb": DeepcopyBomb()},
            "audit_metadata": DeepcopyBomb(),
        }
    )

    repaired = fill_source_numeric_na(
        frame,
        feature_columns=("sales", "planned_price"),
    )
    train, validation, test = temporal_split_by_ratio_or_dates(repaired)
    train, validation, test, _, features = normalize_features(
        train,
        validation,
        test,
        feature_columns=("sales", "planned_price"),
    )
    build_tabular_sequence(train, horizon=1, window_size=3, feature_columns=features)

    for working in (repaired, train, validation, test):
        assert HEAVY_KEYS.isdisjoint(working.attrs)
        assert working.attrs["split_role"] == "source"
        assert isinstance(get_protocol_frame_context(working), ProtocolFrameContext)
    assert deepcopy(context) is context
    pd.testing.assert_frame_equal(canonical_observed, canonical_before, check_exact=True)


def test_actual_cnn_raw_scaler_and_audit_live_only_in_context() -> None:
    frame = _frame().dropna().reset_index(drop=True)
    bind_actual_cnn_source_frame(
        frame,
        source_key=("S1", "I1"),
        group_cols=("entity_id", "item_id"),
        feature_cols=("sales", "planned_price"),
    )
    frame.attrs.update(
        split_role="source",
        split_mode="ratio",
        split_config={"train_ratio": 0.8, "val_ratio": 0.1},
    )
    train, validation, test = temporal_split_by_ratio_or_dates(frame)
    train, validation, test, _, _ = normalize_features(
        train,
        validation,
        test,
        feature_columns=("sales", "planned_price"),
    )
    context = get_protocol_frame_context(train)
    assert context is not None
    assert context.raw_partition is not None
    assert context.fitted_scaler is not None
    assert context.scaler_feature_cols == ("sales", "planned_price")
    assert context.actual_cnn_audit
    assert {
        "protocol_raw_partition",
        "protocol_fitted_scaler",
        "protocol_actual_cnn_audit",
    }.isdisjoint(train.attrs)
