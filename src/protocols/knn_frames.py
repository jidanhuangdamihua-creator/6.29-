"""Construct and retrieve the date-bounded KNN observation frames."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .experiment_protocol import ObservationWindow, ProtocolViolation
from .gate1_transformation import normalized_frame_digest
from src.utils.dataframe_attrs import (
    lightweight_frame_attrs,
    temporarily_detached_attrs,
)


_CONFIGURED_FRAME_ATTR = "protocol_knn_observed_frame"


def _normalized_dates(frame: pd.DataFrame, *, role: str) -> pd.Series:
    if "date" not in frame.columns:
        raise ProtocolViolation(f"{role} frame requires date column")
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ProtocolViolation(f"{role} frame contains invalid dates")
    return dates


def canonical_knn_frame_digest(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    feature_cols: Sequence[str] | None = None,
    ignore_columns: Sequence[str] = (),
) -> str:
    """Digest the actual KNN frame after deterministic key/date ordering."""
    with temporarily_detached_attrs(frame):
        dates = _normalized_dates(frame, role="KNN")
        ordered = frame.copy()
    ordered["date"] = dates
    if ignore_columns:
        ordered = ordered.drop(columns=list(ignore_columns), errors="ignore")
    if feature_cols is not None:
        normalized_features = tuple(str(column) for column in feature_cols)
        missing = [column for column in normalized_features if column not in ordered.columns]
        if missing:
            raise ProtocolViolation(
                f"KNN frame is missing declared feature columns: {missing!r}"
            )
        ordered = ordered.loc[:, [*group_cols, "date", *normalized_features]].copy()
    sort_cols = [column for column in (*group_cols, "date") if column in ordered.columns]
    if not sort_cols:
        raise ProtocolViolation("KNN frame digest requires group or date columns")
    ordered = ordered.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return normalized_frame_digest(ordered)


def build_observed_knn_frame(
    frame: pd.DataFrame,
    *,
    window: ObservationWindow,
    role: str,
    group_cols: Sequence[str],
    feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return the inclusive observed copy used by KNN, never the full model frame."""
    with temporarily_detached_attrs(frame):
        parsed_dates = _normalized_dates(frame, role=role)
    observed_mask = parsed_dates.between(
        pd.Timestamp(window.knn_observed_start),
        pd.Timestamp(window.knn_observed_end),
        inclusive="both",
    )
    with temporarily_detached_attrs(frame):
        observed = frame.loc[observed_mask].copy()
    if observed.empty:
        raise ProtocolViolation(f"{role} KNN observed frame is empty")
    if feature_cols is not None:
        normalized_features = tuple(str(column) for column in feature_cols)
        missing = [column for column in normalized_features if column not in observed.columns]
        if missing:
            raise ProtocolViolation(
                f"{role} KNN observed frame is missing declared feature columns: {missing!r}"
            )
        for column in normalized_features:
            numeric = pd.to_numeric(observed[column], errors="coerce")
            if numeric.isna().any():
                raise ProtocolViolation(
                    f"{role} KNN observed feature {column!r} contains non-numeric values"
                )
    observed["date"] = parsed_dates.loc[observed_mask].to_numpy()
    observed.attrs = {
        key: value
        for key, value in lightweight_frame_attrs(frame.attrs).items()
        if key != _CONFIGURED_FRAME_ATTR
    }
    observed.attrs.update(
        {
            "knn_frame_role": str(role),
            "knn_observed_start": window.knn_observed_start.isoformat(),
            "knn_observed_end": window.knn_observed_end.isoformat(),
            "knn_observed_days": window.observed_days,
            "knn_boundary": "inclusive",
            "knn_feature_columns": list(feature_cols or ()),
            "feature_scope": "historical_observed",
            "max_allowed_date_relation": "date<=origin",
            "knn_frame_min_date": observed["date"].min().strftime("%Y-%m-%d"),
            "knn_frame_max_date": observed["date"].max().strftime("%Y-%m-%d"),
            "knn_frame_digest": canonical_knn_frame_digest(
                observed,
                group_cols=group_cols,
                feature_cols=(
                    None
                    if tuple(feature_cols or ()) == ("sales",)
                    else feature_cols
                ),
                ignore_columns=("promo",) if tuple(feature_cols or ()) == ("sales",) else (),
            ),
        }
    )
    return observed


def get_configured_knn_frame(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    """Retrieve a configured observed frame and fail closed when absent."""
    configured = frame.attrs.get(_CONFIGURED_FRAME_ATTR)
    if not isinstance(configured, pd.DataFrame):
        raise ProtocolViolation(
            f"{role} frame is missing configured KNN observed frame"
        )
    observed = configured.copy()
    observed.attrs = configured.attrs.copy()
    return observed


__all__ = [
    "build_observed_knn_frame",
    "canonical_knn_frame_digest",
    "get_configured_knn_frame",
]
