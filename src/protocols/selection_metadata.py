"""Protocol-owned canonical metadata for persisted source selections."""

from __future__ import annotations

from typing import Any

from .experiment_protocol import ExperimentProtocol, ObservationWindow, ProtocolViolation


CANONICAL_SELECTION_METADATA_FIELDS = (
    "knn_feature_columns",
    "historical_feature_columns",
    "forecast_excluded_columns",
    "feature_scope",
    "max_allowed_date_relation",
    "knn_observed_start",
    "knn_observed_end",
)


def build_selection_metadata_contract(
    protocol: ExperimentProtocol,
    *,
    observed_start: object | None = None,
    window: ObservationWindow | None = None,
) -> dict[str, Any]:
    """Build canonical historical-observed selection metadata from protocol authority."""
    if window is None:
        if protocol.dataset_id in {"D1", "D2"}:
            window = protocol.observation_window()
        else:
            if observed_start is None:
                raise ProtocolViolation(
                    f"{protocol.dataset_id} selection metadata requires observed_start"
                )
            window = protocol.observation_window(observed_start)
    elif observed_start is not None:
        expected = protocol.observation_window(observed_start)
        if expected != window:
            raise ProtocolViolation(
                f"{protocol.dataset_id} selection metadata window disagrees with protocol"
            )

    expected_features = list(protocol.knn_feature_columns)
    return {
        "knn_feature_columns": expected_features,
        "historical_feature_columns": list(expected_features),
        "forecast_excluded_columns": ["promo"] if protocol.dataset_id == "D2" else [],
        "feature_scope": "historical_observed",
        "max_allowed_date_relation": "date<=origin",
        "knn_observed_start": window.knn_observed_start.isoformat(),
        "knn_observed_end": window.knn_observed_end.isoformat(),
    }


__all__ = [
    "CANONICAL_SELECTION_METADATA_FIELDS",
    "build_selection_metadata_contract",
]
