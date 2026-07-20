from __future__ import annotations

import json
from pathlib import Path

from src.constants import (
    SOURCE_HISTORY_CALENDAR,
    SOURCE_HISTORY_COMPLETENESS_POLICY,
    SOURCE_HISTORY_DAYS,
)
from src.protocols.experiment_protocol import get_experiment_protocol


KNN_ROOT = Path("configs/solidified/knn/Dataset4")
TARGETS = ("166_258", "166_432", "166_433", "166_313", "166_311")


def _payload(scenario: str) -> dict[str, object]:
    return json.loads(
        (KNN_ROOT / f"knn_{scenario}_info_sharing.json").read_text(encoding="utf-8")
    )


def test_d4_knn_authority_is_rebuilt_under_exact_180_day_contract() -> None:
    protocol = get_experiment_protocol("D4")
    expected_features = list(protocol.knn_feature_columns)
    for scenario in ("without", "with"):
        payload = _payload(scenario)
        assert payload["source_history_days"] == SOURCE_HISTORY_DAYS
        assert payload["source_history_expected_date_count"] == SOURCE_HISTORY_DAYS
        assert payload["source_history_start"] == "2024-07-19"
        assert payload["source_history_end"] == "2025-01-14"
        assert payload["source_history_calendar"] == SOURCE_HISTORY_CALENDAR
        assert payload["source_history_completeness_policy"] == SOURCE_HISTORY_COMPLETENESS_POLICY
        assert payload["source_history_inclusive_end"] is True
        assert payload["selection_authority"] == "shared_protocol"
        assert tuple(payload["results"]) == TARGETS

        for target_id in TARGETS:
            metadata = payload["selection_metadata"][target_id]
            assert metadata["knn_feature_columns"] == expected_features
            assert metadata["historical_feature_columns"] == expected_features
            assert metadata["forecast_excluded_columns"] == []
            assert metadata["feature_scope"] == "historical_observed"
            assert metadata["max_allowed_date_relation"] == "date<=origin"
            assert metadata["knn_observed_start"] == "2024-12-16"
            assert metadata["knn_observed_end"] == "2025-01-14"
            assert metadata["consumer_frame_rows"] == 3 * SOURCE_HISTORY_DAYS
            assert len(metadata["source_history_frame_digest"]) == 64
            assert len(metadata["consumer_frame_digest"]) == 64
            candidate_keys = {
                tuple(str(part) for part in key)
                for key in metadata["candidate_pool_digest_input"]["candidate_keys"]
            }
            selected_keys = {
                tuple(str(part) for part in row["source_key"])
                for row in metadata["selected_sources_runtime"]
            }
            assert ("729", "424") not in candidate_keys
            assert ("729", "424") not in selected_keys
            assert len(selected_keys) == 3

        result_keys = {
            str(row["source_entity"])
            for rows in payload["results"].values()
            for row in rows
        }
        assert "729_424" not in result_keys

    # The old with-sharing authority ranked 729_424 first for this target.
    # Retaining K=3 after its exclusion proves a fresh Top-K was selected.
    rebuilt_top_k = tuple(
        row["source_entity"] for row in _payload("with")["results"]["166_258"]
    )
    assert len(rebuilt_top_k) == 3
    assert rebuilt_top_k != ("729_424", "530_155", "356_242")
