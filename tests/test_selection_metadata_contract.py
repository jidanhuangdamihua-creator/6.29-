from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.protocols import formal_deployment_manifest as deployment
from src.protocols.candidate_pool import build_candidate_pool_digest
from src.protocols.experiment_protocol import get_experiment_protocol
from src.protocols.selection_metadata import build_selection_metadata_contract
from tools.operations.gate1x_real_input_readiness import _verify_runtime_knn_metadata


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FIELDS = (
    "knn_feature_columns",
    "historical_feature_columns",
    "forecast_excluded_columns",
    "feature_scope",
    "max_allowed_date_relation",
    "knn_observed_start",
    "knn_observed_end",
)


@pytest.mark.parametrize(
    ("dataset_id", "observed_start"),
    (
        ("D1", None),
        ("D2", None),
        ("D4", "2024-12-16"),
        ("D5", "2017-01-17"),
        ("D6", "2015-10-26"),
    ),
)
def test_selection_metadata_contract_is_derived_from_protocol(
    dataset_id: str,
    observed_start: str | None,
) -> None:
    protocol = get_experiment_protocol(dataset_id)
    metadata = build_selection_metadata_contract(
        protocol,
        observed_start=observed_start,
    )
    window = (
        protocol.observation_window()
        if observed_start is None
        else protocol.observation_window(observed_start)
    )

    assert set(CANONICAL_FIELDS).issubset(metadata)
    assert metadata["knn_feature_columns"] == list(protocol.knn_feature_columns)
    assert metadata["historical_feature_columns"] == list(protocol.knn_feature_columns)
    assert metadata["forecast_excluded_columns"] == (["promo"] if dataset_id == "D2" else [])
    assert metadata["feature_scope"] == "historical_observed"
    assert metadata["max_allowed_date_relation"] == "date<=origin"
    assert metadata["knn_observed_start"] == window.knn_observed_start.isoformat()
    assert metadata["knn_observed_end"] == window.knn_observed_end.isoformat()


@pytest.mark.parametrize("missing_field", CANONICAL_FIELDS)
def test_d4_d6_manifest_rejects_missing_canonical_metadata(
    tmp_path: Path,
    missing_field: str,
) -> None:
    source = ROOT / "configs" / "solidified" / "knn" / "Dataset5" / "knn_with_info_sharing.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    target_id = sorted(payload["selection_metadata"])[0]
    payload["selection_metadata"][target_id].pop(missing_field, None)
    path = tmp_path / source.name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment._verify_d4_d6_knn_payload(path, dataset_id=5)
    assert captured.value.code == "D4_D6_KNN_METADATA_MISMATCH"


def test_d4_d6_manifest_rejects_knn_end_that_is_not_origin(tmp_path: Path) -> None:
    source = ROOT / "configs" / "solidified" / "knn" / "Dataset5" / "knn_with_info_sharing.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    target_id = sorted(payload["selection_metadata"])[0]
    payload["selection_metadata"][target_id]["knn_observed_end"] = "2017-02-14"
    path = tmp_path / source.name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment._verify_d4_d6_knn_payload(path, dataset_id=5)
    assert captured.value.code == "D4_D6_KNN_METADATA_MISMATCH"


def test_d4_d6_manifest_rejects_old_300_day_identity(tmp_path: Path) -> None:
    source = ROOT / "configs" / "solidified" / "knn" / "Dataset5" / "knn_with_info_sharing.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["source_history_days"] = 300
    payload["source_history_expected_date_count"] = 300
    path = tmp_path / source.name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment._verify_d4_d6_knn_payload(path, dataset_id=5)
    assert captured.value.code == "D4_D6_SOURCE_HISTORY_IDENTITY_MISMATCH"


def test_d4_d6_manifest_rejects_frame_cardinality_that_is_not_180_days(tmp_path: Path) -> None:
    source = ROOT / "configs" / "solidified" / "knn" / "Dataset5" / "knn_with_info_sharing.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    target_id = sorted(payload["selection_metadata"])[0]
    protocol = get_experiment_protocol("D5")
    observed_start = payload["selection_metadata"][target_id].get(
        "target_observed_start", "2017-01-17"
    )
    payload["selection_metadata"][target_id].update(
        build_selection_metadata_contract(protocol, observed_start=observed_start)
    )
    metadata = payload["selection_metadata"][target_id]
    metadata["selection_authority"] = "shared_protocol"
    metadata["protocol_version"] = protocol.protocol_version
    target_key = [part for part in target_id.split("_")]
    candidate_keys = [row["source_key"] for row in metadata["selected_sources_runtime"]]
    digest_input = {
        "protocol_version": payload["protocol_version"],
        "dataset_id": "D5",
        "scenario": "with",
        "target_key": target_key,
        "group_cols": payload["group_cols"],
        "candidate_keys": candidate_keys,
        "observed_start": metadata["knn_observed_start"],
        "observed_end": metadata["knn_observed_end"],
        "feature_cols": metadata["feature_cols"],
        "source_history_days": metadata["source_history_days"],
        "source_history_start": str(metadata["source_history_start"])[:10],
        "source_history_end": str(metadata["source_history_end"])[:10],
        "source_history_completeness_policy": metadata["source_history_completeness_policy"],
        "source_history_frame_digest": metadata["source_history_frame_digest"],
    }
    metadata["candidate_pool_digest_input"] = digest_input
    metadata["candidate_pool_digest"] = build_candidate_pool_digest(**digest_input)
    metadata["consumer_frame_rows"] = 3 * 300
    path = tmp_path / source.name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(deployment.DeploymentManifestError) as captured:
        deployment._verify_d4_d6_knn_payload(path, dataset_id=5)
    assert captured.value.code == "D4_D6_CONSUMER_FRAME_CARDINALITY"


def test_readiness_does_not_parse_inclusive_end_boolean_as_date() -> None:
    source = ROOT / "configs" / "solidified" / "knn" / "Dataset5" / "knn_with_info_sharing.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    target_id = sorted(payload["selection_metadata"])[0]
    metadata = dict(payload["selection_metadata"][target_id])
    metadata["selected_count"] = len(metadata["selected_sources_runtime"])
    candidate_keys = {
        tuple(str(part) for part in key)
        for key in metadata["candidate_pool_digest_input"]["candidate_keys"]
    }

    proof = _verify_runtime_knn_metadata(
        dataset_id=5,
        scenario="with",
        target_id=target_id,
        metadata=metadata,
        payload=payload,
        expected_candidate_keys=candidate_keys,
    )

    assert proof["consumer_frame_rows"] == 3 * 180
