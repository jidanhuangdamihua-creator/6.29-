"""Read-only Gate 1X D1-D6 real-input readiness preflight.

This command is intentionally unable to create a build, call a producer,
write parquet, create a manifest, or publish a deployment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import sys
from typing import Any, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.gate1_transformation import (  # noqa: E402
    CONTRACT_DIGEST,
    COMBINED_FORMAL_IDENTITY_DIGEST,
    Gate1Failure,
    SchemaRegistry,
    build_d6_calendar_view,
    canonical_digest,
    dataset_contract,
    load_formal_identity,
    slice_dataset_roles,
    select_source_history_candidates,
    stream_source_history_candidates,
)
from src.protocols.d2_source_calendarization import (  # noqa: E402
    D2_FROZEN_SOURCE_CANDIDATE_KEYS,
    D2_SOURCE_MISSING_DATES,
    slice_d2_source_frame,
    verify_d2_source_frame,
)
from src.protocols.formal_input_paths import (  # noqa: E402
    formal_dataset_identity,
    formal_input_paths,
    resolve_formal_dataset_paths,
)
from src.constants import SOURCE_HISTORY_DAYS  # noqa: E402
from src.protocols.candidate_pool import (  # noqa: E402
    build_candidate_pool_digest,
    build_consumer_fingerprint,
    build_source_pool_fingerprint,
)
from src.protocols.experiment_protocol import get_experiment_protocol  # noqa: E402


PARQUET_DIR = ROOT / "数据集" / "固化数据"
RAW_DIR = ROOT / "数据集" / "原始数据"
RAW_INPUTS = {
    1: (RAW_DIR / "Dataset 1/train.csv", RAW_DIR / "Dataset 1/test.csv"),
    2: (),
    3: (RAW_DIR / "Dataset 3 rossmann-store-sales/train.csv", RAW_DIR / "Dataset 3 rossmann-store-sales/test.csv"),
    4: (RAW_DIR / "Dataset 4叮咚数据集/train_sample_100.csv",),
    5: (RAW_DIR / "Dataset 5Favorita/train.csv", RAW_DIR / "Dataset 5Favorita/test.csv", RAW_DIR / "Dataset 5Favorita/oil.csv", RAW_DIR / "Dataset 5Favorita/holidays_events.csv"),
    6: (RAW_DIR / "Dataset 6m5-forecasting-accuracy/sales_train_validation.csv", RAW_DIR / "Dataset 6m5-forecasting-accuracy/sell_prices.csv", RAW_DIR / "Dataset 6m5-forecasting-accuracy/calendar.csv"),
}


def _file_record(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True, "size_bytes": int(path.stat().st_size)}


def _parquet_meta(path: Path) -> dict[str, object]:
    record = _file_record(path)
    if not record["exists"]:
        return record
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        record.update({"row_count": int(parquet.metadata.num_rows), "schema_fields": list(parquet.schema_arrow.names)})
    except Exception as exc:
        record.update({"metadata_error": f"{type(exc).__name__}: {exc}"})
    return record


def resolve_readiness_formal_input_identity(
    dataset_id: int,
    *,
    repository_root: Path = ROOT,
) -> dict[str, object]:
    return formal_dataset_identity(
        resolve_formal_dataset_paths(
            dataset_id,
            repository_root=repository_root,
        )
    )


def _target_frame(root: Path, dataset: int) -> pd.DataFrame:
    return pd.read_parquet(formal_input_paths(root, dataset)["target"])


def _source_frame(root: Path, dataset: int) -> pd.DataFrame | None:
    if dataset == 2:
        return pd.read_parquet(formal_input_paths(root, dataset)["source"])
    if dataset <= 3:
        return pd.read_parquet(formal_input_paths(root, dataset)["source"])
    return None


@lru_cache(maxsize=None)
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _load_json_file(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_d1_d2_knn_readiness(root: Path, dataset_id: int) -> dict[str, object]:
    """Read and validate the sealed D1/D2 selection manifests without producing them."""
    protocol = get_experiment_protocol(dataset_id)
    expected_features = list(protocol.knn_feature_columns)
    config_root = root / "configs" / "solidified" / "knn" / f"Dataset{dataset_id}"
    scenarios: dict[str, object] = {}
    for scenario in ("without", "with"):
        path = config_root / f"knn_{scenario}_info_sharing.json"
        if not path.is_file() or path.is_symlink():
            raise Gate1Failure("KNN_AUTHORITY_MISSING", str(path))
        payload = _load_json_file(path)
        actual_features = payload.get("knn_feature_columns", payload.get("feature_cols"))
        if actual_features != expected_features:
            raise Gate1Failure(
                "KNN_SCHEMA_MISMATCH",
                f"D{dataset_id}/{scenario}: expected {expected_features!r}, got {actual_features!r}",
            )
        selection_metadata = payload.get("selection_metadata")
        if not isinstance(selection_metadata, Mapping):
            raise Gate1Failure("KNN_METADATA_MISSING", f"D{dataset_id}/{scenario}")
        for target_id, metadata in selection_metadata.items():
            if not isinstance(metadata, Mapping):
                raise Gate1Failure("KNN_METADATA_INVALID", f"D{dataset_id}/{scenario}/{target_id}")
            if (
                metadata.get("knn_feature_columns") != expected_features
                or metadata.get("historical_feature_columns") != expected_features
                or metadata.get("forecast_excluded_columns")
                != (["promo"] if dataset_id == 2 else [])
                or metadata.get("feature_scope") != "historical_observed"
                or metadata.get("max_allowed_date_relation") != "date<=origin"
                or metadata.get("knn_observed_end") != protocol.observation_window().origin.isoformat()
            ):
                raise Gate1Failure(
                    "KNN_METADATA_MISMATCH",
                    f"D{dataset_id}/{scenario}/{target_id}",
                )
        scenarios[scenario] = {
            "path": str(path.relative_to(root)),
            "knn_feature_columns": expected_features,
            "selection_metadata_targets": sorted(str(key) for key in selection_metadata),
        }
    return {
        "knn_feature_columns": expected_features,
        "historical_feature_columns": expected_features,
        "forecast_excluded_columns": ["promo"] if dataset_id == 2 else [],
        "scenarios": scenarios,
    }


def _verify_d2_sealed_identity(root: Path, source_path: Path, target_path: Path) -> dict[str, object]:
    directory = source_path.parent
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise Gate1Failure("D2_SEALED_IDENTITY", f"missing D2 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "source": (source_path, 48654, "466391bb7e89067663d2d8f882834819896620c56bbbdc1959b81df938080ab2"),
        "target": (target_path, 1802, "fbfe0df5a5624504b00a8ea701ca7dd250ab46232d29f82473dcf4d0df712588"),
    }
    artifacts: dict[str, object] = {}
    expected_schema_fingerprints = manifest.get("schema_fingerprints", {})
    for role, (path, expected_rows, expected_hash) in expected.items():
        if not path.is_file() or path.is_symlink():
            raise Gate1Failure("D2_SEALED_IDENTITY", f"missing D2 {role}: {path}")
        meta = _parquet_meta(path)
        actual_hash = _sha256_file(path)
        artifact = manifest.get("artifacts", {}).get(role, {})
        if actual_hash != expected_hash or artifact.get("sha256") != expected_hash:
            raise Gate1Failure("D2_SEALED_IDENTITY", f"D2 {role} parquet hash mismatch")
        if meta.get("row_count") != expected_rows or artifact.get("row_count") != expected_rows:
            raise Gate1Failure("D2_SEALED_IDENTITY", f"D2 {role} row count mismatch")
        if artifact.get("path") != path.name or artifact.get("size_bytes") != path.stat().st_size:
            raise Gate1Failure("D2_SEALED_IDENTITY", f"D2 {role} manifest artifact mismatch")
        schema_path = directory / f"{role}_schema.json"
        if not schema_path.is_file() or schema_path.is_symlink():
            raise Gate1Failure("D2_SEALED_IDENTITY", f"missing D2 {role} schema sidecar")
        schema_sidecar = json.loads(schema_path.read_text(encoding="utf-8"))
        schema_frame = pd.read_parquet(path)
        schema_payload = {
            "column_order": list(schema_frame.columns),
            "columns": [
                {
                    "name": str(column),
                    "pandas_dtype": str(schema_frame[column].dtype),
                    "null_count": int(schema_frame[column].isna().sum()),
                }
                for column in schema_frame.columns
            ],
        }
        schema_digest = canonical_digest(schema_payload)
        if schema_sidecar.get("row_count") != expected_rows or schema_sidecar.get("parquet_sha256") != expected_hash:
            raise Gate1Failure("D2_SEALED_IDENTITY", f"D2 {role} schema identity mismatch")
        if schema_sidecar.get("schema_digest") != schema_digest or expected_schema_fingerprints.get(role) != schema_digest:
            raise Gate1Failure("D2_SEALED_IDENTITY", f"D2 {role} schema fingerprint mismatch")
        artifacts[role] = {
            "path": str(path),
            "row_count": expected_rows,
            "sha256": actual_hash,
            "size_bytes": int(path.stat().st_size),
            "schema_fields": meta.get("schema_fields", []),
            "schema_digest": schema_digest,
        }

    zero_dates: dict[str, dict[str, object]] = {}
    for role, (path, _expected_rows, _expected_hash) in expected.items():
        frame = pd.read_parquet(path, columns=["date", "sales"])
        dates = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        zero_dates[role] = {
            date_text: {
                "rows": int((dates == pd.Timestamp(date_text)).sum()),
                "sales_zero": bool(frame.loc[dates == pd.Timestamp(date_text), "sales"].eq(0).all()),
            }
            for date_text in D2_SOURCE_MISSING_DATES
        }
    expected_date_rows = {"source": 27, "target": 1}
    if any(
        item[date_text]["rows"] != expected_date_rows[role]
        or not item[date_text]["sales_zero"]
        for role, item in zero_dates.items()
        for date_text in D2_SOURCE_MISSING_DATES
    ):
        raise Gate1Failure("D2_SEALED_IDENTITY", "D2 frozen dates are missing or sales are not all zero")
    return {
        "status": "passed",
        "manifest_path": str(manifest_path),
        "manifest_version": manifest.get("manifest_version"),
        "artifacts": artifacts,
        "precalendarized": True,
        "runtime_calendarization": False,
        "verified_zero_sales_dates": list(D2_SOURCE_MISSING_DATES),
        "zero_date_evidence": zero_dates,
    }


def _source_post_origin(path: Path, origin: object) -> int | None:
    try:
        import pyarrow.parquet as pq

        cutoff = pd.Timestamp(origin)
        count = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=["date"], batch_size=500_000):
            dates = pd.to_datetime(batch.column("date").to_pandas(), errors="coerce").dt.normalize()
            count += int((dates > cutoff).sum())
        return count
    except Exception:
        return None


def _calendarize_target_counts(target: pd.DataFrame, spec: Any) -> tuple[dict[str, int], list[dict[str, object]], int]:
    target = target.copy()
    target["date"] = pd.to_datetime(target["date"], errors="coerce").dt.normalize()
    counts: dict[str, int] = {}
    repairs: list[dict[str, object]] = []
    expected = pd.date_range(spec.blind_start, spec.blind_end, freq="D")
    for key in spec.target_keys:
        mask = pd.Series(True, index=target.index)
        for field, value in zip(spec.key_fields, key):
            mask &= target[field].map(str).eq(str(value))
        dates = target.loc[mask & target["date"].between(expected.min(), expected.max()), "date"]
        missing = expected.difference(pd.DatetimeIndex(dates))
        rendered_key = "/".join(key)
        counts[rendered_key] = int(len(dates) + len(missing))
        repairs.extend({"key": list(key), "date": timestamp.strftime("%Y-%m-%d"), "sales": 0, "rule": "calendarize_missing_blind_day"} for timestamp in missing)
    return counts, repairs, int(sum(counts.values()))


def _d6_target_with_calendar(root: Path, target: pd.DataFrame) -> pd.DataFrame:
    calendar_path = root / "数据集" / "原始数据" / "Dataset 6m5-forecasting-accuracy/calendar.csv"
    if not calendar_path.is_file():
        return target
    calendar = pd.read_csv(calendar_path)
    view = build_d6_calendar_view(calendar, store_state="CA")
    target = target.copy()
    target["date"] = pd.to_datetime(target["date"], errors="raise").dt.normalize()
    view["date"] = pd.to_datetime(view["date"], errors="raise").dt.normalize()
    return target.drop(columns=[column for column in ("weekday", "wday", "wm_yr_wk", "snap") if column in target.columns], errors="ignore").merge(view, on="date", how="left", validate="many_to_one")


def _verify_d4_consumer_authority(
    root: Path,
    *,
    target_key: tuple[str, str],
    scenario: str,
    candidate_keys: list[tuple[str, str]],
) -> dict[str, object]:
    authority_path = (
        root
        / "configs"
        / "solidified"
        / "knn"
        / "Dataset4"
        / f"knn_{scenario}_info_sharing.json"
    )
    payload = _load_json_file(authority_path)
    target_id = "_".join(target_key)
    metadata = payload.get("selection_metadata", {}).get(target_id)
    if not isinstance(metadata, dict):
        raise Gate1Failure(
            "D4_STALE_SELECTION_AUTHORITY",
            f"missing D4 {scenario} metadata for {target_id}",
        )
    digest_input = metadata.get("candidate_pool_digest_input")
    if not isinstance(digest_input, dict):
        raise Gate1Failure(
            "D4_STALE_SELECTION_AUTHORITY",
            f"missing D4 candidate digest input for {target_id}",
        )
    authority_candidates = sorted(
        tuple(str(part) for part in key)
        for key in digest_input.get("candidate_keys", [])
    )
    expected_candidates = sorted(candidate_keys)
    if authority_candidates != expected_candidates:
        raise Gate1Failure(
            "D4_PATH_PARITY_MISMATCH",
            f"readiness/formal consumer candidate mismatch for {scenario}/{target_id}",
        )
    candidate_digest = build_candidate_pool_digest(**digest_input)
    if candidate_digest != metadata.get("candidate_pool_digest"):
        raise Gate1Failure(
            "D4_STALE_SELECTION_AUTHORITY",
            f"candidate digest mismatch for {scenario}/{target_id}",
        )
    source_pool_fingerprint = build_source_pool_fingerprint(
        protocol_version=metadata["protocol_version"],
        dataset_id="D4",
        scenario=scenario,
        target_key=target_key,
        group_cols=("store_id", "product_id"),
        candidate_keys=expected_candidates,
    )
    if source_pool_fingerprint != metadata.get("source_pool_fingerprint"):
        raise Gate1Failure(
            "D4_STALE_SELECTION_AUTHORITY",
            f"source pool fingerprint mismatch for {scenario}/{target_id}",
        )
    selected_sources = metadata.get("selected_sources_runtime")
    if not isinstance(selected_sources, list):
        raise Gate1Failure(
            "D4_STALE_SELECTION_AUTHORITY",
            f"missing selected consumer sources for {scenario}/{target_id}",
        )
    consumer_fingerprint = build_consumer_fingerprint(
        protocol_version=metadata["protocol_version"],
        dataset_id="D4",
        scenario=scenario,
        target_key=target_key,
        source_pool_fingerprint=source_pool_fingerprint,
        candidate_pool_digest=candidate_digest,
        selection_result_digest=metadata["selection_result_digest"],
        ordered_top_k=selected_sources,
    )
    if consumer_fingerprint != metadata.get("consumer_fingerprint"):
        raise Gate1Failure(
            "D4_STALE_SELECTION_AUTHORITY",
            f"consumer fingerprint mismatch for {scenario}/{target_id}",
        )
    target_store, target_product = target_key
    candidate_set = set(expected_candidates)
    cross_store_same_product_count = sum(
        key[0] != target_store and key[1] == target_product
        for key in candidate_set
    )
    proof = {
        "entity_key_fields": ["store_id", "product_id"],
        "exact_target_tuple": list(target_key),
        "exact_target_tuple_excluded": target_key not in candidate_set,
        "cross_store_same_product_available_count": cross_store_same_product_count,
        "cross_store_same_product_retained_count": cross_store_same_product_count,
        "same_store_other_product_retained_count": sum(
            key[0] == target_store and key[1] != target_product
            for key in candidate_set
        ),
        "cross_store_other_product_retained_count": sum(
            key[0] != target_store and key[1] != target_product
            for key in candidate_set
        ),
        "candidate_digest_verified": True,
        "consumer_fingerprint_verified": True,
    }
    if not proof["exact_target_tuple_excluded"]:
        raise Gate1Failure(
            "D4_PRODUCT_ONLY_CANDIDATE_EXCLUSION",
            f"exact D4 target entered candidate pool for {scenario}/{target_id}",
        )
    if scenario == "with" and not (
        proof["cross_store_same_product_retained_count"] > 0
        and proof["same_store_other_product_retained_count"] > 0
        and proof["cross_store_other_product_retained_count"] > 0
    ):
        raise Gate1Failure(
            "D4_READINESS_SEMANTIC_PROOF_MISSING",
            f"D4 exact-key behavior matrix is incomplete for {target_id}",
        )
    stored_proof = metadata.get("d4_validation_proof")
    if stored_proof != proof or metadata.get(
        "d4_validation_proof_digest"
    ) != canonical_digest(proof):
        raise Gate1Failure(
            "D4_READINESS_SEMANTIC_PROOF_MISSING",
            f"D4 validation proof mismatch for {scenario}/{target_id}",
        )
    manifest = payload.get("d4_manifest_identity")
    if not isinstance(manifest, dict):
        raise Gate1Failure(
            "D4_STALE_SELECTION_AUTHORITY",
            f"missing D4 manifest identity for {scenario}",
        )
    manifest_payload = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_identity_digest"
    }
    if (
        manifest.get("scenario") != scenario
        or manifest.get("entity_key_fields") != ["store_id", "product_id"]
        or manifest.get("source_parquet_sha256")
        != _sha256_file(
            resolve_formal_dataset_paths(4, repository_root=root).source_path
        )
        or manifest.get("target_parquet_sha256")
        != _sha256_file(
            resolve_formal_dataset_paths(4, repository_root=root).target_path
        )
        or manifest.get("manifest_identity_digest")
        != canonical_digest(manifest_payload)
    ):
        raise Gate1Failure(
            "D4_STALE_SELECTION_AUTHORITY",
            f"D4 manifest identity mismatch for {scenario}",
        )
    manifest_consumer = manifest.get("target_consumers", {}).get(target_id)
    expected_manifest_consumer = {
        "candidate_pool_digest": candidate_digest,
        "selection_result_digest": metadata["selection_result_digest"],
        "source_pool_fingerprint": source_pool_fingerprint,
        "consumer_fingerprint": consumer_fingerprint,
        "validation_proof_digest": metadata["d4_validation_proof_digest"],
    }
    if manifest_consumer != expected_manifest_consumer:
        raise Gate1Failure(
            "D4_STALE_SELECTION_AUTHORITY",
            f"D4 manifest consumer mismatch for {scenario}/{target_id}",
        )
    return {
        "status": "passed",
        "scenario": scenario,
        "target_key": list(target_key),
        "candidate_count": len(expected_candidates),
        "candidate_pool_digest": candidate_digest,
        "selection_result_digest": metadata["selection_result_digest"],
        "source_pool_fingerprint": source_pool_fingerprint,
        "consumer_fingerprint": consumer_fingerprint,
        "validation_proof_digest": metadata["d4_validation_proof_digest"],
        "manifest_identity_digest": manifest["manifest_identity_digest"],
        "exact_key_proof": proof,
    }


def _base_report(root: Path, parent_root: Path, old_root: Path, dataset: int, identity: Mapping[str, object] | None) -> dict[str, object]:
    spec = dataset_contract(dataset)
    protocol = get_experiment_protocol(dataset)
    input_paths = formal_input_paths(root, dataset)
    source_path = input_paths["source"]
    target_path = input_paths["target"]
    target_keys = [list(key) for key in spec.target_keys]
    return {
        "formal_identity": dict(identity or {"contract_digest": CONTRACT_DIGEST, "combined_formal_identity_digest": COMBINED_FORMAL_IDENTITY_DIGEST}),
        "dataset": f"D{dataset}",
        "formal_input": resolve_readiness_formal_input_identity(
            dataset,
            repository_root=root,
        ),
        "raw_inputs": [_file_record(path) for path in RAW_INPUTS[dataset]],
        "parent_inputs": {"source": _parquet_meta(parent_root / source_path.relative_to(root)), "target": _parquet_meta(parent_root / target_path.relative_to(root))},
        "old_sealed_inputs": {"source": _parquet_meta(old_root / source_path.relative_to(root)), "target": _parquet_meta(old_root / target_path.relative_to(root))},
        "source_entities": [],
        "target_entities": target_keys,
        "origin": spec.origin.isoformat(),
        "source_history_start": spec.source_history_start.isoformat(),
        "source_history_end": spec.source_history_end.isoformat(),
        "target_train_start": spec.target_train_start.isoformat(),
        "target_train_end": spec.target_train_end.isoformat(),
        "validation_start": spec.validation_start.isoformat(),
        "validation_end": spec.validation_end.isoformat(),
        "blind_start": spec.blind_start.isoformat(),
        "blind_end": spec.blind_end.isoformat(),
        "knn_start": spec.knn_start.isoformat(),
        "knn_end": spec.knn_end.isoformat(),
        "knn_feature_columns": list(protocol.knn_feature_columns),
        "historical_feature_columns": list(protocol.knn_feature_columns),
        "forecast_excluded_columns": ["promo"] if spec.dataset == "D2" else [],
        "knn_consumer_schema": {
            "knn_feature_columns": list(protocol.knn_feature_columns),
            "historical_feature_columns": list(protocol.knn_feature_columns),
            "forecast_excluded_columns": ["promo"] if spec.dataset == "D2" else [],
            "feature_scope": "historical_observed",
            "max_allowed_date_relation": "date<=origin",
        },
        "before_rows": {"source": _parquet_meta(source_path).get("row_count"), "target": _parquet_meta(target_path).get("row_count")},
        "after_slicing_rows": {},
        "expected_calendarized_rows": spec.expected_blind_rows,
        "missing_exact_keys": [],
        "duplicate_exact_keys": 0,
        "post_origin_history_rows": _source_post_origin(source_path, spec.origin),
        "pre_or_equal_origin_forecast_rows": 0,
        "schema_fields": {},
        "worker_safe_fields": list(__import__("src.protocols.gate1_transformation", fromlist=["SchemaRegistry"]).SchemaRegistry().allowed(spec.dataset, "worker")),
        "evaluator_truth_fields": [],
        "audit_fields": [],
        "field_exclusions": {},
        "cardinality": {},
        "proof_inputs_available": {"formal_identity": identity is not None, "raw_authority": False, "sealed_identity": False, "parent": False, "views": False, "calendarization": False, "field_specific_repairs": False},
        "status": "failed",
        "failure_code": None,
    }


def _dataset_report(root: Path, parent_root: Path, old_root: Path, dataset_id: int, identity: Mapping[str, object] | None = None) -> dict[str, object]:
    try:
        spec = dataset_contract(dataset_id)
        report = _base_report(root, parent_root, old_root, dataset_id, identity)
        input_paths = formal_input_paths(root, dataset_id)
        source_path = input_paths["source"]
        target_path = input_paths["target"]
        target = _target_frame(root, dataset_id)
        if dataset_id == 6:
            target = _d6_target_with_calendar(root, target)
        counts, repairs, calendarized_rows = _calendarize_target_counts(target, spec)
        report["missing_exact_keys"] = repairs
        report["duplicate_exact_keys"] = int(target.duplicated([*spec.key_fields, "date"]).sum())
        report["after_slicing_rows"] = {"target_observed": int((pd.to_datetime(target.date) <= pd.Timestamp(spec.origin)).sum()), "target_train": int(pd.to_datetime(target.date).between(pd.Timestamp(spec.target_train_start), pd.Timestamp(spec.target_train_end)).sum()), "validation": int(pd.to_datetime(target.date).between(pd.Timestamp(spec.validation_start), pd.Timestamp(spec.validation_end)).sum()), "blind": calendarized_rows}
        report["pre_or_equal_origin_forecast_rows"] = int(((pd.to_datetime(target.date) >= pd.Timestamp(spec.blind_start)) & (pd.to_datetime(target.date) <= pd.Timestamp(spec.origin))).sum())
        report["cardinality"] = {"target_keys": counts, "worker_safe_blind": calendarized_rows, "evaluator_truth": calendarized_rows, "expected_blind": spec.expected_blind_rows}
        report["evaluator_truth_fields"] = list(target.columns)
        report["audit_fields"] = list(target.columns)
        protocol = get_experiment_protocol(dataset_id)
        report["schema_fields"] = {
            "worker": report["worker_safe_fields"],
            "knn": [*spec.key_fields, "date", *protocol.knn_feature_columns],
            "forecast_consumer": [
                column
                for column in target.columns
                if column != "promo" or spec.dataset != "D2"
            ],
        }
        if dataset_id in {1, 2}:
            report["selection_authority"] = _verify_d1_d2_knn_readiness(root, dataset_id)
        report["field_exclusions"] = {"D2": ["PROMO", "promo", "Promo"], "D3": ["Open", "Customers", "Promo"], "D4": [*__import__("src.protocols.gate1_transformation", fromlist=["D4_AUDIT_ONLY"]).D4_AUDIT_ONLY], "D5": ["transactions", "week"], "D6": ["sales"]}.get(spec.dataset, [])
        report["proof_inputs_available"].update({"parent": True, "views": True, "calendarization": True, "field_specific_repairs": True})
        source = _source_frame(root, dataset_id)
        if source is not None:
            report["source_entities"] = [list(key) for key in sorted({tuple(str(value) for value in row) for row in source.loc[:, list(spec.key_fields)].drop_duplicates().itertuples(index=False, name=None)})]
            try:
                if dataset_id == 2:
                    sealed_identity = _verify_d2_sealed_identity(root, source_path, target_path)
                    source.attrs = source.attrs.copy()
                    source.attrs["split_role"] = "source"
                    source, calendar_proof = verify_d2_source_frame(
                        slice_d2_source_frame(source),
                        candidate_keys=D2_FROZEN_SOURCE_CANDIDATE_KEYS,
                    )
                    selected, source_proof = select_source_history_candidates(spec.dataset, source, "with-sharing", require_complete=True)
                    source_proof["authority"] = "sealed:D2/source.parquet"
                    source_proof["calendarization"] = {**calendar_proof.to_dict(), "status": "verified_precalendarized", "runtime_calendarization": False}
                    source_proof["sealed_identity"] = sealed_identity
                    report["sealed_identity"] = sealed_identity
                    report["proof_inputs_available"].update({"raw_authority": False, "sealed_identity": True, "calendarization": True})
                else:
                    selected, source_proof = select_source_history_candidates(spec.dataset, source, "with-sharing", require_complete=True)
                report["source_selection"] = source_proof
                report["proof_inputs_available"].update({"raw_authority": dataset_id != 2, "source_eligibility": True, "bounded_stream": True})
                report["source_entities"] = source_proof.get("candidate_keys", report["source_entities"])
            except Gate1Failure as exc:
                report["failure_code"] = exc.code
                report["error"] = str(exc)
        else:
            if dataset_id == 4:
                source_proof = stream_source_history_candidates(
                    spec.dataset,
                    source_path,
                    "with-sharing",
                    target_frame=target,
                    current_target_key=spec.target_keys[0],
                    source_history_days=SOURCE_HISTORY_DAYS,
                )
                universe = {
                    tuple(str(part) for part in key)
                    for key in source_proof[
                        "available_source_keys_before_target_exclusion"
                    ]
                }
                per_target: dict[str, object] = {}
                for raw_target_key in spec.target_keys:
                    target_key = tuple(str(part) for part in raw_target_key)
                    target_id = "/".join(target_key)
                    scenario_reports: dict[str, object] = {}
                    for scenario in ("without", "with"):
                        candidates = sorted(
                            key
                            for key in universe
                            if key != target_key
                            and (scenario == "with" or key[0] == target_key[0])
                        )
                        scenario_reports[scenario] = _verify_d4_consumer_authority(
                            root,
                            target_key=target_key,
                            scenario=scenario,
                            candidate_keys=candidates,
                        )
                    per_target[target_id] = scenario_reports
                report["source_selection"] = {
                    "status": "passed",
                    "entity_key_fields": list(spec.key_fields),
                    "exclusion_scope": "current_exact_target_tuple_only",
                    "targets": per_target,
                    "stream_proof": source_proof,
                }
                report["source_entities"] = [list(key) for key in sorted(universe)]
                report["post_origin_history_rows"] = source_proof[
                    "post_origin_history_rows"
                ]
            else:
                source_proof = stream_source_history_candidates(
                    spec.dataset,
                    source_path,
                    "with-sharing",
                    target_frame=target,
                    allow_approved_calendarization=(dataset_id == 5),
                )
                report["source_selection"] = source_proof
                report["source_entities"] = source_proof["complete_candidate_keys"]
                report["post_origin_history_rows"] = source_proof["post_origin_history_rows"]
            report["proof_inputs_available"].update({"raw_authority": True, "source_eligibility": True, "bounded_stream": True})
        if report["duplicate_exact_keys"]:
            report["failure_code"] = "DUPLICATE_EXACT_KEY_DATE"
        if report["pre_or_equal_origin_forecast_rows"]:
            report["failure_code"] = "FORECAST_ORIGIN"
        report["status"] = "passed" if report["failure_code"] is None and calendarized_rows == spec.expected_blind_rows else "failed"
        return report
    except Exception as exc:
        return {"dataset": f"D{dataset_id}", "status": "failed", "failure_code": getattr(exc, "code", "READINESS_ERROR"), "error": f"{type(exc).__name__}: {exc}"}


def run_readiness(
    *,
    root: Path = ROOT,
    parent_root: Path | None = None,
    old_sealed_root: Path | None = None,
    require_deployment: bool = False,
) -> dict[str, object]:
    root = Path(root).resolve()
    parent_root = Path(parent_root or root).resolve()
    old_sealed_root = Path(old_sealed_root or root).resolve()
    try:
        identity = load_formal_identity(root)
        identity_error = None
    except Gate1Failure as exc:
        identity = None
        identity_error = str(exc)
    datasets = [_dataset_report(root, parent_root, old_sealed_root, index, identity) for index in range(1, 7)]
    if identity_error:
        for item in datasets:
            item["status"] = "failed"
            item["failure_code"] = "FORMAL_IDENTITY"
            item["error"] = identity_error
    deployment_error: tuple[str, str] | None = None
    if require_deployment and not identity_error:
        try:
            from src.protocols.formal_deployment_manifest import (
                canonical_json_bytes,
                sha256_bytes,
                validate_deployment_manifest,
            )

            preflight = validate_deployment_manifest(root)
            manifest = preflight["manifest"]
            proofs = preflight["proofs"]
            for item in datasets:
                dataset = str(item["dataset"])
                entry = manifest["datasets"][dataset]
                proof = proofs[dataset]
                base_digest = sha256_bytes(canonical_json_bytes(item))
                if base_digest != proof["readiness_proof_digest"]:
                    item["status"] = "failed"
                    item["failure_code"] = "READINESS_PROOF_MISMATCH"
                item.update(
                    {
                        "source_path": entry["source"]["path"],
                        "target_path": entry["target"]["path"],
                        "source_sha256": entry["source"]["sha256"],
                        "target_sha256": entry["target"]["sha256"],
                        "source_size": entry["source"]["size_bytes"],
                        "target_size": entry["target"]["size_bytes"],
                        "source_schema_digest": entry["source_schema_digest"],
                        "target_schema_digest": entry["target_schema_digest"],
                        "consumer_fingerprint": entry["consumer_fingerprint"],
                        "proof_digest": proof["proof_identity_sha256"],
                        "formal_identity_match": proof["formal_identity"] == manifest["formal_identity"],
                        "manifest_identity_match": proof["consumer_fingerprint"] == entry["consumer_fingerprint"],
                    }
                )
                if not item["formal_identity_match"] or not item["manifest_identity_match"]:
                    item["status"] = "failed"
                    item["failure_code"] = "MANIFEST_IDENTITY_MISMATCH"
        except Exception as exc:
            deployment_error = (
                str(getattr(exc, "code", "FINAL_PREFLIGHT_NOT_READY")),
                f"{type(exc).__name__}: {exc}",
            )
            for item in datasets:
                item["status"] = "failed"
                item["failure_code"] = deployment_error[0]
                item["error"] = deployment_error[1]
    failures = [item for item in datasets if item.get("status") != "passed"]
    ready = len(datasets) - len(failures)
    status = "passed" if not failures else "failed"
    return {
        "status": status,
        "preflight_status": "ready" if status == "passed" else "blocked",
        "datasets_ready": ready,
        "datasets_total": 6,
        "failure_code": None if not failures else failures[0].get("failure_code"),
        "formal_identity": identity or {"status": "failed", "error": identity_error},
        "datasets": datasets,
        "read_only": True,
        "writes_performed": False,
        "producer_calls_performed": 0,
        "private_build_created": False,
        "deployment_created": False,
        "manifest_candidate_created": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 1X real-input readiness; read-only")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--parent-root", type=Path, default=None)
    parser.add_argument("--old-sealed-root", type=Path, default=None)
    parser.add_argument("--read-only", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_readiness(
        root=args.root,
        parent_root=args.parent_root,
        old_sealed_root=args.old_sealed_root,
        require_deployment=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
