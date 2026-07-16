#!/usr/bin/env python3
"""Read-only D1-D6 protocol preflight; it never trains or rewrites datasets."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.candidate_pool import (
    InsufficientCandidatePoolError,
    PreparedDailySequencePool,
    prepare_daily_sequence_pool,
)
from src.protocols.experiment_protocol import ProtocolViolation
from src.protocols.runner_adapter import configure_protocol_frames, validate_predictor_safe_view
from src.protocols.adopt_validation import (
    VALIDATION_POLICY_DIGEST,
    VALIDATION_POLICY_VERSION,
)
from src.protocols.artifact_schemas import (
    artifact_schema_registry_digest,
    get_artifact_schema_registry,
    get_worker_manifest_schema,
)
from src.protocols.feature_schema import (
    audit_future_known_lineage,
    get_future_known_lineage,
    get_knn_schema,
    get_predictor_schema,
)
from src.protocols.sealing_protocol import (
    SEALING_PROTOCOL_VERSION,
    get_source_pretrain_window,
    get_target_window,
)
from src.source_selection.source_selector import SourceSelector


PARQUET_DIR = ROOT / "数据集" / "固化数据"
SEALED_ROOT = PARQUET_DIR / "d1_d6_sealed_v1"
D1D2_PROTOCOL_PARQUET_DIR = (
    ROOT / "数据集" / "派生数据" / "d1d2_protocol_v1"
)


@dataclass(frozen=True)
class DatasetSealState:
    dataset_id: str
    state: str
    failure_codes: tuple[str, ...]
    report_path: str
    report_sha256: str | None
    validation_policy_digest: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _window_payload(dataset_id: int) -> dict[str, dict[str, str]]:
    target = get_target_window(dataset_id)
    source = get_source_pretrain_window(dataset_id)
    return {
        "target": {
            "train_start": target.train_start.isoformat(),
            "train_end": target.train_end.isoformat(),
            "validation_start": target.validation_start.isoformat(),
            "validation_end": target.validation_end.isoformat(),
            "blind_start": target.blind_start.isoformat(),
            "blind_end": target.blind_end.isoformat(),
        },
        "source": {
            "pretrain_start": source.pretrain_start.isoformat(),
            "pretrain_end": source.pretrain_end.isoformat(),
            "knn_start": source.knn_start.isoformat(),
            "knn_end": source.knn_end.isoformat(),
        },
    }


def validate_dataset_seal(
    sealed_root: Path,
    dataset_id: int,
    *,
    required_policy_digest: str = VALIDATION_POLICY_DIGEST,
) -> DatasetSealState:
    """Validate one immutable dataset proof without reading model inputs."""

    directory = Path(sealed_root) / f"dataset{int(dataset_id)}"
    report_path = directory / "validation_report.json"
    failures: list[str] = []
    report_digest: str | None = None
    policy_digest: str | None = None
    required = {
        "manifest.json", "validation_report.json", "source.parquet", "target.parquet",
        "predictor_schema.json", "knn_schema.json", "source_sales_canonicalization.json",
        "calendarization_audit.json", "source_schema.json", "target_schema.json", "provenance.json",
    }
    if dataset_id >= 3:
        required.add("adopt_validation_report.json")
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        failures.append("SEALED_ARTIFACT_MISSING")
    try:
        report = _read_object(report_path)
        report_digest = _sha256(report_path)
        policy_digest = str(report.get("validation_policy_digest") or "") or None
        if report.get("status") != "validated" or report.get("failure_reasons"):
            failures.extend(str(item) for item in report.get("failure_reasons", ()))
            failures.append("DATASET_VALIDATION_FAILED")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        report = {}
        failures.append("VALIDATION_REPORT_INVALID")

    try:
        manifest = _read_object(directory / "manifest.json")
        if manifest.get("dataset_id") != f"D{dataset_id}":
            failures.append("DATASET_IDENTITY_MISMATCH")
        expected_provenance = "raw_rebuilt" if dataset_id <= 2 else "adopted_solidified"
        if manifest.get("provenance_level") != expected_provenance:
            failures.append("PROVENANCE_LEVEL_MISMATCH")
        if manifest.get("windows") != _window_payload(dataset_id):
            failures.append("WINDOW_CONTRACT_MISMATCH")
        if manifest.get("validation_policy_version") != VALIDATION_POLICY_VERSION:
            failures.append("VALIDATION_POLICY_VERSION_MISMATCH")
        if manifest.get("validation_policy_digest") != required_policy_digest:
            failures.append("VALIDATION_POLICY_DIGEST_MISMATCH")
        if policy_digest != required_policy_digest:
            failures.append("VALIDATION_POLICY_DIGEST_MISMATCH")
        if dataset_id >= 3:
            adopt_report = _read_object(directory / "adopt_validation_report.json")
            if (
                adopt_report.get("status") != "validated"
                or adopt_report.get("passed") is not True
                or adopt_report.get("failure_reasons")
                or adopt_report.get("validation_policy_digest") != required_policy_digest
            ):
                failures.append("ADOPT_VALIDATION_REPORT_INVALID")
        predictor = _read_object(directory / "predictor_schema.json")
        knn = _read_object(directory / "knn_schema.json")
        if predictor != get_predictor_schema(dataset_id).descriptor():
            failures.append("PREDICTOR_SCHEMA_MISMATCH")
        if knn != get_knn_schema(dataset_id).descriptor():
            failures.append("KNN_SCHEMA_MISMATCH")
        repair = _read_object(directory / "source_sales_canonicalization.json")
        if repair.get("version") != "source_sales_canonicalization/v1":
            failures.append("SOURCE_REPAIR_REPORT_INVALID")
        if manifest.get("source_sales_repair_mask_sha256") != repair.get("repair_mask_sha256"):
            failures.append("SOURCE_REPAIR_IDENTITY_MISMATCH")
        for artifact in manifest.get("artifacts", {}).values():
            artifact_path = directory / str(artifact["path"])
            if artifact_path.stat().st_size != int(artifact["size_bytes"]):
                failures.append("SEALED_ARTIFACT_SIZE_MISMATCH")
            if _sha256(artifact_path).removeprefix("sha256:") != artifact["sha256"]:
                failures.append("SEALED_ARTIFACT_HASH_MISMATCH")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        failures.append("SEALED_MANIFEST_INVALID")

    return DatasetSealState(
        dataset_id=f"D{dataset_id}",
        state="sealed" if not failures else "adopt_validation_failed",
        failure_codes=tuple(dict.fromkeys(failures)),
        report_path=str(report_path),
        report_sha256=report_digest,
        validation_policy_digest=policy_digest,
    )


def validate_formal_entry_preflight(
    sealed_root: Path = SEALED_ROOT,
    *,
    run_id: str = "pre-attempt",
    required_policy_digest: str = VALIDATION_POLICY_DIGEST,
) -> dict[str, Any]:
    """Return the layered six-dataset gate. This function never creates an attempt."""

    states = tuple(
        validate_dataset_seal(
            Path(sealed_root), dataset_id, required_policy_digest=required_policy_digest
        )
        for dataset_id in range(1, 7)
    )
    worker_fields = set(get_worker_manifest_schema().field_names)
    forbidden_worker_fields = {
        "y_true", "truth_key", "evaluator_path", "evaluator_capability_id"
    }
    d2_fingerprint_valid = False
    if all(state.state == "sealed" for state in states):
        try:
            d2_dir = Path(sealed_root) / "dataset2"
            source = pd.read_parquet(d2_dir / "source.parquet")
            target = pd.read_parquet(d2_dir / "target.parquet")
            source, target = configure_protocol_frames(
                source,
                target,
                dataset_id="D2",
                scenario="without",
                group_cols=("brand_id", "item_id"),
                observed_start="2018-06-01",
            )
            selection = SourceSelector().select_top_k_sources(
                target,
                source,
                feature_cols=("sales", "year", "month", "week", "day"),
                k=3,
                group_cols=("brand_id", "item_id"),
            )
            d2_fingerprint_valid = (
                selection["meta"]["feature_cols"] == ["sales", "promo"]
                and [tuple(row["source_key"]) for row in selection["sources"]]
                == [("1", "4"), ("1", "6"), ("1", "8")]
                and all(
                    math.isclose(float(row["distance"]), expected, abs_tol=0.02)
                    for row, expected in zip(
                        selection["sources"], (24.98, 26.85, 26.85)
                    )
                )
            )
        except Exception:
            d2_fingerprint_valid = False
    checks = {
        "six_dataset_seals": all(state.state == "sealed" for state in states),
        "single_validation_policy": all(
            state.validation_policy_digest == required_policy_digest for state in states
        ),
        "artifact_schema_registry": bool(
            artifact_schema_registry_digest()
            and get_artifact_schema_registry().schema_names
        ),
        "cache_isolation_contract": not bool(worker_fields & forbidden_worker_fields),
        "d2_knn_fingerprint_contract": d2_fingerprint_valid,
        "future_known_lineage": all(
            audit_future_known_lineage(
                get_predictor_schema(dataset_id),
                get_future_known_lineage(dataset_id),
                cutoff=get_target_window(dataset_id).validation_end,
            ).valid
            for dataset_id in range(1, 7)
        ),
    }
    failure_codes = [
        f"{state.dataset_id}:{code}"
        for state in states
        for code in state.failure_codes
    ]
    failure_codes.extend(
        f"GLOBAL:{name}" for name, passed in checks.items() if not passed
    )
    return {
        "run_id": str(run_id),
        "protocol_version": SEALING_PROTOCOL_VERSION,
        "validation_policy_version": VALIDATION_POLICY_VERSION,
        "validation_policy_digest": required_policy_digest,
        "schema_registry_version": "artifact_schema_registry_v1",
        "schema_digests": {
            "artifact_registry": artifact_schema_registry_digest(),
            **{
                f"D{dataset_id}_predictor": get_predictor_schema(dataset_id).digest
                for dataset_id in range(1, 7)
            },
            **{
                f"D{dataset_id}_knn": get_knn_schema(dataset_id).digest
                for dataset_id in range(1, 7)
            },
        },
        "dataset_states": [asdict(state) for state in states],
        "checks": checks,
        "failure_codes": failure_codes,
        "status": "ready" if not failure_codes else "blocked",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def resolve_parquet_dir(
    dataset_id: int,
    explicit_dir: Optional[Path] = None,
) -> Path:
    """Resolve the authoritative parquet directory for protocol preflight."""
    if explicit_dir is not None:
        return Path(explicit_dir)
    if dataset_id in (1, 2):
        return D1D2_PROTOCOL_PARQUET_DIR
    return PARQUET_DIR


DATASET_CONFIG = {
    1: {"group_cols": ("store_id", "item_id"), "observed_start": "2017-06-05"},
    2: {"group_cols": ("brand_id", "item_id"), "observed_start": "2018-06-05"},
    3: {"group_cols": ("store_id",), "observed_start": "2015-01-03"},
    4: {
        "group_cols": ("store_id", "product_id"),
        "grouping_col": "second_category_id",
        "observed_start": "2024-12-16",
    },
    5: {
        "group_cols": ("store_nbr", "item_nbr"),
        "grouping_col": "family",
        "observed_start": "2017-01-17",
    },
    6: {
        "group_cols": ("store_id", "item_id"),
        "grouping_col": "dept_id",
        "observed_start": "2015-10-26",
    },
}


def validate_protocol_frames(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    dataset_id: object,
    scenario: str,
    group_cols: Sequence[str],
    observed_start: object,
    k: int,
    grouping_col: str | None = None,
    prepared_pool: PreparedDailySequencePool | None = None,
    exclusion_sample_limit: int = 20,
) -> dict[str, Any]:
    try:
        source, target = configure_protocol_frames(
            source_df,
            target_df,
            dataset_id=dataset_id,
            scenario=scenario,
            group_cols=group_cols,
            grouping_col=grouping_col,
            observed_start=observed_start,
            prepared_pool=prepared_pool,
        )
        selection = SourceSelector().select_top_k_sources(
            target,
            source,
            feature_cols=("sales",),
            k=int(k),
            group_cols=tuple(group_cols),
            weight_mode="inverse_distance",
        )
        meta = selection["meta"]
        exclusion_summary = summarize_candidate_exclusions(
            meta["source_skip_diagnostics"],
            sample_limit=exclusion_sample_limit,
        )
        ordered_top_k = [
            {
                "source_rank": item["source_rank"],
                "source_key": item["source_key"],
                "distance": item["distance"],
                "weight": item["weight"],
                "tie_group": item["tie_group"],
            }
            for item in selection["sources"]
        ]
        return {
            "status": "passed",
            "dataset_id": target.attrs["protocol_dataset_id"],
            "scenario": target.attrs["protocol_scenario"],
            "target_key": target.attrs["protocol_target_key"],
            "candidate_count": len(target.attrs["protocol_candidate_keys"]),
            "candidate_pool_digest": meta["candidate_pool_digest"],
            "candidate_pool_digest_input_summary": summarize_candidate_digest_input(
                meta["candidate_pool_digest_input"],
                sample_limit=exclusion_sample_limit,
            ),
            "selection_result_digest": meta["selection_result_digest"],
            "candidate_count_total": meta["candidate_source_count"],
            "candidate_count_valid": meta["valid_source_count"],
            **exclusion_summary,
            "requested_k": meta["requested_k"],
            "effective_k": meta["effective_k"],
            "ordered_top_k": ordered_top_k,
            "cnn_provenance_validated": meta["cnn_provenance_validated"],
            "knn_observed_start": target.attrs["knn_observed_start"],
            "knn_observed_end": target.attrs["knn_observed_end"],
            "error": "",
        }
    except Exception as exc:  # preflight must return an auditable failure report
        exclusions = (
            exc.exclusions
            if isinstance(exc, InsufficientCandidatePoolError)
            else ()
        )
        return {
            "status": "failed",
            "dataset_id": str(dataset_id),
            "scenario": str(scenario),
            **summarize_candidate_exclusions(
                exclusions,
                sample_limit=exclusion_sample_limit,
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }


def validate_predictor_preflight(
    frame: pd.DataFrame,
    *,
    dataset_id: object,
    passthrough_cols: Sequence[str] = ("date",),
) -> dict[str, Any]:
    """Validate the model view without changing KNN candidate eligibility."""
    try:
        validate_predictor_safe_view(
            frame,
            dataset_id=dataset_id,
            passthrough_cols=passthrough_cols,
        )
        schema = get_predictor_schema(dataset_id)
        return {
            "status": "passed",
            "failure_code": "",
            "predictor_schema_digest": schema.digest,
            "predictor_fields": list(schema.ordered_names),
            "error": "",
        }
    except ProtocolViolation as exc:
        message = str(exc)
        return {
            "status": "failed",
            "failure_code": message.split(":", 1)[0],
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize_candidate_exclusions(
    exclusions: Sequence[dict[str, Any] | Any],
    *,
    sample_limit: int = 20,
) -> dict[str, Any]:
    if sample_limit < 0:
        raise ValueError("candidate exclusion sample limit must be non-negative")
    normalized = [dict(item) for item in exclusions]
    reason_counts = Counter(str(item.get("reason", "unknown")) for item in normalized)
    return {
        "candidate_exclusion_count": len(normalized),
        "candidate_exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "candidate_exclusion_samples": normalized[:sample_limit],
        "candidate_exclusions_truncated": len(normalized) > sample_limit,
    }


def summarize_candidate_digest_input(
    digest_input: dict[str, Any] | Any,
    *,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Keep default console output bounded without changing production digest input."""
    payload = dict(digest_input)
    candidate_keys = list(payload.pop("candidate_keys", []))
    return {
        **payload,
        "candidate_keys_count": len(candidate_keys),
        "candidate_keys_sample": candidate_keys[:sample_limit],
        "candidate_keys_truncated": len(candidate_keys) > sample_limit,
    }


def _target_groups(target: pd.DataFrame, group_cols: Sequence[str]):
    if len(group_cols) == 1:
        for key, group in target.groupby(group_cols[0], sort=False):
            yield (key,), group
        return
    for key, group in target.groupby(list(group_cols), sort=False):
        yield tuple(key), group


def build_preflight_reports(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    dataset_id: object,
    scenario: str,
    group_cols: Sequence[str],
    observed_start: object,
    k: int,
    grouping_col: str | None = None,
    exclusion_sample_limit: int = 20,
    pool_factory: Callable[..., PreparedDailySequencePool] = prepare_daily_sequence_pool,
) -> list[dict[str, Any]]:
    """Prepare source observations once and validate every target against that pool."""
    metadata_cols = (str(grouping_col),) if grouping_col else ()
    observed_start_ts = pd.Timestamp(observed_start).normalize()
    observed_end_ts = observed_start_ts + pd.Timedelta(days=29)
    prepared_pool = pool_factory(
        source,
        group_cols=tuple(group_cols),
        observed_start=observed_start,
        observed_end=observed_end_ts,
        pretrain_start=observed_end_ts - pd.Timedelta(days=179),
        pretrain_end=observed_end_ts,
        knn_feature_cols=get_knn_schema(dataset_id).ordered_names,
        required_feature_cols=get_knn_schema(dataset_id).ordered_names,
        metadata_cols=metadata_cols,
    )
    stub_columns = list(dict.fromkeys([*group_cols, "date", "sales", *metadata_cols]))
    source_stub = pd.DataFrame(columns=stub_columns)
    reports = []
    for target_key, target_group in _target_groups(target, group_cols):
        report = validate_protocol_frames(
            source_stub,
            target_group,
            dataset_id=dataset_id,
            scenario=scenario,
            group_cols=group_cols,
            grouping_col=grouping_col,
            observed_start=observed_start,
            k=k,
            prepared_pool=prepared_pool,
            exclusion_sample_limit=exclusion_sample_limit,
        )
        report["target_key_from_file"] = target_key
        reports.append(report)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=[f"d{i}" for i in range(1, 7)])
    parser.add_argument("--scenario", choices=("without", "with"), required=True)
    parser.add_argument("--parquet-dir", type=Path)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--exclusion-sample-limit", type=int, default=20)
    args = parser.parse_args()
    dataset_id = int(args.dataset[1:])
    parquet_dir = resolve_parquet_dir(dataset_id, args.parquet_dir)
    cfg = DATASET_CONFIG[dataset_id]
    source_columns = list(
        dict.fromkeys(
            [*cfg["group_cols"], "date", "sales", *([cfg["grouping_col"]] if cfg.get("grouping_col") else [])]
        )
    )
    target_columns = list(source_columns)
    source = pd.read_parquet(
        parquet_dir / f"dataset{dataset_id}-source.parquet",
        columns=source_columns,
    )
    target = pd.read_parquet(
        parquet_dir / f"dataset{dataset_id}-target.parquet",
        columns=target_columns,
    )
    reports = build_preflight_reports(
        source,
        target,
        dataset_id=dataset_id,
        scenario=args.scenario,
        group_cols=cfg["group_cols"],
        grouping_col=cfg.get("grouping_col"),
        observed_start=cfg["observed_start"],
        k=args.k,
        exclusion_sample_limit=args.exclusion_sample_limit,
    )
    print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))
    if not reports or any(report["status"] != "passed" for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
