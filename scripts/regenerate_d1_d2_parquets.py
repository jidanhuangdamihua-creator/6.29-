#!/usr/bin/env python3
"""Regenerate complete D1/D2 protocol parquets; never infer missing domains."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.protocols.experiment_protocol import PROTOCOL_VERSION, ProtocolViolation
from src.data_processing.sealed_daily import (
    D2_CANONICALIZATION_RULE_ID,
    FILL_POLICY_ENGINE_VERSION,
    RUNTIME_FILL_POLICY,
    SHARED_FILL_POLICY_CONFIG_DIGEST,
    calendarize_and_fill,
    canonicalize_source_sales,
    d2_approved_calendarize,
    publish_sealed_dataset,
    sha256_file,
    validate_target_truth,
)
from src.protocols.feature_schema import get_knn_schema, get_predictor_schema
from src.protocols.adopt_validation import (
    VALIDATION_POLICY_DIGEST,
    VALIDATION_POLICY_VERSION,
    validator_code_digest,
)
from src.protocols.sealing_protocol import (
    SEALING_PROTOCOL_VERSION,
    get_source_pretrain_window,
    get_target_window,
)


DEFAULT_D1_INPUT = ROOT / "数据集" / "原始数据" / "Dataset 1" / "train.csv"
DEFAULT_D2_INPUT = ROOT / "数据集" / "原始数据" / "Dataset 2" / "hierarchical_sales_data.csv"
DEFAULT_OUTPUT_DIR = ROOT / "数据集" / "固化数据" / "d1_d6_sealed_v1"


def _rename_required(frame: pd.DataFrame, aliases: dict[str, Sequence[str]]) -> pd.DataFrame:
    renamed = frame.copy()
    for canonical, options in aliases.items():
        if canonical in renamed.columns:
            continue
        match = next((name for name in options if name in renamed.columns), None)
        if match is None:
            raise ProtocolViolation(
                f"raw regeneration input is missing {canonical!r}; aliases={tuple(options)!r}"
            )
        renamed = renamed.rename(columns={match: canonical})
    return renamed


def _finalize_daily_frame(
    frame: pd.DataFrame,
    *,
    domain_col: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result[domain_col] = pd.to_numeric(result[domain_col], errors="raise").astype(int)
    result["item_id"] = pd.to_numeric(result["item_id"], errors="raise").astype(int)
    result["sales"] = pd.to_numeric(result["sales"], errors="raise")
    if result["date"].isna().any():
        raise ProtocolViolation("raw regeneration input contains invalid dates")
    key_cols = [domain_col, "item_id", "date"]
    if result.duplicated(key_cols).any():
        raise ProtocolViolation(f"raw regeneration input contains duplicate {key_cols}")
    result["entity_id"] = result[domain_col].astype(str)
    result["year"] = result["date"].dt.year.astype(int)
    result["month"] = result["date"].dt.month.astype(int)
    result["week"] = result["date"].dt.isocalendar().week.astype(int)
    result["day"] = result["date"].dt.day.astype(int)
    result = result.sort_values([domain_col, "item_id", "date"]).reset_index(drop=True)
    result.attrs.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "zero_demand_calendarization_declared": False,
            "generation_contract": "complete_explicit_source_pool",
        }
    )
    return result


def _assert_exact_entities(
    frame: pd.DataFrame,
    *,
    domain_col: str,
    expected: set[tuple[int, int]],
    role: str,
) -> None:
    actual = set(
        frame[[domain_col, "item_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ProtocolViolation(
            f"{role} entity contract mismatch: missing={missing!r} extra={extra!r}"
        )


def build_d1_protocol_frames(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normalized = _rename_required(
        raw,
        {
            "date": ("Date",),
            "store_id": ("store", "Store"),
            "item_id": ("item", "Item"),
            "sales": ("Sales",),
        },
    )
    normalized = _finalize_daily_frame(normalized, domain_col="store_id")
    source_expected = {(store, item) for store in range(1, 4) for item in range(1, 10)}
    source = normalized[
        normalized[["store_id", "item_id"]]
        .apply(tuple, axis=1)
        .isin(source_expected)
    ].copy()
    target = normalized[
        (normalized["store_id"] == 1) & (normalized["item_id"] == 10)
    ].copy()
    _assert_exact_entities(source, domain_col="store_id", expected=source_expected, role="D1 source")
    _assert_exact_entities(target, domain_col="store_id", expected={(1, 10)}, role="D1 target")
    return source, target


def build_d2_protocol_frames(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "DATE" in raw.columns and any(str(column).startswith("QTY_B") for column in raw.columns):
        rows = []
        for brand in range(1, 4):
            for item in range(1, 11):
                demand_col = f"QTY_B{brand}_{item}"
                if demand_col not in raw.columns:
                    raise ProtocolViolation(f"raw D2 wide input is missing required column {demand_col!r}")
                promo_col = f"PROMO_B{brand}_{item}"
                rows.append(
                    pd.DataFrame(
                        {
                            "date": raw["DATE"],
                            "brand_id": brand,
                            "item_id": item,
                            "sales": raw[demand_col],
                            "promo": raw[promo_col] if promo_col in raw.columns else 0,
                        }
                    )
                )
        raw = pd.concat(rows, ignore_index=True)
    normalized = _rename_required(
        raw,
        {
            "date": ("Date",),
            "brand_id": ("brand", "Brand"),
            "item_id": ("item", "Item"),
            "sales": ("Sales",),
        },
    )
    if "promo" not in normalized.columns:
        normalized["promo"] = 0
    normalized = _finalize_daily_frame(normalized, domain_col="brand_id")
    source_expected = {(brand, item) for brand in range(1, 4) for item in range(1, 10)}
    source = normalized[
        normalized[["brand_id", "item_id"]]
        .apply(tuple, axis=1)
        .isin(source_expected)
    ].copy()
    target = normalized[
        (normalized["brand_id"] == 1) & (normalized["item_id"] == 10)
    ].copy()
    _assert_exact_entities(source, domain_col="brand_id", expected=source_expected, role="D2 source")
    _assert_exact_entities(target, domain_col="brand_id", expected={(1, 10)}, role="D2 target")
    return source, target


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_pair(dataset_id: int, source: pd.DataFrame, target: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / f"dataset{dataset_id}-source.parquet"
    target_path = output_dir / f"dataset{dataset_id}-target.parquet"
    existing = [path for path in (source_path, target_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing protocol parquet(s): {existing!r}")
    source.to_parquet(source_path, index=False)
    target.to_parquet(target_path, index=False)


def _date_dict(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _parent_record(path: Path) -> Dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        return {
            "path": str(candidate),
            "sha256": None,
            "size_bytes": None,
            "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "mtime_ns": None,
        }
    stat = candidate.stat()
    return {
        "path": str(candidate),
        "sha256": sha256_file(candidate),
        "size_bytes": int(stat.st_size),
        "observed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _window_descriptor(dataset_id: int) -> Dict[str, Any]:
    target = get_target_window(dataset_id)
    source = get_source_pretrain_window(dataset_id)
    return {
        "target": {
            "train_start": _date_dict(target.train_start),
            "train_end": _date_dict(target.train_end),
            "validation_start": _date_dict(target.validation_start),
            "validation_end": _date_dict(target.validation_end),
            "blind_start": _date_dict(target.blind_start),
            "blind_end": _date_dict(target.blind_end),
        },
        "source": {
            "pretrain_start": _date_dict(source.pretrain_start),
            "pretrain_end": _date_dict(source.pretrain_end),
            "knn_start": _date_dict(source.knn_start),
            "knn_end": _date_dict(source.knn_end),
        },
    }


def build_and_seal_dataset(
    dataset_id: int,
    raw: pd.DataFrame,
    *,
    output_dir: Path,
    raw_input_path: Path,
) -> Path:
    """Rebuild D1/D2 from raw input and atomically publish one sealed dataset."""

    dataset_number = int(dataset_id)
    if dataset_number not in (1, 2):
        raise ValueError("raw rebuild is only approved for D1 and D2")
    if dataset_number == 1:
        source, target = build_d1_protocol_frames(raw)
        group_cols = ("store_id", "item_id")
        source = calendarize_and_fill(source, group_cols=group_cols, fill_rules={})
        target = calendarize_and_fill(target, group_cols=group_cols, fill_rules={})
    else:
        source, target = build_d2_protocol_frames(raw)
        group_cols = ("brand_id", "item_id")
        source = d2_approved_calendarize(source, group_cols=group_cols, fill_sales=False)
        target = d2_approved_calendarize(target, group_cols=group_cols)

    source, source_sales_audit = canonicalize_source_sales(source)
    validate_target_truth(target)
    source.attrs["dataset_name"] = f"Dataset{dataset_number}"
    target.attrs["dataset_name"] = f"Dataset{dataset_number}"

    predictor = get_predictor_schema(f"D{dataset_number}")
    knn = get_knn_schema(f"D{dataset_number}")
    parent = _parent_record(Path(raw_input_path))
    canonicalization = {
        "rule_id": D2_CANONICALIZATION_RULE_ID if dataset_number == 2 else "none",
        "approved_dates": ["2018-06-02"] if dataset_number == 2 else [],
        "target_sales_repair_performed": False,
    }
    calendar_audit = {
        "engine_version": FILL_POLICY_ENGINE_VERSION,
        "shared_with_raw_rebuild": True,
        "runtime_policy": RUNTIME_FILL_POLICY,
        "source": {
            "synthetic_date_count": int(source.attrs.get("synthetic_date_count", 0)),
            "config_digest": source.attrs.get("fill_policy_config_digest"),
        },
        "target": {
            "synthetic_date_count": int(target.attrs.get("synthetic_date_count", 0)),
            "config_digest": target.attrs.get("fill_policy_config_digest"),
        },
    }
    manifest = {
        "manifest_version": "sealed_dataset_manifest_v1",
        "dataset_id": f"D{dataset_number}",
        "sealed_root_version": SEALING_PROTOCOL_VERSION,
        "provenance_level": "raw_rebuilt",
        "content_validation_level": "raw_rebuilt",
        "adopted_content_validated": False,
        "content_validation_notes": "D1/D2 were rebuilt from the declared raw input using the reviewed protocol.",
        "parent_artifacts": {"raw_input": parent},
        "parent_artifact_sha256": parent["sha256"],
        "parent_artifact_size_bytes": parent["size_bytes"],
        "parent_artifact_observed_at": parent["observed_at"],
        "parent_artifact_mtime_ns": parent["mtime_ns"],
        "parent_artifact_first_seen_at": None,
        "parent_artifact_first_seen_source": None,
        "parent_artifact_first_seen_reliability": "unavailable",
        "fill_policy_engine_version": FILL_POLICY_ENGINE_VERSION,
        "fill_policy_shared_with_raw_rebuild": True,
        "fill_policy_config_digest": SHARED_FILL_POLICY_CONFIG_DIGEST,
        "runtime_fill_policy": RUNTIME_FILL_POLICY,
        "validation_policy_version": VALIDATION_POLICY_VERSION,
        "validation_policy_digest": VALIDATION_POLICY_DIGEST,
        "validator_code_digest": validator_code_digest(),
        "source_sales_canonicalization_version": source_sales_audit["version"],
        "source_sales_repair_mask_sha256": source_sales_audit["repair_mask_sha256"],
        "source_sales_repair_reason_counts": source_sales_audit["repair_reason_counts"],
        "predictor_feature_schema_digest": predictor.digest,
        "knn_feature_schema_digest": knn.digest,
        "windows": _window_descriptor(dataset_number),
        "dataset_canonicalization": canonicalization,
    }
    validation = {
        "status": "validated",
        "failure_reasons": [],
        "validation_policy_version": VALIDATION_POLICY_VERSION,
        "validation_policy_digest": VALIDATION_POLICY_DIGEST,
        "validator_code_digest": validator_code_digest(),
        "checks": [
            "raw input identity recorded",
            "target truth finite and non-negative",
            "source sales canonicalized before formal use",
            "feature schemas recorded",
        ],
    }
    sidecars = {
        "calendarization_audit.json": calendar_audit,
        "source_sales_canonicalization.json": source_sales_audit,
        "provenance.json": manifest["parent_artifacts"],
    }
    return publish_sealed_dataset(
        output_dir,
        dataset_number,
        source_frame=source,
        target_frame=target,
        manifest=manifest,
        validation_report=validation,
        sidecars=sidecars,
        predictor_schema=predictor.descriptor(),
        knn_schema=knn.descriptor(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("d1", "d2", "all"), default="all")
    parser.add_argument("--d1-input", type=Path, default=DEFAULT_D1_INPUT)
    parser.add_argument(
        "--d2-input",
        type=Path,
        default=DEFAULT_D2_INPUT,
        help="D2 raw long table or DATE + QTY_B<brand>_<item> wide table.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.dataset in {"d1", "all"}:
        build_and_seal_dataset(
            1,
            _read_table(args.d1_input),
            output_dir=args.output_dir,
            raw_input_path=args.d1_input,
        )
    if args.dataset in {"d2", "all"}:
        if args.d2_input is None:
            parser.error("--d2-input is required for D2; incomplete solidified parquet is not a raw source")
        build_and_seal_dataset(
            2,
            _read_table(args.d2_input),
            output_dir=args.output_dir,
            raw_input_path=args.d2_input,
        )


if __name__ == "__main__":
    main()
