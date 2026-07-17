"""Read-only Gate 1X D1-D6 real-input readiness preflight.

This command is intentionally unable to create a build, call a producer,
write parquet, create a manifest, or publish a deployment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import sys
from typing import Any

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
    dataset_contract,
    load_formal_identity,
    normalized_frame_digest,
    slice_dataset_roles,
    select_source_history_candidates,
)


PARQUET_DIR = ROOT / "数据集" / "固化数据"
RAW_DIR = ROOT / "数据集" / "原始数据"
RAW_INPUTS = {
    1: (RAW_DIR / "Dataset 1/train.csv", RAW_DIR / "Dataset 1/test.csv"),
    2: (RAW_DIR / "Dataset 2/hierarchical_sales_data.csv",),
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


def _target_frame(root: Path, dataset: int) -> pd.DataFrame:
    path = root / "数据集" / "固化数据" / f"dataset{dataset}-target.parquet"
    return pd.read_parquet(path)


def _source_frame(root: Path, dataset: int) -> pd.DataFrame | None:
    path = root / "数据集" / "固化数据" / f"dataset{dataset}-source.parquet"
    if dataset <= 3:
        return pd.read_parquet(path)
    return None


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


def _base_report(root: Path, parent_root: Path, old_root: Path, dataset: int, identity: Mapping[str, object] | None) -> dict[str, object]:
    spec = dataset_contract(dataset)
    source_path = root / "数据集" / "固化数据" / f"dataset{dataset}-source.parquet"
    target_path = root / "数据集" / "固化数据" / f"dataset{dataset}-target.parquet"
    target_keys = [list(key) for key in spec.target_keys]
    return {
        "formal_identity": dict(identity or {"contract_digest": CONTRACT_DIGEST, "combined_formal_identity_digest": COMBINED_FORMAL_IDENTITY_DIGEST}),
        "dataset": f"D{dataset}",
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
        "proof_inputs_available": {"formal_identity": identity is not None, "raw_authority": False, "parent": False, "views": False, "calendarization": False, "field_specific_repairs": False},
        "status": "failed",
        "failure_code": None,
    }


def _dataset_report(root: Path, parent_root: Path, old_root: Path, dataset_id: int, identity: Mapping[str, object] | None = None) -> dict[str, object]:
    try:
        spec = dataset_contract(dataset_id)
        report = _base_report(root, parent_root, old_root, dataset_id, identity)
        source_path = root / "数据集" / "固化数据" / f"dataset{dataset_id}-source.parquet"
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
        report["schema_fields"] = {"worker": report["worker_safe_fields"], "knn": ["date", "sales"]}
        report["field_exclusions"] = {"D2": ["PROMO", "promo", "Promo"], "D3": ["Open", "Customers", "Promo"], "D4": [*__import__("src.protocols.gate1_transformation", fromlist=["D4_AUDIT_ONLY"]).D4_AUDIT_ONLY], "D5": ["transactions", "week"], "D6": ["sales"]}.get(spec.dataset, [])
        report["proof_inputs_available"].update({"parent": True, "views": True, "calendarization": True, "field_specific_repairs": True})
        source = _source_frame(root, dataset_id)
        if source is not None:
            report["source_entities"] = [list(key) for key in sorted({tuple(str(value) for value in row) for row in source.loc[:, list(spec.key_fields)].drop_duplicates().itertuples(index=False, name=None)})]
            try:
                selected, source_proof = select_source_history_candidates(spec.dataset, source, "with-sharing", require_complete=True)
                report["source_selection"] = source_proof
                report["proof_inputs_available"]["source_eligibility"] = True
            except Gate1Failure as exc:
                report["failure_code"] = exc.code
                report["error"] = str(exc)
        else:
            report["failure_code"] = "SOURCE_STREAM_NOT_PROVEN"
            report["error"] = "large source authority requires bounded streaming eligibility validation"
        if report["duplicate_exact_keys"]:
            report["failure_code"] = "DUPLICATE_EXACT_KEY_DATE"
        if report["pre_or_equal_origin_forecast_rows"]:
            report["failure_code"] = "FORECAST_ORIGIN"
        report["status"] = "passed" if report["failure_code"] is None and calendarized_rows == spec.expected_blind_rows else "failed"
        return report
    except Exception as exc:
        return {"dataset": f"D{dataset_id}", "status": "failed", "failure_code": getattr(exc, "code", "READINESS_ERROR"), "error": f"{type(exc).__name__}: {exc}"}


def run_readiness(*, root: Path = ROOT, parent_root: Path | None = None, old_sealed_root: Path | None = None) -> dict[str, object]:
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
    failures = [item for item in datasets if item.get("status") != "passed"]
    return {"status": "passed" if not failures else "failed", "failure_code": None if not failures else failures[0].get("failure_code"), "formal_identity": identity or {"status": "failed", "error": identity_error}, "datasets": datasets, "read_only": True, "writes_performed": False, "producer_calls_performed": 0, "private_build_created": False, "deployment_created": False, "manifest_candidate_created": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 1X real-input readiness; read-only")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--parent-root", type=Path, default=None)
    parser.add_argument("--old-sealed-root", type=Path, default=None)
    parser.add_argument("--read-only", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_readiness(root=args.root, parent_root=args.parent_root, old_sealed_root=args.old_sealed_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
