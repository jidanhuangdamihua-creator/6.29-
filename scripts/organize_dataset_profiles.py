#!/usr/bin/env python3
"""Organize verified Dataset1-Dataset6 profiles from raw-data scans.

This script treats existing files under outputs/ as review leads only.  Facts
written to formal profiles come from a fresh raw-data scan executed through
scripts.scan_dataset_profiles_d1_d6.
"""

from __future__ import annotations

import argparse
import shutil
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import scan_dataset_profiles_d1_d6 as scanner

DEFAULT_DATA_ROOT = PROJECT_ROOT / "数据集" / "原始数据"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_profiles"

VERIFIED_STATUSES = {"VERIFIED", "SUPPORTED", "INCOMPLETE"}
REQUIRED_WINDOW_LABEL = "15 train + 15 validation + 180 test = 210 days"


def _json_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _stringify(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_stringify(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    for col in view.columns:
        view[col] = view[col].map(_stringify)
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[h]).replace("\n", " ") for h in headers) + " |")
    return "\n".join(lines)


def snapshot_old_outputs(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Read existing review outputs before overwriting them."""
    snapshots: Dict[str, Dict[str, Any]] = {}
    for idx in range(1, 7):
        dataset = f"Dataset{idx}"
        ds_dir = output_dir / dataset
        snapshots[dataset] = {
            "summary_json": _json_load(ds_dir / "dataset_profile_summary.json") or {},
            "summary_md": (ds_dir / "dataset_profile_summary.md").read_text(encoding="utf-8", errors="ignore")
            if (ds_dir / "dataset_profile_summary.md").is_file()
            else "",
            "global_overview_md": (output_dir / "d1_d6_complete_overview.md").read_text(encoding="utf-8", errors="ignore")
            if (output_dir / "d1_d6_complete_overview.md").is_file()
            else "",
        }
    return snapshots


def scan_raw_dataset_files(
    data_root: Path,
    output_dir: Path,
    max_rows: int,
    chunk_size: int,
    max_probe_rows: int,
) -> Tuple[List[Dict[str, Any]], scanner.DatasetDiscovery]:
    """Scan D1-D6 from the raw data root and return raw-derived summaries."""
    discovery = scanner.discover_dataset_paths(data_root)
    scanner.write_discovery_reports(discovery, output_dir)
    results: List[Dict[str, Any]] = []
    for idx in range(1, 7):
        name = f"Dataset{idx}"
        dataset = discovery.datasets.get(name)
        if dataset is None:
            msg = f"{name} raw files not confidently detected."
            print(f"[{name}] {msg}", flush=True)
            results.append({"Dataset": name, "dataset": name, "scan_status": "FAILED", "error": msg})
            continue
        results.append(scanner.scan_one_dataset(dataset, output_dir, max_rows, chunk_size, max_probe_rows))
    return results, discovery


def infer_dataset_schema(result: Dict[str, Any], ds_dir: Path) -> Dict[str, Any]:
    """Return schema facts confirmed by the raw scan outputs."""
    features = _read_csv(ds_dir / "feature_profile.csv")
    columns = features["feature_name"].dropna().astype(str).tolist() if "feature_name" in features else []
    return {
        "raw_file_list": [result.get("main_table_file"), *(result.get("auxiliary_files") or [])],
        "file_shape": f"{result.get('main_table_rows_scanned')} rows x {result.get('main_table_cols')} columns",
        "columns": columns,
        "date_column_candidates": [result.get("date_col")] if result.get("date_col") else [],
        "target_sales_column_candidates": [result.get("sales_col")] if result.get("sales_col") else [],
        "entity_key_candidates": result.get("entity_cols") or [],
        "static_feature_columns": features.loc[
            features.get("shared_or_entity_specific", pd.Series(dtype=str)).astype(str).eq("entity_specific"),
            "feature_name",
        ].astype(str).tolist()
        if not features.empty and "shared_or_entity_specific" in features
        else [],
        "dynamic_feature_columns": features.loc[
            features.get("recommended_for_rfe", pd.Series(dtype=bool)).astype(str).str.lower().eq("true"),
            "feature_name",
        ].astype(str).tolist()
        if not features.empty and "recommended_for_rfe" in features
        else [],
        "metadata_columns": features.loc[
            features.get("role", pd.Series(dtype=str)).astype(str).str.contains("entity id|categorical", case=False, na=False),
            "feature_name",
        ].astype(str).tolist()
        if not features.empty and "role" in features
        else [],
    }


def _entity_part_counts(entity_time_span: pd.DataFrame) -> Dict[str, int]:
    counts: Dict[str, set[str]] = {}
    if "entity_id" not in entity_time_span:
        return {}
    for entity_id in entity_time_span["entity_id"].dropna().astype(str):
        for part in entity_id.split("|"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            counts.setdefault(key, set()).add(value)
    return {key: len(values) for key, values in counts.items()}


def audit_entity_keys(result: Dict[str, Any], ds_dir: Path) -> pd.DataFrame:
    """Audit candidate entity keys using raw-scan entity outputs."""
    ent_span = _read_csv(ds_dir / "entity_time_span.csv")
    part_counts = _entity_part_counts(ent_span)
    entity_cols = [str(c) for c in (result.get("entity_cols") or [])]
    recommended = " + ".join(entity_cols or ["global_entity"])
    candidates: List[str] = []
    for col in entity_cols:
        candidates.append(col)
    if len(entity_cols) >= 2:
        candidates.append(recommended)
    for generic in ("store", "item", "sku", "product", "brand", "city", "category", "store_id", "item_id"):
        if generic in part_counts and generic not in candidates:
            candidates.append(generic)
    if not candidates:
        candidates.append("global_entity")

    rows = []
    entity_count = int(result.get("entity_count") or 0)
    for candidate in candidates:
        is_recommended = candidate == recommended or (candidate == "global_entity" and not entity_cols)
        unique_count = entity_count if is_recommended else part_counts.get(candidate)
        duplicate_status = "INCOMPLETE"
        duplicate_evidence = "Exact date + candidate-key duplicate sets are not retained for every large-table scan."
        if is_recommended and result.get("scan_coverage") in {"FULL_SCAN", "CHUNKED_FULL_SCAN"}:
            duplicate_status = "SUPPORTED"
            duplicate_evidence = (
                "Recommended entity count and per-entity spans were rebuilt from the raw scan; "
                "large-table duplicate checks remain a follow-up unless a row-level duplicate file is present."
            )
        rows.append({
            "candidate_key": candidate,
            "unique_count": unique_count,
            "duplicate_check_under_date_key": duplicate_status,
            "recommended_entity_key": recommended if is_recommended else "",
            "confidence": 0.90 if is_recommended and entity_cols else 0.55,
            "raw_scan_evidence": duplicate_evidence,
        })
    return pd.DataFrame(rows)


def audit_time_span(result: Dict[str, Any], ds_dir: Path) -> Dict[str, Any]:
    ent_span = _read_csv(ds_dir / "entity_time_span.csv")
    gaps = _read_csv(ds_dir / "entity_gap_report.csv")
    return {
        "global_min_date": result.get("global_min_date"),
        "global_max_date": result.get("global_max_date"),
        "per_entity_min_date_available": "min_date" in ent_span,
        "per_entity_max_date_available": "max_date" in ent_span,
        "span_days_distribution": result.get("span_distribution") or {},
        "effective_nonzero_sales_days_distribution": result.get("valid_sales_days_distribution") or {},
        "missing_date_gaps_max": gaps["max_gap_days"].max() if not gaps.empty and "max_gap_days" in gaps else None,
        "short_history_entities": result.get("entities_not_meeting_210_days"),
    }


def audit_data_quality(result: Dict[str, Any], ds_dir: Path) -> Dict[str, Any]:
    quality = _read_csv(ds_dir / "entity_data_quality.csv")
    features = _read_csv(ds_dir / "feature_profile.csv")
    sales_dist = _read_csv(ds_dir / "sales_distribution_by_entity.csv")
    price_rows = features[
        features.get("role", pd.Series(dtype=str)).astype(str).str.contains("price", case=False, na=False)
    ] if not features.empty else pd.DataFrame()
    promo_rows = features[
        features.get("role", pd.Series(dtype=str)).astype(str).str.contains("promo", case=False, na=False)
    ] if not features.empty else pd.DataFrame()
    holiday_rows = features[
        features.get("role", pd.Series(dtype=str)).astype(str).str.contains("holiday|open|status", case=False, na=False)
    ] if not features.empty else pd.DataFrame()
    return {
        "missing_values_ratio": result.get("missing_ratio"),
        "zero_sales_ratio": result.get("zero_sales_ratio"),
        "negative_sales_evidence": "not confirmed by current summary; see raw sales_distribution_by_entity.csv",
        "duplicate_rows_evidence": "INCOMPLETE: row-level duplicate counts are not retained by all large-table scans.",
        "outliers_evidence": "see sales_distribution_by_entity.csv",
        "calendar_gaps_evidence": "see entity_gap_report.csv",
        "price_columns": price_rows["feature_name"].astype(str).tolist() if "feature_name" in price_rows else [],
        "promotion_columns": promo_rows["feature_name"].astype(str).tolist() if "feature_name" in promo_rows else [],
        "holiday_or_open_store_columns": holiday_rows["feature_name"].astype(str).tolist() if "feature_name" in holiday_rows else [],
        "entity_quality_rows": len(quality),
        "sales_distribution_rows": len(sales_dist),
    }


def verify_window_claims_against_raw_scan(result: Dict[str, Any]) -> Dict[str, Any]:
    entity_count = int(result.get("entity_count") or 0)
    feasible = int(result.get("entities_meeting_210_days") or 0)
    status = "SUPPORTED_BY_RAW_DATA" if feasible > 0 else "NOT_SUPPORTED_BY_RAW_DATA"
    return {
        "window": REQUIRED_WINDOW_LABEL,
        "required_days": scanner.REQUIRED_DAYS,
        "entities_meeting_required_days": feasible,
        "entity_count": entity_count,
        "status": status,
        "notes": (
            "Raw scan supports feasibility for at least one entity, but does not confirm any paper-specific "
            "observed/validation/test calendar split or target/source split."
        )
        if feasible > 0
        else "No entity met the required 210-day history in the raw scan.",
    }


def verify_source_target_claims_against_raw_scan(result: Dict[str, Any]) -> Dict[str, Any]:
    method = result.get("source_target_method")
    if method == "from_existing_project_config":
        status = "SUPPORTED"
        notes = "Project config target/source entities were mapped into raw entity ids; still verify paper semantics manually."
    else:
        status = "INCOMPLETE"
        notes = "Source/target rows are data-quality candidates only and are excluded from confirmed profile conclusions."
    return {
        "source_target_method": method,
        "candidate_targets": result.get("candidate_targets") or [],
        "candidate_sources": result.get("candidate_sources") or [],
        "verification_status": status,
        "notes": notes,
    }


def verify_old_claims_against_raw_scan(
    old_snapshot: Dict[str, Any],
    result: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare old output claims with fresh raw-scan facts."""
    old = old_snapshot.get("summary_json") or {}
    dataset = result.get("dataset") or result.get("Dataset")
    rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    comparable = [
        "date_col", "sales_col", "entity_cols", "global_min_date", "global_max_date",
        "global_total_days", "entity_count", "main_entity_grain", "feature_count",
    ]
    for field in comparable:
        if field not in old:
            continue
        old_value = old.get(field)
        new_value = result.get(field)
        old_norm = _stringify(old_value)
        new_norm = _stringify(new_value)
        decision = "VERIFIED" if old_norm == new_norm else "CONFLICTED"
        reason = "Old claim matches the fresh raw scan." if decision == "VERIFIED" else "Old claim differs from fresh raw scan."
        row = {
            "dataset": dataset,
            "claim": f"{field} = {old_norm}",
            "source_file": f"{dataset}/dataset_profile_summary.json",
            "raw_scan_evidence": f"{field} = {new_norm}",
            "decision": decision,
            "reason": reason,
        }
        rows.append(row)
        if decision == "CONFLICTED":
            rejected.append({
                "dataset": dataset,
                "claim": row["claim"],
                "source_file": row["source_file"],
                "status": "CONFLICTED",
                "reason": reason,
                "raw_scan_evidence": row["raw_scan_evidence"],
                "required_follow_up": "Use the fresh raw scan value or manually inspect the raw file.",
            })

    for field in ("candidate_targets", "candidate_sources", "source_target_reason"):
        if old.get(field):
            evidence = verify_source_target_claims_against_raw_scan(result)
            status = "UNVERIFIED" if evidence["verification_status"] == "INCOMPLETE" else "SUPPORTED"
            row = {
                "dataset": dataset,
                "claim": f"{field} = {_stringify(old.get(field))}",
                "source_file": f"{dataset}/dataset_profile_summary.json",
                "raw_scan_evidence": evidence["notes"],
                "decision": status,
                "reason": "Concrete source/target claims require protocol-level confirmation, not only data-quality ranking.",
            }
            rows.append(row)
            if status == "UNVERIFIED":
                rejected.append({
                    "dataset": dataset,
                    "claim": row["claim"],
                    "source_file": row["source_file"],
                    "status": "UNVERIFIED",
                    "reason": row["reason"],
                    "raw_scan_evidence": row["raw_scan_evidence"],
                    "required_follow_up": "Confirm target/source protocol from paper/config before adding to formal profile.",
                })
    return pd.DataFrame(rows), pd.DataFrame(rejected)


def write_rejected_claims(path: Path, rejected_df: pd.DataFrame) -> None:
    lines = ["# Rejected Claims", ""]
    if rejected_df.empty:
        lines.append("_No rejected or excluded historical claims were detected._")
    for _, row in rejected_df.iterrows():
        lines.extend([
            "## Rejected Claim",
            "",
            f"Claim: {row.get('claim')}",
            f"Source file: {row.get('source_file')}",
            f"Status: {row.get('status')}",
            f"Reason rejected: {row.get('reason')}",
            f"Raw scan evidence: {row.get('raw_scan_evidence')}",
            "Impact if used: Could make the formal dataset profile overstate an unverified or conflicting fact.",
            f"Suggested correction: {row.get('required_follow_up')}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _profile_rows(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset = result.get("dataset") or result.get("Dataset")
    raw_source = result.get("main_table_file") or result.get("data_path")
    status = "VERIFIED" if result.get("scan_coverage") in {"FULL_SCAN", "CHUNKED_FULL_SCAN"} else "INCOMPLETE"
    confidence = 0.95 if status == "VERIFIED" else 0.45
    fields = {
        "raw_file_list": [result.get("main_table_file"), *(result.get("auxiliary_files") or [])],
        "file_shape": f"{result.get('main_table_rows_scanned')} rows x {result.get('main_table_cols')} columns",
        "date_col": result.get("date_col"),
        "sales_col": result.get("sales_col"),
        "recommended_entity_key": " + ".join(result.get("entity_cols") or ["global_entity"]),
        "global_date_range": f"{result.get('global_min_date')} to {result.get('global_max_date')}",
        "global_total_days": result.get("global_total_days"),
        "entity_count": result.get("entity_count"),
        "entities_meeting_210_days": result.get("entities_meeting_210_days"),
        "missing_ratio": result.get("missing_ratio"),
        "zero_sales_ratio": result.get("zero_sales_ratio"),
        "feature_count": result.get("feature_count"),
        "rfe_candidate_count": result.get("rfe_candidate_count"),
    }
    rows = []
    for field, value in fields.items():
        rows.append({
            "dataset": dataset,
            "field": field,
            "value": _stringify(value),
            "evidence_source": raw_source,
            "verification_status": status if value not in (None, "", []) else "MISSING",
            "confidence": confidence if value not in (None, "", []) else 0.0,
            "notes": "Fresh raw-data scan result.",
        })
    window = verify_window_claims_against_raw_scan(result)
    rows.append({
        "dataset": dataset,
        "field": "15+15+180 feasibility",
        "value": window["status"],
        "evidence_source": raw_source,
        "verification_status": "SUPPORTED" if window["status"] == "SUPPORTED_BY_RAW_DATA" else "INCOMPLETE",
        "confidence": 0.80,
        "notes": window["notes"],
    })
    st = verify_source_target_claims_against_raw_scan(result)
    rows.append({
        "dataset": dataset,
        "field": "source_target_split",
        "value": st["source_target_method"] or "",
        "evidence_source": raw_source,
        "verification_status": st["verification_status"],
        "confidence": 0.65 if st["verification_status"] == "SUPPORTED" else 0.35,
        "notes": st["notes"],
    })
    return rows


def write_verified_profiles(
    result: Dict[str, Any],
    ds_dir: Path,
    schema: Dict[str, Any],
    key_audit: pd.DataFrame,
    time_audit: Dict[str, Any],
    quality_audit: Dict[str, Any],
    verification_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
) -> pd.DataFrame:
    """Write formal per-dataset profile files from verified/supported facts."""
    profile_df = pd.DataFrame(_profile_rows(result))
    formal_df = profile_df[profile_df["verification_status"].isin(VERIFIED_STATUSES)].copy()
    _write_csv(key_audit, ds_dir / "candidate_key_audit.csv")

    raw_scan = {
        "dataset": result.get("dataset") or result.get("Dataset"),
        "schema": schema,
        "entity_key_audit": key_audit.to_dict(orient="records"),
        "time_span_audit": time_audit,
        "data_quality_audit": quality_audit,
        "source_target_audit": verify_source_target_claims_against_raw_scan(result),
        "window_audit": verify_window_claims_against_raw_scan(result),
    }
    (ds_dir / "raw_scan_report.json").write_text(
        json.dumps(raw_scan, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    raw_md = [
        f"# Raw Scan Report: {raw_scan['dataset']}",
        "",
        "## Data Files And Fields",
        "",
        _df_to_markdown(pd.DataFrame([
            {"item": key, "value": _stringify(value)} for key, value in schema.items()
        ]), max_rows=50),
        "",
        "## Entity Key Audit",
        "",
        _df_to_markdown(key_audit, max_rows=50),
        "",
        "## Time Span Audit",
        "",
        _df_to_markdown(pd.DataFrame([time_audit]), max_rows=10),
        "",
        "## Data Quality Audit",
        "",
        _df_to_markdown(pd.DataFrame([quality_audit]), max_rows=10),
        "",
        "## Window Audit",
        "",
        _df_to_markdown(pd.DataFrame([verify_window_claims_against_raw_scan(result)]), max_rows=10),
        "",
        "## Source / Target Audit",
        "",
        _df_to_markdown(pd.DataFrame([verify_source_target_claims_against_raw_scan(result)]), max_rows=10),
    ]
    (ds_dir / "raw_scan_report.md").write_text("\n".join(raw_md), encoding="utf-8")

    profile_lines = [
        f"# Formal Dataset Profile: {raw_scan['dataset']}",
        "",
        "Only VERIFIED, SUPPORTED, or explicitly INCOMPLETE facts are included here. Historical output claims are not used as fact sources.",
        "",
        "## Profile Fields",
        "",
        _df_to_markdown(formal_df, max_rows=60),
        "",
        "## Entity Key",
        "",
        _df_to_markdown(key_audit, max_rows=40),
    ]
    (ds_dir / "dataset_profile.md").write_text("\n".join(profile_lines), encoding="utf-8")

    _write_csv(verification_df, ds_dir / "verification_report.csv")
    verification_lines = [
        f"# Verification Report: {raw_scan['dataset']}",
        "",
        _df_to_markdown(verification_df, max_rows=80),
    ]
    (ds_dir / "verification_report.md").write_text("\n".join(verification_lines), encoding="utf-8")
    write_rejected_claims(ds_dir / "rejected_claims.md", rejected_df)

    warnings = []
    for warning in result.get("warnings") or []:
        warnings.append({
            "Status": "INCOMPLETE",
            "Reason": warning,
            "Evidence from raw scan": result.get("main_table_file"),
            "Original source file": "raw scan",
            "Required follow-up": "Review raw file and project protocol before promoting to confirmed profile fact.",
        })
    for _, row in rejected_df.iterrows():
        warnings.append({
            "Status": row.get("status"),
            "Reason": row.get("reason"),
            "Evidence from raw scan": row.get("raw_scan_evidence"),
            "Original source file": row.get("source_file"),
            "Required follow-up": row.get("required_follow_up"),
        })
    if not key_audit.empty and key_audit["duplicate_check_under_date_key"].astype(str).eq("INCOMPLETE").any():
        warnings.append({
            "Status": "INCOMPLETE",
            "Reason": "Some candidate-key duplicate checks remain incomplete for large-table scans.",
            "Evidence from raw scan": "candidate_key_audit.csv",
            "Original source file": result.get("main_table_file"),
            "Required follow-up": "Run an exact date+candidate-key duplicate job if this key is needed as a confirmed profile fact.",
        })
    warning_df = pd.DataFrame(warnings)
    _write_csv(warning_df, ds_dir / "warnings_and_gaps.csv")
    warning_lines = [
        f"# Warnings And Gaps: {raw_scan['dataset']}",
        "",
        _df_to_markdown(warning_df, max_rows=80),
    ]
    (ds_dir / "warnings_and_gaps.md").write_text("\n".join(warning_lines), encoding="utf-8")
    return profile_df


def write_global_verified_outputs(
    results: Sequence[Dict[str, Any]],
    output_dir: Path,
    all_profiles: pd.DataFrame,
    all_verifications: pd.DataFrame,
    all_rejected: pd.DataFrame,
    all_warnings: pd.DataFrame,
) -> None:
    raw_rows = []
    for result in results:
        raw_rows.append({
            "dataset": result.get("dataset") or result.get("Dataset"),
            "scan_status": result.get("scan_status"),
            "scan_coverage": result.get("scan_coverage"),
            "raw_path": result.get("data_path"),
            "main_table_file": result.get("main_table_file"),
            "rows_scanned": result.get("main_table_rows_scanned"),
            "columns": result.get("main_table_cols"),
            "date_col": result.get("date_col"),
            "sales_col": result.get("sales_col"),
            "entity_cols": _stringify(result.get("entity_cols")),
            "global_min_date": result.get("global_min_date"),
            "global_max_date": result.get("global_max_date"),
            "entity_count": result.get("entity_count"),
            "entities_meeting_210_days": result.get("entities_meeting_210_days"),
            "missing_ratio": result.get("missing_ratio"),
            "zero_sales_ratio": result.get("zero_sales_ratio"),
        })
    raw_df = pd.DataFrame(raw_rows)
    _write_csv(raw_df, output_dir / "raw_scan_summary.csv")
    (output_dir / "raw_scan_summary.md").write_text(
        "\n".join(["# Raw Scan Summary", "", _df_to_markdown(raw_df, max_rows=20)]),
        encoding="utf-8",
    )

    _write_csv(all_profiles, output_dir / "all_datasets_summary.csv")
    (output_dir / "all_datasets_summary.md").write_text(
        "\n".join([
            "# All Datasets Summary",
            "",
            "Every row includes evidence status. Confirmed conclusions are limited to VERIFIED/SUPPORTED rows.",
            "",
            _df_to_markdown(all_profiles, max_rows=120),
        ]),
        encoding="utf-8",
    )

    _write_csv(all_verifications, output_dir / "verification_report.csv")
    (output_dir / "verification_report.md").write_text(
        "\n".join(["# Verification Report", "", _df_to_markdown(all_verifications, max_rows=200)]),
        encoding="utf-8",
    )
    _write_csv(all_rejected, output_dir / "rejected_claims.csv")
    write_rejected_claims(output_dir / "rejected_claims.md", all_rejected)
    _write_csv(all_warnings, output_dir / "unresolved_issues.csv")
    (output_dir / "unresolved_issues.md").write_text(
        "\n".join(["# Unresolved Issues", "", _df_to_markdown(all_warnings, max_rows=200)]),
        encoding="utf-8",
    )


def _source_folder_files(source_folder: Path) -> List[str]:
    return [
        str(path.resolve())
        for path in sorted(source_folder.rglob("*"))
        if path.is_file()
    ]


def _dataset_source_folder_candidates(outputs_root: Path, profile_dir: Path) -> List[Path]:
    outputs_root = outputs_root.resolve()
    profile_dir = profile_dir.resolve()
    if not outputs_root.is_dir():
        return []
    candidates = []
    for child in sorted(outputs_root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if child.resolve() == profile_dir:
            continue
        if child.name.lower().startswith("dataset"):
            candidates.append(child)
    return candidates


def write_dataset_profile_inventory_and_provenance(
    output_dir: Path,
    results: Sequence[Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Record dataset-prefixed source folders before cleanup."""
    profile_dir = output_dir.resolve()
    outputs_root = profile_dir.parent.resolve()
    candidates = _dataset_source_folder_candidates(outputs_root, profile_dir)
    dataset_status = {
        str(result.get("dataset") or result.get("Dataset")): result.get("scan_status")
        for result in results
    }
    required_outputs = [
        profile_dir / "raw_scan_summary.md",
        profile_dir / "verification_report.md",
        profile_dir / "rejected_claims.md",
    ]
    d1_d6_complete = all(
        (profile_dir / f"Dataset{idx}" / "dataset_profile.md").is_file()
        for idx in range(1, 7)
    )
    global_outputs_complete = all(path.is_file() for path in required_outputs)
    can_archive = d1_d6_complete and global_outputs_complete

    inventory_rows = []
    manifest_sources = []
    for candidate in candidates:
        files = _source_folder_files(candidate)
        status = "ARCHIVED" if can_archive else "INCOMPLETE"
        inventory_rows.append({
            "source_folder": str(candidate.resolve()),
            "archived_into": str(profile_dir),
            "file_count": len(files),
            "status": status,
            "d1_d6_profiles_complete": d1_d6_complete,
            "global_required_outputs_complete": global_outputs_complete,
            "notes": "Dataset-prefixed outputs source folder recorded before protected cleanup.",
        })
        manifest_sources.append({
            "source_folder": str(candidate.resolve()),
            "archived_into": str(profile_dir),
            "files": files,
            "status": status,
        })

    inventory_df = pd.DataFrame(inventory_rows)
    _write_csv(inventory_df, profile_dir / "dataset_profile_inventory.csv")
    manifest = {
        "created_at": _utc_now_iso(),
        "outputs_root": str(outputs_root),
        "dataset_profiles": str(profile_dir),
        "dataset_status": dataset_status,
        "required_outputs": [str(path) for path in required_outputs],
        "source_folders": manifest_sources,
    }
    (profile_dir / "provenance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return inventory_df, manifest


def _validate_cleanup_candidate(outputs_root: Path, candidate: Path) -> None:
    outputs_root = outputs_root.resolve()
    candidate = candidate.resolve()
    if outputs_root not in candidate.parents:
        raise RuntimeError(f"Refuse to delete outside outputs: {candidate}")
    if candidate.name == "dataset_profiles":
        raise RuntimeError("Refuse to delete dataset_profiles")
    if not candidate.name.lower().startswith("dataset"):
        raise RuntimeError(f"Refuse to delete non-dataset folder: {candidate}")


def _load_cleanup_inventory(profile_dir: Path) -> pd.DataFrame:
    path = profile_dir / "dataset_profile_inventory.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_cleanup_provenance(profile_dir: Path) -> Dict[str, Any]:
    path = profile_dir / "provenance_manifest.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _profile_outputs_ready(profile_dir: Path) -> Tuple[bool, List[str]]:
    missing = []
    for required in ("raw_scan_summary.md", "verification_report.md", "rejected_claims.md"):
        if not (profile_dir / required).is_file():
            missing.append(required)
    for idx in range(1, 7):
        rel = f"Dataset{idx}/dataset_profile.md"
        if not (profile_dir / f"Dataset{idx}" / "dataset_profile.md").is_file():
            missing.append(rel)
    return not missing, missing


def _provenance_source_map(provenance: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    source_map = {}
    for entry in provenance.get("source_folders") or []:
        source = entry.get("source_folder")
        if source:
            source_map[str(Path(source).resolve())] = entry
    return source_map


def _inventory_sources(inventory_df: pd.DataFrame) -> set[str]:
    if inventory_df.empty or "source_folder" not in inventory_df:
        return set()
    return {str(Path(value).resolve()) for value in inventory_df["source_folder"].dropna().astype(str)}


def _source_files_recorded(candidate: Path, provenance_entry: Dict[str, Any]) -> bool:
    recorded = {str(Path(value).resolve()) for value in provenance_entry.get("files") or []}
    actual = set(_source_folder_files(candidate))
    return actual.issubset(recorded)


def cleanup_archived_dataset_source_folders(
    outputs_root: Path,
    profile_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Delete archived dataset-prefixed source folders under outputs only."""
    outputs_root = outputs_root.resolve()
    profile_dir = profile_dir.resolve()
    deleted_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    ready, missing_outputs = _profile_outputs_ready(profile_dir)
    inventory_df = _load_cleanup_inventory(profile_dir)
    provenance = _load_cleanup_provenance(profile_dir)
    inventory_sources = _inventory_sources(inventory_df)
    provenance_sources = _provenance_source_map(provenance)
    candidates = _dataset_source_folder_candidates(outputs_root, profile_dir)

    for candidate in candidates:
        resolved = candidate.resolve()
        row_base = {
            "deleted_path": str(resolved),
            "deleted_at": "",
            "reason": "Archived dataset-prefixed source folder consolidated into outputs/dataset_profiles.",
            "confirmed_in_inventory": str(resolved) in inventory_sources,
            "confirmed_in_provenance": str(resolved) in provenance_sources,
            "safety_check_status": "PENDING",
            "notes": "",
        }
        try:
            _validate_cleanup_candidate(outputs_root, resolved)
            if not ready:
                raise RuntimeError(f"Required dataset profile outputs missing: {', '.join(missing_outputs)}")
            if not row_base["confirmed_in_inventory"]:
                raise RuntimeError("Source folder not recorded in dataset_profile_inventory.csv")
            if not row_base["confirmed_in_provenance"]:
                raise RuntimeError("Source folder not recorded in provenance_manifest.json")
            if not _source_files_recorded(resolved, provenance_sources[str(resolved)]):
                raise RuntimeError("Source folder files are not fully recorded in provenance_manifest.json")
            shutil.rmtree(resolved)
            row_base["deleted_at"] = _utc_now_iso()
            row_base["safety_check_status"] = "PASSED_AND_DELETED"
            row_base["notes"] = "Deleted after inventory, provenance, profile-output, and path safety checks."
            deleted_rows.append(row_base)
        except Exception as exc:
            row_base["safety_check_status"] = "SKIPPED"
            row_base["notes"] = str(exc)
            skipped_rows.append(row_base)

    deleted_df = pd.DataFrame(deleted_rows, columns=[
        "deleted_path", "deleted_at", "reason", "confirmed_in_inventory",
        "confirmed_in_provenance", "safety_check_status", "notes",
    ])
    _write_csv(deleted_df, profile_dir / "deleted_source_folders_manifest.csv")
    manifest_lines = [
        "# Deleted Source Folders Manifest",
        "",
        "## Deleted Folders",
        "",
        _df_to_markdown(deleted_df, max_rows=100),
    ]
    if skipped_rows:
        manifest_lines.extend([
            "",
            "## Skipped Folders",
            "",
            _df_to_markdown(pd.DataFrame(skipped_rows), max_rows=100),
        ])
    (profile_dir / "deleted_source_folders_manifest.md").write_text(
        "\n".join(manifest_lines), encoding="utf-8"
    )
    return {"deleted": deleted_rows, "skipped": skipped_rows}


def organize_dataset_profiles(
    data_root: Path,
    output_dir: Path,
    max_rows: int,
    chunk_size: int,
    max_probe_rows: int,
) -> List[Dict[str, Any]]:
    old_snapshots = snapshot_old_outputs(output_dir)
    results, discovery = scan_raw_dataset_files(data_root, output_dir, max_rows, chunk_size, max_probe_rows)
    scanner.write_global_reports(results, output_dir)

    profile_frames: List[pd.DataFrame] = []
    verification_frames: List[pd.DataFrame] = []
    rejected_frames: List[pd.DataFrame] = []
    warning_frames: List[pd.DataFrame] = []

    for result in results:
        dataset = result.get("dataset") or result.get("Dataset")
        ds_dir = output_dir / str(dataset)
        ds_dir.mkdir(parents=True, exist_ok=True)
        if result.get("scan_status") == "FAILED":
            rejected = pd.DataFrame([{
                "dataset": dataset,
                "claim": f"{dataset} raw files not confidently detected.",
                "source_file": str(data_root),
                "status": "INCOMPLETE",
                "reason": result.get("error"),
                "raw_scan_evidence": result.get("error"),
                "required_follow_up": "Locate and register the raw files before building a formal profile.",
            }])
            write_rejected_claims(ds_dir / "rejected_claims.md", rejected)
            rejected_frames.append(rejected)
            warning_frames.append(rejected.rename(columns={
                "status": "Status",
                "reason": "Reason",
                "raw_scan_evidence": "Evidence from raw scan",
                "source_file": "Original source file",
                "required_follow_up": "Required follow-up",
            }))
            continue

        schema = infer_dataset_schema(result, ds_dir)
        key_audit = audit_entity_keys(result, ds_dir)
        time_audit = audit_time_span(result, ds_dir)
        quality_audit = audit_data_quality(result, ds_dir)
        verification_df, rejected_df = verify_old_claims_against_raw_scan(old_snapshots.get(str(dataset), {}), result)
        profile_df = write_verified_profiles(
            result, ds_dir, schema, key_audit, time_audit, quality_audit, verification_df, rejected_df
        )
        profile_frames.append(profile_df)
        verification_frames.append(verification_df)
        rejected_frames.append(rejected_df)
        warnings_df = _read_csv(ds_dir / "warnings_and_gaps.csv")
        if not warnings_df.empty:
            warnings_df.insert(0, "dataset", dataset)
            warning_frames.append(warnings_df)

    all_profiles = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()
    all_verifications = pd.concat(verification_frames, ignore_index=True) if verification_frames else pd.DataFrame()
    all_rejected = pd.concat(rejected_frames, ignore_index=True) if rejected_frames else pd.DataFrame()
    all_warnings = pd.concat(warning_frames, ignore_index=True) if warning_frames else pd.DataFrame()
    write_global_verified_outputs(results, output_dir, all_profiles, all_verifications, all_rejected, all_warnings)
    write_dataset_profile_inventory_and_provenance(output_dir, results)
    cleanup_summary = cleanup_archived_dataset_source_folders(output_dir.parent, output_dir)
    print_terminal_summary(results, all_verifications, all_rejected, all_warnings, discovery, cleanup_summary, output_dir)
    return results


def print_terminal_summary(
    results: Sequence[Dict[str, Any]],
    verifications: pd.DataFrame,
    rejected: pd.DataFrame,
    warnings: pd.DataFrame,
    discovery: scanner.DatasetDiscovery,
    cleanup_summary: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    output_dir: Optional[Path] = None,
) -> None:
    print("\nRaw datasets scanned:")
    for idx in range(1, 7):
        name = f"Dataset{idx}"
        found = discovery.datasets.get(name)
        result = next((r for r in results if (r.get("dataset") or r.get("Dataset")) == name), {})
        if found:
            print(f"{name}: {found.path} ({result.get('scan_status')})")
        else:
            print(f"{name}: {name} raw files not confidently detected.")

    print("\nVerified claims:")
    verified_count = int(verifications["decision"].eq("VERIFIED").sum()) if "decision" in verifications else 0
    print(f"{verified_count}")

    print("\nRejected claims:")
    print(len(rejected))

    print("\nConflicted claims:")
    conflicted = int(verifications["decision"].eq("CONFLICTED").sum()) if "decision" in verifications else 0
    print(conflicted)

    print("\nFields confirmed by raw scan:")
    for result in results:
        name = result.get("dataset") or result.get("Dataset")
        if result.get("scan_status") != "FAILED":
            print(f"{name}: date={result.get('date_col')}, sales={result.get('sales_col')}, entity={result.get('entity_cols')}")

    print("\nFields still incomplete:")
    if warnings.empty:
        print("None recorded.")
    else:
        print(f"{len(warnings)} unresolved warning/gap rows; see outputs/dataset_profiles/unresolved_issues.md")

    print("\nOld outputs used only as references:")
    print("Existing outputs/dataset_profiles files were snapshotted before scanning and used only for claim verification.")

    cleanup_summary = cleanup_summary or {"deleted": [], "skipped": []}
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    outputs_root = output_dir.resolve().parent
    print("\nCleanup completed.")
    print("\nDeleted archived dataset source folders:")
    if cleanup_summary.get("deleted"):
        for row in cleanup_summary["deleted"]:
            print(f"- {row.get('deleted_path')}")
    else:
        print("- None")

    print("\nKept final dataset profile folder:")
    try:
        print(str(output_dir.resolve().relative_to(PROJECT_ROOT)))
    except ValueError:
        print(str(output_dir.resolve()))

    print("\nSkipped folders:")
    if cleanup_summary.get("skipped"):
        for row in cleanup_summary["skipped"]:
            print(f"- {row.get('deleted_path')}: {row.get('notes')}")
    else:
        print("- None")

    print("\nSafety check:")
    print("All delete operations were restricted to:")
    print(str(outputs_root))
    print("\nNo files outside outputs were deleted.")
    print("No raw dataset files were deleted.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verified D1-D6 dataset profiles from raw scans.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="D1-D6 raw data root.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Dataset profile output directory.")
    parser.add_argument("--max-rows", type=int, default=100_000, help="Fallback row cap for non-chunked huge tables.")
    parser.add_argument("--chunk-size", type=int, default=100_000, help="Chunk size for full scans.")
    parser.add_argument("--max-probe-rows", type=int, default=1000, help="Rows used to score candidate raw files.")
    args = parser.parse_args()

    organize_dataset_profiles(
        data_root=Path(args.data_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        max_rows=args.max_rows,
        chunk_size=args.chunk_size,
        max_probe_rows=args.max_probe_rows,
    )


if __name__ == "__main__":
    main()
