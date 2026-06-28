#!/usr/bin/env python3
"""轻量验证脚本：检查 paper_results.csv / full_paper_results.csv / extended_results.csv
中是否包含并正确填充 KNN distance/weight 审计字段和 RFE candidate/removed 审计字段。

此脚本不需要重新训练，只检查已有 CSV 文件。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

CSV_PATHS = [
    ROOT / "outputs" / "experiment_results" / "paper_results.csv",
    ROOT / "outputs" / "experiment_results" / "full_paper_results.csv",
    ROOT / "outputs" / "experiment_results" / "extended_results.csv",
    # Also check the paper_alignment paths:
    ROOT / "outputs" / "paper_alignment" / "runs" / "paper_results.csv",
]

REQUIRED_COLUMNS = [
    "selected_sources",
    "selected_source_distances",
    "selected_source_weights_raw",
    "selected_source_weights_normalized",
]

RFE_COLUMNS = [
    "rfe_candidate_features",
    "rfe_removed_features",
]

ALL_REQUIRED_COLUMNS = REQUIRED_COLUMNS + RFE_COLUMNS

METHODS_WITH_KNN = {"SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"}
METHODS_WITHOUT_KNN = {"No-TL"}
METHOD_RFE = "MSML-TL-RFE"

METHODS_WITH_KNN = {"SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"}
METHODS_WITHOUT_KNN = {"No-TL"}


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _parse_pipe_float_list(value: Any) -> List[float]:
    """将管道拼接的字符串解析为浮点数列表。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return []
        return [float(value)]
    s = str(value).strip()
    if not s or s.lower() in ("", "nan", "none", "not_recorded_by_code", "not_applicable"):
        return []
    parts = s.split("|")
    result: List[float] = []
    for p in parts:
        p = p.strip()
        try:
            v = float(p)
            if not math.isnan(v):
                result.append(v)
        except (ValueError, TypeError):
            continue
    return result


def _count_pipe_parts(value: Any) -> int:
    """统计管道拼接的字段数量。"""
    if value is None:
        return 0
    s = str(value).strip()
    if not s or s.lower() in ("", "nan", "none", "not_recorded_by_code", "not_applicable"):
        return 0
    return len(s.split("|"))


def _check_file(path: Path) -> Dict[str, Any]:
    """检查单个 CSV 文件。"""
    report: Dict[str, Any] = {
        "path": str(path),
        "exists": False,
        "has_all_columns": False,
        "missing_columns": [],
        "total_rows": 0,
        "knn_rows": 0,
        "knn_rows_with_distances": 0,
        "consistency_errors": [],
        "warnings": [],
    }

    if not path.exists():
        report["warnings"].append(f"File not found: {path}")
        return report

    report["exists"] = True
    df = pd.read_csv(path)

    report["total_rows"] = len(df)

    # Check columns (both KNN and RFE)
    missing = [c for c in ALL_REQUIRED_COLUMNS if c not in df.columns]
    report["missing_columns"] = missing
    report["has_all_columns"] = len(missing) == 0

    if missing:
        report["warnings"].append(f"Missing columns: {missing}")
        # Continue checking other columns even if some are missing

    # Check KNN/TL method rows
    knn_df = df[df["method"].isin(METHODS_WITH_KNN)] if "method" in df.columns else pd.DataFrame()
    report["knn_rows"] = len(knn_df)

    if knn_df.empty:
        report["warnings"].append("No KNN/TL method rows found in this CSV.")

    for idx, row in knn_df.iterrows():
        method = row.get("method", "unknown")
        sources_str = str(row.get("selected_sources", ""))
        dist_str = str(row.get("selected_source_distances", ""))
        raw_str = str(row.get("selected_source_weights_raw", ""))
        norm_str = str(row.get("selected_source_weights_normalized", ""))

        n_sources = _count_pipe_parts(sources_str)
        n_dist = _count_pipe_parts(dist_str)
        n_raw = _count_pipe_parts(raw_str)
        n_norm = _count_pipe_parts(norm_str)

        # --- Check 1: selected_sources non-empty for KNN methods ---
        if n_sources == 0:
            report["consistency_errors"].append(
                f"Row {idx}: method={method}, selected_sources is empty but KNN method expected to have sources."
            )

        # --- Check 2: If distances non-empty, count must match ---
        if n_dist > 0:
            report["knn_rows_with_distances"] += 1
            if n_dist != n_sources:
                report["consistency_errors"].append(
                    f"Row {idx}: method={method}, "
                    f"distance count ({n_dist}) != source count ({n_sources})"
                )
            if n_raw != n_sources:
                report["consistency_errors"].append(
                    f"Row {idx}: method={method}, "
                    f"raw_weight count ({n_raw}) != source count ({n_sources})"
                )
            if n_norm != n_sources:
                report["consistency_errors"].append(
                    f"Row {idx}: method={method}, "
                    f"norm_weight count ({n_norm}) != source count ({n_sources})"
                )

            # --- Check 3: Weight consistency ---
            distances = _parse_pipe_float_list(dist_str)
            raw_weights = _parse_pipe_float_list(raw_str)
            norm_weights = _parse_pipe_float_list(norm_str)

            for i, (d, r) in enumerate(zip(distances, raw_weights)):
                if d <= 0:
                    continue
                expected_raw = 1.0 / d
                if abs(r - expected_raw) > max(1e-5 * expected_raw, 1e-10):
                    report["consistency_errors"].append(
                        f"Row {idx}: method={method}, source {i}: "
                        f"raw_weight {r:.10f} != 1/distance ({expected_raw:.10f})"
                    )

            if norm_weights:
                valid_norm = [w for w in norm_weights if not math.isnan(w)]
                if valid_norm:
                    norm_sum = sum(valid_norm)
                    if abs(norm_sum - 1.0) > 1e-6:
                        report["consistency_errors"].append(
                            f"Row {idx}: method={method}, "
                            f"normalized weight sum = {norm_sum:.10f} (expected ~1.0)"
                        )

            # --- Check 4: Order consistency ---
            if len(distances) == len(raw_weights) == len(norm_weights) and len(distances) > 0:
                for i in range(len(distances)):
                    if not math.isnan(norm_weights[i]) and not math.isnan(raw_weights[i]):
                        expected_norm = raw_weights[i] / sum(w for w in raw_weights if not math.isnan(w))
                        if abs(norm_weights[i] - expected_norm) > 1e-6:
                            report["consistency_errors"].append(
                                f"Row {idx}: method={method}, source {i}: "
                                f"norm_weight {norm_weights[i]:.10f} != raw/sum ({expected_norm:.10f})"
                            )

    # --- Check 5: No-TL rows should have empty KNN/distance fields ---
    no_tl_df = df[df["method"].isin(METHODS_WITHOUT_KNN)] if "method" in df.columns else pd.DataFrame()
    for idx, row in no_tl_df.iterrows():
        for col in REQUIRED_COLUMNS[1:]:  # Skip selected_sources (may be empty)
            val = str(row.get(col, "")).strip()
            if val and val.lower() not in ("", "nan", "none", "not_recorded_by_code", "not_applicable"):
                report["warnings"].append(
                    f"Row {idx}: method=No-TL, {col} is '{val}' but expected empty for No-TL."
                )

    # --- Check 6: RFE candidate / removed feature fields ---
    rfe_df = df[df["method"] == METHOD_RFE] if "method" in df.columns else pd.DataFrame()
    if not rfe_df.empty and all(c in df.columns for c in RFE_COLUMNS):
        for idx, row in rfe_df.iterrows():
            candidate_str = str(row.get("rfe_candidate_features", ""))
            selected_str = str(row.get("rfe_selected_features", ""))
            removed_str = str(row.get("rfe_removed_features", ""))
            final_str = str(row.get("selected_features", ""))

            candidate_parts = [p.strip() for p in candidate_str.split("|") if p.strip()]
            selected_parts = [p.strip() for p in selected_str.split("|") if p.strip()]
            removed_parts = [p.strip() for p in removed_str.split("|") if p.strip()]
            final_parts = [p.strip() for p in final_str.split("|") if p.strip()]

            # rfe_selected_features must be non-empty for MSML-TL-RFE
            if not selected_parts:
                report["consistency_errors"].append(
                    f"Row {idx}: method={METHOD_RFE}, rfe_selected_features is empty."
                )

            # rfe_candidate_features must be non-empty for MSML-TL-RFE
            if not candidate_parts:
                report["consistency_errors"].append(
                    f"Row {idx}: method={METHOD_RFE}, rfe_candidate_features is empty."
                )

            # --- Set relationship check: selected ⊆ candidate ---
            if candidate_parts and selected_parts:
                candidate_set = set(candidate_parts)
                selected_set = set(selected_parts)
                if not selected_set.issubset(candidate_set):
                    extra = selected_set - candidate_set
                    report["consistency_errors"].append(
                        f"Row {idx}: method={METHOD_RFE}, "
                        f"rfe_selected_features contains features not in rfe_candidate_features: {extra}"
                    )

            # --- Set relationship: removed == candidate - selected ---
            if candidate_parts and selected_parts:
                candidate_set = set(candidate_parts)
                selected_set = set(selected_parts)
                removed_set = set(removed_parts) if removed_parts else set()
                expected_removed = candidate_set - selected_set
                if removed_set != expected_removed:
                    report["consistency_errors"].append(
                        f"Row {idx}: method={METHOD_RFE}, "
                        f"rfe_removed_features ({removed_set}) != candidate - selected ({expected_removed})"
                    )

            # --- Sales semantic check: sales should NOT be in candidate (it's excluded from RFE) ---
            if candidate_parts and "sales" in candidate_parts:
                report["consistency_errors"].append(
                    f"Row {idx}: method={METHOD_RFE}, "
                    f"'sales' found in rfe_candidate_features but should be excluded from RFE candidate pool."
                )
            if removed_parts and "sales" in removed_parts:
                report["consistency_errors"].append(
                    f"Row {idx}: method={METHOD_RFE}, "
                    f"'sales' found in rfe_removed_features but sales is not an RFE candidate."
                )

            # Sales should appear in selected_features (added back as history input)
            if final_parts and "sales" not in final_parts:
                report["warnings"].append(
                    f"Row {idx}: method={METHOD_RFE}, "
                    f"'sales' not found in selected_features. "
                    f"Expected sales to be added back as fixed history input."
                )
    elif not rfe_df.empty:
        missing_rfe = [c for c in RFE_COLUMNS if c not in df.columns]
        if missing_rfe:
            report["warnings"].append(
                f"RFE columns missing, skipping RFE-specific checks: {missing_rfe}"
            )

    return report


def main() -> int:
    print("=" * 70)
    print("KNN Distance/Weight CSV Field Verification")
    print("=" * 70)

    found_any = False
    all_ok = True
    reports: List[Dict[str, Any]] = []

    for csv_path in CSV_PATHS:
        print(f"\n--- Checking: {csv_path} ---")
        report = _check_file(csv_path)
        reports.append(report)

        if not report["exists"]:
            print(f"  ⚠️  File not found (may not have been generated yet)")
            continue

        found_any = True
        status = "✅" if report["has_all_columns"] else "❌"
        print(f"  Columns: {status}")
        if report["missing_columns"]:
            print(f"    Missing: {report['missing_columns']}")

        print(f"  Total rows: {report['total_rows']}")
        print(f"  KNN/TL rows: {report['knn_rows']}")
        print(f"  KNN/TL rows with distances: {report['knn_rows_with_distances']}")

        if report["consistency_errors"]:
            all_ok = False
            print(f"  ❌ Consistency errors: {len(report['consistency_errors'])}")
            for err in report["consistency_errors"][:5]:
                print(f"    - {err}")
            if len(report["consistency_errors"]) > 5:
                print(f"    ... and {len(report['consistency_errors']) - 5} more")
        else:
            print(f"  ✅ No consistency errors")

        if report["warnings"]:
            print(f"  ⚠️  Warnings: {len(report['warnings'])}")
            for w in report["warnings"][:3]:
                print(f"    - {w}")

    print("\n" + "=" * 70)
    if not found_any:
        print("⚠️  No CSV files found. Run experiments first to generate them.")
        print("   Then re-run this verification script.")
        return 0

    if all_ok:
        print("✅ All checks passed!")
        return 0
    else:
        print("❌ Some checks failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
