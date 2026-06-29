"""Run a unified audit for paper protocol alignment.

Outputs:
- outputs/paper_alignment/paper_protocol_audit.json
- outputs/paper_alignment/paper_protocol_audit.md
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.experiment_runner import prepare_base_data_for_experiments
from src.data_processing.data_preprocessing import temporal_split_by_ratio_or_dates
from paper_reproduction_protocol import (
    MULTI_SOURCE_TL_METHODS,
    assess_metric_alignment,
    assess_source_pretrained_alignment,
    assess_split_alignment,
    get_extended_source_counts,
    get_paper_source_counts,
    load_paper_protocol,
    resolve_experiment_track,
    validate_paper_protocol_config,
)


PASS = "PASS"
PARTIAL = "PARTIAL"
FAIL = "FAIL"

PARTIAL_CODE_RESULT_GAP = "code_or_result_not_closed_loop"
PARTIAL_EXTERNAL_GAP = "external_evidence_or_metadata_missing"


def _load_config() -> Dict[str, Any]:
    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_todo_like(value: Any) -> bool:
    text = str(value or "").strip().upper()
    if not text:
        return True
    return text.startswith("TODO") or "UNCONFIRMED" in text


def _map_status_to_level(status: str, has_evidence: bool) -> str:
    raw = str(status or "").strip().upper()
    if raw == FAIL:
        return FAIL
    if raw in {"PARTIAL", "TODO", "EXTENDED", "WARN", "WARNING"}:
        return PARTIAL
    if raw in {"ALIGNED", "PASS", "NOT_APPLICABLE"}:
        return PASS if has_evidence else PARTIAL
    return PARTIAL


def _collect_metric_alignment(protocol: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    metric_cfg = dict(protocol.get("metric_protocol", {}))
    metric_status = assess_metric_alignment(protocol)

    unresolved: List[str] = []
    fixes: List[str] = []

    has_metric_space_evidence = not _is_todo_like(metric_status.get("paper_metric_space"))
    has_accuracy_evidence = not _is_todo_like(metric_cfg.get("paper_accuracy_definition"))
    has_evidence = bool(has_metric_space_evidence and has_accuracy_evidence)

    runtime_metric = _assess_metric_runtime_closure()

    if not has_metric_space_evidence:
        unresolved.append("metric_space 缺少可证实的论文证据，当前仅可判定为部分一致。")
        fixes.append("补充论文原文/附录/作者代码中对 metric_space 的明确定义并写入配置。")
    if not has_accuracy_evidence:
        unresolved.append("Accuracy 定义缺少可证实的论文证据，当前仅可判定为部分一致。")
        fixes.append("补充论文对 Accuracy 的精确定义，并在配置中替换 TODO 占位值。")

    raw_status = str(metric_status.get("metric_alignment_status", "PARTIAL"))
    if runtime_metric["runtime_checked"]:
        if runtime_metric["runtime_pass"]:
            level = PASS if has_evidence else PARTIAL
        else:
            level = FAIL
            unresolved.append("metric 结果层未完全闭环：存在结果行未满足论文度量口径。")
            fixes.append("检查 full_paper_results.csv 中 paper_metric_aligned / inverse_transform_applied 标记并修复评估链路。")
    else:
        level = _map_status_to_level(raw_status, has_evidence)
        unresolved.append("metric 未找到可用结果层审计文件，暂按配置状态评估。")
        fixes.append("先生成 outputs/experiment_results/full_paper_results.csv 再执行统一审计。")

    result = {
        "level": level,
        "raw_status": raw_status,
        "has_evidence": has_evidence,
        "runtime_metric_closure": runtime_metric,
        "details": {
            "paper_metric_space": metric_status.get("paper_metric_space"),
            "current_metric_space": metric_status.get("current_metric_space"),
            "strict_paper_metrics": metric_status.get("strict_paper_metrics"),
            "metric_alignment_notes": metric_status.get("metric_alignment_notes"),
            "paper_accuracy_definition": metric_cfg.get("paper_accuracy_definition", "TODO_UNCONFIRMED"),
        },
    }
    return result, unresolved, fixes


def _collect_split_alignment(cfg: Dict[str, Any], protocol: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    split_cfg = dict(protocol.get("split_protocol", {}))
    paper_reference = split_cfg.get("paper_reference", "TODO_UNCONFIRMED")
    strict_split_enabled = _strict_split_enabled(cfg)

    unresolved: List[str] = []
    fixes: List[str] = []
    dataset_rows: List[Dict[str, Any]] = []

    for dataset_name, dataset_path in cfg.get("dataset_paths", {}).items():
        base = prepare_base_data_for_experiments(
            dataset_name=dataset_name,
            data_path=dataset_path,
            config=cfg,
            verbose_mode="summary",
        )
        split = _assess_split_alignment_runtime(
            cfg=cfg,
            dataset_name=dataset_name,
            base_data=base,
            strict_split_enabled=strict_split_enabled,
            day_tolerance=3,
        )
        dataset_rows.append(
            {
                "dataset": dataset_name,
                "raw_status": split.get("split_alignment_status", "PARTIAL"),
                "target_start_date": split.get("target_start_date"),
                "target_end_date": split.get("target_end_date"),
                "target_window_days": split.get("target_window_days"),
                "target_window_expected_days": split.get("target_window_expected_days"),
                "target_window_range_days": split.get("target_window_range_days"),
                "target_window_unique_days": split.get("target_window_unique_days"),
                "target_split_mode": split.get("target_split_mode"),
                "source_split_mode": split.get("source_split_mode"),
                "split_runtime_matches_config": split.get("split_runtime_matches_config"),
                "strict_split_enabled": split.get("strict_split_enabled"),
                "expected_train_days": split.get("expected_train_days"),
                "actual_train_days": split.get("actual_train_days"),
                "expected_val_days": split.get("expected_val_days"),
                "actual_val_days": split.get("actual_val_days"),
                "expected_test_days": split.get("expected_test_days"),
                "actual_test_days": split.get("actual_test_days"),
                "split_alignment_notes": split.get("split_alignment_notes"),
            }
        )

    has_evidence = not _is_todo_like(paper_reference)
    if not has_evidence:
        unresolved.append("数据切分缺少可证实的论文绝对窗口证据，当前仅按相对窗口复刻。")
        fixes.append("补充论文绝对日期边界或作者官方切分实现证据，并更新 paper_reference。")

    per_dataset_levels = [PASS if str(row.get("raw_status", "PARTIAL")).upper() == PASS else FAIL for row in dataset_rows]
    if any(level == FAIL for level in per_dataset_levels):
        level = FAIL
    elif not has_evidence:
        level = PARTIAL
    else:
        level = PASS

    result = {
        "level": level,
        "paper_reference": paper_reference,
        "has_evidence": has_evidence,
        "strict_split_enabled": strict_split_enabled,
        "evaluation_basis": "strict_dataset_target_split_days" if strict_split_enabled else "paper_split_protocol_default_window",
        "datasets": dataset_rows,
    }
    return result, unresolved, fixes


def _collect_source_alignment(cfg: Dict[str, Any], protocol: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    source_cfg = dict(protocol.get("paper_source_protocol", {}))
    cap = int(source_cfg.get("max_pretrained_models_from_paper", 5))
    paper_counts = [int(v) for v in get_paper_source_counts(protocol)]
    extended_counts = [int(v) for v in get_extended_source_counts(protocol)]

    unresolved: List[str] = []
    fixes: List[str] = []
    rows: List[Dict[str, Any]] = []

    methods = ["No-TL", "SS-TL", *sorted(MULTI_SOURCE_TL_METHODS)]
    for method_name in methods:
        if method_name == "No-TL":
            requested_counts = [0]
        elif method_name == "SS-TL":
            requested_counts = [1]
        else:
            requested_counts = paper_counts + extended_counts

        for requested_count in requested_counts:
            track = resolve_experiment_track(method_name, int(requested_count), protocol)
            actual_count = 0 if method_name == "No-TL" else (1 if method_name == "SS-TL" else int(requested_count))
            status = assess_source_pretrained_alignment(
                method_name=method_name,
                requested_source_count=int(requested_count),
                actual_pretrained_model_count=actual_count,
                protocol=protocol,
                experiment_track=track,
            )
            rows.append(
                {
                    "method": method_name,
                    "requested_source_count": int(requested_count),
                    "actual_pretrained_model_count": int(actual_count),
                    "experiment_track": track,
                    "raw_status": status.get("source_pretrained_alignment_status", "PARTIAL"),
                    "notes": status.get("source_pretrained_alignment_notes", ""),
                }
            )

    # Source protocol has stronger explicit paper wording in the current repository.
    has_evidence = True
    if cap != 5:
        unresolved.append("max_pretrained_models_from_paper 不是 5，违反论文主约束。")
        fixes.append("将 max_pretrained_models_from_paper 恢复为 5。")
        has_evidence = False
    if any(v > 5 for v in paper_counts):
        unresolved.append("paper_source_counts 包含超过 5 的配置。")
        fixes.append("将 paper_source_counts 限制在不超过 5 的集合。")
        has_evidence = False

    row_levels: List[str] = []
    for row in rows:
        raw_status = str(row.get("raw_status", "PARTIAL")).upper()
        # Extended-track rows are expected in a mixed paper+extended configuration
        # and should not downgrade paper-protocol alignment by themselves.
        if raw_status == "EXTENDED":
            row_levels.append(PASS)
        else:
            row_levels.append(_map_status_to_level(raw_status, has_evidence))
    if any(level == FAIL for level in row_levels):
        level = FAIL
    elif any(level == PARTIAL for level in row_levels):
        level = PARTIAL
    else:
        level = PASS

    region_gap = _detect_dataset3_region_metadata_gap(cfg)
    if region_gap["present"] and level != FAIL:
        level = PARTIAL
        unresolved.append(region_gap["message"])
        fixes.append("补充 Dataset3 门店 region 元数据后，启用 same_region 严格过滤并重新生成 source_identification 报告。")

    result = {
        "level": level,
        "has_evidence": has_evidence,
        "paper_cap": cap,
        "paper_source_counts": paper_counts,
        "extended_source_counts": extended_counts,
        "dataset3_region_metadata_gap": region_gap,
        "checks": rows,
    }
    return result, unresolved, fixes


def _strict_split_enabled(cfg: Dict[str, Any]) -> bool:
    paper_cfg = cfg.get("paper_reproduction", {})
    return bool(
        paper_cfg.get("strict_paper_split", False)
        or paper_cfg.get("paper_strict_split", False)
        or paper_cfg.get("strict_paper_mode", False)
        or paper_cfg.get("paper_strict_mode", False)
    )


def _nunique_days(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df["date"].nunique())


def _ordered_time_ok(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    if train_df.empty or val_df.empty or test_df.empty:
        return False
    train_end = train_df["date"].max()
    val_start = val_df["date"].min()
    val_end = val_df["date"].max()
    test_start = test_df["date"].min()
    return bool(train_end < val_start <= val_end < test_start)


def _resolve_expected_split_days(
    cfg: Dict[str, Any],
    dataset_name: str,
    strict_split_enabled: bool,
) -> Dict[str, int]:
    split_cfg = cfg.get("paper_reproduction", {}).get("paper_split_protocol", {})
    default_observed_days = int(split_cfg.get("target_observed_window_days", 30))
    default_forecast_days = int(split_cfg.get("target_forecast_window_days", 180))

    strict_ds_cfg = cfg.get("paper_reproduction", {}).get("strict_dataset_protocol", {})
    ds_cfg = strict_ds_cfg.get(dataset_name, {})
    split_days = ds_cfg.get("target_split_days", {}) if strict_split_enabled else {}

    expected_train_days = int(split_days.get("train_days", max(default_observed_days - 15, 1)))
    expected_val_days = int(split_days.get("val_days", min(15, default_observed_days)))
    expected_test_days = int(split_days.get("test_days", default_forecast_days))
    return {
        "expected_train_days": expected_train_days,
        "expected_val_days": expected_val_days,
        "expected_test_days": expected_test_days,
        "expected_observed_days": expected_train_days + expected_val_days,
    }


def _assess_split_alignment_runtime(
    cfg: Dict[str, Any],
    dataset_name: str,
    base_data: Dict[str, Any],
    strict_split_enabled: bool,
    day_tolerance: int,
) -> Dict[str, Any]:
    target_df = base_data.get("target_df")
    source_df = base_data.get("source_df")
    if target_df is None or source_df is None or target_df.empty:
        return {
            "split_alignment_status": FAIL,
            "target_start_date": "N/A",
            "target_end_date": "N/A",
            "target_window_days": -1,
            "target_window_expected_days": -1,
            "target_window_range_days": -1,
            "target_window_unique_days": -1,
            "target_split_mode": "unknown",
            "source_split_mode": "unknown",
            "split_runtime_matches_config": False,
            "strict_split_enabled": strict_split_enabled,
            "expected_train_days": -1,
            "actual_train_days": -1,
            "expected_val_days": -1,
            "actual_val_days": -1,
            "expected_test_days": -1,
            "actual_test_days": -1,
            "split_alignment_notes": "target/source 数据为空，无法校验 split。",
        }

    expected = _resolve_expected_split_days(
        cfg=cfg,
        dataset_name=dataset_name,
        strict_split_enabled=strict_split_enabled,
    )
    target_train, target_val, target_test = temporal_split_by_ratio_or_dates(target_df)

    actual_train_days = _nunique_days(target_train)
    actual_val_days = _nunique_days(target_val)
    actual_test_days = _nunique_days(target_test)
    actual_observed_days = _nunique_days(pd.concat([target_train, target_val], ignore_index=True))

    train_ok = abs(actual_train_days - expected["expected_train_days"]) <= day_tolerance
    val_ok = abs(actual_val_days - expected["expected_val_days"]) <= day_tolerance
    test_ok = abs(actual_test_days - expected["expected_test_days"]) <= day_tolerance
    observed_ok = abs(actual_observed_days - expected["expected_observed_days"]) <= day_tolerance
    validation_ok = _ordered_time_ok(target_train, target_val, target_test)

    status = PASS if all([train_ok, val_ok, test_ok, observed_ok, validation_ok]) else FAIL

    window_days = int((target_df["date"].max() - target_df["date"].min()).days + 1)
    expected_total_days = int(expected["expected_observed_days"] + expected["expected_test_days"])
    range_days = int(target_df.attrs.get("target_window_range_days", window_days))
    unique_days = int(target_df.attrs.get("target_window_unique_days", _nunique_days(target_df)))
    start_date = pd.Timestamp(target_df["date"].min()).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(target_df["date"].max()).strftime("%Y-%m-%d")

    return {
        "split_alignment_status": status,
        "target_start_date": start_date,
        "target_end_date": end_date,
        "target_window_days": window_days,
        "target_window_expected_days": expected_total_days,
        "target_window_range_days": range_days,
        "target_window_unique_days": unique_days,
        "target_split_mode": str(target_df.attrs.get("split_mode", "unknown")),
        "source_split_mode": str(source_df.attrs.get("split_mode", "unknown")),
        "split_runtime_matches_config": bool(status == PASS),
        "strict_split_enabled": strict_split_enabled,
        "expected_train_days": expected["expected_train_days"],
        "actual_train_days": actual_train_days,
        "expected_val_days": expected["expected_val_days"],
        "actual_val_days": actual_val_days,
        "expected_test_days": expected["expected_test_days"],
        "actual_test_days": actual_test_days,
        "split_alignment_notes": "strict dataset-specific split 校验" if strict_split_enabled else "paper split 默认窗口校验",
    }


def _assess_metric_runtime_closure() -> Dict[str, Any]:
    results_path = ROOT / "outputs" / "experiment_results" / "full_paper_results.csv"
    if not results_path.exists():
        return {
            "runtime_checked": False,
            "runtime_pass": False,
            "checked_rows": 0,
            "pass_rows": 0,
            "fail_rows": 0,
            "source": str(results_path),
            "note": "results file missing",
        }

    df = pd.read_csv(results_path)
    if df.empty:
        return {
            "runtime_checked": False,
            "runtime_pass": False,
            "checked_rows": 0,
            "pass_rows": 0,
            "fail_rows": 0,
            "source": str(results_path),
            "note": "results file empty",
        }

    paper_metric_aligned = df.get("paper_metric_aligned", False).astype(str).str.lower().isin({"1", "true", "yes", "y"})
    inverse_transform_applied = df.get("inverse_transform_applied", False).astype(str).str.lower().isin({"1", "true", "yes", "y"})
    metric_space_current = df.get("metric_space_current", "").astype(str)
    metric_space_paper = df.get("metric_space_paper", "").astype(str)

    is_paper_original_metric = paper_metric_aligned & (
        (metric_space_current == metric_space_paper) | inverse_transform_applied
    )
    checked_rows = int(len(df))
    pass_rows = int(is_paper_original_metric.sum())

    return {
        "runtime_checked": True,
        "runtime_pass": bool(pass_rows == checked_rows),
        "checked_rows": checked_rows,
        "pass_rows": pass_rows,
        "fail_rows": int(checked_rows - pass_rows),
        "source": str(results_path),
        "note": "computed from full_paper_results.csv",
    }


def _detect_dataset3_region_metadata_gap(cfg: Dict[str, Any]) -> Dict[str, Any]:
    strict_ds = cfg.get("paper_reproduction", {}).get("strict_dataset_protocol", {}).get("Dataset3", {})
    status = str(strict_ds.get("region_alignment_status", "")).upper()
    todo = str(strict_ds.get("region_alignment_todo", "")).strip()

    fallback_note = ""
    source_report = ROOT / "outputs" / "paper_alignment" / "source_identification_report.csv"
    if source_report.exists():
        try:
            sr = pd.read_csv(source_report)
            ds3 = sr[sr.get("dataset").astype(str) == "Dataset3"] if "dataset" in sr.columns else pd.DataFrame()
            if not ds3.empty and "source_pool_scope_note" in ds3.columns:
                notes = [str(v).strip() for v in ds3["source_pool_scope_note"].dropna().tolist() if str(v).strip()]
                fallback_note = notes[0] if notes else ""
        except Exception:
            fallback_note = ""

    present = bool(status == PARTIAL or _is_todo_like(todo) or "region" in fallback_note.lower())
    message = "Dataset3 region 元数据缺失：without-information-sharing 仍使用 region fallback（外部元数据缺口）。"
    if fallback_note:
        message = f"{message} 证据: {fallback_note}"

    return {
        "present": present,
        "status": status or "UNKNOWN",
        "todo": todo,
        "fallback_note": fallback_note,
        "message": message,
    }


def _categorize_partial_reasons(audit: Dict[str, Any]) -> Dict[str, List[str]]:
    code_or_result: List[str] = []
    external: List[str] = []

    metric = audit.get("metric_alignment", {})
    split = audit.get("split_alignment", {})
    source = audit.get("source_protocol_alignment", {})

    if str(metric.get("level", "")).upper() == FAIL:
        code_or_result.append("metric: 结果层存在未满足论文度量口径的实验行。")
    elif str(metric.get("level", "")).upper() == PARTIAL:
        if metric.get("has_evidence", False):
            code_or_result.append("metric: 结果层闭环状态不完整或未产出可验证结果。")
        else:
            external.append("metric: 外部论文证据不足（metric_space / accuracy 定义仍待确认）。")

    split_rows = split.get("datasets", [])
    split_fail_datasets = [r.get("dataset", "N/A") for r in split_rows if str(r.get("raw_status", "")).upper() == FAIL]
    if split_fail_datasets:
        code_or_result.append("split: 数据切分与期望窗口不一致（{}）。".format(", ".join(split_fail_datasets)))
    elif str(split.get("level", "")).upper() == PARTIAL:
        external.append("split: 外部论文绝对窗口证据不足（paper_reference 未确认）。")

    region_gap = source.get("dataset3_region_metadata_gap", {})
    if bool(region_gap.get("present", False)):
        external.append(str(region_gap.get("message", "Dataset3 region 元数据缺失")))

    if str(source.get("level", "")).upper() == PARTIAL and not bool(region_gap.get("present", False)):
        code_or_result.append("source_protocol: 选源/预训练协议存在未闭环项。")

    return {
        PARTIAL_CODE_RESULT_GAP: list(dict.fromkeys(code_or_result)),
        PARTIAL_EXTERNAL_GAP: list(dict.fromkeys(external)),
    }


def _overall_level(levels: List[str]) -> str:
    normalized = [str(v).upper() for v in levels]
    if FAIL in normalized:
        return FAIL
    if PARTIAL in normalized:
        return PARTIAL
    return PASS


def _write_markdown(path: Path, audit: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# 论文协议统一审计报告")
    lines.append("")
    lines.append(f"- 生成时间: {audit['generated_at']}")
    lines.append(f"- 审计结论: {audit['overall_level']}")
    lines.append("")
    lines.append("## 1. metric_alignment")
    lines.append("")
    metric = audit["metric_alignment"]
    lines.append(f"- 结论: {metric['level']}")
    lines.append(f"- 原始状态: {metric['raw_status']}")
    lines.append(f"- 证据充分: {metric['has_evidence']}")
    lines.append(f"- paper_metric_space: {metric['details'].get('paper_metric_space')}")
    lines.append(f"- current_metric_space: {metric['details'].get('current_metric_space')}")
    lines.append(f"- strict_paper_metrics: {metric['details'].get('strict_paper_metrics')}")
    lines.append(f"- notes: {metric['details'].get('metric_alignment_notes')}")
    runtime_metric = metric.get("runtime_metric_closure", {})
    lines.append(
        f"- runtime_metric_closure: checked={runtime_metric.get('runtime_checked')} pass={runtime_metric.get('runtime_pass')} rows={runtime_metric.get('pass_rows')}/{runtime_metric.get('checked_rows')}"
    )
    lines.append("")
    lines.append("## 2. split_alignment")
    lines.append("")
    split = audit["split_alignment"]
    lines.append(f"- 结论: {split['level']}")
    lines.append(f"- paper_reference: {split['paper_reference']}")
    lines.append(f"- 证据充分: {split['has_evidence']}")
    lines.append(f"- strict_split_enabled: {split.get('strict_split_enabled')}")
    lines.append(f"- 评估基准: {split.get('evaluation_basis')}")
    lines.append("")
    lines.append("| dataset | raw_status | target_start_date | target_end_date | target_window_days | expected_days | runtime_matches_config |")
    lines.append("|---|---|---|---|---:|---:|---|")
    for row in split.get("datasets", []):
        lines.append(
            "| {dataset} | {raw_status} | {target_start_date} | {target_end_date} | {target_window_days} | {target_window_expected_days} | {split_runtime_matches_config} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("## 3. source_protocol_alignment")
    lines.append("")
    source = audit["source_protocol_alignment"]
    lines.append(f"- 结论: {source['level']}")
    lines.append(f"- 证据充分: {source['has_evidence']}")
    lines.append(f"- paper_cap: {source['paper_cap']}")
    lines.append(f"- paper_source_counts: {source['paper_source_counts']}")
    lines.append(f"- extended_source_counts: {source['extended_source_counts']}")
    region_gap = source.get("dataset3_region_metadata_gap", {})
    lines.append(f"- Dataset3 region 元数据缺口: {region_gap.get('present')}")
    if region_gap.get("present"):
        lines.append(f"- Dataset3 region 说明: {region_gap.get('message')}")
    lines.append("")
    lines.append("## 4. partial_reason_breakdown")
    lines.append("")
    reasons = audit.get("partial_reason_breakdown", {})
    lines.append(f"- {PARTIAL_CODE_RESULT_GAP}:")
    code_items = reasons.get(PARTIAL_CODE_RESULT_GAP, [])
    if code_items:
        for item in code_items:
            lines.append(f"  - {item}")
    else:
        lines.append("  - 无")
    lines.append(f"- {PARTIAL_EXTERNAL_GAP}:")
    external_items = reasons.get(PARTIAL_EXTERNAL_GAP, [])
    if external_items:
        for item in external_items:
            lines.append(f"  - {item}")
    else:
        lines.append("  - 无")
    lines.append("")
    lines.append("## 5. unresolved_items")
    lines.append("")
    unresolved = audit.get("unresolved_items", [])
    if unresolved:
        for item in unresolved:
            lines.append(f"- {item}")
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("## 6. recommended_fixes")
    lines.append("")
    fixes = audit.get("recommended_fixes", [])
    if fixes:
        for item in fixes:
            lines.append(f"- {item}")
    else:
        lines.append("- 无")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg = _load_config()
    protocol = load_paper_protocol(cfg)

    if _strict_split_enabled(cfg):
        cfg.setdefault("paper_reproduction", {})["strict_paper_mode"] = True
        cfg["paper_reproduction"]["paper_strict_mode"] = True
        cfg["paper_reproduction"]["strict_paper_split"] = True
        cfg["paper_reproduction"]["paper_strict_split"] = True

    unresolved_items: List[str] = []
    recommended_fixes: List[str] = []

    config_validation = validate_paper_protocol_config(protocol=protocol, strict_paper_mode=False)
    for msg in config_validation.get("failures", []):
        unresolved_items.append(f"配置失败: {msg}")
        recommended_fixes.append(f"修复配置失败项: {msg}")
    for msg in config_validation.get("warnings", []):
        unresolved_items.append(f"配置警告: {msg}")

    metric_alignment, metric_unresolved, metric_fixes = _collect_metric_alignment(protocol)
    split_alignment, split_unresolved, split_fixes = _collect_split_alignment(cfg, protocol)
    source_alignment, source_unresolved, source_fixes = _collect_source_alignment(cfg, protocol)

    unresolved_items.extend(metric_unresolved)
    unresolved_items.extend(split_unresolved)
    unresolved_items.extend(source_unresolved)
    recommended_fixes.extend(metric_fixes)
    recommended_fixes.extend(split_fixes)
    recommended_fixes.extend(source_fixes)

    # Deduplicate while preserving order.
    unresolved_items = list(dict.fromkeys(unresolved_items))
    recommended_fixes = list(dict.fromkeys(recommended_fixes))

    overall = _overall_level(
        [metric_alignment["level"], split_alignment["level"], source_alignment["level"]]
    )

    partial_reason_breakdown = _categorize_partial_reasons(
        {
            "metric_alignment": metric_alignment,
            "split_alignment": split_alignment,
            "source_protocol_alignment": source_alignment,
        }
    )

    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_level": overall,
        "metric_alignment": metric_alignment,
        "split_alignment": split_alignment,
        "source_protocol_alignment": source_alignment,
        "partial_reason_breakdown": partial_reason_breakdown,
        "unresolved_items": unresolved_items,
        "recommended_fixes": recommended_fixes,
    }

    out_dir = ROOT / "outputs" / "paper_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "paper_protocol_audit.json"
    out_md = out_dir / "paper_protocol_audit.md"

    out_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out_md, audit)

    print("Paper protocol audit completed")
    print(out_json)
    print(out_md)
    print(f"Overall level: {overall}")


if __name__ == "__main__":
    main()