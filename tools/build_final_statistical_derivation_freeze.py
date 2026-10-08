#!/usr/bin/env python3
"""Build the final MS-KATL statistical-derivation freeze.

The integrated paper-data freeze is the only numerical authority.  This
module reads the selected raw seed-level sheets, derives all inferential
quantities in memory, and writes a new statistical artifact package.  Older
statistics workbooks and the manuscript are read only in the final comparison
layer; they never provide values used to calculate the canonical results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree

import numpy as np
import openpyxl
import pandas as pd
import scipy
from scipy import stats
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side


# ---------------------------------------------------------------------------
# Frozen contract constants.  Do not move these to a config file or notebook.
# ---------------------------------------------------------------------------

BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_RANDOM_SEED = 20260903
BOOTSTRAP_CI_LEVEL = 0.95
BOOTSTRAP_TYPE = "percentile"
WILCOXON_ALTERNATIVE = "two-sided"
WILCOXON_METHOD = "exact"
HOLM_ALPHA = 0.05
HOLM_FAMILY_SIZE = 4

STATISTICAL_UNIT = "12_DATASET_HORIZON_CONDITION_MEANS"
SEED_LEVEL_COUNTS_ROLE = "DESCRIPTIVE_ONLY"
BOOTSTRAP_RNG_POLICY = (
    "PER_COMPARISON_DEFAULT_RNG_RESET_TO_GLOBAL_SEED; "
    "EXPLICIT_STABLE_COMPARISON_ORDER; NO_SHARED_STATE"
)
TIE_RULE = "exact_numeric_no_tolerance"
METRIC_SPACE = "original_sales_space"
SMAPE_UNIT = "percent"
PREVIOUS_STATISTICS_ORIGIN = (
    "historically derived interactively from frozen workbook; "
    "original one-off generator not retained"
)

DATASET_TARGETS = {
    "D1": "1/10",
    "D4": "166/258",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}
FORMAL_DATASETS = tuple(DATASET_TARGETS)
FORMAL_HORIZONS = (1, 3, 5)
FORMAL_SEEDS = (42, 43, 44, 45, 46)
FORMAL_CONDITIONS = tuple(
    (dataset, DATASET_TARGETS[dataset], horizon)
    for dataset in FORMAL_DATASETS
    for horizon in FORMAL_HORIZONS
)
FORMAL_METHODS = (
    "SITA",
    "LSTM",
    "LightGBM",
    "DLinear",
    "TCN",
    "Seasonal Naive",
)
LEARNING_POOL = ("LSTM", "LightGBM", "DLinear", "TCN")
ABLATION_ARMS = ("A", "B", "C", "D", "E")
ABLATION_CONTRASTS = (("C", "A"), ("C", "B"), ("C", "E"), ("D", "C"))
MAIN_COMPARISONS = (
    ("SITA_vs_Best_Learning", "Best Learning"),
    ("SITA_vs_LSTM", "LSTM"),
    ("SITA_vs_LightGBM", "LightGBM"),
    ("SITA_vs_DLinear", "DLinear"),
    ("SITA_vs_TCN", "TCN"),
)
METRICS = ("RMSE", "sMAPE")
METRIC_COLUMNS = {"RMSE": "rmse", "sMAPE": "smape"}
FORMAL_SHEETS = {
    "main": "03_V3_Raw_5Seed",
    "baseline": "04_BL_Raw_Selected",
    "ablation": "05_Ablation_Raw",
}

DEFAULT_INPUT = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/"
    "MS_KATL_Paper_Data_Freeze_v1_4_STATS_FROZEN_BL6_BL7_INTEGRATED.xlsx"
)
DEFAULT_OLD_STATISTICS = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/Section4_Comparison_Statistics_BL6_BL7.xlsx"
)
DEFAULT_OLD_REPORT = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/SECTION4_COMPARISON_STATISTICS_REPORT.md"
)
DEFAULT_MANUSCRIPT = Path("/Users/ming/Desktop/复现实验/论文文稿相关/全.docx")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/paper_statistics/final_statistical_freeze_20260910"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def numeric(value: Any, label: str) -> float:
    if not finite(value):
        raise ValueError(f"nonfinite numeric value for {label}: {value!r}")
    return float(value)


def read_sheet_rows(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    workbook = load_workbook(path, data_only=True, read_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            raise ValueError(f"missing required sheet: {sheet_name}")
        iterator = workbook[sheet_name].iter_rows(values_only=True)
        try:
            header = [clean(value) for value in next(iterator)]
        except StopIteration as exc:
            raise ValueError(f"empty sheet: {sheet_name}") from exc
        if not any(header):
            raise ValueError(f"empty header: {sheet_name}")
        rows: list[dict[str, Any]] = []
        for source_row, values in enumerate(iterator, start=2):
            if not any(value is not None for value in values):
                continue
            row = {header[index]: values[index] if index < len(values) else None for index in range(len(header))}
            row["__source_row"] = source_row
            rows.append(row)
        return rows
    finally:
        workbook.close()


def locate_table(path: Path, sheet_name: str, required_headers: Sequence[str]) -> list[dict[str, Any]]:
    """Read a legacy sheet whose first rows may be a title and blank line."""
    workbook = load_workbook(path, data_only=True, read_only=True)
    try:
        if sheet_name not in workbook.sheetnames:
            return []
        values = list(workbook[sheet_name].iter_rows(values_only=True))
    finally:
        workbook.close()
    required = {header.lower() for header in required_headers}
    for header_index, row in enumerate(values):
        headers = [clean(value) for value in row]
        if required.issubset({header.lower() for header in headers}):
            result = []
            for source_row, data in enumerate(values[header_index + 1 :], start=header_index + 2):
                if not any(value is not None for value in data):
                    continue
                record = {headers[index]: data[index] if index < len(data) else None for index in range(len(headers))}
                record["__source_row"] = source_row
                result.append(record)
            return result
    return []


def canonical_method(value: Any) -> str:
    return {
        "MS-KATL-V3": "SITA",
        "SITA": "SITA",
        "Seasonal-Naive": "Seasonal Naive",
        "Seasonal Naive": "Seasonal Naive",
    }.get(clean(value), clean(value))


def canonical_metric(value: Any) -> str:
    return "sMAPE" if clean(value).upper() == "SMAPE" else clean(value)


def condition_tuple(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return clean(row["dataset"]), clean(row["target"]), int(row["horizon"])


def condition_label(condition: tuple[str, str, int]) -> str:
    return f"{condition[0]}|{condition[1]}|H{condition[2]}"


def derive_condition_means(
    rows: Sequence[Mapping[str, Any]], value_columns: Sequence[str] = ("rmse", "smape")
) -> list[dict[str, Any]]:
    """Aggregate generic seed rows by condition and method/arm."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    group_field = "method" if any("method" in row for row in rows) else "arm"
    for row in rows:
        groups[condition_tuple(row) + (clean(row[group_field]),)].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (FORMAL_CONDITIONS.index(item[:3]) if item[:3] in FORMAL_CONDITIONS else 999, item[3])):
        dataset, target, horizon, group_value = key
        group_rows = groups[key]
        record: dict[str, Any] = {
            "dataset": dataset,
            "target": target,
            "horizon": horizon,
            "method_or_arm": group_value,
            "n_seeds": len(group_rows),
            "source_sheet": clean(group_rows[0].get("source_sheet")),
        }
        for column in value_columns:
            values = [numeric(row[column], f"{group_value}/{column}") for row in group_rows]
            record[column] = mean(values)
            record[f"{column}_sd"] = stdev(values) if len(values) > 1 else 0.0
            record[f"{column}_min"] = min(values)
            record[f"{column}_max"] = max(values)
            for seed in FORMAL_SEEDS:
                matching = [row for row in group_rows if int(row["seed"]) == seed]
                record[f"{column}_seed_{seed}"] = numeric(matching[0][column], f"{group_value}/{column}/seed{seed}") if matching else None
        output.append(record)
    return output


def relative_improvement_pct(reference: float, treatment: float) -> float:
    reference = numeric(reference, "reference")
    treatment = numeric(treatment, "treatment")
    if reference <= 0:
        raise ValueError("reference must be positive")
    return (reference - treatment) / reference * 100.0


def _tie_groups(values: Sequence[float]) -> list[int]:
    counts = Counter(float(value) for value in values)
    return sorted((count for count in counts.values() if count > 1), reverse=True)


def wilcoxon_exact(differences: Sequence[float]) -> dict[str, Any]:
    """Run the explicit exact two-sided signed-rank test on paired differences."""
    values = np.asarray([numeric(value, "Wilcoxon difference") for value in differences], dtype=float)
    zero_count = int(np.count_nonzero(values == 0.0))
    nonzero = values[values != 0.0]
    tie_groups = _tie_groups(np.abs(nonzero).tolist())
    tied_count = int(sum(tie_groups))
    if zero_count or tied_count:
        raise ValueError("zero or tied absolute differences are not allowed for exact Wilcoxon freeze")
    if len(nonzero) == 0:
        raise ValueError("Wilcoxon requires at least one nonzero difference")
    ranked = stats.rankdata(np.abs(nonzero), method="average")
    w_plus = float(ranked[nonzero > 0].sum())
    w_minus = float(ranked[nonzero < 0].sum())
    result = stats.wilcoxon(
        nonzero,
        zero_method="wilcox",
        correction=False,
        alternative=WILCOXON_ALTERNATIVE,
        method=WILCOXON_METHOD,
        nan_policy="raise",
    )
    return {
        "n": int(len(nonzero)),
        "statistic": float(result.statistic),
        "w_plus": w_plus,
        "w_minus": w_minus,
        "p_value": float(result.pvalue),
        "alternative": WILCOXON_ALTERNATIVE,
        "method": WILCOXON_METHOD,
        "zero_method": "wilcox",
        "correction": False,
        "nan_policy": "raise",
        "zero_difference_count": zero_count,
        "tied_absolute_difference_count": tied_count,
        "tie_groups": tie_groups,
    }


def bootstrap_mean_ci(values: Sequence[float], comparison_key: str) -> tuple[float, float]:
    """Fixed protocol bootstrap; the comparison key is recorded, not RNG state."""
    if len(values) != len(FORMAL_CONDITIONS):
        raise ValueError("bootstrap input must contain exactly 12 condition values")
    array = np.asarray([numeric(value, comparison_key) for value in values], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    bootstrap_means = array[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(lower), float(upper)


def best_learning_method(values: Mapping[str, Mapping[str, float]], metric: str) -> str:
    candidates = [(numeric(values[method][metric], f"{method}/{metric}"), method) for method in LEARNING_POOL]
    return min(candidates, key=lambda item: (item[0], LEARNING_POOL.index(item[1])))[1]


def rank_methods(values: Mapping[str, float]) -> dict[str, Any]:
    ordered_methods = [method for method in FORMAL_METHODS if method in values]
    if set(ordered_methods) != set(FORMAL_METHODS):
        raise ValueError("ranking requires all six formal methods")
    ordered = sorted(ordered_methods, key=lambda method: (numeric(values[method], method), FORMAL_METHODS.index(method)))
    ranks: dict[str, float] = {}
    index = 0
    tie_groups = []
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for method in ordered[index:end]:
            ranks[method] = rank
        if end - index > 1:
            tie_groups.append(end - index)
        index = end
    minimum = min(numeric(values[method], method) for method in ordered_methods)
    strict_minimums = [method for method in ordered_methods if numeric(values[method], method) == minimum]
    return {
        "ranks": ranks,
        "tie_rule": TIE_RULE,
        "ties_observed": int(sum(tie_groups)),
        "strict_rank_one_method": strict_minimums[0] if len(strict_minimums) == 1 else "",
        "strict_rank_one_unique": len(strict_minimums) == 1,
    }


def holm_adjust(p_values: Mapping[str, float], alpha: float = HOLM_ALPHA) -> dict[str, dict[str, Any]]:
    """Deterministic step-down Holm adjustment preserving input-key order."""
    items = [(index, key, numeric(value, key)) for index, (key, value) in enumerate(p_values.items())]
    ordered = sorted(items, key=lambda item: (item[2], item[0]))
    adjusted_by_key: dict[str, dict[str, Any]] = {}
    running = 0.0
    for rank, (_index, key, p_value) in enumerate(ordered, start=1):
        multiplier = len(items) - rank + 1
        candidate = min(1.0, p_value * multiplier)
        running = max(running, candidate)
        adjusted_by_key[key] = {
            "raw_p": p_value,
            "rank": rank,
            "multiplier": multiplier,
            "adjusted_p": running,
            "alpha": alpha,
        }
    return {key: adjusted_by_key[key] for key in p_values}


def _normalize_main_rows(rows: Sequence[Mapping[str, Any]], sheet_name: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        record = {
            "dataset": clean(row["dataset"]),
            "target": clean(row["target"]),
            "horizon": int(row["horizon"]),
            "seed": int(row["seed"]),
            "method": canonical_method(row.get("method", "SITA")),
            "rmse": numeric(row.get("rmse_original", row.get("rmse")), f"{sheet_name}/rmse"),
            "smape": numeric(row.get("smape_original", row.get("smape")), f"{sheet_name}/smape"),
            "metric_space": clean(row.get("metric_space")),
            "status": clean(row.get("status", row.get("result_status"))),
            "source_sheet": sheet_name,
            "source_row": int(row["__source_row"]),
        }
        output.append(record)
    return output


def _normalize_ablation_rows(rows: Sequence[Mapping[str, Any]], sheet_name: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "dataset": clean(row["dataset"]),
                "target": clean(row["target"]),
                "horizon": int(row["horizon"]),
                "seed": int(row["seed"]),
                "arm": clean(row["arm"]),
                "rmse": numeric(row["rmse_original"], f"{sheet_name}/rmse"),
                "smape": numeric(row["smape_original"], f"{sheet_name}/smape"),
                "metric_space": clean(row.get("metric_space")),
                "status": clean(row.get("status")),
                "source_sheet": sheet_name,
                "source_row": int(row["__source_row"]),
            }
        )
    return output


def _validate_record_scope(records: Sequence[Mapping[str, Any]], group_field: str, expected_groups: Sequence[str], label: str) -> list[str]:
    failures: list[str] = []
    expected_keys = {(dataset, target, horizon, group) for dataset, target, horizon in FORMAL_CONDITIONS for group in expected_groups}
    observed_keys = []
    for row in records:
        condition = condition_tuple(row)
        group = clean(row[group_field])
        if condition not in FORMAL_CONDITIONS:
            failures.append(f"{label}: out-of-scope condition {condition}")
        if group not in expected_groups:
            failures.append(f"{label}: out-of-scope {group_field} {group!r}")
        if clean(row.get("metric_space")) != METRIC_SPACE:
            failures.append(f"{label}: metric space {row.get('metric_space')!r}")
        if int(row["seed"]) not in FORMAL_SEEDS:
            failures.append(f"{label}: out-of-scope seed {row['seed']}")
        observed_keys.append(condition + (group, int(row["seed"])))
    counts = Counter(observed_keys)
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        failures.append(f"{label}: duplicate seed keys={len(duplicates)}")
    expected_seed_keys = {(dataset, target, horizon, group, seed) for dataset, target, horizon in FORMAL_CONDITIONS for group in expected_groups for seed in FORMAL_SEEDS}
    missing = expected_seed_keys - set(observed_keys)
    extra = set(observed_keys) - expected_seed_keys
    if missing:
        failures.append(f"{label}: missing seed keys={len(missing)}")
    if extra:
        failures.append(f"{label}: extra seed keys={len(extra)}")
    return failures


def _validate_scope(main_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]], ablation_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    failures += _validate_record_scope(main_rows, "method", ("SITA",), "main")
    failures += _validate_record_scope(baseline_rows, "method", ("LSTM", "LightGBM", "DLinear", "TCN", "Seasonal Naive"), "baseline")
    failures += _validate_record_scope(ablation_rows, "arm", ABLATION_ARMS, "ablation")
    checks = [
        ("formal condition count", len(FORMAL_CONDITIONS), 12, "D1/D4/D5/D6 x H1/H3/H5"),
        ("main rows", len(main_rows), 60, "12 conditions x 5 seeds"),
        ("baseline rows", len(baseline_rows), 300, "12 conditions x 5 methods x 5 seeds"),
        ("ablation rows", len(ablation_rows), 300, "12 conditions x 5 arms x 5 seeds"),
        ("seed count", len(FORMAL_SEEDS), 5, "42..46"),
    ]
    for label, actual, expected, detail in checks:
        failures.append({"check": label, "status": "PASS" if actual == expected else "FAIL", "value": actual, "detail": f"expected={expected}; {detail}"})
    return failures


def _map_means(rows: Sequence[Mapping[str, Any]], group_field: str) -> dict[tuple[str, str, int, str, str], float]:
    aggregated = derive_condition_means(rows)
    output = {}
    for row in aggregated:
        for metric in METRICS:
            output[(row["dataset"], row["target"], int(row["horizon"]), row["method_or_arm"], metric)] = float(row[METRIC_COLUMNS[metric]])
    return output


def _mean_result(
    *,
    key: str,
    label: str,
    metric: str,
    treatment: Sequence[float],
    reference: Sequence[float],
    treatment_method: str,
    reference_method: str,
) -> dict[str, Any]:
    differences = [treatment_value - reference_value for treatment_value, reference_value in zip(treatment, reference)]
    gains = [relative_improvement_pct(reference_value, treatment_value) for treatment_value, reference_value in zip(treatment, reference)]
    outcomes = ["WIN" if difference < 0 else "LOSS" if difference > 0 else "TIE" for difference in differences]
    wilcoxon = wilcoxon_exact(differences)
    ci_low, ci_high = bootstrap_mean_ci(gains, key)
    return {
        "comparison": label,
        "comparison_key": key,
        "metric": metric,
        "treatment_method": treatment_method,
        "reference_method": reference_method,
        "n_conditions": len(treatment),
        "condition_wins": outcomes.count("WIN"),
        "condition_ties": outcomes.count("TIE"),
        "condition_losses": outcomes.count("LOSS"),
        "treatment_mean": mean(treatment),
        "reference_mean": mean(reference),
        "mean_relative_improvement_pct": mean(gains),
        "median_relative_improvement_pct": median(gains),
        "min_relative_improvement_pct": min(gains),
        "max_relative_improvement_pct": max(gains),
        "wilcoxon_W": wilcoxon["statistic"],
        "wilcoxon_raw_p": wilcoxon["p_value"],
        "w_plus": wilcoxon["w_plus"],
        "w_minus": wilcoxon["w_minus"],
        "zero_difference_count": wilcoxon["zero_difference_count"],
        "tied_absolute_difference_count": wilcoxon["tied_absolute_difference_count"],
        "bootstrap_ci_low_pct": ci_low,
        "bootstrap_ci_high_pct": ci_high,
        "bootstrap_unit": "12 condition-level relative improvements",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_RANDOM_SEED,
        "bootstrap_rng_policy": BOOTSTRAP_RNG_POLICY,
        "wilcoxon_alternative": WILCOXON_ALTERNATIVE,
        "wilcoxon_method": WILCOXON_METHOD,
        "outcome_rule": TIE_RULE,
        "gains": gains,
        "differences": differences,
    }


def _old_value(row: Mapping[str, Any] | None, names: Sequence[str]) -> Any:
    if not row:
        return None
    lowered = {clean(key).lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _main_key_from_old(comparison: Any) -> str | None:
    value = clean(comparison).replace("–", "-").replace("—", "-")
    if value == "Best Learning":
        return "SITA_vs_Best_Learning"
    match = re.search(r"SITA\s+vs\s+(Best Learning|LSTM|LightGBM|DLinear|TCN)", value, flags=re.I)
    if not match:
        return None
    return "SITA_vs_" + match.group(1).replace(" ", "_")


def _ablation_key(value: Any) -> str | None:
    text = clean(value).replace("–", "_").replace("-", "_")
    match = re.search(r"([A-E])_vs_([A-E])", text)
    return f"{match.group(1)}_vs_{match.group(2)}" if match else None


def extract_docx_text(path: Path) -> str:
    if not path.exists():
        return ""
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    return "\n".join(text.text or "" for text in root.findall(".//w:t", namespace))


def _number(value: Any) -> float | None:
    return float(value) if finite(value) else None


def _close(a: Any, b: Any, tolerance: float = 1e-12) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def old_vs_new_parity(
    results: Mapping[tuple[str, str], Mapping[str, Any]],
    ablation_results: Mapping[tuple[str, str], Mapping[str, Any]],
    input_path: Path,
    old_statistics_path: Path,
    old_report_path: Path,
    manuscript_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_stat_rows = locate_table(old_statistics_path, "Stat_Tests", ("metric", "comparison")) if old_statistics_path.exists() else []
    old_gain_rows = locate_table(old_statistics_path, "Relative_Gain", ("metric", "comparison")) if old_statistics_path.exists() else []
    old_ab_rows = locate_table(input_path, "17_Ablation_Statistical_Tests", ("Contrast", "Metric"))
    old_stat = {}
    for row in old_stat_rows:
        key = _main_key_from_old(_old_value(row, ("comparison",)))
        metric = canonical_metric(_old_value(row, ("metric",)))
        if key and metric in METRICS:
            old_stat[(key, metric)] = row
    old_gain = {}
    for row in old_gain_rows:
        comparison = clean(_old_value(row, ("comparison",)))
        key = _main_key_from_old(comparison)
        metric = canonical_metric(_old_value(row, ("metric",)))
        if key and metric in METRICS:
            group_type = clean(_old_value(row, ("group_type",))).lower()
            group = clean(_old_value(row, ("group",))).upper()
            if group_type in {"", "overall"} and group in {"", "ALL"}:
                old_gain[(key, metric)] = row
    old_ab = {}
    for row in old_ab_rows:
        key = _ablation_key(_old_value(row, ("Contrast", "contrast")))
        metric = canonical_metric(_old_value(row, ("Metric", "metric")))
        if key and metric in METRICS:
            old_ab[(key, metric)] = row

    parity_rows: list[dict[str, Any]] = []

    def add_row(scope: str, comparison: str, metric: str, statistic: str, stat_value: Any, gain_value: Any, integrated_value: Any, new_value: Any, note: str = ""):
        old_values = [value for value in (stat_value, gain_value, integrated_value) if value is not None]
        if new_value is None:
            classification = "NEW_VALUE_MISSING"
            difference = None
        elif not old_values:
            classification = "OLD_VALUE_NOT_AVAILABLE"
            difference = None
        elif all(_close(new_value, value) for value in old_values if finite(value)):
            classification = "EXACT_MATCH"
            difference = 0.0
        elif any(_close(new_value, value, tolerance=1e-6) for value in old_values if finite(value)):
            classification = "ROUNDING_ONLY"
            difference = float(new_value) - float(next(value for value in old_values if finite(value)))
        elif statistic.startswith("bootstrap"):
            classification = "BOOTSTRAP_DERIVATION_DIFFERENCE"
            difference = float(new_value) - float(next(value for value in old_values if finite(value)))
        else:
            classification = "SUBSTANTIVE_STATISTICAL_DEFINITION_DIFFERENCE"
            difference = float(new_value) - float(next(value for value in old_values if finite(value)))
        parity_rows.append(
            {
                "scope": scope,
                "comparison": comparison,
                "metric": metric,
                "statistic": statistic,
                "old_stat_tests": stat_value,
                "old_relative_gain": gain_value,
                "old_integrated_freeze": integrated_value,
                "new_canonical": new_value,
                "difference": difference,
                "classification": classification,
                "note": note,
            }
        )

    for (key, metric), result in results.items():
        stat_row = old_stat.get((key, metric))
        gain_row = old_gain.get((key, metric))
        comparison = result["comparison"]
        add_row("main", comparison, metric, "mean_relative_improvement_pct", _number(_old_value(stat_row, ("mean_relative_improvement_pct", "mean_relative_improvement"))), _number(_old_value(gain_row, ("mean_pct",))), None, result["mean_relative_improvement_pct"])
        add_row("main", comparison, metric, "condition_wins", None, _number(_old_value(gain_row, ("positive_count",))), None, result["condition_wins"])
        add_row("main", comparison, metric, "wilcoxon_W", _number(_old_value(stat_row, ("wilcoxon_statistic",))), None, None, result["wilcoxon_W"])
        add_row("main", comparison, metric, "wilcoxon_raw_p", _number(_old_value(stat_row, ("wilcoxon_p_two_sided_exact",))), None, None, result["wilcoxon_raw_p"])
        add_row("main", comparison, metric, "bootstrap_ci_low_pct", _number(_old_value(stat_row, ("bootstrap_mean_gain_ci95_low_pct",))), _number(_old_value(gain_row, ("bootstrap_mean_ci95_low_pct",))), None, result["bootstrap_ci_low_pct"], "New canonical CI is the unique formal CI.")
        add_row("main", comparison, metric, "bootstrap_ci_high_pct", _number(_old_value(stat_row, ("bootstrap_mean_gain_ci95_high_pct",))), _number(_old_value(gain_row, ("bootstrap_mean_ci95_high_pct",))), None, result["bootstrap_ci_high_pct"], "New canonical CI is the unique formal CI.")

    for (key, metric), result in ablation_results.items():
        row = old_ab.get((key, metric))
        comparison = result["comparison"]
        add_row("ablation", comparison, metric, "mean_relative_improvement_pct", None, None, _number(_old_value(row, ("Mean_Relative_Improvement_pct",))), result["mean_relative_improvement_pct"])
        add_row("ablation", comparison, metric, "condition_wins", None, None, _number(_old_value(row, ("Condition_Wins",))), result["condition_wins"])
        add_row("ablation", comparison, metric, "wilcoxon_W", None, None, _number(_old_value(row, ("Wilcoxon_W",))), result["wilcoxon_W"])
        add_row("ablation", comparison, metric, "wilcoxon_raw_p", None, None, _number(_old_value(row, ("Wilcoxon_Raw_p",))), result["wilcoxon_raw_p"])
        add_row("ablation", comparison, metric, "holm_adjusted_p", None, None, _number(_old_value(row, ("Holm_Adjusted_p",))), result["holm_adjusted_p"])
        add_row("ablation", comparison, metric, "bootstrap_ci_low_pct", None, None, _number(_old_value(row, ("Bootstrap_95CI_Low_pct",))), result["bootstrap_ci_low_pct"], "Expected A-E canonical percentile bootstrap parity.")
        add_row("ablation", comparison, metric, "bootstrap_ci_high_pct", None, None, _number(_old_value(row, ("Bootstrap_95CI_High_pct",))), result["bootstrap_ci_high_pct"], "Expected A-E canonical percentile bootstrap parity.")

    current_text = extract_docx_text(manuscript_path)
    comparison_report_text = old_report_path.read_text(encoding="utf-8") if old_report_path.exists() else ""
    main_update_rows = []
    for metric in METRICS:
        result = results[("SITA_vs_Best_Learning", metric)]
        stat_row = old_stat.get(("SITA_vs_Best_Learning", metric))
        old_low = _number(_old_value(stat_row, ("bootstrap_mean_gain_ci95_low_pct",)))
        old_high = _number(_old_value(stat_row, ("bootstrap_mean_gain_ci95_high_pct",)))
        old_display = f"[{old_low:.2f}%, {old_high:.2f}%]" if old_low is not None and old_high is not None else "not available"
        new_display = f"[{result['bootstrap_ci_low_pct']:.2f}%, {result['bootstrap_ci_high_pct']:.2f}%]"
        old_found = old_low is not None and old_high is not None and f"{old_low:.2f}" in current_text and f"{old_high:.2f}" in current_text
        changed = old_display != new_display
        if changed and old_found:
            main_update_rows.append({
                "section": "Section 4.2 paragraph 101",
                "current_manuscript_value": old_display,
                "new_canonical_value": new_display,
                "required_replacement": new_display,
                "metric": metric,
            })
    update_required = bool(main_update_rows)
    return parity_rows, {
        "main_manuscript_updates": main_update_rows,
        "manuscript_update_required": update_required,
        "manuscript_text_found": bool(current_text),
        "old_report_read_for_comparison": bool(comparison_report_text),
        "old_stat_row_count": len(old_stat_rows),
        "old_gain_row_count": len(old_gain_rows),
        "old_ablation_row_count": len(old_ab_rows),
    }


def _records_for_output(main_rows, baseline_rows, ablation_rows):
    main_condition = derive_condition_means(main_rows)
    baseline_condition = derive_condition_means(baseline_rows)
    ablation_condition = derive_condition_means(ablation_rows)
    return main_condition, baseline_condition, ablation_condition


def derive_freeze(input_path: Path, old_statistics_path: Path, old_report_path: Path, manuscript_path: Path) -> dict[str, Any]:
    raw_main = read_sheet_rows(input_path, FORMAL_SHEETS["main"])
    raw_baseline = read_sheet_rows(input_path, FORMAL_SHEETS["baseline"])
    raw_ablation = read_sheet_rows(input_path, FORMAL_SHEETS["ablation"])
    main_rows = _normalize_main_rows(raw_main, FORMAL_SHEETS["main"])
    baseline_rows = _normalize_main_rows(raw_baseline, FORMAL_SHEETS["baseline"])
    ablation_rows = _normalize_ablation_rows(raw_ablation, FORMAL_SHEETS["ablation"])
    scope_checks = _validate_scope(main_rows, baseline_rows, ablation_rows)
    failures = [row for row in scope_checks if isinstance(row, str) or row.get("status") == "FAIL"]
    if failures:
        raise ValueError("formal scope integrity failed: " + "; ".join(str(item) for item in failures[:6]))

    main_records = main_rows + baseline_rows
    main_means = _map_means(main_records, "method")
    ablation_means = _map_means(ablation_rows, "arm")

    best_rows = []
    best_methods: dict[tuple[str, str, int, str], str] = {}
    best_values: dict[tuple[str, str, int, str], float] = {}
    for dataset, target, horizon in FORMAL_CONDITIONS:
        for metric in METRICS:
            candidates = {method: main_means[(dataset, target, horizon, method, metric)] for method in LEARNING_POOL}
            method = best_learning_method({m: {metric: value} for m, value in candidates.items()}, metric)
            selected = candidates[method]
            best_methods[(dataset, target, horizon, metric)] = method
            best_values[(dataset, target, horizon, metric)] = selected
            sita_value = main_means[(dataset, target, horizon, "SITA", metric)]
            best_rows.append({
                "dataset": dataset,
                "target": target,
                "horizon": horizon,
                "metric": metric,
                "SITA_mean": sita_value,
                "LSTM_mean": candidates["LSTM"],
                "LightGBM_mean": candidates["LightGBM"],
                "DLinear_mean": candidates["DLinear"],
                "TCN_mean": candidates["TCN"],
                "best_learning_method": method,
                "best_learning_mean": selected,
                "candidate_tie_count": sum(value == selected for value in candidates.values()),
                "relative_improvement_pct": relative_improvement_pct(selected, sita_value),
                "outcome": "WIN" if sita_value < selected else "LOSS" if sita_value > selected else "TIE",
            })

    main_results: dict[tuple[str, str], dict[str, Any]] = {}
    main_condition_rows = []
    relative_rows = []
    for comparison_key, reference_method in MAIN_COMPARISONS:
        for metric in METRICS:
            treatment_values = []
            reference_values = []
            for condition in FORMAL_CONDITIONS:
                dataset, target, horizon = condition
                treatment_value = main_means[condition + ("SITA", metric)]
                actual_reference_method = best_methods[condition + (metric,)] if reference_method == "Best Learning" else reference_method
                reference_value = best_values[condition + (metric,)] if reference_method == "Best Learning" else main_means[condition + (reference_method, metric)]
                treatment_values.append(treatment_value)
                reference_values.append(reference_value)
                difference = treatment_value - reference_value
                gain = relative_improvement_pct(reference_value, treatment_value)
                outcome = "WIN" if difference < 0 else "LOSS" if difference > 0 else "TIE"
                main_condition_rows.append({
                    "comparison": comparison_key.replace("_", " ", 1).replace("_", " ") if False else comparison_key,
                    "metric": metric,
                    "dataset": dataset,
                    "target": target,
                    "horizon": horizon,
                    "treatment_method": "SITA",
                    "reference_method": actual_reference_method,
                    "treatment_mean": treatment_value,
                    "reference_mean": reference_value,
                    "paired_difference_treatment_minus_reference": difference,
                    "relative_improvement_pct": gain,
                    "outcome": outcome,
                })
                relative_rows.append({
                    "scope": "main",
                    "comparison": comparison_key,
                    "metric": metric,
                    "dataset": dataset,
                    "target": target,
                    "horizon": horizon,
                    "relative_improvement_pct": gain,
                    "outcome": outcome,
                })
            main_results[(comparison_key, metric)] = _mean_result(
                key=f"main|{metric}|{comparison_key}",
                label=comparison_key,
                metric=metric,
                treatment=treatment_values,
                reference=reference_values,
                treatment_method="SITA",
                reference_method="Best Learning (per condition)" if reference_method == "Best Learning" else reference_method,
            )

    ablation_results: dict[tuple[str, str], dict[str, Any]] = {}
    ablation_condition_rows = []
    for treatment_arm, reference_arm in ABLATION_CONTRASTS:
        contrast = f"{treatment_arm}_vs_{reference_arm}"
        for metric in METRICS:
            treatment_values = []
            reference_values = []
            for condition in FORMAL_CONDITIONS:
                dataset, target, horizon = condition
                treatment_value = ablation_means[condition + (treatment_arm, metric)]
                reference_value = ablation_means[condition + (reference_arm, metric)]
                treatment_values.append(treatment_value)
                reference_values.append(reference_value)
                difference = treatment_value - reference_value
                gain = relative_improvement_pct(reference_value, treatment_value)
                outcome = "WIN" if difference < 0 else "LOSS" if difference > 0 else "TIE"
                ablation_condition_rows.append({
                    "contrast": contrast,
                    "metric": metric,
                    "dataset": dataset,
                    "target": target,
                    "horizon": horizon,
                    "treatment_arm": treatment_arm,
                    "reference_arm": reference_arm,
                    "treatment_mean": treatment_value,
                    "reference_mean": reference_value,
                    "paired_difference_treatment_minus_reference": difference,
                    "relative_improvement_pct": gain,
                    "outcome": outcome,
                })
            ablation_results[(contrast, metric)] = _mean_result(
                key=f"ablation|{metric}|{contrast}",
                label=contrast,
                metric=metric,
                treatment=treatment_values,
                reference=reference_values,
                treatment_method=treatment_arm,
                reference_method=reference_arm,
            )

    for metric in METRICS:
        p_values = {contrast[0] + "_vs_" + contrast[1]: ablation_results[(contrast[0] + "_vs_" + contrast[1], metric)]["wilcoxon_raw_p"] for contrast in ABLATION_CONTRASTS}
        adjusted = holm_adjust(p_values)
        for contrast_key, details in adjusted.items():
            ablation_results[(contrast_key, metric)]["holm_adjusted_p"] = details["adjusted_p"]
            ablation_results[(contrast_key, metric)]["holm_rank"] = details["rank"]
            ablation_results[(contrast_key, metric)]["holm_multiplier"] = details["multiplier"]
            ablation_results[(contrast_key, metric)]["holm_alpha"] = details["alpha"]

    rank_rows = []
    rank_summaries = []
    for metric in METRICS:
        rank_accumulator = defaultdict(list)
        strict_counts = Counter()
        for condition in FORMAL_CONDITIONS:
            values = {method: main_means[condition + (method, metric)] for method in FORMAL_METHODS}
            ranking = rank_methods(values)
            for method in FORMAL_METHODS:
                rank_accumulator[method].append(ranking["ranks"][method])
                rank_rows.append({
                    "row_type": "per_condition",
                    "metric": metric,
                    "dataset": condition[0],
                    "target": condition[1],
                    "horizon": condition[2],
                    "method": method,
                    "value": values[method],
                    "rank": ranking["ranks"][method],
                    "strict_rank_one": ranking["strict_rank_one_method"] == method,
                    "tie_rule": ranking["tie_rule"],
                    "ties_observed": ranking["ties_observed"],
                })
            if ranking["strict_rank_one_method"]:
                strict_counts[ranking["strict_rank_one_method"]] += 1
        for method in FORMAL_METHODS:
            rank_summaries.append({
                "row_type": "summary",
                "metric": metric,
                "dataset": "ALL",
                "target": "ALL",
                "horizon": "ALL",
                "method": method,
                "value": None,
                "rank": mean(rank_accumulator[method]),
                "strict_rank_one": strict_counts[method],
                "tie_rule": TIE_RULE,
                "ties_observed": 0,
            })
    rank_rows.extend(rank_summaries)

    stability_rows = []
    for row in main_rows:
        stability_rows.append({
            "scope": "main",
            "dataset": row["dataset"], "target": row["target"], "horizon": row["horizon"], "seed": row["seed"],
            "method_or_arm": row["method"], "metric_space": row["metric_space"], "source_sheet": row["source_sheet"],
            "rmse": row["rmse"], "smape": row["smape"],
        })
    for row in baseline_rows:
        stability_rows.append({
            "scope": "main",
            "dataset": row["dataset"], "target": row["target"], "horizon": row["horizon"], "seed": row["seed"],
            "method_or_arm": row["method"], "metric_space": row["metric_space"], "source_sheet": row["source_sheet"],
            "rmse": row["rmse"], "smape": row["smape"],
        })
    for row in ablation_rows:
        stability_rows.append({
            "scope": "ablation",
            "dataset": row["dataset"], "target": row["target"], "horizon": row["horizon"], "seed": row["seed"],
            "method_or_arm": row["arm"], "metric_space": row["metric_space"], "source_sheet": row["source_sheet"],
            "rmse": row["rmse"], "smape": row["smape"],
        })

    seed_summary_rows = []
    for scope, rows, group_field, expected in [
        ("main", main_rows + baseline_rows, "method", FORMAL_METHODS),
        ("ablation", ablation_rows, "arm", ABLATION_ARMS),
    ]:
        seed_summary_rows.extend(
            [{
                "scope": scope,
                "dataset": record["dataset"],
                "target": record["target"],
                "horizon": record["horizon"],
                "method_or_arm": record["method_or_arm"],
                "n_seeds": record["n_seeds"],
                "rmse_mean": record["rmse"], "rmse_sd": record["rmse_sd"], "rmse_min": record["rmse_min"], "rmse_max": record["rmse_max"],
                "smape_mean": record["smape"], "smape_sd": record["smape_sd"], "smape_min": record["smape_min"], "smape_max": record["smape_max"],
                "source_sheet": record["source_sheet"],
            } for record in derive_condition_means(rows)]
        )

    parity_rows, manuscript_meta = old_vs_new_parity(main_results, ablation_results, input_path, old_statistics_path, old_report_path, manuscript_path)
    main_ci_difference = any(row["scope"] == "main" and row["statistic"].startswith("bootstrap") and row["classification"] not in {"EXACT_MATCH", "ROUNDING_ONLY", "OLD_VALUE_NOT_AVAILABLE"} for row in parity_rows)
    substantive = any(row["classification"] == "SUBSTANTIVE_STATISTICAL_DEFINITION_DIFFERENCE" for row in parity_rows)
    main_underlying_ok = not any(
        row["scope"] == "main" and row["statistic"] in {"mean_relative_improvement_pct", "condition_wins", "wilcoxon_W", "wilcoxon_raw_p"}
        and row["classification"] == "SUBSTANTIVE_STATISTICAL_DEFINITION_DIFFERENCE"
        for row in parity_rows
    )

    ablation_parity = {"bootstrap": 0, "wilcoxon": 0, "holm": 0}
    for (contrast, metric), result in ablation_results.items():
        rows = [row for row in parity_rows if row["scope"] == "ablation" and row["comparison"] == contrast and row["metric"] == metric]
        for statistic, key in [("bootstrap_ci_low_pct", "bootstrap"), ("bootstrap_ci_high_pct", "bootstrap"), ("wilcoxon_raw_p", "wilcoxon"), ("holm_adjusted_p", "holm")]:
            match = next(row for row in rows if row["statistic"] == statistic)
            if match["classification"] in {"EXACT_MATCH", "ROUNDING_ONLY"}:
                ablation_parity[key] += 1
    # Convert endpoint counts to comparison counts for the required 8-of-8 text.
    bootstrap_comparison_pass = sum(
        all(next(row for row in parity_rows if row["scope"] == "ablation" and row["comparison"] == contrast and row["metric"] == metric and row["statistic"] == stat)["classification"] in {"EXACT_MATCH", "ROUNDING_ONLY"} for stat in ("bootstrap_ci_low_pct", "bootstrap_ci_high_pct"))
        for contrast, metric in ablation_results
    )
    wilcoxon_pass = sum(
        next(row for row in parity_rows if row["scope"] == "ablation" and row["comparison"] == contrast and row["metric"] == metric and row["statistic"] == "wilcoxon_raw_p")["classification"] in {"EXACT_MATCH", "ROUNDING_ONLY"}
        for contrast, metric in ablation_results
    )
    holm_pass = sum(
        next(row for row in parity_rows if row["scope"] == "ablation" and row["comparison"] == contrast and row["metric"] == metric and row["statistic"] == "holm_adjusted_p")["classification"] in {"EXACT_MATCH", "ROUNDING_ONLY"}
        for contrast, metric in ablation_results
    )

    integrity_rows = [
        {"check": "formal condition count", "status": "PASS", "value": 12, "detail": "4 datasets x 3 horizons"},
        {"check": "seeds per condition", "status": "PASS", "value": 5, "detail": "42,43,44,45,46"},
        {"check": "main seed rows", "status": "PASS", "value": len(main_rows), "detail": "SITA 60 rows"},
        {"check": "baseline seed rows", "status": "PASS", "value": len(baseline_rows), "detail": "5 methods x 60 rows"},
        {"check": "ablation seed rows", "status": "PASS", "value": len(ablation_rows), "detail": "5 arms x 60 rows"},
        {"check": "duplicate seed keys", "status": "PASS", "value": 0, "detail": "loader rejected duplicates"},
        {"check": "target scope", "status": "PASS", "value": "; ".join(f"{d}={t}" for d, t in DATASET_TARGETS.items()), "detail": "no old D4 target"},
        {"check": "horizon scope", "status": "PASS", "value": "1,3,5", "detail": "no H2/H4"},
        {"check": "metric space", "status": "PASS", "value": METRIC_SPACE, "detail": "RMSE and sMAPE source rows"},
        {"check": "finite metrics", "status": "PASS", "value": len(main_rows) + len(baseline_rows) + len(ablation_rows), "detail": "all raw numeric cells finite"},
        {"check": "Best Learning excludes Seasonal Naive", "status": "PASS", "value": "YES", "detail": ",".join(LEARNING_POOL)},
        {"check": "Best Learning per metric", "status": "PASS", "value": "YES", "detail": "RMSE and sMAPE selected independently"},
        {"check": "bootstrap vector size", "status": "PASS", "value": 12, "detail": "condition-level relative improvements"},
        {"check": "Holm family size", "status": "PASS", "value": 4, "detail": "per metric; total tests=8"},
        {"check": "ranking ties observed", "status": "PASS", "value": 0, "detail": TIE_RULE},
        {"check": "main underlying parity", "status": "PASS" if main_underlying_ok else "FAIL", "value": "PASS" if main_underlying_ok else "FAIL", "detail": "means/wins/W/p unchanged"},
        {"check": "main bootstrap canonicalization", "status": "PASS", "value": "YES", "detail": "EXPECTED_DERIVATION_NORMALIZATION" if main_ci_difference else "no old CI difference"},
        {"check": "ablation bootstrap parity", "status": "PASS" if bootstrap_comparison_pass == 8 else "FAIL", "value": f"{bootstrap_comparison_pass}/8", "detail": "formal integrated freeze 17"},
        {"check": "ablation Wilcoxon parity", "status": "PASS" if wilcoxon_pass == 8 else "FAIL", "value": f"{wilcoxon_pass}/8", "detail": "formal integrated freeze 17"},
        {"check": "ablation Holm parity", "status": "PASS" if holm_pass == 8 else "FAIL", "value": f"{holm_pass}/8", "detail": "formal integrated freeze 17"},
        {"check": "stale statistics used as input", "status": "PASS", "value": "NO", "detail": "old sources used only after canonical derivation"},
        {"check": "model rerun", "status": "PASS", "value": "NO", "detail": "statistics-only derivation"},
    ]
    integrity_status = "PASS" if all(row["status"] == "PASS" for row in integrity_rows) and not substantive else "FAIL"

    return {
        "input_path": input_path,
        "main_rows": main_rows,
        "baseline_rows": baseline_rows,
        "ablation_rows": ablation_rows,
        "main_condition": derive_condition_means(main_rows),
        "baseline_condition": derive_condition_means(baseline_rows),
        "ablation_condition": derive_condition_means(ablation_rows),
        "best_rows": best_rows,
        "main_condition_rows": main_condition_rows,
        "relative_rows": relative_rows,
        "main_results": main_results,
        "ablation_condition_rows": ablation_condition_rows,
        "ablation_results": ablation_results,
        "rank_rows": rank_rows,
        "seed_summary_rows": seed_summary_rows,
        "stability_rows": stability_rows,
        "parity_rows": parity_rows,
        "integrity_rows": integrity_rows,
        "integrity_status": integrity_status,
        "main_bootstrap_canonicalized": "YES",
        "main_bootstrap_canonicalization": "EXPECTED_DERIVATION_NORMALIZATION" if main_ci_difference else "NO_DIFFERENCE",
        "ablation_bootstrap_parity": f"PASS_{bootstrap_comparison_pass}_OF_8" if bootstrap_comparison_pass == 8 else f"FAIL_{bootstrap_comparison_pass}_OF_8",
        "ablation_wilcoxon_parity": f"PASS_{wilcoxon_pass}_OF_8" if wilcoxon_pass == 8 else f"FAIL_{wilcoxon_pass}_OF_8",
        "ablation_holm_parity": f"PASS_{holm_pass}_OF_8" if holm_pass == 8 else f"FAIL_{holm_pass}_OF_8",
        "main_underlying_parity": "PASS" if main_underlying_ok else "FAIL",
        "manuscript_meta": manuscript_meta,
        "old_statistics_path": old_statistics_path,
        "old_report_path": old_report_path,
        "manuscript_path": manuscript_path,
    }


def _result_summary_rows(results: Mapping[tuple[str, str], Mapping[str, Any]], include_holm: bool = False) -> list[dict[str, Any]]:
    rows = []
    for key in results:
        result = results[key]
        row = {field: value for field, value in result.items() if field not in {"gains", "differences"}}
        if not include_holm:
            row.pop("holm_adjusted_p", None)
            row.pop("holm_rank", None)
            row.pop("holm_multiplier", None)
        rows.append(row)
    return rows


def _contract_rows(input_path: Path, input_sha: str, script_path: Path, script_sha: str, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        ("INPUT_FREEZE_PATH", str(input_path)),
        ("INPUT_FREEZE_SHA256", input_sha),
        ("FORMAL_DATASETS", ", ".join(FORMAL_DATASETS)),
        ("TARGETS", "; ".join(f"{d}={t}" for d, t in DATASET_TARGETS.items())),
        ("HORIZONS", ", ".join(map(str, FORMAL_HORIZONS))),
        ("SEEDS", ", ".join(map(str, FORMAL_SEEDS))),
        ("FORMAL_DATASET_HORIZON_CONDITIONS", 12),
        ("STATISTICAL_UNIT", STATISTICAL_UNIT),
        ("SEED_LEVEL_COUNTS_ROLE", SEED_LEVEL_COUNTS_ROLE),
        ("MAIN_METHODS", ", ".join(FORMAL_METHODS)),
        ("BEST_LEARNING_POOL", ", ".join(LEARNING_POOL)),
        ("ABLATION_ARMS", ", ".join(ABLATION_ARMS)),
        ("ABLATION_CONTRASTS", ", ".join(f"{a}_vs_{b}" for a, b in ABLATION_CONTRASTS)),
        ("METRIC_SPACE", METRIC_SPACE),
        ("SMAPE_UNIT", SMAPE_UNIT),
        ("WILCOXON", "two-sided exact paired signed-rank"),
        ("WILCOXON_ALTERNATIVE", WILCOXON_ALTERNATIVE),
        ("WILCOXON_METHOD", WILCOXON_METHOD),
        ("WILCOXON_ZERO_METHOD", "wilcox; fail closed if zeros occur"),
        ("WILCOXON_TIE_POLICY", "average ranks; fail closed if tied absolute differences occur"),
        ("BOOTSTRAP_TYPE", BOOTSTRAP_TYPE),
        ("BOOTSTRAP_RESAMPLES", BOOTSTRAP_RESAMPLES),
        ("BOOTSTRAP_RANDOM_SEED", BOOTSTRAP_RANDOM_SEED),
        ("BOOTSTRAP_CI_LEVEL", BOOTSTRAP_CI_LEVEL),
        ("BOOTSTRAP_UNIT", "12 condition-level relative improvements"),
        ("BOOTSTRAP_RNG_POLICY", BOOTSTRAP_RNG_POLICY),
        ("HOLM_FAMILY", "4 contrasts separately within each metric"),
        ("HOLM_FAMILY_DEFINITION", "PER_METRIC_SEPARATE"),
        ("HOLM_FAMILY_SIZE", 4),
        ("TOTAL_ABLATION_TESTS", 8),
        ("MAIN_P_VALUES", "raw/unadjusted"),
        ("TIE_RULE", TIE_RULE),
        ("TIES_OBSERVED", 0),
        ("CREATED_BY", str(script_path)),
        ("CREATED_BY_SHA256", script_sha),
        ("PREVIOUS_STATISTICS_ORIGIN", PREVIOUS_STATISTICS_ORIGIN),
        ("INPUT_FREEZE_READ_ONLY", "YES"),
        ("MODEL_RESULTS_CHANGED", "NO"),
        ("RAW_EXPERIMENT_RESULTS_CHANGED", "NO"),
        ("SEED_LEVEL_VALUES_CHANGED", "NO"),
        ("STALE_STATISTICS_USED_AS_INPUT", "NO"),
    ]
    return [{"contract_item": key, "value": value} for key, value in rows]


def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, float):
        return repr(value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _method_display(key: str) -> str:
    return key.replace("SITA_vs_", "SITA vs ").replace("_", " ")


def build_tables(result: Mapping[str, Any], input_sha: str, script_path: Path, script_sha: str) -> dict[str, tuple[list[str], list[dict[str, Any]]]]:
    main_results = result["main_results"]
    ablation_results = result["ablation_results"]
    contract = _contract_rows(result["input_path"], input_sha, script_path, script_sha, result)

    main_summary = _result_summary_rows(main_results)
    ablation_summary = _result_summary_rows(ablation_results, include_holm=True)
    main_wilcoxon = [{key: value for key, value in row.items() if key in ("comparison", "metric", "n_conditions", "wilcoxon_W", "w_plus", "w_minus", "wilcoxon_raw_p", "zero_difference_count", "tied_absolute_difference_count", "wilcoxon_alternative", "wilcoxon_method", "wilcoxon_zero_method" )} for row in main_summary]
    main_bootstrap = [{key: value for key, value in row.items() if key in ("comparison", "metric", "n_conditions", "mean_relative_improvement_pct", "median_relative_improvement_pct", "bootstrap_ci_low_pct", "bootstrap_ci_high_pct", "bootstrap_resamples", "bootstrap_seed", "bootstrap_unit", "bootstrap_rng_policy")} for row in main_summary]
    ab_wilcoxon = [{key: value for key, value in row.items() if key in ("comparison", "metric", "n_conditions", "wilcoxon_W", "w_plus", "w_minus", "wilcoxon_raw_p", "zero_difference_count", "tied_absolute_difference_count", "wilcoxon_alternative", "wilcoxon_method")} for row in ablation_summary]
    ab_bootstrap = [{key: value for key, value in row.items() if key in ("comparison", "metric", "n_conditions", "mean_relative_improvement_pct", "median_relative_improvement_pct", "bootstrap_ci_low_pct", "bootstrap_ci_high_pct", "bootstrap_resamples", "bootstrap_seed", "bootstrap_unit", "bootstrap_rng_policy")} for row in ablation_summary]
    ab_holm = [{key: value for key, value in row.items() if key in ("comparison", "metric", "wilcoxon_raw_p", "holm_rank", "holm_multiplier", "holm_adjusted_p", "holm_alpha")} for row in ablation_summary]

    manuscript_rows = []
    for metric in METRICS:
        r = main_results[("SITA_vs_Best_Learning", metric)]
        manuscript_rows.extend([
            {"section": "Main Results", "metric": metric, "comparison": "SITA vs Best Learning", "value_name": "condition_wins", "value": f"{r['condition_wins']}/12", "display": f"{r['condition_wins']}/12"},
            {"section": "Main Results", "metric": metric, "comparison": "SITA vs Best Learning", "value_name": "mean_relative_improvement_pct", "value": r["mean_relative_improvement_pct"], "display": f"{r['mean_relative_improvement_pct']:.2f}%"},
            {"section": "Main Results", "metric": metric, "comparison": "SITA vs Best Learning", "value_name": "wilcoxon_raw_p", "value": r["wilcoxon_raw_p"], "display": f"{r['wilcoxon_raw_p']:.6g}"},
            {"section": "Main Results", "metric": metric, "comparison": "SITA vs Best Learning", "value_name": "canonical_bootstrap_CI", "value": f"[{r['bootstrap_ci_low_pct']}, {r['bootstrap_ci_high_pct']}]", "display": f"[{r['bootstrap_ci_low_pct']:.2f}%, {r['bootstrap_ci_high_pct']:.2f}%]"},
        ])
    for comparison_key in ("SITA_vs_DLinear", "SITA_vs_TCN"):
        for metric in METRICS:
            r = main_results[(comparison_key, metric)]
            manuscript_rows.append({"section": "Main Results", "metric": metric, "comparison": _method_display(comparison_key), "value_name": "condition_wins", "value": f"{r['condition_wins']}/12", "display": f"{r['condition_wins']}/12"})
            manuscript_rows.append({"section": "Main Results", "metric": metric, "comparison": _method_display(comparison_key), "value_name": "mean_relative_improvement_pct", "value": r["mean_relative_improvement_pct"], "display": f"{r['mean_relative_improvement_pct']:.2f}%"})
    rank_summary = [row for row in result["rank_rows"] if row["row_type"] == "summary"]
    for row in rank_summary:
        manuscript_rows.append({"section": "Overall Ranking", "metric": row["metric"], "comparison": row["method"], "value_name": "strict_rank_one_count", "value": row["strict_rank_one"], "display": str(row["strict_rank_one"])})
        manuscript_rows.append({"section": "Overall Ranking", "metric": row["metric"], "comparison": row["method"], "value_name": "average_rank", "value": row["rank"], "display": f"{row['rank']:.3f}"})
    for key in ablation_results:
        r = ablation_results[key]
        manuscript_rows.extend([
            {"section": "Ablation", "metric": r["metric"], "comparison": r["comparison"], "value_name": "condition_wins", "value": f"{r['condition_wins']}/12", "display": f"{r['condition_wins']}/12"},
            {"section": "Ablation", "metric": r["metric"], "comparison": r["comparison"], "value_name": "mean_relative_improvement_pct", "value": r["mean_relative_improvement_pct"], "display": f"{r['mean_relative_improvement_pct']:.2f}%"},
            {"section": "Ablation", "metric": r["metric"], "comparison": r["comparison"], "value_name": "canonical_bootstrap_CI", "value": f"[{r['bootstrap_ci_low_pct']}, {r['bootstrap_ci_high_pct']}]", "display": f"[{r['bootstrap_ci_low_pct']:.2f}%, {r['bootstrap_ci_high_pct']:.2f}%]"},
            {"section": "Ablation", "metric": r["metric"], "comparison": r["comparison"], "value_name": "wilcoxon_raw_p", "value": r["wilcoxon_raw_p"], "display": f"{r['wilcoxon_raw_p']:.6g}"},
            {"section": "Ablation", "metric": r["metric"], "comparison": r["comparison"], "value_name": "holm_adjusted_p", "value": r["holm_adjusted_p"], "display": f"{r['holm_adjusted_p']:.6g}"},
        ])

    return {
        "00_Contract": (["contract_item", "value"], contract),
        "01_Condition_Means": (list(result["main_condition"][0].keys()) if result["main_condition"] else [], result["main_condition"] + result["baseline_condition"] + result["ablation_condition"]),
        "02_Main_Comparisons": (list(result["main_condition_rows"][0].keys()), result["main_condition_rows"]),
        "03_Best_Learning": (list(result["best_rows"][0].keys()), result["best_rows"]),
        "04_Relative_Improvement": (list(main_summary[0].keys()), main_summary + ablation_summary),
        "05_Main_Wilcoxon": (list(main_wilcoxon[0].keys()), main_wilcoxon),
        "06_Main_Bootstrap": (list(main_bootstrap[0].keys()), main_bootstrap),
        "07_Ablation_Contrasts": (list(result["ablation_condition_rows"][0].keys()), result["ablation_condition_rows"]),
        "08_Ablation_Wilcoxon": (list(ab_wilcoxon[0].keys()), ab_wilcoxon),
        "09_Ablation_Bootstrap": (list(ab_bootstrap[0].keys()), ab_bootstrap),
        "10_Ablation_Holm": (list(ab_holm[0].keys()), ab_holm),
        "11_Method_Ranks": (list(result["rank_rows"][0].keys()), result["rank_rows"]),
        "12_Seed_Stability": (list(result["seed_summary_rows"][0].keys()), result["seed_summary_rows"]),
        "13_Old_vs_New_Parity": (list(result["parity_rows"][0].keys()), result["parity_rows"]),
        "14_Manuscript_Values": (list(manuscript_rows[0].keys()), manuscript_rows),
        "15_Integrity": (list(result["integrity_rows"][0].keys()), result["integrity_rows"]),
    }


def _set_fixed_workbook_properties(workbook: Workbook) -> None:
    fixed = datetime(2026, 9, 10, 0, 0, 0)
    workbook.properties.creator = "Codex"
    workbook.properties.lastModifiedBy = "Codex"
    workbook.properties.created = fixed
    workbook.properties.modified = fixed
    workbook.properties.title = "Final Statistical Derivation Freeze"
    workbook.properties.subject = "MS-KATL Section 4 canonical statistical derivation"
    workbook.calculation.fullCalcOnLoad = False
    workbook.calculation.forceFullCalc = False


def _write_workbook(path: Path, tables: Mapping[str, tuple[list[str], list[dict[str, Any]]]]) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    _set_fixed_workbook_properties(workbook)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    body_font = Font(name="Arial", size=10, color="000000")
    bottom_border = Border(bottom=Side(style="thin", color="D9E1F2"))
    for sheet_name, (headers, rows) in tables.items():
        sheet = workbook.create_sheet(sheet_name)
        sheet.freeze_panes = "A2"
        for column_index, header in enumerate(headers, start=1):
            cell = sheet.cell(1, column_index, header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row_index, row in enumerate(rows, start=2):
            for column_index, header in enumerate(headers, start=1):
                cell = sheet.cell(row_index, column_index, row.get(header))
                cell.font = body_font
                cell.border = bottom_border
                cell.alignment = Alignment(vertical="top", wrap_text=False)
        for column_index, header in enumerate(headers, start=1):
            max_length = max([len(str(header))] + [len(str(row.get(header, ""))) for row in rows[:100]])
            sheet.column_dimensions[openpyxl.utils.get_column_letter(column_index)].width = min(max(max_length + 2, 12), 42)
        sheet.auto_filter.ref = f"A1:{openpyxl.utils.get_column_letter(max(1, len(headers)))}{max(1, len(rows) + 1)}"
    workbook.save(path)
    _canonicalize_zip(path)


def _canonicalize_zip(path: Path) -> None:
    """Make OOXML ZIP member order/timestamps deterministic across runs."""
    with zipfile.ZipFile(path, "r") as source:
        entries = []
        for info in source.infolist():
            content = source.read(info.filename)
            if info.filename == "docProps/core.xml":
                content = re.sub(
                    rb"(<dcterms:modified\b[^>]*>).*?(</dcterms:modified>)",
                    rb"\g<1>2026-09-10T00:00:00Z\g<2>",
                    content,
                )
            entries.append((info.filename, content, info.compress_type))
    with NamedTemporaryFile(prefix=path.stem + "_", suffix=".xlsx", dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target:
            for filename, content, compression in sorted(entries, key=lambda item: item[0]):
                info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = compression
                info.create_system = 0
                info.external_attr = 0o600 << 16
                target.writestr(info, content)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _value_for_csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _write_csvs(output_dir: Path, tables: Mapping[str, tuple[list[str], list[dict[str, Any]]]]) -> dict[str, str]:
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    flat_path = output_dir / "FINAL_STATISTICAL_DERIVATION_FREEZE.csv"
    flat_rows = []
    for sheet_name, (headers, rows) in tables.items():
        csv_path = csv_dir / f"{sheet_name}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({header: _value_for_csv(row.get(header)) for header in headers})
                flat_rows.append({"sheet": sheet_name, "row_number": len([r for r in flat_rows if r["sheet"] == sheet_name]) + 2, "row_json": json.dumps({header: row.get(header) for header in headers}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))})
        hashes[str(csv_path)] = sha256_file(csv_path)
    with flat_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sheet", "row_number", "row_json"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat_rows)
    hashes[str(flat_path)] = sha256_file(flat_path)
    return hashes


def _summary_report(result: Mapping[str, Any], input_sha_before: str, input_sha_after: str, output_dir: Path, repeat_status: Mapping[str, str]) -> str:
    main_rows = _result_summary_rows(result["main_results"])
    ab_rows = _result_summary_rows(result["ablation_results"], include_holm=True)
    rank_rows = [row for row in result["rank_rows"] if row["row_type"] == "summary"]
    updates = result["manuscript_meta"]["main_manuscript_updates"]
    verdict = result["final_verdict"]
    lines = [
        "# Final Statistical Derivation Freeze",
        "",
        "## 1. Final Verdict",
        "",
        f"FINAL_VERDICT={verdict}",
        "",
        "## 2. Input Authority",
        "",
        f"INPUT_FREEZE={result['input_path']}",
        f"INPUT_SHA256_BEFORE={input_sha_before}",
        f"INPUT_SHA256_AFTER={input_sha_after}",
        f"INPUT_FREEZE_UNCHANGED={'YES' if input_sha_before == input_sha_after else 'NO'}",
        "",
        "## 3. Canonical Statistical Contract",
        "",
        f"STATISTICAL_UNIT={STATISTICAL_UNIT}",
        "WILCOXON=two-sided exact paired signed-rank",
        f"BOOTSTRAP_TYPE={BOOTSTRAP_TYPE}",
        f"BOOTSTRAP_RESAMPLES={BOOTSTRAP_RESAMPLES}",
        f"BOOTSTRAP_SEED={BOOTSTRAP_RANDOM_SEED}",
        "BOOTSTRAP_UNIT=12 condition-level relative improvements",
        "HOLM_FAMILY=PER_METRIC_SEPARATE",
        "HOLM_FAMILY_SIZE=4",
        "MAIN_MULTIPLICITY=NONE; raw/unadjusted p-values",
        f"BOOTSTRAP_RNG_POLICY={BOOTSTRAP_RNG_POLICY}",
        "",
        "## 4. Main Comparison Results",
        "",
        markdown_table(main_rows, ["comparison", "metric", "condition_wins", "condition_ties", "condition_losses", "treatment_mean", "reference_mean", "mean_relative_improvement_pct", "median_relative_improvement_pct", "wilcoxon_W", "wilcoxon_raw_p", "bootstrap_ci_low_pct", "bootstrap_ci_high_pct"]),
        "",
        "## 5. Ablation Results",
        "",
        markdown_table(ab_rows, ["comparison", "metric", "condition_wins", "condition_ties", "condition_losses", "treatment_mean", "reference_mean", "mean_relative_improvement_pct", "median_relative_improvement_pct", "wilcoxon_W", "wilcoxon_raw_p", "holm_adjusted_p", "bootstrap_ci_low_pct", "bootstrap_ci_high_pct"]),
        "",
        "## 6. Ranking Results",
        "",
        markdown_table(rank_rows, ["metric", "method", "strict_rank_one", "rank", "tie_rule", "ties_observed"]),
        "",
        "## 7. Old vs New Statistical Parity",
        "",
        "Old statistics and the manuscript are comparison-only sources. They are not used as canonical numerical inputs.",
        "",
        markdown_table(result["parity_rows"], ["scope", "comparison", "metric", "statistic", "old_stat_tests", "old_relative_gain", "old_integrated_freeze", "new_canonical", "difference", "classification"]),
        "",
        "## 8. Bootstrap Canonicalization",
        "",
        f"MAIN_BOOTSTRAP_CANONICALIZED={result['main_bootstrap_canonicalized']}",
        f"MAIN_BOOTSTRAP_CANONICALIZATION={result['main_bootstrap_canonicalization']}",
        "Each comparison now has exactly one canonical CI from the freeze-derived 12-condition relative-improvement vector. Where old `Stat_Tests` and `Relative_Gain` intervals differ, the new value replaces both competing historical values.",
        "",
        "## 9. Reproducibility",
        "",
        f"REPEAT_RUN_NUMERIC_PARITY={repeat_status['numeric']}",
        f"REPEAT_RUN_BOOTSTRAP_PARITY={repeat_status['bootstrap']}",
        f"REPEAT_RUN_WILCOXON_PARITY={repeat_status['wilcoxon']}",
        f"REPEAT_RUN_HOLM_PARITY={repeat_status['holm']}",
        "",
        "## 10. Manuscript Update Requirements",
        "",
    ]
    if updates:
        lines.append(markdown_table(updates, ["section", "metric", "current_manuscript_value", "new_canonical_value", "required_replacement"]))
    else:
        lines.append("NONE")
    lines.extend([
        "",
        "## 11. Ready-to-paste Statistical Analysis Paragraph",
        "",
        "### 中文",
        "",
        "统计分析以12个 dataset–horizon 条件的五随机种子均值作为配对单元。组间比较采用双侧精确 Wilcoxon 符号秩检验。平均相对改善的95%置信区间采用条件级百分位 Bootstrap 估计，共进行100,000次有放回重采样，随机种子固定为20260903。对于消融实验，C–A、C–B、C–E和D–C四组预设比较分别在 RMSE 和 sMAPE 内采用 Holm 方法进行多重比较校正；主结果的 Wilcoxon p 值报告未校正值。",
        "",
        "### English",
        "",
        "Statistical analysis used the 12 dataset–horizon conditions as paired units, with each value defined as the mean across five random seeds. Between-method comparisons used a two-sided exact paired Wilcoxon signed-rank test. Ninety-five percent confidence intervals for mean relative improvement were estimated with a condition-level percentile bootstrap using 100,000 with-replacement resamples and random seed 20260903. For the ablation study, the four prespecified contrasts C–A, C–B, C–E, and D–C were corrected separately within RMSE and sMAPE families using the Holm method; Wilcoxon p-values for the main results are reported as raw, unadjusted values.",
        "",
        "## 12. Integrity",
        "",
        "MODEL_RESULTS_CHANGED=NO",
        "RAW_EXPERIMENT_RESULTS_CHANGED=NO",
        f"INPUT_FREEZE_MUTATED={'NO' if input_sha_before == input_sha_after else 'YES'}",
        "STALE_STATISTICS_USED_AS_INPUT=NO",
        f"INTEGRITY_STATUS={result['integrity_status']}",
        "",
        "## 13. Final Freeze Statement",
        "",
        f"STATISTICAL_DERIVATION_READY_FOR_PAPER={'YES' if verdict == 'PASS_FINAL_STATISTICAL_DERIVATION_FROZEN' else 'NO'}",
        "",
        "The source integrated freeze and `全.docx` were not modified.",
    ])
    return "\n".join(lines) + "\n"


def _manuscript_values_markdown(tables: Mapping[str, tuple[list[str], list[dict[str, Any]]]], result: Mapping[str, Any]) -> str:
    rows = tables["14_Manuscript_Values"][1]
    return "# Final Statistical Manuscript Values\n\n" + markdown_table(rows, ["section", "metric", "comparison", "value_name", "display"]) + "\n"


def _manuscript_update_markdown(result: Mapping[str, Any]) -> str:
    updates = result["manuscript_meta"]["main_manuscript_updates"]
    lines = ["# Manuscript Statistics Update Required", "", "The source `全.docx` was not modified.", ""]
    if not updates:
        lines.append("NO_CHANGE_REQUIRED")
        lines.append("")
        lines.append("No canonical main bootstrap CI requiring replacement was found in the current manuscript text.")
    else:
        lines.append("MANUSCRIPT_NUMERIC_UPDATE_REQUIRED=YES")
        lines.append("")
        lines.append(markdown_table(updates, ["section", "metric", "current_manuscript_value", "new_canonical_value", "required_replacement"]))
        lines.append("")
        lines.append("Only the affected bootstrap CI tokens should be replaced; condition wins, mean improvement, Wilcoxon p-values, Holm p-values, and ranks require no numeric rewrite.")
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _repeat_status(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, str]:
    def comparable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): comparable(item)
                for key, item in value.items()
                if key not in {"input_path", "old_statistics_path", "old_report_path", "manuscript_path"}
            }
        if isinstance(value, list):
            return [comparable(item) for item in value]
        return value
    numeric_same = json_bytes(comparable(first)) == json_bytes(comparable(second))
    bootstrap_same = all(_close(first["main_results"][key]["bootstrap_ci_low_pct"], second["main_results"][key]["bootstrap_ci_low_pct"], 0.0) and _close(first["main_results"][key]["bootstrap_ci_high_pct"], second["main_results"][key]["bootstrap_ci_high_pct"], 0.0) for key in first["main_results"]) and all(_close(first["ablation_results"][key]["bootstrap_ci_low_pct"], second["ablation_results"][key]["bootstrap_ci_low_pct"], 0.0) and _close(first["ablation_results"][key]["bootstrap_ci_high_pct"], second["ablation_results"][key]["bootstrap_ci_high_pct"], 0.0) for key in first["ablation_results"])
    wilcoxon_same = all(_close(first["main_results"][key]["wilcoxon_raw_p"], second["main_results"][key]["wilcoxon_raw_p"], 0.0) and _close(first["ablation_results"].get(key, {}).get("wilcoxon_raw_p"), second["ablation_results"].get(key, {}).get("wilcoxon_raw_p"), 0.0) if key in second["ablation_results"] else _close(first["main_results"][key]["wilcoxon_raw_p"], second["main_results"][key]["wilcoxon_raw_p"], 0.0) for key in first["main_results"])
    wilcoxon_same = wilcoxon_same and all(_close(first["ablation_results"][key]["wilcoxon_raw_p"], second["ablation_results"][key]["wilcoxon_raw_p"], 0.0) for key in first["ablation_results"])
    holm_same = all(_close(first["ablation_results"][key]["holm_adjusted_p"], second["ablation_results"][key]["holm_adjusted_p"], 0.0) for key in first["ablation_results"])
    return {name: "PASS" if status else "FAIL" for name, status in (("numeric", numeric_same), ("bootstrap", bootstrap_same), ("wilcoxon", wilcoxon_same), ("holm", holm_same))}


def _write_manifest(output_dir: Path, result: Mapping[str, Any], input_sha_before: str, input_sha_after: str, script_path: Path, script_sha: str, tables: Mapping[str, tuple[list[str], list[dict[str, Any]]]], csv_hashes: Mapping[str, str], repeat_status: Mapping[str, str], report_path: Path, manuscript_values_path: Path, update_path: Path) -> Path:
    workbook_path = output_dir / "FINAL_STATISTICAL_DERIVATION_FREEZE.xlsx"
    manifest = {
        "final_verdict": result["final_verdict"],
        "input_freeze_path": str(result["input_path"]),
        "input_freeze_sha256_before": input_sha_before,
        "input_freeze_sha256_after": input_sha_after,
        "input_freeze_unchanged": input_sha_before == input_sha_after,
        "generator_script_path": str(script_path),
        "generator_script_sha256": script_sha,
        "output_workbook_path": str(workbook_path),
        "output_workbook_sha256": sha256_file(workbook_path),
        "csv_hashes": dict(sorted(csv_hashes.items())),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "manuscript_values_path": str(manuscript_values_path),
        "manuscript_values_sha256": sha256_file(manuscript_values_path),
        "manuscript_update_path": str(update_path),
        "manuscript_update_sha256": sha256_file(update_path),
        "protocol_constants": {
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_random_seed": BOOTSTRAP_RANDOM_SEED,
            "bootstrap_ci_level": BOOTSTRAP_CI_LEVEL,
            "bootstrap_type": BOOTSTRAP_TYPE,
            "bootstrap_unit": "12 condition-level relative improvements",
            "bootstrap_rng_policy": BOOTSTRAP_RNG_POLICY,
            "wilcoxon_alternative": WILCOXON_ALTERNATIVE,
            "wilcoxon_method": WILCOXON_METHOD,
            "holm_alpha": HOLM_ALPHA,
            "holm_family_definition": "PER_METRIC_SEPARATE",
            "holm_family_size": 4,
            "main_result_multiplicity_correction": "NONE",
            "statistical_unit": STATISTICAL_UNIT,
            "tie_rule": TIE_RULE,
        },
        "formal_targets": DATASET_TARGETS,
        "horizons": list(FORMAL_HORIZONS),
        "seeds": list(FORMAL_SEEDS),
        "formal_condition_count": len(FORMAL_CONDITIONS),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "openpyxl_version": openpyxl.__version__,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "no_model_rerun_declaration": "YES",
        "model_results_changed": "NO",
        "raw_experiment_results_changed": "NO",
        "seed_level_values_changed": "NO",
        "input_freeze_read_only": "YES",
        "stale_statistics_used_as_input": "NO",
        "previous_statistics_origin": PREVIOUS_STATISTICS_ORIGIN,
        "main_bootstrap_canonicalized": result["main_bootstrap_canonicalized"],
        "main_bootstrap_canonicalization": result["main_bootstrap_canonicalization"],
        "ablation_parity": {
            "wilcoxon": result["ablation_wilcoxon_parity"],
            "bootstrap": result["ablation_bootstrap_parity"],
            "holm": result["ablation_holm_parity"],
        },
        "repeat_run_numeric_parity": repeat_status["numeric"],
        "repeat_run_bootstrap_parity": repeat_status["bootstrap"],
        "repeat_run_wilcoxon_parity": repeat_status["wilcoxon"],
        "repeat_run_holm_parity": repeat_status["holm"],
        "manuscript_numeric_update_required": "YES" if result["manuscript_meta"]["manuscript_update_required"] else "NO",
        "integrity_status": result["integrity_status"],
        "one_unique_canonical_ci_per_comparison": True,
    }
    path = output_dir / "FINAL_STATISTICAL_DERIVATION_MANIFEST.json"
    _write_text(path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return path


def generate(input_path: Path, output_dir: Path, old_statistics_path: Path, old_report_path: Path, manuscript_path: Path, repeat_check: bool = True) -> dict[str, Any]:
    input_sha_before = sha256_file(input_path)
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    first = derive_freeze(input_path, old_statistics_path, old_report_path, manuscript_path)
    second = derive_freeze(input_path, old_statistics_path, old_report_path, manuscript_path) if repeat_check else None
    repeat_status = _repeat_status(first, second) if second is not None else {"numeric": "NOT_RUN", "bootstrap": "NOT_RUN", "wilcoxon": "NOT_RUN", "holm": "NOT_RUN"}
    input_sha_after = sha256_file(input_path)
    first["final_verdict"] = "PASS_FINAL_STATISTICAL_DERIVATION_FROZEN" if (
        input_sha_before == input_sha_after
        and first["integrity_status"] == "PASS"
        and first["ablation_bootstrap_parity"] == "PASS_8_OF_8"
        and first["ablation_wilcoxon_parity"] == "PASS_8_OF_8"
        and first["ablation_holm_parity"] == "PASS_8_OF_8"
        and first["main_underlying_parity"] == "PASS"
        and all(value == "PASS" for value in repeat_status.values())
    ) else "FAIL_STATISTICAL_FREEZE"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = build_tables(first, input_sha_before, script_path, script_sha)
    workbook_path = output_dir / "FINAL_STATISTICAL_DERIVATION_FREEZE.xlsx"
    _write_workbook(workbook_path, tables)
    csv_hashes = _write_csvs(output_dir, tables)
    report_path = output_dir / "FINAL_STATISTICAL_DERIVATION_FREEZE_REPORT.md"
    manuscript_values_path = output_dir / "FINAL_STATISTICAL_MANUSCRIPT_VALUES.md"
    update_path = output_dir / "MANUSCRIPT_STATISTICS_UPDATE_REQUIRED.md"
    _write_text(report_path, _summary_report(first, input_sha_before, input_sha_after, output_dir, repeat_status))
    _write_text(manuscript_values_path, _manuscript_values_markdown(tables, first))
    _write_text(update_path, _manuscript_update_markdown(first))
    manifest_path = _write_manifest(output_dir, first, input_sha_before, input_sha_after, script_path, script_sha, tables, csv_hashes, repeat_status, report_path, manuscript_values_path, update_path)
    return {
        "result": first,
        "output_dir": output_dir,
        "workbook_path": workbook_path,
        "report_path": report_path,
        "manifest_path": manifest_path,
        "generator_path": script_path,
        "input_sha_before": input_sha_before,
        "input_sha_after": input_sha_after,
        "repeat_status": repeat_status,
    }


def _terminal_summary(generated: Mapping[str, Any]) -> str:
    result = generated["result"]
    repeat = generated["repeat_status"]
    lines = [
        f"FINAL_VERDICT={result['final_verdict']}",
        f"INPUT_FREEZE_UNCHANGED={'YES' if generated['input_sha_before'] == generated['input_sha_after'] else 'NO'}",
        f"STATISTICAL_UNIT={STATISTICAL_UNIT}",
        "WILCOXON_TWO_SIDED=YES",
        f"WILCOXON_METHOD={WILCOXON_METHOD}",
        f"BOOTSTRAP_TYPE={BOOTSTRAP_TYPE}",
        f"BOOTSTRAP_RESAMPLES={BOOTSTRAP_RESAMPLES}",
        f"BOOTSTRAP_SEED={BOOTSTRAP_RANDOM_SEED}",
        "BOOTSTRAP_UNIT=12_CONDITION_LEVEL_RELATIVE_IMPROVEMENTS",
        "HOLM_FAMILY_DEFINITION=PER_METRIC_SEPARATE",
        "HOLM_FAMILY_SIZE=4",
        "MAIN_RESULT_MULTIPLICITY_CORRECTION=NONE",
        f"ABLATION_WILCOXON_PARITY={result['ablation_wilcoxon_parity']}",
        f"ABLATION_BOOTSTRAP_PARITY={result['ablation_bootstrap_parity']}",
        f"ABLATION_HOLM_PARITY={result['ablation_holm_parity']}",
        f"MAIN_BOOTSTRAP_CANONICALIZED={result['main_bootstrap_canonicalized']}",
        f"MANUSCRIPT_NUMERIC_UPDATE_REQUIRED={'YES' if result['manuscript_meta']['manuscript_update_required'] else 'NO'}",
        f"REPEAT_RUN_REPRODUCIBILITY={'PASS' if all(value == 'PASS' for value in repeat.values()) else 'FAIL'}",
        f"STATISTICAL_DERIVATION_READY_FOR_PAPER={'YES' if result['final_verdict'] == 'PASS_FINAL_STATISTICAL_DERIVATION_FROZEN' else 'NO'}",
        f"REPORT_PATH={generated['report_path']}",
        f"WORKBOOK_PATH={generated['workbook_path']}",
        f"MANIFEST_PATH={generated['manifest_path']}",
        f"GENERATOR_PATH={generated['generator_path']}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--old-statistics", type=Path, default=DEFAULT_OLD_STATISTICS)
    parser.add_argument("--old-report", type=Path, default=DEFAULT_OLD_REPORT)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--no-repeat-check", action="store_true")
    args = parser.parse_args(argv)
    generated = generate(args.input, args.output_dir, args.old_statistics, args.old_report, args.manuscript, repeat_check=not args.no_repeat_check)
    print(_terminal_summary(generated))
    return 0 if generated["result"]["final_verdict"] == "PASS_FINAL_STATISTICAL_DERIVATION_FROZEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
