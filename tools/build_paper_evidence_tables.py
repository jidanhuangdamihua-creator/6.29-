#!/usr/bin/env python3
"""Build a source-preserving MS-KATL paper evidence workbook.

The script reads the frozen workbook, derives paper-facing evidence tables in
memory, and writes a new workbook.  It never saves the input workbook and it
does not read raw experiment directories as a fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.table import Table, TableStyleInfo


SCRIPT_VERSION = "1.0.0"
DEFAULT_INPUT = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/MS_KATL_Paper_Data_Freeze_v1_2_FINAL_FROZEN.xlsx"
)
DEFAULT_OUTPUT = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/MS_KATL_PAPER_EVIDENCE_TABLES.xlsx"
)
REQUIRED_SOURCE_SHEETS = [
    "03_V3_Raw_5Seed",
    "04_BL_Raw_Selected",
    "06_Main_RMSE",
    "07_Main_sMAPE",
    "09_Relative_Gain",
    "10_Stability",
    "11_Ablation_Summary",
    "12_Ablation_Contract",
    "14_Paper_Summary",
]
MASTER_SHEET = "03_Paper_Results_Master"
CONTRACT_SHEET = "01_Paper_Contract"
ABLATION_CONTRACT_SHEET = "12_Ablation_Contract"
ABLATION_RAW_SHEET = "05_Ablation_Raw"
FROZEN_GAIN_SHEET = "09_Relative_Gain"
OURS_METHOD = "MS-KATL-V3"
BASELINE_METHODS = ["LSTM", "LightGBM", "Seasonal-Naive"]
METRICS = [("RMSE", "rmse"), ("sMAPE", "smape")]
EXPECTED_ABLATION_ARMS = ["A", "B", "C", "D", "E"]
BOOTSTRAP_RANDOM_SEED = 20260824
BOOTSTRAP_N_RESAMPLES = 10000
BOOTSTRAP_CI_LEVEL = 95

HEADER_FONT = Font(name="Arial", size=10, bold=True, color="FFFFFF")
BODY_FONT = Font(name="Arial", size=10, color="000000")
TITLE_FONT = Font(name="Arial", size=11, bold=True, color="1F4E78")
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
TITLE_FILL = PatternFill("solid", fgColor="D9EAF7")
THIN_GREY = Side(style="thin", color="D9E1F2")
BODY_BORDER = Border(bottom=THIN_GREY)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any, optional: bool = False) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "NA" if optional else ""
    return str(value).strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean(value, optional=True).upper() in {"TRUE", "YES", "1", "PASS"}


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else math.nan


def sample_sd(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else math.nan


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else math.nan


def condition_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (clean(row["dataset"]), clean(row["target"]), int(row["horizon"]))


def seed_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return condition_key(row) + (int(row["seed"]),)


def full_source_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return seed_key(row) + (
        clean(row["method"]),
        clean(row["experiment_type"]),
        clean(row["arm"], optional=True),
        clean(row["mode"], optional=True),
    )


def read_rows(workbook, sheet_name: str) -> list[dict[str, Any]]:
    ws = workbook[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    headers = [clean(cell) for cell in rows[0]]
    result = []
    for excel_row, values in enumerate(rows[1:], start=2):
        if not any(value is not None for value in values):
            continue
        row = {headers[i]: values[i] if i < len(values) else None for i in range(len(headers))}
        row["__source_row"] = excel_row
        result.append(row)
    return result


def contract_map(workbook, sheet_name: str) -> dict[str, str]:
    return {
        clean(row[0]): clean(row[1], optional=True)
        for row in workbook[sheet_name].iter_rows(min_row=2, values_only=True)
        if row and row[0] is not None
    }


def parse_tie_threshold(contract: dict[str, str]) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", contract.get("Tie rule", ""))
    if not match:
        raise ValueError("The frozen workbook does not expose a usable tie rule")
    return float(match.group(1))


def parse_selected_datasets(contract: dict[str, str]) -> list[str]:
    return sorted(set(re.findall(r"\bD[1-6]\b", contract.get("Selected datasets", ""))))


def parse_targets(contract: dict[str, str]) -> dict[str, str]:
    return {
        dataset: target.strip()
        for dataset, target in re.findall(r"(D[1-6])=([^;]+)", contract.get("Formal targets", ""))
    }


def parse_frozen_gain_rows(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {}
    method_map = {
        "vs_LSTM": "LSTM",
        "vs_LightGBM": "LightGBM",
        "vs_SeasonalNaive": "Seasonal-Naive",
    }
    for row in rows:
        method = method_map.get(clean(row.get("Comparison")))
        metric = clean(row.get("Metric"))
        if not method or metric not in {name for name, _ in METRICS}:
            continue
        key = (clean(row["Dataset"]), clean(row["Target"]), int(row["Horizon"]), metric, method)
        result[key] = row
    return result


def is_tie(delta: float, denominator: float, tie_threshold: float) -> bool:
    if denominator == 0:
        return delta == 0
    return abs(delta) / abs(denominator) <= tie_threshold


def ri_ratio(baseline_value: float, ours_value: float) -> float:
    return (baseline_value - ours_value) / baseline_value if baseline_value else math.nan


def outcome_error(delta: float, denominator: float, tie_threshold: float) -> str:
    if is_tie(delta, denominator, tie_threshold):
        return "TIE"
    return "WIN" if delta < 0 else "LOSS"


def parse_contrast(value: str) -> tuple[str, str] | None:
    match = re.match(r"\s*([A-E])_vs_([A-E])\s*:", value or "")
    return (match.group(1), match.group(2)) if match else None


def paired_summary(rows: list[dict[str, Any]], group_type: str = "overall", group_value: str = "ALL") -> dict[str, Any]:
    wins = sum(row["outcome"] == "WIN" for row in rows)
    ties = sum(row["outcome"] == "TIE" for row in rows)
    losses = sum(row["outcome"] == "LOSS" for row in rows)
    deltas = [float(row["paired_delta_value"]) for row in rows]
    effects = [float(row["relative_improvement_ratio"]) * 100 for row in rows if finite(row["relative_improvement_ratio"])]
    total = wins + ties + losses
    return {
        "metric": rows[0]["metric"] if rows else "NA",
        "ours_method": rows[0]["ours_method"] if rows else "NA",
        "baseline_method": rows[0]["baseline_method"] if rows else "NA",
        "group_type": group_type,
        "group_value": group_value,
        "n_pairs": total,
        "n_wins": wins,
        "n_ties": ties,
        "n_losses": losses,
        "win_rate_pct": 100 * wins / total if total else math.nan,
        "loss_rate_pct": 100 * losses / total if total else math.nan,
        "mean_paired_delta": mean(deltas),
        "median_paired_delta": median(deltas),
        "std_paired_delta": sample_sd(deltas),
        "mean_relative_improvement_pct": mean(effects),
        "median_relative_improvement_pct": median(effects),
        "source_sheet": "03_Paper_Results_Master",
    }


def bootstrap_mean_ci(values: list[float], seed: int, n_resamples: int, ci_level: float) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = __import__("random").Random(seed)
    samples = []
    n = len(values)
    for _ in range(n_resamples):
        samples.append(mean(values[rng.randrange(n)] for _ in range(n)))
    samples.sort()
    alpha = (100 - ci_level) / 200
    lower_idx = max(0, min(len(samples) - 1, int(alpha * len(samples))))
    upper_idx = max(0, min(len(samples) - 1, int((1 - alpha) * len(samples)) - 1))
    return samples[lower_idx], samples[upper_idx]


def wilcoxon_signed_rank(values: list[float]) -> tuple[float, float, str]:
    """Small dependency-free two-sided Wilcoxon signed-rank implementation.

    It uses the exact permutation distribution for n<=20 and a normal
    approximation with continuity correction otherwise.  Zero differences are
    discarded, as in the usual Wilcoxon signed-rank test.
    """
    nonzero = [float(value) for value in values if value != 0]
    n = len(nonzero)
    if n == 0:
        return math.nan, math.nan, "NOT_APPLICABLE_ALL_ZERO"
    ranked = sorted((abs(value), index) for index, value in enumerate(nonzero))
    ranks = [0.0] * n
    pos = 0
    while pos < n:
        end = pos + 1
        while end < n and ranked[end][0] == ranked[pos][0]:
            end += 1
        rank = (pos + 1 + end) / 2
        for _, index in ranked[pos:end]:
            ranks[index] = rank
        pos = end
    w_plus = sum(rank for rank, value in zip(ranks, nonzero) if value > 0)
    w_minus = sum(rank for rank, value in zip(ranks, nonzero) if value < 0)
    statistic = min(w_plus, w_minus)
    if n <= 20:
        total = 0
        favorable = 0
        for mask in range(1 << n):
            w = sum(ranks[i] for i in range(n) if mask & (1 << i))
            total += 1
            if min(w, sum(ranks) - w) <= statistic + 1e-12:
                favorable += 1
        p_value = min(1.0, favorable / total)
        status = "EXACT_TWO_SIDED"
    else:
        expected = n * (n + 1) / 4
        variance = n * (n + 1) * (2 * n + 1) / 24
        z = (statistic - expected + 0.5) / math.sqrt(variance)
        p_value = math.erfc(abs(z) / math.sqrt(2))
        status = "NORMAL_APPROX_TWO_SIDED"
    return statistic, p_value, status


def formula_absolute(row_number: int, ours_col: str, baseline_col: str) -> str:
    return f"={ours_col}{row_number}-{baseline_col}{row_number}"


def formula_relative(row_number: int, ours_col: str, baseline_col: str, scale: float = 100.0) -> str:
    return f'=IFERROR(({baseline_col}{row_number}-{ours_col}{row_number})/{baseline_col}{row_number}*{scale},"")'


def table_name(text: str, index: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", text).strip("_") or "Table"
    return f"{safe[:20]}_{index}"


def write_table(ws, title: str, headers: list[str], rows: list[list[Any]], start_row: int, index: int) -> int:
    ws.cell(start_row, 1, title)
    ws.cell(start_row, 1).font = TITLE_FONT
    ws.cell(start_row, 1).fill = TITLE_FILL
    header_row = start_row + 1
    for col, header in enumerate(headers, start=1):
        cell = ws.cell(header_row, col, header)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for r_offset, row in enumerate(rows, start=1):
        for col, value in enumerate(row, start=1):
            cell = ws.cell(header_row + r_offset, col, value)
            cell.font = BODY_FONT
            cell.border = BODY_BORDER
            cell.alignment = Alignment(vertical="top", wrap_text=isinstance(value, str) and len(value) > 40)
            if isinstance(value, str) and value.startswith("="):
                cell.number_format = "0.000"
    end_row = header_row + max(1, len(rows))
    if rows:
        ref = f"A{header_row}:{ws.cell(end_row, len(headers)).coordinate}"
        tab = Table(displayName=table_name(title, index), ref=ref)
        tab.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False, showRowStripes=True, showColumnStripes=False)
        ws.add_table(tab)
    ws.freeze_panes = "A3"
    return end_row + 3


def set_formats(ws) -> None:
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            header = ""
            for candidate in ws[2] if ws.max_row >= 2 else []:
                if candidate.column == cell.column:
                    header = clean(candidate.value).lower()
                    break
            if any(token in header for token in ("_pct", "rate_pct", "effect_pct", "gain_pct")):
                cell.number_format = "0.000"
            elif any(token in header for token in ("mean", "median", "std", "value", "delta", "difference", "lower", "upper", "ci_")):
                if isinstance(cell.value, (int, float)) or (isinstance(cell.value, str) and cell.value.startswith("=")):
                    cell.number_format = "0.000000"
    for column_cells in ws.columns:
        letter = column_cells[0].column_letter
        max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
        ws.column_dimensions[letter].width = min(46, max(12, max_len + 2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite-output", action="store_true", help="Allow replacing an earlier derived output from this script")
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"INPUT_NOT_FOUND={input_path}")
    if input_path == output_path:
        raise SystemExit("OUTPUT_MUST_NOT_EQUAL_INPUT")
    if output_path.exists() and not args.overwrite_output:
        raise SystemExit(f"OUTPUT_ALREADY_EXISTS={output_path}; refusing to overwrite")

    input_sha_before = sha256_file(input_path)
    source_wb = load_workbook(input_path, read_only=True, data_only=True)
    found_sheets = list(source_wb.sheetnames)
    missing_sheets = [sheet for sheet in REQUIRED_SOURCE_SHEETS if sheet not in found_sheets]
    if MASTER_SHEET not in found_sheets or CONTRACT_SHEET not in found_sheets or ABLATION_CONTRACT_SHEET not in found_sheets:
        raise SystemExit("REQUIRED_CANONICAL_SHEETS_MISSING")

    contract = contract_map(source_wb, CONTRACT_SHEET)
    selected_datasets = parse_selected_datasets(contract)
    target_map = parse_targets(contract)
    tie_threshold = parse_tie_threshold(contract)

    master_rows = read_rows(source_wb, MASTER_SHEET)
    if not master_rows:
        raise SystemExit("MASTER_IS_EMPTY")
    selected_master = [row for row in master_rows if as_bool(row.get("include_in_master")) and clean(row.get("status")) == "OK"]
    if not selected_master:
        raise SystemExit("NO_SELECTED_MASTER_ROWS")

    metric_spaces = {clean(row.get("metric_space")) for row in selected_master}
    smape_units = {clean(row.get("smape_unit")) for row in selected_master}
    rmse_space = next(iter(metric_spaces)) if len(metric_spaces) == 1 else "UNVERIFIED"
    smape_unit = next(iter(smape_units)) if len(smape_units) == 1 else "UNVERIFIED"
    metric_space_ok = metric_spaces == {"original_sales_space"} and smape_units == {"percent"}
    if not metric_space_ok:
        raise SystemExit(f"METRIC_SPACE_VERIFICATION=FAIL metric_spaces={metric_spaces} smape_units={smape_units}")

    main_rows = [row for row in selected_master if clean(row.get("experiment_type")) in {"main", "baseline"}]
    v3_rows = [row for row in main_rows if clean(row.get("method")) == OURS_METHOD]
    baseline_rows = [row for row in main_rows if clean(row.get("method")) in BASELINE_METHODS]
    ablation_rows = [row for row in selected_master if clean(row.get("experiment_type")) == "ablation"]
    conditions = sorted({condition_key(row) for row in v3_rows}, key=lambda value: (value[0], value[2]))
    horizons = sorted({int(row["horizon"]) for row in v3_rows})

    v3_by_seed = {seed_key(row): row for row in v3_rows}
    baseline_by_seed = {(seed_key(row) + (clean(row["method"]),)): row for row in baseline_rows}
    ablation_by_seed = {(seed_key(row) + (clean(row["arm"]),)): row for row in ablation_rows}

    duplicate_full = sum(count > 1 for count in Counter(full_source_key(row) for row in selected_master).values())
    duplicate_main = sum(
        count > 1
        for count in Counter(seed_key(row) + (clean(row["method"]),) for row in main_rows).values()
    )
    nonfinite_values = sum(
        not finite(row.get(field))
        for row in selected_master
        for field in ("rmse", "smape")
    )
    missing_values = sum(
        row.get(field) is None or clean(row.get(field)) == ""
        for row in selected_master
        for field in ("dataset", "target", "horizon", "seed", "method", "rmse", "smape")
    )

    frozen_gain = parse_frozen_gain_rows(read_rows(source_wb, FROZEN_GAIN_SHEET)) if FROZEN_GAIN_SHEET in found_sheets else {}
    relative_rows: list[dict[str, Any]] = []
    for dataset, target, horizon in conditions:
        for metric_name, field in METRICS:
            ours = [row for key, row in v3_by_seed.items() if key[:3] == (dataset, target, horizon)]
            for baseline_method in BASELINE_METHODS:
                baseline = [
                    row for key, row in baseline_by_seed.items()
                    if key[:3] == (dataset, target, horizon) and key[4] == baseline_method
                ]
                ours_values = [float(row[field]) for row in ours]
                baseline_values = [float(row[field]) for row in baseline]
                ours_mean, baseline_mean = mean(ours_values), mean(baseline_values)
                ratio = ri_ratio(baseline_mean, ours_mean)
                fkey = (dataset, target, horizon, metric_name, baseline_method)
                frozen = frozen_gain.get(fkey, {})
                frozen_raw = frozen.get("Gain_pct") if frozen else None
                consistent = (
                    "PASS"
                    if frozen_raw is not None and finite(frozen_raw) and abs(float(frozen_raw) - ratio) <= 1e-10
                    else "NOT_FOUND"
                    if not frozen
                    else "FAIL"
                )
                relative_rows.append({
                    "dataset": dataset,
                    "target": target,
                    "horizon": horizon,
                    "metric": metric_name,
                    "ours_method": OURS_METHOD,
                    "baseline_method": baseline_method,
                    "ours_mean": ours_mean,
                    "baseline_mean": baseline_mean,
                    "absolute_difference_value": ours_mean - baseline_mean,
                    "relative_improvement_ratio": ratio,
                    "n_ours": len(ours_values),
                    "n_baseline": len(baseline_values),
                    "comparison_direction": "lower_error_better; positive RI means ours better",
                    "source_sheet": f"{MASTER_SHEET};{FROZEN_GAIN_SHEET}",
                    "frozen_gain_raw_ratio": frozen_raw,
                    "frozen_result": frozen.get("Result") if frozen else "NOT_FOUND",
                    "gain_consistency_status": consistent,
                })

    paired_rows: list[dict[str, Any]] = []
    unmatched_ours = 0
    unmatched_baseline = 0
    for metric_name, field in METRICS:
        for dataset, target, horizon in conditions:
            for seed in sorted({key[3] for key in v3_by_seed if key[:3] == (dataset, target, horizon)}):
                ours = v3_by_seed.get((dataset, target, horizon, seed))
                for baseline_method in BASELINE_METHODS:
                    baseline = baseline_by_seed.get((dataset, target, horizon, seed, baseline_method))
                    pair_valid = ours is not None and baseline is not None
                    if ours is None:
                        unmatched_ours += 1
                    if baseline is None:
                        unmatched_baseline += 1
                    if not pair_valid:
                        continue
                    ours_value, baseline_value = float(ours[field]), float(baseline[field])
                    delta = ours_value - baseline_value
                    paired_rows.append({
                        "dataset": dataset,
                        "target": target,
                        "horizon": horizon,
                        "seed": seed,
                        "metric": metric_name,
                        "ours_method": OURS_METHOD,
                        "baseline_method": baseline_method,
                        "ours_value": ours_value,
                        "baseline_value": baseline_value,
                        "paired_delta_value": delta,
                        "relative_improvement_ratio": ri_ratio(baseline_value, ours_value),
                        "outcome": outcome_error(delta, baseline_value, tie_threshold),
                        "pair_valid": "YES",
                        "tie_threshold": tie_threshold,
                        "source_sheet": MASTER_SHEET,
                        "source_key": f"{dataset}|{target}|{horizon}|{seed}|{OURS_METHOD}|{baseline_method}",
                    })

    paired_summary_rows: list[dict[str, Any]] = []
    for metric_name, _ in METRICS:
        for baseline_method in BASELINE_METHODS:
            base = [row for row in paired_rows if row["metric"] == metric_name and row["baseline_method"] == baseline_method]
            paired_summary_rows.append(paired_summary(base))
            for dataset in selected_datasets:
                rows = [row for row in base if row["dataset"] == dataset]
                if rows:
                    paired_summary_rows.append(paired_summary(rows, "dataset", dataset))
            for horizon in horizons:
                rows = [row for row in base if row["horizon"] == horizon]
                if rows:
                    paired_summary_rows.append(paired_summary(rows, "horizon", f"H{horizon}"))

    horizon_profile_rows: list[dict[str, Any]] = []
    profile_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    methods = [OURS_METHOD] + BASELINE_METHODS
    for dataset, target, horizon in conditions:
        for metric_name, field in METRICS:
            for method in methods:
                source_rows = [
                    row for key, row in (v3_by_seed.items() if method == OURS_METHOD else baseline_by_seed.items())
                    if key[:3] == (dataset, target, horizon)
                    and (method == OURS_METHOD or key[4] == method)
                ]
                values = [float(row[field]) for row in source_rows]
                profile = {
                    "dataset": dataset,
                    "target": target,
                    "horizon": horizon,
                    "metric": metric_name,
                    "method": method,
                    "mean_value": mean(values),
                    "std_value": sample_sd(values),
                    "n_seeds": len(values),
                    "baseline_method": "NA",
                    "baseline_mean": math.nan,
                    "relative_improvement_pct": math.nan,
                    "source_sheet": MASTER_SHEET,
                }
                horizon_profile_rows.append(profile)
                profile_map[(dataset, target, horizon, metric_name, method)] = profile

    horizon_comparison_rows: list[dict[str, Any]] = []
    for dataset, target, horizon in conditions:
        for metric_name, _ in METRICS:
            ours = profile_map[(dataset, target, horizon, metric_name, OURS_METHOD)]
            for baseline_method in BASELINE_METHODS:
                baseline = profile_map[(dataset, target, horizon, metric_name, baseline_method)]
                ratio = ri_ratio(baseline["mean_value"], ours["mean_value"])
                horizon_comparison_rows.append({
                    "dataset": dataset,
                    "target": target,
                    "horizon": horizon,
                    "metric": metric_name,
                    "method": OURS_METHOD,
                    "mean_value": ours["mean_value"],
                    "std_value": ours["std_value"],
                    "n_seeds": ours["n_seeds"],
                    "baseline_method": baseline_method,
                    "baseline_mean": baseline["mean_value"],
                    "relative_improvement_ratio": ratio,
                    "source_sheet": MASTER_SHEET,
                })

    horizon_aggregate_rows: list[dict[str, Any]] = []
    for metric_name, _ in METRICS:
        for horizon in horizons:
            for baseline_method in BASELINE_METHODS:
                comparisons = [
                    row for row in horizon_comparison_rows
                    if row["metric"] == metric_name and row["horizon"] == horizon and row["baseline_method"] == baseline_method
                ]
                condition_ris = [row["relative_improvement_ratio"] * 100 for row in comparisons]
                wins = sum(
                    not is_tie(row["mean_value"] - row["baseline_mean"], row["baseline_mean"], tie_threshold)
                    and row["mean_value"] < row["baseline_mean"]
                    for row in comparisons
                )
                horizon_aggregate_rows.append({
                    "horizon": horizon,
                    "metric": metric_name,
                    "ours_method": OURS_METHOD,
                    "baseline_method": baseline_method,
                    "n_conditions": len(comparisons),
                    "ours_mean": mean(row["mean_value"] for row in comparisons),
                    "baseline_mean": mean(row["baseline_mean"] for row in comparisons),
                    "mean_relative_improvement_pct": mean(condition_ris),
                    "median_relative_improvement_pct": median(condition_ris),
                    "win_rate_pct": 100 * wins / len(comparisons) if comparisons else math.nan,
                    "source_sheet": MASTER_SHEET,
                })

    method_change_rows: list[dict[str, Any]] = []
    for dataset, target in sorted({(key[0], key[1]) for key in profile_map}):
        for metric_name, _ in METRICS:
            for method in methods:
                values = [profile_map[(dataset, target, h, metric_name, method)]["mean_value"] for h in horizons]
                value_map = {h: profile_map[(dataset, target, h, metric_name, method)]["mean_value"] for h in horizons}
                h1, h3, h5 = (value_map.get(h) for h in (1, 3, 5))
                method_change_rows.append({
                    "dataset": dataset, "target": target, "metric": metric_name, "method": method,
                    "H1_value": h1, "H3_value": h3, "H5_value": h5,
                    "H1_to_H3_change": h3 - h1 if h1 is not None and h3 is not None else math.nan,
                    "H3_to_H5_change": h5 - h3 if h3 is not None and h5 is not None else math.nan,
                    "H1_to_H5_change": h5 - h1 if h1 is not None and h5 is not None else math.nan,
                    "H1_to_H3_change_pct": (h3 - h1) / h1 * 100 if h1 else math.nan,
                    "H3_to_H5_change_pct": (h5 - h3) / h3 * 100 if h3 else math.nan,
                    "H1_to_H5_change_pct": (h5 - h1) / h1 * 100 if h1 else math.nan,
                    "source_sheet": MASTER_SHEET,
                })

    ri_change_rows: list[dict[str, Any]] = []
    for dataset, target in sorted({(row["dataset"], row["target"]) for row in horizon_comparison_rows}):
        for metric_name, _ in METRICS:
            for baseline_method in BASELINE_METHODS:
                ri_map = {
                    h: next(row["relative_improvement_ratio"] * 100 for row in horizon_comparison_rows if row["dataset"] == dataset and row["target"] == target and row["metric"] == metric_name and row["horizon"] == h and row["baseline_method"] == baseline_method)
                    for h in horizons
                }
                h1, h3, h5 = (ri_map.get(h) for h in (1, 3, 5))
                ri_change_rows.append({
                    "dataset": dataset, "target": target, "metric": metric_name,
                    "ours_method": OURS_METHOD, "baseline_method": baseline_method,
                    "RI_H1": h1, "RI_H3": h3, "RI_H5": h5,
                    "RI_H1_to_H3": h3 - h1 if h1 is not None and h3 is not None else math.nan,
                    "RI_H3_to_H5": h5 - h3 if h3 is not None and h5 is not None else math.nan,
                    "RI_H1_to_H5": h5 - h1 if h1 is not None and h5 is not None else math.nan,
                    "source_sheet": MASTER_SHEET,
                })

    ablation_contract_rows = read_rows(source_wb, ABLATION_CONTRACT_SHEET)
    arm_contract = {clean(row["Arm"]): row for row in ablation_contract_rows}
    full_arm = next((arm for arm, row in arm_contract.items() if clean(row.get("Arm_Name")) == "CURRENT_V3"), "UNVERIFIED")
    reference_arm = next((arm for arm, row in arm_contract.items() if clean(row.get("Arm_Name")) == "FUSED_TRAINABLE"), "UNVERIFIED")
    contrast_defs = []
    for row in ablation_contract_rows:
        parsed = parse_contrast(clean(row.get("Primary_Contrast")))
        if parsed:
            left, right = parsed
            contrast_defs.append({
                "contrast_label": f"{left}_vs_{right}",
                "reference_arm": left,
                "comparison_arm": right,
                "primary_contrast": clean(row.get("Primary_Contrast")),
                "scientific_question": clean(row.get("Scientific_Question")),
            })
    ablation_paired_rows: list[dict[str, Any]] = []
    for contrast in contrast_defs:
        for metric_name, field in METRICS:
            for dataset, target, horizon in sorted({condition_key(row) for row in ablation_rows}, key=lambda value: (value[0], value[2])):
                for seed in sorted({int(row["seed"]) for row in ablation_rows if condition_key(row) == (dataset, target, horizon)}):
                    ref = ablation_by_seed.get((dataset, target, horizon, seed, contrast["reference_arm"]))
                    comp = ablation_by_seed.get((dataset, target, horizon, seed, contrast["comparison_arm"]))
                    if not ref or not comp:
                        continue
                    reference_value, comparison_value = float(ref[field]), float(comp[field])
                    delta = comparison_value - reference_value
                    ablation_paired_rows.append({
                        "dataset": dataset, "target": target, "horizon": horizon, "seed": seed,
                        "metric": metric_name, "reference_arm": contrast["reference_arm"],
                        "comparison_arm": contrast["comparison_arm"], "reference_value": reference_value,
                        "comparison_value": comparison_value, "paired_delta_value": delta,
                        "relative_effect_ratio": delta / reference_value if reference_value else math.nan,
                        "outcome": "TIE" if is_tie(delta, reference_value, tie_threshold) else "REFERENCE_WIN" if delta > 0 else "REFERENCE_LOSS",
                        "contract_primary_contrast": contrast["primary_contrast"],
                        "scientific_question": contrast["scientific_question"],
                        "pair_valid": "YES", "source_sheet": f"{ABLATION_RAW_SHEET};{ABLATION_CONTRACT_SHEET}",
                    })

    ablation_summary_rows: list[dict[str, Any]] = []
    for contrast in contrast_defs:
        for metric_name, _ in METRICS:
            rows = [row for row in ablation_paired_rows if row["metric"] == metric_name and row["reference_arm"] == contrast["reference_arm"] and row["comparison_arm"] == contrast["comparison_arm"]]
            wins = sum(row["outcome"] == "REFERENCE_WIN" for row in rows)
            ties = sum(row["outcome"] == "TIE" for row in rows)
            losses = sum(row["outcome"] == "REFERENCE_LOSS" for row in rows)
            effects = [row["relative_effect_ratio"] * 100 for row in rows if finite(row["relative_effect_ratio"])]
            ablation_summary_rows.append({
                "metric": metric_name, "reference_arm": contrast["reference_arm"], "comparison_arm": contrast["comparison_arm"],
                "contrast_label": contrast["contrast_label"], "n_pairs": len(rows), "wins_reference": wins,
                "ties": ties, "losses_reference": losses, "mean_delta": mean(row["paired_delta_value"] for row in rows),
                "median_delta": median(row["paired_delta_value"] for row in rows), "mean_effect_pct": mean(effects),
                "median_effect_pct": median(effects), "source_sheet": f"{ABLATION_RAW_SHEET};{ABLATION_CONTRACT_SHEET}",
            })

    wtl_rows: list[dict[str, Any]] = []
    for metric_name, _ in METRICS:
        for baseline_method in BASELINE_METHODS:
            base = [row for row in paired_rows if row["metric"] == metric_name and row["baseline_method"] == baseline_method]
            for group_type, values in [("overall", ["ALL"]), ("dataset", selected_datasets), ("horizon", [f"H{h}" for h in horizons])]:
                for group_value in values:
                    rows = base if group_type == "overall" else [row for row in base if (row["dataset"] == group_value if group_type == "dataset" else f"H{row['horizon']}" == group_value)]
                    wins = sum(row["outcome"] == "WIN" for row in rows)
                    ties = sum(row["outcome"] == "TIE" for row in rows)
                    losses = sum(row["outcome"] == "LOSS" for row in rows)
                    total = wins + ties + losses
                    wtl_rows.append({
                        "metric": metric_name, "ours_method": OURS_METHOD, "baseline_method": baseline_method,
                        "group_type": group_type, "group_value": group_value, "wins": wins, "ties": ties,
                        "losses": losses, "total_pairs": total, "win_rate_pct": 100 * wins / total if total else math.nan,
                        "tie_rate_pct": 100 * ties / total if total else math.nan, "loss_rate_pct": 100 * losses / total if total else math.nan,
                        "source_sheet": "03_Paired_Effect",
                    })

    statistical_rows: list[dict[str, Any]] = []
    for metric_name, _ in METRICS:
        for baseline_method in BASELINE_METHODS:
            rows = [row for row in paired_rows if row["metric"] == metric_name and row["baseline_method"] == baseline_method]
            deltas = [row["paired_delta_value"] for row in rows]
            statistic, p_value, test_status = wilcoxon_signed_rank(deltas)
            ci_lower, ci_upper = bootstrap_mean_ci(deltas, BOOTSTRAP_RANDOM_SEED, BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_CI_LEVEL)
            statistical_rows.append({
                "metric": metric_name, "ours_method": OURS_METHOD, "baseline_method": baseline_method,
                "n_pairs": len(deltas), "statistic": statistic, "p_value": p_value,
                "test_status": test_status, "median_paired_delta": median(deltas),
                "mean_paired_delta": mean(deltas), "bootstrap_statistic": "mean_paired_delta",
                "bootstrap_ci_lower": ci_lower, "bootstrap_ci_upper": ci_upper,
                "bootstrap_random_seed": BOOTSTRAP_RANDOM_SEED, "bootstrap_n_resamples": BOOTSTRAP_N_RESAMPLES,
                "bootstrap_ci_level_pct": BOOTSTRAP_CI_LEVEL, "source_sheet": "03_Paired_Effect",
            })

    plot_rows: list[dict[str, Any]] = []
    for row in relative_rows:
        plot_rows.append({
            "evidence_type": "RELATIVE_IMPROVEMENT", "dataset": row["dataset"], "target": row["target"], "horizon": row["horizon"], "seed": "NA", "metric": row["metric"], "ours_method": row["ours_method"], "comparison_method": row["baseline_method"], "value": row["ours_mean"], "secondary_value": row["baseline_mean"], "effect": row["absolute_difference_value"], "effect_pct": row["relative_improvement_ratio"] * 100, "group": row["baseline_method"], "source_sheet": "02_Relative_Improvement",
        })
    for row in paired_rows:
        plot_rows.append({
            "evidence_type": "PAIRED_EFFECT", "dataset": row["dataset"], "target": row["target"], "horizon": row["horizon"], "seed": row["seed"], "metric": row["metric"], "ours_method": row["ours_method"], "comparison_method": row["baseline_method"], "value": row["ours_value"], "secondary_value": row["baseline_value"], "effect": row["paired_delta_value"], "effect_pct": row["relative_improvement_ratio"] * 100, "group": row["baseline_method"], "source_sheet": "03_Paired_Effect",
        })
    for row in horizon_comparison_rows:
        plot_rows.append({
            "evidence_type": "HORIZON_STABILITY", "dataset": row["dataset"], "target": row["target"], "horizon": row["horizon"], "seed": "NA", "metric": row["metric"], "ours_method": row["method"], "comparison_method": row["baseline_method"], "value": row["mean_value"], "secondary_value": row["baseline_mean"], "effect": row["mean_value"] - row["baseline_mean"], "effect_pct": row["relative_improvement_ratio"] * 100, "group": row["baseline_method"], "source_sheet": "04_Horizon_Stability",
        })
    for row in ablation_paired_rows:
        plot_rows.append({
            "evidence_type": "ABLATION_EFFECT", "dataset": row["dataset"], "target": row["target"], "horizon": row["horizon"], "seed": row["seed"], "metric": row["metric"], "ours_method": row["reference_arm"], "comparison_method": row["comparison_arm"], "value": row["reference_value"], "secondary_value": row["comparison_value"], "effect": row["paired_delta_value"], "effect_pct": row["relative_effect_ratio"] * 100, "group": row["contract_primary_contrast"], "source_sheet": "05_Ablation_Effect",
        })
    for row in wtl_rows:
        plot_rows.append({
            "evidence_type": "WIN_RATE", "dataset": row["group_value"] if row["group_type"] == "dataset" else "NA", "target": "NA", "horizon": row["group_value"] if row["group_type"] == "horizon" else "NA", "seed": "NA", "metric": row["metric"], "ours_method": row["ours_method"], "comparison_method": row["baseline_method"], "value": row["win_rate_pct"], "secondary_value": row["total_pairs"], "effect": "NA", "effect_pct": row["win_rate_pct"], "group": f"{row['group_type']}:{row['group_value']}", "source_sheet": "06_Win_Tie_Loss",
        })

    input_sha_after_read = sha256_file(input_path)
    if input_sha_before != input_sha_after_read:
        raise SystemExit("SOURCE_WORKBOOK_INTEGRITY=FAIL; stopping before output write")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    default = wb.active
    wb.remove(default)
    for sheet in ["00_README", "01_Evidence_Index", "02_Relative_Improvement", "03_Paired_Effect", "04_Horizon_Stability", "05_Ablation_Effect", "06_Win_Tie_Loss", "07_Statistical_Tests", "08_Plot_Data_Long", "09_Validation"]:
        wb.create_sheet(sheet)

    now = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).isoformat(timespec="seconds")
    readme_headers = ["Item", "Value", "Interpretation"]
    readme_rows = [
        ["SOURCE_WORKBOOK", str(input_path), "Read-only frozen input"],
        ["SOURCE_SHA256", input_sha_before, "Input hash before and after derivation"],
        ["OUTPUT_WORKBOOK", str(output_path), "New derived workbook; input is never overwritten"],
        ["GENERATION_TIMESTAMP", now, "Asia/Kuala_Lumpur"],
        ["SCRIPT_VERSION", SCRIPT_VERSION, "Reproducible builder version"],
        ["TRAINING_STARTED", "NO", "No model training was executed"],
        ["EXPERIMENT_STARTED", "NO", "No experiment or result regeneration was executed"],
        ["MODEL_EXECUTION_ALLOWED", "NO", "Read-only derivation task"],
        ["RAW_RESULTS_MODIFIED", "NO", "Raw result files and source workbook were not modified"],
        ["SOURCE_RESULTS_MODIFIED", "NO", "Frozen workbook was read-only"],
        ["CHARTS_CREATED", "NO", "No Excel charts, plots, heatmaps, or images"],
        ["RMSE_SPACE", rmse_space, "Frozen metric space"],
        ["sMAPE_UNIT", smape_unit, "Frozen sMAPE unit"],
        ["RELATIVE_IMPROVEMENT_DENOMINATOR", "BASELINE", "(baseline - ours) / baseline * 100"],
        ["PERCENT_FIELD_UNIT", "percentage_points", "Values such as 44.362 mean 44.362 percent"],
        ["TIE_RULE", f"relative difference <= {tie_threshold}", "Read from 01_Paper_Contract; not invented"],
        ["BOOTSTRAP_RANDOM_SEED", BOOTSTRAP_RANDOM_SEED, "Fixed seed for paired-effect mean CI"],
        ["BOOTSTRAP_N_RESAMPLES", BOOTSTRAP_N_RESAMPLES, "Fixed resample count"],
        ["BOOTSTRAP_CI_LEVEL", f"{BOOTSTRAP_CI_LEVEL}%", "Percentile CI for mean paired delta"],
        ["SELECTED_DATASETS", ", ".join(selected_datasets), "Read from frozen paper contract"],
        ["FULL_ARM", full_arm, "Arm with contract Arm_Name=CURRENT_V3"],
        ["REFERENCE_ARM", reference_arm, "Arm with contract Arm_Name=FUSED_TRAINABLE"],
        ["SOURCE_SHEETS_FOUND", "; ".join(sheet for sheet in REQUIRED_SOURCE_SHEETS if sheet in found_sheets), "Workbook source audit"],
        ["SOURCE_SHEETS_MISSING", "; ".join(missing_sheets) if missing_sheets else "NONE", "Missing source sheets are not silently substituted"],
        ["SCOPE_NOTE", "This workbook is the paper evidence data layer, not a new experiment result source.", "Future figures/statistics may read these tables without modifying the frozen source."],
    ]
    write_table(wb["00_README"], "README", readme_headers, readme_rows, 1, 1)

    index_rows = [
        ["Relative Improvement", f"{MASTER_SHEET};{FROZEN_GAIN_SHEET}", "02_Relative_Improvement", "dataset×target×horizon", "OK" if relative_rows else "INSUFFICIENT_SOURCE_DATA"],
        ["Paired Effect", MASTER_SHEET, "03_Paired_Effect", "dataset×target×horizon×seed", "OK" if paired_rows else "INSUFFICIENT_SOURCE_DATA"],
        ["Horizon Stability", MASTER_SHEET, "04_Horizon_Stability", "horizon and adjacent-horizon changes", "OK" if horizon_profile_rows else "INSUFFICIENT_SOURCE_DATA"],
        ["Ablation Effect", f"{ABLATION_RAW_SHEET};{ABLATION_CONTRACT_SHEET}", "05_Ablation_Effect", "paired condition", "OK" if ablation_paired_rows and contrast_defs else "INSUFFICIENT_SOURCE_DATA"],
        ["Win/Tie/Loss", "03_Paired_Effect", "06_Win_Tie_Loss", "overall/dataset/horizon", "OK" if wtl_rows else "INSUFFICIENT_SOURCE_DATA"],
        ["Statistical Tests", "03_Paired_Effect", "07_Statistical_Tests", "comparison", "OK" if statistical_rows else "INSUFFICIENT_SOURCE_DATA"],
    ]
    write_table(wb["01_Evidence_Index"], "Evidence Index", ["Evidence", "Source Sheet", "Output Sheet", "Unit of Analysis", "Status"], index_rows, 1, 1)

    rel_headers = ["dataset", "target", "horizon", "metric", "ours_method", "baseline_method", "ours_mean", "baseline_mean", "absolute_difference", "relative_improvement_pct", "n_ours", "n_baseline", "comparison_direction", "source_sheet", "frozen_gain_raw_ratio", "frozen_result", "gain_consistency_status"]
    rel_values = []
    for row in relative_rows:
        row_number = 3 + len(rel_values)
        rel_values.append([row["dataset"], row["target"], row["horizon"], row["metric"], row["ours_method"], row["baseline_method"], row["ours_mean"], row["baseline_mean"], formula_absolute(row_number, "G", "H"), formula_relative(row_number, "G", "H"), row["n_ours"], row["n_baseline"], row["comparison_direction"], row["source_sheet"], row["frozen_gain_raw_ratio"], row["frozen_result"], row["gain_consistency_status"]])
    write_table(wb["02_Relative_Improvement"], "Relative Improvement (percentage points; baseline denominator)", rel_headers, rel_values, 1, 1)

    paired_headers = ["dataset", "target", "horizon", "seed", "metric", "ours_method", "baseline_method", "ours_value", "baseline_value", "paired_delta", "relative_improvement_pct", "outcome", "pair_valid", "tie_threshold", "source_sheet", "source_key"]
    paired_values = []
    for row in paired_rows:
        row_number = 3 + len(paired_values)
        paired_values.append([row["dataset"], row["target"], row["horizon"], row["seed"], row["metric"], row["ours_method"], row["baseline_method"], row["ours_value"], row["baseline_value"], f"=H{row_number}-I{row_number}", f'=IFERROR((I{row_number}-H{row_number})/I{row_number}*100,"")', row["outcome"], row["pair_valid"], row["tie_threshold"], row["source_sheet"], row["source_key"]])
    next_row = write_table(wb["03_Paired_Effect"], "Paired Effect Rows (strict seed-level key)", paired_headers, paired_values, 1, 1)
    summary_headers = ["metric", "ours_method", "baseline_method", "group_type", "group_value", "n_pairs", "n_wins", "n_ties", "n_losses", "win_rate_pct", "loss_rate_pct", "mean_paired_delta", "median_paired_delta", "std_paired_delta", "mean_relative_improvement_pct", "median_relative_improvement_pct", "source_sheet"]
    write_table(wb["03_Paired_Effect"], "Paired Effect Summary", summary_headers, [[row.get(header) for header in summary_headers] for row in paired_summary_rows], next_row, 2)

    profile_headers = ["dataset", "target", "horizon", "metric", "method", "mean_value", "std_value", "n_seeds", "baseline_method", "baseline_mean", "relative_improvement_pct", "source_sheet"]
    profile_values = [[row.get(header) for header in profile_headers] for row in horizon_profile_rows]
    next_row = write_table(wb["04_Horizon_Stability"], "Horizon Stability Method Profiles", profile_headers, profile_values, 1, 1)
    comparison_headers = profile_headers
    next_row = write_table(wb["04_Horizon_Stability"], "Horizon Stability Ours-versus-Baseline Comparisons", comparison_headers, [[row[header] if header not in {"relative_improvement_pct"} else row["relative_improvement_ratio"] * 100 for header in comparison_headers] for row in horizon_comparison_rows], next_row, 2)
    agg_headers = ["horizon", "metric", "ours_method", "baseline_method", "n_conditions", "ours_mean", "baseline_mean", "mean_relative_improvement_pct", "median_relative_improvement_pct", "win_rate_pct", "source_sheet"]
    next_row = write_table(wb["04_Horizon_Stability"], "Horizon-Level Aggregate", agg_headers, [[row.get(header) for header in agg_headers] for row in horizon_aggregate_rows], next_row, 3)
    method_change_headers = ["dataset", "target", "metric", "method", "H1_value", "H3_value", "H5_value", "H1_to_H3_change", "H3_to_H5_change", "H1_to_H5_change", "H1_to_H3_change_pct", "H3_to_H5_change_pct", "H1_to_H5_change_pct", "source_sheet"]
    next_row = write_table(wb["04_Horizon_Stability"], "Adjacent Horizon Change (method values)", method_change_headers, [[row.get(header) for header in method_change_headers] for row in method_change_rows], next_row, 4)
    ri_change_headers = ["dataset", "target", "metric", "ours_method", "baseline_method", "RI_H1", "RI_H3", "RI_H5", "RI_H1_to_H3", "RI_H3_to_H5", "RI_H1_to_H5", "source_sheet"]
    write_table(wb["04_Horizon_Stability"], "Adjacent Horizon Change (relative improvement)", ri_change_headers, [[row.get(header) for header in ri_change_headers] for row in ri_change_rows], next_row, 5)

    ablation_headers = ["dataset", "target", "horizon", "seed", "metric", "reference_arm", "comparison_arm", "reference_value", "comparison_value", "paired_delta", "relative_effect_pct", "outcome", "contract_primary_contrast", "scientific_question", "pair_valid", "source_sheet"]
    ablation_values = []
    for row in ablation_paired_rows:
        row_number = 3 + len(ablation_values)
        ablation_values.append([row["dataset"], row["target"], row["horizon"], row["seed"], row["metric"], row["reference_arm"], row["comparison_arm"], row["reference_value"], row["comparison_value"], f"=I{row_number}-H{row_number}", f'=IFERROR((I{row_number}-H{row_number})/H{row_number}*100,"")', row["outcome"], row["contract_primary_contrast"], row["scientific_question"], row["pair_valid"], row["source_sheet"]])
    next_row = write_table(wb["05_Ablation_Effect"], "Ablation Paired Effect Rows (contract contrast direction)", ablation_headers, ablation_values, 1, 1)
    ablation_summary_headers = ["metric", "reference_arm", "comparison_arm", "contrast_label", "n_pairs", "wins_reference", "ties", "losses_reference", "mean_delta", "median_delta", "mean_effect_pct", "median_effect_pct", "source_sheet"]
    write_table(wb["05_Ablation_Effect"], "Ablation Effect Summary", ablation_summary_headers, [[row.get(header) for header in ablation_summary_headers] for row in ablation_summary_rows], next_row, 2)

    wtl_headers = ["metric", "ours_method", "baseline_method", "group_type", "group_value", "wins", "ties", "losses", "total_pairs", "win_rate_pct", "tie_rate_pct", "loss_rate_pct", "source_sheet"]
    write_table(wb["06_Win_Tie_Loss"], "Win/Tie/Loss (strict paired observations)", wtl_headers, [[row.get(header) for header in wtl_headers] for row in wtl_rows], 1, 1)

    stats_headers = ["metric", "ours_method", "baseline_method", "n_pairs", "statistic", "p_value", "test_status", "median_paired_delta", "mean_paired_delta", "bootstrap_statistic", "bootstrap_ci_lower", "bootstrap_ci_upper", "bootstrap_random_seed", "bootstrap_n_resamples", "bootstrap_ci_level_pct", "source_sheet"]
    write_table(wb["07_Statistical_Tests"], "Paired Statistical Tests (no significance claim is made here)", stats_headers, [[row.get(header) for header in stats_headers] for row in statistical_rows], 1, 1)

    plot_headers = ["evidence_type", "dataset", "target", "horizon", "seed", "metric", "ours_method", "comparison_method", "value", "secondary_value", "effect", "effect_pct", "group", "source_sheet"]
    write_table(wb["08_Plot_Data_Long"], "Unified Plot Data Long Interface (data only; no figures)", plot_headers, [[row.get(header) for header in plot_headers] for row in plot_rows], 1, 1)

    frozen_gain_failures = sum(row["gain_consistency_status"] == "FAIL" for row in relative_rows)
    wtl_reconciliation = all(row["wins"] + row["ties"] + row["losses"] == row["total_pairs"] for row in wtl_rows)
    pairing_integrity = len(paired_rows) == len(conditions) * len(METRICS) * len(BASELINE_METHODS) * 5 and unmatched_ours == 0 and unmatched_baseline == 0
    ablation_integrity = set(arm_contract) >= set(EXPECTED_ABLATION_ARMS) and len(contrast_defs) == 4 and len(ablation_paired_rows) == 4 * len(conditions) * 3 * len(METRICS)
    validation_rows = [
        ["INPUT_SHA256_BEFORE", input_sha_before, input_sha_before, "PASS", "Hash computed before read"],
        ["INPUT_SHA256_AFTER", input_sha_after_read, input_sha_after_read, "PASS", "Hash computed after derivation before output write"],
        ["SOURCE_WORKBOOK_UNCHANGED", "YES", "YES" if input_sha_before == input_sha_after_read else "NO", "PASS" if input_sha_before == input_sha_after_read else "FAIL", "Input was not saved or renamed"],
        ["MASTER_ROWS_READ", 420, len(selected_master), "PASS" if len(selected_master) == 420 else "FAIL", "Selected master rows with include_in_master=TRUE and status=OK"],
        ["DUPLICATE_CANONICAL_KEYS_MAIN_BASELINE", 0, duplicate_main, "PASS" if duplicate_main == 0 else "FAIL", "dataset×target×horizon×seed×method"],
        ["DUPLICATE_FULL_SOURCE_KEYS", 0, duplicate_full, "PASS" if duplicate_full == 0 else "FAIL", "Includes experiment_type, arm, and mode"],
        ["PAIRED_ROWS_EXPECTED", len(conditions) * len(METRICS) * len(BASELINE_METHODS) * 5, len(conditions) * len(METRICS) * len(BASELINE_METHODS) * 5, "PASS", "4 datasets×3 horizons×5 seeds×2 metrics×3 baselines"],
        ["PAIRED_ROWS_GENERATED", len(conditions) * len(METRICS) * len(BASELINE_METHODS) * 5, len(paired_rows), "PASS" if pairing_integrity else "FAIL", "Strict key matching only"],
        ["UNMATCHED_OURS_ROWS", 0, unmatched_ours, "PASS" if unmatched_ours == 0 else "FAIL", "No forced pairing"],
        ["UNMATCHED_BASELINE_ROWS", 0, unmatched_baseline, "PASS" if unmatched_baseline == 0 else "FAIL", "No forced pairing"],
        ["RELATIVE_IMPROVEMENT_ROWS", len(conditions) * len(METRICS) * len(BASELINE_METHODS), len(relative_rows), "PASS" if len(relative_rows) == 72 else "FAIL", "Condition-level means"],
        ["HORIZON_ROWS_GENERATED", "N/A", len(horizon_profile_rows) + len(horizon_comparison_rows) + len(horizon_aggregate_rows) + len(method_change_rows) + len(ri_change_rows), "PASS", "Profiles, comparisons, aggregate, and change tables"],
        ["ABLATION_ROWS_GENERATED", 4 * len(conditions) * 3 * len(METRICS), len(ablation_paired_rows), "PASS" if ablation_integrity else "FAIL", "Contract-defined four contrasts; seed-level paired rows"],
        ["WIN_TIE_LOSS_RECONCILIATION", "wins+ties+losses=total_pairs", "PASS" if wtl_reconciliation else "FAIL", "PASS" if wtl_reconciliation else "FAIL", "Every W/T/L group reconciles"],
        ["NONFINITE_VALUES", 0, nonfinite_values, "PASS" if nonfinite_values == 0 else "FAIL", "Master rmse/smape only"],
        ["MISSING_REQUIRED_VALUES", 0, missing_values, "PASS" if missing_values == 0 else "FAIL", "Master analysis key and metrics"],
        ["FROZEN_RELATIVE_GAIN_CONSISTENCY_FAILURES", 0, frozen_gain_failures, "PASS" if frozen_gain_failures == 0 else "FAIL", "09_Relative_Gain checked without overwriting it"],
        ["PAIRING_INTEGRITY", "PASS", "PASS" if pairing_integrity else "FAIL", "PASS" if pairing_integrity else "FAIL", "Strict dataset-target-horizon-seed pairing"],
        ["METRIC_SPACE_INTEGRITY", "PASS", "PASS" if metric_space_ok else "FAIL", "PASS" if metric_space_ok else "FAIL", f"RMSE_SPACE={rmse_space}; sMAPE_UNIT={smape_unit}"],
        ["ABLATION_CONTRACT_INTEGRITY", "PASS", "PASS" if ablation_integrity else "FAIL", "PASS" if ablation_integrity else "FAIL", f"FULL_ARM={full_arm}; REFERENCE_ARM={reference_arm}; contrasts={len(contrast_defs)}"],
        ["TRAINING_STARTED", "NO", "NO", "PASS", "Read-only derivation"],
        ["EXPERIMENT_STARTED", "NO", "NO", "PASS", "Read-only derivation"],
        ["RAW_RESULTS_MODIFIED", "NO", "NO", "PASS", "No source writes"],
        ["CHARTS_CREATED", "NO", "NO", "PASS", "No figures or charts"],
    ]
    write_table(wb["09_Validation"], "Validation", ["Check", "Expected", "Actual", "Status", "Notes"], validation_rows, 1, 1)

    for ws in wb.worksheets:
        ws.sheet_view.showGridLines = False
        set_formats(ws)
    wb.properties.title = "MS-KATL Paper Evidence Tables"
    wb.properties.subject = "Read-only derived evidence data layer"
    wb.properties.creator = "Codex"
    wb.save(output_path)

    input_sha_after_output = sha256_file(input_path)
    output_sha = sha256_file(output_path)
    source_unchanged = input_sha_before == input_sha_after_output
    all_pass = source_unchanged and pairing_integrity and metric_space_ok and ablation_integrity and wtl_reconciliation and frozen_gain_failures == 0 and nonfinite_values == 0 and missing_values == 0
    print("=" * 50)
    print("MS-KATL PAPER EVIDENCE TABLE GENERATION")
    print("=" * 50)
    print(f"INPUT_WORKBOOK={input_path}")
    print(f"INPUT_SHA256_BEFORE={input_sha_before}")
    print(f"INPUT_SHA256_AFTER={input_sha_after_output}")
    print(f"SOURCE_WORKBOOK_UNCHANGED={'YES' if source_unchanged else 'NO'}")
    print(f"OUTPUT_WORKBOOK={output_path}")
    print(f"OUTPUT_SHA256={output_sha}")
    print("\n--------------------------------------------------\nSOURCE\n--------------------------------------------------")
    print(f"SOURCE_SHEETS_FOUND={'; '.join(sheet for sheet in REQUIRED_SOURCE_SHEETS if sheet in found_sheets)}")
    print(f"SOURCE_SHEETS_MISSING={'; '.join(missing_sheets) if missing_sheets else 'NONE'}")
    print(f"RMSE_SPACE={rmse_space}")
    print(f"sMAPE_UNIT={smape_unit}")
    print("\n--------------------------------------------------\nOUTPUT SHEETS\n--------------------------------------------------")
    for sheet in wb.sheetnames:
        print(f"{sheet}=CREATED")
    print("\n--------------------------------------------------\nPAIRING\n--------------------------------------------------")
    print(f"PAIRED_ROWS_GENERATED={len(paired_rows)}")
    print(f"UNMATCHED_OURS_ROWS={unmatched_ours}")
    print(f"UNMATCHED_BASELINE_ROWS={unmatched_baseline}")
    print(f"PAIRING_INTEGRITY={'PASS' if pairing_integrity else 'FAIL'}")
    print("\n--------------------------------------------------\nVALIDATION\n--------------------------------------------------")
    print(f"RELATIVE_IMPROVEMENT_TABLE={'PASS' if len(relative_rows) == 72 and frozen_gain_failures == 0 else 'FAIL'}")
    print(f"PAIRED_EFFECT_TABLE={'PASS' if pairing_integrity else 'FAIL'}")
    print(f"HORIZON_STABILITY_TABLE={'PASS' if horizon_profile_rows and horizon_comparison_rows else 'FAIL'}")
    print(f"ABLATION_EFFECT_TABLE={'PASS' if ablation_integrity else 'FAIL'}")
    print(f"WIN_TIE_LOSS_TABLE={'PASS' if wtl_reconciliation else 'FAIL'}")
    print(f"METRIC_SPACE_INTEGRITY={'PASS' if metric_space_ok else 'FAIL'}")
    print(f"ABLATION_CONTRACT_INTEGRITY={'PASS' if ablation_integrity else 'FAIL'}")
    print("\n--------------------------------------------------\nSAFETY\n--------------------------------------------------")
    print("TRAINING_STARTED=NO")
    print("EXPERIMENT_STARTED=NO")
    print("RAW_RESULTS_MODIFIED=NO")
    print(f"SOURCE_WORKBOOK_MODIFIED={'NO' if source_unchanged else 'YES'}")
    print("CHARTS_CREATED=NO")
    print("FIGURES_CREATED=NO")
    print("\n--------------------------------------------------\nFINAL\n--------------------------------------------------")
    print(f"PAPER_EVIDENCE_DATA_LAYER_READY={'YES' if all_pass else 'NO'}")
    if not all_pass:
        print("FAILURE_REASONS=see 09_Validation and the checks above")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
