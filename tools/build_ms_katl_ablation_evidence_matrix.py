#!/usr/bin/env python3
"""Build a standalone MS-KATL A-E evidence workbook from one frozen XLSX.

The input is deliberately restricted to the exact frozen workbook supplied by
the paper-data directory.  The reader uses the XLSX XML package directly so
the audit does not need to import a project data loader or read repository
results.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    import xlsxwriter
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("xlsxwriter is required for workbook generation") from exc


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"

DATA_DIR = Path("/Users/ming/Desktop/复现实验/论文文稿相关")
INPUT_PATH = DATA_DIR / "MS_KATL_Paper_Data_Freeze_v1_2_FINAL_FROZEN.xlsx"
OUTPUT_PATH = DATA_DIR / "MS_KATL_Ablation_Evidence_Matrix.xlsx"
EXPECTED_DATASETS = ["D1", "D4", "D5", "D6"]
EXPECTED_HORIZONS = [1, 3, 5]
EXPECTED_SEEDS = [42, 43, 44]
EXPECTED_ARMS = ["A", "B", "C", "D", "E"]
CONTRASTS = [
    ("C_vs_A", "C", "A"),
    ("C_vs_B", "C", "B"),
    ("C_vs_E", "C", "E"),
    ("D_vs_C", "D", "C"),
]
METRICS = ["RMSE", "sMAPE"]
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_REPS = 10000


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def col_to_num(col: str) -> int:
    total = 0
    for ch in col:
        if ch.isalpha():
            total = total * 26 + ord(ch.upper()) - 64
    return total


def cell_ref_parts(ref: str) -> tuple[int, int]:
    m = re.match(r"([A-Za-z]+)([0-9]+)$", ref)
    if not m:
        raise ValueError(f"Invalid cell reference: {ref}")
    return int(m.group(2)), col_to_num(m.group(1))


def text_value(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext())


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values = []
    for si in root:
        values.append(text_value(si))
    return values


def workbook_sheet_targets(zf: zipfile.ZipFile) -> dict[str, str]:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {}
    for rel in rels:
        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target", "")
        if target.startswith("/"):
            target = target.lstrip("/")
        elif not target.startswith("xl/"):
            target = "xl/" + target
        rel_map[rid] = target
    result = {}
    for sheet in wb.find(f"{{{NS_MAIN}}}sheets"):
        name = sheet.attrib["name"]
        rid = sheet.attrib[f"{{{NS_REL}}}id"]
        result[name] = rel_map[rid]
    return result


def scalar_cell_value(cell: ET.Element, shared: list[str]):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return text_value(cell.find(f"{{{NS_MAIN}}}is"))
    v = cell.find(f"{{{NS_MAIN}}}v")
    raw = "" if v is None else v.text or ""
    if cell_type == "s":
        return shared[int(raw)] if raw else ""
    if cell_type == "b":
        return raw == "1"
    if raw == "":
        return ""
    try:
        number = float(raw)
        if number.is_integer():
            return int(number)
        return number
    except ValueError:
        return raw


def read_sheet(path: Path, sheet_name: str) -> list[list[object]]:
    with zipfile.ZipFile(path) as zf:
        targets = workbook_sheet_targets(zf)
        if sheet_name not in targets:
            raise KeyError(f"Missing sheet {sheet_name}")
        root = ET.fromstring(zf.read(targets[sheet_name]))
        shared = read_shared_strings(zf)
    rows: dict[int, dict[int, object]] = {}
    max_col = 0
    for row in root.iter(f"{{{NS_MAIN}}}row"):
        row_num = int(row.attrib["r"])
        cells = {}
        for cell in row:
            if local_name(cell.tag) != "c":
                continue
            _, col_num = cell_ref_parts(cell.attrib["r"])
            cells[col_num] = scalar_cell_value(cell, shared)
            max_col = max(max_col, col_num)
        rows[row_num] = cells
    if not rows:
        return []
    max_row = max(rows)
    return [[rows.get(r, {}).get(c, "") for c in range(1, max_col + 1)] for r in range(1, max_row + 1)]


def records_from_sheet(rows: list[list[object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    header = [str(x).strip() for x in rows[0]]
    header = [x if x else f"_unnamed_{i}" for i, x in enumerate(header)]
    records = []
    for row in rows[1:]:
        if not any(str(x).strip() for x in row):
            continue
        padded = row + [""] * (len(header) - len(row))
        records.append(dict(zip(header, padded[: len(header)])))
    return records


def norm_key(value: object) -> str:
    return str(value).strip()


def norm_upper(value: object) -> str:
    return norm_key(value).upper()


def num(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_field(fields: list[str], candidates: list[str], required: bool = True) -> str | None:
    normalized = {re.sub(r"[^a-z0-9]", "", f.lower()): f for f in fields}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    for field in fields:
        compact = re.sub(r"[^a-z0-9]", "", field.lower())
        if any(re.sub(r"[^a-z0-9]", "", c.lower()) in compact for c in candidates):
            return field
    if required:
        raise KeyError(f"Cannot resolve field from {candidates}; fields={fields}")
    return None


def resolve_raw_fields(records: list[dict[str, object]]) -> dict[str, str | None]:
    fields = list(records[0]) if records else []
    return {
        "dataset": find_field(fields, ["dataset", "dataset_id"]),
        "target": find_field(fields, ["target", "target_id"]),
        "horizon": find_field(fields, ["horizon", "h"]),
        "seed": find_field(fields, ["seed", "random_seed"]),
        "arm": find_field(fields, ["arm", "ablation_arm"]),
        "metric_space": find_field(fields, ["metric_space", "metricspace"], required=False),
        "status": find_field(fields, ["status", "source_status", "result_status"], required=False),
        "rmse": find_field(fields, ["rmse", "rmse_original", "RMSE"], required=False),
        "smape": find_field(fields, ["smape", "smape_original", "sMAPE"], required=False),
    }


def parse_contract(rows: list[list[object]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        values = [norm_key(x) for x in row if norm_key(x)]
        if len(values) >= 2:
            result[values[0]] = values[1]
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def iqr(values: list[float]) -> float:
    return percentile(values, 0.75) - percentile(values, 0.25)


def wilcoxon_exact(values: list[float]) -> tuple[float, float, float, float, int]:
    nonzero = [v for v in values if abs(v) > 1e-12]
    n = len(nonzero)
    if not nonzero:
        return 0.0, 0.0, 1.0, 1.0, 0
    abs_values = [abs(v) for v in nonzero]
    order = sorted(range(n), key=lambda i: abs_values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs_values[order[j]] == abs_values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    w_pos = sum(r for r, v in zip(ranks, nonzero) if v > 0)
    w_neg = sum(r for r, v in zip(ranks, nonzero) if v < 0)
    statistic = min(w_pos, w_neg)
    scale = 2
    scaled = [int(round(r * scale)) for r in ranks]
    observed = int(round(w_pos * scale))
    distribution = []
    for mask in range(1 << n):
        total = sum(scaled[i] for i in range(n) if mask & (1 << i))
        distribution.append(total)
    lower = sum(x <= observed for x in distribution) / len(distribution)
    upper = sum(x >= observed for x in distribution) / len(distribution)
    p_value = min(1.0, 2.0 * min(lower, upper))
    return statistic, w_pos, w_neg, p_value, n


def bootstrap_mean_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(BOOTSTRAP_SEED)
    means = []
    n = len(values)
    for _ in range(BOOTSTRAP_REPS):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    return percentile(means, 0.025), percentile(means, 0.975)


def winner(delta: float) -> str:
    if delta > 1e-12:
        return "TREATMENT_WIN"
    if delta < -1e-12:
        return "REFERENCE_WIN"
    return "TIE"


def direction(win_count: int, total: int) -> str:
    if not total:
        return "MIXED"
    rate = win_count / total
    if rate > 0.5:
        return "FAVORABLE"
    if rate < 0.5:
        return "UNFAVORABLE"
    return "MIXED"


def consistency(win_count: int, total: int) -> str:
    rate = win_count / total if total else 0.0
    if rate >= 0.90:
        return "VERY_HIGH"
    if rate >= 0.75:
        return "HIGH"
    if rate >= 0.60:
        return "MODERATE"
    return "LOW"


def scientific_question(contrast: str) -> str:
    return {
        "C_vs_A": "完整 C 机制相对于基础/弱化配置是否产生总体增益？",
        "C_vs_B": "target-specific adaptation 相对于 frozen fused initialization 是否带来额外收益？",
        "C_vs_E": "multi-source fused initialization 相对于 best-single-source initialization 是否更有效？",
        "D_vs_C": "D 相对于正式 C 配置增加/改变的机制是否带来额外收益？",
    }[contrast]


def effect_interpretation(median_ri: float) -> str:
    if median_ri >= 20:
        return "LARGE_POSITIVE"
    if median_ri >= 10:
        return "MODERATE_POSITIVE"
    if median_ri > 0:
        return "MODEST_POSITIVE"
    if abs(median_ri) <= 1e-12:
        return "NO_MEDIAN_CHANGE"
    return "UNFAVORABLE_MEDIAN"


def claim_text(grade: str, heterogeneity: str) -> tuple[str, str]:
    if grade == "STRONG" and heterogeneity in {"LOW", "MODERATE"}:
        return (
            "consistently improves and provides a robust advantage at the condition level",
            "uniformly large improvement; equally effective across all datasets",
        )
    if grade == "STRONG":
        return (
            "consistently improves at the condition level, although the magnitude of the gain varies across datasets",
            "uniformly large improvement; equally effective across all datasets",
        )
    if grade == "MODERATE":
        return (
            "generally improves and shows an overall favorable tendency",
            "consistently outperforms in all settings; universally improves",
        )
    if grade == "LIMITED":
        return (
            "shows benefits in several conditions; the benefit is dataset-dependent",
            "robustly improves; universally superior; consistently effective",
        )
    return (
        "shows conflicting or insufficient directional evidence",
        "robust improvement; universal superiority; consistent effectiveness",
    )


def grade_for(win_rate: float, seed_rate: float, dataset_pos: int, dataset_neg: int, dir_value: str) -> str:
    if dir_value == "MIXED" or dataset_pos and dataset_neg and win_rate < 0.70:
        return "CONFLICTING"
    if win_rate >= 0.90 and seed_rate >= 0.75 and dataset_neg == 0:
        return "STRONG"
    if win_rate >= 0.70 and seed_rate >= 0.60 and dataset_pos >= 2:
        return "MODERATE"
    if win_rate > 0.50:
        return "LIMITED"
    return "CONFLICTING"


def build_analysis(raw: list[dict[str, object]], fields: dict[str, str | None], contract: dict[str, str]):
    filtered = []
    for row in raw:
        dataset = norm_upper(row[fields["dataset"]])
        if dataset not in EXPECTED_DATASETS:
            continue
        normalized = {
            "dataset": dataset,
            "target": norm_key(row[fields["target"]]),
            "horizon": int(float(row[fields["horizon"]])),
            "seed": int(float(row[fields["seed"]])),
            "arm": norm_upper(row[fields["arm"]]),
            "metric_space": norm_key(row[fields["metric_space"]]) if fields["metric_space"] else "",
            "status": norm_upper(row[fields["status"]]) if fields["status"] else "",
            "RMSE": num(row[fields["rmse"]]) if fields["rmse"] else None,
            "sMAPE": num(row[fields["smape"]]) if fields["smape"] else None,
        }
        filtered.append(normalized)
    key_fields = ["dataset", "target", "horizon", "seed", "arm"]
    key_counts = Counter(tuple(row[x] for x in key_fields) for row in filtered)
    conditions = sorted({(r["dataset"], r["target"], r["horizon"]) for r in filtered})
    paired_units = sorted({(r["dataset"], r["target"], r["horizon"], r["seed"]) for r in filtered})
    by_key = {tuple(row[x] for x in key_fields): row for row in filtered if key_counts[tuple(row[x] for x in key_fields)] == 1}
    seed_rows = []
    condition_rows = []
    dataset_rows = []
    horizon_rows = []
    summary_rows = []
    stat_rows = []
    for contrast, treatment, reference in CONTRASTS:
        for metric in METRICS:
            seed_effects = []
            for dataset, target, horizon, seed in paired_units:
                t = by_key.get((dataset, target, horizon, seed, treatment), {})
                r = by_key.get((dataset, target, horizon, seed, reference), {})
                tv = t.get(metric)
                rv = r.get(metric)
                if tv is None or rv is None:
                    continue
                delta = rv - tv
                ri = delta / rv * 100 if rv else None
                seed_effects.append({
                    "dataset": dataset, "target": target, "horizon": horizon, "seed": seed,
                    "contrast": contrast, "metric": metric, "treatment_arm": treatment,
                    "reference_arm": reference, "treatment_value": tv, "reference_value": rv,
                    "delta": delta, "relative_improvement_pct": ri, "winner": winner(delta),
                })
                seed_rows.append(seed_effects[-1])
            for dataset, target, horizon in conditions:
                vals = [x for x in seed_effects if (x["dataset"], x["target"], x["horizon"]) == (dataset, target, horizon)]
                if not vals:
                    continue
                tvs = [x["treatment_value"] for x in vals]
                rvs = [x["reference_value"] for x in vals]
                treatment_mean = sum(tvs) / len(tvs)
                reference_mean = sum(rvs) / len(rvs)
                delta = reference_mean - treatment_mean
                ri = delta / reference_mean * 100 if reference_mean else None
                wins = sum(x["winner"] == "TREATMENT_WIN" for x in vals)
                ties = sum(x["winner"] == "TIE" for x in vals)
                losses = sum(x["winner"] == "REFERENCE_WIN" for x in vals)
                condition_rows.append({
                    "dataset": dataset, "target": target, "horizon": horizon,
                    "contrast": contrast, "metric": metric, "treatment_arm": treatment,
                    "reference_arm": reference, "n_seeds": len(vals),
                    "treatment_mean": treatment_mean, "treatment_std": statistics.stdev(tvs) if len(tvs) > 1 else 0.0,
                    "reference_mean": reference_mean, "reference_std": statistics.stdev(rvs) if len(rvs) > 1 else 0.0,
                    "delta": delta, "relative_improvement_pct": ri,
                    "winner": winner(delta), "direction": "FAVORABLE" if delta > 1e-12 else ("UNFAVORABLE" if delta < -1e-12 else "MIXED"),
                    "seed_wins": wins, "seed_ties": ties, "seed_losses": losses,
                    "seed_win_rate_pct": wins / len(vals) * 100,
                })
            conds = [x for x in condition_rows if x["contrast"] == contrast and x["metric"] == metric]
            cond_deltas = [x["delta"] for x in conds]
            cond_ris = [x["relative_improvement_pct"] for x in conds if x["relative_improvement_pct"] is not None]
            cwins = sum(x["winner"] == "TREATMENT_WIN" for x in conds)
            cties = sum(x["winner"] == "TIE" for x in conds)
            closses = sum(x["winner"] == "REFERENCE_WIN" for x in conds)
            dataset_effects = []
            for dataset in EXPECTED_DATASETS:
                ds = [x for x in conds if x["dataset"] == dataset]
                ds_ris = [x["relative_improvement_pct"] for x in ds if x["relative_improvement_pct"] is not None]
                ds_w = sum(x["winner"] == "TREATMENT_WIN" for x in ds)
                ds_t = sum(x["winner"] == "TIE" for x in ds)
                ds_l = sum(x["winner"] == "REFERENCE_WIN" for x in ds)
                ds_mean_delta = sum(x["delta"] for x in ds) / len(ds) if ds else None
                ds_mean_ri = sum(ds_ris) / len(ds_ris) if ds_ris else None
                ds_dir = "POSITIVE" if ds_mean_ri is not None and ds_mean_ri > 1e-12 else ("NEGATIVE" if ds_mean_ri is not None and ds_mean_ri < -1e-12 else "MIXED")
                dataset_effects.append({"dataset": dataset, "contrast": contrast, "metric": metric, "n_conditions": len(ds),
                                        "condition_wins": ds_w, "condition_ties": ds_t, "condition_losses": ds_l,
                                        "condition_win_rate_pct": ds_w / len(ds) * 100 if ds else None,
                                        "mean_delta": ds_mean_delta, "median_delta": statistics.median([x["delta"] for x in ds]) if ds else None,
                                        "mean_relative_improvement_pct": ds_mean_ri,
                                        "median_relative_improvement_pct": statistics.median(ds_ris) if ds_ris else None,
                                        "min_relative_improvement_pct": min(ds_ris) if ds_ris else None,
                                        "max_relative_improvement_pct": max(ds_ris) if ds_ris else None,
                                        "dataset_direction": ds_dir,
                                        "dataset_effect_interpretation": effect_interpretation(statistics.median(ds_ris) if ds_ris else 0.0)})
                dataset_rows.append(dataset_effects[-1])
            for horizon in EXPECTED_HORIZONS:
                hs = [x for x in conds if x["horizon"] == horizon]
                hris = [x["relative_improvement_pct"] for x in hs if x["relative_improvement_pct"] is not None]
                hw = sum(x["winner"] == "TREATMENT_WIN" for x in hs)
                ht = sum(x["winner"] == "TIE" for x in hs)
                hl = sum(x["winner"] == "REFERENCE_WIN" for x in hs)
                horizon_rows.append({"horizon": horizon, "contrast": contrast, "metric": metric,
                                     "n_dataset_conditions": len(hs), "wins": hw, "ties": ht, "losses": hl,
                                     "win_rate_pct": hw / len(hs) * 100 if hs else None,
                                     "mean_relative_improvement_pct": sum(hris) / len(hris) if hris else None,
                                     "median_relative_improvement_pct": statistics.median(hris) if hris else None,
                                     "horizon_direction": direction(hw, len(hs))})
            seed_all = [x for x in seed_rows if x["contrast"] == contrast and x["metric"] == metric]
            sw = sum(x["winner"] == "TREATMENT_WIN" for x in seed_all)
            st = sum(x["winner"] == "TIE" for x in seed_all)
            sl = sum(x["winner"] == "REFERENCE_WIN" for x in seed_all)
            ds_selected = [x for x in dataset_effects]
            dpos = sum(x["dataset_direction"] == "POSITIVE" for x in ds_selected)
            dmix = sum(x["dataset_direction"] == "MIXED" for x in ds_selected)
            dneg = sum(x["dataset_direction"] == "NEGATIVE" for x in ds_selected)
            sign_reversal = any(x > 1e-12 for x in cond_ris) and any(x < -1e-12 for x in cond_ris)
            range_ri = max(cond_ris) - min(cond_ris) if cond_ris else None
            iqr_ri = iqr(cond_ris) if cond_ris else None
            if sign_reversal or (range_ri is not None and range_ri >= 30) or (iqr_ri is not None and iqr_ri >= 15):
                hetero = "HIGH"
            elif (range_ri is not None and range_ri >= 15) or (iqr_ri is not None and iqr_ri >= 7.5) or dmix:
                hetero = "MODERATE"
            else:
                hetero = "LOW"
            dir_value = direction(cwins, len(conds))
            seed_rate = sw / len(seed_all) if seed_all else 0.0
            cond_rate = cwins / len(conds) if conds else 0.0
            grade = grade_for(cond_rate, seed_rate, dpos, dneg, dir_value)
            allowed, prohibited = claim_text(grade, hetero)
            stat, wpos, wneg, pval, n_nonzero = wilcoxon_exact(cond_deltas)
            ci_low, ci_high = bootstrap_mean_ci(cond_deltas)
            stat_rows.append({"contrast": contrast, "metric": metric, "test_name": "Wilcoxon signed-rank (exact, two-sided; zero deltas dropped)",
                              "n": n_nonzero, "statistic": stat, "p_value": pval, "bootstrap_CI_low": ci_low,
                              "bootstrap_CI_high": ci_high, "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": BOOTSTRAP_SEED,
                              "w_plus": wpos, "w_minus": wneg})
            summary_rows.append({"contrast": contrast, "metric": metric, "treatment_arm": treatment, "reference_arm": reference,
                                 "scientific_question": scientific_question(contrast), "n_seed_pairs": len(seed_all),
                                 "seed_wins": sw, "seed_ties": st, "seed_losses": sl, "seed_win_rate_pct": seed_rate * 100,
                                 "n_conditions": len(conds), "condition_wins": cwins, "condition_ties": cties, "condition_losses": closses,
                                 "condition_win_rate_pct": cond_rate * 100, "n_datasets": len(ds_selected), "dataset_positive": dpos,
                                 "dataset_mixed": dmix, "dataset_negative": dneg, "mean_delta": sum(cond_deltas) / len(cond_deltas) if cond_deltas else None,
                                 "median_delta": statistics.median(cond_deltas) if cond_deltas else None,
                                 "mean_relative_improvement_pct": sum(cond_ris) / len(cond_ris) if cond_ris else None,
                                 "median_relative_improvement_pct": statistics.median(cond_ris) if cond_ris else None,
                                 "min_condition_relative_improvement_pct": min(cond_ris) if cond_ris else None,
                                 "max_condition_relative_improvement_pct": max(cond_ris) if cond_ris else None,
                                 "IQR_condition_relative_improvement_pct": iqr_ri, "direction": dir_value,
                                 "direction_consistency": consistency(cwins, len(conds)), "heterogeneity": hetero,
                                 "effect_magnitude_interpretation": effect_interpretation(statistics.median(cond_ris) if cond_ris else 0.0),
                                 "direction_strength": consistency(cwins, len(conds)), "effect_strength": effect_interpretation(statistics.median(cond_ris) if cond_ris else 0.0),
                                 "overall_evidence_grade": grade, "allowed_claim": allowed, "prohibited_claim": prohibited,
                                 "statistical_note": f"{stat_rows[-1]['test_name']}; n={n_nonzero}; p={pval:.6g}; bootstrap mean-delta 95% CI [{ci_low:.6g}, {ci_high:.6g}]"})
    return filtered, seed_rows, condition_rows, dataset_rows, horizon_rows, summary_rows, stat_rows, key_counts, conditions, paired_units


def fmt_num(value):
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def write_table(ws, headers, rows, formats, freeze=True, filter_row=0):
    for c, header in enumerate(headers):
        ws.write(0, c, header, formats["header"])
    for r, row in enumerate(rows, start=1):
        for c, header in enumerate(headers):
            value = row.get(header, "")
            if value is None:
                value = ""
            if isinstance(value, float) and not math.isfinite(value):
                value = ""
            if header.endswith("_pct") or "relative_improvement_pct" in header or header in {"IQR_condition_relative_improvement_pct", "bootstrap_CI_low", "bootstrap_CI_high"}:
                ws.write_number(r, c, float(value), formats["pct"] if value != "" else formats["text"])
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                ws.write_number(r, c, value, formats["number"])
            else:
                ws.write(r, c, value, formats["text"])
    if headers:
        ws.autofilter(0, 0, len(rows), len(headers) - 1)
        ws.freeze_panes(1, 0) if freeze else None
        for c, header in enumerate(headers):
            width = min(48, max(12, len(header) + 2))
            if header in {"scientific_question", "allowed_claim", "prohibited_claim", "statistical_note"}:
                width = 52
            ws.set_column(c, c, width)


def write_workbook(path: Path, source_hash: str, raw_records, fields, contract, analysis, qc):
    filtered, seed_rows, condition_rows, dataset_rows, horizon_rows, summary_rows, stat_rows, key_counts, conditions, paired_units = analysis
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(path))
    wb.set_properties({"title": "MS-KATL A-E Ablation Evidence Matrix", "comments": "Read-only derivation from the exact frozen workbook."})
    formats = {
        "header": wb.add_format({"bold": True, "font_name": "Arial", "font_size": 10, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "border": 1, "text_wrap": True, "valign": "vcenter"}),
        "text": wb.add_format({"font_name": "Arial", "font_size": 10, "valign": "top", "text_wrap": True}),
        "number": wb.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.000000", "valign": "top"}),
        "pct": wb.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.00%", "valign": "top"}),
        "title": wb.add_format({"bold": True, "font_name": "Arial", "font_size": 14, "font_color": "#1F4E78"}),
        "label": wb.add_format({"bold": True, "font_name": "Arial", "font_size": 10, "bg_color": "#D9EAF7", "border": 1}),
        "warn": wb.add_format({"font_name": "Arial", "font_size": 10, "font_color": "#9C0006", "bg_color": "#FFC7CE", "text_wrap": True}),
        "good": wb.add_format({"font_name": "Arial", "font_size": 10, "font_color": "#006100", "bg_color": "#C6EFCE"}),
        "formula": wb.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.000000", "font_color": "#000000"}),
        "formula_pct": wb.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.00%", "font_color": "#000000"}),
    }
    readme = wb.add_worksheet("00_README")
    readme.write(0, 0, "MS-KATL A-E Ablation Evidence Matrix", formats["title"])
    readme.set_column(0, 0, 34); readme.set_column(1, 1, 100)
    meta = [
        ("WORKBOOK_PURPOSE", "Standalone read-only evidence matrix derived from the exact frozen workbook."),
        ("SOURCE_WORKBOOK", str(INPUT_PATH)), ("SOURCE_WORKBOOK_SHA256", source_hash),
        ("OUTPUT_WORKBOOK", str(path)), ("GENERATION_TIMESTAMP", datetime.now(timezone.utc).isoformat()),
        ("READ_ONLY_ANALYSIS", "YES"), ("TRAINING_STARTED", "NO"), ("EXPERIMENT_STARTED", "NO"),
        ("SOURCE_WORKBOOK_MODIFIED", "NO"), ("METRIC_DIRECTION", "RMSE and sMAPE: LOWER_IS_BETTER"),
        ("DELTA_DEFINITION", "reference_metric - treatment_metric; positive means treatment wins"),
        ("RELATIVE_IMPROVEMENT_DEFINITION", "(reference_metric - treatment_metric) / reference_metric * 100"),
        ("PRIMARY_EVIDENCE_LEVEL", "CONDITION_LEVEL"), ("SEED_LEVEL_ROLE", "STABILITY_EVIDENCE"),
        ("DATASET_LEVEL_ROLE", "HETEROGENEITY_EVIDENCE"), ("EVIDENCE_MATRIX_STATUS", qc["status"]),
        ("HETEROGENEITY_RULE", "LOW: range<15pp and IQR<7.5pp; MODERATE: range>=15pp or IQR>=7.5pp; HIGH: sign reversal or range>=30pp or IQR>=15pp."),
        ("STATISTICS_RULE", f"Wilcoxon exact two-sided on condition-level deltas; bootstrap mean-delta percentile CI, reps={BOOTSTRAP_REPS}, seed={BOOTSTRAP_SEED}."),
        ("SOURCE_SCOPE_RULE", f"Only {DATA_DIR} was read for input data; no repository result files were used."),
    ]
    for r, (key, value) in enumerate(meta, start=2):
        readme.write(r, 0, key, formats["label"]); readme.write(r, 1, value, formats["text"])
    start = len(meta) + 4
    readme.write(start, 0, "ARM", formats["header"]); readme.write(start, 1, "SOURCE_CONTRACT_DEFINITION", formats["header"])
    arm_def = {"A": "TARGET_NATIVE_INIT", "B": "CONTROLLED_FROZEN_FUSED", "C": "FUSED_TRAINABLE", "D": "CURRENT_V3", "E": "BEST_SOURCE_TRAINABLE"}
    for i, arm in enumerate(EXPECTED_ARMS, start=start + 1):
        readme.write(i, 0, arm, formats["text"]); readme.write(i, 1, contract.get(arm, arm_def[arm]), formats["text"])
    summary_headers = ["contrast", "metric", "treatment_arm", "reference_arm", "scientific_question", "n_seed_pairs", "seed_wins", "seed_ties", "seed_losses", "seed_win_rate_pct", "n_conditions", "condition_wins", "condition_ties", "condition_losses", "condition_win_rate_pct", "n_datasets", "dataset_positive", "dataset_mixed", "dataset_negative", "mean_delta", "median_delta", "mean_relative_improvement_pct", "median_relative_improvement_pct", "min_condition_relative_improvement_pct", "max_condition_relative_improvement_pct", "IQR_condition_relative_improvement_pct", "direction", "direction_consistency", "heterogeneity", "effect_magnitude_interpretation", "direction_strength", "effect_strength", "overall_evidence_grade", "allowed_claim", "prohibited_claim", "statistical_note"]
    ws = wb.add_worksheet("01_Evidence_Summary")
    write_table(ws, summary_headers, summary_rows, formats)
    ws.conditional_format(1, summary_headers.index("overall_evidence_grade"), len(summary_rows), summary_headers.index("overall_evidence_grade"), {"type": "text", "criteria": "containing", "value": "STRONG", "format": formats["good"]})
    ws.conditional_format(1, summary_headers.index("overall_evidence_grade"), len(summary_rows), summary_headers.index("overall_evidence_grade"), {"type": "text", "criteria": "containing", "value": "CONFLICTING", "format": formats["warn"]})
    condition_headers = ["dataset", "target", "horizon", "contrast", "metric", "treatment_arm", "reference_arm", "n_seeds", "treatment_mean", "treatment_std", "reference_mean", "reference_std", "delta", "relative_improvement_pct", "winner", "direction", "seed_wins", "seed_ties", "seed_losses", "seed_win_rate_pct"]
    write_table(wb.add_worksheet("02_Condition_Effects"), condition_headers, condition_rows, formats)
    seed_headers = ["dataset", "target", "horizon", "seed", "contrast", "metric", "treatment_arm", "reference_arm", "treatment_value", "reference_value", "delta", "relative_improvement_pct", "winner"]
    write_table(wb.add_worksheet("03_Seed_Paired_Effects"), seed_headers, seed_rows, formats)
    dataset_headers = ["dataset", "contrast", "metric", "n_conditions", "condition_wins", "condition_ties", "condition_losses", "condition_win_rate_pct", "mean_delta", "median_delta", "mean_relative_improvement_pct", "median_relative_improvement_pct", "min_relative_improvement_pct", "max_relative_improvement_pct", "dataset_direction", "dataset_effect_interpretation"]
    write_table(wb.add_worksheet("04_Dataset_Heterogeneity"), dataset_headers, dataset_rows, formats)
    horizon_headers = ["horizon", "contrast", "metric", "n_dataset_conditions", "wins", "ties", "losses", "win_rate_pct", "mean_relative_improvement_pct", "median_relative_improvement_pct", "horizon_direction"]
    write_table(wb.add_worksheet("05_Horizon_Stability"), horizon_headers, horizon_rows, formats)
    stat_headers = ["contrast", "metric", "test_name", "n", "statistic", "p_value", "bootstrap_CI_low", "bootstrap_CI_high", "bootstrap_reps", "bootstrap_seed", "w_plus", "w_minus"]
    stat_ws = wb.add_worksheet("07_Statistical_Tests")
    write_table(stat_ws, stat_headers, stat_rows, formats)
    qc_ws = wb.add_worksheet("06_QC_Audit")
    qc_ws.write(0, 0, "QC_ITEM", formats["header"]); qc_ws.write(0, 1, "VALUE", formats["header"]); qc_ws.write(0, 2, "EXPECTED_OR_NOTE", formats["header"])
    for r, (key, value, note) in enumerate(qc["rows"], start=1):
        fmt = formats["good"] if str(value) in {"0", "YES", "OK", "180", "36", "8", "96", "32", "12"} else formats["warn"] if str(value) not in {"NO", "INCOMPLETE"} and ("MISSING" in key or "DUPLICATE" in key or "NON_OK" in key) else formats["text"]
        qc_ws.write(r, 0, key, formats["label"]); qc_ws.write(r, 1, value, fmt); qc_ws.write(r, 2, note, formats["text"])
    qc_ws.autofilter(0, 0, len(qc["rows"]), 2); qc_ws.freeze_panes(1, 0); qc_ws.set_column(0, 0, 32); qc_ws.set_column(1, 1, 22); qc_ws.set_column(2, 2, 72)
    wb.close()


def inspect(path: Path):
    raw = records_from_sheet(read_sheet(path, "05_Ablation_Raw"))
    contract_rows = read_sheet(path, "12_Ablation_Contract")
    print("INPUT", path)
    print("RAW_ROWS", len(raw))
    print("RAW_FIELDS", list(raw[0]) if raw else [])
    print("RAW_DATASETS", sorted({norm_upper(row.get("dataset", "")) for row in raw}))
    if raw:
        fields = resolve_raw_fields(raw)
        print("RESOLVED_FIELDS", fields)
        print("RAW_HEAD", raw[:3])
    print("CONTRACT_ROWS", len(contract_rows))
    print("CONTRACT_HEAD", contract_rows[:20])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    if args.input != INPUT_PATH:
        raise SystemExit("Input path must be the exact allowed frozen workbook")
    if not args.input.is_file():
        print("DATA_SCOPE_STATUS=INCOMPLETE")
        print("MISSING_REQUIRED_INPUT=MS_KATL_Paper_Data_Freeze_v1_2_FINAL_FROZEN.xlsx")
        print(f"SEARCHED_DIRECTORY={DATA_DIR}")
        print("EXTERNAL_DATA_SEARCHED=NO")
        return 2
    if args.inspect:
        inspect(args.input)
        return 0
    source_before = sha256(args.input)
    raw = records_from_sheet(read_sheet(args.input, "05_Ablation_Raw"))
    contract_rows = read_sheet(args.input, "12_Ablation_Contract")
    contract = parse_contract(contract_rows)
    if not raw:
        raise SystemExit("DATA_SCOPE_STATUS=INCOMPLETE; MISSING_REQUIRED_INPUT=05_Ablation_Raw rows")
    fields = resolve_raw_fields(raw)
    analysis = build_analysis(raw, fields, contract)
    filtered, seed_rows, condition_rows, dataset_rows, horizon_rows, summary_rows, stat_rows, key_counts, conditions, paired_units = analysis
    duplicate_key_count = sum(1 for v in key_counts.values() if v > 1)
    missing_arms = 0
    for unit in paired_units:
        present = {r["arm"] for r in filtered if tuple(r[x] for x in ["dataset", "target", "horizon", "seed"]) == unit}
        missing_arms += len(set(EXPECTED_ARMS) - present)
    invalid_space = sum(1 for r in filtered if r["metric_space"] and r["metric_space"] != "original_sales_space")
    non_ok = sum(1 for r in filtered if r["status"] and r["status"] != "OK")
    missing_rmse = sum(r["RMSE"] is None for r in filtered)
    missing_smape = sum(r["sMAPE"] is None for r in filtered)
    actual_conditions = len(conditions)
    status = "COMPLETE" if (len(filtered) == 180 and len(paired_units) == 36 and duplicate_key_count == 0 and missing_arms == 0 and non_ok == 0 and missing_rmse == 0 and missing_smape == 0 and invalid_space == 0 and len(summary_rows) == 8) else "INCOMPLETE"
    qc_rows = [
        ("SOURCE_ROWS", len(filtered), "Expected 180 paper-facing A-E rows after restricting to D1/D4/D5/D6"),
        ("EXPECTED_SOURCE_ROWS", 180, "4 datasets x 3 horizons x 3 seeds x 5 arms"),
        ("PAIRED_UNITS", len(paired_units), "Expected 36 dataset x target x horizon x seed units"),
        ("EXPECTED_PAIRED_UNITS", 36, "4 datasets x 3 horizons x 3 seeds"),
        ("MISSING_ARM_COUNT", missing_arms, "Must be 0"),
        ("DUPLICATE_KEY_COUNT", duplicate_key_count, "Must be 0"),
        ("NON_OK_STATUS_COUNT", non_ok, "Must be 0"),
        ("MISSING_RMSE_COUNT", missing_rmse, "Must be 0"),
        ("MISSING_SMAPE_COUNT", missing_smape, "Must be 0"),
        ("INVALID_METRIC_SPACE_COUNT", invalid_space, "Must be 0; expected original_sales_space"),
        ("CONTRAST_COUNT", len(CONTRASTS), "Expected 4"),
        ("EXPECTED_CONTRAST_COUNT", 4, "C vs A, C vs B, C vs E, D vs C"),
        ("SEED_EFFECT_ROWS", len(seed_rows), "Expected 288"),
        ("CONDITION_EFFECT_ROWS", len(condition_rows), "Expected 96"),
        ("DATASET_EFFECT_ROWS", len(dataset_rows), "Expected 32"),
        ("SOURCE_SHA256_BEFORE", source_before, "Exact frozen input"),
        ("SOURCE_SHA256_AFTER", sha256(args.input), "Must equal before"),
        ("SOURCE_WORKBOOK_UNCHANGED", "YES" if source_before == sha256(args.input) else "NO", "Must be YES"),
        ("EVIDENCE_MATRIX_STATUS", status, "COMPLETE only when all critical QC passes"),
    ]
    qc = {"rows": qc_rows, "status": status}
    write_workbook(args.output, source_before, raw, fields, contract, analysis, qc)
    print("SOURCE_ROWS", len(filtered))
    print("PAIRED_UNITS", len(paired_units))
    print("MISSING_ARM_COUNT", missing_arms)
    print("DUPLICATE_KEY_COUNT", duplicate_key_count)
    print("NON_OK_STATUS_COUNT", non_ok)
    print("CONTRASTS", len(CONTRASTS))
    print("METRICS", ",".join(METRICS))
    for row in summary_rows:
        print(f"{row['contrast']}_{row['metric']}_CONDITION_WTL={row['condition_wins']}/{row['condition_ties']}/{row['condition_losses']}")
        print(f"{row['contrast']}_{row['metric']}_GRADE={row['overall_evidence_grade']}")
    print("EVIDENCE_MATRIX_STATUS", status)
    print("OUTPUT_WORKBOOK", args.output)
    print("SHEETS", "00_README,01_Evidence_Summary,02_Condition_Effects,03_Seed_Paired_Effects,04_Dataset_Heterogeneity,05_Horizon_Stability,06_QC_Audit,07_Statistical_Tests")
    return 0 if status == "COMPLETE" else 3


if __name__ == "__main__":
    raise SystemExit(main())
