"""Read-only integrity checks for the ablation workbook and run artifacts.

A failed check is returned to the caller. This module does not edit the
workbook, run directories, or experimental configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

from analysis.formulas import original_sales_scores


WORKBOOK_PATH = Path("/Users/ming/Desktop/复现实验/论文文稿相关/原始实验结果/消融实验全部结果.xlsx")
SEEDS = (42, 43, 44, 45, 46)
DATASETS = ("D1", "D4", "D5", "D6")
HORIZONS = (1, 3, 5)
TEST_RANGES = {
    "D1": ("2017-07-01", "2017-12-27"),
    "D4": ("2025-01-15", "2025-07-13"),
    "D5": ("2017-02-16", "2017-08-14"),
    "D6": ("2015-11-25", "2016-05-22"),
}
CELL_KEYS = ["分支", "协议", "数据集", "预测步长", "种子"]
METRIC_ATOL = 1e-8
PREDICTION_ATOL = 1e-8
PLANNED_ARMS = (
    ("F1", "main"),
    ("F2", "main"),
    ("F3", "main"),
    ("F4", "main"),
    ("P1", "main"),
    ("P2", "main"),
    ("P3", "main"),
    ("F1", "alt_lr"),
    ("F2", "alt_lr"),
    ("F3", "alt_lr"),
    ("F4", "alt_lr"),
    ("F4", "random3"),
    ("F4", "bottom3"),
    ("F4", "k5"),
    ("F4", "k10"),
    ("F4", "ell15"),
    ("SITA_TCN", "main"),
    ("TCN_REVIN", "main"),
)
MISSING_NOT_RUN = ("F4", "k10")


@dataclass
class IntegrityReport:
    ok: bool
    workbook_path: Path
    runs_path: Path | None
    cell_count: int
    planned_cell_count: int
    missing_not_run: list[str] = field(default_factory=list)
    issues: list[dict[str, object]] = field(default_factory=list)
    notes: dict[str, object] = field(default_factory=dict)

    def add(self, kind: str, detail: str, **extra: object) -> None:
        row: dict[str, object] = {"kind": kind, "detail": detail}
        row.update(extra)
        self.issues.append(row)
        self.ok = False


def expected_test_dates(dataset: str) -> tuple[str, ...]:
    start, end = TEST_RANGES[dataset]
    return tuple(date.strftime("%Y-%m-%d") for date in pd.date_range(start, end, freq="D"))


def verify_workbook(workbook: Path = WORKBOOK_PATH) -> tuple[IntegrityReport, pd.DataFrame | None]:
    """Check the workbook against the frozen grid, formulas, curves, and runs."""
    report = IntegrityReport(
        ok=True,
        workbook_path=workbook,
        runs_path=None,
        cell_count=0,
        planned_cell_count=len(PLANNED_ARMS) * len(DATASETS) * len(HORIZONS) * len(SEEDS),
    )
    if not workbook.is_file():
        report.add("missing_workbook", f"未找到结果工作簿：{workbook}")
        return report, None
    sheets = pd.read_excel(
        workbook,
        sheet_name=["逐格", "逐日预测", "五种子均值", "训练曲线", "审计", "说明"],
    )
    cells = sheets["逐格"]
    daily = sheets["逐日预测"]
    means = sheets["五种子均值"]
    curves = sheets["训练曲线"]
    report.cell_count = int(len(cells))
    report.notes["说明"] = {str(row["项目"]): row["内容"] for _, row in sheets["说明"].iterrows()}
    _check_plan(report, cells)
    _check_cells(report, cells)
    _check_daily(report, cells, daily)
    _check_means(report, cells, means)
    _check_curves(report, cells, curves)
    _check_audit(report, sheets["审计"])
    runs_path = _runs_path(report)
    report.runs_path = runs_path
    if runs_path is not None:
        _check_runs(report, cells, daily, runs_path)
        _check_audit_files(report, sheets["审计"], runs_path)
    _check_official_metric_function(report, daily)
    return report, cells if report.ok else None


def _check_plan(report: IntegrityReport, cells: pd.DataFrame) -> None:
    observed = set(zip(cells["分支"].astype(str), cells["协议"].astype(str)))
    planned = set(PLANNED_ARMS)
    missing = sorted(planned - observed)
    extra = sorted(observed - planned)
    allowed = {MISSING_NOT_RUN}
    unexpected_missing = [item for item in missing if item not in allowed]
    if unexpected_missing:
        report.add("missing_arm", "计划中的配置缺失，且不是已注明未运行的 F4/k10", missing=unexpected_missing)
    if extra:
        report.add("unexpected_arm", "工作簿含有计划外的分支或协议", extra=extra)
    if MISSING_NOT_RUN not in missing:
        report.add("k10_present", "工作簿注明 F4/k10 未运行，但表中出现了该配置")
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                report.missing_not_run.append(f"F4:k10:{dataset}:h{horizon}:seed{seed}")
    if len(report.missing_not_run) != 60:
        report.add("k10_count", f"F4/k10 应记为 60 个未运行 cell，实际列出 {len(report.missing_not_run)} 个")
    note = str(report.notes.get("说明", {}).get("未包含", ""))
    if "F4/k10" not in note:
        report.add("k10_note", "说明页没有写明 F4/k10 未运行")


def _check_cells(report: IntegrityReport, cells: pd.DataFrame) -> None:
    duplicated = cells.duplicated(CELL_KEYS)
    if duplicated.any():
        report.add("duplicate_cell", "逐格结果有重复身份", count=int(duplicated.sum()))
    if not set(cells["种子"].tolist()) <= set(SEEDS):
        report.add("seed", "逐格种子超出 42–46", seeds=sorted(set(cells["种子"].tolist())))
    for (arm, protocol, dataset, horizon), block in cells.groupby(["分支", "协议", "数据集", "预测步长"], sort=False):
        seeds = sorted(int(seed) for seed in block["种子"])
        if seeds != list(SEEDS):
            report.add(
                "seed_coverage",
                "该条件没有完整的种子 42–46",
                arm=str(arm),
                protocol=str(protocol),
                dataset=str(dataset),
                horizon=int(horizon),
                seeds=seeds,
            )
    for column in ("测试RMSE", "测试sMAPE", "验证RMSE", "验证sMAPE"):
        values = cells[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            report.add("nonfinite_metric", f"逐格 {column} 含有非有限值", count=int((~np.isfinite(values)).sum()))
    if not (cells["测试天数"] == 180).all():
        report.add("test_days", "逐格测试天数不是 180")
    if not (cells["状态"].astype(str) == "COMPLETED").all():
        report.add("status", "逐格状态含有未完成记录", statuses=sorted(cells["状态"].astype(str).unique()))
    if not (cells["数值空间"].astype(str) == "original_sales").all():
        report.add("value_space", "逐格数值空间不是 original_sales")
    if cells["未评分测试日期"].notna().any():
        report.add("unscored", "逐格含有未评分测试日期")


def _check_daily(report: IntegrityReport, cells: pd.DataFrame, daily: pd.DataFrame) -> None:
    if daily.duplicated(CELL_KEYS + ["日期"]).any():
        report.add("duplicate_date", "逐日预测有重复日期")
    if not (daily["权重版本"].astype(str) == "EMA").all():
        report.add("weight_variant", "逐日预测的权重版本不是 EMA")
    true = daily["真实销量"].to_numpy(dtype=np.float64)
    pred = daily["预测销量"].to_numpy(dtype=np.float64)
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        report.add("nonfinite_prediction", "逐日预测含有非有限销量")
    expected = {dataset: set(expected_test_dates(dataset)) for dataset in DATASETS}
    for dataset, dates in expected.items():
        if len(dates) != 180:
            report.add("calendar", f"{dataset} 的冻结测试区间不是 180 天", count=len(dates))
    scored = []
    grouped = daily.groupby(CELL_KEYS, sort=False)
    sizes = grouped.size()
    cell_index = cells.set_index(CELL_KEYS)
    if set(sizes.index) != set(cell_index.index):
        report.add("daily_keys", "逐日预测的 cell 身份与逐格结果不能一一对应")
        return
    if len(sizes) == 0 or sizes.min() != 180 or sizes.max() != 180 or len(sizes) != len(cells):
        report.add(
            "daily_coverage",
            "逐日预测的 cell 数或每个 cell 的天数不是 180",
            groups=int(len(sizes)),
            min_rows=int(sizes.min()) if len(sizes) else 0,
            max_rows=int(sizes.max()) if len(sizes) else 0,
        )
        return
    worst_rmse = 0.0
    worst_smape = 0.0
    for key, block in grouped:
        dataset = str(key[2])
        observed = set(block["日期"].map(_iso_date))
        if observed != expected[dataset]:
            report.add(
                "date_set",
                "逐日预测日期与冻结测试区间不一致",
                arm=key[0],
                protocol=key[1],
                dataset=dataset,
                horizon=int(key[3]),
                seed=int(key[4]),
                missing=sorted(expected[dataset] - observed)[:5],
                extra=sorted(observed - expected[dataset])[:5],
            )
            break
        rmse, smape = original_sales_scores(block["真实销量"].to_numpy(), block["预测销量"].to_numpy())
        recorded = cell_index.loc[key]
        rmse_gap = abs(float(recorded["测试RMSE"]) - rmse)
        smape_gap = abs(float(recorded["测试sMAPE"]) - smape)
        worst_rmse = max(worst_rmse, rmse_gap)
        worst_smape = max(worst_smape, smape_gap)
        if rmse_gap > METRIC_ATOL or smape_gap > METRIC_ATOL:
            scored.append(
                {
                    "arm": key[0],
                    "protocol": key[1],
                    "dataset": dataset,
                    "horizon": int(key[3]),
                    "seed": int(key[4]),
                    "recorded_rmse": float(recorded["测试RMSE"]),
                    "recomputed_rmse": rmse,
                    "recorded_smape": float(recorded["测试sMAPE"]),
                    "recomputed_smape": smape,
                }
            )
    report.notes["max_abs_rmse_gap"] = worst_rmse
    report.notes["max_abs_smape_gap"] = worst_smape
    if scored:
        report.add("metric_recompute", "由逐日预测重算的 RMSE 或 sMAPE 与逐格记录不一致", examples=scored[:20], count=len(scored))


def _check_means(report: IntegrityReport, cells: pd.DataFrame, means: pd.DataFrame) -> None:
    if len(means) != len(cells) // len(SEEDS):
        report.add("mean_rows", "五种子均值行数不等于逐格行数除以 5", rows=int(len(means)))
    merged_rows = []
    for (arm, protocol, dataset, horizon), block in cells.groupby(CELL_KEYS[:4], sort=False):
        merged_rows.append(
            {
                "分支": arm,
                "协议": protocol,
                "数据集": dataset,
                "预测步长": horizon,
                "rmse_mean": float(block["测试RMSE"].mean()),
                "rmse_std": float(block["测试RMSE"].std(ddof=1)),
                "smape_mean": float(block["测试sMAPE"].mean()),
                "smape_std": float(block["测试sMAPE"].std(ddof=1)),
            }
        )
    recalculated = pd.DataFrame(merged_rows)
    compared = means.merge(recalculated, on=CELL_KEYS[:4], how="outer", indicator=True)
    if not (compared["_merge"] == "both").all():
        report.add("mean_keys", "五种子均值与逐格条件无法一一对应")
        return
    for recorded, fresh, label in (
        ("测试RMSE均值", "rmse_mean", "RMSE均值"),
        ("测试RMSE标准差", "rmse_std", "RMSE标准差"),
        ("测试sMAPE均值", "smape_mean", "sMAPE均值"),
        ("测试sMAPE标准差", "smape_std", "sMAPE标准差"),
    ):
        gap = (compared[recorded] - compared[fresh]).abs()
        if gap.max() > METRIC_ATOL:
            report.add("mean_value", f"五种子{label}与逐格重算不一致", max_abs=float(gap.max()))
    for seed in SEEDS:
        for metric, column in (("测试RMSE", f"种子{seed}_测试RMSE"), ("测试sMAPE", f"种子{seed}_测试sMAPE")):
            seed_cells = cells.loc[cells["种子"] == seed, CELL_KEYS[:4] + [metric]].rename(columns={metric: "value"})
            seed_means = means[CELL_KEYS[:4] + [column]].rename(columns={column: "recorded"})
            joined = seed_means.merge(seed_cells, on=CELL_KEYS[:4], how="outer", indicator=True)
            if not (joined["_merge"] == "both").all() or (joined["recorded"] - joined["value"]).abs().max() > METRIC_ATOL:
                report.add("seed_column", f"五种子表的种子 {seed} {metric} 与逐格不一致")
                return


def _check_curves(report: IntegrityReport, cells: pd.DataFrame, curves: pd.DataFrame) -> None:
    if curves.duplicated(CELL_KEYS + ["轮次"]).any():
        report.add("duplicate_epoch", "训练曲线有重复轮次")
    for column in ("训练损失", "验证损失", "验证RMSE", "验证sMAPE"):
        values = curves[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            report.add("nonfinite_curve", f"训练曲线 {column} 含有非有限值")
    sizes = curves.groupby(CELL_KEYS, sort=False).size()
    if len(sizes) != len(cells) or sizes.min() != 50 or sizes.max() != 50:
        report.add("curve_rows", "训练曲线不是每个 cell 恰好 50 行", groups=int(len(sizes)))
        return
    epochs = curves.groupby(CELL_KEYS, sort=False)["轮次"].agg(lambda values: tuple(sorted(int(value) for value in values)))
    if any(value != tuple(range(1, 51)) for value in epochs):
        report.add("epoch_index", "训练曲线轮次不是 1 到 50")
        return
    last = curves.loc[curves["轮次"] == 50, CELL_KEYS + ["验证RMSE", "验证sMAPE"]]
    compared = cells[CELL_KEYS + ["验证RMSE", "验证sMAPE"]].merge(last, on=CELL_KEYS, suffixes=("_cell", "_curve"))
    if len(compared) != len(cells):
        report.add("curve_join", "第 50 轮训练曲线无法与逐格对齐")
        return
    for metric in ("验证RMSE", "验证sMAPE"):
        gap = (compared[f"{metric}_cell"] - compared[f"{metric}_curve"]).abs()
        if gap.max() > METRIC_ATOL:
            report.add("curve_metric", f"逐格{metric}与第 50 轮训练曲线不一致", max_abs=float(gap.max()))


def _check_audit(report: IntegrityReport, audit: pd.DataFrame) -> None:
    required = {"failed", "hash_mismatch", "insufficient_candidates", "missing"}
    if not required <= set(audit.columns):
        report.add("audit_columns", "审计表缺少规定的分类列", columns=list(audit.columns))
        return
    for _, row in audit.iterrows():
        for column in required:
            if str(row[column]).strip() != "[]":
                report.add("audit_entry", "审计表含有非空异常分类", file=str(row["文件"]), column=column, value=str(row[column]))


def _check_audit_files(report: IntegrityReport, audit: pd.DataFrame, runs: Path) -> None:
    """Compare the workbook audit sheet with the JSON files beside the runs directory."""
    audit_dir = runs.parent / "audit"
    if not audit_dir.is_dir():
        report.add("audit_dir", "runs 同级目录没有 audit", path=str(audit_dir))
        return
    files = {}
    for path in sorted(audit_dir.glob("*.json")):
        files[path.name] = json.loads(path.read_text())
    sheet_names = [str(name) for name in audit["文件"].tolist()]
    report.notes["audit_files"] = sheet_names
    report.notes["phase_2_audit_present"] = "phase_2_audit.json" in files or "phase_2_audit.json" in sheet_names
    if sorted(sheet_names) != sorted(files) or len(sheet_names) != len(set(sheet_names)):
        report.add(
            "audit_file_list",
            "工作簿审计表列出的文件与 runs 同级 audit 目录不一致",
            sheet=sheet_names,
            disk=sorted(files),
        )
        return
    required = ("failed", "hash_mismatch", "insufficient_candidates", "missing")
    for name, payload in files.items():
        for key in required:
            if payload.get(key) != []:
                report.add("audit_json", "审计 JSON 的分类不是空列表", file=name, column=key, value=payload.get(key))


def _runs_path(report: IntegrityReport) -> Path | None:
    location = report.notes.get("说明", {}).get("记录位置")
    if not isinstance(location, str) or not location.strip():
        report.add("runs_note", "说明页没有给出 runs 记录位置")
        return None
    path = Path(location.strip())
    if not path.is_dir():
        report.add("runs_missing", "说明页给出的 runs 目录不存在", path=str(path))
        return None
    return path


def _check_runs(report: IntegrityReport, cells: pd.DataFrame, daily: pd.DataFrame, runs: Path) -> None:
    done_dirs = {path.parent.relative_to(runs).as_posix(): path.parent for path in runs.rglob("DONE.json")}
    expected_dirs = {
        f"{arm}/{protocol}/{dataset}_h{int(horizon)}/seed{int(seed)}"
        for arm, protocol, dataset, horizon, seed in cells[CELL_KEYS].itertuples(index=False, name=None)
    }
    missing = sorted(expected_dirs - set(done_dirs))
    extra = sorted(set(done_dirs) - expected_dirs)
    if missing:
        report.add("run_missing", "工作簿中的 cell 在 runs 中没有 DONE.json", examples=missing[:20], count=len(missing))
    if extra:
        report.add("run_extra", "runs 中有工作簿未记录的 DONE.json", examples=extra[:20], count=len(extra))
    if missing or extra:
        return
    grouped = {key: block for key, block in daily.groupby(CELL_KEYS, sort=False)}
    worst = 0.0
    examples = []
    rows = list(cells[CELL_KEYS + ["测试RMSE", "测试sMAPE"]].itertuples(index=False))
    for row_number, row in enumerate(rows, start=1):
        arm, protocol, dataset, horizon, seed, rmse, smape = row
        directory = done_dirs[f"{arm}/{protocol}/{dataset}_h{int(horizon)}/seed{int(seed)}"]
        prediction_path = directory / "predictions.parquet"
        record_path = directory / "cell_record.json"
        if not prediction_path.is_file() or not record_path.is_file():
            report.add("run_artifact", "runs cell 缺少预测文件或 cell_record.json", directory=str(directory))
            return
        frame = pd.read_parquet(prediction_path)
        workbook_rows = grouped[(arm, protocol, dataset, horizon, seed)].copy()
        frame = frame.copy()
        frame["日期"] = frame["label_date"].map(_iso_date)
        workbook_rows["日期"] = workbook_rows["日期"].map(_iso_date)
        frame = frame.sort_values("日期")
        workbook_rows = workbook_rows.sort_values("日期")
        if frame["日期"].tolist() != workbook_rows["日期"].tolist():
            report.add("run_dates", "runs 预测日期与工作簿逐日预测不一致", directory=str(directory))
            return
        true_gap = np.max(np.abs(frame["y_true_original"].to_numpy(dtype=np.float64) - workbook_rows["真实销量"].to_numpy(dtype=np.float64)))
        pred_gap = np.max(np.abs(frame["y_pred_original"].to_numpy(dtype=np.float64) - workbook_rows["预测销量"].to_numpy(dtype=np.float64)))
        worst = max(worst, float(true_gap), float(pred_gap))
        record = json.loads(record_path.read_text())
        record_rmse = float(record["pair"]["rmse"])
        record_smape = float(record["pair"]["smape"])
        if row_number % 200 == 0:
            print(f"已核对 runs {row_number}/{len(rows)}", file=sys.stderr)
        if (
            true_gap > PREDICTION_ATOL
            or pred_gap > PREDICTION_ATOL
            or abs(record_rmse - float(rmse)) > METRIC_ATOL
            or abs(record_smape - float(smape)) > METRIC_ATOL
            or str(record.get("status")) != "COMPLETED"
        ):
            examples.append(
                {
                    "directory": str(directory),
                    "true_gap": float(true_gap),
                    "pred_gap": float(pred_gap),
                    "record_rmse": record_rmse,
                    "workbook_rmse": float(rmse),
                    "record_smape": record_smape,
                    "workbook_smape": float(smape),
                }
            )
    report.notes["max_abs_run_prediction_gap"] = worst
    if examples:
        report.add("run_mismatch", "runs 预测或 cell_record 指标与工作簿不一致", examples=examples[:20], count=len(examples))


def _check_official_metric_function(report: IntegrityReport, daily: pd.DataFrame) -> None:
    """Stop if the local formula drifts from the frozen metric implementation."""
    try:
        from sita_ablation.metrics import evaluate_original_sales
    except Exception as exc:
        report.add("metric_import", "无法导入冻结指标函数 sita_ablation.metrics.evaluate_original_sales", error=str(exc))
        return
    worst = 0.0
    checked = 0
    for _key, block in daily.groupby(CELL_KEYS, sort=False):
        local_rmse, local_smape = original_sales_scores(block["真实销量"].to_numpy(), block["预测销量"].to_numpy())
        official = evaluate_original_sales(block["真实销量"].to_numpy(), block["预测销量"].to_numpy())
        worst = max(worst, abs(local_rmse - official.rmse), abs(local_smape - official.smape))
        checked += 1
        if worst > 0.0:
            report.add(
                "metric_formula",
                "分析用指标公式与 sita_ablation.metrics.evaluate_original_sales 不一致",
                local_rmse=local_rmse,
                official_rmse=official.rmse,
                local_smape=local_smape,
                official_smape=official.smape,
            )
            return
    report.notes["official_metric_cells_checked"] = checked
    report.notes["official_metric_max_abs_gap"] = worst


def _iso_date(value: object) -> str:
    text = str(value)
    return text[:10]
