"""Reproducible entry point for the 2026-10-03 ablation statistics.

Run from ``消融实验``::

    python -m analysis.stats

The command reads the workbook and run artifacts, writes integrity output,
and stops before inference if the recomputed metrics disagree with the records.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import traceback
import unittest

import numpy as np
import pandas as pd
import scipy

from analysis.comparisons import analyze_cells, descriptive_condition_table, validation_selection
from analysis.report import write_report
from analysis.verify import WORKBOOK_PATH, verify_workbook


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "20261003"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    test_result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromName("analysis.tests.test_stats")
    )
    if not test_result.wasSuccessful():
        print("统计单元测试未通过，停止读取实验结果。", file=sys.stderr)
        return 1
    log_lines = [
        f"started: {started}",
        f"python: {sys.version}",
        f"platform: {platform.platform()}",
        f"numpy: {np.__version__}",
        f"pandas: {pd.__version__}",
        f"scipy: {scipy.__version__}",
        f"workbook: {WORKBOOK_PATH}",
        f"command: {sys.executable} -m analysis.stats",
        f"unit_tests: analysis.tests.test_stats passed {test_result.testsRun}",
    ]
    try:
        digest = hashlib.sha256(WORKBOOK_PATH.read_bytes()).hexdigest()
        log_lines.append(f"workbook_sha256: {digest}")
        report, cells = verify_workbook(WORKBOOK_PATH)
        notes = dict(report.notes)
        notes["workbook_sha256"] = digest
        notes["runs_path"] = "" if report.runs_path is None else str(report.runs_path)
        _write_json(OUTPUT_DIR / "integrity_report.json", _integrity_payload(report, notes))
        if not report.ok or cells is None:
            log_lines.append(f"integrity: failed ({len(report.issues)} issues)")
            log_lines.append("inference: not run")
            _write_log(log_lines + [f"finished: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}"])
            print(f"完整性核对未通过，已写入 {OUTPUT_DIR / 'integrity_report.json'}", file=sys.stderr)
            return 2
        log_lines.append("integrity: passed")
        log_lines.append(f"cells: {report.cell_count}")
        log_lines.append(f"missing_f4_k10: {len(report.missing_not_run)}")
        summary, detail, dataset_summary, disagreement = analyze_cells(cells)
        descriptive = descriptive_condition_table(cells)
        validation = validation_selection(cells)
        descriptive.to_csv(OUTPUT_DIR / "descriptive_summary.csv", index=False)
        detail.to_csv(OUTPUT_DIR / "paired_ri_detail.csv", index=False)
        summary.to_csv(OUTPUT_DIR / "comparison_results.csv", index=False)
        dataset_summary.to_csv(OUTPUT_DIR / "dataset_ri_summary.csv", index=False)
        disagreement.to_csv(OUTPUT_DIR / "rmse_smape_disagreement.csv", index=False)
        validation.to_csv(OUTPUT_DIR / "validation_selection.csv", index=False)
        _write_family_slices(summary)
        write_report(
            OUTPUT_DIR / "统计分析报告.md",
            summary=summary,
            validation=validation,
            disagreement=disagreement,
            notes=notes,
            missing_k10=len(report.missing_not_run),
            cell_count=report.cell_count,
            planned_cell_count=report.planned_cell_count,
        )
        _write_workbook(summary, detail, descriptive, dataset_summary, disagreement, validation)
        log_lines.extend(
            [
                "inference: completed",
                "bootstrap_seed: 20261001",
                "bootstrap_resamples: 100000",
                "resample_unit: 12 dataset-by-horizon conditions",
                "holm: family A (5 contrasts) and family B (4 contrasts), separately for RMSE and sMAPE",
                f"outputs: {OUTPUT_DIR}",
            ]
        )
        print(f"统计分析已写入 {OUTPUT_DIR}")
        return 0
    except Exception:
        log_lines.append("status: error")
        log_lines.append(traceback.format_exc())
        print("统计分析中断，见 run_log.txt", file=sys.stderr)
        return 1
    finally:
        if log_lines[-1] != f"finished: held":
            log_lines.append(f"finished: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
        _write_log(log_lines)


def _write_family_slices(summary: pd.DataFrame) -> None:
    slices = {
        "table_G1_2x2.csv": summary["比较族"].isin(["A", "交互"]),
        "table_G2_param_usage.csv": summary["比较族"] == "B",
        "table_G5_tcn.csv": summary["比较族"] == "G5",
        "table_S1_lr.csv": summary["比较族"] == "S1",
        "table_S2_selection.csv": summary["比较族"] == "S2",
        "table_S3_K.csv": summary["比较族"] == "S3",
        "table_S4_window.csv": summary["比较族"] == "S4",
    }
    for name, mask in slices.items():
        summary.loc[mask].to_csv(OUTPUT_DIR / name, index=False)


def _write_workbook(*frames: pd.DataFrame) -> None:
    names = [
        "比较结果",
        "配对RI明细",
        "描述性汇总",
        "分数据集RI",
        "指标分歧",
        "验证选择",
    ]
    with pd.ExcelWriter(OUTPUT_DIR / "ablation_statistics_20261003.xlsx") as writer:
        for name, frame in zip(names, frames):
            frame.to_excel(writer, sheet_name=name, index=False)


def _integrity_payload(report, notes: dict[str, object]) -> dict[str, object]:
    return {
        "ok": report.ok,
        "workbook_path": str(report.workbook_path),
        "runs_path": None if report.runs_path is None else str(report.runs_path),
        "cell_count": report.cell_count,
        "planned_cell_count": report.planned_cell_count,
        "missing_not_run_count": len(report.missing_not_run),
        "missing_not_run": report.missing_not_run,
        "issues": report.issues,
        "notes": notes,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_log(lines: list[str]) -> None:
    (OUTPUT_DIR / "run_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
