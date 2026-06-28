"""One-click benchmark launcher.

This launcher prepares runtime environment, runs full paper experiments,
and prints a unified summary report.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from dataset_registry import get_dataset_display_name, list_dataset_names

REQ_FILE = ROOT / "requirements.txt"
VENV_DIR = Path.home() / ".msml_tl_env"
VENV_PYTHON = VENV_DIR / "bin" / "python"
MARKER_FILE = VENV_DIR / ".requirements_fingerprint"
FULL_RESULTS_CSV = ROOT / "outputs" / "experiment_results" / "full_paper_results.csv"
DATASET_PATHS_JSON = ROOT / "configs" / "dataset_paths.json"
DISPLAY_METHOD_MAPPING: List[Tuple[str, str, bool]] = [
    ("No-TL", "No-TL", False),
    ("SS-TL", "SS-TL", False),
    ("MSWA-TL", "MSWA-TL", False),
    ("MSSB-TL", "MSSB-TL", False),
    ("MSADW-TL", "MSML-TL", True),
]


def _run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def _capture(cmd: List[str], cwd: Optional[Path] = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.stdout.strip()


def _ensure_python_version() -> None:
    if sys.version_info < (3, 9):
        raise RuntimeError(f"Python version must be >= 3.9, current={sys.version.split()[0]}")


def _prepare_shared_venv() -> None:
    if VENV_PYTHON.exists():
        print(f"复用共享虚拟环境: {VENV_DIR}")
        return
    print(f"创建共享虚拟环境: {VENV_DIR}")
    _run([sys.executable, "-m", "venv", str(VENV_DIR)], cwd=ROOT)


def _check_python_architecture() -> None:
    system_arch = platform.machine()
    venv_arch = _capture([str(VENV_PYTHON), "-c", "import platform; print(platform.machine())"])
    print(f"系统架构: {system_arch}")
    print(f"虚拟环境 Python 架构: {venv_arch}")


def _requirements_fingerprint() -> str:
    req_text = REQ_FILE.read_text(encoding="utf-8")
    py_ver = _capture([str(VENV_PYTHON), "-c", "import sys; print(sys.version)"])
    payload = f"{req_text}\n---\n{py_ver}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _install_dependencies_if_needed() -> None:
    current_fp = _requirements_fingerprint()
    previous_fp = ""
    if MARKER_FILE.exists():
        previous_fp = MARKER_FILE.read_text(encoding="utf-8").strip()

    if previous_fp == current_fp:
        print("依赖已安装且版本匹配，跳过。")
        return

    _run([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"], cwd=ROOT)
    _run([str(VENV_PYTHON), "-m", "pip", "install", "-r", str(REQ_FILE)], cwd=ROOT)
    MARKER_FILE.write_text(current_fp, encoding="utf-8")


def _launch_experiment() -> None:
    _run([str(VENV_PYTHON), str(ROOT / "scripts" / "run_full_paper_experiments.py")], cwd=ROOT)


def _load_dataset_paths() -> Dict[str, str]:
    with DATASET_PATHS_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_shape_from_string(shape_text: str) -> str:
    text = str(shape_text)
    m = re.search(r"\((\d+)\s*,\s*(\d+)\)", text)
    if not m:
        return "N/A"
    return f"({m.group(1)}, {m.group(2)})"


def _get_domain_stats(dataset_name: str, data_path: str) -> Tuple[str, int]:
    try:
        from experiment_runner import prepare_base_data_for_experiments

        p = Path(data_path)
        if not p.is_absolute():
            p = ROOT / p

        bundle = prepare_base_data_for_experiments(dataset_name=dataset_name, data_path=str(p), config=None)
        target_df = bundle["target_df"]
        source_df = bundle["source_df"]
        source_count = int(len(source_df[["entity_id", "item_id"]].drop_duplicates()))
        return str(tuple(target_df.shape)), source_count
    except Exception:
        return "N/A", -1


SMAPE_MISSING_MESSAGE = "No sMAPE column found. Please rerun experiments after metric update."


def _primary_metric_col(df: pd.DataFrame) -> Optional[str]:
    if "smape" in df.columns:
        return "smape"
    if "original_scale_smape" in df.columns:
        return "original_scale_smape"
    print(SMAPE_MISSING_MESSAGE)
    return None


def _best_smape(success_df: pd.DataFrame, dataset: str, method: str) -> Optional[float]:
    metric_col = _primary_metric_col(success_df)
    if metric_col is None:
        return None
    sub = success_df[(success_df["dataset"] == dataset) & (success_df["method"] == method)]
    if sub.empty:
        return None
    return float(pd.to_numeric(sub[metric_col], errors="coerce").min())


def _method_display_rows(
    success_df: pd.DataFrame, dataset: str
) -> List[Tuple[str, str, bool, Optional[float]]]:
    rows: List[Tuple[str, str, bool, Optional[float]]] = []
    for display_name, internal_name, is_alias in DISPLAY_METHOD_MAPPING:
        rows.append((display_name, internal_name, is_alias, _best_smape(success_df, dataset, internal_name)))
    return rows


def _display_method_label(display_name: str, internal_name: str, is_alias: bool) -> str:
    if is_alias:
        return f"{display_name} (internal: {internal_name})"
    return display_name


def _na_reason_for_fixed_protocol(
    success_df: pd.DataFrame,
    dataset: str,
    internal_method: str,
    scenario: str,
    source_count: int,
) -> str:
    dataset_method = success_df[
        (success_df["dataset"] == dataset) & (success_df["method"] == internal_method)
    ]
    if dataset_method.empty:
        return "No successful run for this dataset+method in full_paper_results.csv"

    scenario_only = dataset_method[dataset_method["information_sharing"].astype(str) == str(scenario)]
    if scenario_only.empty:
        available = sorted(dataset_method["information_sharing"].astype(str).dropna().unique().tolist())
        return f"No row under scenario={scenario}. available_scenarios={available}"

    source_only = scenario_only[pd.to_numeric(scenario_only["source_count"], errors="coerce") == int(source_count)]
    if source_only.empty:
        available_k = sorted(
            pd.to_numeric(scenario_only["source_count"], errors="coerce").dropna().astype(int).unique().tolist()
        )
        return f"No row under source_count={source_count}. available_source_counts={available_k}"

    return "N/A"


def _build_fixed_protocol_summary(
    success_df: pd.DataFrame,
    datasets: List[str],
    scenario: str,
    source_count: int,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for dataset in datasets:
        for display_name, internal_name, is_alias in DISPLAY_METHOD_MAPPING:
            label = _display_method_label(display_name, internal_name, is_alias)
            sub = success_df[
                (success_df["dataset"] == dataset)
                & (success_df["method"] == internal_name)
                & (success_df["information_sharing"].astype(str) == str(scenario))
                & (pd.to_numeric(success_df["source_count"], errors="coerce") == int(source_count))
            ]

            if sub.empty:
                records.append(
                    {
                        "dataset": dataset,
                        "method": label,
                        "internal_method": internal_name,
                        "scenario": scenario,
                        "source_count": source_count,
                        "smape": "N/A",
                        "original_scale_smape": "N/A",
                        "rmse": "N/A",
                        "accuracy": "N/A",
                        "status": "N/A",
                        "na_reason": _na_reason_for_fixed_protocol(
                            success_df=success_df,
                            dataset=dataset,
                            internal_method=internal_name,
                            scenario=scenario,
                            source_count=source_count,
                        ),
                    }
                )
                continue

            metric_col = _primary_metric_col(sub)
            if metric_col is None:
                best_idx = sub.index[0]
            else:
                best_idx = pd.to_numeric(sub[metric_col], errors="coerce").idxmin()
            row = sub.loc[best_idx]
            records.append(
                {
                    "dataset": dataset,
                    "method": label,
                    "internal_method": internal_name,
                    "scenario": scenario,
                    "source_count": int(row["source_count"]),
                    "smape": float(row["smape"]) if "smape" in row else float("nan"),
                    "original_scale_smape": float(row["original_scale_smape"]) if "original_scale_smape" in row else float("nan"),
                    "rmse": float(row["rmse"]),
                    "accuracy": float(row["accuracy"]),
                    "status": "OK",
                    "na_reason": "",
                }
            )
    return pd.DataFrame(records)


def _print_unified_report() -> None:
    if not FULL_RESULTS_CSV.exists():
        print("未找到实验结果文件，无法生成统一报告。")
        return

    df = pd.read_csv(FULL_RESULTS_CSV)
    success_df = df[df["error"].fillna("").astype(str).str.strip().eq("")].copy()

    print("=" * 80)
    print("统一实验评估流水线")
    print("Multi-Source Transfer Learning Benchmark")
    print("=" * 80)
    print("方法名映射: MSADW-TL(展示别名) + MSML-TL(项目内部方法) 会同时展示")
    print(
        "SS-TL source 选择提示: 当前 SS-TL 使用固定排序后的第一个 source，"
        "不是多源方法中的相似源 top-k 机制。"
    )

    dataset_paths = _load_dataset_paths()
    datasets = list_dataset_names()

    smape_table_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    for dataset in datasets:
        print(f"\n{dataset} ({get_dataset_display_name(dataset)})")
        data_path = dataset_paths.get(dataset, "")
        target_shape, source_count = _get_domain_stats(dataset, data_path)
        print(f"目标域 shape: {target_shape}")
        if source_count >= 0:
            print(f"源域数量: {source_count}")
        else:
            print("源域数量: N/A")

        rows = _method_display_rows(success_df, dataset)
        for i, (display_name, internal_name, is_alias, smape) in enumerate(rows, start=1):
            if smape is None:
                if is_alias:
                    print(f"[{i}/5] {display_name}: sMAPE = N/A (project method: {internal_name})")
                else:
                    print(f"[{i}/5] {display_name}: sMAPE = N/A")
            else:
                if is_alias:
                    print(f"[{i}/5] {display_name}: sMAPE = {smape:.6f} (project method: {internal_name})")
                else:
                    print(f"[{i}/5] {display_name}: sMAPE = {smape:.6f}")
                smape_table_rows.append(
                    {
                        "dataset": dataset,
                        "method": display_name,
                        "smape": smape,
                    }
                )

        ds_success = success_df[success_df["dataset"] == dataset]
        if not ds_success.empty:
            metric_col = _primary_metric_col(ds_success)
            if metric_col is None:
                continue
            best_idx = pd.to_numeric(ds_success[metric_col], errors="coerce").idxmin()
            best_one = ds_success.loc[best_idx]
            best_rows.append(
                {
                    "dataset": dataset,
                    "method": str(best_one["method"]),
                    "smape": float(best_one[metric_col]),
                    "rmse": float(best_one["rmse"]),
                    "accuracy": float(best_one["accuracy"]),
                    "prediction_shape": _safe_shape_from_string(best_one["prediction_shape"]),
                }
            )

    print("\n📊 实验结果汇总")

    if smape_table_rows:
        smape_df = pd.DataFrame(smape_table_rows)
        pivot = smape_df.pivot_table(index="dataset", columns="method", values="smape", aggfunc="min")
        print("\nBest-over-all-configs summary")
        print("说明: 该摘要按 dataset+method 跨 scenario/source_count 取最小 sMAPE。")
        print("警告: 不可直接视为论文固定协议对比表。")
        print("\nsMAPE 对比表（best-over-all-configs）")
        print(pivot.to_string())

        print("\n提升百分比（相对 SS-TL，best-over-all-configs）")
        for dataset in list_dataset_names():
            ds = smape_df[smape_df["dataset"] == dataset]
            base = ds.loc[ds["method"] == "SS-TL", "smape"]
            if base.empty:
                print(f"{dataset} ({get_dataset_display_name(dataset)}): SS-TL 基线缺失")
                continue
            baseline = float(base.min())
            parts: List[str] = []
            for m in ["MSWA-TL", "MSSB-TL", "MSADW-TL"]:
                val = ds.loc[ds["method"] == m, "smape"]
                if val.empty:
                    parts.append(f"{m}=N/A")
                    continue
                improve = (baseline - float(val.min())) / baseline * 100.0
                parts.append(f"{m}={improve:.2f}%")
            print(f"{dataset} ({get_dataset_display_name(dataset)}): " + ", ".join(parts))

    if best_rows:
        best_df = pd.DataFrame(best_rows)
        print("\nBest-over-all-configs summary（每个数据集单条最优）")
        print("说明: 该摘要会跨 scenario/source_count 取最优，不能当作固定协议结果表。")
        print(best_df.to_string(index=False))

        global_idx = best_df["smape"].idxmin()
        global_best = best_df.loc[global_idx]
        print("\n全局最佳方法")
        print(
            f"dataset={global_best['dataset']}, method={global_best['method']}, "
            f"smape={float(global_best['smape']):.6f}, rmse={float(global_best['rmse']):.6f}, accuracy={float(global_best['accuracy']):.6f}, "
            f"prediction_shape={global_best['prediction_shape']}"
        )

    # Fixed protocol snapshot for auditability: do not cross scenario/source_count.
    fixed_protocol_scenario = "with_information_sharing"
    fixed_protocol_source_count = 3
    fixed_df = _build_fixed_protocol_summary(
        success_df=success_df,
        datasets=datasets,
        scenario=fixed_protocol_scenario,
        source_count=fixed_protocol_source_count,
    )
    print("\nFixed-protocol summary")
    print(
        "协议: scenario=with_information_sharing, source_count=3; "
        "每个 dataset+method 仅使用该协议下单条结果，不跨 scenario/k 取最优。"
    )
    print("如无结果将显示 N/A，并给出原因。")
    print(
        fixed_df[
            ["dataset", "method", "scenario", "source_count", "smape", "original_scale_smape", "rmse", "accuracy", "status", "na_reason"]
        ].to_string(index=False)
    )

    total = len(df)
    success_count = len(success_df)
    failed = total - success_count
    print("\n总体统计")
    print(f"total_runs={total}, success={success_count}, failed={failed}")


def main() -> None:
    print("[1/5] 检查系统架构...")
    _ensure_python_version()
    print(f"当前 Python: {sys.executable}")
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"系统架构: {platform.machine()}")

    print("[2/5] 准备共享虚拟环境（固定路径，避免项目搬家后重复下载）...")
    _prepare_shared_venv()

    print("[3/5] 校验 Python 架构...")
    _check_python_architecture()

    print("[4/5] 安装依赖（仅首次）...")
    _install_dependencies_if_needed()

    print("[5/5] 启动实验脚本...")
    _launch_experiment()

    _print_unified_report()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"启动失败: {type(exc).__name__}: {exc}")
        sys.exit(1)
