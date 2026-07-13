"""
模块11：论文结果表与图表生成

职责：
1. 读取模块10输出的实验结果 CSV
2. 生成论文风格结果表
3. 生成 sMAPE/Accuracy 对比条形图
4. 保存表格与图表到输出目录
5. 生成简洁文本摘要
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.metric_contract import filter_formally_comparable_smape_rows

try:
    from src.utils.environment import setup_logging
except ImportError:
    setup_logging = None


LOGGER_NAME = "experiment"
REQUIRED_COLUMNS = ["method", "rmse", "accuracy", "prediction_shape"]
SMAPE_MISSING_MESSAGE = "No sMAPE column found. Please rerun experiments after metric update."


def _get_logger() -> logging.Logger:
    """获取统一日志器；若尚未初始化，则按默认参数初始化。"""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers and setup_logging is not None:
        setup_logging(log_level="INFO", log_file=None)
        logger = logging.getLogger(LOGGER_NAME)
    return logger


def load_results_csv(csv_path: str) -> pd.DataFrame:
    """
    读取实验结果 CSV 并校验必要列。

    Args:
        csv_path: 结果 CSV 路径。

    Returns:
        结果 DataFrame。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 缺失必要列。
    """
    logger = _get_logger()
    path = Path(csv_path)
    logger.info("[load_results_csv] Start. path=%s", path)

    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns in results CSV: "
            f"{missing}. Required={REQUIRED_COLUMNS}"
        )

    logger.info("[load_results_csv] Finished. rows=%d columns=%s", len(df), list(df.columns))
    return df


def _primary_smape_column(results_df: pd.DataFrame) -> Optional[str]:
    if "smape" in results_df.columns:
        return "smape"
    if "original_scale_smape" in results_df.columns:
        return "original_scale_smape"
    print(SMAPE_MISSING_MESSAGE)
    _get_logger().warning(SMAPE_MISSING_MESSAGE)
    return None


def filter_formally_comparable_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows eligible for formal original-sales sMAPE reporting."""
    eligible, _ = filter_formally_comparable_smape_rows(results_df)
    return eligible.reset_index(drop=True)


def sort_results_by_rmse(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    按 sMAPE 升序排序；保留旧函数名以兼容历史导入。

    Args:
        results_df: 原始结果表。

    Returns:
        排序后的新 DataFrame。
    """
    metric_col = _primary_smape_column(results_df)
    if metric_col is None:
        return results_df.copy().reset_index(drop=True)
    sorted_df = results_df.sort_values(by=metric_col, ascending=True).reset_index(drop=True)
    return sorted_df


def sort_results_by_method_order(
    results_df: pd.DataFrame,
    method_order: Sequence[str],
) -> pd.DataFrame:
    """按给定方法顺序排序，未在顺序中的方法排在末尾。"""
    out = results_df.copy()
    order_map = {m: i for i, m in enumerate(method_order)}
    out["_method_order"] = out["method"].astype(str).map(order_map).fillna(len(order_map)).astype(int)
    out = out.sort_values(by=["_method_order", "method"], ascending=[True, True]).drop(columns=["_method_order"])
    return out.reset_index(drop=True)


def add_rank_column(
    results_df: pd.DataFrame,
    metric_col: str = "rmse",
    ascending: bool = True,
) -> pd.DataFrame:
    """
    为结果表新增 rank 列。

    Args:
        results_df: 输入结果表。
        metric_col: 排名依据指标列。
        ascending: 是否升序排名。

    Returns:
        带 rank 列的新 DataFrame。
    """
    if metric_col not in results_df.columns:
        raise ValueError(f"metric_col '{metric_col}' not found in DataFrame columns: {list(results_df.columns)}")

    ranked_df = results_df.sort_values(by=metric_col, ascending=ascending).reset_index(drop=True).copy()
    ranked_df.insert(0, "rank", range(1, len(ranked_df) + 1))
    return ranked_df


def format_results_table(results_df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """
    生成论文展示风格的格式化结果表。

    Args:
        results_df: 输入结果表。
        decimals: 数值保留小数位。

    Returns:
        格式化后的 DataFrame，列顺序为 rank/method/rmse/accuracy/prediction_shape。
    """
    required = ["rank", "method", "rmse", "accuracy", "prediction_shape"]
    missing = [c for c in required if c not in results_df.columns]
    if missing:
        raise ValueError(f"Cannot format table, missing columns: {missing}")

    optional = [c for c in ["dataset", "include_sales_in_knn"] if c in results_df.columns]
    metric_cols = [c for c in ["smape", "original_scale_smape"] if c in results_df.columns]
    cols = ["rank"] + optional + metric_cols + [c for c in required if c != "rank"]
    out = results_df[cols].copy()
    for c in metric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(decimals)
    out["rmse"] = pd.to_numeric(out["rmse"], errors="coerce").round(decimals)
    out["accuracy"] = pd.to_numeric(out["accuracy"], errors="coerce").round(decimals)
    out["prediction_shape"] = out["prediction_shape"].astype(str)
    return out


def save_formatted_table(results_df: pd.DataFrame, output_path: str) -> None:
    """
    保存格式化结果表到 CSV，并自动创建目录。

    Args:
        results_df: 格式化后的结果表。
        output_path: 输出 CSV 路径。
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False, encoding="utf-8")


def plot_rmse_bar_chart(results_df: pd.DataFrame, output_path: str) -> None:
    """
    绘制并保存 sMAPE 对比条形图；保留旧函数名以兼容历史导入。

    Args:
        results_df: 输入结果表。
        output_path: 输出 PNG 路径。
    """
    plot_df = sort_results_by_rmse(results_df)
    metric_col = _primary_smape_column(plot_df)
    if metric_col is None:
        return
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["method"].astype(str), pd.to_numeric(plot_df[metric_col], errors="coerce"))
    plt.title("sMAPE Comparison")
    plt.xlabel("method")
    plt.ylabel(metric_col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_accuracy_bar_chart(results_df: pd.DataFrame, output_path: str) -> None:
    """
    绘制并保存 Accuracy 对比条形图。

    Args:
        results_df: 输入结果表。
        output_path: 输出 PNG 路径。
    """
    plot_df = sort_results_by_rmse(results_df)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["method"].astype(str), pd.to_numeric(plot_df["accuracy"], errors="coerce"))
    plt.title("Accuracy Comparison")
    plt.xlabel("method")
    plt.ylabel("accuracy")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_method_average_rank_bar_chart(average_rank_df: pd.DataFrame, output_path: str) -> None:
    """绘制并保存方法平均排名柱状图（越小越好）。"""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if average_rank_df.empty:
        plt.figure(figsize=(9, 5))
        plt.title("Method Average Rank (empty)")
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    df = average_rank_df.copy()
    if "method" not in df.columns or "average_rank" not in df.columns:
        raise ValueError("average_rank_df must contain columns: method, average_rank")

    df = df.sort_values(by="average_rank", ascending=True).reset_index(drop=True)

    plt.figure(figsize=(10, 5))
    plt.bar(df["method"].astype(str), pd.to_numeric(df["average_rank"], errors="coerce"))
    plt.title("Method Average Rank (Lower is Better)")
    plt.xlabel("method")
    plt.ylabel("average_rank")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_wilcoxon_significance_table(wilcoxon_df: pd.DataFrame, output_path: str) -> None:
    """保存 Wilcoxon 配对检验结果表。"""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wilcoxon_df.to_csv(out_path, index=False, encoding="utf-8")


def generate_results_summary(results_df: pd.DataFrame) -> str:
    """
    生成简洁文本摘要。

    Args:
        results_df: 输入结果表。

    Returns:
        摘要字符串。
    """
    if results_df.empty:
        return "Best method: N/A\nBest sMAPE: N/A\nBest Accuracy: N/A\nWorst method: N/A\nTotal methods: 0"

    metric_col = _primary_smape_column(results_df)
    if metric_col is None:
        return SMAPE_MISSING_MESSAGE
    sorted_df = sort_results_by_rmse(results_df)
    best_row = sorted_df.iloc[0]
    worst_row = sorted_df.iloc[-1]

    summary = (
        f"Best method: {best_row['method']}\n"
        f"Best sMAPE: {float(best_row[metric_col]):.4f}\n"
        f"Best Accuracy: {float(best_row['accuracy']):.4f}\n"
        f"Worst method: {worst_row['method']}\n"
        f"Total methods: {len(sorted_df)}"
    )
    return summary


def run_result_visualization(
    csv_path: str,
    output_dir: str = "outputs/results_reports",
    method_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    运行结果表与图表生成全流程。

    Args:
        csv_path: 模块10导出的结果 CSV 路径。
        output_dir: 输出目录。

    Returns:
        {
          "formatted_table_path": str,
          "rmse_plot_path": str,
          "accuracy_plot_path": str,
          "summary": str,
          "results_df": pd.DataFrame,
        }
    """
    logger = _get_logger()
    logger.info("[run_result_visualization] Start. csv_path=%s output_dir=%s", csv_path, output_dir)

    results_df = load_results_csv(csv_path)
    results_df = filter_formally_comparable_results(results_df)
    if "error" in results_df.columns:
        results_df = results_df[results_df["error"].fillna("").astype(str).str.strip().eq("")].copy()

    metric_col = _primary_smape_column(results_df)
    if metric_col is None:
        return {
            "formatted_table_path": "",
            "rmse_plot_path": "",
            "smape_plot_path": "",
            "accuracy_plot_path": "",
            "summary": SMAPE_MISSING_MESSAGE,
            "results_df": results_df,
        }

    rmse_rank_df = add_rank_column(sort_results_by_rmse(results_df), metric_col=metric_col, ascending=True)
    if method_order is not None:
        ordered_df = sort_results_by_method_order(rmse_rank_df, method_order=method_order)
    else:
        ordered_df = rmse_rank_df
    formatted_df = format_results_table(ordered_df, decimals=4)

    input_name = Path(csv_path).stem
    dataset_prefix = input_name.replace("_results", "")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    formatted_table_path = out_dir / f"{dataset_prefix}_results_formatted.csv"
    rmse_plot_path = out_dir / f"{dataset_prefix}_smape_bar.png"
    accuracy_plot_path = out_dir / f"{dataset_prefix}_accuracy_bar.png"

    save_formatted_table(formatted_df, str(formatted_table_path))
    plot_rmse_bar_chart(formatted_df, str(rmse_plot_path))
    plot_accuracy_bar_chart(formatted_df, str(accuracy_plot_path))
    summary = generate_results_summary(formatted_df)

    logger.info("[run_result_visualization] Finished.")

    return {
        "formatted_table_path": str(formatted_table_path),
        "rmse_plot_path": str(rmse_plot_path),
        "smape_plot_path": str(rmse_plot_path),
        "accuracy_plot_path": str(accuracy_plot_path),
        "summary": summary,
        "results_df": formatted_df,
    }
