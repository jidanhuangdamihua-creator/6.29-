from __future__ import annotations

import pandas as pd

from .progress_tracker import format_seconds


def print_pipeline_header() -> None:
    print("=" * 80)
    print("                    统一实验评估流水线")
    print("               Multi-Source Transfer Learning Benchmark")
    print("=" * 80)


def print_dataset_header(dataset_name: str, target_shape: str, source_count: int) -> None:
    print("\n" + "=" * 70)
    print(f"评估 {dataset_name}")
    print("=" * 70)
    print(f"目标域: {target_shape}")
    print(f"源域数量: {source_count if source_count >= 0 else 'N/A'}")


def print_global_progress(current: int, total: int, dataset_name: str, method_name: str, eta_seconds: float) -> None:
    eta_text = format_seconds(eta_seconds)
    print(
        f"总体进度: [{current}/{total}]  当前数据集: {dataset_name}  当前方法: {method_name}  ETA: {eta_text}"
    )


def print_method_start(method_idx: int, method_total: int, method_name: str) -> None:
    print(f"[{method_idx}/{method_total}] 运行 {method_name} ...")


def _fmt_metric(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "nan"


def print_method_result(
    method_name: str,
    rmse: float,
    accuracy: float,
    smape: float | None = None,
    original_scale_smape: float | None = None,
) -> None:
    parts = [
        f"{method_name}",
        f"sMAPE: {_fmt_metric(smape)}",
        f"Original-scale sMAPE: {_fmt_metric(original_scale_smape)}",
        f"RMSE: {rmse:.6f}",
        f"Accuracy: {accuracy:.6f}",
    ]
    print("  " + "  ".join(parts))


def print_final_summary(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        print("\n结果为空，未生成汇总表。")
        return

    ok_df = results_df[results_df["error"].fillna("").astype(str).str.strip().eq("")].copy()
    if ok_df.empty:
        print("\n全部实验失败，未生成有效汇总。")
        return

    if "smape" not in ok_df.columns:
        print("\nNo sMAPE column found. Please rerun experiments after metric update.")
        return

    print("\nsMAPE 对比表")
    pivot = ok_df.pivot_table(index="dataset", columns="method", values="smape", aggfunc="min")
    print(pivot.to_string())

    if "No-TL" in ok_df["method"].unique():
        print("\n相对 No-TL 提升百分比（按 sMAPE，越高越好）")
        for dataset_name in sorted(ok_df["dataset"].dropna().unique().tolist()):
            ds = ok_df[ok_df["dataset"] == dataset_name]
            base = ds.loc[ds["method"] == "No-TL", "smape"]
            if base.empty:
                continue
            base_smape = float(base.min())
            parts = []
            for method_name in sorted(ds["method"].unique().tolist()):
                if method_name == "No-TL":
                    continue
                m_smape_series = ds.loc[ds["method"] == method_name, "smape"]
                if m_smape_series.empty:
                    continue
                m_smape = float(m_smape_series.min())
                improvement = (base_smape - m_smape) / base_smape * 100.0
                parts.append(f"{method_name}={improvement:.2f}%")
            if parts:
                print(f"{dataset_name}: " + ", ".join(parts))

    print("\n各数据集最佳方法")
    idx = ok_df.groupby("dataset")["smape"].idxmin()
    best_df = ok_df.loc[idx, ["dataset", "method", "smape", "original_scale_smape", "rmse", "accuracy", "prediction_shape"]]
    print(best_df.sort_values("dataset").to_string(index=False))


def print_completion() -> None:
    print("\n✅ 实验完成")
