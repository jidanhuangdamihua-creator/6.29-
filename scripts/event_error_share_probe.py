#!/usr/bin/env python3
"""Event-day squared-error share for the existing SITA point forecasts.

Reads the 225 saved predictions under
outputs/sita_all_targets_d4_d6_5seed and the sealed target parquets.
Does not train, and does not write anything except its own report.

Date alignment follows the audit already recorded in
outputs/source_error_borrowing_pilot/audit/audit_report.md:
row i has label date = test_dates[9 + horizon + i]. This script checks
that y_true_original matches the sealed sales on that date.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PRED_ROOT = REPO / "outputs/sita_all_targets_d4_d6_5seed/predictions/mode=without/v3"
OUT_DIR = REPO / "outputs/event_error_share_probe"
ATOL = 1e-4

DATASETS = {
    "D4": {
        "parquet": REPO / "数据集/固化数据/d1_d6_sealed_v1/dataset4/target.parquet",
        "train_start": "2024-12-16",
        "test_end": "2025-07-13",
        "learning_rate": "3e-5",
        "products": ["166_258", "166_311", "166_313", "166_432", "166_433"],
        "columns": ["activity_flag", "discount", "holiday_flag", "stock_hour6_22_cnt"],
    },
    "D5": {
        "parquet": REPO / "数据集/固化数据/d1_d6_sealed_v1/dataset5/target.parquet",
        "train_start": "2017-01-17",
        "test_end": "2017-08-15",
        "learning_rate": "3e-4",
        "products": ["48_1159414", "48_1159415", "48_1349808", "48_320682", "48_364606"],
        "columns": ["is_holiday", "onpromotion"],
    },
    "D6": {
        "parquet": REPO / "数据集/固化数据/d1_d6_sealed_v1/dataset6/target.parquet",
        "train_start": "2015-10-26",
        "test_end": "2016-05-22",
        "learning_rate": "3e-5",
        "products": [
            "CA_1_FOODS_3_080",
            "CA_1_FOODS_3_377",
            "CA_1_FOODS_3_555",
            "CA_1_FOODS_3_586",
            "CA_1_FOODS_3_668",
        ],
        "columns": ["is_event_1", "is_event_2", "snap", "sell_price"],
    },
}
HORIZONS = (1, 3, 5)
SEEDS = (42, 43, 44, 45, 46)


def test_dates(spec: dict) -> pd.DatetimeIndex:
    full = pd.date_range(spec["train_start"], spec["test_end"], freq="D")
    test = full[-180:]
    if len(test) != 180:
        raise SystemExit(f"test window is not 180 days: {len(test)}")
    return test


def event_frame(spec: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    frame = pd.read_parquet(spec["parquet"], columns=["entity_id", "date", "sales", *spec["columns"]])
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["entity_id"] = frame["entity_id"].astype(str)
    frame = frame[frame["entity_id"].isin(spec["products"]) & frame["date"].isin(set(dates))]
    if frame.duplicated(["entity_id", "date"]).any():
        raise SystemExit("duplicate product-date rows")
    return frame


def masks_for(dataset: str, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pre-specified event definitions. Thresholds were fixed from flag rates, before seeing errors."""
    if dataset == "D4":
        holiday = rows["holiday_flag"].to_numpy() == 1
        activity = rows["activity_flag"].to_numpy() == 1
        deep_discount = rows["discount"].to_numpy(dtype=float) < 0.9
        stockout = rows["stock_hour6_22_cnt"].to_numpy() == 0
        retail = holiday | activity | deep_discount
        return {
            "holiday": holiday,
            "activity": activity,
            "deep_discount": deep_discount,
            "retail_event": retail,
            "stock_hours_zero": stockout,
        }
    if dataset == "D5":
        holiday = rows["is_holiday"].to_numpy() == 1
        return {"holiday": holiday}
    event = rows["is_event_1"].to_numpy() == 1
    snap = rows["snap"].to_numpy() == 1
    return {"calendar_event": event, "snap": snap, "event_or_snap": event | snap}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    max_error = 0.0
    n_files = 0
    for dataset, spec in DATASETS.items():
        dates = test_dates(spec)
        history = pd.date_range(dates[0] - pd.Timedelta(days=30), dates[0] - pd.Timedelta(days=1), freq="D")
        raw = event_frame(spec, dates.append(history))
        for product in spec["products"]:
            product_rows = raw[raw["entity_id"] == product].set_index("date").sort_index()
            for horizon in HORIZONS:
                count = 180 - 10 - horizon + 1
                label = dates[9 + horizon: 9 + horizon + count]
                flags = masks_for(dataset, product_rows.loc[label])
                sales = product_rows.loc[label, "sales"].to_numpy(dtype=float)
                history_rows = product_rows.reindex(history)
                history_flags = masks_for(dataset, history_rows.fillna(0))
                for seed in SEEDS:
                    path = (
                        PRED_ROOT / dataset / product
                        / f"h{horizon}_s{seed}_{spec['learning_rate']}" / "predictions.csv"
                    )
                    pred = pd.read_csv(path)
                    n_files += 1
                    if len(pred) != count:
                        raise SystemExit(f"row count {len(pred)} != {count} at {path}")
                    y_true = pred["y_true_original"].to_numpy(dtype=float)
                    y_pred = pred["y_pred_original"].to_numpy(dtype=float)
                    error = float(np.max(np.abs(y_true - sales)))
                    max_error = max(max_error, error)
                    if error > ATOL:
                        raise SystemExit(f"date alignment failed at {path}: {error}")
                    sse = (y_true - y_pred) ** 2
                    total = float(sse.sum())
                    for name, mask in flags.items():
                        n_event = int(mask.sum())
                        event_sse = float(sse[mask].sum()) if n_event else 0.0
                        records.append({
                            "dataset": dataset,
                            "product": product,
                            "horizon": horizon,
                            "seed": seed,
                            "event": name,
                            "n_days": count,
                            "n_event_days": n_event,
                            "day_share": n_event / count,
                            "sse_share": event_sse / total if total else np.nan,
                            "sse_over_day": (event_sse / total) / (n_event / count) if total and n_event else np.nan,
                            "rmse_event": float(np.sqrt(sse[mask].mean())) if n_event else np.nan,
                            "rmse_other": float(np.sqrt(sse[~mask].mean())) if n_event < count else np.nan,
                            "history_event_days": int(history_flags[name].sum()),
                        })
    detail = pd.DataFrame(records)
    if n_files != 225:
        raise SystemExit(f"expected 225 files, read {n_files}")
    detail.to_csv(OUT_DIR / "cell_event_shares.csv", index=False)

    seed_mean = (
        detail.groupby(["dataset", "product", "horizon", "event"], as_index=False)
        [["day_share", "sse_share", "sse_over_day", "rmse_event", "rmse_other", "n_event_days", "history_event_days"]]
        .mean()
    )
    product_mean = (
        seed_mean.groupby(["dataset", "product", "event"], as_index=False)
        [["day_share", "sse_share", "sse_over_day", "rmse_event", "rmse_other", "n_event_days", "history_event_days"]]
        .mean()
    )
    product_mean.to_csv(OUT_DIR / "product_event_shares.csv", index=False)
    summary = (
        product_mean.groupby(["dataset", "event"], as_index=False)
        .agg(
            day_share=("day_share", "mean"),
            sse_share=("sse_share", "mean"),
            sse_over_day=("sse_over_day", "mean"),
            sse_over_day_min=("sse_over_day", "min"),
            sse_over_day_max=("sse_over_day", "max"),
            rmse_event=("rmse_event", "mean"),
            rmse_other=("rmse_other", "mean"),
            history_event_days=("history_event_days", "mean"),
            n_products=("product", "nunique"),
        )
    )
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    write_report(summary, product_mean, max_error, n_files)
    print(f"FILES={n_files} MAX_ALIGNMENT_ERROR={max_error:.3g}")
    print(summary.round(3).to_string(index=False))


def write_report(summary: pd.DataFrame, product_mean: pd.DataFrame, max_error: float, n_files: int) -> None:
    lines = [
        "# 事件日误差占比探测",
        "",
        "底层预测沿用已有的 225 组 SITA 点预测。学习率是在测试集上选出的，所以这些数字只用来判断“事件响应”这个方向有没有误差空间，不作为论文证据。",
        "",
        f"读取预测文件 {n_files} 个。标签日原始销量与 `y_true_original` 的最大绝对误差是 {max_error:.3g}。",
        "日期规则与已有审计一致：第 i 行的标签日是测试窗第 `9+h+i` 天。每个产品先对 5 个种子取平均，再对 3 个预测步长等权，最后对 5 个产品等权。",
        "",
        "## 事件定义（看误差之前定下的）",
        "",
        "- D4 假日：`holiday_flag=1`。活动：`activity_flag=1`。深折扣：`discount<0.9`。零售事件是这三者的并集。`discount<1` 覆盖了大部分天，不用作事件。",
        "- D4 另列 `stock_hour6_22_cnt=0`，只作为缺货诊断，不属于事件响应方向。",
        "- D5 假日：`is_holiday=1`。这批产品的 `onpromotion` 在测试窗里全部是 False，没有促销可分析。",
        "- D6 日历事件：`is_event_1=1`（`is_event_2` 全是 0）。SNAP：`snap=1`。测试窗里这 5 个产品的 `sell_price` 没有变化。",
        "- 历史对照是测试窗开始前的 30 天，用来看目标在适配期内见没见过这类事件。",
        "",
        "## 汇总",
        "",
        "天数占比是事件天占评价点的比例。误差占比是这些天的平方误差占全部平方误差的比例。倍数是误差占比除以天数占比，大于 1 表示事件天的单位误差更高。",
        "",
        "| 数据集 | 事件 | 天数占比 | 误差占比 | 倍数 | 倍数范围 | 事件天 RMSE | 其余天 RMSE | 前 30 天事件数 |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.event} | {row.day_share:.3f} | {row.sse_share:.3f} | "
            f"{row.sse_over_day:.2f} | {row.sse_over_day_min:.2f}–{row.sse_over_day_max:.2f} | "
            f"{row.rmse_event:.3f} | {row.rmse_other:.3f} | {row.history_event_days:.1f} |"
        )
    lines += ["", "## 逐产品倍数", ""]
    pivot = product_mean.pivot_table(index=["dataset", "product"], columns="event", values="sse_over_day")
    header = "| 数据集 | 产品 | " + " | ".join(pivot.columns) + " |"
    lines.append(header)
    lines.append("| --- | --- | " + " | ".join(["---:"] * len(pivot.columns)) + " |")
    for (dataset, product), row in pivot.round(2).iterrows():
        cells = " | ".join("" if pd.isna(value) else f"{value:.2f}" for value in row)
        lines.append(f"| {dataset} | {product} | {cells} |")
    lines += [
        "",
        "## 怎么读",
        "",
        "倍数接近 1，说明这类天并没有集中误差，迁移事件响应的空间小。倍数明显大于 1，而且前 30 天几乎没出现过，才符合“目标没见过、源产品见过”的方向。缺货诊断如果倍数高，说明误差里有一部分来自零库存，那是另一个问题。",
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "run_notes.json").write_text(json.dumps({
        "n_files": n_files,
        "max_alignment_error": max_error,
        "uses_test_selected_learning_rate": True,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
