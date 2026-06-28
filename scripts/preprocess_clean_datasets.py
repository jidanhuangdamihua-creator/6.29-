#!/usr/bin/env python3
"""Export D1-D6 clean tables following the preprocessing contract document.

The first executable gate is phase1: D1, D3, and D4. It validates the shared
date/entity_id/item_id/sales contract before high-cost D2/D5/D6 jobs run.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "preprocessing"
DEFAULT_PATHS = {
    "D1": ROOT / "数据集" / "Dataset1-Challenge.csv",
    "D2": ROOT / "数据集" / "Dataset2-pasta.csv",
    "D3": ROOT / "数据集" / "Dataset3-Rossmann.csv",
    "D4": ROOT / "数据集" / "原始数据" / "Dataset 4叮咚数据集" / "data" / "train.parquet",
    "D5": ROOT / "数据集" / "原始数据" / "Dataset 5Favorita",
    "D6": ROOT / "数据集" / "原始数据" / "Dataset 6m5-forecasting-accuracy",
}
QTY_RE = re.compile(r"^QTY_(B\d+)_(\d+)$")
PROMO_RE = re.compile(r"^PROMO_(B\d+)_(\d+)$")


def entity_coverage_summary_from_counts(counts: Iterable[int]) -> dict[str, object]:
    coverage = np.array(list(counts), dtype=np.int64)
    if coverage.size == 0:
        return {
            "min_entity_coverage_days": 0,
            "median_entity_coverage_days": 0.0,
            "mean_entity_coverage_days": 0.0,
            "max_entity_coverage_days": 0,
            "entities_ge_510_days": 0,
            "entity_coverage_ge_510_rate": 0.0,
        }
    ge_510 = coverage >= 510
    return {
        "min_entity_coverage_days": int(coverage.min()),
        "median_entity_coverage_days": float(np.median(coverage)),
        "mean_entity_coverage_days": float(coverage.mean()),
        "max_entity_coverage_days": int(coverage.max()),
        "entities_ge_510_days": int(ge_510.sum()),
        "entity_coverage_ge_510_rate": float(ge_510.mean()),
    }


def _with_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = date.dt.year
    out["month"] = date.dt.month
    out["week"] = date.dt.isocalendar().week.astype("int64")
    out["day"] = date.dt.day
    out["date"] = date.dt.strftime("%Y-%m-%d")
    return out


def _state_holiday_code(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    mapped = values.map({"0": 0, "a": 1, "b": 2, "c": 3})
    numeric = pd.to_numeric(values, errors="coerce")
    return mapped.fillna(numeric).fillna(0).astype("int64")


def clean_d1_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "store", "item", "sales"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"D1 missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"], errors="coerce"),
            "entity_id": pd.to_numeric(raw["store"], errors="coerce").astype("Int64"),
            "item_id": pd.to_numeric(raw["item"], errors="coerce").astype("Int64"),
            "sales": pd.to_numeric(raw["sales"], errors="coerce"),
        }
    )
    out = out.dropna().astype({"entity_id": "int64", "item_id": "int64"})
    out = _with_datetime_features(out)
    return out[["date", "entity_id", "item_id", "sales", "year", "month", "week", "day"]]


def _melt_d2_wide_columns(raw: pd.DataFrame, columns: list[str], value_name: str, pattern: re.Pattern[str]) -> pd.DataFrame:
    long = raw[["DATE"] + columns].melt(id_vars=["DATE"], var_name="wide_key", value_name=value_name)
    extracted = long["wide_key"].str.extract(pattern)
    long["entity_id"] = extracted[0]
    long["item_id"] = pd.to_numeric(extracted[1], errors="coerce")
    return long.drop(columns=["wide_key"])


def clean_d2_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    if "DATE" not in raw.columns:
        raise ValueError("D2 missing DATE column")
    qty_cols = [col for col in raw.columns if QTY_RE.match(str(col))]
    promo_cols = [col for col in raw.columns if PROMO_RE.match(str(col))]
    if not qty_cols:
        raise ValueError("D2 requires QTY_Bx_y columns")

    qty_long = _melt_d2_wide_columns(raw, qty_cols, "sales", QTY_RE)
    if promo_cols:
        promo_long = _melt_d2_wide_columns(raw, promo_cols, "promo", PROMO_RE)
        out = qty_long.merge(promo_long, on=["DATE", "entity_id", "item_id"], how="left")
    else:
        out = qty_long
        out["promo"] = 0

    out = out.rename(columns={"DATE": "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["item_id"] = out["item_id"].astype("Int64")
    out["sales"] = pd.to_numeric(out["sales"], errors="coerce")
    out["promo"] = pd.to_numeric(out["promo"], errors="coerce").fillna(0).astype("int64")
    out = out.dropna(subset=["date", "entity_id", "item_id", "sales"]).astype({"item_id": "int64"})
    out = _with_datetime_features(out)
    return out[["date", "entity_id", "item_id", "sales", "promo", "year", "month", "week", "day"]]


def clean_d3_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"Date", "Store", "Sales", "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"D3 missing required columns: {sorted(missing)}")

    store_id = pd.to_numeric(raw["Store"], errors="coerce").astype("Int64")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Date"], errors="coerce"),
            "entity_id": store_id,
            "item_id": 1,
            "store_id": store_id,
            "sales": pd.to_numeric(raw["Sales"], errors="coerce"),
            "customers_leakage_risk": pd.to_numeric(raw["Customers"], errors="coerce"),
            "open": pd.to_numeric(raw["Open"], errors="coerce"),
            "promo": pd.to_numeric(raw["Promo"], errors="coerce"),
            "state_holiday": _state_holiday_code(raw["StateHoliday"]),
            "school_holiday": pd.to_numeric(raw["SchoolHoliday"], errors="coerce"),
            "store_type": raw["StoreType"].astype("string") if "StoreType" in raw.columns else "unknown",
        }
    )
    out = out.dropna(subset=["date", "entity_id", "store_id", "sales"])
    out[["entity_id", "item_id", "store_id"]] = out[["entity_id", "item_id", "store_id"]].astype("int64")
    out = _with_datetime_features(out)
    return out[
        [
            "date",
            "entity_id",
            "item_id",
            "store_id",
            "sales",
            "customers_leakage_risk",
            "open",
            "promo",
            "state_holiday",
            "school_holiday",
            "store_type",
            "year",
            "month",
            "week",
            "day",
        ]
    ]


def _as_sequence(value: object) -> list[float]:
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Expected list-like value, got {type(value).__name__}")


def _list_stat(series: pd.Series, fn: Callable[[list[float]], float]) -> pd.Series:
    return series.map(lambda value: fn(_as_sequence(value)))


def clean_d4_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
        "dt",
        "sale_amount",
        "hours_sale",
        "stock_hour6_22_cnt",
        "hours_stock_status",
        "activity_flag",
        "discount",
        "holiday_flag",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"D4 missing required columns: {sorted(missing)}")

    out = raw.copy()
    out["date"] = pd.to_datetime(out["dt"], errors="coerce")
    out["entity_id"] = out["city_id"].astype("string") + "_" + out["store_id"].astype("string")
    out["item_id"] = pd.to_numeric(out["product_id"], errors="coerce").astype("Int64")
    out["sales"] = pd.to_numeric(out["sale_amount"], errors="coerce")
    out["stock_hour_6_22_cnt"] = pd.to_numeric(out["stock_hour6_22_cnt"], errors="coerce")
    out["activity_flag"] = out["activity_flag"].astype("int64")
    out["holiday_flag"] = out["holiday_flag"].astype("int64")

    out["hours_sale_sum_leakage_risk"] = _list_stat(out["hours_sale"], sum)
    out["hours_sale_max_leakage_risk"] = _list_stat(out["hours_sale"], max)
    out["hours_sale_nonzero_hours_leakage_risk"] = _list_stat(
        out["hours_sale"], lambda seq: sum(1 for value in seq if float(value) > 0)
    )
    out["hours_stock_sum_leakage_risk"] = _list_stat(out["hours_stock_status"], sum)
    out["hours_stock_max_leakage_risk"] = _list_stat(out["hours_stock_status"], max)
    out["hours_stock_nonzero_hours_leakage_risk"] = _list_stat(
        out["hours_stock_status"], lambda seq: sum(1 for value in seq if float(value) > 0)
    )

    out = _with_datetime_features(out)
    columns = [
        "date",
        "entity_id",
        "item_id",
        "city_id",
        "store_id",
        "product_id",
        "sales",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "stock_hour_6_22_cnt",
        "activity_flag",
        "discount",
        "holiday_flag",
        "precpt",
        "avg_temperature",
        "avg_humidity",
        "avg_wind_level",
        "hours_sale_sum_leakage_risk",
        "hours_sale_max_leakage_risk",
        "hours_sale_nonzero_hours_leakage_risk",
        "hours_stock_sum_leakage_risk",
        "hours_stock_max_leakage_risk",
        "hours_stock_nonzero_hours_leakage_risk",
        "year",
        "month",
        "week",
        "day",
    ]
    return out[columns].dropna(subset=["date", "entity_id", "item_id", "sales"])


def preprocess_d5_oil(oil_raw: pd.DataFrame) -> pd.DataFrame:
    oil = oil_raw.copy()
    oil["date"] = pd.to_datetime(oil["date"], errors="coerce")
    oil = oil.sort_values("date").set_index("date")
    oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce")
    oil = oil.resample("D").asfreq()
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill()
    oil["oil_price"] = oil["dcoilwtico"].shift(1)
    oil = oil.reset_index()
    oil["date"] = oil["date"].dt.strftime("%Y-%m-%d")
    return oil[["date", "oil_price"]]


def _d5_effective_holiday_mask(holidays: pd.DataFrame) -> pd.Series:
    holiday_type = holidays["type"].astype("string")
    transferred = holidays["transferred"].astype(bool)
    return (
        holiday_type.eq("Transfer")
        | (holiday_type.isin(["Holiday", "Additional", "Bridge"]) & ~transferred)
    ) & ~holiday_type.eq("Work Day")


def preprocess_d5_holidays(holidays_raw: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    holidays = holidays_raw.copy()
    holidays["date"] = pd.to_datetime(holidays["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    holidays["is_effective_holiday"] = _d5_effective_holiday_mask(holidays).astype("int64")
    holidays = holidays[holidays["is_effective_holiday"].eq(1)].copy()

    rows: list[pd.DataFrame] = []
    national = holidays[holidays["locale"].eq("National")]
    if not national.empty:
        rows.append(
            stores[["store_nbr"]]
            .merge(national[["date", "is_effective_holiday"]], how="cross")
            .rename(columns={"is_effective_holiday": "is_holiday"})
        )
    regional = holidays[holidays["locale"].eq("Regional")]
    if not regional.empty:
        rows.append(
            stores[["store_nbr", "state"]].merge(
                regional[["date", "locale_name", "is_effective_holiday"]],
                left_on="state",
                right_on="locale_name",
                how="inner",
            )[["store_nbr", "date", "is_effective_holiday"]].rename(columns={"is_effective_holiday": "is_holiday"})
        )
    local = holidays[holidays["locale"].eq("Local")]
    if not local.empty:
        rows.append(
            stores[["store_nbr", "city"]].merge(
                local[["date", "locale_name", "is_effective_holiday"]],
                left_on="city",
                right_on="locale_name",
                how="inner",
            )[["store_nbr", "date", "is_effective_holiday"]].rename(columns={"is_effective_holiday": "is_holiday"})
        )

    if not rows:
        return pd.DataFrame(columns=["store_nbr", "date", "is_holiday"])
    out = pd.concat(rows, ignore_index=True)
    return out.groupby(["store_nbr", "date"], as_index=False)["is_holiday"].max()


def clean_d5_store_dataframe(
    store_train: pd.DataFrame,
    *,
    items: pd.DataFrame,
    store_row: pd.Series,
    oil: pd.DataFrame,
    transactions: pd.DataFrame,
    holidays_by_store: pd.DataFrame,
    global_end_date: pd.Timestamp,
) -> pd.DataFrame:
    observed = store_train.copy()
    observed["date"] = pd.to_datetime(observed["date"], errors="coerce")
    observed["store_nbr"] = pd.to_numeric(observed["store_nbr"], errors="coerce").astype("int64")
    observed["item_nbr"] = pd.to_numeric(observed["item_nbr"], errors="coerce").astype("int64")
    observed["unit_sales"] = pd.to_numeric(observed["unit_sales"], errors="coerce").clip(lower=0)
    observed["onpromotion"] = pd.to_numeric(observed["onpromotion"], errors="coerce").fillna(0).astype("int64")

    starts = observed.groupby(["store_nbr", "item_nbr"], as_index=False)["date"].min()
    pieces = []
    for row in starts.itertuples(index=False):
        dates = pd.date_range(row.date, global_end_date, freq="D")
        pieces.append(pd.DataFrame({"store_nbr": row.store_nbr, "item_nbr": row.item_nbr, "date": dates}))
    full = pd.concat(pieces, ignore_index=True)
    out = full.merge(observed, on=["store_nbr", "item_nbr", "date"], how="left")
    out["sales"] = out["unit_sales"].fillna(0)
    out["onpromotion"] = out["onpromotion"].fillna(0).astype("int64")

    out = out.merge(items, on="item_nbr", how="left")
    for col in ["city", "state", "type", "cluster"]:
        out[col] = store_row[col]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out.merge(oil, on="date", how="left")
    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.merge(tx[["date", "store_nbr", "transactions"]], on=["date", "store_nbr"], how="left")
    out = out.merge(holidays_by_store, on=["store_nbr", "date"], how="left")
    out["transactions"] = out["transactions"].fillna(0).astype("int64")
    out["is_holiday"] = out["is_holiday"].fillna(0).astype("int64")
    out["entity_id"] = out["store_nbr"].astype("int64")
    out["item_id"] = out["item_nbr"].astype("int64")
    out = _with_datetime_features(out)
    columns = [
        "date",
        "entity_id",
        "item_id",
        "store_nbr",
        "item_nbr",
        "sales",
        "onpromotion",
        "family",
        "class",
        "perishable",
        "city",
        "state",
        "type",
        "cluster",
        "transactions",
        "oil_price",
        "is_holiday",
        "year",
        "month",
        "week",
        "day",
    ]
    return out[columns]


def clean_d6_chunk(sales_chunk: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]
    missing = set(id_cols).difference(sales_chunk.columns)
    if missing:
        raise ValueError(f"D6 sales chunk missing required columns: {sorted(missing)}")
    value_vars = [col for col in sales_chunk.columns if str(col).startswith("d_")]
    if not value_vars:
        raise ValueError("D6 sales chunk has no d_* columns")

    melted = sales_chunk.melt(id_vars=id_cols, value_vars=value_vars, var_name="d", value_name="sales")
    cal_cols = [
        "d",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]
    out = melted.merge(calendar[cal_cols], on="d", how="left")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["entity_id"] = out["store_id"]
    out["sales"] = pd.to_numeric(out["sales"], errors="coerce").astype("int64")
    out["is_event_1"] = out["event_name_1"].notna().astype("int64")
    out["is_event_2"] = out["event_name_2"].notna().astype("int64")
    out["event_type_1"] = out["event_type_1"].fillna("None")
    out["snap"] = np.select(
        [out["state_id"].eq("CA"), out["state_id"].eq("TX"), out["state_id"].eq("WI")],
        [out["snap_CA"], out["snap_TX"], out["snap_WI"]],
        default=0,
    ).astype("int64")

    out = out.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    out["price_available"] = out["sell_price"].notna().astype("int64")
    out["week"] = out["date"].dt.isocalendar().week.astype("int64")
    out["day"] = out["date"].dt.day.astype("int64")
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    columns = [
        "date",
        "entity_id",
        "item_id",
        "store_id",
        "dept_id",
        "cat_id",
        "state_id",
        "sales",
        "sell_price",
        "price_available",
        "weekday",
        "wday",
        "is_event_1",
        "is_event_2",
        "event_type_1",
        "snap",
        "year",
        "month",
        "week",
        "day",
    ]
    return out[columns]


def validate_clean_frame(dataset: str, df: pd.DataFrame, *, full_dataset: bool = False) -> dict[str, object]:
    required = {"date", "entity_id", "item_id", "sales", "year", "month", "week", "day"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{dataset} missing contract columns: {missing}")
    if df.empty:
        raise ValueError(f"{dataset} clean frame is empty")
    if df[["date", "entity_id", "item_id", "sales"]].isna().any().any():
        raise ValueError(f"{dataset} has NA in contract columns")
    if pd.to_numeric(df["sales"], errors="coerce").min() < 0:
        raise ValueError(f"{dataset} has negative sales")

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError(f"{dataset} has invalid dates")

    report: dict[str, object] = {
        "dataset": dataset,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "entity_item_groups": int(df.groupby(["entity_id", "item_id"]).ngroups),
        "date_min": str(parsed_dates.min().date()),
        "date_max": str(parsed_dates.max().date()),
        "sales_min": float(pd.to_numeric(df["sales"], errors="coerce").min()),
    }

    if full_dataset:
        expected_groups = {"D1": 500, "D2": 118}.get(dataset)
        if expected_groups is not None and report["entity_item_groups"] != expected_groups:
            raise ValueError(f"{dataset} expected {expected_groups} entity-item groups, got {report['entity_item_groups']}")
    if dataset == "D2":
        if "promo" not in df.columns:
            raise ValueError("D2 missing promo column")
        promo_values = set(pd.to_numeric(df["promo"], errors="coerce").dropna().astype(int).unique().tolist())
        if not promo_values.issubset({0, 1}):
            raise ValueError(f"D2 promo must be binary, got {sorted(promo_values)}")
        if full_dataset:
            counts = df.groupby("entity_id")["item_id"].nunique().to_dict()
            expected_counts = {"B1": 42, "B2": 45, "B3": 21, "B4": 10}
            if counts != expected_counts:
                raise ValueError(f"D2 expected item counts {expected_counts}, got {counts}")
    if dataset == "D3":
        if not (df["entity_id"].astype("int64") == df["store_id"].astype("int64")).all():
            raise ValueError("D3 entity_id must equal store_id")
        if df["item_id"].nunique() != 1 or int(df["item_id"].iloc[0]) != 1:
            raise ValueError("D3 item_id must be constant 1")
        if "customers_leakage_risk" not in df.columns or "customers" in df.columns:
            raise ValueError("D3 customers must be exposed only as customers_leakage_risk")
    if dataset == "D4":
        leakage_cols = [col for col in df.columns if col.endswith("_leakage_risk")]
        if len(leakage_cols) != 6:
            raise ValueError(f"D4 expected 6 leakage risk columns, got {leakage_cols}")
        if full_dataset and df["city_id"].nunique() != 18:
            raise ValueError(f"D4 expected 18 city_id values, got {df['city_id'].nunique()}")
        report["city_count"] = int(df["city_id"].nunique())
        if len(df) > 1 and df["hours_sale_sum_leakage_risk"].nunique() > 1 and df["sales"].nunique() > 1:
            report["d4_hours_sale_corr"] = float(df["hours_sale_sum_leakage_risk"].corr(df["sales"]))
    if dataset == "D6":
        if "price_available" not in df.columns:
            raise ValueError("D6 missing price_available column")
        available = pd.to_numeric(df["price_available"], errors="coerce")
        if not set(available.dropna().astype(int).unique().tolist()).issubset({0, 1}):
            raise ValueError("D6 price_available must be binary")
        if df.loc[available.eq(1), "sell_price"].isna().any():
            raise ValueError("D6 price_available == 1 rows must have sell_price")
        if df.loc[df["sell_price"].isna(), "price_available"].ne(0).any():
            raise ValueError("D6 sell_price NaN must only appear when price_available == 0")
    if dataset == "D5":
        if "onpromotion" not in df.columns or "is_holiday" not in df.columns:
            raise ValueError("D5 missing onpromotion or is_holiday")
        for col in ["onpromotion", "is_holiday"]:
            values = set(pd.to_numeric(df[col], errors="coerce").dropna().astype(int).unique().tolist())
            if not values.issubset({0, 1}):
                raise ValueError(f"D5 {col} must be binary, got {sorted(values)}")
    return report


def validate_phase1_clean_frame(dataset: str, df: pd.DataFrame, *, full_dataset: bool = False) -> dict[str, object]:
    return validate_clean_frame(dataset, df, full_dataset=full_dataset)


def _read_dataset(dataset: str, path: Path) -> pd.DataFrame:
    if dataset in {"D1", "D2", "D3"}:
        return pd.read_csv(path, low_memory=False)
    if dataset == "D4":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset for phase1: {dataset}")


def clean_dataset(dataset: str, raw: pd.DataFrame) -> pd.DataFrame:
    if dataset == "D1":
        return clean_d1_dataframe(raw)
    if dataset == "D2":
        return clean_d2_dataframe(raw)
    if dataset == "D3":
        return clean_d3_dataframe(raw)
    if dataset == "D4":
        return clean_d4_dataframe(raw)
    raise ValueError(f"Unsupported dataset for phase1: {dataset}")


def export_dataset(dataset: str, input_path: Path, output_dir: Path) -> dict[str, object]:
    if dataset == "D6":
        return export_d6_dataset(input_path, output_dir)
    if dataset == "D5":
        return export_d5_dataset(input_path, output_dir)
    raw = _read_dataset(dataset, input_path)
    clean = clean_dataset(dataset, raw)
    report = validate_clean_frame(dataset, clean, full_dataset=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}_clean.csv"
    clean.to_csv(out_path, index=False)
    report["output_path"] = str(out_path)
    return report


def export_d6_dataset(input_dir: Path, output_dir: Path, chunksize: int = 1000) -> dict[str, object]:
    sales_path = input_dir / "sales_train_evaluation.csv"
    calendar_path = input_dir / "calendar.csv"
    prices_path = input_dir / "sell_prices.csv"
    for path in [sales_path, calendar_path, prices_path]:
        if not path.exists():
            raise FileNotFoundError(f"D6 required file not found: {path}")

    calendar = pd.read_csv(calendar_path)
    prices = pd.read_csv(prices_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "D6_clean.csv"
    if out_path.exists():
        out_path.unlink()

    rows = 0
    groups: set[tuple[str, str]] = set()
    sales_min: float | None = None
    date_min: str | None = None
    date_max: str | None = None
    first = True

    for chunk in pd.read_csv(sales_path, chunksize=chunksize, low_memory=False):
        clean = clean_d6_chunk(chunk, calendar, prices)
        validate_clean_frame("D6", clean)
        clean.to_csv(out_path, index=False, mode="w" if first else "a", header=first)
        first = False
        rows += len(clean)
        groups.update(zip(clean["item_id"].astype(str), clean["store_id"].astype(str)))
        chunk_sales_min = float(clean["sales"].min())
        sales_min = chunk_sales_min if sales_min is None else min(sales_min, chunk_sales_min)
        chunk_date_min = str(clean["date"].min())
        chunk_date_max = str(clean["date"].max())
        date_min = chunk_date_min if date_min is None else min(date_min, chunk_date_min)
        date_max = chunk_date_max if date_max is None else max(date_max, chunk_date_max)

    report = {
        "dataset": "D6",
        "rows": int(rows),
        "columns": 20,
        "entity_item_groups": int(len(groups)),
        "date_min": date_min,
        "date_max": date_max,
        "sales_min": float(sales_min if sales_min is not None else np.nan),
        "output_path": str(out_path),
    }
    if report["rows"] != 59181090:
        raise ValueError(f"D6 expected 59,181,090 rows, got {report['rows']}")
    if report["entity_item_groups"] != 30490:
        raise ValueError(f"D6 expected 30,490 entity-item groups, got {report['entity_item_groups']}")
    return report


def _partition_d5_train_by_store(train_path: Path, temp_dir: Path, chunksize: int = 1_000_000) -> pd.Timestamp:
    temp_dir.mkdir(parents=True, exist_ok=True)
    headers_written: set[int] = set()
    global_end: pd.Timestamp | None = None
    usecols = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
    for chunk in pd.read_csv(train_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["unit_sales"] = pd.to_numeric(chunk["unit_sales"], errors="coerce").clip(lower=0)
        chunk_end = chunk["date"].max()
        global_end = chunk_end if global_end is None else max(global_end, chunk_end)
        chunk["date"] = chunk["date"].dt.strftime("%Y-%m-%d")
        for store_nbr, group in chunk.groupby("store_nbr", sort=False):
            store = int(store_nbr)
            out_path = temp_dir / f"store_{store}.csv"
            group.to_csv(out_path, index=False, mode="a", header=store not in headers_written)
            headers_written.add(store)
    if global_end is None:
        raise ValueError("D5 train.csv is empty")
    return global_end


def export_d5_dataset(input_dir: Path, output_dir: Path) -> dict[str, object]:
    train_path = input_dir / "train.csv"
    items = pd.read_csv(input_dir / "items.csv")
    stores = pd.read_csv(input_dir / "stores.csv")
    oil = preprocess_d5_oil(pd.read_csv(input_dir / "oil.csv"))
    transactions = pd.read_csv(input_dir / "transactions.csv")
    holidays = preprocess_d5_holidays(pd.read_csv(input_dir / "holidays_events.csv"), stores)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "_d5_store_partitions"
    global_end = _partition_d5_train_by_store(train_path, temp_dir)
    out_path = output_dir / "D5_clean.csv"
    if out_path.exists():
        out_path.unlink()

    rows = 0
    groups = 0
    coverage_days: list[int] = []
    date_min: str | None = None
    first = True
    for store_row in stores.sort_values("store_nbr").itertuples(index=False):
        store_series = pd.Series(store_row._asdict())
        store_path = temp_dir / f"store_{int(store_series['store_nbr'])}.csv"
        if not store_path.exists():
            continue
        store_train = pd.read_csv(store_path, low_memory=False)
        clean = clean_d5_store_dataframe(
            store_train,
            items=items,
            store_row=store_series,
            oil=oil,
            transactions=transactions,
            holidays_by_store=holidays,
            global_end_date=global_end,
        )
        validate_clean_frame("D5", clean)
        clean.to_csv(out_path, index=False, mode="w" if first else "a", header=first)
        first = False
        rows += len(clean)
        group_sizes = clean.groupby(["entity_id", "item_id"]).size()
        groups += group_sizes.size
        coverage_days.extend(int(value) for value in group_sizes.tolist())
        store_min = str(clean["date"].min())
        date_min = store_min if date_min is None else min(date_min, store_min)

    report = {
        "dataset": "D5",
        "rows": int(rows),
        "columns": 21,
        "entity_item_groups": int(groups),
        "date_min": date_min,
        "date_max": str(global_end.date()),
        "sales_min": 0.0,
        "output_path": str(out_path),
        "temp_dir": str(temp_dir),
    }
    report.update(entity_coverage_summary_from_counts(coverage_days))
    return report


def export_phase1(datasets: Iterable[str], output_dir: Path) -> list[dict[str, object]]:
    reports = []
    for dataset in datasets:
        input_path = DEFAULT_PATHS[dataset]
        if not input_path.exists():
            raise FileNotFoundError(f"{dataset} input not found: {input_path}")
        reports.append(export_dataset(dataset, input_path, output_dir))
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["phase1"], default="phase1")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DEFAULT_PATHS), default=["D1", "D3", "D4"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_OUTPUT_DIR / "phase1_validation_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = export_phase1(args.datasets, args.output_dir)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    for report in reports:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print(f"validation_report={args.report_json}")


if __name__ == "__main__":
    main()
