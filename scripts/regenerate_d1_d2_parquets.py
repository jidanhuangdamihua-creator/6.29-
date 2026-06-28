#!/usr/bin/env python3
"""Regenerate D1/D2 solidified parquet files from raw CSV inputs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
D1_RAW = ROOT / "数据集" / "原始数据" / "Dataset 1" / "train.csv"
D2_RAW = ROOT / "数据集" / "原始数据" / "Dataset 2.csv"
OUT_DIR = ROOT / "数据集" / "固化数据"

D1_SCHEMA = ["date", "store_id", "item_id", "entity_id", "sales", "year", "month", "week", "day"]
D2_SCHEMA = ["date", "brand_id", "item_id", "entity_id", "sales", "promo", "year", "month", "week", "day"]

QTY_RE = re.compile(r"^QTY_B(\d+)_(\d+)$")
PROMO_RE = re.compile(r"^PROMO_B(\d+)_(\d+)$")


def add_datetime_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise").astype("datetime64[ns]")
    out["year"] = out["date"].dt.year.astype("int64")
    out["month"] = out["date"].dt.month.astype("int64")
    out["week"] = out["date"].dt.isocalendar().week.astype("int64")
    out["day"] = out["date"].dt.day.astype("int64")
    return out


def assert_columns(frame: pd.DataFrame, expected: list[str], label: str) -> None:
    actual = list(frame.columns)
    assert actual == expected, f"{label} columns mismatch: expected {expected}, got {actual}"


def assert_datetime_ns(frame: pd.DataFrame, label: str) -> None:
    dtype = frame["date"].dtype
    assert dtype == "datetime64[ns]", f"{label} date dtype must be datetime64[ns], got {dtype}"


def assert_target_window_covered(frame: pd.DataFrame, json_paths: Iterable[Path], label: str) -> None:
    date_min = frame["date"].min()
    date_max = frame["date"].max()
    for path in json_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        window = data["target_train_window"]
        start = pd.Timestamp(window["start"])
        end = pd.Timestamp(window["end"])
        assert date_min <= start <= date_max, f"{label} does not cover {path}: start {start.date()}"
        assert date_min <= end <= date_max, f"{label} does not cover {path}: end {end.date()}"


def build_d1() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(D1_RAW)
    required = {"date", "store", "item", "sales"}
    missing = sorted(required.difference(raw.columns))
    assert not missing, f"D1 missing required columns: {missing}"

    work = raw.loc[pd.to_numeric(raw["store"], errors="coerce") == 1].copy()
    work = work.rename(columns={"store": "store_id", "item": "item_id"})
    work["store_id"] = pd.to_numeric(work["store_id"], errors="raise").astype("int64")
    work["item_id"] = pd.to_numeric(work["item_id"], errors="raise").astype("int64")
    work["sales"] = pd.to_numeric(work["sales"], errors="raise")
    work["entity_id"] = work["store_id"].astype(str) + "_" + work["item_id"].astype(str)
    work = add_datetime_features(work)
    work = work[D1_SCHEMA].sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    target = work.loc[work["item_id"] == 10].copy().reset_index(drop=True)
    source = work.loc[work["item_id"] != 10].copy().reset_index(drop=True)

    assert target["entity_id"].drop_duplicates().tolist() == ["1_10"], "D1 target entity_id must be ['1_10']"
    assert int(source["item_id"].nunique()) == 49, f"D1 source item_id.nunique() must be 49, got {source['item_id'].nunique()}"
    assert_columns(source, D1_SCHEMA, "D1 source")
    assert_columns(target, D1_SCHEMA, "D1 target")
    assert_datetime_ns(source, "D1 source")
    assert_datetime_ns(target, "D1 target")
    assert_target_window_covered(
        target,
        [
            ROOT / "outputs" / "knn_selection" / "Dataset1" / "knn_without_info_sharing.json",
            ROOT / "outputs" / "knn_selection" / "Dataset1" / "knn_with_info_sharing.json",
        ],
        "D1 target",
    )
    return source, target


def melt_d2(raw: pd.DataFrame, columns: list[str], value_name: str, pattern: re.Pattern[str]) -> pd.DataFrame:
    long = raw[["DATE"] + columns].melt(id_vars=["DATE"], var_name="wide_key", value_name=value_name)
    extracted = long["wide_key"].str.extract(pattern)
    assert not extracted.isna().any().any(), f"D2 failed to parse {value_name} wide columns"
    long["brand_id"] = pd.to_numeric(extracted[0], errors="raise").astype("int64")
    long["item_id"] = pd.to_numeric(extracted[1], errors="raise").astype("int64")
    return long.drop(columns=["wide_key"])


def build_d2() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(D2_RAW)
    print("D2 raw columns:")
    print(list(raw.columns))

    assert "DATE" in raw.columns, "D2 missing DATE column"
    qty_cols = [col for col in raw.columns if QTY_RE.match(str(col))]
    promo_cols = [col for col in raw.columns if PROMO_RE.match(str(col))]
    assert qty_cols, "D2 requires QTY_Bx_y columns"

    qty_long = melt_d2(raw, qty_cols, "sales", QTY_RE)
    if promo_cols:
        promo_long = melt_d2(raw, promo_cols, "promo", PROMO_RE)
        work = qty_long.merge(promo_long, on=["DATE", "brand_id", "item_id"], how="left")
        work["promo"] = pd.to_numeric(work["promo"], errors="coerce").fillna(0).astype("int64")
    else:
        work = qty_long
        work["promo"] = 0

    work = work.rename(columns={"DATE": "date"})
    work = work.loc[work["brand_id"] == 1].copy()
    work["sales"] = pd.to_numeric(work["sales"], errors="raise")
    work["entity_id"] = work["brand_id"].astype(str) + "_" + work["item_id"].astype(str)
    work = add_datetime_features(work)
    work = work[D2_SCHEMA].sort_values(["brand_id", "item_id", "date"]).reset_index(drop=True)

    target = work.loc[work["item_id"] == 10].copy().reset_index(drop=True)
    source = work.loc[work["item_id"] != 10].copy().reset_index(drop=True)

    assert target["entity_id"].drop_duplicates().tolist() == ["1_10"], "D2 target entity_id must be ['1_10']"
    expected_source_entity = "1_" + source["item_id"].astype(str)
    assert source["entity_id"].equals(expected_source_entity), "D2 source entity_id must equal 1_{item_id}"
    assert_columns(source, D2_SCHEMA, "D2 source")
    assert_columns(target, D2_SCHEMA, "D2 target")
    assert_datetime_ns(source, "D2 source")
    assert_datetime_ns(target, "D2 target")
    assert_target_window_covered(
        target,
        [
            ROOT / "outputs" / "knn_selection" / "Dataset2" / "knn_without_info_sharing.json",
            ROOT / "outputs" / "knn_selection" / "Dataset2" / "knn_with_info_sharing.json",
        ],
        "D2 target",
    )
    return source, target


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def summarize(path: Path, expected_columns: list[str]) -> None:
    frame = pd.read_parquet(path)
    assert_columns(frame, expected_columns, path.name)
    assert_datetime_ns(frame, path.name)
    print(
        f"{path.name}: rows={len(frame)}, "
        f"entity_nunique={frame['entity_id'].nunique()}, "
        f"date_min={frame['date'].min().date()}, "
        f"date_max={frame['date'].max().date()}"
    )


def main() -> None:
    d1_source, d1_target = build_d1()
    d2_source, d2_target = build_d2()

    outputs = [
        (d1_source, OUT_DIR / "dataset1-source.parquet", D1_SCHEMA),
        (d1_target, OUT_DIR / "dataset1-target.parquet", D1_SCHEMA),
        (d2_source, OUT_DIR / "dataset2-source.parquet", D2_SCHEMA),
        (d2_target, OUT_DIR / "dataset2-target.parquet", D2_SCHEMA),
    ]
    for frame, path, _ in outputs:
        write_parquet_atomic(frame, path)

    print("Generated parquet summaries:")
    for _, path, schema in outputs:
        summarize(path, schema)


if __name__ == "__main__":
    main()
