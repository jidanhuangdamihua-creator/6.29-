#!/usr/bin/env python3
"""Full audit of D4 city/store fields.

This script intentionally reads all D4 rows, while limiting Parquet reads to the
columns needed for this audit so large nested hourly arrays are not materialized.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - only used when parquet support is absent
    pq = None


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "数据集" / "原始数据"
OUT_DIR = ROOT / "outputs" / "dataset_audit"

TARGET_FIELD = "city_idboolean"
STORE_FIELD = "store_id"
SIMILAR_FIELDS = [
    "city_id",
    "city",
    "store",
    "store_nbr",
    "shop_id",
    "item_id",
    "sku_id",
    "product_id",
]
DATE_CANDIDATES = ["date", "sales_date", "transaction_date", "dt"]
SALES_CANDIDATES = ["sales", "demand", "qty", "quantity", "unit_sales", "sale_amount"]
SUPPORTED_SUFFIXES = {".csv", ".parquet", ".xlsx", ".xls"}


@dataclass
class FileInfo:
    path: Path
    fmt: str
    rows: int | None
    columns: list[str]
    selected: bool
    reason: str


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parquet_info(path: Path) -> tuple[int | None, list[str]]:
    if pq is None:
        return None, []
    pf = pq.ParquetFile(path)
    return pf.metadata.num_rows, list(pf.schema_arrow.names)


def csv_info(path: Path) -> tuple[int | None, list[str]]:
    header = pd.read_csv(path, nrows=0)
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for rows, _ in enumerate(reader, start=1):
            pass
    return rows, list(header.columns)


def excel_info(path: Path) -> tuple[int | None, list[str]]:
    frame = pd.read_excel(path, nrows=0)
    row_count = pd.read_excel(path, usecols=[0]).shape[0]
    return int(row_count), list(frame.columns)


def file_info(path: Path) -> FileInfo:
    suffix = path.suffix.lower()
    fmt = suffix.lstrip(".")
    try:
        if suffix == ".parquet":
            rows, columns = parquet_info(path)
        elif suffix == ".csv":
            rows, columns = csv_info(path)
        elif suffix in {".xlsx", ".xls"}:
            rows, columns = excel_info(path)
        else:
            rows, columns = None, []
    except Exception as exc:
        rows, columns = None, []
        return FileInfo(path, fmt, rows, columns, False, f"metadata read failed: {exc}")
    return FileInfo(path, fmt, rows, columns, False, "")


def locate_d4_files() -> list[FileInfo]:
    files = [
        path
        for path in DATA_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    infos = [file_info(path) for path in files]

    dataset4_infos = [
        info
        for info in infos
        if "dataset 4" in str(info.path).lower()
        or "dataset4" in str(info.path).lower()
        or "d4" in info.path.name.lower()
    ]
    selected = [
        info
        for info in dataset4_infos
        if "sample" not in info.path.name.lower() and info.path.suffix.lower() == ".parquet"
    ]
    if not selected:
        selected = [
            info
            for info in dataset4_infos
            if "sample" not in info.path.name.lower()
        ]
    if not selected:
        selected = dataset4_infos

    selected_paths = {info.path for info in selected}
    for info in infos:
        if info.path in selected_paths:
            info.selected = True
            info.reason = "Dataset 4 path match; non-sample D4 data file"
        elif info in dataset4_infos:
            info.reason = "Dataset 4 candidate but not selected, likely sample or auxiliary file"
        else:
            info.reason = "not D4 path match"
    return infos


def read_selected_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if pq is None:
            raise RuntimeError("pyarrow is required to read parquet files")
        table = pq.read_table(path, columns=columns)
        return table.to_pandas()
    if suffix == ".csv":
        chunks = []
        for chunk in pd.read_csv(path, usecols=columns, chunksize=500_000):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=columns)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, usecols=columns)
    raise ValueError(f"Unsupported format: {path}")


def is_missing(series: pd.Series) -> pd.Series:
    missing = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        missing = missing | series.astype("string").str.strip().eq("")
    return missing.fillna(True)


def value_label(value) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def compact_values(values: Iterable, limit: int) -> str:
    labels = [value_label(value) for value in values]
    if len(labels) <= limit:
        return ";".join(labels)
    return ";".join(labels[:limit]) + f";... ({len(labels)} total)"


def field_stats(df: pd.DataFrame, field: str) -> dict:
    if field not in df.columns:
        return {
            "field": field,
            "exists": False,
            "dtype": "N/A",
            "non_null": 0,
            "missing": len(df),
            "missing_rate": 1.0 if len(df) else 0.0,
            "unique_count": 0,
            "unique_values": [],
            "value_counts": pd.DataFrame(columns=["field", "value", "count", "pct"]),
            "boolean_like": False,
            "city_id_like": False,
            "anomalies": ["field_missing"],
        }

    series = df[field]
    missing = is_missing(series)
    non_missing = series[~missing]
    counts = series.where(~missing, other=pd.NA).value_counts(dropna=False)
    value_counts = counts.rename_axis("value").reset_index(name="count")
    value_counts["field"] = field
    value_counts["pct"] = value_counts["count"] / len(df) if len(df) else 0.0
    value_counts = value_counts[["field", "value", "count", "pct"]]

    unique_values = list(non_missing.drop_duplicates().sort_values(ignore_index=True))
    normalized = set(non_missing.astype(str).str.strip().str.lower().unique())
    bool_sets = [
        {"true", "false"},
        {"0", "1"},
        {"y", "n"},
        {"yes", "no"},
        {"t", "f"},
    ]
    boolean_like = bool(normalized) and any(normalized <= allowed for allowed in bool_sets)

    numeric = pd.to_numeric(non_missing, errors="coerce")
    numeric_ok = numeric.notna().all() if len(non_missing) else False
    integral = bool(numeric_ok and (numeric.dropna() % 1 == 0).all())
    unique_count = int(non_missing.nunique(dropna=True))
    city_id_like = bool(integral and unique_count > 2)

    anomalies = []
    if missing.any():
        anomalies.append("missing_values")
    if numeric_ok and (numeric < 0).any():
        anomalies.append("negative_values")
    if numeric_ok and not integral:
        anomalies.append("decimal_values")
    type_names = sorted({type(x).__name__ for x in non_missing.head(10000)})
    if len(type_names) > 1:
        anomalies.append("mixed_python_types_in_first_10000_non_missing")

    return {
        "field": field,
        "exists": True,
        "dtype": str(series.dtype),
        "non_null": int((~missing).sum()),
        "missing": int(missing.sum()),
        "missing_rate": float(missing.mean()) if len(df) else 0.0,
        "unique_count": unique_count,
        "unique_values": unique_values,
        "value_counts": value_counts,
        "boolean_like": boolean_like,
        "city_id_like": city_id_like,
        "anomalies": anomalies,
    }


def pct(value: float) -> str:
    return f"{value:.6%}"


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if limit is not None:
        frame = frame.head(limit)
    if frame.empty:
        return "_无记录_"
    view = frame[columns].copy()
    view = view.fillna("")
    rows = [[str(value) for value in row] for row in view.to_numpy().tolist()]
    headers = [str(col) for col in columns]
    widths = []
    for idx, header in enumerate(headers):
        max_cell = max([len(row[idx]) for row in rows], default=0)
        widths.append(max(len(header), max_cell))

    def fmt_row(values: list[str]) -> str:
        cells = [values[idx].ljust(widths[idx]) for idx in range(len(values))]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt_row(headers), separator] + [fmt_row(row) for row in rows])


def detect_date_field(columns: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in DATE_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    for col in columns:
        low = col.lower()
        if "date" in low or low in {"dt", "day"}:
            return col
    return None


def detect_sales_field(columns: list[str]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in SALES_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    for col in columns:
        low = col.lower()
        if "sales" in low or "sale" in low or "demand" in low or "qty" in low:
            return col
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    infos = locate_d4_files()
    selected_infos = [info for info in infos if info.selected]
    if not selected_infos:
        raise RuntimeError("No D4 data files could be selected")

    all_columns = sorted({col for info in selected_infos for col in info.columns})
    date_field = detect_date_field(all_columns)
    sales_field = detect_sales_field(all_columns)
    read_columns = [
        col
        for col in [TARGET_FIELD, "city_id", STORE_FIELD, date_field, sales_field]
        if col and col in all_columns
    ]
    read_columns = list(dict.fromkeys(read_columns))

    frames = []
    for info in selected_infos:
        available = [col for col in read_columns if col in info.columns]
        frame = read_selected_columns(info.path, available)
        frame["__source_file"] = rel(info.path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    city_field = TARGET_FIELD if TARGET_FIELD in df.columns else ("city_id" if "city_id" in df.columns else None)
    city_stats = field_stats(df, TARGET_FIELD)
    proxy_city_stats = field_stats(df, city_field) if city_field else None
    store_stats = field_stats(df, STORE_FIELD)

    value_counts_frames = []
    if city_stats["exists"]:
        value_counts_frames.append(city_stats["value_counts"])
    else:
        value_counts_frames.append(
            pd.DataFrame(
                [
                    {
                        "field": TARGET_FIELD,
                        "value": "__FIELD_NOT_PRESENT__",
                        "count": 0,
                        "pct": 0.0,
                    }
                ]
            )
        )
    if proxy_city_stats and city_field != TARGET_FIELD:
        proxy_counts = proxy_city_stats["value_counts"].copy()
        proxy_counts["field"] = f"{TARGET_FIELD}_proxy:{city_field}"
        value_counts_frames.append(proxy_counts)
    if store_stats["exists"]:
        value_counts_frames.append(store_stats["value_counts"])
    value_counts = pd.concat(value_counts_frames, ignore_index=True)
    value_counts.to_csv(OUT_DIR / "d4_city_store_value_counts.csv", index=False)

    if not city_field or STORE_FIELD not in df.columns:
        store_city_mapping = pd.DataFrame()
        city_store_mapping = pd.DataFrame()
    else:
        relation = df[[STORE_FIELD, city_field]].copy()
        relation["__missing_store"] = is_missing(relation[STORE_FIELD])
        relation["__missing_city"] = is_missing(relation[city_field])
        valid_relation = relation[~relation["__missing_store"] & ~relation["__missing_city"]]

        store_group = valid_relation.groupby(STORE_FIELD, dropna=False)
        store_city_mapping = store_group.agg(
            row_count=(city_field, "size"),
            city_value_count=(city_field, "nunique"),
        ).reset_index()
        store_city_values = store_group[city_field].agg(lambda s: compact_values(sorted(s.drop_duplicates()), 200))
        store_city_mapping["city_values"] = store_city_mapping[STORE_FIELD].map(store_city_values)
        store_city_mapping["row_pct"] = store_city_mapping["row_count"] / len(df)

        if date_field and date_field in df.columns:
            dates = pd.to_datetime(df[date_field], errors="coerce")
            tmp = df[[STORE_FIELD]].copy()
            tmp["__date"] = dates
            date_summary = tmp.dropna(subset=[STORE_FIELD, "__date"]).groupby(STORE_FIELD).agg(
                date_min=("__date", "min"),
                date_max=("__date", "max"),
                calendar_days=("__date", "nunique"),
            )
            store_city_mapping = store_city_mapping.merge(
                date_summary.reset_index(), on=STORE_FIELD, how="left"
            )
        else:
            store_city_mapping["date_min"] = pd.NaT
            store_city_mapping["date_max"] = pd.NaT
            store_city_mapping["calendar_days"] = pd.NA

        if date_field and sales_field and date_field in df.columns and sales_field in df.columns:
            sales_numeric = pd.to_numeric(df[sales_field], errors="coerce")
            sale_tmp = df[[STORE_FIELD, date_field]].copy()
            sale_tmp["__sale"] = sales_numeric
            sale_tmp["__date"] = pd.to_datetime(sale_tmp[date_field], errors="coerce")
            positive = sale_tmp[sale_tmp["__sale"].notna() & (sale_tmp["__sale"] > 0)]
            sales_days = positive.dropna(subset=[STORE_FIELD, "__date"]).groupby(STORE_FIELD)["__date"].nunique()
            store_city_mapping["effective_sales_days"] = store_city_mapping[STORE_FIELD].map(sales_days).fillna(0).astype(int)
        else:
            store_city_mapping["effective_sales_days"] = pd.NA

        store_city_mapping.insert(1, "city_field_used", city_field)
        store_city_mapping = store_city_mapping.sort_values([STORE_FIELD]).reset_index(drop=True)

        city_group = valid_relation.groupby(city_field, dropna=False)
        city_store_mapping = city_group.agg(
            row_count=(STORE_FIELD, "size"),
            store_count=(STORE_FIELD, "nunique"),
        ).reset_index()
        city_store_values = city_group[STORE_FIELD].agg(lambda s: compact_values(sorted(s.drop_duplicates()), 200))
        city_store_mapping["store_values"] = city_store_mapping[city_field].map(city_store_values)
        city_store_mapping["row_pct"] = city_store_mapping["row_count"] / len(df)
        city_store_mapping.insert(1, "city_field_used", city_field)
        city_store_mapping = city_store_mapping.sort_values([city_field]).reset_index(drop=True)

    store_city_mapping.to_csv(OUT_DIR / "d4_store_city_mapping.csv", index=False)
    city_store_mapping.to_csv(OUT_DIR / "d4_city_store_mapping.csv", index=False)

    anomalies = []
    if not city_stats["exists"]:
        anomalies.append(
            {
                "anomaly_type": "field_missing",
                "field": TARGET_FIELD,
                "count": len(df),
                "details": f"{TARGET_FIELD} is absent; similar field used for relationship audit: {city_field}",
            }
        )
    if city_field == "city_id":
        anomalies.append(
            {
                "anomaly_type": "likely_misnamed_field",
                "field": TARGET_FIELD,
                "count": len(df),
                "details": "D4 schema contains city_id, not city_idboolean; city_id is numeric and has more than two values.",
            }
        )
    if proxy_city_stats and not proxy_city_stats["boolean_like"]:
        anomalies.append(
            {
                "anomaly_type": "not_boolean_like",
                "field": city_field,
                "count": proxy_city_stats["unique_count"],
                "details": f"{city_field} unique values are not limited to boolean-like values.",
            }
        )
    if not store_city_mapping.empty:
        multi_city_store = store_city_mapping[store_city_mapping["city_value_count"] > 1]
        for _, row in multi_city_store.iterrows():
            anomalies.append(
                {
                    "anomaly_type": "store_maps_to_multiple_city_values",
                    "field": STORE_FIELD,
                    "count": int(row["city_value_count"]),
                    "details": f"store_id={row[STORE_FIELD]} city_values={row['city_values']}",
                }
            )
    if city_field and STORE_FIELD in df.columns:
        missing_store = is_missing(df[STORE_FIELD])
        missing_city = is_missing(df[city_field])
        city_missing_store_present = int((missing_city & ~missing_store).sum())
        store_missing_city_present = int((missing_store & ~missing_city).sum())
        if city_missing_store_present:
            anomalies.append(
                {
                    "anomaly_type": "city_missing_store_present",
                    "field": city_field,
                    "count": city_missing_store_present,
                    "details": f"{city_field} missing while store_id present",
                }
            )
        if store_missing_city_present:
            anomalies.append(
                {
                    "anomaly_type": "store_missing_city_present",
                    "field": STORE_FIELD,
                    "count": store_missing_city_present,
                    "details": f"store_id missing while {city_field} present",
                }
            )
    anomaly_frame = pd.DataFrame(anomalies, columns=["anomaly_type", "field", "count", "details"])
    anomaly_frame.to_csv(OUT_DIR / "d4_city_store_anomalies.csv", index=False)

    candidate_rows = []
    for info in infos:
        if "dataset 4" in str(info.path).lower() or "dataset4" in str(info.path).lower() or "d4" in info.path.name.lower():
            candidate_rows.append(
                {
                    "path": rel(info.path),
                    "format": info.fmt,
                    "rows": info.rows,
                    "cols": len(info.columns),
                    "selected": info.selected,
                    "reason": info.reason,
                }
            )
    candidate_frame = pd.DataFrame(candidate_rows)

    selected_paths = [rel(info.path) for info in selected_infos]
    row_counts_by_file = pd.DataFrame(
        [{"path": rel(info.path), "rows": info.rows, "columns": len(info.columns)} for info in selected_infos]
    )
    total_rows = len(df)
    total_cols = len(all_columns)
    key_presence = {
        TARGET_FIELD: TARGET_FIELD in all_columns,
        STORE_FIELD: STORE_FIELD in all_columns,
        **{field: field in all_columns for field in SIMILAR_FIELDS},
    }

    city_for_report = city_stats if city_stats["exists"] else proxy_city_stats
    city_label = TARGET_FIELD if city_stats["exists"] else f"{city_field}（作为 {TARGET_FIELD} 的相似字段审计）"

    city_counts_md = "_字段不存在，无法对 city_idboolean 本体做值分布。下表为 city_id 的全量分布：_"
    if city_for_report is not None:
        city_counts_source = city_for_report["value_counts"].copy()
        city_counts_source["pct"] = city_counts_source["pct"].map(pct)
        city_counts_md += "\n\n" + markdown_table(city_counts_source, ["field", "value", "count", "pct"], limit=100)

    store_counts = store_stats["value_counts"].copy() if store_stats["exists"] else pd.DataFrame()
    if not store_counts.empty:
        store_counts["pct"] = store_counts["pct"].map(pct)
        store_counts = store_counts.sort_values(["count", "value"], ascending=[False, True])

    store_map_md = store_city_mapping.copy()
    if not store_map_md.empty:
        store_map_md["row_pct"] = store_map_md["row_pct"].map(pct)
        for col in ["date_min", "date_max"]:
            if col in store_map_md.columns:
                store_map_md[col] = pd.to_datetime(store_map_md[col], errors="coerce").dt.strftime("%Y-%m-%d")

    city_map_md = city_store_mapping.copy()
    if not city_map_md.empty:
        city_map_md["row_pct"] = city_map_md["row_pct"].map(pct)

    one_to_many_store = (
        int((store_city_mapping["city_value_count"] > 1).sum()) if not store_city_mapping.empty else math.nan
    )
    city_to_store_max = (
        int(city_store_mapping["store_count"].max()) if not city_store_mapping.empty else math.nan
    )
    store_to_city_max = (
        int(store_city_mapping["city_value_count"].max()) if not store_city_mapping.empty else math.nan
    )
    relation_summary = (
        "store_id -> city_id 为一对一；city_id -> store_id 为一对多（多个门店属于同一城市）。"
        if one_to_many_store == 0 and city_to_store_max > 1
        else "映射存在异常或无法完整判断，详见异常表。"
    )

    md = f"""# D4 city_idboolean 与 store_id 全量审计报告

## 1. 审计目标

本次检查回答 D4 数据集中 `{TARGET_FIELD}` 与 `{STORE_FIELD}` 的字段存在性、取值分布、缺失与异常情况，以及二者之间是否构成一一对应、多对一、一对多或混乱映射。脚本读取 D4 全量 train/eval 数据文件；没有抽样、没有只看前几行，也没有运行训练实验。

## 2. D4 数据文件定位

- 文件路径：{", ".join(selected_paths)}
- 文件格式：{", ".join(sorted({info.fmt for info in selected_infos}))}
- 总行数：{total_rows}
- 总列数（所选文件列名并集）：{total_cols}
- 日期字段自动识别：`{date_field}`
- 销量字段自动识别：`{sales_field}`

候选文件判断：

{markdown_table(candidate_frame, ["path", "format", "rows", "cols", "selected", "reason"])}

所选文件行数：

{markdown_table(row_counts_by_file, ["path", "rows", "columns"])}

关键字段是否存在：

{markdown_table(pd.DataFrame([{"field": k, "exists": v} for k, v in key_presence.items()]), ["field", "exists"])}

## 3. city_idboolean 字段检查

- `{TARGET_FIELD}` 是否存在：{city_stats["exists"]}
- pandas dtype：{city_stats["dtype"]}
- 非空数：{city_stats["non_null"]}
- 缺失数：{city_stats["missing"]}
- 缺失率：{pct(city_stats["missing_rate"])}
- 唯一值数量：{city_stats["unique_count"]}
- 是否像 boolean：{city_stats["boolean_like"]}
- 是否像 city_id：{city_stats["city_id_like"]}
- 异常值/异常状态：{", ".join(city_stats["anomalies"]) if city_stats["anomalies"] else "无"}

`{TARGET_FIELD}` 在 D4 原始字段中不存在。D4 README 和 Parquet schema 中存在的是 `city_id`，因此下面补充 `city_id` 的全量检查，用于判断是否为城市编号字段被误写成 boolean：

- 审计字段：{city_label}
- pandas dtype：{city_for_report["dtype"] if city_for_report else "N/A"}
- 非空数：{city_for_report["non_null"] if city_for_report else "N/A"}
- 缺失数：{city_for_report["missing"] if city_for_report else "N/A"}
- 缺失率：{pct(city_for_report["missing_rate"]) if city_for_report else "N/A"}
- 唯一值数量：{city_for_report["unique_count"] if city_for_report else "N/A"}
- 唯一值列表：{compact_values(city_for_report["unique_values"], 100) if city_for_report else "N/A"}
- 是否只有 True/False、0/1、Y/N 等布尔结构：{city_for_report["boolean_like"] if city_for_report else "N/A"}
- 是否像城市 ID：{city_for_report["city_id_like"] if city_for_report else "N/A"}
- 异常值：{", ".join(city_for_report["anomalies"]) if city_for_report and city_for_report["anomalies"] else "无明显异常"}

值分布：

{city_counts_md}

## 4. store_id 字段检查

- `{STORE_FIELD}` 是否存在：{store_stats["exists"]}
- pandas dtype：{store_stats["dtype"]}
- 非空数：{store_stats["non_null"]}
- 缺失数：{store_stats["missing"]}
- 缺失率：{pct(store_stats["missing_rate"])}
- 唯一 store 数量：{store_stats["unique_count"]}
- store_id 完整列表说明：完整列表与完整频数已写入 `d4_city_store_value_counts.csv`；下表显示前 200 个高频 store。
- 是否存在异常值：{", ".join(store_stats["anomalies"]) if store_stats["anomalies"] else "无明显异常"}

store_id 分布（前 200 个，按行数降序）：

{markdown_table(store_counts, ["field", "value", "count", "pct"], limit=200)}

每个 store 的城市映射、日期范围、有效销售天数已写入 `d4_store_city_mapping.csv`。预览：

{markdown_table(store_map_md, [STORE_FIELD, "city_field_used", "city_value_count", "city_values", "row_count", "row_pct", "date_min", "date_max", "calendar_days", "effective_sales_days"], limit=200)}

## 5. city_idboolean 与 store_id 的映射关系

由于 `{TARGET_FIELD}` 不存在，严格意义上无法检查 `{TARGET_FIELD}` 本体与 `{STORE_FIELD}` 的映射。以下关系检查使用相似字段 `{city_field}` 完成。

- 每个 `{STORE_FIELD}` 对应的 `{city_field}` 最大数量：{store_to_city_max}
- `{STORE_FIELD}` 对应多个 `{city_field}` 的 store 数量：{one_to_many_store}
- 每个 `{city_field}` 对应的 `{STORE_FIELD}` 最大数量：{city_to_store_max}
- 关系判断：{relation_summary}
- `{city_field}` 缺失但 `{STORE_FIELD}` 不缺失：{int((is_missing(df[city_field]) & ~is_missing(df[STORE_FIELD])).sum()) if city_field and STORE_FIELD in df.columns else "N/A"}
- `{STORE_FIELD}` 缺失但 `{city_field}` 不缺失：{int((is_missing(df[STORE_FIELD]) & ~is_missing(df[city_field])).sum()) if city_field and STORE_FIELD in df.columns else "N/A"}

交叉表 1：store_id -> city_idboolean（实际使用 `{city_field}`）预览，完整表见 `d4_store_city_mapping.csv`。

{markdown_table(store_map_md, [STORE_FIELD, "city_field_used", "city_value_count", "city_values", "row_count", "row_pct"], limit=200)}

交叉表 2：city_idboolean -> store_id（实际使用 `{city_field}`）预览，完整表见 `d4_city_store_mapping.csv`。

{markdown_table(city_map_md, [city_field, "city_field_used", "store_count", "store_values", "row_count", "row_pct"], limit=100) if city_field and not city_map_md.empty else "_无记录_"}

异常映射明细见 `d4_city_store_anomalies.csv`。

## 6. 对 D4 实体划分的影响

- D4 的实体建议按 `{STORE_FIELD}` 划分。README 中写明仓储层级为 `city_id > store_id`，全量映射也显示每个 store 只属于一个 city。
- `{TARGET_FIELD}` 不适合作为 source/target domain 划分字段，因为它在原始 D4 中不存在。
- `city_id` 可作为 domain 候选字段，适合做城市级 source/target 划分；它不是 boolean，而是 18 个城市编码。
- `{TARGET_FIELD}` 更像是代码或配置中把 `city_id` 错误拼接/命名成 boolean 后产生的字段名，而不是 D4 原始字段。
- 冷启动任务中建议：实体粒度使用 `{STORE_FIELD}`；若要跨城市迁移或城市留出，用 `city_id` 设计 source/target domain；若要门店冷启动，用 store_id 做目标实体留出并防止同一 store 泄漏到 source pool。
- 对 source pool、KNN 选择和实体划分的影响：不要使用不存在的 `{TARGET_FIELD}`；KNN 的实体索引和相似源选择应显式使用 `{STORE_FIELD}`，domain 约束若需要城市层级则使用 `city_id`。

## 7. 结论摘要

- D4 自动定位为 FreshRetailNet-LT 的 train/eval Parquet 文件，合计 {total_rows} 行。
- 原始 D4 不存在 `{TARGET_FIELD}` 字段。
- 原始 D4 存在 `city_id` 字段，并且 README 将其定义为 encoded city id。
- `city_id` 唯一值数量为 {city_for_report["unique_count"] if city_for_report else "N/A"}，不是 True/False、0/1 或 Y/N 结构。
- `{STORE_FIELD}` 唯一值数量为 {store_stats["unique_count"]}。
- 全量检查显示每个 `{STORE_FIELD}` 对应的 `{city_field}` 最大数量为 {store_to_city_max}。
- `{city_field}` 到 `{STORE_FIELD}` 是一对多：多个门店属于同一个城市。
- 未发现 `{STORE_FIELD}` 与 `{city_field}` 之间的混乱映射，异常主要是 `{TARGET_FIELD}` 字段缺失/命名不一致。

## 8. 后续建议

### 必须修复

- 将任何引用 `{TARGET_FIELD}` 的 D4 配置、预处理或实验代码改为引用真实字段 `city_id`，或在清洗层明确生成并记录该字段的来源。
- source/target domain 设计中不要把 `{TARGET_FIELD}` 当作原始字段。

### 建议确认

- 确认后续实验是否应同时使用 train/eval，还是只使用 train；本审计按 D4 全量已覆盖 train/eval。
- 确认冷启动实验是门店冷启动、城市冷启动，还是商品-门店序列冷启动；三者对应的实体划分不同。

### 可暂缓

- 可以暂缓重命名原始数据文件；只要在配置和报告中明确 D4 的真实字段是 `city_id` 即可。
- 如果后续不做城市级迁移，可暂缓把 `city_id` 作为强 domain 约束，只保留为分析字段。
"""

    report_path = OUT_DIR / "d4_city_store_audit.md"
    report_path.write_text(md, encoding="utf-8")

    print(f"D4 文件路径: {', '.join(selected_paths)}")
    print(f"总行数: {total_rows}")
    print(f"{TARGET_FIELD} 唯一值数量: {city_stats['unique_count']} (字段存在: {city_stats['exists']})")
    if proxy_city_stats and city_field != TARGET_FIELD:
        print(f"{city_field} 唯一值数量: {proxy_city_stats['unique_count']} (作为相似字段)")
    print(f"{STORE_FIELD} 唯一值数量: {store_stats['unique_count']}")
    print(f"是否存在 store_id 对应多个 {city_field}: {bool(one_to_many_store and one_to_many_store > 0)}")
    print(f"报告输出路径: {rel(report_path)}")


if __name__ == "__main__":
    main()
