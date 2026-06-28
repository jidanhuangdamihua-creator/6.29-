#!/usr/bin/env python3
"""Specialized Dataset5/Favorita negative unit_sales audit.

This script is intentionally read-only for data/model code. It locates the
Favorita train file, scans it in chunks, writes a timestamp-safe report folder,
and audits current project code for negative-sales handling.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


D5_KEYWORDS = ("D5", "Dataset5", "dataset5", "Dataset 5", "Favorita", "favorita")
SALES_CANDIDATES = ("unit_sales", "sales", "demand", "qty", "quantity", "Sale", "Sales")
ITEM_CANDIDATES = ("item_nbr", "item_id", "item", "sku", "product_id", "Product")
DATE_CANDIDATES = ("date", "Date", "ds")
STORE_CANDIDATES = ("store_nbr", "store_id", "store", "Store")
PY_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "outputs", "reports", "logs", "_git_backup"}


def make_output_dir(root: Path, preferred_name: str) -> Path:
    base = root / preferred_name
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    stamped = root / f"{preferred_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stamped.mkdir(parents=True, exist_ok=False)
    return stamped


def read_columns(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    if suffix == ".parquet":
        return list(pd.read_parquet(path, columns=[]).columns)
    return []


def count_csv_rows(path: Path) -> int:
    total_newlines = 0
    last_byte = b""
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024 * 8)
            if not block:
                break
            total_newlines += block.count(b"\n")
            last_byte = block[-1:]
    if total_newlines == 0:
        return 0
    data_lines = total_newlines - 1
    if last_byte and last_byte != b"\n":
        data_lines += 1
    return max(data_lines, 0)


def choose_column(columns: Iterable[str], preferred: str, candidates: Iterable[str]) -> Tuple[Optional[str], str]:
    col_list = list(columns)
    if preferred in col_list:
        return preferred, f"exact:{preferred}"
    lowered = {str(c).lower(): c for c in col_list}
    for candidate in candidates:
        if candidate.lower() in lowered:
            chosen = lowered[candidate.lower()]
            return chosen, f"candidate:{candidate}"
    return None, "not_found"


def find_dataset_candidates(root: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    scanned: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in PY_SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in {".csv", ".parquet"}:
            continue
        text = str(path)
        keyword_hit = any(keyword in text for keyword in D5_KEYWORDS)
        name_hit = path.name.lower() in {"train.csv", "sales_train.csv"}
        if not (keyword_hit or name_hit):
            continue
        try:
            columns = read_columns(path)
        except Exception as exc:  # pragma: no cover - defensive reporting
            scanned.append({"path": str(path), "columns": "", "error": str(exc)})
            continue
        row = {"path": str(path), "columns": ",".join(columns), "error": ""}
        scanned.append(row)
        sales_col, _ = choose_column(columns, "unit_sales", SALES_CANDIDATES)
        item_col, _ = choose_column(columns, "item_nbr", ITEM_CANDIDATES)
        score = 0
        score += 100 if "unit_sales" in columns else 0
        score += 50 if "item_nbr" in columns else 0
        score += 25 if path.name.lower() == "train.csv" else 0
        score += 10 if any(keyword in text for keyword in D5_KEYWORDS) else 0
        score += 5 if "date" in columns else 0
        if sales_col and item_col:
            candidates.append({"path": path, "columns": columns, "score": score})
    candidates.sort(key=lambda x: (-int(x["score"]), len(str(x["path"]))))
    return candidates, scanned


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def describe_series(values: pd.Series) -> pd.Series:
    if values.empty:
        return pd.Series(
            {
                "count": 0.0,
                "mean": math.nan,
                "std": math.nan,
                "min": math.nan,
                "25%": math.nan,
                "50%": math.nan,
                "75%": math.nan,
                "max": math.nan,
            }
        )
    return values.describe(percentiles=[0.25, 0.5, 0.75])


def format_float(value: Any, digits: int = 8) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "nan"
    return f"{f:.{digits}g}"


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_无数据_"
    view = df.head(max_rows).copy()
    lines = [
        "| " + " | ".join(str(c) for c in view.columns) + " |",
        "| " + " | ".join("---" for _ in view.columns) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for col in view.columns:
            val = row[col]
            if isinstance(val, float):
                values.append(format_float(val))
            else:
                values.append(str(val).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def update_sum(target: Dict[Any, float], series: pd.Series) -> None:
    for key, value in series.items():
        target[key] += float(value)


def update_count(target: Dict[Any, int], series: pd.Series) -> None:
    for key, value in series.items():
        target[key] += int(value)


def enclosing_functions_by_line(path: Path) -> Dict[int, str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    mapping: Dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            for line_no in range(node.lineno, end + 1):
                mapping[line_no] = node.name
    return mapping


def audit_code(root: Path, audit_script_path: Path) -> pd.DataFrame:
    patterns = [
        ("unit_sales", re.compile(r"\bunit_sales\b")),
        ("item_nbr", re.compile(r"\bitem_nbr\b")),
        ("negative_filter", re.compile(r"(unit_sales|sales)\s*<\s*0")),
        ("clip_lower_zero", re.compile(r"\.clip\([^)]*lower\s*=\s*0|\.clip\(\s*0")),
        ("log1p", re.compile(r"log1p")),
        (
            "drop_negative_sales_hint",
            re.compile(
                r"(drop|dropna|filter|query).*?(unit_sales|sales|negative|负值)"
                r"|(?:unit_sales|sales|negative|负值).*?(drop|dropna|filter|query)",
                re.IGNORECASE,
            ),
        ),
        ("dataset5", re.compile(r"Dataset5|dataset5|Favorita|D5")),
        ("normalize_features", re.compile(r"def normalize_features|MinMaxScaler|scaler\.fit")),
    ]
    rows: List[Dict[str, Any]] = []
    for path in root.rglob("*.py"):
        if path.resolve() == audit_script_path.resolve():
            continue
        if any(part in PY_SKIP_DIRS for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        funcs = enclosing_functions_by_line(path)
        for idx, line in enumerate(lines, start=1):
            matches = [name for name, pattern in patterns if pattern.search(line)]
            if not matches:
                continue
            rows.append(
                {
                    "file": str(path),
                    "line": idx,
                    "function": funcs.get(idx, ""),
                    "match_type": ",".join(matches),
                    "code": line.strip(),
                }
            )
    return pd.DataFrame(rows)


def classify_code_handling(code_hits: pd.DataFrame) -> Dict[str, Any]:
    if code_hits.empty:
        return {
            "handles_negative_unit_sales": False,
            "has_log1p_unit_sales_risk": False,
            "summary": "未发现相关 Python 代码命中。",
        }
    code_joined = "\n".join(code_hits["code"].astype(str).tolist())
    unit_hits = code_hits[code_hits["match_type"].str.contains("unit_sales", na=False)]
    negative_hits = code_hits[code_hits["match_type"].str.contains("negative_filter", na=False)]
    clip_hits = code_hits[
        code_hits["match_type"].str.contains("clip_lower_zero", na=False)
        & code_hits["code"].str.contains("unit_sales|sales", case=False, regex=True)
    ]
    log_hits = code_hits[code_hits["match_type"].str.contains("log1p", na=False)]
    unit_log_risk = bool(
        not log_hits.empty
        and log_hits["code"].str.contains("unit_sales|sales", case=False, regex=True).any()
    )
    dataset5_hits = code_hits[code_hits["match_type"].str.contains("dataset5", na=False)]
    return {
        "unit_sales_hit_count": int(len(unit_hits)),
        "negative_filter_hit_count": int(len(negative_hits)),
        "clip_sales_hit_count": int(len(clip_hits)),
        "log1p_hit_count": int(len(log_hits)),
        "dataset5_hit_count": int(len(dataset5_hits)),
        "handles_negative_unit_sales": bool(len(negative_hits) or len(clip_hits)),
        "has_log1p_unit_sales_risk": unit_log_risk,
        "summary": (
            "未发现项目代码中对 Dataset5 `unit_sales < 0` 的显式过滤、删除或 clip 到 0；"
            "也未发现 `np.log1p(unit_sales)` / `log1p(sales)` 的直接命中。"
        )
        if not unit_log_risk
        else "发现可能涉及 sales/unit_sales 的 log1p，请人工复核。"
        if "log1p" in code_joined
        else "未发现 log1p 相关风险命中。",
    }


def plot_outputs(negative_items: pd.DataFrame, neg_values: pd.Series, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    if negative_items.empty:
        plt.text(0.5, 0.5, "No negative rows", ha="center", va="center")
    else:
        plt.hist(negative_items["neg_ratio"], bins=50, color="#4c78a8", edgecolor="white")
        plt.xlabel("item-level negative row ratio")
        plt.ylabel("item count")
    plt.tight_layout()
    plt.savefig(out_dir / "dataset5_item_neg_ratio_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    if neg_values.empty:
        plt.text(0.5, 0.5, "No negative unit_sales", ha="center", va="center")
    else:
        plt.hist(neg_values, bins=80, color="#d04f3a", edgecolor="white")
        plt.xlabel("negative unit_sales")
        plt.ylabel("row count")
    plt.tight_layout()
    plt.savefig(out_dir / "dataset5_negative_unit_sales_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    top = negative_items.sort_values("neg_rows", ascending=False).head(20)
    if top.empty:
        plt.text(0.5, 0.5, "No negative rows", ha="center", va="center")
    else:
        labels = top["item_nbr"].astype(str)
        plt.barh(labels[::-1], top["neg_rows"].iloc[::-1], color="#6b9e49")
        plt.xlabel("negative row count")
        plt.ylabel("item_nbr")
    plt.tight_layout()
    plt.savefig(out_dir / "dataset5_top_negative_items_bar.png", dpi=160)
    plt.close()


def scan_train(
    train_path: Path,
    columns: List[str],
    total_rows_precount: int,
    sales_col: str,
    item_col: str,
    date_col: Optional[str],
    store_col: Optional[str],
    chunksize: int,
) -> Dict[str, Any]:
    usecols = [sales_col, item_col]
    for optional in [date_col, store_col, "id", "onpromotion"]:
        if optional and optional in columns and optional not in usecols:
            usecols.append(optional)

    total_by_item: Dict[Any, int] = defaultdict(int)
    neg_by_item: Dict[Any, int] = defaultdict(int)
    neg_sum_by_item: Dict[Any, float] = defaultdict(float)
    neg_min_by_item: Dict[Any, float] = {}
    neg_date_rows: Dict[Any, int] = defaultdict(int)
    neg_date_items: Dict[Any, set] = defaultdict(set)
    neg_store_rows: Dict[Any, int] = defaultdict(int)
    neg_store_items: Dict[Any, set] = defaultdict(set)
    all_items = set()
    neg_items = set()
    neg_values_parts: List[pd.Series] = []
    top_records = pd.DataFrame()
    total_rows = 0
    neg_rows_total = 0

    dtype_map = {sales_col: "float64"}
    for col in [item_col, store_col]:
        if col:
            dtype_map[col] = "Int64"

    reader = pd.read_csv(train_path, usecols=usecols, chunksize=chunksize, low_memory=False)
    for chunk_no, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        chunk[sales_col] = pd.to_numeric(chunk[sales_col], errors="coerce")

        item_counts = chunk.groupby(item_col, dropna=False).size()
        update_count(total_by_item, item_counts)
        all_items.update(chunk[item_col].dropna().unique().tolist())

        neg = chunk[chunk[sales_col] < 0].copy()
        if neg.empty:
            if chunk_no % 10 == 0:
                print(f"[scan] chunks={chunk_no} rows={total_rows:,} neg_rows={neg_rows_total:,}")
            continue

        neg_rows_total += len(neg)
        neg_items.update(neg[item_col].dropna().unique().tolist())
        neg_values_parts.append(neg[sales_col].astype("float64"))

        update_count(neg_by_item, neg.groupby(item_col, dropna=False).size())
        update_sum(neg_sum_by_item, neg.groupby(item_col, dropna=False)[sales_col].sum())
        for key, value in neg.groupby(item_col, dropna=False)[sales_col].min().items():
            f_value = float(value)
            if key not in neg_min_by_item or f_value < neg_min_by_item[key]:
                neg_min_by_item[key] = f_value

        if date_col:
            update_count(neg_date_rows, neg.groupby(date_col, dropna=False).size())
            for key, values in neg.groupby(date_col, dropna=False)[item_col].unique().items():
                neg_date_items[key].update(pd.Series(values).dropna().tolist())

        if store_col:
            update_count(neg_store_rows, neg.groupby(store_col, dropna=False).size())
            for key, values in neg.groupby(store_col, dropna=False)[item_col].unique().items():
                neg_store_items[key].update(pd.Series(values).dropna().tolist())

        candidate_top = neg.nsmallest(50, sales_col)
        top_records = pd.concat([top_records, candidate_top], ignore_index=True).nsmallest(50, sales_col)
        if chunk_no % 10 == 0:
            print(f"[scan] chunks={chunk_no} rows={total_rows:,} neg_rows={neg_rows_total:,}")

    if total_rows != total_rows_precount:
        print(f"[warn] parsed row count {total_rows:,} differs from pre-count {total_rows_precount:,}")

    item_rows = []
    for item, neg_count in neg_by_item.items():
        total_count = total_by_item.get(item, 0)
        item_rows.append(
            {
                "item_nbr": item,
                "total_rows": int(total_count),
                "neg_rows": int(neg_count),
                "neg_ratio": float(neg_count / total_count) if total_count else math.nan,
                "min_negative_unit_sales": float(neg_min_by_item.get(item, math.nan)),
                "mean_negative_unit_sales": float(neg_sum_by_item[item] / neg_count) if neg_count else math.nan,
            }
        )
    negative_items = pd.DataFrame(item_rows)
    if not negative_items.empty:
        negative_items = negative_items.sort_values(["neg_rows", "neg_ratio"], ascending=[False, False])

    by_date = pd.DataFrame(
        [
            {"date": key, "neg_rows": rows, "neg_item_count": len(neg_date_items.get(key, set()))}
            for key, rows in neg_date_rows.items()
        ]
    )
    if not by_date.empty:
        by_date = by_date.sort_values("date")

    by_store = pd.DataFrame(
        [
            {"store_nbr": key, "neg_rows": rows, "neg_item_count": len(neg_store_items.get(key, set()))}
            for key, rows in neg_store_rows.items()
        ]
    )
    if not by_store.empty:
        by_store = by_store.sort_values("store_nbr")

    neg_values = pd.concat(neg_values_parts, ignore_index=True) if neg_values_parts else pd.Series(dtype="float64")
    return {
        "total_rows": int(total_rows),
        "total_item_count": int(len(all_items)),
        "neg_rows_total": int(neg_rows_total),
        "neg_item_count": int(len(neg_items)),
        "negative_items": negative_items,
        "top_records": top_records,
        "by_date": by_date,
        "by_store": by_store,
        "neg_values": neg_values,
    }


def load_item_metadata(train_path: Path, item_col: str) -> Optional[pd.DataFrame]:
    for candidate in [train_path.parent / "items.csv", train_path.parent / "item.csv"]:
        if candidate.is_file():
            meta = pd.read_csv(candidate)
            if item_col in meta.columns:
                return meta
            if "item_nbr" in meta.columns and item_col != "item_nbr":
                return meta.rename(columns={"item_nbr": item_col})
    return None


def build_metadata_tables(negative_items: pd.DataFrame, item_meta: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if item_meta is None or negative_items.empty:
        return {}
    merged = negative_items.merge(item_meta, on="item_nbr", how="left")
    tables = {}
    for col in ["family", "class", "perishable"]:
        if col in merged.columns:
            tables[col] = (
                merged.groupby(col, dropna=False)
                .agg(
                    neg_rows=("neg_rows", "sum"),
                    neg_item_count=("item_nbr", "nunique"),
                    mean_item_neg_ratio=("neg_ratio", "mean"),
                    max_item_neg_ratio=("neg_ratio", "max"),
                )
                .reset_index()
                .sort_values("neg_rows", ascending=False)
            )
    return tables


def determine_nature(
    total_rows: int,
    total_item_count: int,
    neg_rows_total: int,
    neg_item_count: int,
    item_neg_desc: pd.Series,
    neg_unit_desc: pd.Series,
) -> Tuple[str, List[str]]:
    row_ratio = neg_rows_total / total_rows if total_rows else 0.0
    item_ratio = neg_item_count / total_item_count if total_item_count else 0.0
    max_item_ratio = float(item_neg_desc.get("max", 0) or 0)
    median_neg = float(neg_unit_desc.get("50%", math.nan))
    min_neg = float(neg_unit_desc.get("min", math.nan))
    reasons = [
        f"负值行占比 {row_ratio:.6%}",
        f"涉及 item 占比 {item_ratio:.6%}",
        f"单 item 最高负值行占比 {max_item_ratio:.6%}",
    ]
    sparse = row_ratio < 0.01
    broad = item_ratio >= 0.5
    concentrated = item_ratio < 0.1 or max_item_ratio >= 0.05
    extreme = bool(not math.isnan(min_neg) and not math.isnan(median_neg) and abs(min_neg) >= max(100.0, abs(median_neg) * 50))
    if extreme:
        reasons.append(f"最小负值 {min_neg:g} 远低于负值中位数 {median_neg:g}")
    if sparse and broad and extreme:
        nature = "稀疏广泛型退货/冲销记录，夹杂极端异常值"
    elif sparse and broad:
        nature = "稀疏广泛型 / 可接受的退货记录型"
    elif concentrated and extreme:
        nature = "少数集中型，且存在极端异常值"
    elif concentrated:
        nature = "少数集中型"
    elif extreme:
        nature = "极端异常型"
    else:
        nature = "可接受的退货记录型"
    return nature, reasons


def write_error_report(out_dir: Path, message: str, scanned: List[Dict[str, Any]]) -> None:
    scanned_df = pd.DataFrame(scanned)
    write_csv(scanned_df, out_dir / "dataset5_scanned_files.csv")
    md = [
        "# Dataset5 Negative Unit Sales Check - Error",
        "",
        message,
        "",
        "## 已扫描文件与字段",
        "",
        markdown_table(scanned_df, max_rows=200),
    ]
    (out_dir / "dataset5_negative_unit_sales_summary.md").write_text("\n".join(md), encoding="utf-8")


def write_summary(
    out_dir: Path,
    train_path: Path,
    columns: List[str],
    row_count: int,
    adopted: Dict[str, str],
    stats: Dict[str, Any],
    item_neg_desc: pd.Series,
    neg_unit_desc: pd.Series,
    code_hits: pd.DataFrame,
    code_handling: Dict[str, Any],
    metadata_tables: Dict[str, pd.DataFrame],
    nature: str,
    nature_reasons: List[str],
) -> None:
    total_rows = stats["total_rows"]
    total_items = stats["total_item_count"]
    neg_rows = stats["neg_rows_total"]
    neg_items = stats["neg_item_count"]
    row_ratio = neg_rows / total_rows if total_rows else math.nan
    item_ratio = neg_items / total_items if total_items else math.nan
    core = pd.DataFrame(
        [
            {"metric": "train_total_rows", "value": total_rows},
            {"metric": "precount_rows", "value": row_count},
            {"metric": "item_nbr_total_count", "value": total_items},
            {"metric": "negative_unit_sales_rows", "value": neg_rows},
            {"metric": "negative_row_ratio", "value": row_ratio},
            {"metric": "negative_item_count", "value": neg_items},
            {"metric": "negative_item_ratio", "value": item_ratio},
        ]
    )
    top_items = stats["negative_items"].sort_values("neg_rows", ascending=False).head(20)
    top_ratio_items = stats["negative_items"].sort_values("neg_ratio", ascending=False).head(20)
    top_dates = stats["by_date"].sort_values("neg_rows", ascending=False).head(20) if not stats["by_date"].empty else pd.DataFrame()

    md = [
        "# Dataset5 / Favorita unit_sales 负值专项检查",
        "",
        "## D5 数据文件来源",
        "",
        f"- train 文件: `{train_path}`",
        f"- 字段名: `{columns}`",
        f"- 读取前行数统计: `{row_count}`",
        f"- 最终采用销量字段: `{adopted['sales_col']}` ({adopted['sales_reason']})",
        f"- 最终采用商品字段: `{adopted['item_col']}` ({adopted['item_reason']})",
        f"- 日期字段: `{adopted.get('date_col') or '不存在'}`",
        f"- 门店字段: `{adopted.get('store_col') or '不存在'}`",
        "",
        "## train 数据规模与负值总体统计",
        "",
        markdown_table(core),
        "",
        "## 每个 item 的负值行占比统计",
        "",
        markdown_table(item_neg_desc.reset_index().rename(columns={"index": "metric", 0: "value"})),
        "",
        "## 负值 unit_sales 分布统计",
        "",
        markdown_table(neg_unit_desc.reset_index().rename(columns={"index": "metric", 0: "value"})),
        "",
        "## 负值最多的 top 20 item_nbr",
        "",
        markdown_table(top_items),
        "",
        "## 负值占自身总记录比例最高的 top 20 item_nbr",
        "",
        markdown_table(top_ratio_items),
    ]
    if not top_dates.empty:
        md.extend(["", "## 负值最多的 top 20 日期", "", markdown_table(top_dates)])
    if not stats["by_store"].empty:
        md.extend(["", "## 按门店分布", "", markdown_table(stats["by_store"])])
    if metadata_tables:
        md.extend(["", "## 商品元信息集中性分析"])
        for name, table in metadata_tables.items():
            md.extend(["", f"### {name}", "", markdown_table(table, max_rows=30)])
    else:
        md.extend(["", "## 商品元信息集中性分析", "", "未发现可合并的 `items.csv` 元信息，或字段不足。"])

    md.extend(
        [
            "",
            "## 当前代码对负值的处理方式",
            "",
            code_handling["summary"],
            "",
            "- `data_preprocessing.py::load_dataset` 当前只支持 Dataset1/2/3；Dataset5/Favorita 不在主 `dataset_registry.py` 中。",
            "- `scripts/scan_dataset5_favorita_cold_start.py::scan_sales_train` 读取 `unit_sales` 并统计有效/零值/缺失，但没有对 `< 0` 做删除、clip 或特殊标记。",
            "- `normalize_features` 使用 `MinMaxScaler`，如果未来把 D5 标准化成 `sales` 后直接进入主流水线，负值会被保留进归一化范围。",
            "- 未发现直接的 `np.log1p(unit_sales)` / `log1p(sales)` 命中，因此当前代码层面没有明确的 log1p 负值 NaN 风险；但若新增 log1p 变换，需要先处理负值。",
            "",
            "### 相关源码命中",
            "",
            markdown_table(code_hits.head(80) if not code_hits.empty else code_hits, max_rows=80),
            "",
            "## 风险判断",
            "",
            f"结论: **{nature}**。",
        ]
    )
    md.extend([f"- {reason}" for reason in nature_reasons])
    high_risk = stats["negative_items"][stats["negative_items"]["neg_ratio"] >= 0.05]
    md.extend(
        [
            f"- 高风险 item 定义为 `neg_ratio >= 5%`，本次命中 {len(high_risk)} 个。",
            "",
            "## 对后续实验的建议",
            "",
            "- 不建议把 Favorita 负销量简单视作 M5 风格的销量字段异常；它更符合退货/冲销语义，需要 D5 专属处理策略。",
            "- 主实验若未来纳入 Dataset5，应显式记录策略：保留负值、clip 到 0、或单独增加退货标记，而不要隐式沿用 Dataset1/2/3 的 `sales` 逻辑。",
            "- 极端负值记录建议在训练前单独做敏感性实验：保留原值 vs clip 到 0 vs winsorize，并报告指标差异。",
            "- 如果模型目标不允许负需求，建议新增 `returned_units = abs(min(unit_sales, 0))` 和 `net_unit_sales = max(unit_sales, 0)` 两列，再比较实验表现。",
        ]
    )
    (out_dir / "dataset5_negative_unit_sales_summary.md").write_text("\n".join(md), encoding="utf-8")


def run(root: Path, output_root: Path, chunksize: int) -> Path:
    out_dir = make_output_dir(output_root, "dataset5_negative_unit_sales_check")
    candidates, scanned = find_dataset_candidates(root)
    if not candidates:
        write_error_report(out_dir, "未能找到包含销量字段和商品字段的 D5/Favorita train 数据文件。", scanned)
        print(f"[error] report written to: {out_dir}")
        return out_dir

    chosen = candidates[0]
    train_path = Path(chosen["path"])
    columns = list(chosen["columns"])
    sales_col, sales_reason = choose_column(columns, "unit_sales", SALES_CANDIDATES)
    item_col, item_reason = choose_column(columns, "item_nbr", ITEM_CANDIDATES)
    date_col, _ = choose_column(columns, "date", DATE_CANDIDATES)
    store_col, _ = choose_column(columns, "store_nbr", STORE_CANDIDATES)
    if not sales_col or not item_col:
        write_error_report(
            out_dir,
            f"找到候选 train `{train_path}`，但无法识别 unit_sales/item_nbr 字段。",
            scanned,
        )
        print(f"[error] report written to: {out_dir}")
        return out_dir

    row_count = count_csv_rows(train_path) if train_path.suffix.lower() == ".csv" else -1
    print(f"D5 train file path: {train_path}")
    print(f"D5 train columns: {columns}")
    print(f"D5 train row count before dataframe read: {row_count:,}")
    print(f"Adopted sales field: {sales_col} ({sales_reason})")
    print(f"Adopted item field: {item_col} ({item_reason})")

    stats = scan_train(
        train_path=train_path,
        columns=columns,
        total_rows_precount=row_count,
        sales_col=sales_col,
        item_col=item_col,
        date_col=date_col,
        store_col=store_col,
        chunksize=chunksize,
    )
    negative_items = stats["negative_items"]
    item_neg_ratio = negative_items["neg_ratio"] if not negative_items.empty else pd.Series(dtype="float64")
    item_neg_desc = describe_series(item_neg_ratio)
    neg_unit_desc = describe_series(stats["neg_values"])
    item_meta = load_item_metadata(train_path, item_col)
    metadata_tables = build_metadata_tables(negative_items, item_meta)

    write_csv(negative_items, out_dir / "dataset5_negative_items.csv")
    write_csv(stats["top_records"], out_dir / "dataset5_negative_top_records.csv")
    if not stats["by_date"].empty:
        write_csv(stats["by_date"], out_dir / "dataset5_negative_by_date.csv")
    if not stats["by_store"].empty:
        write_csv(stats["by_store"], out_dir / "dataset5_negative_by_store.csv")
    for name, table in metadata_tables.items():
        write_csv(table, out_dir / f"dataset5_negative_by_{name}.csv")

    plot_outputs(negative_items, stats["neg_values"], out_dir)
    code_hits = audit_code(root, Path(__file__))
    write_csv(code_hits, out_dir / "dataset5_negative_code_audit_hits.csv")
    code_handling = classify_code_handling(code_hits)
    nature, reasons = determine_nature(
        total_rows=stats["total_rows"],
        total_item_count=stats["total_item_count"],
        neg_rows_total=stats["neg_rows_total"],
        neg_item_count=stats["neg_item_count"],
        item_neg_desc=item_neg_desc,
        neg_unit_desc=neg_unit_desc,
    )
    write_summary(
        out_dir=out_dir,
        train_path=train_path,
        columns=columns,
        row_count=row_count,
        adopted={
            "sales_col": sales_col,
            "sales_reason": sales_reason,
            "item_col": item_col,
            "item_reason": item_reason,
            "date_col": date_col or "",
            "store_col": store_col or "",
        },
        stats=stats,
        item_neg_desc=item_neg_desc,
        neg_unit_desc=neg_unit_desc,
        code_hits=code_hits,
        code_handling=code_handling,
        metadata_tables=metadata_tables,
        nature=nature,
        nature_reasons=reasons,
    )

    neg_row_ratio = stats["neg_rows_total"] / stats["total_rows"] if stats["total_rows"] else math.nan
    neg_item_ratio = stats["neg_item_count"] / stats["total_item_count"] if stats["total_item_count"] else math.nan
    print("\n=== Dataset5 negative unit_sales terminal summary ===")
    print(f"D5 train 文件路径: {train_path}")
    print(f"train 总行数: {stats['total_rows']:,}")
    print(f"item 总数: {stats['total_item_count']:,}")
    print(f"unit_sales < 0 总行数: {stats['neg_rows_total']:,}")
    print(f"负值行占比: {neg_row_ratio:.8%}")
    print(f"出现负值的 item 数: {stats['neg_item_count']:,}")
    print(f"出现负值的 item 占比: {neg_item_ratio:.8%}")
    print("\nitem_neg_ratio.describe():")
    print(item_neg_desc.to_string())
    print("\nneg_rows['unit_sales'].describe():")
    print(neg_unit_desc.to_string())
    print(f"\n当前代码是否处理负值: {'是' if code_handling['handles_negative_unit_sales'] else '否'}")
    print(f"风险判断: {nature}")
    print(f"报告输出路径: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Dataset5/Favorita negative unit_sales records.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root to scan.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Output root.")
    parser.add_argument("--chunksize", type=int, default=5_000_000, help="CSV chunk size.")
    args = parser.parse_args()
    run(args.root.resolve(), args.output_root, args.chunksize)


if __name__ == "__main__":
    main()
