#!/usr/bin/env python3
"""
数据集档案扫描工具 — 极简配置，自动推断列名与源/目标域划分。

用法:
  python3 scripts/scan_dataset_profiles.py --config configs/dataset_profile_scan_config.json --verbose
  python3 scripts/scan_dataset_profiles.py --config configs/dataset_profile_scan_config.json --dataset Dataset1 --verbose
  python3 scripts/scan_dataset_profiles.py --config configs/dataset_profile_scan_config.json --infer-only --verbose
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
DATE_KEYWORDS = ("date", "日期", "day", "time", "ds", "dt", "calendar")
SALES_KEYWORDS = (
    "sales", "sale", "demand", "qty", "quantity", "units", "volume",
    "target", "y", "销量", "销售", "需求", "数量",
)
SALES_EXCLUDE_KEYWORDS = (
    "id", "store_id", "item_id", "sku_id", "dept", "category", "price",
    "promo", "holiday", "open", "customer", "key", "index", "code",
)
ENTITY_KEYWORDS = (
    "store", "shop", "item", "product", "sku", "goods", "dept", "brand",
    "warehouse", "node", "entity", "门店", "店铺", "商品", "sku", "品类", "仓库",
)
STORE_KEYWORDS = ("store", "shop", "warehouse", "门店", "店铺", "仓库", "node")
ITEM_KEYWORDS = ("item", "product", "sku", "goods", "商品")
CONFIDENCE_THRESHOLD = 0.70
SOURCE_TOP_K = 20


# ---------------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if "datasets" not in cfg or not cfg["datasets"]:
        raise ValueError("Config must contain a non-empty 'datasets' list.")
    normalized_datasets: List[Dict[str, str]] = []
    for ds in cfg["datasets"]:
        if "name" not in ds or "path" not in ds:
            raise ValueError("Each dataset entry requires 'name' and 'path' only.")
        normalized_datasets.append({"name": str(ds["name"]), "path": str(ds["path"])})
    cfg["datasets"] = normalized_datasets
    cfg.setdefault("output_root", "outputs/dataset_profiles")
    cfg.setdefault("cold_start_window", {"train_days": 15, "val_days": 15, "test_days": 180})
    return cfg


class ScanLogger:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self.warnings: List[str] = []
        self.lines: List[str] = []

    def info(self, msg: str) -> None:
        line = f"[INFO] {msg}"
        self.lines.append(line)
        if self.verbose:
            print(line)

    def warn(self, msg: str) -> None:
        line = f"[WARN] {msg}"
        self.warnings.append(msg)
        self.lines.append(line)
        if self.verbose:
            print(line)

    def error(self, msg: str) -> None:
        line = f"[ERROR] {msg}"
        self.lines.append(line)
        if self.verbose:
            print(line)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _is_sample_file(path: Path) -> bool:
    lower = path.name.lower()
    return "sample" in lower or "sample_" in lower


def _resolve_data_file(path: Path, logger: ScanLogger, allow_sample: bool = False) -> Path:
    if path.is_file():
        if path.suffix.lower() == ".ipynb":
            sibling_csv = path.parent / "(Dataset 3.csv"
            if sibling_csv.is_file():
                logger.info(f"Notebook path given; using sibling CSV: {sibling_csv}")
                return sibling_csv
            for candidate in sorted(path.parent.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True):
                logger.info(f"Notebook path given; using CSV in same dir: {candidate}")
                return candidate
            raise FileNotFoundError(f"No readable data file found near notebook: {path}")
        if _is_sample_file(path) and not allow_sample:
            raise FileNotFoundError(
                f"Refusing to use sample file for formal scan: {path}. "
                "Pass --sample-ok only for explicit sample-mode diagnostics."
            )
        if path.suffix.lower() in SUPPORTED_EXTENSIONS or path.suffix.lower() == ".7z":
            return path
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    nested_data = path / "data"
    preferred_locations = []
    for name in ("train.parquet", "train.csv", "train.xlsx", "train.xls", "data.parquet", "data.csv", "sales.csv"):
        if nested_data.is_dir():
            preferred_locations.append(nested_data / name)
        preferred_locations.append(path / name)

    for candidate in preferred_locations:
        if candidate.is_file() and (allow_sample or not _is_sample_file(candidate)):
            return candidate

    all_files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        all_files.extend(path.rglob(f"*{ext}"))
    formal_files = [
        f for f in all_files
        if "sample_submission" not in f.name.lower()
        and (f.name.lower().startswith("train") or "test" not in f.name.lower())
        and not _is_sample_file(f)
    ]
    if formal_files:
        formal_files.sort(
            key=lambda p: (
                p.suffix.lower() == ".parquet",
                p.name.lower().startswith("train"),
                p.stat().st_size,
            ),
            reverse=True,
        )
        logger.info(f"Auto-selected full data file: {formal_files[0]}")
        return formal_files[0]

    sample_files = [
        f for f in all_files
        if "sample_submission" not in f.name.lower()
        and (f.name.lower().startswith("train") or "test" not in f.name.lower())
        and _is_sample_file(f)
    ]
    if sample_files:
        sample_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        if allow_sample:
            logger.warn(
                f"SAMPLE_ONLY: using sample file {sample_files[0]}; "
                "INVALID_FOR_FORMAL_EXPERIMENT."
            )
            return sample_files[0]
        raise FileNotFoundError(
            f"Only sample files found under {path}; refusing formal scan. "
            "Pass --sample-ok only for explicit sample-mode diagnostics."
        )

    if not formal_files:
        compressed = list(path.rglob("*.7z")) + list(path.rglob("*.zip"))
        if compressed:
            raise FileNotFoundError(
                f"Only compressed archives found under {path}; please extract first: "
                f"{[str(c) for c in compressed[:3]]}"
            )
        raise FileNotFoundError(f"No supported data files found under directory: {path}")
    raise FileNotFoundError(f"No formal data file found under directory: {path}")


def _read_tabular_file(file_path: Path, logger: ScanLogger) -> pd.DataFrame:
    ext = file_path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(file_path, low_memory=False)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    if ext == ".parquet":
        try:
            return pd.read_parquet(file_path)
        except ImportError as exc:
            raise ImportError(
                "Parquet support requires pyarrow or fastparquet. "
                "Install with: pip install pyarrow"
            ) from exc
    raise ValueError(f"Unsupported extension: {ext}")


def _detect_wide_qty_columns(columns: Sequence[str]) -> List[str]:
    return [c for c in columns if re.match(r"(?i)^QTY_", str(c))]


def _melt_wide_sales(df: pd.DataFrame, date_col: str, logger: ScanLogger) -> pd.DataFrame:
    qty_cols = _detect_wide_qty_columns(df.columns)
    if not qty_cols:
        return df

    logger.info(f"Detected wide-format QTY columns ({len(qty_cols)}); melting to long format.")
    long_df = df[[date_col] + qty_cols].melt(
        id_vars=[date_col], var_name="_qty_key", value_name="_sales_melted"
    )
    meta = long_df["_qty_key"].str.extract(r"(?i)QTY_([^_]+)_(\d+)")
    long_df["_entity_part"] = meta[0]
    long_df["_item_part"] = pd.to_numeric(meta[1], errors="coerce")
    long_df = long_df.drop(columns=["_qty_key"])
    return long_df


def load_dataset(path_str: str, logger: ScanLogger, allow_sample: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "raw_path": path_str,
        "resolved_path": None,
        "load_notes": [],
        "sample_only": False,
        "formal_validity": "FORMAL_CANDIDATE",
    }
    path = Path(path_str).expanduser()
    data_file = _resolve_data_file(path, logger, allow_sample=allow_sample)
    meta["resolved_path"] = str(data_file)
    meta["sample_only"] = _is_sample_file(data_file)
    if meta["sample_only"]:
        meta["formal_validity"] = "INVALID_FOR_FORMAL_EXPERIMENT"
        meta["load_notes"].append("SAMPLE_ONLY: file name contains sample; invalid for formal experiment.")
        logger.warn("SAMPLE_ONLY: INVALID_FOR_FORMAL_EXPERIMENT; sample file is not valid for formal dataset profiling.")

    if data_file.suffix.lower() == ".7z":
        raise FileNotFoundError(f"Compressed file must be extracted before scanning: {data_file}")

    raw_df = _read_tabular_file(data_file, logger)
    logger.info(f"Loaded {len(raw_df)} rows x {len(raw_df.columns)} cols from {data_file.name}")
    meta["raw_columns"] = list(raw_df.columns)
    meta["raw_shape"] = raw_df.shape
    return raw_df, meta


# ---------------------------------------------------------------------------
# Column inference
# ---------------------------------------------------------------------------

def _col_name_score(col: str, keywords: Sequence[str]) -> float:
    lower = str(col).lower()
    best = 0.0
    for kw in keywords:
        if kw.lower() == lower:
            best = max(best, 1.0)
        elif kw.lower() in lower:
            best = max(best, 0.85)
    return best


def _contains_array_like_values(series: pd.Series) -> bool:
    sample = series.dropna().head(20)
    return any(isinstance(v, (list, tuple, np.ndarray, dict)) for v in sample)


def _safe_nunique(series: pd.Series) -> int:
    if _contains_array_like_values(series):
        return -1
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return -1


def _datetime_convert_stats(series: pd.Series) -> Tuple[float, int]:
    sample = series.dropna().head(5000)
    if sample.empty:
        return 0.0, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        converted = pd.to_datetime(sample, errors="coerce", utc=False)
    success_rate = float(converted.notna().mean())
    unique_dates = int(converted.dropna().dt.normalize().nunique())
    return success_rate, unique_dates


def infer_date_col(df: pd.DataFrame) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    for col in df.columns:
        name_score = _col_name_score(col, DATE_KEYWORDS)
        success_rate, unique_dates = _datetime_convert_stats(df[col])
        numeric = pd.to_numeric(df[col].dropna().head(5000), errors="coerce")
        numeric_success = float(numeric.notna().mean()) if len(numeric) else 0.0
        looks_like_binary_flag = numeric_success > 0.95 and df[col].nunique(dropna=True) <= 2
        if name_score == 0 and success_rate < 0.8:
            continue
        composite = 0.45 * name_score + 0.35 * success_rate + 0.20 * min(unique_dates / 100.0, 1.0)
        if looks_like_binary_flag and str(col).lower() not in {"date", "dt", "ds", "日期"}:
            composite *= 0.25
        candidates.append({
            "col": col,
            "score": composite,
            "name_score": name_score,
            "success_rate": success_rate,
            "unique_dates": unique_dates,
        })

    if not candidates:
        return {
            "inferred_date_col": None,
            "date_col_confidence": 0.0,
            "date_col_reason": "No column matched date keywords or datetime conversion.",
        }

    candidates.sort(key=lambda x: (x["score"], x["success_rate"], x["unique_dates"], x["name_score"]), reverse=True)
    best = candidates[0]
    confidence = float(min(1.0, best["score"]))
    reason = (
        f"Selected '{best['col']}': name_score={best['name_score']:.2f}, "
        f"convert_success={best['success_rate']:.2f}, unique_dates={best['unique_dates']}"
    )
    return {
        "inferred_date_col": best["col"],
        "date_col_confidence": confidence,
        "date_col_reason": reason,
    }


def _is_sales_excluded(col: str) -> bool:
    lower = str(col).lower()
    if lower in {"id", "index", "entity_id", "item_id", "store_id", "sku_id"}:
        return True
    for kw in SALES_EXCLUDE_KEYWORDS:
        if kw in lower and not any(sk in lower for sk in ("sales", "qty", "quantity", "demand", "销量")):
            return True
    return False


def infer_sales_col(df: pd.DataFrame, date_col: Optional[str], entity_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    entity_cols = entity_cols or []
    exclude = {date_col, *entity_cols, "_sales_melted", "_entity_part", "_item_part"}
    candidates: List[Dict[str, Any]] = []

    for col in df.columns:
        if col in exclude or _is_sales_excluded(col):
            continue
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        valid_ratio = float(numeric.notna().mean())
        if valid_ratio < 0.5:
            continue
        non_neg_ratio = float((numeric >= 0).sum() / max(numeric.notna().sum(), 1))
        unique_count = int(numeric.dropna().nunique())
        is_constant = unique_count <= 1
        name_score = _col_name_score(col, SALES_KEYWORDS)
        if name_score == 0 and non_neg_ratio < 0.9:
            continue
        zero_ratio = float((numeric == 0).sum() / max(numeric.notna().sum(), 1))
        variability = 0.0 if is_constant else min(unique_count / 100.0, 1.0)
        composite = (
            0.40 * name_score + 0.25 * non_neg_ratio + 0.20 * valid_ratio
            + 0.10 * variability - 0.05 * max(zero_ratio - 0.95, 0)
        )
        candidates.append({
            "col": col,
            "score": composite,
            "name_score": name_score,
            "non_neg_ratio": non_neg_ratio,
            "unique_count": unique_count,
            "zero_ratio": zero_ratio,
        })

    if "_sales_melted" in df.columns:
        candidates.append({
            "col": "_sales_melted",
            "score": 0.95,
            "name_score": 1.0,
            "non_neg_ratio": 1.0,
            "unique_count": int(df["_sales_melted"].nunique()),
            "zero_ratio": float((df["_sales_melted"] == 0).mean()),
        })

    if not candidates:
        return {
            "inferred_sales_col": None,
            "sales_col_confidence": 0.0,
            "sales_col_reason": "No numeric column matched sales heuristics.",
        }

    candidates.sort(key=lambda x: (x["score"], x["name_score"], x["non_neg_ratio"], x["unique_count"]), reverse=True)
    best = candidates[0]
    confidence = float(min(1.0, max(0.0, best["score"])))
    reason = (
        f"Selected '{best['col']}': name_score={best['name_score']:.2f}, "
        f"non_neg_ratio={best['non_neg_ratio']:.2f}, unique_values={best['unique_count']}"
    )
    return {
        "inferred_sales_col": best["col"],
        "sales_col_confidence": confidence,
        "sales_col_reason": reason,
    }


def _entity_col_score(col: str, series: pd.Series, date_col: Optional[str], sales_col: Optional[str]) -> float:
    if col in (date_col, sales_col):
        return 0.0
    if _contains_array_like_values(series):
        return 0.0
    name_score = _col_name_score(col, ENTITY_KEYWORDS)
    nunique = _safe_nunique(series)
    if nunique < 0:
        return 0.0
    n = len(series)
    if nunique <= 1 or nunique >= max(n * 0.9, 2):
        cardinality_score = 0.1
    elif 2 <= nunique <= 50000:
        cardinality_score = 0.8
    else:
        cardinality_score = 0.5
    return 0.6 * name_score + 0.4 * cardinality_score


def infer_entity_cols(
    df: pd.DataFrame,
    date_col: Optional[str],
    sales_col: Optional[str],
) -> Dict[str, Any]:
    if "_entity_part" in df.columns and "_item_part" in df.columns:
        return {
            "inferred_entity_cols": ["_entity_part", "_item_part"],
            "entity_col_confidence": 0.92,
            "entity_col_reason": "Wide-format QTY columns parsed into entity/item parts.",
            "entity_inference_status": "OK",
        }

    scored: List[Tuple[str, float, str]] = []
    for col in df.columns:
        if col in (date_col, sales_col) or col.startswith("_"):
            continue
        score = _entity_col_score(col, df[col], date_col, sales_col)
        if score > 0.2:
            category = "store" if _col_name_score(col, STORE_KEYWORDS) > 0 else (
                "item" if _col_name_score(col, ITEM_KEYWORDS) > 0 else "generic"
            )
            scored.append((col, score, category))

    scored.sort(key=lambda x: x[1], reverse=True)
    store_cols = [c for c, _, cat in scored if cat == "store"]
    item_cols = [c for c, _, cat in scored if cat == "item"]
    generic_cols = [c for c, _, cat in scored if cat == "generic"]

    chosen: List[str] = []
    reason_parts: List[str] = []

    if store_cols and item_cols:
        chosen = [store_cols[0], item_cols[0]]
        reason_parts.append(f"store+item combo: {chosen}")
    elif item_cols:
        chosen = [item_cols[0]]
        reason_parts.append(f"item column: {item_cols[0]}")
    elif store_cols:
        chosen = [store_cols[0]]
        reason_parts.append(f"store column: {store_cols[0]}")
    elif generic_cols:
        chosen = [generic_cols[0]]
        reason_parts.append(f"generic entity column: {generic_cols[0]}")

    if not chosen:
        return {
            "inferred_entity_cols": [],
            "entity_col_confidence": 0.3,
            "entity_col_reason": "No entity column detected; will use global_entity fallback.",
            "entity_inference_status": "FALLBACK_GLOBAL_ENTITY",
        }

    avg_score = float(np.mean([s for c, s, _ in scored if c in chosen]))
    confidence = min(1.0, avg_score)
    return {
        "inferred_entity_cols": chosen,
        "entity_col_confidence": confidence,
        "entity_col_reason": "; ".join(reason_parts) if reason_parts else "Heuristic entity selection.",
        "entity_inference_status": "OK",
    }


def infer_columns(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    date_info = infer_date_col(df)
    date_col = date_info["inferred_date_col"]

    entity_info = infer_entity_cols(df, date_col, None)
    entity_cols = entity_info.get("inferred_entity_cols") or []

    sales_info = infer_sales_col(df, date_col, entity_cols)
    sales_col = sales_info["inferred_sales_col"]

    if not entity_info.get("inferred_entity_cols") and sales_col:
        entity_info = infer_entity_cols(df, date_col, sales_col)
        entity_cols = entity_info.get("inferred_entity_cols") or []

    result = {
        "dataset_name": dataset_name,
        **date_info,
        **sales_info,
        **entity_info,
    }

    low_conf = any(
        result.get(k, 1.0) < CONFIDENCE_THRESHOLD
        for k in ("date_col_confidence", "sales_col_confidence", "entity_col_confidence")
    )
    result["inferred_status"] = "LOW_CONFIDENCE" if low_conf else "OK"
    result["needs_manual_review"] = low_conf
    return result


def build_entity_id(row: pd.Series, entity_cols: List[str]) -> str:
    if not entity_cols:
        return "global_entity"
    parts = []
    for col in entity_cols:
        val = row[col]
        if pd.isna(val):
            val = "NA"
        parts.append(f"{col}={val}")
    return "|".join(parts)


def build_entity_id_series(df: pd.DataFrame, entity_cols: List[str]) -> pd.Series:
    if not entity_cols:
        return pd.Series("global_entity", index=df.index)

    entity_id: Optional[pd.Series] = None
    for col in entity_cols:
        values = df[col].where(df[col].notna(), "NA").astype(str)
        part = col + "=" + values
        entity_id = part if entity_id is None else entity_id + "|" + part
    return entity_id if entity_id is not None else pd.Series("global_entity", index=df.index)


def apply_inferred_schema(
    df: pd.DataFrame,
    inference: Dict[str, Any],
    logger: ScanLogger,
) -> pd.DataFrame:
    work = df.copy()
    date_col = inference.get("inferred_date_col")
    sales_col = inference.get("inferred_sales_col")
    entity_cols = inference.get("inferred_entity_cols") or []

    if date_col and date_col in work.columns:
        work["_date"] = pd.to_datetime(work[date_col], errors="coerce")
        bad_dates = int(work["_date"].isna().sum())
        if bad_dates:
            logger.warn(f"Date column '{date_col}': {bad_dates} values could not be parsed.")

    if sales_col and sales_col in work.columns:
        before_na = int(work[sales_col].isna().sum()) if sales_col in work.columns else 0
        work["_sales"] = pd.to_numeric(work[sales_col], errors="coerce")
        new_na = int(work["_sales"].isna().sum()) - before_na
        if new_na > 0:
            logger.warn(f"Sales column '{sales_col}': {new_na} values coerced to NaN.")

    if entity_cols:
        work["_entity_id"] = build_entity_id_series(work, entity_cols)
    else:
        work["_entity_id"] = "global_entity"
        inference.setdefault("entity_inference_status", "FALLBACK_GLOBAL_ENTITY")

    return work


# ---------------------------------------------------------------------------
# Source / target inference
# ---------------------------------------------------------------------------

def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _find_dataset_protocol_in_mapping(obj: Any, dataset_name: str) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        direct = obj.get(dataset_name)
        if isinstance(direct, dict):
            return dict(direct)
        for value in obj.values():
            found = _find_dataset_protocol_in_mapping(value, dataset_name)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_dataset_protocol_in_mapping(item, dataset_name)
            if found:
                return found
    return None


def _extract_protocol_from_python_text(text: str, dataset_name: str) -> Optional[Dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key_node, value_node in zip(node.keys, node.values):
            if not isinstance(key_node, ast.Constant) or key_node.value != dataset_name:
                continue
            try:
                value = ast.literal_eval(value_node)
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, dict):
                return dict(value)
    return None


def _iter_project_protocol_candidate_files() -> List[Path]:
    candidates: List[Path] = []
    explicit_files = [
        PROJECT_ROOT / "config.yaml",
        PROJECT_ROOT / "config.yml",
        PROJECT_ROOT / "config.py",
        PROJECT_ROOT / "run_full_paper_experiments.py",
        PROJECT_ROOT / "data_preprocessing.py",
        PROJECT_ROOT / "scripts" / "run_full_paper_experiments.py",
    ]
    candidates.extend([p for p in explicit_files if p.is_file()])

    for dirname in ("configs", "scripts", "src"):
        root = PROJECT_ROOT / dirname
        if not root.is_dir():
            continue
        for ext in ("*.json", "*.yaml", "*.yml", "*.py"):
            candidates.extend(root.rglob(ext))

    seen: set[Path] = set()
    unique: List[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _search_protocol_in_project_files(dataset_name: str) -> Optional[Dict[str, Any]]:
    for path in _iter_project_protocol_candidate_files():
        try:
            if path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        lower_text = text.lower()
        if dataset_name not in text or ("source" not in lower_text and "target" not in lower_text):
            continue

        try:
            if path.suffix.lower() == ".json":
                found = _find_dataset_protocol_in_mapping(json.loads(text), dataset_name)
                if found:
                    return found
            elif path.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml
                except ImportError:
                    continue
                found = _find_dataset_protocol_in_mapping(yaml.safe_load(text), dataset_name)
                if found:
                    return found
            elif path.suffix.lower() == ".py":
                found = _extract_protocol_from_python_text(text, dataset_name)
                if found:
                    return found
        except Exception:
            continue
    return None


def _search_strict_protocol_in_project(dataset_name: str) -> Optional[Dict[str, Any]]:
    try:
        from src.data_processing.data_preprocessing import STRICT_DATASET_PROTOCOL
        if dataset_name in STRICT_DATASET_PROTOCOL:
            return dict(STRICT_DATASET_PROTOCOL[dataset_name])
    except ImportError:
        pass

    cfg_path = PROJECT_ROOT / "configs" / "default_config.json"
    cfg = _load_json_if_exists(cfg_path)
    if cfg:
        proto = cfg.get("paper_reproduction", {}).get("strict_dataset_protocol", {})
        if isinstance(proto, dict) and dataset_name in proto:
            return dict(proto[dataset_name])

    yaml_path = PROJECT_ROOT / "config.yaml"
    if yaml_path.is_file():
        try:
            import yaml
            with yaml_path.open("r", encoding="utf-8") as fh:
                ycfg = yaml.safe_load(fh)
            proto = (ycfg or {}).get("paper_reproduction", {}).get("strict_dataset_protocol", {})
            if isinstance(proto, dict) and dataset_name in proto:
                return dict(proto[dataset_name])
        except Exception:
            pass

    return _search_protocol_in_project_files(dataset_name)


def _map_protocol_to_entity_ids(
    dataset_name: str,
    protocol: Dict[str, Any],
    entity_cols: List[str],
    work_df: pd.DataFrame,
) -> Tuple[List[str], List[str], str]:
    """Map known project protocol to entity_id strings."""
    col_map = {c.lower(): c for c in entity_cols}

    def find_col(*aliases: str) -> Optional[str]:
        for alias in aliases:
            if alias in entity_cols:
                return alias
            if alias.lower() in col_map:
                return col_map[alias.lower()]
        return None

    store_col = find_col("entity_id", "store", "Store", "store_id", "_entity_part")
    item_col = find_col("item_id", "item", "Item", "product_id", "sku_id", "_item_part")

    target_entities: List[str] = []
    source_entities: List[str] = []

    if dataset_name == "Dataset1":
        te = protocol.get("target_entity_id", 1)
        ti = protocol.get("target_item_id", 10)
        allowed = protocol.get("allowed_entities", [1, 2, 3])
        source_items = protocol.get("source_item_ids", list(range(1, 10)))
        if store_col and item_col:
            target_entities = [build_entity_id(pd.Series({store_col: te, item_col: ti}), [store_col, item_col])]
            for ent in allowed:
                for it in source_items:
                    source_entities.append(
                        build_entity_id(pd.Series({store_col: ent, item_col: it}), [store_col, item_col])
                    )
        elif item_col:
            target_entities = [f"{item_col}={ti}"]
            source_entities = [f"{item_col}={it}" for it in source_items]
    elif dataset_name == "Dataset2":
        te = protocol.get("target_entity_id", "B1")
        ti = protocol.get("target_item_id", 10)
        if store_col and item_col:
            target_entities = [build_entity_id(pd.Series({store_col: te, item_col: ti}), [store_col, item_col])]
            if work_df is not None and not work_df.empty:
                all_ids = work_df["_entity_id"].unique()
                source_entities = [e for e in all_ids if e != target_entities[0]][:SOURCE_TOP_K]
        elif item_col:
            target_entities = [f"{item_col}={ti}"]
    elif dataset_name == "Dataset3":
        ts = protocol.get("target_store_id", 10)
        col = item_col or store_col or (entity_cols[0] if entity_cols else None)
        if col:
            target_entities = [f"{col}={ts}"]
            if work_df is not None and not work_df.empty:
                all_ids = work_df["_entity_id"].unique()
                source_entities = [e for e in all_ids if e != target_entities[0]][:SOURCE_TOP_K]

    reason = f"Loaded from project strict protocol for {dataset_name}: {protocol}"
    return source_entities, target_entities, reason


def _auto_candidate_source_target(
    work_df: pd.DataFrame,
    required_days: int,
    logger: ScanLogger,
) -> Tuple[List[str], List[str], float, str]:
    if "_date" not in work_df.columns or "_sales" not in work_df.columns:
        return [], [], 0.2, "Missing date/sales columns for auto source-target inference."

    entity_stats: List[Dict[str, Any]] = []
    for ent_id, grp in work_df.groupby("_entity_id"):
        dates = grp["_date"].dropna().sort_values()
        if dates.empty:
            continue
        span_days = (dates.max() - dates.min()).days + 1
        valid_sales_days = int(grp.loc[grp["_sales"].notna(), "_date"].nunique())
        sales = grp["_sales"].dropna()
        zero_ratio = float((sales == 0).mean()) if len(sales) else 1.0
        cv = float(sales.std() / sales.mean()) if len(sales) and sales.mean() > 0 else 999.0
        gap_count = int(dates.diff().dt.days.fillna(1).sub(1).clip(lower=0).sum())
        score = (
            (1.0 if span_days >= required_days else span_days / required_days) * 0.35
            + min(valid_sales_days / required_days, 1.0) * 0.35
            + max(0, 1.0 - zero_ratio) * 0.15
            + max(0, 1.0 - min(cv, 5) / 5) * 0.10
            + max(0, 1.0 - gap_count / max(span_days, 1)) * 0.05
        )
        entity_stats.append({
            "entity_id": ent_id,
            "span_days": span_days,
            "valid_sales_days": valid_sales_days,
            "score": score,
            "gap_count": gap_count,
        })

    if len(entity_stats) < 2:
        return [], [], 0.3, "Fewer than 2 entities; cannot form source-target structure."

    entity_stats.sort(key=lambda x: x["score"], reverse=True)
    target = entity_stats[0]["entity_id"]
    sources = [e["entity_id"] for e in entity_stats[1:SOURCE_TOP_K + 1]]
    confidence = float(min(0.85, entity_stats[0]["score"]))
    reason = (
        "Auto-generated candidate split by data quality "
        f"(required_days={required_days}, target={target}, sources={len(sources)}). "
        "该源/目标域划分为程序根据数据质量自动生成的候选方案，不一定等同于论文或原实验设定，需要人工复核。"
    )
    logger.info(reason)
    return sources, [target], confidence, reason


def infer_source_target_entities(
    dataset_name: str,
    work_df: pd.DataFrame,
    entity_cols: List[str],
    cold_start_window: Dict[str, int],
    logger: ScanLogger,
) -> Dict[str, Any]:
    required_days = (
        int(cold_start_window.get("train_days", 15))
        + int(cold_start_window.get("val_days", 15))
        + int(cold_start_window.get("test_days", 180))
    )

    protocol = _search_strict_protocol_in_project(dataset_name)
    if protocol:
        sources, targets, reason = _map_protocol_to_entity_ids(
            dataset_name, protocol, entity_cols, work_df
        )
        if targets:
            return {
                "inferred_source_entities": sources,
                "inferred_target_entities": targets,
                "source_target_inference_method": "from_existing_project_config",
                "source_target_confidence": 0.90,
                "source_target_reason": reason,
            }

    sources, targets, confidence, reason = _auto_candidate_source_target(
        work_df, required_days, logger
    )
    return {
        "inferred_source_entities": sources,
        "inferred_target_entities": targets,
        "source_target_inference_method": "auto_candidate_by_data_profile",
        "source_target_confidence": confidence,
        "source_target_reason": reason,
    }


# ---------------------------------------------------------------------------
# Analysis layers
# ---------------------------------------------------------------------------

def analyze_time_span(work_df: pd.DataFrame) -> pd.DataFrame:
    if "_date" not in work_df.columns:
        return pd.DataFrame([{"note": "date column unavailable"}])
    dates = work_df["_date"].dropna()
    if dates.empty:
        return pd.DataFrame([{"note": "no valid dates"}])
    return pd.DataFrame([{
        "min_date": dates.min(),
        "max_date": dates.max(),
        "span_days": (dates.max() - dates.min()).days + 1,
        "unique_dates": int(dates.dt.normalize().nunique()),
        "total_rows": len(work_df),
    }])


def analyze_entities(work_df: pd.DataFrame) -> pd.DataFrame:
    if "_entity_id" not in work_df.columns or "_date" not in work_df.columns:
        return pd.DataFrame()
    rows = []
    for ent_id, grp in work_df.groupby("_entity_id"):
        dates = grp["_date"].dropna().sort_values()
        rows.append({
            "entity_id": ent_id,
            "row_count": len(grp),
            "min_date": dates.min() if not dates.empty else pd.NaT,
            "max_date": dates.max() if not dates.empty else pd.NaT,
            "span_days": (dates.max() - dates.min()).days + 1 if len(dates) else 0,
            "unique_dates": int(dates.dt.normalize().nunique()) if not dates.empty else 0,
        })
    return pd.DataFrame(rows).sort_values("row_count", ascending=False)


def analyze_data_quality(work_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gap_rows = []
    if "_entity_id" in work_df.columns and "_date" in work_df.columns:
        for ent_id, grp in work_df.groupby("_entity_id"):
            dates = grp["_date"].dropna().sort_values().dt.normalize().unique()
            if len(dates) < 2:
                continue
            full_range = pd.date_range(dates.min(), dates.max(), freq="D")
            missing = len(full_range) - len(dates)
            gap_rows.append({
                "entity_id": ent_id,
                "observed_days": len(dates),
                "expected_days": len(full_range),
                "missing_days": missing,
                "missing_ratio": missing / len(full_range) if len(full_range) else 0,
            })
    gap_df = pd.DataFrame(gap_rows)

    density_rows = [{
        "total_rows": len(work_df),
        "date_missing": int(work_df["_date"].isna().sum()) if "_date" in work_df else None,
        "sales_missing": int(work_df["_sales"].isna().sum()) if "_sales" in work_df else None,
        "entity_count": int(work_df["_entity_id"].nunique()) if "_entity_id" in work_df else None,
    }]
    return pd.DataFrame(density_rows), gap_df


def analyze_features(work_df: pd.DataFrame, inference: Dict[str, Any]) -> pd.DataFrame:
    skip = {
        inference.get("inferred_date_col"),
        inference.get("inferred_sales_col"),
        "_date", "_sales", "_entity_id",
        *(inference.get("inferred_entity_cols") or []),
    }
    skip = {c for c in skip if c}
    rows = []
    for col in work_df.columns:
        if col in skip or col.startswith("_"):
            continue
        series = work_df[col]
        rows.append({
            "feature": col,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_ratio": float(series.isna().mean()),
            "unique_count": _safe_nunique(series),
        })
    columns = ["feature", "dtype", "missing_count", "missing_ratio", "unique_count"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("missing_ratio", ascending=False)


def analyze_sales_distribution(work_df: pd.DataFrame) -> pd.DataFrame:
    if "_sales" not in work_df.columns or "_entity_id" not in work_df.columns:
        return pd.DataFrame()
    rows = []
    for ent_id, grp in work_df.groupby("_entity_id"):
        sales = grp["_sales"].dropna()
        if sales.empty:
            continue
        rows.append({
            "entity_id": ent_id,
            "count": len(sales),
            "mean": float(sales.mean()),
            "std": float(sales.std()),
            "min": float(sales.min()),
            "p50": float(sales.median()),
            "p95": float(sales.quantile(0.95)),
            "max": float(sales.max()),
            "zero_ratio": float((sales == 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def analyze_source_target(st_info: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{
        "inference_method": st_info.get("source_target_inference_method"),
        "confidence": st_info.get("source_target_confidence"),
        "target_count": len(st_info.get("inferred_target_entities") or []),
        "source_count": len(st_info.get("inferred_source_entities") or []),
        "target_entities": "|".join(st_info.get("inferred_target_entities") or []),
        "source_entities_sample": "|".join((st_info.get("inferred_source_entities") or [])[:10]),
        "reason": st_info.get("source_target_reason"),
    }])


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def _inference_md_section(inference: Dict[str, Any], st_info: Dict[str, Any]) -> str:
    needs_review = inference.get("needs_manual_review") or inference.get("inferred_status") == "LOW_CONFIDENCE"
    lines = [
        "## 0. 自动推断结果",
        "",
        f"- 自动识别的日期列：{inference.get('inferred_date_col')}",
        f"- 自动识别的销量列：{inference.get('inferred_sales_col')}",
        f"- 自动识别的实体列：{inference.get('inferred_entity_cols')}",
        f"- 日期列推断置信度：{inference.get('date_col_confidence', 0):.2f}",
        f"- 销量列推断置信度：{inference.get('sales_col_confidence', 0):.2f}",
        f"- 实体列推断置信度：{inference.get('entity_col_confidence', 0):.2f}",
        f"- 源/目标域推断方式：{st_info.get('source_target_inference_method')}",
        f"- 是否需要人工复核：{'是' if needs_review else '否'}",
        "",
    ]
    if st_info.get("source_target_inference_method") == "auto_candidate_by_data_profile":
        lines.append(
            "> 该源/目标域划分为程序根据数据质量自动生成的候选方案，"
            "不一定等同于论文或原实验设定，需要人工复核。"
        )
        lines.append("")
    return "\n".join(lines)


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_无数据_"
    view = df.head(max_rows)
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def write_dataset_reports(
    dataset_name: str,
    output_dir: Path,
    inference: Dict[str, Any],
    st_info: Dict[str, Any],
    time_span_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    density_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    sales_dist_df: pd.DataFrame,
    st_df: pd.DataFrame,
    logger: ScanLogger,
    scan_status: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    inferred_row = {**inference, **st_info}
    write_csv(pd.DataFrame([inferred_row]), output_dir / "inferred_columns_report.csv")

    write_csv(entity_df if not entity_df.empty else time_span_df, output_dir / "entity_time_span.csv")
    write_csv(sales_dist_df, output_dir / "entity_sales_distribution.csv")
    write_csv(feature_df, output_dir / "feature_missing_report.csv")
    write_csv(gap_df, output_dir / "gap_report.csv")
    write_csv(st_df, output_dir / "source_target_report.csv")

    summary_csv_rows = [{
        "dataset_name": dataset_name,
        "scan_status": scan_status,
        "inferred_date_col": inference.get("inferred_date_col"),
        "inferred_sales_col": inference.get("inferred_sales_col"),
        "inferred_entity_cols": str(inference.get("inferred_entity_cols")),
        "entity_count": int(entity_df["entity_id"].nunique()) if not entity_df.empty and "entity_id" in entity_df else None,
        "min_date": time_span_df.iloc[0]["min_date"] if not time_span_df.empty and "min_date" in time_span_df else None,
        "max_date": time_span_df.iloc[0]["max_date"] if not time_span_df.empty and "max_date" in time_span_df else None,
        "warnings_count": len(logger.warnings),
    }]
    write_csv(pd.DataFrame(summary_csv_rows), output_dir / "dataset_profile_summary.csv")

    key_warnings: List[str] = list(logger.warnings)
    low_conf = any(
        inference.get(k, 1.0) < CONFIDENCE_THRESHOLD
        for k in ("date_col_confidence", "sales_col_confidence", "entity_col_confidence")
    )
    if low_conf:
        key_warnings.insert(
            0,
            "日期列/销量列/实体列为低置信度自动推断结果，正式实验前需要人工复核。",
        )

    title_prefix = "SAMPLE_ONLY INVALID_FOR_FORMAL_EXPERIMENT：" if inference.get("sample_only") else ""
    md_parts = [
        f"# {title_prefix}数据集档案：{dataset_name}",
        "",
        _inference_md_section(inference, st_info),
        "## 1. 时间跨度",
        _df_to_markdown(time_span_df),
        "",
        "## 2. 实体划分",
        _df_to_markdown(entity_df),
        "",
        "## 3. 数据密度与质量",
        _df_to_markdown(density_df),
        "",
        "## 4. 特征维度",
        _df_to_markdown(feature_df, max_rows=30),
        "",
        "## 5. 销售量分布",
        _df_to_markdown(sales_dist_df, max_rows=30),
        "",
        "## 源/目标域",
        _df_to_markdown(st_df),
        "",
        "## 关键警告",
    ]
    if key_warnings:
        md_parts.extend(f"- {w}" for w in key_warnings)
    else:
        md_parts.append("- 无")
    if inference.get("sample_only"):
        md_parts.append("- SAMPLE_ONLY：该报告来自 sample 文件，不可用于正式实验判断。")
    md_parts.append("")

    (output_dir / "dataset_profile_summary.md").write_text("\n".join(md_parts), encoding="utf-8")
    (output_dir / "scan_log.txt").write_text("\n".join(logger.lines), encoding="utf-8")


def write_global_summary(
    run_dir: Path,
    all_summaries: List[Dict[str, Any]],
    all_inferences: List[Dict[str, Any]],
) -> None:
    summary_df = pd.DataFrame(all_summaries)
    write_csv(summary_df, run_dir / "summary_all_datasets.csv")

    infer_df = pd.DataFrame(all_inferences)
    write_csv(infer_df, run_dir / "inferred_columns_all_datasets.csv")

    md_lines = ["# 数据集档案扫描总览", "", f"运行时间：{run_dir.name}", ""]
    if not summary_df.empty:
        md_lines.append(_df_to_markdown(summary_df))
    else:
        md_lines.append("_无成功扫描的数据集_")
    (run_dir / "summary_all_datasets.md").write_text("\n".join(md_lines), encoding="utf-8")


def _candidate_entity_field_confidence(
    work_df: pd.DataFrame,
    date_col: Optional[str],
    sales_col: Optional[str],
) -> pd.DataFrame:
    rows = []
    for col in work_df.columns:
        if col in {date_col, sales_col, "_date", "_sales", "_entity_id"} or str(col).startswith("_"):
            continue
        series = work_df[col]
        rows.append({
            "field": col,
            "confidence": float(_entity_col_score(col, series, date_col, sales_col)),
            "unique_values": _safe_nunique(series),
            "missing_ratio": float(series.isna().mean()),
            "dtype": str(series.dtype),
        })
    if not rows:
        return pd.DataFrame(columns=["field", "confidence", "unique_values", "missing_ratio", "dtype"])
    return pd.DataFrame(rows).sort_values(["confidence", "unique_values"], ascending=[False, False])


def _full_entity_stats(work_df: pd.DataFrame, required_days: int) -> pd.DataFrame:
    stats_input = work_df[["_entity_id", "_date", "_sales"]].copy()
    stats_input["_date_norm"] = stats_input["_date"].dt.normalize()
    stats_input["_valid_sales_date"] = stats_input["_date_norm"].where(stats_input["_sales"].notna())
    stats_input["_sales_missing"] = stats_input["_sales"].isna().astype(int)
    stats_input["_zero_sales"] = ((stats_input["_sales"] == 0) & stats_input["_sales"].notna()).astype(int)
    stats_input["_sales_non_missing"] = stats_input["_sales"].notna().astype(int)

    stats = stats_input.groupby("_entity_id", sort=False).agg(
        row_count=("_date", "size"),
        min_date=("_date", "min"),
        max_date=("_date", "max"),
        unique_dates=("_date_norm", "nunique"),
        valid_sales_days=("_valid_sales_date", "nunique"),
        sales_missing_count=("_sales_missing", "sum"),
        zero_sales_count=("_zero_sales", "sum"),
        sales_non_missing_count=("_sales_non_missing", "sum"),
    ).reset_index().rename(columns={"_entity_id": "entity_id"})

    stats["span_days"] = (stats["max_date"] - stats["min_date"]).dt.days + 1
    stats["sales_missing_rate"] = stats["sales_missing_count"] / stats["row_count"].replace(0, np.nan)
    stats["zero_sales_ratio"] = stats["zero_sales_count"] / stats["sales_non_missing_count"].replace(0, np.nan)
    stats["date_density"] = stats["unique_dates"] / stats["span_days"].replace(0, np.nan)
    stats["meets_cold_start_210_days"] = (
        (stats["span_days"] >= required_days)
        & (stats["valid_sales_days"] >= required_days)
    )
    stats["source_target_quality_score"] = (
        np.minimum(stats["valid_sales_days"] / required_days, 1.0) * 0.45
        + stats["date_density"].fillna(0) * 0.35
        + (1.0 - stats["sales_missing_rate"].fillna(1.0)) * 0.20
    )
    return stats.sort_values(
        ["meets_cold_start_210_days", "source_target_quality_score", "valid_sales_days"],
        ascending=[False, False, False],
    )


def write_dataset4_full_reports(
    output_dir: Path,
    work_df: pd.DataFrame,
    inference: Dict[str, Any],
    load_meta: Dict[str, Any],
    cold_start_window: Dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    required_days = (
        int(cold_start_window.get("train_days", 15))
        + int(cold_start_window.get("val_days", 15))
        + int(cold_start_window.get("test_days", 180))
    )
    resolved_path = str(load_meta.get("resolved_path") or "")
    total_rows = int(len(work_df))
    invalid_reasons: List[str] = []
    if total_rows <= 1000:
        invalid_reasons.append("total_rows <= 1000，可能仍然读取的是样本文件。")
    if "sample" in Path(resolved_path).name.lower():
        invalid_reasons.append("文件名包含 sample，INVALID_FOR_FORMAL_EXPERIMENT。")

    field_df = _candidate_entity_field_confidence(
        work_df,
        inference.get("inferred_date_col"),
        inference.get("inferred_sales_col"),
    )
    write_csv(field_df, output_dir / "dataset4_full_entity_candidate_fields.csv")

    entity_stats = _full_entity_stats(work_df, required_days)
    write_csv(entity_stats, output_dir / "dataset4_full_entity_stats.csv")

    eligible = entity_stats[entity_stats["meets_cold_start_210_days"]].copy()
    candidates = []
    if not eligible.empty:
        target = eligible.iloc[0]
        candidates.append({
            "role": "candidate_target",
            "entity_id": target["entity_id"],
            "candidate": True,
            "quality_score": float(target["source_target_quality_score"]),
            "span_days": int(target["span_days"]),
            "valid_sales_days": int(target["valid_sales_days"]),
            "reason": "Top full-data entity by valid sales days, date density, and low missingness; requires manual confirmation.",
        })
        for _, row in eligible.iloc[1:SOURCE_TOP_K + 1].iterrows():
            candidates.append({
                "role": "candidate_source",
                "entity_id": row["entity_id"],
                "candidate": True,
                "quality_score": float(row["source_target_quality_score"]),
                "span_days": int(row["span_days"]),
                "valid_sales_days": int(row["valid_sales_days"]),
                "reason": "Candidate source selected from entities satisfying cold-start history; requires manual confirmation.",
            })
    source_target_df = pd.DataFrame(candidates)
    write_csv(source_target_df, output_dir / "dataset4_full_source_target_candidates.csv")

    dates = work_df["_date"].dropna()
    sales = work_df["_sales"]
    global_summary = {
        "resolved_path": resolved_path,
        "formal_validity": "INVALID_FOR_FORMAL_EXPERIMENT" if invalid_reasons else "FORMAL_FULL_DATA",
        "total_rows": total_rows,
        "min_date": dates.min() if not dates.empty else pd.NaT,
        "max_date": dates.max() if not dates.empty else pd.NaT,
        "unique_dates": int(dates.dt.normalize().nunique()) if not dates.empty else 0,
        "final_entity_unit_candidate": "store_id + product_id",
        "entity_count": int(entity_stats["entity_id"].nunique()),
        "span_days_min": float(entity_stats["span_days"].min()) if not entity_stats.empty else 0,
        "span_days_median": float(entity_stats["span_days"].median()) if not entity_stats.empty else 0,
        "span_days_max": float(entity_stats["span_days"].max()) if not entity_stats.empty else 0,
        "valid_sales_days_min": float(entity_stats["valid_sales_days"].min()) if not entity_stats.empty else 0,
        "valid_sales_days_median": float(entity_stats["valid_sales_days"].median()) if not entity_stats.empty else 0,
        "valid_sales_days_max": float(entity_stats["valid_sales_days"].max()) if not entity_stats.empty else 0,
        "sales_missing_rate": float(sales.isna().mean()) if len(sales) else 0.0,
        "zero_sales_ratio": float(((sales == 0) & sales.notna()).sum() / max(int(sales.notna().sum()), 1)),
        "date_density_median": float(entity_stats["date_density"].median()) if not entity_stats.empty else 0,
        "cold_start_required_days": required_days,
        "cold_start_entity_count": int(entity_stats["meets_cold_start_210_days"].sum()) if not entity_stats.empty else 0,
        "cold_start_entity_ratio": float(entity_stats["meets_cold_start_210_days"].mean()) if not entity_stats.empty else 0.0,
        "candidate_target_count": 1 if not eligible.empty else 0,
        "candidate_source_count": int(max(len(eligible) - 1, 0)),
    }

    summary_df = pd.DataFrame([global_summary])
    write_csv(summary_df, output_dir / "dataset4_full_profile_summary.csv")

    md_lines = [
        "# Dataset4 Full Profile Summary",
        "",
        "## 强制校验",
        "",
        f"- resolved_path: `{resolved_path}`",
        f"- formal_validity: {global_summary['formal_validity']}",
        f"- total_rows: {total_rows}",
    ]
    if invalid_reasons:
        md_lines.extend(f"- WARNING: {reason}" for reason in invalid_reasons)
    else:
        md_lines.append("- 强制校验通过：未读取 sample 文件，行数大于 1000。")
    md_lines.extend([
        "",
        "## 全局概览",
        _df_to_markdown(summary_df),
        "",
        "## 实体字段候选置信度",
        _df_to_markdown(field_df.head(30)),
        "",
        "## 最终实体单位候选",
        "",
        "- 建议候选：`store_id + product_id`",
        "- 理由：Dataset4 是门店-商品粒度销售数据，单独 `store_id` 会混合商品，单独 `product_id` 会在多门店场景混合门店需求。",
        "",
        "## Source/Target 候选",
        "",
        "以下划分为程序根据完整数据质量自动生成的 candidate，不等同于论文或原实验设定，需要人工确认。",
        "",
        _df_to_markdown(source_target_df.head(30)),
        "",
    ])
    (output_dir / "dataset4_full_profile_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    review_lines = [
        "# Dataset4 Full Entity Manual Review",
        "",
        "## 结论",
        "",
        "最终实体单位候选为 `store_id + product_id`，但 source/target 仍标记为 candidate，需要人工确认。",
        "",
        "## 时间跨度与销售密度分布",
        "",
        _df_to_markdown(pd.DataFrame([{
            "entity_count": global_summary["entity_count"],
            "span_days_min": global_summary["span_days_min"],
            "span_days_median": global_summary["span_days_median"],
            "span_days_max": global_summary["span_days_max"],
            "valid_sales_days_min": global_summary["valid_sales_days_min"],
            "valid_sales_days_median": global_summary["valid_sales_days_median"],
            "valid_sales_days_max": global_summary["valid_sales_days_max"],
            "sales_missing_rate": global_summary["sales_missing_rate"],
            "zero_sales_ratio": global_summary["zero_sales_ratio"],
            "date_density_median": global_summary["date_density_median"],
            "cold_start_entity_count": global_summary["cold_start_entity_count"],
            "cold_start_entity_ratio": global_summary["cold_start_entity_ratio"],
        }])),
        "",
        "## 实体明细文件",
        "",
        "- `dataset4_full_entity_stats.csv` 包含每个实体的 row_count、span_days、unique_dates、valid_sales_days、销售缺失率、零销量比例、日期密度、是否满足 210 天 cold-start 窗口。",
        "- `dataset4_full_source_target_candidates.csv` 包含自动 source/target 候选方案，全部为 candidate。",
    ]
    (output_dir / "dataset4_full_entity_manual_review.md").write_text("\n".join(review_lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Scan orchestration
# ---------------------------------------------------------------------------

def scan_single_dataset(
    ds_cfg: Dict[str, Any],
    run_dir: Path,
    cold_start_window: Dict[str, int],
    infer_only: bool,
    allow_sample: bool,
    logger: ScanLogger,
) -> Dict[str, Any]:
    name = ds_cfg["name"]
    ds_logger = ScanLogger(logger.verbose)
    ds_out = run_dir / name
    result: Dict[str, Any] = {"dataset_name": name, "scan_status": "FAILED"}

    try:
        raw_df, load_meta = load_dataset(ds_cfg["path"], ds_logger, allow_sample=allow_sample)
        ds_logger.info(f"Load meta: {load_meta}")

        date_guess = infer_date_col(raw_df)
        date_col = date_guess.get("inferred_date_col")
        if date_col and _detect_wide_qty_columns(raw_df.columns):
            raw_df = _melt_wide_sales(raw_df, date_col, ds_logger)

        inference = infer_columns(raw_df, name)
        inference["resolved_path"] = load_meta.get("resolved_path")
        inference["sample_only"] = bool(load_meta.get("sample_only"))
        inference["formal_validity"] = load_meta.get("formal_validity")
        work_df = apply_inferred_schema(raw_df, inference, ds_logger)

        st_info = infer_source_target_entities(
            name, work_df, inference.get("inferred_entity_cols") or [], cold_start_window, ds_logger
        )
        inference.update(st_info)

        if infer_only:
            write_csv(pd.DataFrame([inference]), ds_out / "inferred_columns_report.csv")
            (ds_out / "scan_log.txt").write_text("\n".join(ds_logger.lines), encoding="utf-8")
            result.update(inference)
            result["scan_status"] = "SUCCESS_WITH_WARNINGS" if ds_logger.warnings or inference.get("inferred_status") == "LOW_CONFIDENCE" else "SUCCESS"
            return result

        time_span_df = analyze_time_span(work_df)
        entity_df = analyze_entities(work_df)
        density_df, gap_df = analyze_data_quality(work_df)
        feature_df = analyze_features(work_df, inference)
        sales_dist_df = analyze_sales_distribution(work_df)
        st_df = analyze_source_target(st_info)

        scan_status = "SUCCESS"
        if ds_logger.warnings or inference.get("inferred_status") == "LOW_CONFIDENCE":
            scan_status = "SUCCESS_WITH_WARNINGS"

        write_dataset_reports(
            name, ds_out, inference, st_info,
            time_span_df, entity_df, density_df, gap_df,
            feature_df, sales_dist_df, st_df, ds_logger, scan_status,
        )
        if name == "Dataset4":
            write_dataset4_full_reports(
                ds_out,
                work_df,
                inference,
                load_meta,
                cold_start_window,
            )
        result.update(inference)
        result["scan_status"] = scan_status

    except Exception as exc:
        ds_logger.error(f"{name} scan failed: {exc}")
        ds_logger.error(traceback.format_exc())
        ds_out.mkdir(parents=True, exist_ok=True)
        (ds_out / "scan_log.txt").write_text("\n".join(ds_logger.lines), encoding="utf-8")
        result["scan_status"] = "FAILED"
        result["error"] = str(exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan dataset profiles with auto column inference.")
    parser.add_argument("--config", required=True, help="Path to dataset_profile_scan_config.json")
    parser.add_argument("--dataset", default=None, help="Scan only this dataset name")
    parser.add_argument("--infer-only", action="store_true", help="Only run column inference pre-check")
    parser.add_argument("--sample-ok", action="store_true", help="Explicitly allow SAMPLE_ONLY scans")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stdout")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = ScanLogger(verbose=args.verbose)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["output_root"]) / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output run directory: {run_dir}")

    datasets = cfg["datasets"]
    if args.dataset:
        datasets = [d for d in datasets if d["name"] == args.dataset]
        if not datasets:
            raise ValueError(f"Dataset '{args.dataset}' not found in config.")

    cold_start = cfg.get("cold_start_window", {})
    all_summaries: List[Dict[str, Any]] = []
    all_inferences: List[Dict[str, Any]] = []

    for ds_cfg in datasets:
        logger.info(f"=== Scanning {ds_cfg['name']} ===")
        result = scan_single_dataset(
            ds_cfg, run_dir, cold_start, args.infer_only, args.sample_ok, logger
        )
        all_summaries.append({
            "dataset_name": result.get("dataset_name"),
            "scan_status": result.get("scan_status"),
            "inferred_date_col": result.get("inferred_date_col"),
            "inferred_sales_col": result.get("inferred_sales_col"),
            "inferred_entity_cols": str(result.get("inferred_entity_cols")),
            "inferred_status": result.get("inferred_status"),
            "source_target_inference_method": result.get("source_target_inference_method"),
            "error": result.get("error"),
        })
        infer_row = {k: v for k, v in result.items() if k not in ("scan_status", "error")}
        infer_row["dataset_name"] = result.get("dataset_name")
        all_inferences.append(infer_row)

    write_global_summary(run_dir, all_summaries, all_inferences)
    logger.info(f"Scan complete. Results saved to: {run_dir}")

    if args.verbose:
        print(f"\nDone. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
