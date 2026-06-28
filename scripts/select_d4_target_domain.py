#!/usr/bin/env python3
"""Select a D4 target store/SKU set for domain adaptation experiments."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = ROOT / "数据集" / "原始数据" / "Dataset 4叮咚数据集" / "data" / "train.parquet"
DEFAULT_PROFILE_DIR = ROOT / "outputs" / "dataset_profiles" / "Dataset4"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "domain_adaptation" / "Dataset4" / "target_selection"

WAREHOUSE_COL = "store_id"
SKU_COL = "product_id"
DATE_COL = "dt"
SALES_COL = "sale_amount"
CATEGORY_COL = "second_category_id"
RAW_COLUMNS = ["city_id", WAREHOUSE_COL, SKU_COL, DATE_COL, SALES_COL, CATEGORY_COL]


@dataclass(frozen=True)
class WindowConfig:
    target_train_days: int = 15
    val_days: int = 15
    test_days: int = 180
    max_source_history_days: int = 300
    min_date_coverage_days: int = 510
    max_gap_days: int = 30
    min_nonzero_ratio: float = 0.70
    max_cv: float = 1.5
    max_spike_ratio: float = 10.0


def compute_target_windows(series_max_date, cfg: WindowConfig = WindowConfig()) -> dict[str, pd.Timestamp]:
    series_max_date = pd.to_datetime(series_max_date)
    test_end = series_max_date
    test_start = test_end - pd.Timedelta(days=cfg.test_days - 1)
    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=cfg.val_days - 1)
    target_train_end = val_start - pd.Timedelta(days=1)
    target_train_start = target_train_end - pd.Timedelta(days=cfg.target_train_days - 1)
    return {
        "target_train_start": target_train_start,
        "target_train_end": target_train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
    }


def compute_source_window(target_train_start, source_history_days: int) -> dict[str, Any]:
    target_train_start = pd.to_datetime(target_train_start)
    source_history_end = target_train_start
    source_history_start = source_history_end - pd.Timedelta(days=source_history_days - 1)
    return {
        "source_history_start": source_history_start,
        "source_history_end": source_history_end,
        "source_history_days": int(source_history_days),
    }


def date_coverage_days(entity_df: pd.DataFrame, date_col: str = DATE_COL) -> int:
    if entity_df.empty or date_col not in entity_df:
        return 0
    dates = pd.to_datetime(entity_df[date_col], errors="coerce").dropna()
    if dates.empty:
        return 0
    return int((dates.max().normalize() - dates.min().normalize()).days + 1)


def max_calendar_gap_days(entity_df: pd.DataFrame, threshold: int | None = None, date_col: str = DATE_COL) -> int | None:
    if entity_df.empty or date_col not in entity_df:
        return None
    dates = (
        pd.to_datetime(entity_df[date_col], errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )
    if dates.empty:
        return None
    full_calendar = pd.date_range(dates.min(), dates.max(), freq="D")
    observed = set(dates)
    max_gap = 0
    current_gap = 0
    for day in full_calendar:
        if day not in observed:
            current_gap += 1
            max_gap = max(max_gap, current_gap)
            if threshold is not None and max_gap > threshold:
                return int(max_gap)
        else:
            current_gap = 0
    return int(max_gap)


def spike_ratio(series: Iterable[float]) -> float:
    s = np.asarray(list(series), dtype=float)
    s = s[np.isfinite(s)]
    nonzero = s[s > 0]
    if len(nonzero) == 0:
        return float("inf")
    med = np.median(nonzero)
    return float("inf") if med <= 0 else float(np.max(s) / med)


def compute_sku_metrics(sku_df: pd.DataFrame, sales_col: str = SALES_COL, date_col: str = DATE_COL) -> dict[str, float | int | None]:
    s = sku_df[sales_col].to_numpy(dtype=float) if sales_col in sku_df else np.asarray([], dtype=float)
    s = s[np.isfinite(s)]
    mean_val = float(np.mean(s)) if len(s) else 0.0
    std_val = float(np.std(s)) if len(s) else 0.0
    cv = float(std_val / mean_val) if mean_val > 0 else float("inf")
    nonzero_ratio = float(np.mean(s > 0)) if len(s) else 0.0
    return {
        "date_coverage_days": date_coverage_days(sku_df, date_col=date_col),
        "max_gap_days": max_calendar_gap_days(sku_df, date_col=date_col),
        "mean": mean_val,
        "std": std_val,
        "nonzero_ratio": nonzero_ratio,
        "cv": cv,
        "spike_ratio": spike_ratio(s),
    }


def extract_sku_summary(series: Iterable[float]) -> list[float] | None:
    s = np.asarray(list(series), dtype=float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return None
    mean_val = float(np.mean(s))
    std_val = float(np.std(s))
    cv = float(std_val / mean_val) if mean_val > 0 else 0.0
    zero_ratio = float(np.mean(s == 0))
    iqr = float(np.percentile(s, 75) - np.percentile(s, 25))
    if len(s) >= 14 and np.std(s[:-7]) > 0 and np.std(s[7:]) > 0:
        acf = float(np.corrcoef(s[:-7], s[7:])[0, 1])
        acf = 0.0 if not np.isfinite(acf) else acf
    else:
        acf = 0.0
    if len(s) >= 2:
        t = np.arange(len(s))
        slope = float(np.polyfit(t, s, 1)[0])
    else:
        slope = 0.0
    return [mean_val, std_val, cv, zero_ratio, iqr, acf, slope]


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def median_heuristic_gamma(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.vstack([X, Y])
    if len(Z) < 2:
        return 1.0
    diff = Z[:, None, :] - Z[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    upper = dist_sq[np.triu_indices_from(dist_sq, k=1)]
    positive = upper[upper > 0]
    if len(positive) == 0:
        return 1.0
    median_dist_sq = float(np.median(positive))
    if not np.isfinite(median_dist_sq) or median_dist_sq <= 0:
        return 1.0
    return float(1.0 / (2.0 * median_dist_sq))


def scaled_mmd(target_features: np.ndarray, source_features: np.ndarray) -> tuple[float, float, StandardScaler]:
    target_features = np.asarray(target_features, dtype=float)
    source_features = np.asarray(source_features, dtype=float)
    if target_features.ndim != 2 or source_features.ndim != 2:
        raise ValueError("target_features and source_features must be 2D arrays")
    if len(target_features) == 0 or len(source_features) == 0:
        raise ValueError("target_features and source_features must be non-empty")
    scaler = StandardScaler()
    combined = np.vstack([target_features, source_features])
    scaler.fit(combined)
    X = scaler.transform(target_features)
    Y = scaler.transform(source_features)
    gamma = median_heuristic_gamma(X, Y)
    mmd = max(compute_mmd(X, Y, gamma=gamma), 0.0)
    return float(mmd), float(gamma), scaler


def permutation_test_mmd(
    target_features: np.ndarray,
    source_features: np.ndarray,
    n_permutations: int = 500,
    random_state: int = 42,
) -> tuple[float, float | None, float]:
    observed, gamma, scaler = scaled_mmd(target_features, source_features)
    if n_permutations <= 0:
        return observed, None, gamma
    rng = np.random.default_rng(random_state)
    X = scaler.transform(np.asarray(target_features, dtype=float))
    Y = scaler.transform(np.asarray(source_features, dtype=float))
    combined = np.vstack([X, Y])
    n_x = len(X)
    null_values = []
    for _ in range(n_permutations):
        idx = rng.permutation(len(combined))
        X_perm = combined[idx[:n_x]]
        Y_perm = combined[idx[n_x:]]
        null_values.append(compute_mmd(X_perm, Y_perm, gamma=gamma))
    p_value = float(np.mean(np.asarray(null_values) >= observed))
    return observed, p_value, gamma


def filter_eligible_skus(metrics_df: pd.DataFrame, cfg: WindowConfig = WindowConfig()) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df.copy()
    max_gap = metrics_df["max_gap_days"].fillna(float("inf"))
    eligible = metrics_df[
        (metrics_df["date_coverage_days"] >= cfg.min_date_coverage_days)
        & (max_gap <= cfg.max_gap_days)
        & (metrics_df["nonzero_ratio"] >= cfg.min_nonzero_ratio)
        & (metrics_df["cv"] <= cfg.max_cv)
        & (metrics_df["spike_ratio"] <= cfg.max_spike_ratio)
    ].copy()
    eligible["eligible_rule"] = "default"
    return eligible


def choose_target_skus(eligible: pd.DataFrame, min_skus: int = 3, max_skus: int = 5) -> tuple[list[int], int]:
    if eligible.empty:
        return [], -1
    category_counts = (
        eligible.groupby(CATEGORY_COL)[SKU_COL]
        .nunique()
        .sort_values(ascending=False, kind="mergesort")
    )
    main_category = category_counts.index[0]
    final_pool = eligible[eligible[CATEGORY_COL] == main_category]
    target_skus = (
        final_pool.sort_values(["cv", SKU_COL], ascending=[True, True])
        .head(max_skus)[SKU_COL]
        .astype(int)
        .tolist()
    )
    if len(target_skus) < min_skus:
        return [], int(main_category)
    return target_skus, int(main_category)


def parse_entity_id(entity_id: str) -> tuple[int, int]:
    match = re.match(r"store_id=(\d+)\|product_id=(\d+)", str(entity_id))
    if not match:
        raise ValueError(f"Bad entity_id: {entity_id}")
    return int(match.group(1)), int(match.group(2))


def aggregate_store_profile(profile_dir: str | Path, candidate_store_limit: int | None = 50) -> pd.DataFrame:
    path = Path(profile_dir) / "source_target_candidate_report.csv"
    if not path.exists():
        return pd.DataFrame(columns=[WAREHOUSE_COL, "profile_product_count"])
    report = pd.read_csv(path)
    if "entity_id" not in report:
        return pd.DataFrame(columns=[WAREHOUSE_COL, "profile_product_count"])
    parsed = report["entity_id"].apply(parse_entity_id)
    report[WAREHOUSE_COL] = parsed.apply(lambda pair: pair[0])
    report[SKU_COL] = parsed.apply(lambda pair: pair[1])
    store_profile = report.groupby(WAREHOUSE_COL).agg(
        profile_product_count=(SKU_COL, "nunique"),
        max_total_calendar_days=("total_calendar_days", "max"),
        median_total_calendar_days=("total_calendar_days", "median"),
        median_nonzero_sales_days=("nonzero_sales_days", "median"),
        median_cv=("coefficient_of_variation", "median"),
        median_quality_score=("quality_score", "median"),
    ).reset_index()
    store_profile = store_profile.sort_values(
        ["max_total_calendar_days", "profile_product_count", "median_quality_score", "median_cv", WAREHOUSE_COL],
        ascending=[False, False, False, True, True],
    )
    if candidate_store_limit is not None:
        store_profile = store_profile.head(candidate_store_limit)
    return store_profile.reset_index(drop=True)


def _selection_history(group: pd.DataFrame, cfg: WindowConfig) -> tuple[pd.DataFrame, dict[str, pd.Timestamp], dict[str, Any]]:
    windows = compute_target_windows(group[DATE_COL].max(), cfg)
    source_window = compute_source_window(windows["target_train_start"], cfg.max_source_history_days)
    history = group[
        (group[DATE_COL] >= source_window["source_history_start"])
        & (group[DATE_COL] < windows["val_start"])
    ].copy()
    return history, windows, source_window


def compute_store_sku_metrics(df: pd.DataFrame, store_id: int, cfg: WindowConfig = WindowConfig()) -> pd.DataFrame:
    store_df = df[df[WAREHOUSE_COL] == store_id]
    rows: list[dict[str, Any]] = []
    for product_id, group in store_df.groupby(SKU_COL, sort=False):
        group = group.sort_values(DATE_COL)
        history, windows, source_window = _selection_history(group, cfg)
        full_metrics = compute_sku_metrics(group)
        history_metrics = compute_sku_metrics(history)
        category_values = group[CATEGORY_COL].dropna().unique()
        rows.append({
            WAREHOUSE_COL: int(store_id),
            SKU_COL: int(product_id),
            CATEGORY_COL: int(category_values[0]) if len(category_values) else -1,
            "series_min_date": group[DATE_COL].min(),
            "series_max_date": group[DATE_COL].max(),
            "date_coverage_days": full_metrics["date_coverage_days"],
            "max_gap_days": full_metrics["max_gap_days"],
            "history_start": source_window["source_history_start"],
            "history_end_exclusive": windows["val_start"],
            "history_observed_days": int(history[DATE_COL].nunique()) if not history.empty else 0,
            "nonzero_ratio": history_metrics["nonzero_ratio"],
            "cv": history_metrics["cv"],
            "spike_ratio": history_metrics["spike_ratio"],
            "history_mean": history_metrics["mean"],
            "history_std": history_metrics["std"],
        })
    return pd.DataFrame(rows)


def _feature_rows_from_groups(groups: Iterable[tuple[Any, pd.DataFrame]]) -> tuple[np.ndarray, list[pd.Series], list[dict[str, Any]]]:
    features = []
    series_list: list[pd.Series] = []
    meta: list[dict[str, Any]] = []
    for key, group in groups:
        summary = extract_sku_summary(group[SALES_COL].to_numpy(dtype=float))
        if summary is None:
            continue
        features.append(summary)
        series_list.append(group[SALES_COL].reset_index(drop=True))
        if isinstance(key, tuple):
            meta.append({WAREHOUSE_COL: int(key[0]), SKU_COL: int(key[1])})
        else:
            meta.append({SKU_COL: int(key)})
    return np.asarray(features, dtype=float), series_list, meta


def build_target_features(
    df: pd.DataFrame,
    target_store: int,
    target_skus: Iterable[int],
    source_window: dict[str, Any],
    windows: dict[str, pd.Timestamp],
) -> tuple[np.ndarray, list[pd.Series], list[dict[str, Any]]]:
    subset = df[
        (df[WAREHOUSE_COL] == target_store)
        & (df[SKU_COL].isin(list(target_skus)))
        & (df[DATE_COL] >= source_window["source_history_start"])
        & (df[DATE_COL] < windows["val_start"])
    ].copy()
    return _feature_rows_from_groups(subset.groupby(SKU_COL, sort=False))


def build_source_features(
    df: pd.DataFrame,
    target_store: int,
    categories: Iterable[int],
    source_window: dict[str, Any],
    cfg: WindowConfig = WindowConfig(),
) -> tuple[np.ndarray, list[pd.Series], pd.DataFrame]:
    categories = list(categories)
    subset = df[
        (df[WAREHOUSE_COL] != target_store)
        & (df[CATEGORY_COL].isin(categories))
        & (df[DATE_COL] >= source_window["source_history_start"])
        & (df[DATE_COL] <= source_window["source_history_end"])
    ].copy()
    features = []
    series_list: list[pd.Series] = []
    rows: list[dict[str, Any]] = []
    for (store_id, product_id), group in subset.groupby([WAREHOUSE_COL, SKU_COL], sort=False):
        coverage = date_coverage_days(group)
        if coverage < cfg.max_source_history_days:
            continue
        summary = extract_sku_summary(group[SALES_COL].to_numpy(dtype=float))
        if summary is None:
            continue
        features.append(summary)
        series_list.append(group.sort_values(DATE_COL)[SALES_COL].reset_index(drop=True))
        rows.append({
            WAREHOUSE_COL: int(store_id),
            SKU_COL: int(product_id),
            CATEGORY_COL: int(group[CATEGORY_COL].iloc[0]),
            "source_history_days_observed": int(group[DATE_COL].nunique()),
            "source_date_coverage_days": int(coverage),
        })
    return np.asarray(features, dtype=float), series_list, pd.DataFrame(rows)


def summarize_series_list(series_list: Iterable[Iterable[float]]) -> dict[str, float]:
    values = {
        "trend": [],
        "seasonality": [],
        "volatility": [],
        "sparsity": [],
        "scale": [],
    }
    for series in series_list:
        feat = extract_sku_summary(series)
        if feat is None:
            continue
        mean_val, _std_val, cv, zero_ratio, _iqr, acf_lag7, trend_slope = feat
        values["trend"].append(trend_slope)
        values["seasonality"].append(acf_lag7)
        values["volatility"].append(cv)
        values["sparsity"].append(zero_ratio)
        values["scale"].append(mean_val)
    return {key: float(np.mean(val)) if len(val) else float("nan") for key, val in values.items()}


def compute_structural_shift(
    target_series_list: Iterable[Iterable[float]],
    source_series_list: Iterable[Iterable[float]],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    target_summary = summarize_series_list(target_series_list)
    source_summary = summarize_series_list(source_series_list)
    shift = {
        key: float(abs(target_summary[key] - source_summary[key]))
        for key in target_summary
    }
    return shift, target_summary, source_summary


def scan_candidate_stores(
    df: pd.DataFrame,
    store_profile: pd.DataFrame | None = None,
    cfg: WindowConfig = WindowConfig(),
    permutations: int = 0,
    random_state: int = 42,
) -> pd.DataFrame:
    if store_profile is not None and not store_profile.empty and WAREHOUSE_COL in store_profile:
        candidate_stores = store_profile[WAREHOUSE_COL].astype(int).drop_duplicates().tolist()
    else:
        candidate_stores = sorted(df[WAREHOUSE_COL].dropna().astype(int).unique().tolist())

    rows: list[dict[str, Any]] = []
    for index, store_id in enumerate(candidate_stores):
        store_df = df[df[WAREHOUSE_COL] == store_id]
        if store_df.empty:
            continue
        sku_metrics = compute_store_sku_metrics(df, store_id, cfg)
        eligible = filter_eligible_skus(sku_metrics, cfg)
        if len(eligible) < 3:
            rows.append({
                WAREHOUSE_COL: int(store_id),
                "eligible_sku_count": int(len(eligible)),
                "status": "insufficient_eligible_skus",
            })
            continue
        main_category = int(eligible.groupby(CATEGORY_COL)[SKU_COL].nunique().idxmax())
        main_eligible = eligible[eligible[CATEGORY_COL] == main_category]
        if len(main_eligible) < 3:
            rows.append({
                WAREHOUSE_COL: int(store_id),
                "eligible_sku_count": int(len(eligible)),
                "main_category": main_category,
                "status": "insufficient_main_category_skus",
            })
            continue
        anchor_date = pd.to_datetime(main_eligible["series_max_date"]).min()
        windows = compute_target_windows(anchor_date, cfg)
        source_window = compute_source_window(windows["target_train_start"], cfg.max_source_history_days)
        target_features, _target_series, _target_meta = build_target_features(
            df, store_id, main_eligible[SKU_COL].astype(int).tolist(), source_window, windows
        )
        source_features, _source_series, source_meta = build_source_features(
            df, store_id, [main_category], source_window, cfg
        )
        if len(target_features) < 3 or len(source_features) == 0:
            rows.append({
                WAREHOUSE_COL: int(store_id),
                "eligible_sku_count": int(len(eligible)),
                "main_category": main_category,
                "target_feature_count": int(len(target_features)),
                "source_sku_count": int(len(source_features)),
                "status": "insufficient_mmd_features",
            })
            continue
        mmd_value, p_value, gamma = permutation_test_mmd(
            target_features,
            source_features,
            n_permutations=permutations,
            random_state=random_state + index,
        )
        rows.append({
            WAREHOUSE_COL: int(store_id),
            "main_category": main_category,
            "eligible_sku_count": int(len(eligible)),
            "main_category_eligible_sku_count": int(main_eligible[SKU_COL].nunique()),
            "target_feature_count": int(len(target_features)),
            "source_sku_count": int(len(source_features)),
            "source_store_count": int(source_meta[WAREHOUSE_COL].nunique()) if not source_meta.empty else 0,
            "store_max_gap_days": max_calendar_gap_days(store_df, threshold=cfg.max_gap_days),
            "anchor_series_max_date": anchor_date,
            "target_train_start": windows["target_train_start"],
            "val_start": windows["val_start"],
            "source_history_start": source_window["source_history_start"],
            "source_history_end": source_window["source_history_end"],
            "mmd": mmd_value,
            "gamma": gamma,
            "p_value": p_value,
            "status": "ok",
        })
    scan = pd.DataFrame(rows)
    if "mmd" in scan and scan["mmd"].notna().any():
        valid_mmd = scan["mmd"].dropna()
        q25 = float(valid_mmd.quantile(0.25))
        q75 = float(valid_mmd.quantile(0.75))
        scan["mmd_q25"] = q25
        scan["mmd_q75"] = q75
        scan["in_mmd_iqr"] = scan["mmd"].between(q25, q75, inclusive="both")
    return scan


def select_target_store(scan_df: pd.DataFrame) -> dict[str, Any]:
    valid = scan_df[scan_df["mmd"].notna()].copy()
    if "status" in valid:
        valid = valid[valid["status"].fillna("ok") == "ok"].copy()
    if valid.empty:
        raise ValueError("No valid candidate stores with MMD values")
    q25 = float(valid["mmd"].quantile(0.25))
    q75 = float(valid["mmd"].quantile(0.75))
    median = float(valid["mmd"].median())
    middle = valid[valid["mmd"].between(q25, q75, inclusive="both")].copy()
    if middle.empty:
        middle = valid.copy()
        middle["mmd_distance_to_median"] = (middle["mmd"] - median).abs()
    else:
        middle["mmd_distance_to_median"] = (middle["mmd"] - median).abs()
    selected = middle.sort_values(
        ["eligible_sku_count", "source_sku_count", "store_max_gap_days", "mmd_distance_to_median", WAREHOUSE_COL],
        ascending=[False, False, True, True, True],
    ).iloc[0].to_dict()
    selected["mmd_q25"] = q25
    selected["mmd_q75"] = q75
    selected["mmd_median"] = median
    return selected


def _select_final_target_skus(metrics_df: pd.DataFrame, cfg: WindowConfig) -> tuple[pd.DataFrame, list[int], int, list[str]]:
    relaxations: list[str] = []
    eligible = filter_eligible_skus(metrics_df, cfg)
    skus, category = choose_target_skus(eligible)
    if skus:
        return eligible, skus, category, relaxations
    relaxed_steps = [
        ("cv <= 2.0", WindowConfig(max_cv=2.0)),
        ("nonzero_ratio >= 0.60", WindowConfig(max_cv=2.0, min_nonzero_ratio=0.60)),
        ("max_gap_days <= 45", WindowConfig(max_cv=2.0, min_nonzero_ratio=0.60, max_gap_days=45)),
    ]
    for label, relaxed_cfg in relaxed_steps:
        relaxations.append(label)
        eligible = filter_eligible_skus(metrics_df, relaxed_cfg)
        skus, category = choose_target_skus(eligible)
        if skus:
            eligible["eligible_rule"] = "relaxed: " + ", ".join(relaxations)
            return eligible, skus, category, relaxations
    return eligible, [], category, relaxations


def read_raw_d4(raw_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(raw_path, columns=RAW_COLUMNS)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# D4 Target Selection Report",
        "",
        "Profile was used only to narrow candidate stores. Final SKU metrics, MMD features, and structural-shift features were recomputed from raw `train.parquet` using no-leakage training history.",
        "",
        f"- Target store: `{result['target_store_id']}`",
        f"- Target SKUs: `{result['target_skus']}`",
        f"- Target categories: `{result['target_categories']}`",
        f"- Source stores: `{result['source_store_count']}`",
        f"- Source SKUs: `{result['source_sku_count']}`",
        f"- MMD: `{result['mmd']['value']}`",
        f"- Permutation p-value: `{result['permutation_test']['p_value']}`",
        f"- MMD IQR rule: Q25=`{result['mmd_selection']['q25']}`, Q75=`{result['mmd_selection']['q75']}`",
        f"- Relaxations: `{result['relaxations']}`",
        "",
        "## Windows",
        "",
        f"- Target train: `{result['target_windows']['target_train_start']}` to `{result['target_windows']['target_train_end']}`",
        f"- Validation: `{result['target_windows']['val_start']}` to `{result['target_windows']['val_end']}`",
        f"- Test: `{result['target_windows']['test_start']}` to `{result['target_windows']['test_end']}`",
        f"- Source history: `{result['source_window']['source_history_start']}` to `{result['source_window']['source_history_end']}`",
        "",
        "## Structural Shift",
        "",
    ]
    for key, value in result["structural_shift"].items():
        lines.append(f"- `{key}`: `{value}`")
    (output_dir / "target_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_target_selection(
    raw_path: str | Path = DEFAULT_RAW_PATH,
    profile_dir: str | Path = DEFAULT_PROFILE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cfg: WindowConfig = WindowConfig(),
    candidate_store_limit: int | None = 50,
    permutations: int = 500,
    random_state: int = 42,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    store_profile = aggregate_store_profile(profile_dir, candidate_store_limit=candidate_store_limit)
    store_profile.to_csv(output_dir / "store_candidate_profile.csv", index=False)

    df = read_raw_d4(raw_path)
    scan = scan_candidate_stores(
        df,
        store_profile=store_profile,
        cfg=cfg,
        permutations=permutations,
        random_state=random_state,
    )
    selected_store = select_target_store(scan)
    scan["selected"] = scan[WAREHOUSE_COL] == int(selected_store[WAREHOUSE_COL])
    scan.to_csv(output_dir / "warehouse_mmd_scan.csv", index=False)

    target_store = int(selected_store[WAREHOUSE_COL])
    target_metrics = compute_store_sku_metrics(df, target_store, cfg)
    eligible, target_skus, main_category, relaxations = _select_final_target_skus(target_metrics, cfg)
    if not target_skus:
        raise ValueError(f"Target store {target_store} did not yield at least 3 target SKUs")
    target_metrics["eligible_default"] = target_metrics[SKU_COL].isin(filter_eligible_skus(target_metrics, cfg)[SKU_COL])
    target_metrics["selected_target_sku"] = target_metrics[SKU_COL].isin(target_skus)
    target_metrics.to_csv(output_dir / "target_sku_metrics.csv", index=False)

    target_categories = sorted(eligible[eligible[SKU_COL].isin(target_skus)][CATEGORY_COL].dropna().astype(int).unique().tolist())
    anchor_date = pd.to_datetime(target_metrics[target_metrics[SKU_COL].isin(target_skus)]["series_max_date"]).min()
    target_windows = compute_target_windows(anchor_date, cfg)
    source_window = compute_source_window(target_windows["target_train_start"], cfg.max_source_history_days)
    target_features, target_series, _target_meta = build_target_features(
        df, target_store, target_skus, source_window, target_windows
    )
    source_features, source_series, source_meta = build_source_features(
        df, target_store, target_categories, source_window, cfg
    )
    if len(target_features) == 0 or len(source_features) == 0:
        raise ValueError("Selected target/source pools did not yield MMD features")
    mmd_value, p_value, gamma = permutation_test_mmd(
        target_features,
        source_features,
        n_permutations=permutations,
        random_state=random_state,
    )
    shift, target_shift_summary, source_shift_summary = compute_structural_shift(target_series, source_series)

    result = {
        "dataset": "Dataset4",
        "raw_data": str(raw_path),
        "target_store_id": target_store,
        "target_skus": [int(sku) for sku in target_skus],
        "category_col": CATEGORY_COL,
        "target_categories": [int(category) for category in target_categories],
        "source_store_count": int(source_meta[WAREHOUSE_COL].nunique()) if not source_meta.empty else 0,
        "source_sku_count": int(len(source_meta)),
        "source_row_count": int(
            df[
                (df[WAREHOUSE_COL] != target_store)
                & (df[CATEGORY_COL].isin(target_categories))
                & (df[DATE_COL] >= source_window["source_history_start"])
                & (df[DATE_COL] <= source_window["source_history_end"])
            ].shape[0]
        ),
        "window_config": asdict(cfg),
        "target_windows": target_windows,
        "source_window": source_window,
        "mmd": {
            "value": mmd_value,
            "gamma": gamma,
            "scaler": "StandardScaler fit jointly on target + source features",
            "feature_space": "7-dimensional SKU summary features",
            "target_feature_count": int(len(target_features)),
            "source_feature_count": int(len(source_features)),
        },
        "mmd_selection": {
            "q25": selected_store.get("mmd_q25"),
            "q75": selected_store.get("mmd_q75"),
            "median": selected_store.get("mmd_median"),
        },
        "permutation_test": {
            "n_permutations": int(permutations),
            "p_value": p_value,
        },
        "structural_shift": shift,
        "target_structural_summary": target_shift_summary,
        "source_structural_summary": source_shift_summary,
        "relaxations": relaxations,
        "profile_usage": "Profile is used only for candidate store narrowing; final decisions are recomputed from raw train.parquet before val/test windows.",
    }
    result = _jsonable(result)
    with (output_dir / "target_selection_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    write_report(result, output_dir)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-store-limit", type=int, default=50)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_target_selection(
        raw_path=args.raw_path,
        profile_dir=args.profile_dir,
        output_dir=args.output_dir,
        candidate_store_limit=args.candidate_store_limit,
        permutations=args.permutations,
        random_state=args.random_state,
    )
    print(f"target_store_id={result['target_store_id']}")
    print(f"target_skus={result['target_skus']}")
    print(f"source_sku_count={result['source_sku_count']}")
    print(f"output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
