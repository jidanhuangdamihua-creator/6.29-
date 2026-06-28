#!/usr/bin/env python3
"""Select a D5 Favorita target store/SKU set for domain adaptation experiments."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "数据集" / "原始数据" / "Dataset 5Favorita"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "domain_adaptation" / "Dataset5" / "target_selection"

DATE_COL = "date"
STORE_COL = "store_nbr"
SKU_COL = "item_nbr"
SALES_COL = "unit_sales"
PROMO_COL = "onpromotion"
FAMILY_COL = "family"
GLOBAL_CALENDAR_DAYS = 1688
TRAIN_END = pd.Timestamp("2017-01-16")
GLOBAL_MAX_DATE = pd.Timestamp("2017-08-15")
PREFERRED_FAMILIES = ["GROCERY I", "BEVERAGES"]
EXCLUDED_FAMILIES = {"BOOKS", "MAGAZINES", "LADIESWEAR", "BABY CARE", "HARDWARE"}
FEATURE_COLUMNS = ["mean", "std", "cv", "coverage_ratio", "iqr", "acf_lag7", "trend_slope"]


@dataclass(frozen=True)
class WindowConfig:
    target_train_days: int = 15
    val_days: int = 15
    test_days: int = 180
    max_source_history_days: int = 300
    min_date_coverage_days: int = 510
    min_coverage_ratio: float = 0.60
    max_onpromotion_ratio: float = 0.30
    max_cv: float = 1.0
    max_spike_ratio: float = 8.0
    hard_transaction_gap_days: int = 30
    audit_transaction_gap_days: int = 7


def _bool_value(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "t", "yes", "y"}
    return bool(value)


def compute_target_windows(global_max_date: str | pd.Timestamp, cfg: WindowConfig = WindowConfig()) -> dict[str, pd.Timestamp]:
    max_date = pd.to_datetime(global_max_date)
    if max_date.normalize() == GLOBAL_MAX_DATE:
        return {
            "train_start": pd.Timestamp("2017-01-17"),
            "train_end": pd.Timestamp("2017-01-31"),
            "val_start": pd.Timestamp("2017-02-01"),
            "val_end": pd.Timestamp("2017-02-15"),
            "test_start": pd.Timestamp("2017-02-16"),
            "test_end": pd.Timestamp("2017-08-15"),
        }
    test_end = max_date
    test_start = test_end - pd.Timedelta(days=cfg.test_days - 1)
    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=cfg.val_days - 1)
    train_end = val_start - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=cfg.target_train_days - 1)
    return {
        "train_start": train_start.normalize(),
        "train_end": train_end.normalize(),
        "val_start": val_start.normalize(),
        "val_end": val_end.normalize(),
        "test_start": test_start.normalize(),
        "test_end": test_end.normalize(),
    }


def compute_source_window(train_start: str | pd.Timestamp, source_history_days: int) -> dict[str, Any]:
    train_start = pd.to_datetime(train_start).normalize()
    source_end = train_start + pd.Timedelta(days=WindowConfig().target_train_days + WindowConfig().val_days - 1)
    source_start = source_end - pd.Timedelta(days=source_history_days - 1)
    return {
        "source_start": source_start,
        "source_end": source_end,
        "source_history_days": int(source_history_days),
    }


def clean_sales_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if DATE_COL in out:
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    if SALES_COL in out:
        out[SALES_COL] = pd.to_numeric(out[SALES_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    if PROMO_COL in out:
        out[PROMO_COL] = out[PROMO_COL].map(_bool_value)
    return out


def complete_daily_series(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    if end < start:
        return pd.DataFrame({DATE_COL: pd.to_datetime([]), SALES_COL: [], PROMO_COL: []})
    if df.empty:
        base = pd.DataFrame({DATE_COL: pd.date_range(start, end, freq="D")})
        base[SALES_COL] = 0.0
        base[PROMO_COL] = False
        return base
    clean = clean_sales_frame(df)
    grouped = clean.groupby(DATE_COL, as_index=False).agg(
        unit_sales=(SALES_COL, "sum"),
        onpromotion=(PROMO_COL, "max") if PROMO_COL in clean else (SALES_COL, "size"),
    )
    full = pd.DataFrame({DATE_COL: pd.date_range(start, end, freq="D")})
    full = full.merge(grouped, on=DATE_COL, how="left")
    full[SALES_COL] = full[SALES_COL].fillna(0.0)
    if PROMO_COL in full:
        full[PROMO_COL] = full[PROMO_COL].map(lambda value: bool(value) if pd.notna(value) else False)
    else:
        full[PROMO_COL] = False
    return full


def _max_missing_run(observed_dates: set[pd.Timestamp], start: pd.Timestamp, end: pd.Timestamp) -> int:
    max_gap = 0
    current = 0
    for day in pd.date_range(start, end, freq="D"):
        if day.normalize() in observed_dates:
            current = 0
        else:
            current += 1
            max_gap = max(max_gap, current)
    return int(max_gap)


def transaction_gap_summary(
    transactions: pd.DataFrame,
    store_nbr: int,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    cfg: WindowConfig = WindowConfig(),
) -> dict[str, Any]:
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    if transactions.empty:
        max_gap = int((end - start).days + 1)
    else:
        tx = transactions.copy()
        tx[DATE_COL] = pd.to_datetime(tx[DATE_COL], errors="coerce").dt.normalize()
        dates = set(tx.loc[tx[STORE_COL].astype(int).eq(int(store_nbr)), DATE_COL].dropna())
        max_gap = _max_missing_run(dates, start, end)
    return {
        "max_transaction_gap_days": int(max_gap),
        "has_gap_gt_7_days": bool(max_gap > cfg.audit_transaction_gap_days),
        "passes_transaction_gap_gate": bool(max_gap <= cfg.hard_transaction_gap_days),
    }


def summarize_sales_series(
    series_df: pd.DataFrame,
    coverage_denominator: int | None = None,
) -> dict[str, float]:
    if series_df.empty:
        return {key: 0.0 for key in FEATURE_COLUMNS}
    sales = pd.to_numeric(series_df[SALES_COL], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    mean_val = float(np.mean(sales)) if len(sales) else 0.0
    std_val = float(np.std(sales)) if len(sales) else 0.0
    cv = float(std_val / mean_val) if mean_val > 0 else float("inf")
    iqr = float(np.percentile(sales, 75) - np.percentile(sales, 25)) if len(sales) else 0.0
    if len(sales) >= 14 and np.std(sales[:-7]) > 0 and np.std(sales[7:]) > 0:
        acf = float(np.corrcoef(sales[:-7], sales[7:])[0, 1])
        acf = 0.0 if not np.isfinite(acf) else acf
    else:
        acf = 0.0
    if len(sales) >= 2:
        slope = float(np.polyfit(np.arange(len(sales)), sales, 1)[0])
    else:
        slope = 0.0
    observed_days = int(pd.to_datetime(series_df[DATE_COL], errors="coerce").dt.normalize().nunique()) if DATE_COL in series_df else len(sales)
    denom = int(coverage_denominator) if coverage_denominator else max(len(sales), 1)
    coverage_ratio = float(observed_days / max(denom, 1))
    return {
        "mean": mean_val,
        "std": std_val,
        "cv": cv,
        "coverage_ratio": coverage_ratio,
        "iqr": iqr,
        "acf_lag7": acf,
        "trend_slope": slope,
    }


def _feature_array(summaries: Iterable[dict[str, float]]) -> np.ndarray:
    return np.asarray([[summary[col] for col in FEATURE_COLUMNS] for summary in summaries], dtype=float)


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
    return max(compute_mmd(X, Y, gamma), 0.0), gamma, scaler


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
    n_target = len(X)
    null_values = []
    for _ in range(n_permutations):
        idx = rng.permutation(len(combined))
        null_values.append(compute_mmd(combined[idx[:n_target]], combined[idx[n_target:]], gamma))
    return observed, float(np.mean(np.asarray(null_values) >= observed)), gamma


def _mean_summary(summaries: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(summaries)
    if not rows:
        return {"trend": float("nan"), "seasonality": float("nan"), "volatility": float("nan"), "coverage": float("nan"), "scale": float("nan")}
    return {
        "trend": float(np.mean([row["trend_slope"] for row in rows])),
        "seasonality": float(np.mean([row["acf_lag7"] for row in rows])),
        "volatility": float(np.mean([row["cv"] for row in rows])),
        "coverage": float(np.mean([row["coverage_ratio"] for row in rows])),
        "scale": float(np.mean([row["mean"] for row in rows])),
    }


def compute_structural_shift(
    target_summaries: Iterable[dict[str, float]],
    source_summaries: Iterable[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    target_summary = _mean_summary(target_summaries)
    source_summary = _mean_summary(source_summaries)
    shift = {key: float(abs(target_summary[key] - source_summary[key])) for key in target_summary}
    return shift, target_summary, source_summary


def compute_sku_metrics(group: pd.DataFrame, train_end: pd.Timestamp = TRAIN_END) -> dict[str, Any]:
    clean = clean_sales_frame(group)
    clean = clean[clean[DATE_COL].notna()].copy()
    if clean.empty:
        return {}
    min_date = clean[DATE_COL].min().normalize()
    full = complete_daily_series(clean[clean[DATE_COL] <= train_end], min_date, train_end)
    observed_days = int(clean.loc[clean[DATE_COL] <= train_end, DATE_COL].dt.normalize().nunique())
    date_coverage_days = int((train_end - min_date).days + 1)
    sales = full[SALES_COL].to_numpy(dtype=float)
    positive = sales[sales > 0]
    mean_val = float(np.mean(sales)) if len(sales) else 0.0
    std_val = float(np.std(sales)) if len(sales) else 0.0
    median_positive = float(np.median(positive)) if len(positive) else 0.0
    return {
        "series_min_date": min_date,
        "series_train_end": train_end,
        "date_coverage_days": date_coverage_days,
        "observed_days": observed_days,
        "coverage_ratio": float(observed_days / max(date_coverage_days, 1)),
        "onpromotion_ratio": float(full[PROMO_COL].mean()) if PROMO_COL in full else 0.0,
        "mean": mean_val,
        "std": std_val,
        "cv": float(std_val / mean_val) if mean_val > 0 else float("inf"),
        "spike_ratio": float(np.max(sales) / median_positive) if median_positive > 0 else float("inf"),
    }


def _eligible(metrics: pd.DataFrame, cfg: WindowConfig) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    return metrics[
        (metrics["date_coverage_days"] >= cfg.min_date_coverage_days)
        & (metrics["coverage_ratio"] >= cfg.min_coverage_ratio)
        & (metrics["onpromotion_ratio"] <= cfg.max_onpromotion_ratio)
        & (metrics["cv"] <= cfg.max_cv)
        & (metrics["spike_ratio"] <= cfg.max_spike_ratio)
        & (~metrics[FAMILY_COL].isin(EXCLUDED_FAMILIES))
    ].copy()


def filter_family_candidates(
    metrics: pd.DataFrame,
    family: str,
    cfg: WindowConfig = WindowConfig(),
) -> tuple[pd.DataFrame, list[str]]:
    family_metrics = metrics[metrics[FAMILY_COL].eq(family)].copy()
    attempts = [
        ("default", cfg),
        ("relaxed_cv_1.5", WindowConfig(max_cv=1.5)),
        ("relaxed_coverage_0.50", WindowConfig(max_cv=1.5, min_coverage_ratio=0.50)),
        ("relaxed_onpromotion_0.40", WindowConfig(max_cv=1.5, min_coverage_ratio=0.50, max_onpromotion_ratio=0.40)),
    ]
    relaxations: list[str] = []
    for label, active_cfg in attempts:
        if label != "default":
            relaxations.append(label)
        eligible = _eligible(family_metrics, active_cfg)
        if len(eligible) >= 3:
            eligible["eligible_rule"] = label
            return eligible, relaxations
    eligible = _eligible(family_metrics, attempts[-1][1])
    eligible["eligible_rule"] = "insufficient_after_relaxation" if not eligible.empty else ""
    return eligible, relaxations


def select_target_skus_with_fallback(metrics: pd.DataFrame) -> dict[str, Any]:
    family_results: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    for family in PREFERRED_FAMILIES:
        eligible, relaxations = filter_family_candidates(metrics, family)
        family_results[family] = (eligible, relaxations)
        if len(eligible) >= 3:
            chosen = eligible.sort_values(["cv", SKU_COL], ascending=[True, True]).head(5)
            return {
                "target_skus": chosen[SKU_COL].astype(int).tolist(),
                "target_family": family,
                "target_families": [family],
                "family_selection_rule": f"{family}: " + (", ".join(relaxations) if relaxations else "default"),
                "eligible_metrics": eligible,
            }
    merged = pd.concat([value[0] for value in family_results.values()], ignore_index=True)
    if len(merged) >= 3:
        chosen = merged.sort_values(["cv", SKU_COL], ascending=[True, True]).head(5)
        families = sorted(chosen[FAMILY_COL].dropna().unique().tolist())
        return {
            "target_skus": chosen[SKU_COL].astype(int).tolist(),
            "target_family": "MIXED",
            "target_families": families,
            "family_selection_rule": "mixed_family_fallback",
            "eligible_metrics": merged,
        }
    return {
        "target_skus": [],
        "target_family": "",
        "target_families": [],
        "family_selection_rule": "insufficient_grocery_i_beverages",
        "eligible_metrics": merged,
    }


def build_source_entities(
    train: pd.DataFrame,
    items: pd.DataFrame,
    target_store: int,
    target_families: Iterable[str],
    min_start: pd.Timestamp | None = None,
    max_end: pd.Timestamp | None = None,
) -> list[dict[str, int]]:
    family_items = items[items[FAMILY_COL].isin(list(target_families))][SKU_COL].astype(int)
    subset = train[
        (train[STORE_COL].astype(int) != int(target_store))
        & (train[SKU_COL].astype(int).isin(family_items.tolist()))
    ].copy()
    if min_start is not None:
        subset = subset[pd.to_datetime(subset[DATE_COL]) >= pd.to_datetime(min_start)]
    if max_end is not None:
        subset = subset[pd.to_datetime(subset[DATE_COL]) <= pd.to_datetime(max_end)]
    pairs = subset[[STORE_COL, SKU_COL]].drop_duplicates().sort_values([STORE_COL, SKU_COL])
    return [{STORE_COL: int(row[STORE_COL]), SKU_COL: int(row[SKU_COL])} for _, row in pairs.iterrows()]


def _read_train_filtered(
    train_path: Path,
    chunksize: int,
    store_filter: set[int] | None = None,
    item_filter: set[int] | None = None,
    date_start: pd.Timestamp | None = None,
    date_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    parts = []
    usecols = [DATE_COL, STORE_COL, SKU_COL, SALES_COL, PROMO_COL]
    dtypes = {STORE_COL: "int32", SKU_COL: "int32", SALES_COL: "float32", PROMO_COL: "string"}
    for chunk in pd.read_csv(train_path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        if store_filter is not None:
            chunk = chunk[chunk[STORE_COL].isin(store_filter)]
        if item_filter is not None:
            chunk = chunk[chunk[SKU_COL].isin(item_filter)]
        chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        if date_start is not None:
            chunk = chunk[chunk[DATE_COL] >= date_start]
        if date_end is not None:
            chunk = chunk[chunk[DATE_COL] <= date_end]
        if not chunk.empty:
            parts.append(clean_sales_frame(chunk))
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True)


def _entity_summaries(
    df: pd.DataFrame,
    store: int | None = None,
    item_ids: Iterable[int] | None = None,
    coverage_denominator: int | None = None,
    train_end: pd.Timestamp = TRAIN_END,
    max_entities: int | None = None,
    random_state: int = 42,
) -> tuple[list[dict[str, float]], np.ndarray, list[dict[str, int]]]:
    subset = df.copy()
    if store is not None:
        subset = subset[subset[STORE_COL].astype(int).eq(int(store))]
    if item_ids is not None:
        subset = subset[subset[SKU_COL].astype(int).isin([int(x) for x in item_ids])]
    if max_entities is not None and not subset.empty:
        pairs = subset[[STORE_COL, SKU_COL]].drop_duplicates().sort_values([STORE_COL, SKU_COL])
        if len(pairs) > max_entities:
            sampled_idx = np.random.default_rng(random_state).choice(pairs.index.to_numpy(), size=max_entities, replace=False)
            pairs = pairs.loc[sorted(sampled_idx)]
            subset = subset.merge(pairs, on=[STORE_COL, SKU_COL], how="inner")
    summaries: list[dict[str, float]] = []
    meta: list[dict[str, int]] = []
    for (store_nbr, item_nbr), group in subset.groupby([STORE_COL, SKU_COL], sort=False):
        group = clean_sales_frame(group)
        group = group[group[DATE_COL] <= train_end]
        if group.empty:
            continue
        start = group[DATE_COL].min().normalize()
        full = complete_daily_series(group, start, train_end)
        summary = summarize_sales_series(full, coverage_denominator=coverage_denominator)
        summaries.append(summary)
        meta.append({STORE_COL: int(store_nbr), SKU_COL: int(item_nbr)})
    return summaries, _feature_array(summaries), meta


def _target_sku_metrics_for_store(train: pd.DataFrame, items: pd.DataFrame, store_nbr: int) -> pd.DataFrame:
    rows = []
    store_df = train[train[STORE_COL].astype(int).eq(int(store_nbr))]
    item_meta = items[[SKU_COL, FAMILY_COL]].drop_duplicates()
    for item_nbr, group in store_df.groupby(SKU_COL, sort=False):
        metrics = compute_sku_metrics(group)
        if not metrics:
            continue
        metrics[STORE_COL] = int(store_nbr)
        metrics[SKU_COL] = int(item_nbr)
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).merge(item_meta, on=SKU_COL, how="left")
    out[FAMILY_COL] = out[FAMILY_COL].fillna("")
    return out


def scan_candidate_stores(
    train: pd.DataFrame,
    items: pd.DataFrame,
    stores: pd.DataFrame,
    transactions: pd.DataFrame,
    cfg: WindowConfig = WindowConfig(),
    store_sku_sample: int = 50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = compute_target_windows(GLOBAL_MAX_DATE, cfg)
    candidate_meta = stores[stores["type"].isin(["A", "B"])].copy()
    if (candidate_meta["city"] == "Quito").any():
        candidate_meta = candidate_meta[candidate_meta["city"].eq("Quito")].copy()
    rng = np.random.default_rng(random_state)
    rows = []
    metrics_parts = []
    store_feature_cache: dict[int, np.ndarray] = {}
    for _, store_row in candidate_meta.sort_values([STORE_COL]).iterrows():
        store_nbr = int(store_row[STORE_COL])
        sku_metrics = _target_sku_metrics_for_store(train, items, store_nbr)
        selection = select_target_skus_with_fallback(sku_metrics) if not sku_metrics.empty else {"target_skus": [], "eligible_metrics": pd.DataFrame()}
        if not sku_metrics.empty:
            metrics_parts.append(sku_metrics)
        store_train = train[train[STORE_COL].astype(int).eq(store_nbr)]
        sku_ids = sorted(store_train[SKU_COL].dropna().astype(int).unique().tolist())
        if len(sku_ids) > store_sku_sample:
            sku_ids = sorted(rng.choice(sku_ids, size=store_sku_sample, replace=False).astype(int).tolist())
        _summaries, features, _meta = _entity_summaries(
            store_train,
            store=store_nbr,
            item_ids=sku_ids,
            coverage_denominator=GLOBAL_CALENDAR_DAYS,
        )
        store_feature_cache[store_nbr] = features
        gap = transaction_gap_summary(transactions, store_nbr, windows["train_start"], windows["test_end"], cfg)
        rows.append({
            STORE_COL: store_nbr,
            "city": store_row.get("city", ""),
            "type": store_row.get("type", ""),
            "is_quito": bool(store_row.get("city", "") == "Quito"),
            "is_type_a": bool(store_row.get("type", "") == "A"),
            **gap,
            "eligible_target_sku_count": int(len(selection.get("eligible_metrics", []))),
            "selected_candidate_sku_count": int(len(selection.get("target_skus", []))),
            "lowest_eligible_cv": float(selection["eligible_metrics"]["cv"].min()) if not selection.get("eligible_metrics", pd.DataFrame()).empty else float("inf"),
            "sampled_store_sku_count": int(len(features)),
            "status": "pending_mmd",
        })
    scan = pd.DataFrame(rows)
    for idx, row in scan.iterrows():
        store_nbr = int(row[STORE_COL])
        target_features = store_feature_cache.get(store_nbr, np.empty((0, len(FEATURE_COLUMNS))))
        source_features = np.vstack([value for key, value in store_feature_cache.items() if key != store_nbr and len(value)]) if len(store_feature_cache) > 1 else np.empty((0, len(FEATURE_COLUMNS)))
        if len(target_features) == 0 or len(source_features) == 0:
            scan.loc[idx, "mmd"] = np.nan
            scan.loc[idx, "gamma"] = np.nan
            scan.loc[idx, "status"] = "insufficient_mmd_features"
            continue
        mmd, gamma, _scaler = scaled_mmd(target_features, source_features)
        scan.loc[idx, "mmd"] = mmd
        scan.loc[idx, "gamma"] = gamma
        scan.loc[idx, "status"] = "ok"
    if scan["mmd"].notna().any():
        valid = scan["mmd"].dropna()
        q25 = float(valid.quantile(0.25))
        q75 = float(valid.quantile(0.75))
        median = float(valid.median())
        scan["mmd_q25"] = q25
        scan["mmd_q75"] = q75
        scan["mmd_median"] = median
        scan["in_mmd_iqr"] = scan["mmd"].between(q25, q75, inclusive="both")
        scan["mmd_distance_to_median"] = (scan["mmd"] - median).abs()
    all_metrics = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    return scan, all_metrics


def select_target_store(scan_df: pd.DataFrame) -> dict[str, Any]:
    valid = scan_df[
        scan_df["passes_transaction_gap_gate"].fillna(False)
        & scan_df["mmd"].notna()
        & scan_df["selected_candidate_sku_count"].ge(3)
    ].copy()
    if valid.empty:
        raise ValueError("No valid D5 candidate stores after transaction, MMD, and SKU gates")
    if "in_mmd_iqr" in valid and valid["in_mmd_iqr"].any():
        valid = valid[valid["in_mmd_iqr"]].copy()
    selected = valid.sort_values(
        ["is_quito", "is_type_a", "mmd_distance_to_median", "eligible_target_sku_count", "lowest_eligible_cv", STORE_COL],
        ascending=[False, False, True, False, True, True],
    ).iloc[0]
    return selected.to_dict()


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
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _write_report(result: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# D5 Target Selection Report",
        "",
        "Final metrics are recomputed from Favorita raw files using no-leakage training history.",
        "Negative sales are clipped to zero and missing natural dates are filled with zero for feature calculation only.",
        "",
        f"- Target store: `{result['target_store']}`",
        f"- Target SKUs: `{result['target_skus']}`",
        f"- Target family: `{result['target_family']}`",
        f"- Target families: `{result['target_families']}`",
        f"- Family rule: `{result['family_selection_rule']}`",
        f"- Source entities: `{len(result['source_entities'])}`",
        f"- MMD: `{result['mmd_value']}`",
        f"- Permutation p-value: `{result['permutation_p_value']}`",
        "",
        "## Transaction Gap Rule",
        "",
        "- Hard exclusion: continuous `>30` days with no transaction record.",
        "- Audit only: `has_gap_gt_7_days`.",
        "",
        "## Structural Shift",
        "",
    ]
    for key, value in result["structural_shift"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Source Pool",
        "",
        "The source pool follows the actual final target family set and is represented by concrete `(store_nbr, item_nbr)` entities.",
    ])
    (output_dir / "target_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_target_selection(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    permutations: int = 500,
    store_sku_sample: int = 50,
    random_state: int = 42,
    chunksize: int = 5_000_000,
    feature_entity_limit: int = 1000,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = pd.read_csv(dataset_root / "items.csv")
    stores = pd.read_csv(dataset_root / "stores.csv")
    transactions = pd.read_csv(dataset_root / "transactions.csv")
    candidate_stores = set(stores.loc[stores["type"].isin(["A", "B"]), STORE_COL].astype(int).tolist())
    if stores["city"].eq("Quito").any():
        candidate_stores = set(stores.loc[stores["type"].isin(["A", "B"]) & stores["city"].eq("Quito"), STORE_COL].astype(int).tolist())
    train = _read_train_filtered(
        dataset_root / "train.csv",
        chunksize=chunksize,
        store_filter=candidate_stores,
        date_end=TRAIN_END,
    )
    scan_df, target_metrics = scan_candidate_stores(
        train,
        items,
        stores,
        transactions,
        store_sku_sample=store_sku_sample,
        random_state=random_state,
    )
    selected_store = select_target_store(scan_df)
    target_store = int(selected_store[STORE_COL])
    store_metrics = target_metrics[target_metrics[STORE_COL].astype(int).eq(target_store)]
    selected_skus = select_target_skus_with_fallback(store_metrics)
    source_item_ids = set(items[items[FAMILY_COL].isin(selected_skus["target_families"])][SKU_COL].astype(int).tolist())
    source_window = compute_source_window(compute_target_windows(GLOBAL_MAX_DATE)["train_start"], WindowConfig().max_source_history_days)
    source_train = _read_train_filtered(
        dataset_root / "train.csv",
        chunksize=chunksize,
        item_filter=source_item_ids,
        date_start=source_window["source_start"],
        date_end=TRAIN_END,
    )
    source_entities = build_source_entities(
        source_train,
        items,
        target_store=target_store,
        target_families=selected_skus["target_families"],
        min_start=source_window["source_start"],
        max_end=TRAIN_END,
    )
    target_summaries, target_features, _target_meta = _entity_summaries(
        train,
        store=target_store,
        item_ids=selected_skus["target_skus"],
        coverage_denominator=GLOBAL_CALENDAR_DAYS,
    )
    source_summaries, source_features, _source_meta = _entity_summaries(
        source_train[source_train[STORE_COL].astype(int).ne(target_store)],
        coverage_denominator=GLOBAL_CALENDAR_DAYS,
        max_entities=feature_entity_limit,
        random_state=random_state,
    )
    mmd_value, p_value, gamma = permutation_test_mmd(
        target_features,
        source_features,
        n_permutations=permutations,
        random_state=random_state,
    )
    shift, target_shift_summary, source_shift_summary = compute_structural_shift(target_summaries, source_summaries)
    windows = compute_target_windows(GLOBAL_MAX_DATE)
    result = {
        "target_store": target_store,
        "target_skus": selected_skus["target_skus"],
        "target_family": selected_skus["target_family"],
        "target_families": selected_skus["target_families"],
        "family_selection_rule": selected_skus["family_selection_rule"],
        "source_stores": sorted({row[STORE_COL] for row in source_entities}),
        "source_skus": sorted({row[SKU_COL] for row in source_entities}),
        "source_entities": source_entities,
        "mmd_value": mmd_value,
        "mmd_gamma": gamma,
        "permutation_p_value": p_value,
        "structural_shift": shift,
        "target_structural_summary": target_shift_summary,
        "source_structural_summary": source_shift_summary,
        "time_windows": {
            **windows,
            **source_window,
        },
        "selection_rules": {
            "transaction_hard_gap_days": 30,
            "transaction_audit_gap_days": 7,
            "structural_shift_semantics": "absolute_difference_to_match_d4",
            "store_mmd_coverage_ratio": "observed_days / 1688",
            "sku_coverage_ratio": "observed_days / date_coverage_days",
            "source_feature_entity_limit": int(feature_entity_limit),
        },
    }
    scan_df.to_csv(output_dir / "store_candidate_profile.csv", index=False, encoding="utf-8-sig")
    target_metrics.to_csv(output_dir / "target_sku_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "target_selection_result.json").write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(_jsonable(result), output_dir)
    return _jsonable(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Dataset5 Favorita target store/SKU set.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--store-sku-sample", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=5_000_000)
    parser.add_argument("--feature-entity-limit", type=int, default=1000)
    args = parser.parse_args()
    result = run_target_selection(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        permutations=args.permutations,
        store_sku_sample=args.store_sku_sample,
        random_state=args.random_state,
        chunksize=args.chunksize,
        feature_entity_limit=args.feature_entity_limit,
    )
    print(
        "D5 target selection saved: "
        f"target_store={result['target_store']}, "
        f"target_skus={len(result['target_skus'])}, "
        f"target_families={result['target_families']}, "
        f"source_entities={len(result['source_entities'])}, "
        f"mmd={result['mmd_value']}, "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
