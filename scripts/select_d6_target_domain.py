#!/usr/bin/env python3
"""Select a D6 M5 target store/SKU set for domain adaptation experiments."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "数据集" / "原始数据" / "Dataset 6m5-forecasting-accuracy"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "domain_adaptation" / "Dataset6" / "target_selection"

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
FEATURE_COLUMNS = ["mean", "std", "cv", "zero_ratio", "iqr", "acf_lag7", "trend_slope"]
STATE_PRIORITY = {"CA": 0, "TX": 1, "WI": 2}


@dataclass(frozen=True)
class WindowConfig:
    target_train_days: int = 15
    val_days: int = 15
    test_days: int = 180
    source_history_days: int = 300
    expected_max_date: str = "2016-05-22"


def _d_number(d_col: str) -> int:
    return int(str(d_col).split("_", 1)[1])


def d_cols_from_frame(df: pd.DataFrame) -> list[str]:
    return sorted([col for col in df.columns if str(col).startswith("d_")], key=_d_number)


def resolve_max_date(calendar: pd.DataFrame, n_d_cols: int = 1941) -> pd.Timestamp:
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
    target_d = f"d_{n_d_cols}"
    match = cal.loc[cal["d"].eq(target_d), "date"]
    if match.empty:
        raise ValueError(f"calendar.csv does not contain {target_d}")
    return pd.Timestamp(match.iloc[0]).normalize()


def compute_target_windows(max_date: str | pd.Timestamp, cfg: WindowConfig = WindowConfig()) -> dict[str, pd.Timestamp]:
    test_end = pd.to_datetime(max_date).normalize()
    test_start = test_end - pd.Timedelta(days=cfg.test_days - 1)
    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=cfg.val_days - 1)
    train_end = val_start - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=cfg.target_train_days - 1)
    return {
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
    }


def compute_source_window(train_start: str | pd.Timestamp, source_history_days: int) -> dict[str, Any]:
    train_start = pd.to_datetime(train_start).normalize()
    source_end = train_start - pd.Timedelta(days=1)
    source_start = source_end - pd.Timedelta(days=source_history_days - 1)
    return {
        "source_start": source_start,
        "source_end": source_end,
        "source_history_days": int(source_history_days),
    }


def get_screening_d_cols(calendar: pd.DataFrame, train_end: str | pd.Timestamp) -> list[str]:
    cal = calendar.copy()
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
    train_end = pd.to_datetime(train_end).normalize()
    eligible = cal[cal["date"] <= train_end].copy()
    if eligible.empty:
        raise ValueError("No calendar d columns found before train_end")
    eligible["_d_num"] = eligible["d"].map(_d_number)
    return eligible.sort_values("_d_num")["d"].astype(str).tolist()


def _percentile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.percentile(values, q, method="linear"))
    except TypeError:
        return float(np.percentile(values, q, interpolation="linear"))


def _sanitize_float(value: float) -> float:
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def extract_sku_summary(sales_array: Iterable[float]) -> dict[str, float]:
    sales = np.asarray(list(sales_array), dtype=float)
    if sales.size == 0:
        return {key: 0.0 for key in FEATURE_COLUMNS}
    mean_val = float(np.mean(sales))
    std_val = float(np.std(sales, ddof=0))
    cv = float(std_val / mean_val) if mean_val > 0 else 0.0
    zero_ratio = float(np.mean(sales == 0))
    iqr = _percentile(sales, 75) - _percentile(sales, 25)
    if len(sales) >= 14 and np.std(sales[:-7], ddof=0) > 0 and np.std(sales[7:], ddof=0) > 0:
        acf = float(np.corrcoef(sales[:-7], sales[7:])[0, 1])
    else:
        acf = 0.0
    if not np.isfinite(acf):
        acf = 0.0
    slope = float(np.polyfit(np.arange(len(sales)), sales, 1)[0]) if len(sales) >= 2 else 0.0
    return {
        "mean": _sanitize_float(mean_val),
        "std": _sanitize_float(std_val),
        "cv": _sanitize_float(cv),
        "zero_ratio": _sanitize_float(zero_ratio),
        "iqr": _sanitize_float(iqr),
        "acf_lag7": _sanitize_float(acf),
        "trend_slope": _sanitize_float(slope),
    }


def compute_sku_screening_metrics(row: pd.Series, sales_array: Iterable[float]) -> dict[str, Any]:
    sales = np.asarray(list(sales_array), dtype=float)
    positive = sales[sales > 0]
    mean_val = float(np.mean(sales)) if len(sales) else 0.0
    std_val = float(np.std(sales, ddof=0)) if len(sales) else 0.0
    cv = float(std_val / mean_val) if mean_val > 0 else float("inf")
    median_positive = float(np.median(positive)) if len(positive) else 0.0
    spike_ratio = float(np.max(sales) / median_positive) if median_positive > 0 else float("inf")
    summary = extract_sku_summary(sales)
    return {
        **summary,
        "item_id": row.get("item_id"),
        "dept_id": row.get("dept_id"),
        "cat_id": row.get("cat_id"),
        "nonzero_ratio": float(np.mean(sales > 0)) if len(sales) else 0.0,
        "cv": cv,
        "spike_ratio": spike_ratio,
    }


def _feature_array(summaries: Iterable[dict[str, float]]) -> np.ndarray:
    rows = list(summaries)
    if not rows:
        return np.empty((0, len(FEATURE_COLUMNS)), dtype=float)
    return np.asarray([[row[col] for col in FEATURE_COLUMNS] for row in rows], dtype=float)


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    XX = rbf_kernel(X, X, gamma=gamma)
    YY = rbf_kernel(Y, Y, gamma=gamma)
    XY = rbf_kernel(X, Y, gamma=gamma)
    return float(XX.mean() + YY.mean() - 2.0 * XY.mean())


def _median_gamma(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.vstack([X, Y])
    if len(Z) < 2:
        return 1.0
    dists = pairwise_distances(Z, metric="euclidean")
    off_diag = dists[~np.eye(len(Z), dtype=bool)]
    nonzero = off_diag[off_diag > 0]
    if len(nonzero) == 0:
        return 1.0
    median_dist = float(np.median(nonzero))
    if median_dist <= 0 or not np.isfinite(median_dist):
        return 1.0
    return float(1.0 / (2.0 * median_dist ** 2))


def scaled_mmd(target_features: np.ndarray, source_features: np.ndarray) -> tuple[float, float, StandardScaler]:
    target_features = np.asarray(target_features, dtype=float)
    source_features = np.asarray(source_features, dtype=float)
    if target_features.ndim != 2 or source_features.ndim != 2:
        raise ValueError("target_features and source_features must be 2D arrays")
    if len(target_features) == 0 or len(source_features) == 0:
        raise ValueError("target_features and source_features must be non-empty")
    scaler = StandardScaler().fit(np.vstack([target_features, source_features]))
    X = scaler.transform(target_features)
    Y = scaler.transform(source_features)
    gamma = _median_gamma(X, Y)
    return max(compute_mmd(X, Y, gamma), 0.0), gamma, scaler


def permutation_p_value(X: np.ndarray, Y: np.ndarray, gamma: float, n_perm: int = 500, random_state: int = 43) -> float:
    observed = compute_mmd(X, Y, gamma)
    if n_perm <= 0:
        return 1.0
    rng = np.random.RandomState(random_state)
    Z = np.vstack([X, Y])
    n_x = len(X)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(Z))
        mmd_perm = compute_mmd(Z[perm[:n_x]], Z[perm[n_x:]], gamma)
        if mmd_perm >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))


def _eligible(metrics: pd.DataFrame, dept_scope: list[str], nonzero_ratio: float, cv: float, spike_ratio: float) -> pd.DataFrame:
    subset = metrics[metrics["dept_id"].isin(dept_scope)].copy()
    return subset[
        (subset["nonzero_ratio"] >= nonzero_ratio)
        & (subset["cv"] <= cv)
        & (subset["spike_ratio"] <= spike_ratio)
    ].copy()


def select_target_skus_with_fallback(metrics: pd.DataFrame) -> dict[str, Any]:
    attempts = []
    round1 = _eligible(metrics, ["FOODS_3"], 0.30, 1.5, 10)
    attempts.append({"fallback_round": 1, "department_scope": ["FOODS_3"], "n_eligible": int(len(round1))})
    if len(round1) >= 3:
        chosen = round1.sort_values(["cv", "item_id"], ascending=[True, True]).head(5)
        return {
            "target_skus": chosen["item_id"].astype(str).tolist(),
            "target_department": "FOODS_3",
            "fallback_round": 1,
            "eligible_metrics": round1,
            "attempts": attempts,
        }
    round2 = _eligible(metrics, ["FOODS_3"], 0.20, 2.0, 15)
    attempts.append({"fallback_round": 2, "department_scope": ["FOODS_3"], "n_eligible": int(len(round2))})
    if len(round2) >= 3:
        chosen = round2.sort_values(["cv", "item_id"], ascending=[True, True]).head(5)
        return {
            "target_skus": chosen["item_id"].astype(str).tolist(),
            "target_department": "FOODS_3",
            "fallback_round": 2,
            "eligible_metrics": round2,
            "attempts": attempts,
        }
    round3 = _eligible(metrics, ["FOODS_1", "FOODS_2", "FOODS_3"], 0.30, 1.5, 10)
    dept_counts = round3.groupby("dept_id")["item_id"].nunique() if not round3.empty else pd.Series(dtype=int)
    candidate_depts = [dept for dept, count in dept_counts.items() if count >= 3]
    attempts.append({"fallback_round": 3, "department_scope": ["FOODS_1", "FOODS_2", "FOODS_3"], "n_eligible": int(len(round3))})
    if candidate_depts:
        dept_rank = (
            round3[round3["dept_id"].isin(candidate_depts)]
            .groupby("dept_id")["cv"]
            .median()
            .sort_values(kind="mergesort")
        )
        target_department = str(dept_rank.index[0])
        pool = round3[round3["dept_id"].eq(target_department)]
        chosen = pool.sort_values(["cv", "item_id"], ascending=[True, True]).head(5)
        return {
            "target_skus": chosen["item_id"].astype(str).tolist(),
            "target_department": target_department,
            "fallback_round": 3,
            "eligible_metrics": pool,
            "attempts": attempts,
        }
    return {
        "target_skus": [],
        "target_department": "",
        "fallback_round": 0,
        "eligible_metrics": pd.DataFrame(),
        "attempts": attempts,
    }


def build_source_entities(df: pd.DataFrame, target_store: str, target_department: str) -> list[dict[str, str]]:
    subset = df[(df["store_id"].astype(str) != str(target_store)) & (df["dept_id"].astype(str) == str(target_department))]
    pairs = subset[["store_id", "item_id"]].drop_duplicates().sort_values(["store_id", "item_id"])
    return [{"store_id": str(row["store_id"]), "item_id": str(row["item_id"])} for _, row in pairs.iterrows()]


def _mean_summary(summaries: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(summaries)
    if not rows:
        return {"scale": float("nan"), "volatility": float("nan"), "sparsity": float("nan"), "seasonality": float("nan"), "trend": float("nan")}
    return {
        "scale": float(np.mean([row["mean"] for row in rows])),
        "volatility": float(np.mean([row["cv"] for row in rows])),
        "sparsity": float(np.mean([row["zero_ratio"] for row in rows])),
        "seasonality": float(np.mean([row["acf_lag7"] for row in rows])),
        "trend": float(np.mean([row["trend_slope"] for row in rows])),
    }


def compute_structural_shift(
    target_summaries: Iterable[dict[str, float]],
    source_summaries: Iterable[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    target_summary = _mean_summary(target_summaries)
    source_summary = _mean_summary(source_summaries)
    shift = {key: float(target_summary[key] - source_summary[key]) for key in target_summary}
    return shift, target_summary, source_summary


def _jsonable(value: Any) -> Any:
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


def _load_inputs(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]
    for name in required:
        if not (dataset_root / name).is_file():
            raise FileNotFoundError(f"Missing required D6 input file: {dataset_root / name}")
    calendar = pd.read_csv(dataset_root / "calendar.csv", usecols=["d", "date"], parse_dates=["date"])
    sales = pd.read_csv(dataset_root / "sales_train_evaluation.csv")
    missing = [col for col in ID_COLS if col not in sales.columns]
    if missing:
        raise ValueError(f"sales_train_evaluation.csv missing columns: {missing}")
    return sales, calendar


def _row_summaries(df: pd.DataFrame, d_cols: list[str]) -> list[dict[str, float]]:
    return [extract_sku_summary(row[d_cols].to_numpy(dtype=float)) for _, row in df.iterrows()]


def _row_features(df: pd.DataFrame, d_cols: list[str]) -> tuple[list[dict[str, float]], np.ndarray]:
    summaries = _row_summaries(df, d_cols)
    return summaries, _feature_array(summaries)


def compute_metrics_for_store(store_df: pd.DataFrame, d_cols: list[str]) -> pd.DataFrame:
    rows = [compute_sku_screening_metrics(row, row[d_cols].to_numpy(dtype=float)) for _, row in store_df.iterrows()]
    return pd.DataFrame(rows)


def _sample_store_rows(store_df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    ordered = store_df.sort_values("item_id", kind="mergesort")
    if len(ordered) <= sample_size:
        return ordered
    positions = np.random.RandomState(seed).choice(np.arange(len(ordered)), size=sample_size, replace=False)
    return ordered.iloc[sorted(positions)].copy()


def scan_candidate_stores(sales: pd.DataFrame, d_cols: list[str], store_sample_size: int, random_seed: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    sampled_features: dict[str, np.ndarray] = {}
    sampled_states: dict[str, str] = {}
    foods3_round1_counts: dict[str, int] = {}
    for store_id, store_df in sales.groupby("store_id", sort=True):
        sampled = _sample_store_rows(store_df, store_sample_size, random_seed)
        _summaries, features = _row_features(sampled, d_cols)
        sampled_features[str(store_id)] = features
        sampled_states[str(store_id)] = str(store_df["state_id"].iloc[0])
        metrics = compute_metrics_for_store(store_df[store_df["dept_id"].astype(str).eq("FOODS_3")], d_cols)
        foods3_round1_counts[str(store_id)] = int(len(_eligible(metrics, ["FOODS_3"], 0.30, 1.5, 10)))
    rows = []
    for store_id, target_features in sampled_features.items():
        source_features = np.vstack([features for other, features in sampled_features.items() if other != store_id])
        store_mmd, gamma, _scaler = scaled_mmd(target_features, source_features)
        rows.append({
            "store_id": store_id,
            "state_id": sampled_states[store_id],
            "store_mmd": store_mmd,
            "store_mmd_gamma": gamma,
            "n_foods3_eligible_r1": foods3_round1_counts[store_id],
        })
    profile = pd.DataFrame(rows)
    values = profile["store_mmd"].to_numpy(dtype=float)
    q25 = _percentile(values, 25)
    median = _percentile(values, 50)
    q75 = _percentile(values, 75)
    profile["q25_mmd"] = q25
    profile["median_mmd"] = median
    profile["q75_mmd"] = q75
    profile["is_in_iqr"] = profile["store_mmd"].between(q25, q75, inclusive="both")
    profile["abs_mmd_distance"] = (profile["store_mmd"] - median).abs()
    profile["state_priority"] = profile["state_id"].map(STATE_PRIORITY).fillna(99).astype(int)
    iqr = profile[profile["is_in_iqr"]].copy()
    ranked = iqr.sort_values(["abs_mmd_distance", "state_priority", "store_id"], ascending=[True, True, True])
    profile["rank_in_iqr"] = np.nan
    for rank, idx in enumerate(ranked.index, start=1):
        profile.loc[idx, "rank_in_iqr"] = rank
    return profile.sort_values("store_id").reset_index(drop=True), sampled_features


def _ranked_candidate_stores(profile: pd.DataFrame) -> pd.DataFrame:
    candidates = profile[profile["is_in_iqr"]].copy()
    if candidates.empty:
        candidates = profile.copy()
    return candidates.sort_values(["abs_mmd_distance", "state_priority", "store_id"], ascending=[True, True, True]).reset_index(drop=True)


def _feature_subset(df: pd.DataFrame, d_cols: list[str], max_rows: int | None = None, random_seed: int = 42) -> tuple[list[dict[str, float]], np.ndarray]:
    subset = df
    if max_rows is not None and len(subset) > max_rows:
        ordered = subset.sort_values(["store_id", "item_id"], kind="mergesort")
        positions = np.random.RandomState(random_seed).choice(np.arange(len(ordered)), size=max_rows, replace=False)
        subset = ordered.iloc[sorted(positions)]
    return _row_features(subset, d_cols)


def _assemble_target_metrics(sales: pd.DataFrame, store_id: str, d_cols: list[str], selected_skus: list[str]) -> pd.DataFrame:
    store_df = sales[sales["store_id"].astype(str).eq(str(store_id))]
    metrics = compute_metrics_for_store(store_df[store_df["dept_id"].isin(["FOODS_1", "FOODS_2", "FOODS_3"])], d_cols)
    metrics["is_selected"] = metrics["item_id"].astype(str).isin(selected_skus)
    return metrics


def _write_report(payload: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# D6 Target Selection Report",
        "",
        f"- Target store: `{payload['target_store']}`",
        f"- Target department: `{payload['target_department']}`",
        f"- Target SKUs: `{payload['target_skus']}`",
        f"- Fallback round: `{payload['fallback_round']}`",
        f"- Source entities: `{len(payload['source_entities'])}`",
        f"- Store MMD: `{payload['store_mmd']}`",
        f"- Final MMD: `{payload['final_mmd']}`",
        f"- Permutation p-value: `{payload['permutation_p_value']}`",
        "",
        "## Windows",
        "",
        f"- Target train: `{payload['train_start']}` to `{payload['train_end']}`",
        f"- Target validation: `{payload['val_start']}` to `{payload['val_end']}`",
        f"- Target test: `{payload['test_start']}` to `{payload['test_end']}`",
        f"- Source history: `{payload['source_start']}` to `{payload['source_end']}`",
        "",
        "## Structural Shift",
        "",
        f"Semantics: `{payload['structural_shift_semantics']}`",
    ]
    for key, value in payload["structural_shift"].items():
        lines.append(f"- `{key}`: `{value}`")
    (output_dir / "target_selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_target_selection(
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    random_seed: int = 42,
    permutation_seed: int = 43,
    n_perm: int = 500,
    store_sample_size: int = 50,
    source_history_days: int = 300,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sales, calendar = _load_inputs(dataset_root)
    d_cols = d_cols_from_frame(sales)
    max_date = resolve_max_date(calendar, n_d_cols=len(d_cols))
    expected = pd.Timestamp(WindowConfig().expected_max_date)
    if len(d_cols) == 1941 and max_date != expected:
        raise ValueError(f"max_date expected {expected.date()}, got {max_date.date()}")
    cfg = WindowConfig(source_history_days=source_history_days)
    windows = compute_target_windows(max_date, cfg)
    source_window = compute_source_window(windows["train_start"], source_history_days)
    screening_d_cols = get_screening_d_cols(calendar, windows["train_end"])
    available = set(d_cols)
    screening_d_cols = [col for col in screening_d_cols if col in available]
    if not screening_d_cols:
        raise ValueError("No screening d columns found in sales table")
    store_profile, _sampled = scan_candidate_stores(sales, screening_d_cols, store_sample_size, random_seed)
    fallback_attempts: list[dict[str, Any]] = []
    selected_result: dict[str, Any] | None = None
    target_metrics = pd.DataFrame()
    store_selection_rank = 0
    for rank, store_row in _ranked_candidate_stores(store_profile).iterrows():
        store_id = str(store_row["store_id"])
        store_df = sales[sales["store_id"].astype(str).eq(store_id)]
        metrics = compute_metrics_for_store(store_df, screening_d_cols)
        result = select_target_skus_with_fallback(metrics)
        for attempt in result["attempts"]:
            fallback_attempts.append({"store_id": store_id, **attempt})
        if len(result["target_skus"]) >= 3:
            selected_result = {"store": store_row.to_dict(), "selection": result}
            target_metrics = _assemble_target_metrics(sales, store_id, screening_d_cols, result["target_skus"])
            store_selection_rank = int(rank) + 1
            break
    if selected_result is None:
        raise ValueError("No D6 store produced at least 3 target SKUs after fallback")
    target_store = str(selected_result["store"]["store_id"])
    target_department = str(selected_result["selection"]["target_department"])
    target_skus = [str(sku) for sku in selected_result["selection"]["target_skus"]]
    target_df = sales[(sales["store_id"].astype(str).eq(target_store)) & (sales["item_id"].astype(str).isin(target_skus))]
    source_df = sales[(~sales["store_id"].astype(str).eq(target_store)) & (sales["dept_id"].astype(str).eq(target_department))]
    target_summaries, target_features = _row_features(target_df, screening_d_cols)
    source_summaries, source_features = _feature_subset(source_df, screening_d_cols, max_rows=1000, random_seed=random_seed)
    final_mmd, final_gamma, scaler = scaled_mmd(target_features, source_features)
    p_value = permutation_p_value(scaler.transform(target_features), scaler.transform(source_features), final_gamma, n_perm=n_perm, random_state=permutation_seed)
    shift, target_shift_summary, source_shift_summary = compute_structural_shift(target_summaries, source_summaries)
    source_entities = build_source_entities(sales, target_store=target_store, target_department=target_department)
    target_entities = [{"store_id": target_store, "item_id": sku} for sku in target_skus]
    payload = {
        "target_store": target_store,
        "target_department": target_department,
        "target_skus": target_skus,
        "fallback_round": int(selected_result["selection"]["fallback_round"]),
        "target_entities": target_entities,
        "source_entities": source_entities,
        "store_mmd": float(selected_result["store"]["store_mmd"]),
        "final_mmd": float(final_mmd),
        "final_mmd_gamma": float(final_gamma),
        "permutation_p_value": float(p_value),
        "store_selection_rank": store_selection_rank,
        "fallback_attempts": fallback_attempts,
        "q25_mmd": float(selected_result["store"]["q25_mmd"]),
        "median_mmd": float(selected_result["store"]["median_mmd"]),
        "q75_mmd": float(selected_result["store"]["q75_mmd"]),
        "structural_shift": shift,
        "structural_shift_semantics": "signed_target_minus_source",
        "target_structural_summary": target_shift_summary,
        "source_structural_summary": source_shift_summary,
        **windows,
        **source_window,
        "selection_rules": {
            "round_1": {"dept": "FOODS_3", "nonzero_ratio": 0.30, "cv": 1.5, "spike_ratio": 10},
            "round_2": {"dept": "FOODS_3", "nonzero_ratio": 0.20, "cv": 2.0, "spike_ratio": 15},
            "round_3": {"dept": "FOODS_*", "nonzero_ratio": 0.30, "cv": 1.5, "spike_ratio": 10},
        },
        "random_seed": int(random_seed),
        "permutation_seed": int(permutation_seed),
        "n_perm": int(n_perm),
        "store_sample_size": int(store_sample_size),
        "feature_columns": FEATURE_COLUMNS,
    }
    store_profile.drop(columns=["state_priority"], errors="ignore").to_csv(output_dir / "store_candidate_profile.csv", index=False, encoding="utf-8-sig")
    metric_cols = [
        "item_id",
        "dept_id",
        "cat_id",
        "nonzero_ratio",
        "cv",
        "spike_ratio",
        "mean",
        "std",
        "zero_ratio",
        "iqr",
        "acf_lag7",
        "trend_slope",
        "is_selected",
    ]
    remaining_metric_cols = [col for col in target_metrics.columns if col not in metric_cols]
    target_metrics = target_metrics[[col for col in metric_cols if col in target_metrics.columns] + remaining_metric_cols]
    target_metrics.to_csv(output_dir / "target_sku_metrics.csv", index=False, encoding="utf-8-sig")
    json_payload = _jsonable(payload)
    (output_dir / "target_selection_result.json").write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(json_payload, output_dir)
    return json_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Dataset6 M5 target store/SKU set.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--permutation-seed", type=int, default=43)
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--store-sample-size", type=int, default=50)
    parser.add_argument("--source-history-days", type=int, default=300)
    args = parser.parse_args()
    result = run_target_selection(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
        permutation_seed=args.permutation_seed,
        n_perm=args.n_perm,
        store_sample_size=args.store_sample_size,
        source_history_days=args.source_history_days,
    )
    print(
        "D6 target selection saved: "
        f"target_store={result['target_store']}, "
        f"target_department={result['target_department']}, "
        f"target_skus={len(result['target_skus'])}, "
        f"source_entities={len(result['source_entities'])}, "
        f"store_mmd={result['store_mmd']}, "
        f"final_mmd={result['final_mmd']}, "
        f"output_dir={args.output_dir}"
    )


if __name__ == "__main__":
    main()
