"""Dataset1 small audit for RMSE, MAE, and MAPE.

This audit is intentionally scoped to Dataset1 and writes only under
outputs/audits/. It first scans existing result files for row-level
y_true/y_pred-like columns. If no usable Dataset1 prediction file is found, it
runs a small Dataset1-only method set through existing training/prediction
functions and records missing metrics when a method entry point does not expose
the prediction arrays needed for MAPE.
"""

from __future__ import annotations

import ast
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

OUT_DIR = ROOT / "outputs" / "audits"
DETAILS_CSV = OUT_DIR / "dataset1_mae_rmse_mape_small_experiment_details.csv"
SUMMARY_CSV = OUT_DIR / "dataset1_mae_rmse_mape_small_experiment_summary.csv"
REPORT_MD = OUT_DIR / "dataset1_mae_rmse_mape_small_experiment.md"

DATASET_ID = "d1"
DATASET = "Dataset1"
SEED = 42
HORIZON = 1
K = 3
METHODS_WITHOUT = ["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL-RFE"]
METHODS_WITH = ["SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL-RFE"]
SCENARIOS = ["without_information_sharing", "with_information_sharing"]

DETAIL_COLUMNS = [
    "dataset_id",
    "dataset",
    "method",
    "model_name",
    "info_sharing",
    "metric_space",
    "seed",
    "horizon",
    "y_true_count",
    "zero_y_true_count",
    "zero_y_true_ratio",
    "rmse",
    "mae",
    "mape_exclude_zero",
    "mape_note",
    "source_file",
    "run_status",
    "error_message",
    "n_test_samples",
    "notes",
]


def _read_default_config() -> dict[str, Any]:
    with (ROOT / "configs" / "default_config.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _as_1d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr.reshape(-1)


def _metrics_from_arrays(y_true: Any, y_pred: Any) -> dict[str, Any]:
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true/y_pred length mismatch: {yt.shape[0]} vs {yp.shape[0]}")
    if yt.shape[0] == 0:
        raise ValueError("empty y_true/y_pred arrays")

    diff = yt - yp
    zero_mask = yt == 0
    nonzero = ~zero_mask
    zero_count = int(zero_mask.sum())
    if int(nonzero.sum()) == 0:
        mape = math.nan
        mape_note = "MAPE undefined because all y_true values are zero."
    else:
        mape = float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])) * 100.0)
        mape_note = "MAPE = mean(abs((y_true - y_pred) / y_true)) * 100, excluding y_true == 0."

    return {
        "y_true_count": int(yt.shape[0]),
        "zero_y_true_count": zero_count,
        "zero_y_true_ratio": float(zero_count / yt.shape[0]),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "mape_exclude_zero": mape,
        "mape_note": mape_note,
        "n_test_samples": int(yt.shape[0]),
    }


def _inverse_sales(values: Any, scaler: Any, feature_columns: Any) -> np.ndarray:
    cols = list(feature_columns)
    if "sales" not in cols:
        raise ValueError("sales column unavailable for original-space inverse transform")
    idx = cols.index("sales")
    scale = np.asarray(getattr(scaler, "scale_"), dtype=np.float64).reshape(-1)
    offset = np.asarray(getattr(scaler, "min_"), dtype=np.float64).reshape(-1)
    if idx >= len(scale) or idx >= len(offset):
        raise ValueError("sales scaler does not contain the sales feature index")
    if np.isclose(float(scale[idx]), 0.0):
        data_min = getattr(scaler, "data_min_", None)
        if data_min is None or idx >= len(data_min):
            return _as_1d(values)
        return np.full_like(_as_1d(values), float(np.asarray(data_min, dtype=np.float64)[idx]))
    return (_as_1d(values) - float(offset[idx])) / float(scale[idx])


def _blank_row(method: str, scenario: str, metric_space: str, status: str, source_file: str, notes: str = "") -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "dataset": DATASET,
        "method": method,
        "model_name": method,
        "info_sharing": scenario,
        "metric_space": metric_space,
        "seed": SEED,
        "horizon": HORIZON,
        "y_true_count": 0,
        "zero_y_true_count": 0,
        "zero_y_true_ratio": math.nan,
        "rmse": math.nan,
        "mae": math.nan,
        "mape_exclude_zero": math.nan,
        "mape_note": "",
        "source_file": source_file,
        "run_status": status,
        "error_message": "",
        "n_test_samples": 0,
        "notes": notes,
    }


def _rows_from_prediction_arrays(
    *,
    method: str,
    scenario: str,
    model_name: str,
    y_true: Any,
    y_pred: Any,
    source_file: str,
    notes: str,
    scaler: Any | None,
    feature_columns: Any | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_space, yt, yp in [
        ("normalized_minmax_space", y_true, y_pred),
        (
            "original_sales_space",
            _inverse_sales(y_true, scaler, feature_columns) if scaler is not None and feature_columns is not None else None,
            _inverse_sales(y_pred, scaler, feature_columns) if scaler is not None and feature_columns is not None else None,
        ),
    ]:
        if yt is None or yp is None:
            row = _blank_row(method, scenario, metric_space, "missing", source_file, notes)
            row["error_message"] = "No scaler/feature_columns available for original-space MAPE."
            rows.append(row)
            continue
        metric = _metrics_from_arrays(yt, yp)
        row = _blank_row(method, scenario, metric_space, "success", source_file, notes)
        row.update(metric)
        row["model_name"] = model_name
        rows.append(row)
    return rows


def _extract_literal_array(series: pd.Series) -> np.ndarray | None:
    values: list[float] = []
    for item in series.dropna():
        if isinstance(item, (int, float, np.number)):
            values.append(float(item))
            continue
        text = str(item).strip()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
        arr = np.asarray(parsed, dtype=np.float64).reshape(-1)
        values.extend([float(v) for v in arr])
    return np.asarray(values, dtype=np.float64) if values else None


def find_existing_prediction_rows() -> tuple[list[dict[str, Any]], list[str]]:
    """Find existing Dataset1 prediction files with explicit y_true/y_pred columns."""
    rows: list[dict[str, Any]] = []
    inspected: list[str] = []
    true_names = {"y_true", "actual", "actuals", "y_test", "test_y"}
    pred_names = {"y_pred", "prediction", "predictions", "y_hat", "forecast"}

    for path in sorted((ROOT / "outputs").rglob("*.csv")):
        if "audits/dataset1_mae_rmse_mape_small_experiment" in str(path):
            continue
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        lower = {str(c).lower(): c for c in header.columns}
        true_col = next((lower[c] for c in true_names if c in lower), None)
        pred_col = next((lower[c] for c in pred_names if c in lower), None)
        if true_col is None or pred_col is None:
            continue
        inspected.append(str(path.relative_to(ROOT)))
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        dataset_col = next((c for c in df.columns if str(c).lower() in {"dataset", "dataset_name", "dataset_id"}), None)
        if dataset_col is not None:
            df = df[df[dataset_col].astype(str).str.contains("Dataset1|d1", case=False, na=False)]
        if df.empty:
            continue
        yt = _extract_literal_array(df[true_col])
        yp = _extract_literal_array(df[pred_col])
        if yt is None or yp is None:
            continue
        method = str(df.get("method", pd.Series(["unknown"])).dropna().iloc[0])
        scenario = str(df.get("information_sharing", df.get("info_sharing", pd.Series(["unknown"]))).dropna().iloc[0])
        rows.extend(
            _rows_from_prediction_arrays(
                method=method,
                scenario=scenario,
                model_name=method,
                y_true=yt,
                y_pred=yp,
                source_file=str(path.relative_to(ROOT)),
                notes="Reused existing row-level prediction file.",
                scaler=None,
                feature_columns=None,
            )
        )
    return rows, inspected


def _prepare_context() -> dict[str, Any]:
    from data_preprocessing import normalize_features, temporal_split_by_ratio_or_dates
    from experiment_runner import prepare_base_data_for_experiments
    from scripts.run_full_paper_experiments import (
        _apply_information_sharing_filter,
        _resolve_dataset_feature_cols,
        _scenario_to_bool,
        load_paper_protocol,
        resolve_strict_paper_mode,
    )

    cfg = _read_default_config()
    cfg["dataset_name"] = DATASET
    cfg.setdefault("single_experiment", {})
    cfg["single_experiment"]["dataset_name"] = DATASET
    cfg["single_experiment"]["source_epochs"] = min(int(cfg["single_experiment"].get("source_epochs", 2)), 2)
    cfg["single_experiment"]["target_epochs"] = min(int(cfg["single_experiment"].get("target_epochs", 2)), 2)
    cfg["single_experiment"]["horizon"] = HORIZON
    cfg["single_experiment"]["k"] = K

    protocol = load_paper_protocol(cfg)
    strict = resolve_strict_paper_mode(cfg, explicit=None)
    base = prepare_base_data_for_experiments(
        dataset_name=DATASET,
        data_path=cfg["dataset_paths"][DATASET],
        config=cfg,
        verbose_mode="summary",
    )
    target_df = base["target_df"].copy()
    feature_cols = _resolve_dataset_feature_cols(DATASET, base["source_df"], target_df, cfg)
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(target_df)
    _, _, _, target_scaler, target_feature_columns = normalize_features(train_df, val_df, test_df)
    return {
        "cfg": cfg,
        "protocol": protocol,
        "strict": strict,
        "base": base,
        "target_df": target_df,
        "feature_cols": feature_cols,
        "target_scaler": target_scaler,
        "target_feature_columns": target_feature_columns,
        "_apply_information_sharing_filter": _apply_information_sharing_filter,
        "_scenario_to_bool": _scenario_to_bool,
    }


def _source_for_scenario(ctx: dict[str, Any], scenario: str) -> pd.DataFrame:
    source_df = ctx["base"]["source_df"].copy()
    target_df = ctx["target_df"].copy()
    filtered = ctx["_apply_information_sharing_filter"](
        dataset_name=DATASET,
        source_df=source_df,
        target_df=target_df,
        use_information_sharing=ctx["_scenario_to_bool"](scenario),
        strict_paper_mode=bool(ctx["strict"]),
        protocol=ctx["protocol"],
        cfg=ctx["cfg"],
    )
    target_df.attrs["information_sharing_scenario"] = filtered.attrs.get("information_sharing_scenario", "")
    ctx["target_df"] = target_df
    return filtered


def _run_notl(ctx: dict[str, Any], scenario: str) -> list[dict[str, Any]]:
    import tensorflow as tf
    from data_preprocessing import build_tabular_sequence, normalize_features, temporal_split_by_ratio_or_dates, to_cnn_tensor
    from src.models.no_tl_model import build_no_tl_cnn_model
    from src.utils.runtime_control import keras_verbose

    cfg = ctx["cfg"]["single_experiment"]
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(ctx["target_df"])
    train_s, val_s, test_s, scaler, feature_columns = normalize_features(train_df, val_df, test_df)
    x_train, y_train = build_tabular_sequence(train_s, horizon=HORIZON, window_size=int(cfg["window_size"]))
    x_val, y_val = build_tabular_sequence(val_s, horizon=HORIZON, window_size=int(cfg["window_size"]))
    x_test, y_test = build_tabular_sequence(test_s, horizon=HORIZON, window_size=int(cfg["window_size"]))
    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("No-TL produced empty train/test windows.")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = build_no_tl_cnn_model(input_shape=to_cnn_tensor(x_train).shape[1:], learning_rate=float(cfg["learning_rate"]))
    fit_kwargs: dict[str, Any] = {
        "epochs": int(cfg["target_epochs"]),
        "batch_size": int(cfg["batch_size"]),
        "verbose": keras_verbose(),
    }
    if len(y_val) > 0:
        fit_kwargs["validation_data"] = (to_cnn_tensor(x_val), y_val)
    model.fit(to_cnn_tensor(x_train), y_train, **fit_kwargs)
    y_pred = model.predict(to_cnn_tensor(x_test), verbose=0)
    return _rows_from_prediction_arrays(
        method="No-TL",
        scenario=scenario,
        model_name="No-TL audit-local existing CNN",
        y_true=y_test,
        y_pred=y_pred,
        source_file="audit-local minimal No-TL run",
        notes="Ran audit-local No-TL using existing split, sequence, CNN, and fit functions.",
        scaler=scaler,
        feature_columns=feature_columns,
    )


def _run_ss(ctx: dict[str, Any], source_df: pd.DataFrame, scenario: str) -> list[dict[str, Any]]:
    from data_preprocessing import temporal_split_by_ratio_or_dates
    from mswa_tl import run_single_source_tl_for_mswa
    from source_selector import SourceSelector

    cfg = ctx["cfg"]["single_experiment"]
    selector = SourceSelector()
    selection = selector.select_top_k_sources(
        target_df=ctx["target_df"],
        source_df=source_df.sort_values(["entity_id", "item_id", "date"]).copy(),
        feature_cols=ctx["feature_cols"],
        k=1,
        weight_mode=str(cfg["weight_mode"]),
        include_sales_in_knn=True,
    )
    selected = selection.get("sources", [])
    if not selected:
        raise ValueError("SS-TL KNN source selection returned no source.")
    key = tuple(selected[0]["source_key"])
    single_source_df = source_df[(source_df["entity_id"] == key[0]) & (source_df["item_id"] == key[1])].copy()
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(ctx["target_df"])
    raw = run_single_source_tl_for_mswa(
        source_sequence_df=single_source_df,
        target_train_df=train_df,
        target_val_df=val_df,
        target_test_df=test_df,
        feature_cols=ctx["feature_cols"],
        horizon=HORIZON,
        window_size=int(cfg["window_size"]),
        learning_rate=float(cfg["learning_rate"]),
        source_epochs=int(cfg["source_epochs"]),
        target_epochs=int(cfg["target_epochs"]),
        batch_size=int(cfg["batch_size"]),
        metric_protocol=ctx["protocol"].get("metric_protocol", {}),
    )
    return _rows_from_prediction_arrays(
        method="SS-TL",
        scenario=scenario,
        model_name="SS-TL via existing single-source TL routine",
        y_true=raw["y_test"],
        y_pred=raw["y_pred"],
        source_file="audit-local minimal SS-TL run",
        notes=f"Selected source={key}; reused existing single-source TL training routine.",
        scaler=ctx["target_scaler"],
        feature_columns=ctx["target_feature_columns"],
    )


def _run_mswa(ctx: dict[str, Any], source_df: pd.DataFrame, scenario: str) -> list[dict[str, Any]]:
    from data_preprocessing import temporal_split_by_ratio_or_dates
    from mswa_tl import run_single_source_tl_for_mswa, weighted_prediction_fusion
    from source_selector import SourceSelector

    cfg = ctx["cfg"]["single_experiment"]
    selector = SourceSelector()
    selection = selector.select_top_k_sources(
        target_df=ctx["target_df"],
        source_df=source_df,
        feature_cols=ctx["feature_cols"],
        k=K,
        weight_mode=str(cfg["weight_mode"]),
        include_sales_in_knn=True,
    )
    selected = selection.get("sources", [])
    if not selected:
        raise ValueError("MSWA-TL source selection returned no source.")
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(ctx["target_df"])
    individuals = []
    for source in selected:
        key = tuple(source["source_key"])
        single_source_df = source_df[(source_df["entity_id"] == key[0]) & (source_df["item_id"] == key[1])].copy()
        one = run_single_source_tl_for_mswa(
            source_sequence_df=single_source_df,
            target_train_df=train_df,
            target_val_df=val_df,
            target_test_df=test_df,
            feature_cols=ctx["feature_cols"],
            horizon=HORIZON,
            window_size=int(cfg["window_size"]),
            learning_rate=float(cfg["learning_rate"]),
            source_epochs=int(cfg["source_epochs"]),
            target_epochs=int(cfg["target_epochs"]),
            batch_size=int(cfg["batch_size"]),
            metric_protocol=ctx["protocol"].get("metric_protocol", {}),
        )
        one["weight"] = float(source["weight"])
        one["source_key"] = key
        individuals.append(one)
    y_true = np.asarray(individuals[0]["y_test"])
    y_pred = weighted_prediction_fusion([r["y_pred"] for r in individuals], [r["weight"] for r in individuals])
    return _rows_from_prediction_arrays(
        method="MSWA-TL",
        scenario=scenario,
        model_name="MSWA-TL existing single-source weighted fusion routines",
        y_true=y_true,
        y_pred=y_pred,
        source_file="audit-local minimal MSWA-TL run",
        notes=f"Selected sources={[r['source_key'] for r in individuals]}; fused existing per-source predictions with KNN weights.",
        scaler=ctx["target_scaler"],
        feature_columns=ctx["target_feature_columns"],
    )


def _run_mssb(ctx: dict[str, Any], source_df: pd.DataFrame, scenario: str) -> list[dict[str, Any]]:
    from data_preprocessing import temporal_split_by_ratio_or_dates
    from mssb_tl import run_single_source_tl_for_mssb
    from source_selector import SourceSelector

    cfg = ctx["cfg"]["single_experiment"]
    selector = SourceSelector()
    selection = selector.select_top_k_sources(
        target_df=ctx["target_df"],
        source_df=source_df,
        feature_cols=ctx["feature_cols"],
        k=K,
        weight_mode=str(cfg["weight_mode"]),
        include_sales_in_knn=True,
    )
    selected = selection.get("sources", [])
    if not selected:
        raise ValueError("MSSB-TL source selection returned no source.")
    train_df, val_df, test_df = temporal_split_by_ratio_or_dates(ctx["target_df"])
    results = []
    for source in selected:
        key = tuple(source["source_key"])
        single_source_df = source_df[(source_df["entity_id"] == key[0]) & (source_df["item_id"] == key[1])].copy()
        one = run_single_source_tl_for_mssb(
            source_sequence_df=single_source_df,
            target_train_df=train_df,
            target_val_df=val_df,
            target_test_df=test_df,
            feature_cols=ctx["feature_cols"],
            horizon=HORIZON,
            window_size=int(cfg["window_size"]),
            learning_rate=float(cfg["learning_rate"]),
            source_epochs=int(cfg["source_epochs"]),
            target_epochs=int(cfg["target_epochs"]),
            batch_size=int(cfg["batch_size"]),
            metric_protocol=ctx["protocol"].get("metric_protocol", {}),
        )
        one["source_key"] = key
        results.append(one)
    best = min(results, key=lambda r: float(r["val_rmse"]))
    return _rows_from_prediction_arrays(
        method="MSSB-TL",
        scenario=scenario,
        model_name="MSSB-TL existing single-source switching routine",
        y_true=best["y_test_true"],
        y_pred=best["y_test_pred"],
        source_file="audit-local minimal MSSB-TL run",
        notes=f"Best validation source={best['source_key']}; reused existing MSSB single-source routine.",
        scaler=ctx["target_scaler"],
        feature_columns=ctx["target_feature_columns"],
    )


def _run_msml_rfe(ctx: dict[str, Any], source_df: pd.DataFrame, scenario: str) -> list[dict[str, Any]]:
    from msml_tl_rfe import run_msml_tl_rfe

    cfg = ctx["cfg"]["single_experiment"]
    raw = run_msml_tl_rfe(
        source_df=source_df,
        target_df=ctx["target_df"],
        feature_cols=ctx["feature_cols"],
        k=K,
        number_of_sources=K,
        horizon=HORIZON,
        window_size=int(cfg["window_size"]),
        weight_mode=str(cfg["weight_mode"]),
        estimator_name=str(cfg["estimator_name"]),
        keep_ratio=float(cfg["keep_ratio"]),
        include_sales_in_knn=True,
        learning_rate=float(cfg["learning_rate"]),
        source_epochs=int(cfg["source_epochs"]),
        target_epochs=int(cfg["target_epochs"]),
        batch_size=int(cfg["batch_size"]),
        random_state=SEED,
        metric_protocol=ctx["protocol"].get("metric_protocol", {}),
        source_selection_window="target_observed_window",
    )
    fused = raw.get("fused_result", {})
    rows = []
    metric_pairs = [
        ("normalized_minmax_space", "normalized_rmse", "normalized_mae"),
        ("original_sales_space", "original_scale_rmse", "original_scale_mae"),
    ]
    for metric_space, rmse_key, mae_key in metric_pairs:
        row = _blank_row(
            "MSML-TL-RFE",
            scenario,
            metric_space,
            "missing_predictions",
            "audit-local minimal MSML-TL-RFE run",
            "Existing MSML-TL-RFE entry point exposes RMSE/MAE but not y_true/y_pred arrays; MAPE not computed.",
        )
        row["model_name"] = "MSML-TL-RFE existing run_msml_tl_rfe"
        row["rmse"] = _safe_float(fused.get(rmse_key, fused.get("rmse") if metric_space == "normalized_minmax_space" else None))
        row["mae"] = _safe_float(fused.get(mae_key, fused.get("mae") if metric_space == "normalized_minmax_space" else None))
        row["n_test_samples"] = int(ast.literal_eval(str(fused.get("prediction_shape", "(0,)")))[0]) if fused.get("prediction_shape") else 0
        row["y_true_count"] = row["n_test_samples"]
        row["mape_note"] = "MAPE requires y_true/y_pred arrays; existing MSML-TL-RFE return value does not expose them."
        rows.append(row)
    return rows


def run_minimal_experiment() -> list[dict[str, Any]]:
    ctx = _prepare_context()
    runners: dict[str, Callable[[dict[str, Any], pd.DataFrame, str], list[dict[str, Any]]]] = {
        "SS-TL": _run_ss,
        "MSWA-TL": _run_mswa,
        "MSSB-TL": _run_mssb,
        "MSML-TL-RFE": _run_msml_rfe,
    }
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        source_df = _source_for_scenario(ctx, scenario)
        methods = METHODS_WITHOUT if scenario == "without_information_sharing" else METHODS_WITH
        for method in methods:
            print(f"[audit] Dataset1 scenario={scenario} method={method}")
            start = time.time()
            try:
                if method == "No-TL":
                    method_rows = _run_notl(ctx, scenario)
                else:
                    method_rows = runners[method](ctx, source_df, scenario)
                for row in method_rows:
                    row["notes"] = (str(row.get("notes", "")) + f" Runtime seconds={time.time() - start:.2f}.").strip()
                rows.extend(method_rows)
            except Exception as exc:
                for metric_space in ["normalized_minmax_space", "original_sales_space"]:
                    row = _blank_row(method, scenario, metric_space, "failed", "audit-local minimal run")
                    row["error_message"] = repr(exc)
                    rows.append(row)
                print(f"[audit] failed scenario={scenario} method={method}: {exc!r}")
    for metric_space in ["normalized_minmax_space", "original_sales_space"]:
        rows.append(
            _blank_row(
                "No-TL",
                "with_information_sharing",
                metric_space,
                "skipped",
                "not run",
                "No-TL is target-only and was requested only for the no-information-sharing method set.",
            )
        )
    return rows


def build_summary(details: pd.DataFrame) -> pd.DataFrame:
    success_like = details[details["run_status"].isin(["success", "missing_predictions"])].copy()
    grouped = (
        success_like.groupby(["method", "info_sharing", "metric_space", "horizon"], dropna=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            mape_exclude_zero_mean=("mape_exclude_zero", "mean"),
            run_count=("run_status", "count"),
        )
        .reset_index()
    )
    return grouped


def _fmt(value: Any, digits: int = 6) -> str:
    val = _safe_float(value)
    if math.isnan(val):
        return ""
    return f"{val:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            value = row[col]
            vals.append(_fmt(value) if isinstance(value, (float, int, np.floating, np.integer)) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(details: pd.DataFrame, summary: pd.DataFrame, reused_existing: bool, inspected: list[str]) -> None:
    lines: list[str] = [
        "# Dataset1 MAE/RMSE/MAPE Small Experiment",
        "",
        "## 实验目的",
        "",
        "只针对 Dataset1（d1）生成无信息共享和有信息共享场景下主要方法的 RMSE、MAE、MAPE 审计结果，不修改主实验逻辑，不覆盖主实验输出。",
        "",
        "## 复用或重跑说明",
        "",
    ]
    if reused_existing:
        lines.append("本次复用了现有逐样本预测结果文件中的 y_true/y_pred。")
    else:
        lines.append("未在现有结果中找到 Dataset1 可直接使用的逐样本 y_true/y_pred/predictions/actuals 文件，因此运行了 Dataset1-only 最小审计实验。")
    lines.extend(
        [
            "",
            f"- 已扫描到候选逐样本文件: {', '.join(inspected) if inspected else '无'}",
            f"- seed: {SEED}",
            f"- horizon: {HORIZON}",
            f"- k: {K}",
            "",
            "## 方法覆盖范围",
            "",
            "- 无信息共享: No-TL, SS-TL, MSWA-TL, MSSB-TL, MSML-TL-RFE。",
            "- 有信息共享: SS-TL, MSWA-TL, MSSB-TL, MSML-TL-RFE。",
            "- No-TL 是 target-only 方法，不使用 source pool；本报告按用户方法范围只在无信息共享集合列出，with_information_sharing 行标记 skipped。",
            "- MSSA-TL 在当前项目主要入口中未发现稳定独立方法名；使用项目已有 MSSB-TL 入口覆盖 switching-based 方法。",
            "",
            "## MAPE 公式与零值处理",
            "",
            "项目现有统一指标函数未提供 MAPE 字段。本审计按用户指定公式计算: `MAPE = mean(abs((y_true - y_pred) / y_true)) * 100`。",
            "",
            "为避免除以 0，计算 `mape_exclude_zero` 时排除 `y_true == 0` 的样本，并同时输出 `zero_y_true_count` 和 `zero_y_true_ratio`。如果方法没有暴露 y_true/y_pred，则不计算 MAPE。",
            "",
        ]
    )
    display_cols = [
        "method",
        "seed",
        "horizon",
        "rmse",
        "mae",
        "mape_exclude_zero",
        "zero_y_true_count",
        "zero_y_true_ratio",
        "n_test_samples",
        "run_status",
    ]
    for metric_space in sorted(details["metric_space"].dropna().unique()):
        lines.extend(["", f"## 无信息共享结果表 - {metric_space}", ""])
        sub = details[(details["info_sharing"] == "without_information_sharing") & (details["metric_space"] == metric_space)].copy()
        sub = sub.sort_values("rmse", na_position="last")
        lines.append(markdown_table(sub, display_cols))
        lines.extend(["", f"## 有信息共享结果表 - {metric_space}", ""])
        sub = details[(details["info_sharing"] == "with_information_sharing") & (details["metric_space"] == metric_space)].copy()
        sub = sub.sort_values("rmse", na_position="last")
        lines.append(markdown_table(sub, display_cols))
    missing = details[~details["run_status"].isin(["success"])]
    lines.extend(["", "## 缺失、失败或跳过", ""])
    if missing.empty:
        lines.append("无。")
    else:
        lines.append(markdown_table(missing, ["method", "info_sharing", "metric_space", "run_status", "error_message", "notes"]))
    lines.extend(["", "## 输出文件", "", f"- `{DETAILS_CSV.relative_to(ROOT)}`", f"- `{SUMMARY_CSV.relative_to(ROOT)}`", f"- `{REPORT_MD.relative_to(ROOT)}`"])
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_terminal_tables(details: pd.DataFrame) -> None:
    cols = ["method", "metric_space", "rmse", "mae", "mape_exclude_zero", "run_status"]
    print("\n新增脚本路径:")
    print(f"- {ROOT / 'scripts/audits/dataset1_mae_rmse_mape_small_experiment.py'}")
    print("新增 CSV/MD 路径:")
    for path in [DETAILS_CSV, SUMMARY_CSV, REPORT_MD]:
        print(f"- {path}")
    for scenario, title in [
        ("without_information_sharing", "Dataset1 无信息共享 MAE/RMSE/MAPE 表"),
        ("with_information_sharing", "Dataset1 有信息共享 MAE/RMSE/MAPE 表"),
    ]:
        print(f"\n{title}")
        sub = details[details["info_sharing"] == scenario].copy().sort_values(["metric_space", "rmse"], na_position="last")
        if sub.empty:
            print("(no rows)")
        else:
            print(sub[cols].to_string(index=False))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_rows, inspected = find_existing_prediction_rows()
    reused_existing = bool(existing_rows)
    rows = existing_rows if reused_existing else run_minimal_experiment()
    details = pd.DataFrame(rows)
    for col in DETAIL_COLUMNS:
        if col not in details.columns:
            details[col] = np.nan
    details = details[DETAIL_COLUMNS]
    summary = build_summary(details)
    details.to_csv(DETAILS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    write_report(details, summary, reused_existing=reused_existing, inspected=inspected)
    print_terminal_tables(details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
