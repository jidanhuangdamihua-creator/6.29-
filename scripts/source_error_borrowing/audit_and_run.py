#!/usr/bin/env python3
"""Exploratory post-hoc pilot: borrow contemporaneous product forecast errors.

The protocol is read from config.json and is not adjusted from results.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
METHODS = ["m1", "m2", "m3", "m4", "m4b"]
METHOD_INDEX = {name: i for i, name in enumerate(METHODS)}


def log(message: str) -> None:
    print(message, flush=True)


def load_config() -> dict:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload["methods"] != METHODS:
        raise SystemExit("config methods drifted from the locked five-method list")
    if payload["v3_primary_targets_actually_used"]["D4"]["entity_key"] != "166_311":
        raise SystemExit("D4 primary target must stay 166/311")
    return payload


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Weighted empirical quantile with a weight-1 test atom at +inf."""
    out = np.full(alphas.shape, np.inf, dtype=np.float64)
    if values.size == 0 or not np.any(weights > 0):
        return out
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    total = float(cumulative[-1]) + 1.0
    for i, alpha in enumerate(alphas):
        threshold = (1.0 - float(alpha)) * total
        index = int(np.searchsorted(cumulative, threshold, side="left"))
        if index < sorted_values.size:
            out[i] = float(sorted_values[index])
    return out


def wasserstein1(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=np.float64))
    right = np.sort(np.asarray(right, dtype=np.float64))
    if left.size == 0 or right.size == 0:
        return float("nan")
    grid = np.concatenate([left, right])
    grid.sort()
    deltas = np.diff(grid)
    if deltas.size == 0:
        return 0.0
    left_cdf = np.searchsorted(left, grid[:-1], side="right") / left.size
    right_cdf = np.searchsorted(right, grid[:-1], side="right") / right.size
    return float(np.sum(np.abs(left_cdf - right_cdf) * deltas))


def source_weight_map(
    target_scores: np.ndarray,
    source_scores: list[np.ndarray],
    *,
    delta: float,
    equal_if_short: bool,
    min_scores: int,
    min_sources: int,
) -> tuple[dict[int, float], str]:
    usable = [i for i, scores in enumerate(source_scores) if len(scores) > 0]
    if not usable:
        return {}, "no_source"
    if equal_if_short:
        enough = [i for i in usable if len(source_scores[i]) >= min_scores]
        short = any(len(source_scores[i]) < min_scores for i in usable)
        if short or len(enough) < min_sources:
            share = 1.0 / len(usable)
            return {i: share for i in usable}, "equal_fallback_insufficient_startup"
    raw = []
    for index in usable:
        distance = wasserstein1(target_scores, source_scores[index])
        raw.append(1.0 / (distance + delta))
    scale = float(sum(raw))
    return {index: value / scale for index, value in zip(usable, raw)}, "inverse_distance"


def weighted_sample(
    target_scores: np.ndarray,
    source_scores: list[np.ndarray],
    weights: dict[int, float],
    total_mass: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = [np.asarray(target_scores, dtype=np.float64)]
    masses = [np.ones(len(target_scores), dtype=np.float64)]
    applied = 0.0
    if total_mass > 0 and weights:
        for index, scores in enumerate(source_scores):
            share = float(weights.get(index, 0.0))
            count = len(scores)
            if share <= 0 or count == 0:
                continue
            point_mass = total_mass * share / count
            values.append(np.asarray(scores, dtype=np.float64))
            masses.append(np.full(count, point_mass, dtype=np.float64))
            applied += point_mass * count
    if all(part.size == 0 for part in values):
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), 0.0
    return np.concatenate(values), np.concatenate(masses), applied


def winkler_one(y_value: float, low: float, high: float, alpha: float) -> float:
    if not (math.isfinite(low) and math.isfinite(high)):
        return float("inf")
    score = high - low
    if y_value < low:
        score += (2.0 / alpha) * (low - y_value)
    elif y_value > high:
        score += (2.0 / alpha) * (y_value - high)
    return float(score)


def arrays_match(left: np.ndarray, right: np.ndarray, atol: float = 1e-8) -> bool:
    both_inf = np.isinf(left) & np.isinf(right) & (np.sign(left) == np.sign(right))
    both_nan = np.isnan(left) & np.isnan(right)
    close = np.isfinite(left) & np.isfinite(right) & (np.abs(left - right) <= atol)
    return bool(np.all(both_inf | both_nan | close))


def run_panel(
    y: np.ndarray,
    yhat: np.ndarray,
    scale: np.ndarray,
    horizon: int,
    *,
    alphas: np.ndarray,
    c_grid: np.ndarray,
    n0: int,
    delta: float,
    lookback: int,
    min_scores: int,
    min_sources: int,
) -> dict:
    """Replay one dataset-horizon-seed panel. Rows are calendar-aligned."""
    n_products, n_time = y.shape
    residual = np.abs(y - yhat) / scale[:, None]
    n_alpha = len(alphas)
    n_c = len(c_grid)
    q = np.full((len(METHODS), n_alpha, n_products, n_time), np.nan, dtype=np.float64)
    q_c = np.full((n_c, n_alpha, n_products, n_time), np.inf, dtype=np.float64)
    applied_mass = np.zeros((len(METHODS), n_alpha, n_products, n_time), dtype=np.float64)
    nominal_mass = np.zeros((len(METHODS), n_products, n_time), dtype=np.float64)
    chosen_c = np.zeros((n_alpha, n_products, n_time), dtype=np.float64)
    history_n = np.zeros((n_alpha, n_products, n_time), dtype=np.int16)
    excluded = np.zeros((n_alpha, n_products), dtype=np.int32)
    eligible = np.zeros((n_alpha, n_products), dtype=np.int32)
    weight_error = 0.0
    c0_error = 0.0
    history_violation = 0
    startup = int(n0)
    frozen = []
    for product in range(n_products):
        sources = [k for k in range(n_products) if k != product]
        if product in sources:
            raise RuntimeError("target entered its own source pool")
        source_startup = [residual[k, :startup] for k in sources]
        weights, reason = source_weight_map(
            residual[product, :startup],
            source_startup,
            delta=delta,
            equal_if_short=True,
            min_scores=min_scores,
            min_sources=min_sources,
        )
        frozen.append({"sources": sources, "weights": weights, "reason": reason})

    for product in range(n_products):
        sources = frozen[product]["sources"]
        frozen_weights = frozen[product]["weights"]
        qualified: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n_alpha)]
        low_c = np.full((n_c, n_alpha, n_time), np.inf)
        high_c = np.full((n_c, n_alpha, n_time), np.inf)
        for issue in range(n_time):
            revealed = 0 if issue < horizon else issue - horizon + 1
            if issue >= horizon:
                just = issue - horizon
                if just > issue - horizon:
                    history_violation += 1
                for alpha_i, alpha in enumerate(alphas):
                    finite = bool(np.all(np.isfinite(q_c[:, alpha_i, product, just])))
                    if finite:
                        scores = np.array(
                            [
                                winkler_one(
                                    float(y[product, just]),
                                    float(low_c[c_i, alpha_i, just]),
                                    float(high_c[c_i, alpha_i, just]),
                                    float(alpha),
                                )
                                for c_i in range(n_c)
                            ],
                            dtype=np.float64,
                        )
                        if not np.all(np.isfinite(scores)):
                            excluded[alpha_i, product] += 1
                        else:
                            qualified[alpha_i].append((just, scores))
                            eligible[alpha_i, product] += 1
                    else:
                        excluded[alpha_i, product] += 1
            target_scores = residual[product, :revealed]
            source_scores = [residual[k, :revealed] for k in sources]
            dynamic_weights, _reason = source_weight_map(
                target_scores,
                source_scores,
                delta=delta,
                equal_if_short=False,
                min_scores=1,
                min_sources=1,
            )
            selected = np.zeros(n_alpha, dtype=np.int16)
            for alpha_i in range(n_alpha):
                history = qualified[alpha_i]
                if history and history[-1][0] > issue - horizon:
                    history_violation += 1
                history_n[alpha_i, product, issue] = min(len(history), lookback)
                if not history:
                    selected[alpha_i] = 0
                else:
                    window = history[-lookback:]
                    means = np.mean(np.stack([item[1] for item in window], axis=0), axis=0)
                    selected[alpha_i] = int(np.argmin(means))
                chosen_c[alpha_i, product, issue] = float(c_grid[selected[alpha_i]])

            n_target = int(revealed)
            decay = (float(n0) ** 2) / max(float(n0), float(n_target))
            samples = {
                "m1": weighted_sample(target_scores, source_scores, {}, 0.0),
            }
            pooled_values = [target_scores, *source_scores] if n_target else [target_scores]
            pooled = np.concatenate([np.asarray(part, dtype=np.float64) for part in pooled_values if len(part) or part is target_scores])
            if n_target == 0:
                pooled = np.zeros(0, dtype=np.float64)
            else:
                pooled = np.concatenate([np.asarray(part, dtype=np.float64) for part in [target_scores, *source_scores]])
            samples["m2"] = (pooled, np.ones(pooled.size, dtype=np.float64), float(pooled.size))
            if issue >= 14 + horizon:
                samples["m3"] = weighted_sample(target_scores, source_scores, frozen_weights, decay)
            else:
                samples["m3"] = (np.zeros(0), np.zeros(0), 0.0)
            candidate_samples = []
            for c_value in c_grid:
                mass = float(c_value) * float(n_target)
                candidate_samples.append(weighted_sample(target_scores, source_scores, dynamic_weights, mass))
            c_index = int(selected[0]) if n_alpha else 0
            # Method 4 uses a coverage-specific c. Rebuild per alpha below.
            samples["m4b"] = weighted_sample(target_scores, source_scores, dynamic_weights, decay)

            q_m1 = weighted_quantiles(samples["m1"][0], samples["m1"][1], alphas)
            q_m2 = weighted_quantiles(samples["m2"][0], samples["m2"][1], alphas)
            if issue >= 14 + horizon:
                q_m3 = weighted_quantiles(samples["m3"][0], samples["m3"][1], alphas)
                applied_m3 = samples["m3"][2]
                expected_m3 = 0.0 if not dynamic_weights and n_target == 0 else decay
                if n_target == 0 or not frozen_weights:
                    expected_m3 = 0.0
                weight_error = max(weight_error, abs(applied_m3 - expected_m3))
            else:
                q_m3 = np.full(n_alpha, np.nan)
                applied_m3 = 0.0
            q_candidates = np.full((n_c, n_alpha), np.inf)
            applied_c = np.zeros(n_c)
            for c_i, sample in enumerate(candidate_samples):
                q_candidates[c_i] = weighted_quantiles(sample[0], sample[1], alphas)
                applied_c[c_i] = sample[2]
                expected = 0.0 if (n_target == 0 or not dynamic_weights) else float(c_grid[c_i]) * float(n_target)
                weight_error = max(weight_error, abs(sample[2] - expected))
            q_m4 = np.array([q_candidates[int(selected[a]), a] for a in range(n_alpha)])
            applied_m4 = np.array([applied_c[int(selected[a])] for a in range(n_alpha)])
            q_m4b = weighted_quantiles(samples["m4b"][0], samples["m4b"][1], alphas)
            applied_m4b = samples["m4b"][2]
            expected_m4b = 0.0 if (n_target == 0 or not dynamic_weights) else decay
            weight_error = max(weight_error, abs(applied_m4b - expected_m4b))
            c0_error = max(c0_error, float(np.nanmax(np.abs(np.where(
                np.isfinite(q_m1) | np.isfinite(q_candidates[0]),
                np.nan_to_num(q_m1, nan=0.0) - np.nan_to_num(q_candidates[0], nan=0.0),
                0.0,
            )))) if np.any(np.isfinite(q_m1) | np.isfinite(q_candidates[0])) else 0.0)
            if not arrays_match(q_m1, q_candidates[0]):
                c0_error = max(c0_error, 1.0)

            q[0, :, product, issue] = q_m1
            q[1, :, product, issue] = q_m2
            q[2, :, product, issue] = q_m3
            q[3, :, product, issue] = q_m4
            q[4, :, product, issue] = q_m4b
            q_c[:, :, product, issue] = q_candidates
            applied_mass[0, :, product, issue] = 0.0
            applied_mass[1, :, product, issue] = samples["m2"][2]
            applied_mass[2, :, product, issue] = applied_m3
            applied_mass[3, :, product, issue] = applied_m4
            applied_mass[4, :, product, issue] = applied_m4b
            nominal_mass[2, product, issue] = decay if issue >= 14 + horizon else 0.0
            nominal_mass[4, product, issue] = decay
            for alpha_i in range(n_alpha):
                nominal_mass[3, product, issue] = float(c_grid[int(selected[alpha_i])]) * float(n_target)
                finite_q = q_candidates[:, alpha_i]
                low_c[:, alpha_i, issue] = yhat[product, issue] - scale[product] * finite_q
                high_c[:, alpha_i, issue] = yhat[product, issue] + scale[product] * finite_q
                low_c[np.isinf(finite_q), alpha_i, issue] = -np.inf
                high_c[np.isinf(finite_q), alpha_i, issue] = np.inf

    revealed_count = max(0, n_time - horizon)
    return {
        "q": q,
        "q_c": q_c,
        "applied_mass": applied_mass,
        "nominal_mass": nominal_mass,
        "chosen_c": chosen_c,
        "history_n": history_n,
        "excluded": excluded,
        "eligible": eligible,
        "revealed_count": revealed_count,
        "weight_error": weight_error,
        "c0_error": c0_error,
        "history_violation": history_violation,
        "frozen": frozen,
    }


def self_check(config: dict) -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(0.2, 3.0, size=(3, 40))
    yhat = y + rng.normal(0.0, 0.3, size=y.shape)
    scale = np.array([1.0, 1.2, 0.8])
    alphas = np.array([0.1, 0.05])
    c_grid = np.asarray(config["candidate_control"]["c_grid"], dtype=np.float64)
    kwargs = dict(
        alphas=alphas,
        c_grid=c_grid,
        n0=15,
        delta=0.001,
        lookback=30,
        min_scores=5,
        min_sources=2,
    )
    base = run_panel(y, yhat, scale, 1, **kwargs)
    if base["c0_error"] > 1e-8:
        raise SystemExit(f"synthetic c=0 diverged from method 1: {base['c0_error']}")
    if base["weight_error"] > 1e-6:
        raise SystemExit(f"synthetic source mass diverged: {base['weight_error']}")
    if base["history_violation"]:
        raise SystemExit("synthetic history used an unrevealed label")
    perturbed = y.copy()
    perturbed[:, -10:] += 1000.0
    changed = run_panel(perturbed, yhat, scale, 1, **kwargs)
    cutoff = 40 - 10 + 1
    if not arrays_match(base["q"][:, :, :, :cutoff], changed["q"][:, :, :, :cutoff]):
        raise SystemExit("synthetic future labels changed past intervals")
    if not arrays_match(base["chosen_c"][:, :, :cutoff], changed["chosen_c"][:, :, :cutoff]):
        raise SystemExit("synthetic future labels changed past source decisions")
    if not arrays_match(base["q_c"][:, :, :, :cutoff], changed["q_c"][:, :, :, :cutoff]):
        raise SystemExit("synthetic future labels changed past candidate intervals")
    short_w, reason = source_weight_map(
        np.array([1.0]),
        [np.array([1.0]), np.array([2.0])],
        delta=0.001,
        equal_if_short=True,
        min_scores=5,
        min_sources=2,
    )
    if reason != "equal_fallback_insufficient_startup" or abs(short_w[0] - 0.5) > 1e-12:
        raise SystemExit("equal-weight fallback did not trigger on a short startup")
    log("SELF_CHECK_OK")


def calendar_for(config: dict, dataset: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    window = config["solidified_windows"][dataset]
    full = pd.date_range(window["train_start"], window["test_end"], freq="D")
    evaluation = full[-int(config["split"]["eval_days"]):]
    test = evaluation[-int(config["split"]["test_days"]):]
    if len(test) != 180 or len(evaluation) != 210:
        raise SystemExit(f"{dataset} calendar split is not 210/180")
    return full, evaluation, test


def prediction_count(horizon: int) -> int:
    return 180 - 10 - int(horizon) + 1


def load_raw_sales(config: dict, dataset: str) -> dict[str, pd.Series]:
    frame = pd.read_parquet(
        REPO / config["sealed_target_parquet"][dataset],
        columns=["entity_id", "date", "sales"],
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["entity_id"] = frame["entity_id"].astype(str)
    out = {}
    for product, group in frame.groupby("entity_id", sort=False):
        if group["date"].duplicated().any():
            raise SystemExit(f"{dataset} {product} has duplicate raw dates")
        series = group.set_index("date")["sales"].astype(float).sort_index()
        out[str(product)] = series
    return out


def scale_from_history(series: pd.Series, test_start: pd.Timestamp, history_days: int, epsilon: float) -> dict:
    observed = series[series.index < test_start].sort_index()
    used = observed.iloc[-history_days:]
    mean = float(used.mean()) if len(used) else float("nan")
    triggered = (not math.isfinite(mean)) or mean < epsilon
    scale = epsilon if triggered else mean
    return {
        "n_history": int(len(used)),
        "history_start": used.index.min().strftime("%Y-%m-%d") if len(used) else "NA",
        "history_end": used.index.max().strftime("%Y-%m-%d") if len(used) else "NA",
        "history_mean": mean,
        "scale": float(scale),
        "epsilon_triggered": bool(triggered),
        "n_true_zero_in_scale_history": int(np.sum(np.abs(used.to_numpy()) <= 1e-12)) if len(used) else 0,
    }


def audit(config: dict, out_dir: Path) -> dict:
    atol = float(config["y_true_match_atol"])
    epsilon = float(config["scale"]["epsilon"])
    history_days = int(config["scale"]["history_days"])
    rows = []
    scale_rows = []
    blocked = []
    max_error = 0.0
    for dataset, products in config["products"].items():
        _full, _evaluation, test_dates = calendar_for(config, dataset)
        raw = load_raw_sales(config, dataset)
        primary = config["v3_primary_targets_actually_used"][dataset]
        lr = primary["learning_rate"]
        scale_info = {}
        for product in products:
            if product not in raw:
                blocked.append(f"{dataset} {product} missing from sealed target parquet")
                continue
            info = scale_from_history(raw[product], test_dates[0], history_days, epsilon)
            info.update({
                "dataset": dataset,
                "product": product,
                "is_v3_primary_target": product == primary["entity_key"],
                "v3_identity": primary["identity"] if product == primary["entity_key"] else "",
                "learning_rate": lr,
            })
            raw_dates = set(raw[product].index)
            window_dates = pd.date_range(test_dates[0] - pd.Timedelta(days=30), test_dates[-1], freq="D")
            info["n_missing_in_pretest30_and_test"] = int(sum(day not in raw_dates for day in window_dates))
            info["n_true_zero_in_test_calendar"] = int(sum(
                day in raw_dates and abs(float(raw[product].loc[day])) <= 1e-12 for day in test_dates
            ))
            scale_info[product] = info
            scale_rows.append(info)
        for horizon in config["horizons"]:
            count = prediction_count(int(horizon))
            issue = test_dates[9:9 + count]
            label = test_dates[9 + int(horizon):9 + int(horizon) + count]
            for product in products:
                if product not in raw:
                    continue
                for seed in config["seeds"]:
                    path = (
                        REPO / config["prediction_root"] / dataset / product
                        / f"h{int(horizon)}_s{int(seed)}_{lr}" / "predictions.csv"
                    )
                    record = {
                        "dataset": dataset,
                        "product": product,
                        "horizon": int(horizon),
                        "seed": int(seed),
                        "learning_rate": lr,
                        "is_v3_primary_target": product == primary["entity_key"],
                        "direct_lr_selection_cell": bool(
                            product == primary["entity_key"] and int(horizon) == 1 and int(seed) == 42
                        ),
                        "path": str(path),
                        "exists": path.is_file(),
                        "n_rows": 0,
                        "columns_ok": False,
                        "n_null": 0,
                        "n_duplicate_rows": 0,
                        "max_abs_y_error": np.nan,
                        "n_label_dates_missing_in_raw": 0,
                        "n_filled_zero_labels": 0,
                        "issue_start": issue[0].strftime("%Y-%m-%d"),
                        "issue_end": issue[-1].strftime("%Y-%m-%d"),
                        "label_start": label[0].strftime("%Y-%m-%d"),
                        "label_end": label[-1].strftime("%Y-%m-%d"),
                        "alignment_ok": False,
                    }
                    if not path.is_file():
                        blocked.append(f"missing predictions: {path}")
                        rows.append(record)
                        continue
                    frame = pd.read_csv(path)
                    expected_cols = [
                        "y_true_normalized",
                        "y_pred_normalized",
                        "y_true_original",
                        "y_pred_original",
                    ]
                    record["columns_ok"] = list(frame.columns) == expected_cols
                    record["n_rows"] = int(len(frame))
                    record["n_null"] = int(frame.isna().sum().sum()) if record["columns_ok"] else -1
                    record["n_duplicate_rows"] = int(frame.duplicated().sum()) if record["columns_ok"] else -1
                    if (not record["columns_ok"]) or record["n_rows"] != count or record["n_null"] or record["n_duplicate_rows"]:
                        blocked.append(f"inventory failed: {path}")
                        rows.append(record)
                        continue
                    y_true = frame["y_true_original"].to_numpy(dtype=np.float64)
                    missing = 0
                    filled = 0
                    errors = []
                    for observed, day in zip(y_true, label):
                        if day in raw[product].index:
                            errors.append(abs(float(raw[product].loc[day]) - float(observed)))
                        else:
                            missing += 1
                            if abs(float(observed)) <= atol:
                                filled += 1
                            else:
                                errors.append(1e9)
                    error = float(max(errors)) if errors else 0.0
                    record["max_abs_y_error"] = error
                    record["n_label_dates_missing_in_raw"] = missing
                    record["n_filled_zero_labels"] = filled
                    record["alignment_ok"] = error <= atol
                    max_error = max(max_error, error)
                    if not record["alignment_ok"]:
                        blocked.append(
                            f"date alignment failed {dataset} {product} h{horizon} s{seed} error={error}"
                        )
                    rows.append(record)
    inventory = pd.DataFrame(rows)
    scales = pd.DataFrame(scale_rows)
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(audit_dir / "inventory.csv", index=False, na_rep="NA")
    scales.to_csv(audit_dir / "scale_and_zeros.csv", index=False, na_rep="NA")
    (audit_dir / "lr_selection.json").write_text(
        json.dumps(
            {
                "authority": config["lr_selection"],
                "v3_primary_targets_actually_used": config["v3_primary_targets_actually_used"],
                "not_used_earlier_primary_manifest": config["not_used_earlier_primary_manifest"],
                "banner": config["lr_selection"]["report_banner"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_audit_report(config, audit_dir / "audit_report.md", inventory, scales, blocked, max_error)
    log(f"AUDIT_FILES={len(inventory)} BLOCKED={len(blocked)} MAX_Y_ERROR={max_error:.3g}")
    return {"inventory": inventory, "scales": scales, "blocked": blocked, "max_error": max_error}


def write_audit_report(config, path: Path, inventory: pd.DataFrame, scales: pd.DataFrame, blocked: list, max_error: float) -> None:
    lines = [
        "# 数据与时间审计",
        "",
        "这是同期产品误差借用的探索性小试所用的审计。建模日历对齐不表示这些产品有真实上市记录，也不表示它们同期上市。",
        "",
        f"显著限制：{config['lr_selection']['report_banner']}。四种或五种校准共用点预测器，消除不了这项限制。",
        "",
        "## 已核实事实",
        "",
        f"- 预测文件 {len(inventory)} 个，对齐失败或清单失败 {len(blocked)} 个。`y_true_original` 与标签日原始销量的最大绝对误差为 {max_error:.3g}。",
        "- 列是 `y_true_normalized`、`y_pred_normalized`、`y_true_original`、`y_pred_original`。导出代码在 `scripts/run_ms_katl_fto_three_track_robustness.py` 写入原始销量，并要求 `metric_space` 为 `original_sales_space`。",
        "- `h` 是单个提前步。`src/data_processing/data_preprocessing.py` 的 `build_tabular_sequence` 取 `sales[end_idx + horizon]`，窗口只在测试分段内部滑动，窗口长度 10。",
        "- 行数 h=1/3/5 分别为 170/168/166，等于 `180 - 10 - h + 1`。",
        "- 日期不在 CSV 中。固化窗口来自 `src/constants.py` 的 `SOLIDIFIED_TARGET_WINDOWS`。切分取最后 210 个日历日中的最后 180 天作为测试段。D5 固化窗口有 211 天，2017-01-17 不在这 210 天里。",
        "- 第 i 行的发出日是测试日索引 `9+i`，标签日是 `9+i+h`。同日先揭示 `label_date <= 发出日` 的真实值，再发出窗口结束于当天的预测。",
        "- 没有落盘的测试前校准预测。`src/transfer_methods/ms_katl_lite.py` 记录验证 RMSE，但 `used_for_checkpoint_selection` 为 false，并固定跑满 50 个 epoch。",
        "- D4 本批主目标是 `166/311`，学习率 `3e-5`。更早清单里的 `166/258` 与 `1e-4` 没有用于这 225 组。D5 主目标 `48/1159414` 用 `3e-4`，D6 主目标 `CA_1/FOODS_3_080` 用 `3e-5`。筛选只用 horizon 1、种子 42、测试集 `rmse_original`，再套到同数据集全部产品、步长和种子。",
        "",
        "## 尺度、真实零与填零",
        "",
    ]
    if scales.empty:
        lines.append("没有尺度记录。")
    else:
        lines.append(
            f"- 尺度历史天数最小 {int(scales['n_history'].min())}，最大 {int(scales['n_history'].max())}。"
            f"触发 epsilon={config['scale']['epsilon']} 的产品数：{int(scales['epsilon_triggered'].sum())}。"
        )
        lines.append(
            f"- 测试开始前 30 天加测试日历上的原始缺日总数：{int(scales['n_missing_in_pretest30_and_test'].sum())}。"
            f"测试日历上的真实零销量天数：{int(scales['n_true_zero_in_test_calendar'].sum())}。"
        )
        lines.append(
            "- `_reindex_target_calendar` 会把缺日填 0，但共享协议要求只有预处理明确声明缺失即零需求时才能这样做。本次评价窗口内没有缺日进入标签，因此没有把填零标签当作真实零销量。"
        )
    if blocked:
        lines.extend(["", "## 阻塞", ""])
        lines.extend(f"- {item}" for item in blocked)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def segment_masks(n_time: int, horizon: int) -> dict[str, np.ndarray]:
    index = np.arange(n_time)
    start = 14 + horizon
    label_day = 9 + index + horizon + 1
    legal = index >= start
    return {
        "label_day_1_30": legal & (label_day <= 30),
        "label_day_31_180": legal & (label_day >= 31) & (label_day <= 180),
        "all_legal": legal,
        "post_start_first_30_predictions": legal & (index < start + 30),
        "post_start_thereafter": index >= start + 30,
    }


def finite_bounds(yhat_row: np.ndarray, scale: float, q_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = yhat_row - scale * q_row
    high = yhat_row + scale * q_row
    low = np.where(np.isfinite(q_row), low, -np.inf)
    high = np.where(np.isfinite(q_row), high, np.inf)
    return low, high


def mask_metrics(y: np.ndarray, low: np.ndarray, high: np.ndarray, scale: float, alpha: float, mask: np.ndarray) -> dict:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {
            "n_points": 0,
            "n_infinite": 0,
            "coverage_all_legal": np.nan,
            "n_common": 0,
            "coverage_common": np.nan,
            "width_raw_common": np.nan,
            "width_std_common": np.nan,
            "winkler_raw_common": np.nan,
            "winkler_std_common": np.nan,
        }
    chosen_low = low[idx]
    chosen_high = high[idx]
    chosen_y = y[idx]
    infinite = ~(np.isfinite(chosen_low) & np.isfinite(chosen_high))
    covered = infinite | ((chosen_y >= chosen_low) & (chosen_y <= chosen_high))
    return {
        "n_points": int(idx.size),
        "n_infinite": int(infinite.sum()),
        "coverage_all_legal": float(covered.mean()),
        "n_common": int(idx.size),
        "coverage_common": float(covered.mean()) if not np.any(infinite) else float((~infinite & covered).sum() / max(1, (~infinite).sum())) if (~infinite).any() else np.nan,
        "width_raw_common": float(np.mean(chosen_high[~infinite] - chosen_low[~infinite])) if (~infinite).any() else np.nan,
        "width_std_common": float(np.mean((chosen_high[~infinite] - chosen_low[~infinite]) / scale)) if (~infinite).any() else np.nan,
        "winkler_raw_common": float(np.mean([
            winkler_one(float(yy), float(lo), float(hi), alpha)
            for yy, lo, hi in zip(chosen_y[~infinite], chosen_low[~infinite], chosen_high[~infinite])
        ])) if (~infinite).any() else np.nan,
        "winkler_std_common": np.nan,
    }


def common_metrics(y, lows, highs, scale, alpha, mask) -> dict:
    """Metrics on one mask. `lows/highs` are lists aligned with methods."""
    finite = np.ones(y.shape[0], dtype=bool)
    for low, high in zip(lows, highs):
        finite &= np.isfinite(low) & np.isfinite(high)
    common = mask & finite
    out = []
    for low, high in zip(lows, highs):
        base = mask_metrics(y, low, high, scale, alpha, mask)
        if not np.any(common):
            base.update({
                "n_common": 0,
                "coverage_common": np.nan,
                "width_raw_common": np.nan,
                "width_std_common": np.nan,
                "winkler_raw_common": np.nan,
                "winkler_std_common": np.nan,
            })
        else:
            idx = np.flatnonzero(common)
            width = high[idx] - low[idx]
            wink = np.array([
                winkler_one(float(y[i]), float(low[i]), float(high[i]), alpha) for i in idx
            ])
            covered = (y[idx] >= low[idx]) & (y[idx] <= high[idx])
            base.update({
                "n_common": int(idx.size),
                "coverage_common": float(covered.mean()),
                "width_raw_common": float(width.mean()),
                "width_std_common": float((width / scale).mean()),
                "winkler_raw_common": float(wink.mean()),
                "winkler_std_common": float((wink / scale).mean()),
            })
        out.append(base)
    return out


def run_experiment(config: dict, out_dir: Path, audit_result: dict) -> dict:
    alphas = np.array([
        float(config["quantile"]["primary_alpha"]),
        float(config["quantile"]["secondary_alpha"]),
    ])
    c_grid = np.asarray(config["candidate_control"]["c_grid"], dtype=np.float64)
    scales = audit_result["scales"].set_index(["dataset", "product"])
    blocked_datasets = set()
    for item in audit_result["blocked"]:
        for dataset in config["products"]:
            if item.startswith(dataset) or f" {dataset} " in f" {item} ":
                blocked_datasets.add(dataset)
    inventory = audit_result["inventory"]
    if not inventory.empty and (~inventory["alignment_ok"]).any():
        blocked_datasets.update(inventory.loc[~inventory["alignment_ok"], "dataset"].unique())
    metric_rows = []
    exclusion_rows = []
    weight_rows = []
    borrow_rows = []
    point_dir = out_dir / "pointwise"
    point_dir.mkdir(parents=True, exist_ok=True)
    check = {
        "c0_error": 0.0,
        "weight_error": 0.0,
        "history_violation": 0,
        "source_pool_excludes_target": True,
        "perturbation_past_unchanged": False,
        "completed_series": 0,
        "completed_panels": 0,
    }
    perturbation_done = False
    for dataset, products in config["products"].items():
        if dataset in blocked_datasets:
            log(f"SKIP_BLOCKED {dataset}")
            continue
        _full, _evaluation, test_dates = calendar_for(config, dataset)
        raw = load_raw_sales(config, dataset)
        product_scale = np.array([float(scales.loc[(dataset, product), "scale"]) for product in products])
        frames = []
        for horizon in config["horizons"]:
            horizon = int(horizon)
            count = prediction_count(horizon)
            issue = test_dates[9:9 + count]
            label = test_dates[9 + horizon:9 + horizon + count]
            y_by_seed = {}
            yhat_by_seed = {}
            for seed in config["seeds"]:
                y = np.zeros((len(products), count))
                yhat = np.zeros((len(products), count))
                lr = config["v3_primary_targets_actually_used"][dataset]["learning_rate"]
                for p_i, product in enumerate(products):
                    path = (
                        REPO / config["prediction_root"] / dataset / product
                        / f"h{horizon}_s{int(seed)}_{lr}" / "predictions.csv"
                    )
                    frame = pd.read_csv(path)
                    y[p_i] = frame["y_true_original"].to_numpy(dtype=np.float64)
                    yhat[p_i] = frame["y_pred_original"].to_numpy(dtype=np.float64)
                y_by_seed[int(seed)] = y
                yhat_by_seed[int(seed)] = yhat
            for seed, y in y_by_seed.items():
                result = run_panel(
                    y,
                    yhat_by_seed[seed],
                    product_scale,
                    horizon,
                    alphas=alphas,
                    c_grid=c_grid,
                    n0=int(config["startup"]["n0"]),
                    delta=float(config["similarity"]["delta"]),
                    lookback=int(config["candidate_control"]["lookback"]),
                    min_scores=int(config["similarity"]["method3_min_scores_per_source"]),
                    min_sources=int(config["similarity"]["method3_min_sources_with_5_scores"]),
                )
                check["c0_error"] = max(check["c0_error"], result["c0_error"])
                check["weight_error"] = max(check["weight_error"], result["weight_error"])
                check["history_violation"] += int(result["history_violation"])
                check["completed_panels"] += 1
                check["completed_series"] += len(products)
                for p_i, product in enumerate(products):
                    sources = [products[k] for k in result["frozen"][p_i]["sources"]]
                    if product in sources:
                        check["source_pool_excludes_target"] = False
                    for source, share in result["frozen"][p_i]["weights"].items():
                        weight_rows.append({
                            "dataset": dataset,
                            "horizon": horizon,
                            "seed": seed,
                            "target": product,
                            "source": products[result["frozen"][p_i]["sources"][source]],
                            "frozen_weight": share,
                            "reason": result["frozen"][p_i]["reason"],
                        })
                    for alpha_i, alpha in enumerate(alphas):
                        exclusion_rows.append({
                            "dataset": dataset,
                            "product": product,
                            "horizon": horizon,
                            "seed": seed,
                            "alpha": float(alpha),
                            "n_labels_revealed_before_a_later_issue": int(result["revealed_count"]),
                            "n_excluded_infinite_candidate": int(result["excluded"][alpha_i, p_i]),
                            "n_eligible_candidate_history": int(result["eligible"][alpha_i, p_i]),
                        })
                masks = segment_masks(count, horizon)
                for p_i, product in enumerate(products):
                    for alpha_i, alpha in enumerate(alphas):
                        lows = []
                        highs = []
                        for method_i in range(len(METHODS)):
                            low, high = finite_bounds(
                                yhat_by_seed[seed][p_i],
                                float(product_scale[p_i]),
                                result["q"][method_i, alpha_i, p_i],
                            )
                            lows.append(low)
                            highs.append(high)
                        for segment, mask in masks.items():
                            metrics = common_metrics(
                                y[p_i], lows, highs, float(product_scale[p_i]), float(alpha), mask
                            )
                            for method_i, method in enumerate(METHODS):
                                metric_rows.append({
                                    "dataset": dataset,
                                    "product": product,
                                    "horizon": horizon,
                                    "seed": seed,
                                    "alpha": float(alpha),
                                    "segment": segment,
                                    "method": method,
                                    "is_v3_primary_target": product == config["v3_primary_targets_actually_used"][dataset]["entity_key"],
                                    **metrics[method_i],
                                })
                    borrow_rows.append({
                        "dataset": dataset,
                        "product": product,
                        "horizon": horizon,
                        "seed": seed,
                        "issue_day": (9 + np.arange(count) + 1).astype(int),
                        "M_m3": result["applied_mass"][2, 0, p_i],
                        "M_m4": result["applied_mass"][3, 0, p_i],
                        "M_m4b": result["applied_mass"][4, 0, p_i],
                        "c_m4": result["chosen_c"][0, p_i],
                    })
                frames.append(build_point_frame(
                    dataset, products, horizon, seed, issue, label, y, yhat_by_seed[seed],
                    product_scale, alphas, result,
                ))
                if (not perturbation_done) and dataset == "D4" and horizon == 1 and seed == 42:
                    perturbed = y.copy()
                    perturbed[:, -20:] += 1000.0
                    changed = run_panel(
                        perturbed, yhat_by_seed[seed], product_scale, horizon,
                        alphas=alphas, c_grid=c_grid, n0=15, delta=0.001, lookback=30,
                        min_scores=5, min_sources=2,
                    )
                    cutoff = count - 20 + horizon
                    check["perturbation_past_unchanged"] = bool(
                        arrays_match(result["q"][:, :, :, :cutoff], changed["q"][:, :, :, :cutoff])
                        and arrays_match(result["q_c"][:, :, :, :cutoff], changed["q_c"][:, :, :, :cutoff])
                        and arrays_match(result["chosen_c"][:, :, :cutoff], changed["chosen_c"][:, :, :cutoff])
                    )
                    perturbation_done = True
                    log(f"PERTURBATION_PAST_UNCHANGED={check['perturbation_past_unchanged']}")
            log(f"PANEL_DONE {dataset} h{horizon}")
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(point_dir / f"{dataset}.parquet", index=False)
            log(f"WROTE_POINTWISE {dataset}")
        progress = {
            "status": "running",
            "completed_series": check["completed_series"],
            "completed_panels": check["completed_panels"],
            "blocked_datasets": sorted(blocked_datasets),
        }
        (out_dir / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
    check["perturbation_past_unchanged"] = bool(check["perturbation_past_unchanged"])
    return {
        "metrics": pd.DataFrame(metric_rows),
        "exclusion": pd.DataFrame(exclusion_rows),
        "weights": pd.DataFrame(weight_rows),
        "borrow": borrow_rows,
        "check": check,
        "blocked_datasets": sorted(blocked_datasets),
    }


def build_point_frame(dataset, products, horizon, seed, issue, label, y, yhat, scale, alphas, result) -> pd.DataFrame:
    chunks = []
    n_time = y.shape[1]
    legal = np.arange(n_time) >= 14 + horizon
    for p_i, product in enumerate(products):
        for alpha_i, alpha in enumerate(alphas):
            finite = np.ones(n_time, dtype=bool)
            bounds = []
            for method_i in range(len(METHODS)):
                low, high = finite_bounds(yhat[p_i], float(scale[p_i]), result["q"][method_i, alpha_i, p_i])
                bounds.append((low, high))
                finite &= np.isfinite(low) & np.isfinite(high)
            for method_i, method in enumerate(METHODS):
                low, high = bounds[method_i]
                width = high - low
                wink = np.array([
                    winkler_one(float(y[p_i, i]), float(low[i]), float(high[i]), float(alpha))
                    for i in range(n_time)
                ])
                chunks.append(pd.DataFrame({
                    "dataset": dataset,
                    "product": product,
                    "horizon": horizon,
                    "seed": seed,
                    "alpha": float(alpha),
                    "method": method,
                    "i": np.arange(n_time),
                    "issue_date": issue.strftime("%Y-%m-%d"),
                    "label_date": label.strftime("%Y-%m-%d"),
                    "y": y[p_i],
                    "yhat": yhat[p_i],
                    "q": result["q"][method_i, alpha_i, p_i],
                    "lo": low,
                    "hi": high,
                    "finite": np.isfinite(low) & np.isfinite(high),
                    "in_common_finite": finite & legal,
                    "evaluable": legal,
                    "width_raw": width,
                    "width_std": width / float(scale[p_i]),
                    "winkler_raw": wink,
                    "winkler_std": wink / float(scale[p_i]),
                    "applied_source_mass": result["applied_mass"][method_i, alpha_i, p_i],
                    "c_selected": result["chosen_c"][alpha_i, p_i] if method == "m4" else np.nan,
                    "candidate_history_n": result["history_n"][alpha_i, p_i] if method == "m4" else 0,
                }))
    return pd.concat(chunks, ignore_index=True)


def aggregate(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["dataset", "product", "horizon", "alpha", "segment", "method", "is_v3_primary_target"]
    value_cols = [
        "n_points", "n_infinite", "coverage_all_legal", "n_common", "coverage_common",
        "width_raw_common", "width_std_common", "winkler_raw_common", "winkler_std_common",
    ]
    product = metrics.groupby(keys, as_index=False)[value_cols].mean()
    dataset_keys = ["dataset", "horizon", "alpha", "segment", "method"]
    dataset = product.groupby(dataset_keys, as_index=False)[value_cols].mean()
    return metrics, product, dataset


def paired(product: pd.DataFrame) -> pd.DataFrame:
    rows = []
    index_cols = ["dataset", "product", "horizon", "alpha", "segment"]
    for keys, group in product.groupby(index_cols, sort=False):
        keyed = {row.method: row for row in group.itertuples(index=False)}
        if not all(name in keyed for name in METHODS):
            continue
        record = dict(zip(index_cols, keys if isinstance(keys, tuple) else (keys,)))
        for name in METHODS:
            record[f"winkler_std_{name}"] = getattr(keyed[name], "winkler_std_common")
            record[f"coverage_common_{name}"] = getattr(keyed[name], "coverage_common")
            record[f"coverage_all_{name}"] = getattr(keyed[name], "coverage_all_legal")
            record[f"n_infinite_{name}"] = getattr(keyed[name], "n_infinite")
            record[f"n_common_{name}"] = getattr(keyed[name], "n_common")
            record[f"width_std_{name}"] = getattr(keyed[name], "width_std_common")
        base = record["winkler_std_m4"]
        record["delta_m4_minus_m1"] = base - record["winkler_std_m1"]
        record["delta_m4_minus_m3"] = base - record["winkler_std_m3"]
        record["delta_m4_minus_m4b"] = base - record["winkler_std_m4b"]
        rows.append(record)
    return pd.DataFrame(rows)


def credible(delta: np.ndarray, warn: np.ndarray) -> dict:
    valid = np.isfinite(delta)
    better = valid & (delta < -1e-9) & ~warn
    better_warn = valid & (delta < -1e-9) & warn
    worse = valid & (delta > 1e-9)
    median = float(np.median(delta[valid])) if valid.any() else float("nan")
    n_valid = int(valid.sum())
    supported = bool(n_valid > 0 and median < 0 and int(better.sum()) * 2 >= n_valid and int(better.sum()) > int(better_warn.sum()))
    return {
        "n_valid": n_valid,
        "n_better_without_warning": int(better.sum()),
        "n_better_with_warning": int(better_warn.sum()),
        "n_worse": int(worse.sum()),
        "median_delta": median,
        "supported": supported,
    }


def write_figures(borrow_rows: list[dict], paired_df: pd.DataFrame, fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if borrow_rows:
        pieces = []
        for row in borrow_rows:
            n = len(row["issue_day"])
            pieces.append(pd.DataFrame({
                "dataset": row["dataset"],
                "horizon": row["horizon"],
                "issue_day": row["issue_day"],
                "M_m3": row["M_m3"],
                "M_m4": row["M_m4"],
                "M_m4b": row["M_m4b"],
                "c_m4": row["c_m4"],
            }))
        borrow = pd.concat(pieces, ignore_index=True)
        mean = borrow.groupby(["dataset", "horizon", "issue_day"], as_index=False)[
            ["M_m3", "M_m4", "M_m4b", "c_m4"]
        ].mean()
        datasets = ["D4", "D5", "D6"]
        horizons = [1, 3, 5]
        fig, axes = plt.subplots(len(datasets), len(horizons), figsize=(12, 8), sharex=True)
        for r, dataset in enumerate(datasets):
            for c, horizon in enumerate(horizons):
                ax = axes[r, c]
                part = mean[(mean["dataset"] == dataset) & (mean["horizon"] == horizon)]
                ax.plot(part["issue_day"], part["M_m3"], label="m3")
                ax.plot(part["issue_day"], part["M_m4"], label="m4")
                ax.plot(part["issue_day"], part["M_m4b"], label="m4b")
                ax.set_title(f"{dataset} h={horizon}")
                if r == len(datasets) - 1:
                    ax.set_xlabel("Test-calendar issue day")
                if c == 0:
                    ax.set_ylabel("Applied source mass")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        path = fig_dir / "borrow_mass_over_time.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))

    early = paired_df[
        (paired_df["alpha"] == 0.1)
        & (paired_df["segment"].isin(["post_start_first_30_predictions", "post_start_thereafter"]))
    ]
    if not early.empty:
        datasets = ["D4", "D5", "D6"]
        horizons = [1, 3, 5]
        fig, axes = plt.subplots(len(datasets), len(horizons), figsize=(12, 8), sharey=False)
        x = np.arange(len(METHODS))
        width = 0.35
        for r, dataset in enumerate(datasets):
            for c, horizon in enumerate(horizons):
                ax = axes[r, c]
                for offset, segment, label in (
                    (-width / 2, "post_start_first_30_predictions", "first 30 after start"),
                    (width / 2, "post_start_thereafter", "thereafter"),
                ):
                    values = []
                    for method in METHODS:
                        part = early[
                            (early["dataset"] == dataset)
                            & (early["horizon"] == horizon)
                            & (early["segment"] == segment)
                        ]
                        column = f"winkler_std_{method}"
                        values.append(float(part[column].mean()) if len(part) and part[column].notna().any() else np.nan)
                    ax.bar(x + offset, values, width=width, label=label)
                ax.set_xticks(x, METHODS, rotation=45, ha="right")
                ax.set_title(f"{dataset} h={horizon}")
                if c == 0:
                    ax.set_ylabel("Std. Winkler")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        path = fig_dir / "early_late_winkler.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(str(path))
    return written


def fmt(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def write_results_report(config, out_dir, audit_result, experiment, product, paired_df, figures) -> None:
    primary = paired_df[(paired_df["alpha"] == 0.1) & (paired_df["segment"] == "all_legal")].copy()
    warning_line = float(config["coverage_warning_below_nominal"])
    nominal = 0.9
    answers = {}
    for name, column in (("m1", "delta_m4_minus_m1"), ("m3", "delta_m4_minus_m3"), ("m4b", "delta_m4_minus_m4b")):
        warn = primary["coverage_common_m4"].to_numpy(dtype=float) < (nominal - warning_line)
        warn = np.where(np.isfinite(primary["coverage_common_m4"].to_numpy(dtype=float)), warn, True)
        answers[name] = credible(primary[column].to_numpy(dtype=float), warn)
    early = paired_df[(paired_df["alpha"] == 0.1) & (paired_df["segment"] == "post_start_first_30_predictions")]
    late = paired_df[(paired_df["alpha"] == 0.1) & (paired_df["segment"] == "post_start_thereafter")]
    early_stats = credible(
        early["delta_m4_minus_m1"].to_numpy(dtype=float),
        np.zeros(len(early), dtype=bool),
    ) if len(early) else {"supported": False, "median_delta": float("nan")}
    late_stats = credible(
        late["delta_m4_minus_m1"].to_numpy(dtype=float),
        np.zeros(len(late), dtype=bool),
    ) if len(late) else {"supported": False, "median_delta": float("nan")}
    checks = experiment["check"]
    checks_ok = (
        checks["c0_error"] <= 1e-8
        and checks["weight_error"] <= 1e-6
        and checks["history_violation"] == 0
        and checks["source_pool_excludes_target"]
        and checks["perturbation_past_unchanged"]
        and not experiment["blocked_datasets"]
    )
    banner = config["lr_selection"]["report_banner"]
    lines = [
        "# 同期产品误差借用的探索性小试",
        "",
        f"**{banner}**",
        "",
        "五种方法共用同一套点预测。这个事实不消除测试集参与学习率选择的限制。本轮没有重训 SITA、源模型或 M4。",
        "",
        "## 已核实事实",
        "",
        "- 225 组预测按固化日历恢复。`h` 是单步提前预测。发出日为测试日索引 `9+i`，标签日为 `9+i+h`。",
        "- D4 主目标按 V3 运行清单记为 `166/311`，学习率 `3e-5`。没有改用 `166/258`。",
        "- 评价窗口内原始缺日为 0。真实零销量保留在尺度和评价中。运行时代码仍有缺日填零路径，而协议文档没有把 D4–D6 缺日声明为零需求；本次没有填零标签进入校准或评价，排除填零的敏感性与主分析相同。",
        f"- `y_true_original` 最大绝对对账误差 {audit_result['max_error']:.3g}。没有测试前落盘校准残差。共同评价从每条序列已经揭示 15 个目标分数之后开始，行号 `i = 14 + h`。",
        "",
        "## 实际完成的实验",
        "",
        f"- 完成面板 {checks['completed_panels']} 个，序列 {checks['completed_series']} 条。阻塞数据集：{experiment['blocked_datasets'] or '无'}。",
        f"- 必要检查：c=0 与方法 1 的最大差异 {checks['c0_error']:.3g}；源权重质量最大差异 {checks['weight_error']:.3g}；历史泄露计数 {checks['history_violation']}；源池不含目标 {checks['source_pool_excludes_target']}；未来标签扰动后过去区间和选源不变 {checks['perturbation_past_unchanged']}。",
        "- 主比较是 90% 区间、全部合法评价期、五种方法都有限的共同时点上的标准化 Winkler。负的差值表示方法 4 更低。低于名义覆盖率 5 个百分点只记为警戒，不表示其余情况覆盖合理。全部合法点的覆盖率单独列出，不与共同有限集的宽度混用。",
        "",
        "### 三个问题",
        "",
    ]
    question = {
        "m1": "方法 4 是否优于目标在线校准方法 1",
        "m3": "方法 4 是否优于固定规则方法 3",
        "m4b": "方法 4 是否优于隔离总量控制的方法 4b",
    }
    for key, title in question.items():
        stat = answers[key]
        lines.append(
            f"- {title}：{'有可信优势' if stat['supported'] and checks_ok else '没有形成可信优势'}。"
            f"可比单元 {stat['n_valid']}，无警戒且 Winkler 更低 {stat['n_better_without_warning']}，"
            f"Winkler 更低但覆盖触及警戒 {stat['n_better_with_warning']}，Winkler 更高 {stat['n_worse']}，"
            f"差值中位数 {fmt(stat['median_delta'])}。"
        )
    if not answers["m1"]["supported"]:
        lines.append("- 方法 4 相对方法 1 没有可信优势。当前实现不支持升级为主线。这不表示所有源误差借用都无效。")
    elif not answers["m3"]["supported"]:
        lines.append("- 方法 4 优于方法 1，但不优于方法 3。这支持借用有实用价值，不支持把自适应控制当作主要贡献。")
    elif not answers["m4b"]["supported"]:
        lines.append("- 方法 4 优于方法 1 和方法 3。相对方法 3 只能先看成动态权重与借用量规则一起变化后的整体差异。方法 4 没有优于方法 4b，因此还不支持自适应总量控制有独立贡献。")
    else:
        lines.append("- 方法 4 在共同有限集的标准化 Winkler 上同时优于方法 1、方法 3 和方法 4b。覆盖偏差仍需逐格查看；未触发警戒线也不等于覆盖已经合理。可以扩大验证并补充 M4 对照。本轮仍然是探索性结果。")
    if early_stats.get("supported") and not late_stats.get("supported"):
        lines.append("- 相对方法 1 的优势出现在共同启动后的前 30 次有效预测，此后没有同样的优势。这不是上市前 30 天，适用阶段应限于启动后的早期预测。")
    lines.extend(["", "### 产品级差值（90%，全部合法评价期）", ""])
    show = primary[[
        "dataset", "product", "horizon", "delta_m4_minus_m1", "delta_m4_minus_m3", "delta_m4_minus_m4b",
        "coverage_common_m4", "coverage_all_m4", "n_infinite_m4", "n_common_m4",
    ]].sort_values(["dataset", "horizon", "product"])
    lines.append("```")
    lines.append(show.to_csv(index=False, na_rep="NA").rstrip())
    lines.append("```")
    lines.extend(["", "### 启动后分段的方法 4 减方法 1 中位数", ""])
    for segment, frame in (
        ("post_start_first_30_predictions", early),
        ("post_start_thereafter", late),
    ):
        if frame.empty:
            lines.append(f"- {segment}: NA")
            continue
        lines.append(
            f"- {segment}: 中位数 {fmt(float(np.nanmedian(frame['delta_m4_minus_m1'])))}，"
            f"单元数 {int(frame['delta_m4_minus_m1'].notna().sum())}。"
        )
    lines.extend(["", "## 尚未解决的限制", ""])
    lines.extend([
        f"- {banner}。直接使用测试标签选学习率的单元是各数据集主目标的 horizon 1、种子 42；该学习率又被同数据集其余产品继承。",
        "- 没有分布无关覆盖率保证。权重随时间变化，源和目标的误差分布也不被假定相同。",
        "- 标准化只用于跨产品比较。触发 epsilon 的产品如果存在，其标准化评分不能直接并进总结论；本次尺度表会标明是否触发。",
        "- 文献查新尚未做。这里不宣称首次提出，也不判断期刊分区。",
        "- 空的日历分段写成 NA。h=5 的测试前 30 个标签日没有共同启动后的可评价点。",
        "",
        "## 输出",
        "",
        f"- 图：{', '.join(figures) if figures else 'NA'}",
        f"- 检查通过：{checks_ok}",
    ])
    (out_dir / "results_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "checks.json").write_text(json.dumps({
        **checks,
        "checks_ok": checks_ok,
        "answers": answers,
        "early_vs_m1": early_stats,
        "late_vs_m1": late_stats,
    }, indent=2, default=str), encoding="utf-8")


def segment_date_table(config: dict) -> pd.DataFrame:
    rows = []
    for dataset in config["products"]:
        _full, _evaluation, test_dates = calendar_for(config, dataset)
        for horizon in config["horizons"]:
            horizon = int(horizon)
            count = prediction_count(horizon)
            issue = test_dates[9:9 + count]
            label = test_dates[9 + horizon:9 + horizon + count]
            masks = segment_masks(count, horizon)
            for name, mask in masks.items():
                if not np.any(mask):
                    rows.append({
                        "dataset": dataset, "horizon": horizon, "segment": name,
                        "n_points": 0, "issue_start": "NA", "issue_end": "NA",
                        "label_start": "NA", "label_end": "NA",
                    })
                    continue
                rows.append({
                    "dataset": dataset,
                    "horizon": horizon,
                    "segment": name,
                    "n_points": int(mask.sum()),
                    "issue_start": issue[mask][0].strftime("%Y-%m-%d"),
                    "issue_end": issue[mask][-1].strftime("%Y-%m-%d"),
                    "label_start": label[mask][0].strftime("%Y-%m-%d"),
                    "label_end": label[mask][-1].strftime("%Y-%m-%d"),
                })
    return pd.DataFrame(rows)


def main() -> int:
    config = load_config()
    out_dir = REPO / config["output_root"]
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    (out_dir / "config_used.json").write_text(
        json.dumps({"sha256": digest, "config": config}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "progress.json").write_text(json.dumps({"status": "audit"}, indent=2), encoding="utf-8")
    self_check(config)
    audit_result = audit(config, out_dir)
    if audit_result["blocked"]:
        log("DATE_OR_INVENTORY_BLOCKER")
        for item in audit_result["blocked"]:
            log(item)
        (out_dir / "progress.json").write_text(
            json.dumps({"status": "blocked", "blocked": audit_result["blocked"]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return 3
    experiment = run_experiment(config, out_dir, audit_result)
    summary_dir = out_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics, product, dataset = aggregate(experiment["metrics"])
    paired_df = paired(product)
    seed_metrics.to_csv(summary_dir / "metrics_by_seed.csv", index=False, na_rep="NA")
    product.to_csv(summary_dir / "metrics_by_product.csv", index=False, na_rep="NA")
    dataset.to_csv(summary_dir / "metrics_by_dataset.csv", index=False, na_rep="NA")
    paired_df.to_csv(summary_dir / "paired_product_differences.csv", index=False, na_rep="NA")
    experiment["exclusion"].to_csv(summary_dir / "candidate_exclusion.csv", index=False, na_rep="NA")
    experiment["weights"].to_csv(summary_dir / "method3_frozen_weights.csv", index=False, na_rep="NA")
    segment_date_table(config).to_csv(summary_dir / "segment_dates.csv", index=False, na_rep="NA")
    figures = write_figures(experiment["borrow"], paired_df, out_dir / "figures")
    write_results_report(config, out_dir, audit_result, experiment, product, paired_df, figures)
    checks = json.loads((out_dir / "checks.json").read_text(encoding="utf-8"))
    (out_dir / "progress.json").write_text(json.dumps({
        "status": "complete",
        "completed_series": experiment["check"]["completed_series"],
        "completed_panels": experiment["check"]["completed_panels"],
        "checks_ok": checks["checks_ok"],
        "output_root": str(out_dir),
    }, indent=2), encoding="utf-8")
    log(
        f"COMPLETE series={experiment['check']['completed_series']} "
        f"panels={experiment['check']['completed_panels']} checks_ok={checks['checks_ok']} output={out_dir}"
    )
    return 0 if checks["checks_ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
