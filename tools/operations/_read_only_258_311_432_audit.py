#!/usr/bin/env python3
"""Read-only provenance + data/prediction audit. Does not train."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/Users/ming/Desktop/复现实验/保留的复现实验修改rfe")
OUT = ROOT / "outputs" / "audits" / "20260818_reference_and_archetype_audit.json"

FORMAL_CSV = (
    ROOT
    / "outputs/runs/20260816_183000_d1d6_630ae4ef_h1s42_d5first_p8"
    / "d4_without/results/dataset4_without_results.csv"
)
FORMAL_MANIFEST = (
    ROOT
    / "docs/experiment_evidence/20260816_183000_d1d6_630ae4ef_h1s42_d5first_p8"
    / "run/d4_without/results/dataset4_without_results.manifest.json"
)
STALE_CSV = ROOT / "outputs/runs/20260727_224425_132701_D4_180d_without/results/dataset4_without_results.csv"
STALE_LOCAL_CSV = ROOT / "outputs/runs/local_d4_without_h1_s42_20260804_151150/results/dataset4_without_results.csv"
BASELINE_CSV = ROOT / "outputs/baselines/d4_baselines_multiseed.csv"
CROSS_ROOT = ROOT / "outputs/runs/20260818_141311_897713_MS_KATL_D4_H1_S42_CROSS_TARGET"
MECH_DIAG = ROOT / "outputs/runs/20260818_134401_999267_MS_KATL_MECHANISM_D4_h1_s42/results/mechanism_ablation_diagnostics.json"
KNN_JSON = ROOT / "configs/solidified/knn/Dataset4/knn_without_info_sharing.json"
TARGET_PARQUET = ROOT / "数据集/固化数据/d1_d6_sealed_v1/dataset4/target.parquet"
SOURCE_PARQUET = ROOT / "数据集/固化数据/d1_d6_sealed_v1/dataset4/source.parquet"

TARGETS = ("166/258", "166/311", "166/432")
ARCHETYPES = {
    "166/258": "success",
    "166/311": "fused_useful_overall_fail",
    "166/432": "fused_negative_transfer",
}
FORMAL_METHODS = ("No-TL", "MSML-TL")
BL_METHODS = (
    "BL1_HistoricalMean",
    "BL2_MovingAverage",
    "BL3_LightGBM",
    "BL4_LSTM",
    "BL5_SeasonalNaive",
)
TRAIN_DAYS = 15
VAL_DAYS = 15
TEST_DAYS = 180
WINDOW = 10
HORIZON = 1
SEED = 42
WINDOW_START = np.datetime64("2024-12-16")
WINDOW_END = np.datetime64("2025-07-13")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def identity_match(row: dict[str, str], *, target: str, method: str, horizon: int, seed: int) -> bool:
    key = str(row.get("target_entity_key") or row.get("target_identity") or "").replace("_", "/")
    if key != target:
        return False
    if str(row.get("method") or "") != method:
        return False
    if int(float(row.get("horizon") or -1)) != horizon:
        return False
    if int(float(row.get("seed") or -1)) != seed:
        return False
    scenario = str(row.get("scenario") or row.get("information_sharing") or "").lower()
    if scenario not in {"without", "without_information_sharing"}:
        return False
    return True


def row_payload(row: dict[str, str], *, csv_row_number: int, run_root: str, source_csv: str, git_commit: str, extra: dict) -> dict:
    return {
        "run_root": run_root,
        "source_csv": source_csv,
        "dataset": row.get("dataset") or row.get("dataset_id") or "D4",
        "target_identity": str(row.get("target_entity_key") or "").replace("_", "/"),
        "scenario": row.get("scenario") or row.get("information_sharing"),
        "horizon": int(float(row.get("horizon"))),
        "seed": int(float(row.get("seed"))),
        "method": row.get("method"),
        "git_commit": git_commit,
        "result_row_identifier": (
            f"csv_row={csv_row_number}|"
            f"{row.get('dataset') or row.get('dataset_id')}|{str(row.get('target_entity_key') or '').replace('_', '/')}"
            f"|{row.get('scenario') or row.get('information_sharing')}|H={row.get('horizon')}|seed={row.get('seed')}|{row.get('method')}"
        ),
        "RMSE": float(row["rmse"]),
        "sMAPE": float(row["smape"]),
        **extra,
    }


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float).reshape(-1)
    mean = float(values.mean()) if values.size else float("nan")
    std = float(values.std(ddof=0)) if values.size else float("nan")
    cv = float(std / mean) if mean not in (0.0, float("nan")) and math.isfinite(mean) and mean != 0.0 else float("nan")
    zero_ratio = float(np.mean(values == 0.0)) if values.size else float("nan")
    if values.size >= 2:
        x = np.arange(values.size, dtype=float)
        slope = float(np.polyfit(x, values, 1)[0])
        ac1 = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    else:
        slope = float("nan")
        ac1 = float("nan")
    ac7 = float(np.corrcoef(values[:-7], values[7:])[0, 1]) if values.size > 7 else float("nan")
    return {
        "n": int(values.size),
        "mean": mean,
        "std": std,
        "cv": cv,
        "min": float(values.min()) if values.size else float("nan"),
        "max": float(values.max()) if values.size else float("nan"),
        "median": float(np.median(values)) if values.size else float("nan"),
        "zero_ratio": zero_ratio,
        "trend_slope_per_day": slope,
        "acf_lag1": ac1,
        "acf_lag7": ac7,
    }


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float).reshape(-1))
    b = np.sort(np.asarray(b, dtype=float).reshape(-1))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def downsample(values: list[float], step: int) -> list[float]:
    return [float(values[i]) for i in range(0, len(values), step)]


def weekly_mean(values: np.ndarray) -> tuple[list[str], list[float]]:
    values = np.asarray(values, dtype=float).reshape(-1)
    labels = []
    means = []
    for start in range(0, values.size, 7):
        chunk = values[start : start + 7]
        labels.append(f"W{start // 7 + 1}")
        means.append(float(chunk.mean()))
    return labels, means


def load_entity(table, store: str, product: str) -> np.ndarray:
    cols = table.column_names
    if "store_id" in cols and "product_id" in cols:
        store_col = table.column("store_id").to_pandas().astype(str)
        product_col = table.column("product_id").to_pandas().astype(str)
        date_col = table.column("date").to_pandas()
        sales_col = table.column("sales").to_pandas()
        mask = (store_col == store) & (product_col == product)
        frame_dates = date_col[mask]
        frame_sales = sales_col[mask]
    elif "entity_id" in cols:
        entity = f"{store}_{product}"
        entity_col = table.column("entity_id").to_pandas().astype(str)
        date_col = table.column("date").to_pandas()
        sales_col = table.column("sales").to_pandas()
        mask = entity_col == entity
        frame_dates = date_col[mask]
        frame_sales = sales_col[mask]
    else:
        raise RuntimeError(f"unrecognized parquet columns: {cols}")
    order = np.argsort(frame_dates.to_numpy())
    sales = np.asarray(frame_sales.to_numpy(), dtype=float)[order]
    raw_dates = np.asarray(frame_dates.to_numpy())[order]
    dates = np.array([np.datetime64(str(value)[:10]) for value in raw_dates])
    mask = (dates >= WINDOW_START) & (dates <= WINDOW_END)
    return dates[mask], sales[mask]


def main() -> None:
    formal_git = json.loads(FORMAL_MANIFEST.read_text())["code_identity"]["git_commit"]
    formal_sha = sha256_file(FORMAL_CSV)
    baseline_sha = sha256_file(BASELINE_CSV)
    stale_sha = sha256_file(STALE_CSV) if STALE_CSV.exists() else None
    stale_local_sha = sha256_file(STALE_LOCAL_CSV) if STALE_LOCAL_CSV.exists() else None

    formal_rows = load_csv(FORMAL_CSV)
    baseline_rows = load_csv(BASELINE_CSV)
    stale_rows = load_csv(STALE_CSV) if STALE_CSV.exists() else []
    stale_local_rows = load_csv(STALE_LOCAL_CSV) if STALE_LOCAL_CSV.exists() else []
    knn = json.loads(KNN_JSON.read_text())
    cross_diag = json.loads((CROSS_ROOT / "results/d4_h1_s42_cross_target_diagnostics.json").read_text())
    mech_diag = json.loads(MECH_DIAG.read_text()) if MECH_DIAG.exists() else {}

    accepted = []
    join_audit = []
    all_targets = ("166/258", "166/311", "166/313", "166/432", "166/433")
    for target in all_targets:
        for method in FORMAL_METHODS:
            hits = []
            for idx, row in enumerate(formal_rows, start=2):
                if identity_match(row, target=target, method=method, horizon=HORIZON, seed=SEED):
                    hits.append((idx, row))
            status = "FOUND" if len(hits) == 1 else ("AMBIGUOUS" if len(hits) > 1 else "MISSING")
            payload = None
            if hits:
                idx, row = hits[0]
                payload = row_payload(
                    row,
                    csv_row_number=idx,
                    run_root=str(FORMAL_CSV.parents[2]),
                    source_csv=str(FORMAL_CSV),
                    git_commit=formal_git,
                    extra={
                        "result_status": row.get("result_status"),
                        "protocol_track": row.get("protocol_track"),
                        "rmse_metric_space": row.get("rmse_metric_space"),
                        "smape_metric_space": row.get("smape_metric_space"),
                        "rmse_current": row.get("rmse_current"),
                        "original_scale_rmse": row.get("original_scale_rmse"),
                        "normalized_rmse": row.get("normalized_rmse"),
                        "sample_count": row.get("sample_count"),
                        "sample_manifest_digest": row.get("sample_manifest_digest"),
                        "metric_date_start": row.get("metric_date_start"),
                        "metric_date_end": row.get("metric_date_end"),
                        "csv_sha256": formal_sha,
                        "accepted": True,
                    },
                )
                accepted.append(payload)
            join_audit.append(
                {
                    "family": "formal",
                    "target_identity": target,
                    "method": method,
                    "join_keys": "dataset=D4 + target_entity_key + scenario in {without,without_information_sharing} + horizon=1 + seed=42 + method",
                    "status": status,
                    "n_hits": len(hits),
                    "row": payload,
                }
            )
        for method in BL_METHODS:
            hits = []
            for idx, row in enumerate(baseline_rows, start=2):
                if identity_match(row, target=target, method=method, horizon=HORIZON, seed=SEED):
                    hits.append((idx, row))
            status = "FOUND" if len(hits) == 1 else ("AMBIGUOUS" if len(hits) > 1 else "MISSING")
            payload = None
            if hits:
                idx, row = hits[0]
                payload = row_payload(
                    row,
                    csv_row_number=idx,
                    run_root=str(BASELINE_CSV.parent),
                    source_csv=str(BASELINE_CSV),
                    git_commit="NOT_IN_CSV_OR_MANIFEST",
                    extra={
                        "result_status": row.get("result_status"),
                        "protocol_track": row.get("protocol_track"),
                        "rmse_metric_space": row.get("rmse_metric_space"),
                        "smape_metric_space": row.get("smape_metric_space"),
                        "sample_count": row.get("sample_count"),
                        "sample_manifest_digest": row.get("sample_manifest_digest"),
                        "csv_sha256": baseline_sha,
                        "accepted": True,
                        "producing_script": "scripts/baselines/run_baselines_multiseed.py",
                    },
                )
                accepted.append(payload)
            join_audit.append(
                {
                    "family": "baseline",
                    "target_identity": target,
                    "method": method,
                    "join_keys": "dataset_id=D4 + target_entity_key + scenario=without + horizon=1 + seed=42 + method",
                    "status": status,
                    "n_hits": len(hits),
                    "row": payload,
                }
            )

    stale_refs = []
    for label, rows, path, sha in (
        ("20260727_smoke_extended_trial", stale_rows, STALE_CSV, stale_sha),
        ("20260804_local_copy_or_rerun", stale_local_rows, STALE_LOCAL_CSV, stale_local_sha),
    ):
        for method in FORMAL_METHODS:
            hits = []
            for idx, row in enumerate(rows, start=2):
                if identity_match(row, target="166/258", method=method, horizon=HORIZON, seed=SEED):
                    hits.append((idx, row))
            if not hits:
                stale_refs.append({"label": label, "method": method, "status": "MISSING"})
                continue
            idx, row = hits[0]
            stale_refs.append(
                row_payload(
                    row,
                    csv_row_number=idx,
                    run_root=str(path.parents[1]),
                    source_csv=str(path),
                    git_commit="UNKNOWN_NOT_IN_RUN_CONFIG",
                    extra={
                        "label": label,
                        "result_status": row.get("result_status"),
                        "protocol_track": row.get("protocol_track"),
                        "rmse_metric_space": row.get("rmse_metric_space"),
                        "sample_count": row.get("sample_count"),
                        "sample_manifest_digest": row.get("sample_manifest_digest"),
                        "metric_date_start": row.get("metric_date_start"),
                        "metric_date_end": row.get("metric_date_end"),
                        "csv_sha256": sha,
                        "accepted": False,
                        "smoke": True,
                    },
                )
            )

    hardcoded = mech_diag.get("reference", {})
    no_tl_stale = next((r for r in stale_refs if r.get("method") == "No-TL" and r.get("label") == "20260727_smoke_extended_trial"), None)
    no_tl_accepted = next(r for r in accepted if r["method"] == "No-TL" and r["target_identity"] == "166/258")
    msml_stale = next((r for r in stale_refs if r.get("method") == "MSML-TL" and r.get("label") == "20260727_smoke_extended_trial"), None)
    msml_accepted = next(r for r in accepted if r["method"] == "MSML-TL" and r["target_identity"] == "166/258")
    bl4_accepted = next(r for r in accepted if r["method"] == "BL4_LSTM" and r["target_identity"] == "166/258")

    target_table = pq.read_table(TARGET_PARQUET)
    source_table = pq.read_table(SOURCE_PARQUET)
    knn_results = knn.get("results", {})

    archetype = {}
    for target in TARGETS:
        store, product = target.split("/")
        dates_210, sales_210 = load_entity(target_table, store, product)
        if sales_210.size != TRAIN_DAYS + VAL_DAYS + TEST_DAYS:
            raise RuntimeError(
                f"{target} formal window length={sales_210.size}, expected 210"
            )
        train = sales_210[:TRAIN_DAYS]
        val = sales_210[TRAIN_DAYS : TRAIN_DAYS + VAL_DAYS]
        test = sales_210[TRAIN_DAYS + VAL_DAYS :]
        eval_sales = test[WINDOW:]  # 170 labels, matching sample_count
        train_min = float(train.min())
        train_max = float(train.max())
        denom = train_max - train_min if train_max != train_min else 1.0

        def to_norm(arr: np.ndarray) -> np.ndarray:
            return (np.asarray(arr, dtype=float) - train_min) / denom

        def to_orig(arr: np.ndarray) -> np.ndarray:
            return np.asarray(arr, dtype=float) * denom + train_min

        pred_dir = CROSS_ROOT / "predictions" / f"{store}_{product}"
        b = np.genfromtxt(pred_dir / "fused_target_only.csv", delimiter=",", skip_header=1)
        c = np.genfromtxt(pred_dir / "raw_target_only.csv", delimiter=",", skip_header=1)
        a = np.genfromtxt(pred_dir / "full_current.csv", delimiter=",", skip_header=1)
        y_true_norm = b[:, 0]
        b_pred_norm = b[:, 1]
        c_pred_norm = c[:, 1]
        a_pred_norm = a[:, 1]
        y_true_orig_from_split = eval_sales[: y_true_norm.size]
        y_true_orig_inv = to_orig(y_true_norm)
        y_true_match = float(np.max(np.abs(y_true_orig_inv - y_true_orig_from_split))) if y_true_orig_from_split.size == y_true_norm.size else float("nan")

        # Rolling-origin BL2: mean of last 7 of the 10-day lookback. Origins start at observed_end + 10 days.
        full = sales_210
        observed_end_idx = TRAIN_DAYS + VAL_DAYS - 1  # day 30
        first_origin = observed_end_idx + WINDOW
        bl2 = []
        labels = []
        for origin in range(first_origin, len(full) - HORIZON):
            window = full[origin - WINDOW + 1 : origin + 1]
            bl2.append(float(window[-7:].mean()))
            labels.append(float(full[origin + HORIZON]))
        bl2 = np.asarray(bl2, dtype=float)
        labels_arr = np.asarray(labels, dtype=float)
        if bl2.size != y_true_norm.size:
            bl2_note = f"BL2 n={bl2.size} vs pred n={y_true_norm.size}"
        else:
            bl2_note = "aligned"

        rmse = lambda yt, yp: float(np.sqrt(np.mean((yt - yp) ** 2)))
        smape = lambda yt, yp: float(np.mean(2.0 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + 1e-8)) * 100.0)
        b_orig = to_orig(b_pred_norm)
        c_orig = to_orig(c_pred_norm)
        a_orig = to_orig(a_pred_norm)
        yt_orig = y_true_orig_from_split if y_true_orig_from_split.size == b_orig.size else to_orig(y_true_norm)

        # Residual vs MA
        if bl2.size == b_orig.size:
            b_minus_ma = b_orig - bl2
            c_minus_ma = c_orig - bl2
            yt_minus_ma = yt_orig - bl2
        else:
            b_minus_ma = np.array([])
            c_minus_ma = np.array([])
            yt_minus_ma = np.array([])

        sources = knn_results.get(f"{store}_{product}", [])
        source_mismatch = []
        for src in sources:
            src_store, src_product = str(src["source_entity"]).split("_")
            src_dates, src_sales = load_entity(source_table, src_store, src_product)
            obs_mask = src_dates <= np.datetime64("2025-01-14")
            src_obs = src_sales[obs_mask]
            source_mismatch.append(
                {
                    "source": src["source_entity"].replace("_", "/"),
                    "knn_distance": float(src["distance"]),
                    "knn_weight": float(src["weight"]),
                    "source_obs30_mean": float(src_obs.mean()) if src_obs.size else float("nan"),
                    "source_obs30_std": float(src_obs.std(ddof=0)) if src_obs.size else float("nan"),
                    "target_obs30_mean": float(np.concatenate([train, val]).mean()),
                    "mean_ratio_source_over_target": (
                        float(src_obs.mean() / np.concatenate([train, val]).mean())
                        if src_obs.size and np.concatenate([train, val]).mean() != 0
                        else float("nan")
                    ),
                    "source_obs_n": int(src_obs.size),
                    "source_window_n": int(src_sales.size),
                }
            )

        # Cross-target diagnostics distances if present
        runtime_distances = None
        targets_block = cross_diag.get("targets") or cross_diag.get("per_target") or {}
        if isinstance(targets_block, dict):
            tblock = targets_block.get(target) or targets_block.get(f"{store}_{product}") or {}
            runtime_distances = tblock.get("source_distances")
        if runtime_distances is None:
            for key in ("comparisons", "target_results", "results"):
                block = cross_diag.get(key)
                if isinstance(block, list):
                    hit = next((item for item in block if str(item.get("target_identity")) == target), None)
                    if hit:
                        runtime_distances = [
                            hit.get("source_1_distance"),
                            hit.get("source_2_distance"),
                            hit.get("source_3_distance"),
                        ]
                        break
                if isinstance(block, dict) and target in block:
                    hit = block[target]
                    runtime_distances = hit.get("source_distances") or [
                        hit.get("source_1_distance"),
                        hit.get("source_2_distance"),
                        hit.get("source_3_distance"),
                    ]
                    break

        shift = {
            "train_to_test_mean_ratio": float(test.mean() / train.mean()) if train.mean() != 0 else float("nan"),
            "train_to_test_std_ratio": float(test.std() / train.std()) if train.std() != 0 else float("nan"),
            "obs30_to_test_mean_ratio": float(test.mean() / np.concatenate([train, val]).mean())
            if np.concatenate([train, val]).mean() != 0
            else float("nan"),
            "ks_train_vs_test": ks_statistic(train, test),
            "ks_obs30_vs_eval170": ks_statistic(np.concatenate([train, val]), eval_sales),
            "mean_delta_test_minus_train": float(test.mean() - train.mean()),
        }

        week_labels, week_true = weekly_mean(yt_orig)
        _, week_b = weekly_mean(b_orig)
        _, week_c = weekly_mean(c_orig)
        _, week_ma = weekly_mean(bl2 if bl2.size == yt_orig.size else yt_orig)
        first_n = 28
        detail_labels = [str(i + 1) for i in range(min(first_n, yt_orig.size))]

        history_status = "MISSING"
        history_tail = None
        if target == "166/258":
            arms = mech_diag.get("arms") or {}
            history_tail = {
                name: (arms.get(name) or {}).get("history_tail")
                for name in ("full_current", "fused_target_only", "raw_target_only")
            }
            if any(history_tail.values()):
                history_status = "PARTIAL_LAST_3_EPOCHS_LOSS_ONLY"

        archetype[target] = {
            "archetype": ARCHETYPES[target],
            "dates": {
                "n_raw": int(sales_210.size),
                "n_210": int(sales_210.size),
                "train_start": str(dates_210[0])[:10] if len(dates_210) else None,
                "train_end": str(dates_210[TRAIN_DAYS - 1])[:10] if len(dates_210) >= TRAIN_DAYS else None,
                "val_end": str(dates_210[TRAIN_DAYS + VAL_DAYS - 1])[:10] if len(dates_210) >= TRAIN_DAYS + VAL_DAYS else None,
                "test_start": str(dates_210[TRAIN_DAYS + VAL_DAYS])[:10] if len(dates_210) > TRAIN_DAYS + VAL_DAYS else None,
                "test_end": str(dates_210[-1])[:10] if len(dates_210) else None,
                "eval_start": str(dates_210[TRAIN_DAYS + VAL_DAYS + WINDOW])[:10]
                if len(dates_210) > TRAIN_DAYS + VAL_DAYS + WINDOW
                else None,
            },
            "train": stats(train),
            "val": stats(val),
            "test_180": stats(test),
            "eval_170": stats(eval_sales),
            "obs_30": stats(np.concatenate([train, val])),
            "shift": shift,
            "scaler_train_min": train_min,
            "scaler_train_max": train_max,
            "y_true_inverse_max_abs_error_vs_split": y_true_match,
            "knn_json_sources": source_mismatch,
            "runtime_source_distances": runtime_distances,
            "predictions": {
                "n": int(y_true_norm.size),
                "space_note": "A/B/C CSV y_pred is normalized_minmax_space; original recovered with train-only MinMax",
                "A_rmse_norm": rmse(y_true_norm, a_pred_norm),
                "B_rmse_norm": rmse(y_true_norm, b_pred_norm),
                "C_rmse_norm": rmse(y_true_norm, c_pred_norm),
                "A_rmse_orig": rmse(yt_orig, a_orig),
                "B_rmse_orig": rmse(yt_orig, b_orig),
                "C_rmse_orig": rmse(yt_orig, c_orig),
                "BL2_rmse_orig": rmse(labels_arr, bl2) if bl2.size == labels_arr.size else float("nan"),
                "BL2_smape_orig": smape(labels_arr, bl2) if bl2.size == labels_arr.size else float("nan"),
                "BL2_rmse_norm": rmse(to_norm(labels_arr), to_norm(bl2)) if bl2.size == labels_arr.size else float("nan"),
                "B_mean_orig": float(b_orig.mean()),
                "C_mean_orig": float(c_orig.mean()),
                "A_mean_orig": float(a_orig.mean()),
                "BL2_mean_orig": float(bl2.mean()) if bl2.size else float("nan"),
                "y_true_mean_orig": float(yt_orig.mean()),
                "B_std_orig": float(b_orig.std(ddof=0)),
                "C_std_orig": float(c_orig.std(ddof=0)),
                "BL2_std_orig": float(bl2.std(ddof=0)) if bl2.size else float("nan"),
                "y_true_std_orig": float(yt_orig.std(ddof=0)),
                "corr_B_true": float(np.corrcoef(b_orig, yt_orig)[0, 1]),
                "corr_C_true": float(np.corrcoef(c_orig, yt_orig)[0, 1]),
                "corr_BL2_true": float(np.corrcoef(bl2, labels_arr)[0, 1]) if bl2.size == labels_arr.size else float("nan"),
                "corr_B_BL2": float(np.corrcoef(b_orig, bl2)[0, 1]) if bl2.size == b_orig.size else float("nan"),
                "B_minus_BL2_mean": float(b_minus_ma.mean()) if b_minus_ma.size else float("nan"),
                "B_minus_BL2_rmse": float(np.sqrt(np.mean(b_minus_ma**2))) if b_minus_ma.size else float("nan"),
                "C_minus_BL2_mean": float(c_minus_ma.mean()) if c_minus_ma.size else float("nan"),
                "true_minus_BL2_rmse": float(np.sqrt(np.mean(yt_minus_ma**2))) if yt_minus_ma.size else float("nan"),
                "bl2_note": bl2_note,
            },
            "weekly": {
                "labels": week_labels,
                "y_true": week_true,
                "B": week_b,
                "C": week_c,
                "BL2": week_ma,
            },
            "first_28": {
                "labels": detail_labels,
                "y_true": [float(x) for x in yt_orig[:first_n]],
                "B": [float(x) for x in b_orig[:first_n]],
                "C": [float(x) for x in c_orig[:first_n]],
                "BL2": [float(x) for x in bl2[:first_n]] if bl2.size >= first_n else [],
            },
            "fused_init_initial_predictions": "MISSING_NOT_SAVED",
            "epoch_1_to_50_prediction_trajectory": history_status,
            "history_tail": history_tail,
        }

    missing_items = [
        {
            "item": "fused initialization before-training predictions",
            "status": "MISSING",
            "reason": "A/B fused init digest exists; no checkpoint dump of epoch-0 train/val/test predictions.",
        },
        {
            "item": "epoch 1→50 train/val/test prediction trajectory",
            "status": "MISSING",
            "reason": "Only last-3-epoch loss tail stored in mechanism diagnostics; test predictions saved only after epoch 50.",
        },
    ]

    payload = {
        "no_tl_068_vs_15908": {
            "answer": (
                "0.68 and 1.5908 are the same identity cell (D4, 166/258, without, H=1, seed=42, "
                "170 samples, same sample_manifest_digest, original_sales RMSE) produced by two different runs. "
                "0.68 is a rounded stale smoke/extended trial from 20260727 (CSV RMSE=0.6793, sMAPE=37.49), "
                "later hardcoded into REFERENCE_METRICS. 1.5908 is the accepted formal confirmed_baseline "
                "from 20260816 commit 630ae4ef. Cross-target joined the accepted formal row. "
                "The mechanism-ablation prompt/hardcode did not."
            ),
            "same_identity": True,
            "same_test_slice": True,
            "same_producing_run": False,
            "stale": no_tl_stale,
            "accepted": no_tl_accepted,
            "hardcoded_reference_metrics": hardcoded,
            "hardcoded_vs_stale_rmse": {
                "hardcoded": hardcoded.get("FORMAL_NO_TL"),
                "stale_run_rmse": None if no_tl_stale is None else no_tl_stale.get("RMSE"),
                "stale_run_smape": None if no_tl_stale is None else no_tl_stale.get("sMAPE"),
            },
            "msml_hardcoded_vs_accepted": {
                "hardcoded": hardcoded.get("MSML_TL"),
                "stale": None if msml_stale is None else {"RMSE": msml_stale.get("RMSE"), "sMAPE": msml_stale.get("sMAPE")},
                "accepted": {"RMSE": msml_accepted["RMSE"], "sMAPE": msml_accepted["sMAPE"]},
            },
            "bl4_hardcoded_vs_accepted": {
                "hardcoded": hardcoded.get("BL4_LSTM"),
                "accepted": {"RMSE": bl4_accepted["RMSE"], "sMAPE": bl4_accepted["sMAPE"]},
            },
        },
        "join_audit": join_audit,
        "accepted_references": accepted,
        "stale_references": stale_refs,
        "missing": missing_items,
        "archetype": archetype,
        "cross_diag_keys": sorted(cross_diag.keys()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"WROTE {OUT}")
    print("NO_TL_STALE", None if no_tl_stale is None else (no_tl_stale["RMSE"], no_tl_stale["sMAPE"]))
    print("NO_TL_ACCEPTED", no_tl_accepted["RMSE"], no_tl_accepted["sMAPE"])
    print("JOIN_MISSING", [j for j in join_audit if j["status"] != "FOUND"])
    for target, block in archetype.items():
        pred = block["predictions"]
        print(
            target,
            "train_mean",
            round(block["train"]["mean"], 4),
            "test_mean",
            round(block["test_180"]["mean"], 4),
            "B_orig",
            round(pred["B_rmse_orig"], 4),
            "BL2",
            round(pred["BL2_rmse_orig"], 4),
            "y_true_match",
            block["y_true_inverse_max_abs_error_vs_split"],
            "bl2_note",
            pred["bl2_note"],
        )


if __name__ == "__main__":
    main()
