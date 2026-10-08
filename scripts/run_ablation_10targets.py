#!/usr/bin/env python3
"""10-target F1-F4/P1/P2 ablation on D4/D5/D6 (main LR).

Reuses 消融实验/runs for the three dagger targets. Does not modify those
files, SITA/MSML sources, or main-table workbooks. Resume via DONE.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "消融实验") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "消融实验"))

from additional_target_robustness.paths import SOURCE_MODEL_CACHE, ensure_ablation_path
from additional_target_robustness.protocol import (
    HORIZONS,
    LEARNING_RATES,
    SEEDS,
    TEST_DAYS,
    ProtocolError,
    build_evaluation_index,
    calendar_for,
    frozen_protocol_sha256,
)
from additional_target_robustness.runner import _load_or_train_source
from additional_target_robustness.source_selection import load_source_histories, select_top3
from additional_target_robustness.target_data import load_target_series, split_target_frame

ensure_ablation_path()
from sita_ablation.adaptation import TargetWindows, WeightSnapshot, adapt_target
from sita_ablation.arms import RunSpec
from sita_ablation.metrics import evaluate_original_sales
from sita_ablation.models import build_target_cnn
from sita_ablation.revin import build_revin
from sita_ablation.scaling import fit_target_model_scaler, transform_model_frame
from sita_ablation.stage_a import NOT_APPLICABLE, SourceRecord, StageABundle, StageAKey
from sita_ablation.windows import build_target_partition_windows, build_test_windows

CONFIG = json.loads((REPO_ROOT / "configs" / "ablation_targets_10.json").read_text(encoding="utf-8"))
OUT_ROOT = REPO_ROOT / "outputs" / "ablation_10targets"
RESULTS_CSV = REPO_ROOT / "results" / "ablation_10targets_5seed.csv"
FAILURES_CSV = REPO_ROOT / "results" / "ablation_10targets_failures.csv"
LOG_PATH = REPO_ROOT / "logs" / "ablation_10targets.log"
MANIFEST = (
    REPO_ROOT
    / "实验结果"
    / "非实验要用的结果"
    / "sita_d456_fixed40_headline_paper180swap"
    / "fixed40_manifest.csv"
)
CONDITIONS = tuple(CONFIG["conditions"])
EXPECTED = 3 * 10 * 6 * 3 * 5
ARM_KWARGS = {
    "F1": dict(revin=False, freeze_conv=False, selection_rule="none", requested_k=None, weight_mode="none", comparison_days=None),
    "F2": dict(revin=True, freeze_conv=False, selection_rule="none", requested_k=None, weight_mode="none", comparison_days=None),
    "F3": dict(revin=False, freeze_conv=False, selection_rule="top", requested_k=3, weight_mode="inverse_distance", comparison_days=30),
    "F4": dict(revin=True, freeze_conv=False, selection_rule="top", requested_k=3, weight_mode="inverse_distance", comparison_days=30),
    "P1": dict(revin=True, freeze_conv=True, selection_rule="top", requested_k=3, weight_mode="inverse_distance", comparison_days=30),
    "P2": dict(revin=True, freeze_conv=False, selection_rule="top", requested_k=1, weight_mode="single", comparison_days=30),
}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"{_utc()} {msg}"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def mae_original(y_true, y_pred) -> float:
    t = np.clip(np.asarray(y_true, dtype=np.float64), 0.0, None)
    p = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    return float(np.mean(np.abs(p - t)))


def identities(dataset: str) -> list[str]:
    return list(CONFIG["datasets"][dataset]["all10"])


def is_dagger(dataset: str, identity: str) -> bool:
    return identity == CONFIG["dagger_targets"][dataset]


_MANIFEST_FRAME: pd.DataFrame | None = None


def manifest_frame() -> pd.DataFrame:
    global _MANIFEST_FRAME
    if _MANIFEST_FRAME is None:
        _MANIFEST_FRAME = pd.read_csv(MANIFEST)
    return _MANIFEST_FRAME


def universe_id(dataset: str, identity: str) -> str | None:
    frame = manifest_frame()
    hit = frame.loc[(frame["dataset"] == dataset) & (frame["target_identity"].astype(str) == identity)]
    if hit.empty:
        return None
    return str(hit.iloc[0]["source_universe_id"])


def cell_dir(dataset: str, identity: str, arm: str, horizon: int, seed: int) -> Path:
    return OUT_ROOT / dataset / identity.replace("/", "_") / arm / f"h{horizon}" / f"s{seed}"


def ablation_run_dir(dataset: str, arm: str, horizon: int, seed: int) -> Path:
    return REPO_ROOT / "消融实验" / "runs" / arm / "main" / f"{dataset}_h{horizon}" / f"seed{seed}"


def inspect(directory: Path) -> str:
    done = directory / "DONE.json"
    if not done.is_file():
        return "MISS"
    payload = json.loads(done.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETED":
        return "NOT_COMPLETED"
    needed = ("RMSE", "sMAPE", "MAE")
    if any(k not in payload or not np.isfinite(float(payload[k])) for k in needed):
        return "INCOMPLETE_METRICS"
    return "HIT"


def append_failure(row: dict) -> None:
    FAILURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(FAILURES_CSV, mode="a", header=not FAILURES_CSV.is_file(), index=False)


def write_done(directory: Path, record: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / "DONE.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(directory / "DONE.json")


def reuse_dagger(dataset: str, identity: str, arm: str, horizon: int, seed: int) -> dict:
    src = ablation_run_dir(dataset, arm, horizon, seed)
    rec = json.loads((src / "cell_record.json").read_text(encoding="utf-8"))
    pred = pd.read_parquet(src / "predictions.parquet")
    y_true = pred["y_true_original"].to_numpy(dtype=np.float64)
    y_pred = pred["y_pred_original"].to_numpy(dtype=np.float64)
    pair = evaluate_original_sales(y_true, y_pred)
    stored = rec.get("pair") or {}
    if abs(float(stored["rmse"]) - pair.rmse) > 1e-10 or abs(float(stored["smape"]) - pair.smape) > 1e-10:
        raise ProtocolError(
            f"reuse mismatch {dataset} {identity} {arm} h={horizon} s={seed}"
        )
    if int(rec.get("n_test", 0)) != TEST_DAYS or len(pred) != TEST_DAYS:
        raise ProtocolError("reused dagger n_test is not 180")
    record = {
        "status": "COMPLETED",
        "dataset": dataset,
        "target": identity,
        "condition": arm,
        "h": int(horizon),
        "seed": int(seed),
        "sMAPE": pair.smape,
        "RMSE": pair.rmse,
        "MAE": mae_original(y_true, y_pred),
        "n_test": TEST_DAYS,
        "reused_from": str(src.relative_to(REPO_ROOT)),
        "trained_at": _utc(),
    }
    write_done(cell_dir(dataset, identity, arm, horizon, seed), record)
    return record


def make_spec(dataset: str, arm: str, horizon: int, seed: int) -> RunSpec:
    kw = dict(ARM_KWARGS[arm])
    return RunSpec(
        arm=arm,
        protocol_tag="main",
        dataset=dataset,
        horizon=int(horizon),
        seed=int(seed),
        backbone="cnn",
        target_learning_rate=float(LEARNING_RATES[dataset]),
        **kw,
    )


def run_new(dataset: str, identity: str, arm: str, horizon: int, seed: int) -> dict:
    import tensorflow as tf

    calendar = calendar_for(dataset)
    target = load_target_series(dataset, identity)
    train, validation, _test = split_target_frame(target, calendar)
    features = calendar.model_features
    scaler = fit_target_model_scaler(train, features)
    index = build_evaluation_index(dataset, horizon)
    if set(index.label_date_text()) & set(scaler.fit_dates):
        raise ProtocolError("target scaler fit includes a test date")
    train_scaled = transform_model_frame(scaler, train)
    validation_scaled = transform_model_frame(scaler, validation)
    observed = pd.concat(
        [validation, target.loc[target["date"] >= pd.Timestamp(calendar.test_start)]],
        ignore_index=True,
    )
    observed_scaled = transform_model_frame(scaler, observed)
    key = tuple(identity.split("/"))
    train_windows = build_target_partition_windows(
        train_scaled, key_fields=calendar.key_fields, key=key,
        feature_columns=features, horizon=horizon, partition="train",
    )
    validation_windows = build_target_partition_windows(
        validation_scaled, key_fields=calendar.key_fields, key=key,
        feature_columns=features, horizon=horizon, partition="validation",
    )
    test_windows = build_test_windows(
        observed_scaled, test_start=calendar.test_start, test_end=calendar.test_end,
        feature_columns=features, horizon=horizon,
    )
    got = tuple(pd.Timestamp(day).strftime("%Y-%m-%d") for day in test_windows.label_date)
    if got != index.label_date_text():
        raise ProtocolError("test windows do not match evaluation index")
    spec = make_spec(dataset, arm, horizon, seed)
    bundle = None
    if spec.selection_rule != "none":
        histories = load_source_histories(dataset, target_identity=identity, target_frame=target)
        selection = select_top3(
            dataset, identity, target, histories, source_universe_id=universe_id(dataset, identity)
        )
        if selection.status != "COMPLETED" or len(selection.selected) != 3:
            raise ProtocolError(f"Top-3 failed: {selection.status} n={len(selection.selected)}")
        sources = []
        for selected in selection.selected:
            trained, _status = _load_or_train_source(
                histories, source_key=selected.key, features=features,
                horizon=horizon, seed=seed, dataset=dataset, key_fields=calendar.key_fields,
            )
            sources.append(
                SourceRecord(
                    rank=selected.rank, source_key=selected.key, weights=trained.weights,
                    scaler=trained.scaler, sample_count=trained.sample_count,
                )
            )
        bundle = StageABundle(
            key=StageAKey(
                dataset=dataset, horizon=horizon, seed=seed, backbone="cnn",
                selection_rule="top", trained_k=3,
                comparison_start=calendar.comparison_start, comparison_end=calendar.comparison_end,
                source_history_start=calendar.history_start, source_history_end=calendar.history_end,
                model_features=features, similarity_variables=calendar.similarity_variables,
                scaling_rule="per_similarity_variable_minmax", draw_rule=None,
                frozen_config_sha256=frozen_protocol_sha256(),
            ),
            status="COMPLETED",
            distances=np.asarray([item.distance for item in selection.selected], dtype=np.float64),
            selected=tuple(item.key for item in selection.selected),
            sources=tuple(sources),
            source_validation_metric=NOT_APPLICABLE,
            source_validation_value_space=NOT_APPLICABLE,
            cache_dir=SOURCE_MODEL_CACHE,
        )
    tf.keras.utils.set_random_seed(int(seed))
    model, _ = build_target_cnn((10, len(features)))
    tensors: dict[str, tuple[np.ndarray, ...]] = {}
    for layer in model.layers:
        arrays = layer.get_weights()
        if arrays:
            tensors[layer.name] = tuple(np.asarray(array, dtype=np.float32) for array in arrays)
    revin = build_revin(len(features))
    tensors["affine_weight"] = (np.asarray(revin.affine_weight.numpy(), dtype=np.float32),)
    tensors["affine_bias"] = (np.asarray(revin.affine_bias.numpy(), dtype=np.float32),)
    result = adapt_target(
        spec, TargetWindows(train_windows, validation_windows, test_windows),
        scaler, WeightSnapshot(tensors), bundle,
    )
    y_true = np.asarray(test_windows.y_original, dtype=np.float64)
    y_pred = np.asarray(result.test_predictions, dtype=np.float64)
    pair = evaluate_original_sales(y_true, y_pred)
    directory = cell_dir(dataset, identity, arm, horizon, seed)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"target_date": list(index.label_date_text()), "y_true": y_true, "y_pred": y_pred}
    ).to_csv(directory / "predictions.csv", index=False)
    tf.keras.backend.clear_session()
    record = {
        "status": "COMPLETED", "dataset": dataset, "target": identity, "condition": arm,
        "h": int(horizon), "seed": int(seed), "sMAPE": pair.smape, "RMSE": pair.rmse,
        "MAE": mae_original(y_true, y_pred), "n_test": TEST_DAYS, "reused_from": None,
        "trained_at": _utc(),
    }
    write_done(directory, record)
    return record


def iter_cells(datasets, conditions):
    for dataset in datasets:
        for identity in identities(dataset):
            for arm in conditions:
                for horizon in HORIZONS:
                    for seed in SEEDS:
                        yield dataset, identity, arm, int(horizon), int(seed)


def assemble(datasets) -> pd.DataFrame:
    rows = []
    for dataset, identity, arm, horizon, seed in iter_cells(datasets, CONDITIONS):
        directory = cell_dir(dataset, identity, arm, horizon, seed)
        if inspect(directory) != "HIT":
            continue
        payload = json.loads((directory / "DONE.json").read_text(encoding="utf-8"))
        rows.append({
            "dataset": dataset, "target": identity, "condition": arm,
            "h": int(horizon), "seed": int(seed), "sMAPE": float(payload["sMAPE"]),
            "RMSE": float(payload["RMSE"]), "MAE": float(payload["MAE"]),
        })
    frame = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        frame = pd.DataFrame(columns=["dataset", "target", "condition", "h", "seed", "sMAPE", "RMSE", "MAE"])
    frame.to_csv(RESULTS_CSV, index=False)
    return frame


def run_all(datasets, conditions, *, limit=None) -> dict:
    already_done = reused = newly_trained = failures = 0
    n_done = 0
    for dataset, identity, arm, horizon, seed in iter_cells(datasets, conditions):
        if limit is not None and n_done >= limit:
            break
        directory = cell_dir(dataset, identity, arm, horizon, seed)
        if inspect(directory) == "HIT":
            already_done += 1
            continue
        try:
            if is_dagger(dataset, identity):
                reuse_dagger(dataset, identity, arm, horizon, seed)
                _log(f"REUSE {dataset} {identity} {arm} h={horizon} s={seed}")
                reused += 1
            else:
                run_new(dataset, identity, arm, horizon, seed)
                _log(f"OK {dataset} {identity} {arm} h={horizon} s={seed}")
                newly_trained += 1
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            n_done += 1
            detail = f"{type(exc).__name__}: {exc}"
            _log(f"FAIL {dataset} {identity} {arm} h={horizon} s={seed} {detail}")
            append_failure({
                "dataset": dataset, "target": identity, "condition": arm,
                "h": horizon, "seed": seed, "error": detail,
                "traceback": traceback.format_exc()[-4000:], "written_at": _utc(),
            })
    frame = assemble(datasets)
    payload = {
        "already_done": already_done,
        "reused": reused,
        "newly_trained": newly_trained,
        "failures": failures,
        "assembled": int(len(frame)),
        "expected": EXPECTED,
        "missing": EXPECTED - len(frame),
    }
    summary_path = REPO_ROOT / "results" / "ablation_10targets_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--datasets", default="D5,D6,D4")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    conditions = tuple(x.strip() for x in args.conditions.split(",") if x.strip())
    if args.status:
        hits = sum(
            1 for dataset, identity, arm, horizon, seed in iter_cells(datasets, conditions)
            if inspect(cell_dir(dataset, identity, arm, horizon, seed)) == "HIT"
        )
        print(json.dumps({"hits": hits, "expected": EXPECTED, "missing": EXPECTED - hits}, indent=2))
        return 0
    if args.smoke:
        dataset, identity = "D4", CONFIG["datasets"]["D4"]["sampled9"][3]
        rec = run_new(dataset, identity, "F1", 1, 42)
        _log(f"SMOKE {dataset} {identity} F1 {rec}")
        print(json.dumps(rec, indent=2))
        return 0 if np.isfinite(rec["RMSE"]) and rec["n_test"] == 180 else 1
    if args.assemble:
        frame = assemble(datasets)
        print(f"assembled {len(frame)} -> {RESULTS_CSV}")
        return 0
    payload = run_all(datasets, conditions, limit=args.limit)
    print(json.dumps(payload, indent=2))
    return 0 if int(payload["failures"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
