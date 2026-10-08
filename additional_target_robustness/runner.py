"""Independent dry-run, canary, and full-run entry.

Target identities are read from the frozen manifest. Source calendars are
hashed and read; they are not rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from additional_target_robustness.baselines import (
    fit_and_predict_patchtst,
    fit_and_predict_timesnet,
    materialize_seasonal_seeds,
    partition_sales,
    predict_seasonal,
)
from additional_target_robustness.io import write_canary
from additional_target_robustness.manifest import (
    ManifestRecord,
    load_manifest,
    require_manifest_target,
    verify_source_caches,
)
from additional_target_robustness.paths import OUTPUT_ROOT, SOURCE_MODEL_CACHE, ensure_ablation_path
from additional_target_robustness.protocol import (
    HORIZONS,
    LEARNING_RATES,
    METHODS,
    TEST_DAYS,
    ProtocolError,
    ProtocolHold,
    build_evaluation_index,
    calendar_for,
    frozen_protocol_sha256,
)
from additional_target_robustness.source_selection import SelectionOutcome, load_source_histories, select_top3
from additional_target_robustness.target_data import load_target_series, sales_by_date, split_target_frame

CANARY_DATASET = "D1"
CANARY_IDENTITY = "2/6"
CANARY_SEED = 42
CANARY_DIR = OUTPUT_ROOT / "canary_D1_2_6"
D1_SOURCE_UNIVERSE = "store=1,item=1-9"


@dataclass
class _Log:
    lines: list[str]

    def write(self, message: str) -> None:
        print(message, flush=True)
        self.lines.append(message)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def seasonal_seed_records(predictions: np.ndarray):
    """Copy one seasonal forecast onto seeds 42–46. The canary itself records seed 42."""
    return materialize_seasonal_seeds(predictions)


def dry_run() -> dict[str, object]:
    """Check the freeze, build D1/2/6, select Top-3, and index 180 dates. No training."""
    record, target, selection, indexes, hashes = _prepare(CANARY_DATASET, CANARY_IDENTITY)
    summary = {
        "status": "DRY_RUN_OK",
        "dataset": record.dataset,
        "target_identity": record.target_identity,
        "source_universe_id": record.source_universe_id,
        "source_universe": D1_SOURCE_UNIVERSE,
        "selection_status": selection.status,
        "selected": [
            {
                "identity": item.identity,
                "distance": item.distance,
                "weight": item.weight,
            }
            for item in selection.selected
        ],
        "n_test": {str(horizon): len(index.rows) for horizon, index in indexes.items()},
        "source_cache_sha256": hashes,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def canary() -> str:
    """Train D1/2/6 at h=1,3,5 and seed 42. Seasonal Naive stays on seed 42 here."""
    log = _Log([])
    status = "BLOCKED_IMPLEMENTATION_ERROR"
    try:
        status = _run_canary(log)
    except ProtocolHold as exc:
        log.write(exc.token)
        status = "BLOCKED_PROTOCOL_MISMATCH"
        _write_blocked(log, status, exc.token)
    except ProtocolError as exc:
        log.write(f"BLOCKED_PROTOCOL_MISMATCH {exc}")
        status = "BLOCKED_PROTOCOL_MISMATCH"
        _write_blocked(log, status, str(exc))
    except Exception as exc:
        log.write("BLOCKED_IMPLEMENTATION_ERROR")
        log.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        status = "BLOCKED_IMPLEMENTATION_ERROR"
        _write_blocked(log, status, str(exc))
    log.write(status)
    return status


def _prepare(
    dataset: str,
    identity: str,
) -> tuple[ManifestRecord, pd.DataFrame, SelectionOutcome, dict[int, object], dict[str, str]]:
    records = load_manifest()
    record = require_manifest_target(records, dataset, identity)
    hashes = verify_source_caches()
    target = load_target_series(dataset, identity)
    calendar = calendar_for(dataset)
    split_target_frame(target, calendar)
    histories = load_source_histories(dataset, target_identity=identity, target_frame=target)
    selection = select_top3(
        dataset,
        identity,
        target,
        histories,
        source_universe_id=record.source_universe_id,
    )
    if selection.status != "COMPLETED" or len(selection.selected) != 3:
        raise ProtocolError(f"{dataset} {identity} selection status is {selection.status}")
    _require_d1_universe(selection, identity)
    indexes = {}
    for horizon in HORIZONS:
        index = build_evaluation_index(dataset, horizon)
        if len(index.rows) != TEST_DAYS:
            raise ProtocolError(f"h={horizon} evaluation index has {len(index.rows)} dates")
        if index.label_date_text()[0] != calendar.test_start or index.label_date_text()[-1] != calendar.test_end:
            raise ProtocolError(f"h={horizon} evaluation index does not cover the frozen test span")
        indexes[horizon] = index
    return record, target, selection, indexes, hashes


def _run_canary(log: _Log) -> str:
    log.write("canary start D1/2/6 seed 42")
    hashes_before = verify_source_caches()
    record, target, selection, indexes, _hashes = _prepare(CANARY_DATASET, CANARY_IDENTITY)
    if hashes_before != _hashes:
        raise ProtocolHold("HOLD_SOURCE_CACHE_HASH_MISMATCH")
    calendar = calendar_for(CANARY_DATASET)
    sales = sales_by_date(target)
    train, validation, test = split_target_frame(target, calendar)
    del test
    train_sales = partition_sales(target, calendar.train_start, calendar.train_end)
    validation_sales = partition_sales(target, calendar.val_start, calendar.val_end)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    cache_rows: list[dict[str, object]] = []
    horizon_checks: dict[str, object] = {}
    for horizon in HORIZONS:
        index = indexes[horizon]
        y_true = np.asarray([sales[day] for day in index.label_date_text()], dtype=np.float64)
        log.write(f"horizon {horizon}: seasonal naive")
        seasonal = predict_seasonal(CANARY_DATASET, CANARY_IDENTITY, horizon, index, sales)
        log.write(f"horizon {horizon}: source cnn and sita")
        sita, usage = _predict_sita(
            target,
            train,
            validation,
            selection,
            index,
            horizon=horizon,
            seed=CANARY_SEED,
        )
        cache_rows.extend(usage)
        log.write(f"horizon {horizon}: patchtst")
        patchtst = fit_and_predict_patchtst(
            train_sales, validation_sales, index, sales, horizon=horizon, seed=CANARY_SEED
        )
        log.write(f"horizon {horizon}: timesnet")
        timesnet = fit_and_predict_timesnet(
            train_sales, validation_sales, index, sales, horizon=horizon, seed=CANARY_SEED
        )
        forecasts = {
            "SITA": sita,
            "SeasonalNaive": seasonal,
            "PatchTST": patchtst,
            "TimesNet": timesnet,
        }
        dates = {method: index.label_date_text() for method in METHODS}
        if any(len(values) != TEST_DAYS for values in forecasts.values()):
            raise ProtocolError(f"h={horizon} did not score {TEST_DAYS} dates")
        if len(set(dates.values())) != 1:
            raise ProtocolError(f"h={horizon} methods do not share target dates")
        for row in index.rows:
            index_rows.append(
                {
                    "horizon": horizon,
                    "label_date": row.label_date.strftime("%Y-%m-%d"),
                    "forecast_origin": row.forecast_origin.strftime("%Y-%m-%d"),
                    "input_start": row.input_start.strftime("%Y-%m-%d"),
                    "input_end": row.input_end.strftime("%Y-%m-%d"),
                    "input_dates": ",".join(day.strftime("%Y-%m-%d") for day in row.input_dates),
                }
            )
        for method, values in forecasts.items():
            _append_predictions(
                prediction_rows,
                metric_rows,
                method=method,
                horizon=horizon,
                dates=dates[method],
                y_true=y_true,
                y_pred=np.asarray(values, dtype=np.float64),
                deterministic_reuse=False,
            )
        horizon_checks[str(horizon)] = {
            "n_test": {method: int(len(values)) for method, values in forecasts.items()},
            "target_dates_equal": True,
            "first_target_date": dates["SITA"][0],
            "last_target_date": dates["SITA"][-1],
            "methods": list(METHODS),
        }
        log.write(
            "horizon {0} dates {1}..{2} n_test {3}".format(
                horizon, dates["SITA"][0], dates["SITA"][-1], TEST_DAYS
            )
        )
    hashes_after = verify_source_caches()
    if hashes_before != hashes_after:
        raise ProtocolHold("HOLD_SOURCE_CACHE_HASH_MISMATCH")
    metrics = pd.DataFrame(metric_rows)
    ready = bool(set(metrics["n_test"].tolist()) == {TEST_DAYS} and len(metrics) == 12)
    status = "READY_FOR_FULL_RUN" if ready else "BLOCKED_PROTOCOL_MISMATCH"
    artifacts = {
        "protocol_check": {
            "status": status,
            "dataset": CANARY_DATASET,
            "target_identity": CANARY_IDENTITY,
            "seed": CANARY_SEED,
            "source_universe_id": record.source_universe_id,
            "source_universe": D1_SOURCE_UNIVERSE,
            "manifest_sha256": "c83d03d337ef2b8c31d1e6b8aa0ea94f2b4ae8f7c6b312c39aedfdaf4035fec9",
            "source_cache_sha256_before": hashes_before,
            "source_cache_sha256_after": hashes_after,
            "source_cache_unchanged": True,
            "horizons": horizon_checks,
            "seasonal_canary_seed": CANARY_SEED,
            "seasonal_deterministic_reuse_in_canary": False,
        },
        "source_selection": _selection_payload(record, selection, cache_rows),
        "cache_usage": {"records": cache_rows},
        "evaluation_index": pd.DataFrame(index_rows),
        "predictions": pd.DataFrame(prediction_rows),
        "metrics": metrics,
        "run_log": log.text(),
    }
    write_canary(CANARY_DIR, artifacts)
    log.write(f"wrote {CANARY_DIR}")
    return status


def _predict_sita(
    target: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selection: SelectionOutcome,
    index,
    *,
    horizon: int,
    seed: int,
    dataset: str = CANARY_DATASET,
    identity: str = CANARY_IDENTITY,
    histories: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    ensure_ablation_path()
    import tensorflow as tf
    from sita_ablation.adaptation import TargetWindows, WeightSnapshot, adapt_target
    from sita_ablation.arms import RunSpec
    from sita_ablation.metrics import evaluate_original_sales
    from sita_ablation.models import build_target_cnn
    from sita_ablation.revin import build_revin
    from sita_ablation.scaling import fit_target_model_scaler, transform_model_frame
    from sita_ablation.stage_a import NOT_APPLICABLE, SourceRecord, StageABundle, StageAKey
    from sita_ablation.windows import build_target_partition_windows, build_test_windows

    calendar = calendar_for(dataset)
    features = calendar.model_features
    scaler = fit_target_model_scaler(train, features)
    test_days = set(index.label_date_text())
    if test_days & set(scaler.fit_dates):
        raise ProtocolError("target scaler fit includes a test date")
    train_scaled = transform_model_frame(scaler, train)
    validation_scaled = transform_model_frame(scaler, validation)
    observed = pd.concat([validation, target.loc[target["date"] >= pd.Timestamp(calendar.test_start)]], ignore_index=True)
    observed_scaled = transform_model_frame(scaler, observed)
    key = tuple(identity.split("/"))
    train_windows = build_target_partition_windows(
        train_scaled,
        key_fields=calendar.key_fields,
        key=key,
        feature_columns=features,
        horizon=horizon,
        partition="train",
    )
    validation_windows = build_target_partition_windows(
        validation_scaled,
        key_fields=calendar.key_fields,
        key=key,
        feature_columns=features,
        horizon=horizon,
        partition="validation",
    )
    test_windows = build_test_windows(
        observed_scaled,
        test_start=calendar.test_start,
        test_end=calendar.test_end,
        feature_columns=features,
        horizon=horizon,
    )
    got_labels = tuple(pd.Timestamp(day).strftime("%Y-%m-%d") for day in test_windows.label_date)
    if got_labels != index.label_date_text():
        raise ProtocolError("SITA test windows do not match the frozen evaluation index")
    got_ends = tuple(pd.Timestamp(day).strftime("%Y-%m-%d") for day in test_windows.input_end)
    expected_ends = tuple(row.input_end.strftime("%Y-%m-%d") for row in index.rows)
    if got_ends != expected_ends:
        raise ProtocolError("SITA forecast origins do not match the frozen evaluation index")
    if histories is None:
        histories = load_source_histories(dataset, target_identity=identity, target_frame=target)
    sources: list[SourceRecord] = []
    usage: list[dict[str, object]] = []
    for selected in selection.selected:
        trained, cache_status = _load_or_train_source(
            histories,
            source_key=selected.key,
            features=features,
            horizon=horizon,
            seed=seed,
            dataset=dataset,
            key_fields=calendar.key_fields,
        )
        sources.append(
            SourceRecord(
                rank=selected.rank,
                source_key=selected.key,
                weights=trained.weights,
                scaler=trained.scaler,
                sample_count=trained.sample_count,
            )
        )
        usage.append(
            {
                "dataset": dataset,
                "source_identity": selected.identity,
                "horizon": horizon,
                "seed": seed,
                "architecture": "cnn",
                "distance": selected.distance,
                "weight": selected.weight,
                "cache_status": cache_status,
                "source_universe_id": selection.source_universe_id,
            }
        )
    stage_key = StageAKey(
        dataset=dataset,
        horizon=horizon,
        seed=seed,
        backbone="cnn",
        selection_rule="top",
        trained_k=3,
        comparison_start=calendar.comparison_start,
        comparison_end=calendar.comparison_end,
        source_history_start=calendar.history_start,
        source_history_end=calendar.history_end,
        model_features=features,
        similarity_variables=calendar.similarity_variables,
        scaling_rule="per_similarity_variable_minmax",
        draw_rule=None,
        frozen_config_sha256=frozen_protocol_sha256(),
    )
    bundle = StageABundle(
        key=stage_key,
        status="COMPLETED",
        distances=np.asarray([item.distance for item in selection.selected], dtype=np.float64),
        selected=tuple(item.key for item in selection.selected),
        sources=tuple(sources),
        source_validation_metric=NOT_APPLICABLE,
        source_validation_value_space=NOT_APPLICABLE,
        cache_dir=SOURCE_MODEL_CACHE,
    )
    tf.keras.utils.set_random_seed(int(seed))
    model, _extractor = build_target_cnn((10, len(features)))
    tensors: dict[str, tuple[np.ndarray, ...]] = {}
    for layer in model.layers:
        arrays = layer.get_weights()
        if arrays:
            tensors[layer.name] = tuple(np.asarray(array, dtype=np.float32) for array in arrays)
    revin = build_revin(len(features))
    tensors["affine_weight"] = (np.asarray(revin.affine_weight.numpy(), dtype=np.float32),)
    tensors["affine_bias"] = (np.asarray(revin.affine_bias.numpy(), dtype=np.float32),)
    spec = RunSpec(
        arm="F4",
        protocol_tag="main",
        dataset=dataset,
        horizon=horizon,
        seed=seed,
        backbone="cnn",
        revin=True,
        freeze_conv=False,
        selection_rule="top",
        requested_k=3,
        weight_mode="inverse_distance",
        comparison_days=30,
        target_learning_rate=LEARNING_RATES[dataset],
    )
    result = adapt_target(
        spec,
        TargetWindows(train_windows, validation_windows, test_windows),
        scaler,
        WeightSnapshot(tensors),
        bundle,
    )
    if result.test_predictions.shape != (TEST_DAYS,):
        raise ProtocolError(f"SITA returned {result.test_predictions.shape[0]} predictions")
    scored = evaluate_original_sales(
        np.asarray(test_windows.y_original, dtype=np.float64),
        np.asarray(result.test_predictions, dtype=np.float64),
    )
    del scored
    tf.keras.backend.clear_session()
    return np.asarray(result.test_predictions, dtype=np.float64), usage


def _load_or_train_source(histories, *, source_key, features, horizon, seed, dataset, key_fields):
    ensure_ablation_path()
    import tensorflow as tf
    from sita_ablation.data import split_source_frame
    from sita_ablation.scaling import fit_source_model_scaler, transform_model_frame
    from sita_ablation.stage_a import fit_source_cnn
    from sita_ablation.windows import build_partition_windows

    series = _one_source(histories, source_key, key_fields)
    identity = "/".join(source_key)
    payload = {
        "dataset": dataset,
        "source_identity": identity,
        "horizon": int(horizon),
        "seed": int(seed),
        "architecture": "cnn",
        "source_data_sha256": _series_sha256(series, features),
        "protocol_sha256": frozen_protocol_sha256(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    directory = _owned_cache(SOURCE_MODEL_CACHE / digest)
    done_path = directory / "DONE.json"
    if done_path.is_file():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("cache_key_sha256") != digest or done.get("status") != "COMPLETED":
            raise ProtocolHold("HOLD_HASH_MISMATCH")
        return _read_source_record(directory, source_key), "CACHE_HIT"
    if directory.exists():
        _remove_tree(directory)
    partial = directory.with_name(directory.name + ".partial")
    if partial.exists():
        _remove_tree(partial)
    partial.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(int(seed))
    split = split_source_frame(series, features)
    scaler = fit_source_model_scaler(split.train, features)
    transformed = transform_model_frame(scaler, split.train)
    windows = build_partition_windows(
        transformed,
        key_fields=key_fields,
        key=source_key,
        feature_columns=features,
        horizon=int(horizon),
        partition="train",
    )
    weights = fit_source_cnn(np.asarray(windows.x, dtype=np.float32), np.asarray(windows.y_scaled, dtype=np.float32))
    _write_source_cache(partial, payload, digest, source_key, weights, scaler, int(windows.y_scaled.shape[0]))
    partial.rename(directory)
    tf.keras.backend.clear_session()
    return _read_source_record(directory, source_key), "TRAINED"


def _one_source(frame: pd.DataFrame, source_key: tuple[str, ...], key_fields: tuple[str, ...]) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for field, value in zip(key_fields, source_key):
        mask &= frame[field].map(lambda item: str(item).strip()) == value
    series = frame.loc[mask].sort_values("date", kind="mergesort").reset_index(drop=True)
    if len(series) != 180 or series["date"].duplicated().any():
        raise ProtocolError(f"source {'/'.join(source_key)} does not have 180 unique history days")
    return series


def _append_predictions(rows, metrics, *, method, horizon, dates, y_true, y_pred, deterministic_reuse):
    ensure_ablation_path()
    from sita_ablation.metrics import evaluate_original_sales

    if len(dates) != TEST_DAYS or len(y_true) != TEST_DAYS or len(y_pred) != TEST_DAYS:
        raise ProtocolError(f"{method} h={horizon} n_test is not {TEST_DAYS}")
    pair = evaluate_original_sales(y_true, y_pred)
    finite_pred = int(np.isfinite(y_pred).sum())
    finite_true = int(np.isfinite(y_true).sum())
    metrics.append(
        {
            "horizon": horizon,
            "method": method,
            "seed": CANARY_SEED,
            "n_test": TEST_DAYS,
            "rmse": pair.rmse,
            "smape": pair.smape,
            "prediction_finite_count": finite_pred,
            "target_finite_count": finite_true,
            "first_target_date": dates[0],
            "last_target_date": dates[-1],
            "deterministic_reuse": bool(deterministic_reuse),
            "source_universe_id": "c57f402fbbc701b7",
        }
    )
    for day, truth, pred in zip(dates, y_true, y_pred):
        rows.append(
            {
                "dataset": CANARY_DATASET,
                "target_identity": CANARY_IDENTITY,
                "horizon": horizon,
                "seed": CANARY_SEED,
                "method": method,
                "target_date": day,
                "y_true": float(truth),
                "y_pred": float(pred),
                "deterministic_reuse": bool(deterministic_reuse),
                "source_universe_id": "c57f402fbbc701b7",
            }
        )


def _selection_payload(record, selection: SelectionOutcome, cache_rows: list[dict[str, object]]) -> dict[str, object]:
    sources = [
        {
            "identity": item.identity,
            "distance": item.distance,
            "weight": item.weight,
            "rank": item.rank,
        }
        for item in selection.selected
    ]
    by_horizon = {}
    for horizon in HORIZONS:
        rows = [row for row in cache_rows if int(row["horizon"]) == horizon]
        by_horizon[str(horizon)] = {
            "sources": sources,
            "cache": rows,
        }
    return {
        "dataset": record.dataset,
        "target_identity": record.target_identity,
        "source_universe_id": record.source_universe_id,
        "source_universe": D1_SOURCE_UNIVERSE,
        "status": selection.status,
        "comparison_start": selection.comparison_start,
        "comparison_end": selection.comparison_end,
        "selection_independent_of_horizon": True,
        "horizons": by_horizon,
    }


def _require_d1_universe(selection: SelectionOutcome, target_identity: str) -> None:
    if selection.dataset != "D1":
        return
    if selection.source_universe_id != "c57f402fbbc701b7":
        raise ProtocolError("D1 source_universe_id is not the frozen pool id")
    for item in selection.selected:
        store, sku = item.key
        if store != "1" or sku not in {str(value) for value in range(1, 10)}:
            raise ProtocolError(f"D1 selected {item.identity} outside store 1 items 1-9")
        if item.identity == target_identity:
            raise ProtocolError("D1 selected the target identity")


def _series_sha256(series: pd.DataFrame, features: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    ordered = series.sort_values("date", kind="mergesort")
    for _, row in ordered.iterrows():
        digest.update(pd.Timestamp(row["date"]).strftime("%Y-%m-%d").encode("utf-8"))
        for column in features:
            value = row[column]
            token = "NA" if pd.isna(value) else repr(value)
            digest.update(b"\0")
            digest.update(token.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_source_cache(directory, payload, digest, source_key, weights, scaler, sample_count) -> None:
    arrays = {}
    layout = {}
    for name, tensors in weights.items():
        layout[name] = len(tensors)
        for index, array in enumerate(tensors):
            arrays[f"{name}__{index}"] = np.asarray(array)
    np.savez_compressed(directory / "weights.npz", **arrays)
    (directory / "weights_layout.json").write_text(json.dumps(layout, sort_keys=True), encoding="utf-8")
    (directory / "key.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    scaler_payload = {
        "role": scaler.role,
        "columns": list(scaler.columns),
        "data_min": scaler.data_min.tolist(),
        "data_max": scaler.data_max.tolist(),
        "scale": scaler.scale.tolist(),
        "min": scaler.min.tolist(),
        "fit_dates": list(scaler.fit_dates),
        "sample_count": sample_count,
        "source_key": list(source_key),
    }
    (directory / "scaler.json").write_text(json.dumps(scaler_payload), encoding="utf-8")
    done = {"status": "COMPLETED", "cache_key_sha256": digest}
    (directory / "DONE.json").write_text(json.dumps(done, sort_keys=True), encoding="utf-8")


def _read_source_record(directory: Path, source_key):
    from sita_ablation.scaling import ModelScaler
    from sita_ablation.stage_a import SourceRecord

    layout = json.loads((directory / "weights_layout.json").read_text(encoding="utf-8"))
    stored = np.load(directory / "weights.npz")
    weights = {
        name: [np.asarray(stored[f"{name}__{index}"]) for index in range(int(count))]
        for name, count in layout.items()
    }
    payload = json.loads((directory / "scaler.json").read_text(encoding="utf-8"))
    scaler = ModelScaler(
        role=payload["role"],
        columns=tuple(payload["columns"]),
        data_min=np.asarray(payload["data_min"], dtype=np.float64),
        data_max=np.asarray(payload["data_max"], dtype=np.float64),
        scale=np.asarray(payload["scale"], dtype=np.float64),
        min=np.asarray(payload["min"], dtype=np.float64),
        fit_dates=tuple(payload["fit_dates"]),
    )
    return SourceRecord(
        rank=0,
        source_key=tuple(source_key),
        weights=weights,
        scaler=scaler,
        sample_count=int(payload["sample_count"]),
    )


def _owned_cache(path: Path) -> Path:
    resolved = path.resolve()
    root = SOURCE_MODEL_CACHE.resolve()
    if resolved != root and root not in resolved.parents:
        raise ProtocolError("source model cache escaped its output directory")
    return path


def _remove_tree(path: Path) -> None:
    if SOURCE_MODEL_CACHE.resolve() not in path.resolve().parents:
        raise ProtocolError("refusing to delete a path outside the source model cache")
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_dir():
            child.rmdir()
    path.rmdir()


def _write_blocked(log: _Log, status: str, detail: str) -> None:
    CANARY_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "detail": detail, "canary": "D1/2/6"}
    (CANARY_DIR / "protocol_check.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (CANARY_DIR / "run_log.txt").write_text(log.text(), encoding="utf-8")
    (CANARY_DIR / "DONE.json").write_text(json.dumps({"status": status, "canary": "D1/2/6"}) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv == ["dry-run"]:
            dry_run()
            return 0
        if argv == ["canary"]:
            status = canary()
            return 0 if status == "READY_FOR_FULL_RUN" else 1
        if argv == ["full-run"]:
            from additional_target_robustness.full_run import full_run

            status = full_run()
            return 0 if status == "COMPLETED" else 1
    except ProtocolHold as exc:
        print(exc.token, flush=True)
        return 2
    except ProtocolError as exc:
        print(f"BLOCKED_PROTOCOL_MISMATCH {exc}", flush=True)
        return 1
    raise SystemExit("usage: python -m additional_target_robustness.runner {dry-run|canary|full-run}")


if __name__ == "__main__":
    raise SystemExit(main())
