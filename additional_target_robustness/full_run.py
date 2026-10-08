"""Full additional-target run. Identities come only from the frozen manifest.

A finished cell is skipped only when its key, prediction file, and completion
marker agree. A conflicting finished cell is marked ``HOLD_HASH_MISMATCH`` and
left in place. The canary directory is read, never rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
from additional_target_robustness.horizon import verify_direct_horizon
from additional_target_robustness.manifest import ManifestRecord, load_manifest, verify_source_caches
from additional_target_robustness.paths import CANARY_DIR, FULL_RUN_DIR
from additional_target_robustness.protocol import (
    HOLD_HASH_MISMATCH,
    HORIZONS,
    MANIFEST_SHA256,
    METHODS,
    SEEDS,
    TEST_DAYS,
    ProtocolError,
    ProtocolHold,
    build_evaluation_index,
    calendar_for,
    frozen_protocol_sha256,
)
from additional_target_robustness.runner import (
    _predict_sita,
    _require_d1_universe,
    _series_sha256,
)
from additional_target_robustness.source_selection import SelectionOutcome, load_source_histories, select_top3
from additional_target_robustness.target_data import load_target_series, sales_by_date, split_target_frame

EXPECTED_CONDITIONS = 40 * 3 * 5
EXPECTED_RECORDS = EXPECTED_CONDITIONS * 4


@dataclass
class PreparedTarget:
    record: ManifestRecord
    selection: SelectionOutcome
    target_sha256: str


class _Log:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def full_run(directory: Path | None = None) -> str:
    root = Path(directory) if directory is not None else FULL_RUN_DIR
    log = _Log(root / "run_log.txt")
    log.write("full-run start")
    contract = verify_direct_horizon()
    log.write(f"horizon contract {json.dumps(contract, sort_keys=True)}")
    records = load_manifest()
    if len(records) != 40 or EXPECTED_RECORDS != 2400:
        raise ProtocolHold("HOLD_TARGET_MANIFEST_MISMATCH")
    source_hashes = verify_source_caches()
    protocol_sha = frozen_protocol_sha256()
    prepared = _preflight(records, log, root)
    state = {
        "expected_records": EXPECTED_RECORDS,
        "expected_conditions": EXPECTED_CONDITIONS,
        "completed": 0,
        "skipped_existing": 0,
        "pending": EXPECTED_RECORDS,
        "failed": 0,
        "held": 0,
        "canary_reused": 0,
        "status": "RUNNING",
    }
    _write_json(root / "progress.json", state)
    _write_json(
        root / "run_manifest.json",
        {
            "command": "python -m additional_target_robustness.runner full-run",
            "cwd_note": "repository root",
            "log": str(root / "run_log.txt"),
            "expected_records": EXPECTED_RECORDS,
            "expected_conditions": EXPECTED_CONDITIONS,
            "manifest_sha256": MANIFEST_SHA256,
            "protocol_sha256": protocol_sha,
            "horizon_contract": contract,
        },
    )
    try:
        _train_all(
            root,
            prepared,
            source_hashes=source_hashes,
            protocol_sha=protocol_sha,
            log=log,
            state=state,
        )
    except ProtocolHold:
        state["status"] = "HELD"
        _write_json(root / "progress.json", _with_pending(state))
        raise
    except Exception:
        state["status"] = "FAILED"
        if int(state["failed"]) == 0:
            state["failed"] = 1
        _write_json(root / "progress.json", _with_pending(state))
        log.write("FAILED\n" + traceback.format_exc())
        raise
    state["pending"] = EXPECTED_RECORDS - int(state["completed"]) - int(state["held"])
    if int(state["completed"]) == EXPECTED_RECORDS and int(state["failed"]) == 0 and int(state["held"]) == 0:
        _write_completeness(root, log)
        state["status"] = "COMPLETED"
        state["pending"] = 0
        _write_json(root / "progress.json", state)
        log.write("COMPLETED")
        return "COMPLETED"
    state["status"] = "INCOMPLETE"
    _write_json(root / "progress.json", state)
    log.write(f"INCOMPLETE completed={state['completed']} pending={state['pending']}")
    return "INCOMPLETE"


def _preflight(records: tuple[ManifestRecord, ...], log: _Log, root: Path) -> list[PreparedTarget]:
    prepared: list[PreparedTarget] = []
    payload = []
    for record in records:
        log.write(f"preflight {record.dataset} {record.target_identity}")
        target = load_target_series(record.dataset, record.target_identity)
        calendar = calendar_for(record.dataset)
        split_target_frame(target, calendar)
        histories = load_source_histories(
            record.dataset,
            target_identity=record.target_identity,
            target_frame=target,
        )
        selection = select_top3(
            record.dataset,
            record.target_identity,
            target,
            histories,
            source_universe_id=record.source_universe_id,
        )
        if selection.status != "COMPLETED" or len(selection.selected) != 3:
            raise ProtocolError(
                f"{record.dataset} {record.target_identity} selection status is {selection.status}"
            )
        _require_d1_universe(selection, record.target_identity)
        if any(item.identity == record.target_identity for item in selection.selected):
            raise ProtocolError(f"{record.dataset} selected itself")
        for horizon in HORIZONS:
            index = build_evaluation_index(record.dataset, horizon)
            if len(index.rows) != TEST_DAYS:
                raise ProtocolError(f"{record.dataset} h={horizon} does not have 180 evaluation dates")
            if any(row.label_date != row.forecast_origin + pd.Timedelta(days=horizon) for row in index.rows):
                raise ProtocolHold("BLOCKED_HORIZON_MISMATCH")
        item = PreparedTarget(
            record=record,
            selection=selection,
            target_sha256=_series_sha256(target, calendar.model_features),
        )
        prepared.append(item)
        payload.append(
            {
                "dataset": record.dataset,
                "target_identity": record.target_identity,
                "source_universe_id": record.source_universe_id,
                "target_sha256": item.target_sha256,
                "selection": _selection_rows(selection),
            }
        )
    _write_json(root / "preflight.json", {"targets": payload, "target_count": len(payload)})
    log.write(f"preflight complete targets={len(prepared)}")
    return prepared


def _train_all(root, prepared, *, source_hashes, protocol_sha, log: _Log, state: dict[str, int]) -> None:
    for item in prepared:
        record = item.record
        log.write(f"load {record.dataset} {record.target_identity}")
        target = load_target_series(record.dataset, record.target_identity)
        calendar = calendar_for(record.dataset)
        train, validation, _test = split_target_frame(target, calendar)
        histories = load_source_histories(
            record.dataset,
            target_identity=record.target_identity,
            target_frame=target,
        )
        selection = select_top3(
            record.dataset,
            record.target_identity,
            target,
            histories,
            source_universe_id=record.source_universe_id,
        )
        if _selection_rows(selection) != _selection_rows(item.selection):
            raise ProtocolError(f"{record.dataset} {record.target_identity} selection changed after preflight")
        sales = sales_by_date(target)
        train_sales = partition_sales(target, calendar.train_start, calendar.train_end)
        validation_sales = partition_sales(target, calendar.val_start, calendar.val_end)
        target_sha = _series_sha256(target, calendar.model_features)
        if target_sha != item.target_sha256:
            raise ProtocolHold(HOLD_HASH_MISMATCH)
        for horizon in HORIZONS:
            index = build_evaluation_index(record.dataset, horizon)
            dates = index.label_date_text()
            y_true = np.asarray([sales[day] for day in dates], dtype=np.float64)
            _write_seasonal(
                root,
                item,
                index=index,
                index_dates=dates,
                y_true=y_true,
                sales=sales,
                horizon=horizon,
                source_hashes=source_hashes,
                protocol_sha=protocol_sha,
                target_sha=target_sha,
                log=log,
                state=state,
            )
            for seed in SEEDS:
                for method in ("SITA", "PatchTST", "TimesNet"):
                    _write_trained_method(
                        root,
                        item,
                        target=target,
                        train=train,
                        validation=validation,
                        histories=histories,
                        index=index,
                        dates=dates,
                        y_true=y_true,
                        train_sales=train_sales,
                        validation_sales=validation_sales,
                        sales=sales,
                        horizon=horizon,
                        seed=seed,
                        method=method,
                        source_hashes=source_hashes,
                        protocol_sha=protocol_sha,
                        target_sha=target_sha,
                        log=log,
                        state=state,
                    )


def _write_seasonal(root, item, *, index, index_dates, y_true, sales, horizon, source_hashes, protocol_sha, target_sha, log, state) -> None:
    record = item.record
    pending = []
    existing_prediction = None
    for seed in SEEDS:
        key = _cell_key(
            record,
            item.selection,
            horizon=horizon,
            seed=seed,
            method="SeasonalNaive",
            source_hashes=source_hashes,
            protocol_sha=protocol_sha,
            target_sha=target_sha,
            dates=index_dates,
        )
        directory = _cell_dir(root, record.dataset, record.target_identity, horizon, seed, "SeasonalNaive")
        decision = _inspect_cell(root, directory, key, index_dates, y_true)
        if decision == "COMPLETE":
            state["completed"] += 1
            state["skipped_existing"] += 1
            stored = pd.read_parquet(directory / "predictions.parquet")["y_pred"].to_numpy(dtype=np.float64)
            if existing_prediction is None:
                existing_prediction = stored
            elif not np.array_equal(existing_prediction, stored):
                _mark_hold(directory, key)
                state["held"] += 1
                _write_json(root / "progress.json", _with_pending(state))
                raise ProtocolHold(HOLD_HASH_MISMATCH)
            continue
        if decision == "HOLD":
            _mark_hold(directory, key)
            state["held"] += 1
            _write_json(root / "progress.json", _with_pending(state))
            raise ProtocolHold(HOLD_HASH_MISMATCH)
        pending.append((seed, directory, key))
    if not pending:
        _write_json(root / "progress.json", _with_pending(state))
        return
    predicted = predict_seasonal(record.dataset, record.target_identity, horizon, index, sales)
    if predicted.shape != (TEST_DAYS,):
        raise ProtocolError("seasonal naive did not return 180 predictions")
    if existing_prediction is not None and not np.array_equal(existing_prediction, predicted):
        raise ProtocolHold(HOLD_HASH_MISMATCH)
    copies = materialize_seasonal_seeds(predicted)
    if any(not np.array_equal(predicted, copy["y_pred"]) for copy in copies):
        raise ProtocolError("seasonal seed copies are not identical")
    canary = _canary_vector(record, "SeasonalNaive", horizon, 42, index_dates, y_true, item.selection)
    origin = "canary_D1_2_6" if canary is not None and np.array_equal(canary, predicted) else "computed"
    by_seed = {int(copy["seed"]): copy for copy in copies}
    for seed, directory, key in pending:
        _commit_cell(
            directory,
            key,
            dates=index_dates,
            y_true=y_true,
            y_pred=np.asarray(by_seed[seed]["y_pred"], dtype=np.float64),
            deterministic_reuse=True,
            prediction_source=origin,
            dataset=record.dataset,
            identity=record.target_identity,
            horizon=horizon,
            seed=seed,
            method="SeasonalNaive",
            source_universe_id=record.source_universe_id,
        )
        state["completed"] += 1
        if origin == "canary_D1_2_6" and int(seed) == 42:
            state["canary_reused"] += 1
        log.write(f"wrote {record.dataset} {record.target_identity} h={horizon} seed={seed} SeasonalNaive {origin}")
    _write_json(root / "progress.json", _with_pending(state))


def _write_trained_method(root, item, **kwargs) -> None:
    record = item.record
    method = kwargs["method"]
    horizon = kwargs["horizon"]
    seed = kwargs["seed"]
    dates = kwargs["dates"]
    y_true = kwargs["y_true"]
    key = _cell_key(
        record,
        item.selection,
        horizon=horizon,
        seed=seed,
        method=method,
        source_hashes=kwargs["source_hashes"],
        protocol_sha=kwargs["protocol_sha"],
        target_sha=kwargs["target_sha"],
        dates=dates,
    )
    directory = _cell_dir(root, record.dataset, record.target_identity, horizon, seed, method)
    decision = _inspect_cell(root, directory, key, dates, y_true)
    state = kwargs["state"]
    log = kwargs["log"]
    if decision == "COMPLETE":
        state["completed"] += 1
        state["skipped_existing"] += 1
        _write_json(root / "progress.json", _with_pending(state))
        return
    if decision == "HOLD":
        _mark_hold(directory, key)
        state["held"] += 1
        _write_json(root / "progress.json", _with_pending(state))
        raise ProtocolHold(HOLD_HASH_MISMATCH)
    reused = _canary_vector(record, method, horizon, seed, dates, y_true, item.selection)
    if reused is not None:
        y_pred = reused
        origin = "canary_D1_2_6"
    else:
        log.write(f"train {record.dataset} {record.target_identity} h={horizon} seed={seed} {method}")
        y_pred = _fit_method(item, **kwargs)
        origin = "trained"
    if y_pred.shape != (TEST_DAYS,) or not np.isfinite(y_pred).all():
        state["failed"] += 1
        _write_json(root / "progress.json", _with_pending(state))
        raise ProtocolError(f"{method} returned a non-finite or short forecast")
    _commit_cell(
        directory,
        key,
        dates=dates,
        y_true=y_true,
        y_pred=y_pred,
        deterministic_reuse=False,
        prediction_source=origin,
        dataset=record.dataset,
        identity=record.target_identity,
        horizon=horizon,
        seed=seed,
        method=method,
        source_universe_id=record.source_universe_id,
    )
    state["completed"] += 1
    if origin == "canary_D1_2_6":
        state["canary_reused"] += 1
    log.write(f"wrote {record.dataset} {record.target_identity} h={horizon} seed={seed} {method} {origin}")
    _write_json(root / "progress.json", _with_pending(state))


def _fit_method(item, **kwargs) -> np.ndarray:
    method = kwargs["method"]
    horizon = int(kwargs["horizon"])
    seed = int(kwargs["seed"])
    index = kwargs["index"]
    if method == "SITA":
        predictions, _usage = _predict_sita(
            kwargs["target"],
            kwargs["train"],
            kwargs["validation"],
            item.selection,
            index,
            horizon=horizon,
            seed=seed,
            dataset=item.record.dataset,
            identity=item.record.target_identity,
            histories=kwargs["histories"],
        )
        return np.asarray(predictions, dtype=np.float64)
    if method == "PatchTST":
        return fit_and_predict_patchtst(
            kwargs["train_sales"],
            kwargs["validation_sales"],
            index,
            kwargs["sales"],
            horizon=horizon,
            seed=seed,
        )
    if method == "TimesNet":
        return fit_and_predict_timesnet(
            kwargs["train_sales"],
            kwargs["validation_sales"],
            index,
            kwargs["sales"],
            horizon=horizon,
            seed=seed,
        )
    raise ProtocolError(f"unknown method {method}")


def _canary_vector(record, method, horizon, seed, dates, y_true, selection: SelectionOutcome):
    if record.dataset != "D1" or record.target_identity != "2/6" or int(seed) != 42:
        return None
    prediction_path = CANARY_DIR / "predictions.parquet"
    protocol_path = CANARY_DIR / "protocol_check.json"
    selection_path = CANARY_DIR / "source_selection.json"
    if not prediction_path.is_file() or not protocol_path.is_file() or not selection_path.is_file():
        return None
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "READY_FOR_FULL_RUN" or protocol.get("manifest_sha256") != MANIFEST_SHA256:
        return None
    if protocol.get("target_identity") != "2/6" or protocol.get("dataset") != "D1":
        return None
    if method == "SITA":
        stored = json.loads(selection_path.read_text(encoding="utf-8"))
        stored_ids = [row["identity"] for row in stored["horizons"][str(horizon)]["sources"]]
        if stored_ids != [item.identity for item in selection.selected]:
            return None
    frame = pd.read_parquet(prediction_path)
    part = frame[
        (frame["dataset"] == "D1")
        & (frame["target_identity"] == "2/6")
        & (frame["method"] == method)
        & (frame["horizon"] == int(horizon))
        & (frame["seed"] == int(seed))
    ].sort_values("target_date")
    if len(part) != TEST_DAYS:
        return None
    found_dates = tuple(pd.Timestamp(day).strftime("%Y-%m-%d") for day in part["target_date"])
    if found_dates != tuple(dates):
        return None
    found_true = part["y_true"].to_numpy(dtype=np.float64)
    if not np.allclose(found_true, y_true, rtol=0.0, atol=1e-8):
        return None
    return part["y_pred"].to_numpy(dtype=np.float64)


def _cell_key(record, selection, *, horizon, seed, method, source_hashes, protocol_sha, target_sha, dates) -> dict[str, object]:
    return {
        "dataset": record.dataset,
        "target_identity": record.target_identity,
        "horizon": int(horizon),
        "seed": int(seed),
        "method": method,
        "manifest_sha256": MANIFEST_SHA256,
        "protocol_sha256": protocol_sha,
        "source_cache_sha256": source_hashes,
        "target_sha256": target_sha,
        "evaluation_dates_sha256": hashlib.sha256("\n".join(dates).encode("utf-8")).hexdigest(),
        "selection": _selection_rows(selection),
        "source_universe_id": record.source_universe_id,
        "n_test": TEST_DAYS,
        "direct_horizon": "y[forecast_origin + horizon]",
    }


def _inspect_cell(root: Path, directory: Path, key: dict[str, object], dates, y_true) -> str:
    done_path = directory / "DONE.json"
    key_path = directory / "key.json"
    prediction_path = directory / "predictions.parquet"
    if not done_path.is_file():
        if directory.exists():
            _remove_tree(directory, root)
        return "MISSING"
    if not key_path.is_file() or not prediction_path.is_file():
        return "HOLD"
    stored = json.loads(key_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    key_sha = _digest(key)
    if _digest(stored) != key_sha or done.get("key_sha256") != key_sha or done.get("status") != "COMPLETED":
        return "HOLD"
    if int(done.get("n_test", 0)) != TEST_DAYS:
        return "HOLD"
    frame = pd.read_parquet(prediction_path)
    if len(frame) != TEST_DAYS:
        return "HOLD"
    found_dates = tuple(pd.Timestamp(day).strftime("%Y-%m-%d") for day in frame["target_date"])
    if found_dates != tuple(dates):
        return "HOLD"
    if not np.allclose(frame["y_true"].to_numpy(dtype=np.float64), y_true, rtol=0.0, atol=1e-8):
        return "HOLD"
    return "COMPLETE"


def _commit_cell(directory, key, *, dates, y_true, y_pred, deterministic_reuse, prediction_source, dataset, identity, horizon, seed, method, source_universe_id) -> None:
    from additional_target_robustness.paths import ensure_ablation_path

    ensure_ablation_path()
    from sita_ablation.metrics import evaluate_original_sales

    if len(dates) != TEST_DAYS or len(y_true) != TEST_DAYS or len(y_pred) != TEST_DAYS:
        raise ProtocolError(f"{method} n_test is not 180")
    pair = evaluate_original_sales(np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64))
    partial = directory.with_name(directory.name + ".partial")
    if partial.exists():
        _remove_tree(partial, directory.parent)
    partial.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "target_identity": identity,
            "horizon": int(horizon),
            "seed": int(seed),
            "method": method,
            "target_date": list(dates),
            "y_true": np.asarray(y_true, dtype=np.float64),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
            "deterministic_reuse": bool(deterministic_reuse),
            "prediction_source": prediction_source,
            "source_universe_id": source_universe_id,
        }
    )
    frame.to_parquet(partial / "predictions.parquet", index=False)
    metrics = {
        "dataset": dataset,
        "target_identity": identity,
        "horizon": int(horizon),
        "seed": int(seed),
        "method": method,
        "n_test": TEST_DAYS,
        "rmse": pair.rmse,
        "smape": pair.smape,
        "prediction_finite_count": int(np.isfinite(y_pred).sum()),
        "target_finite_count": int(np.isfinite(y_true).sum()),
        "first_target_date": dates[0],
        "last_target_date": dates[-1],
        "deterministic_reuse": bool(deterministic_reuse),
        "prediction_source": prediction_source,
        "source_universe_id": source_universe_id,
    }
    _write_json(partial / "metrics.json", metrics)
    _write_json(partial / "key.json", key)
    _write_json(
        partial / "DONE.json",
        {
            "status": "COMPLETED",
            "key_sha256": _digest(key),
            "n_test": TEST_DAYS,
            "prediction_source": prediction_source,
            "deterministic_reuse": bool(deterministic_reuse),
        },
    )
    partial.rename(directory)


def _mark_hold(directory: Path, key: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    hold_path = directory / "HOLD.json"
    if hold_path.exists():
        return
    _write_json(
        hold_path,
        {
            "status": HOLD_HASH_MISMATCH,
            "expected_key_sha256": _digest(key),
        },
    )


def _write_completeness(root: Path, log: _Log) -> None:
    rows = []
    seen = set()
    for metrics_path in sorted((root / "cells").rglob("metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        identity = (
            metrics["dataset"],
            metrics["target_identity"],
            int(metrics["horizon"]),
            int(metrics["seed"]),
            metrics["method"],
        )
        if identity in seen:
            raise ProtocolError(f"duplicate cell {identity}")
        seen.add(identity)
        if int(metrics["n_test"]) != TEST_DAYS:
            raise ProtocolError(f"{identity} n_test is not 180")
        prediction_path = metrics_path.with_name("predictions.parquet")
        frame = pd.read_parquet(prediction_path)
        if len(frame) != TEST_DAYS:
            raise ProtocolError(f"{identity} predictions are not 180 rows")
        rows.append(metrics)
    if len(rows) != EXPECTED_RECORDS:
        raise ProtocolError(f"completeness saw {len(rows)} records")
    table = pd.DataFrame(rows)
    grouped = table.groupby(["dataset", "target_identity", "horizon", "seed"], sort=False)
    for key, part in grouped:
        if len(part) != 4 or set(part["method"]) != set(METHODS):
            raise ProtocolError(f"condition {key} does not have four methods")
        dates = []
        truths = []
        for _, row in part.iterrows():
            frame = pd.read_parquet(
                _cell_dir(root, row["dataset"], row["target_identity"], int(row["horizon"]), int(row["seed"]), row["method"])
                / "predictions.parquet"
            )
            dates.append(tuple(pd.Timestamp(day).strftime("%Y-%m-%d") for day in frame["target_date"]))
            truths.append(frame["y_true"].to_numpy(dtype=np.float64))
        if any(item != dates[0] for item in dates) or any(not np.array_equal(item, truths[0]) for item in truths):
            raise ProtocolError(f"condition {key} dates or labels differ across methods")
    expected = {
        (record.dataset, record.target_identity, horizon, seed, method)
        for record in load_manifest()
        for horizon in HORIZONS
        for seed in SEEDS
        for method in METHODS
    }
    if seen != expected:
        raise ProtocolError("completed identities do not match the frozen grid")
    table.sort_values(["dataset", "target_identity", "horizon", "seed", "method"]).to_csv(root / "metrics.csv", index=False)
    _write_json(
        root / "completeness.json",
        {
            "status": "COMPLETE",
            "records": EXPECTED_RECORDS,
            "conditions": EXPECTED_CONDITIONS,
            "n_test": TEST_DAYS,
            "duplicate_identities": 0,
            "missing_identities": 0,
        },
    )
    log.write(f"completeness records={EXPECTED_RECORDS}")


def _selection_rows(selection: SelectionOutcome) -> list[dict[str, object]]:
    return [
        {
            "identity": item.identity,
            "distance": item.distance,
            "weight": item.weight,
            "rank": item.rank,
        }
        for item in selection.selected
    ]


def _cell_dir(root: Path, dataset: str, identity: str, horizon: int, seed: int, method: str) -> Path:
    return root / "cells" / dataset / identity.replace("/", "__") / f"h{int(horizon)}" / f"s{int(seed)}" / method


def _digest(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _with_pending(state: dict[str, int]) -> dict[str, int]:
    updated = dict(state)
    updated["pending"] = EXPECTED_RECORDS - int(state["completed"]) - int(state["held"])
    return updated


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _remove_tree(path: Path, root: Path) -> None:
    resolved = path.resolve()
    base = root.resolve()
    if resolved != base and base not in resolved.parents:
        raise ProtocolError("refusing to delete a path outside the full-run directory")
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()
