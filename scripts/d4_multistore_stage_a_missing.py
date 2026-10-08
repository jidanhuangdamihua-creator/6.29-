#!/usr/bin/env python3
"""Train CACHE_MISS Stage A source CNNs for the D4 60-target robustness freeze.

This entry reuses additional_target_robustness.runner._load_or_train_source.
It does not run target adaptation, SITA prediction, PatchTST, TimesNet, or
Seasonal Naive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from additional_target_robustness.paths import SOURCE_MODEL_CACHE, ensure_ablation_path
from additional_target_robustness.protocol import ProtocolHold, calendar_for
from additional_target_robustness.runner import (
    _load_or_train_source,
    _one_source,
    _read_source_record,
    _series_sha256,
)
from additional_target_robustness.source_selection import load_source_histories

ensure_ablation_path()
from sita_ablation.data import split_source_frame
from sita_ablation.scaling import fit_source_model_scaler, transform_model_frame
from sita_ablation.windows import WINDOW_SIZE, build_partition_windows

OUT_DIR = REPO_ROOT / "outputs" / "d4_multistore_60target_robustness_20261004"
REQ_PATH = OUT_DIR / "stage_a_requirements.csv"
USAGE_PATH = OUT_DIR / "source_usage_summary.csv"
CANARY_DIR = OUT_DIR / "stage_a_canary"
CANARY_STORES = (155, 274, 576, 139, 161)
TRAIN_STORES = CANARY_STORES
EXPECTED_MISS = 1125
EXPECTED_REQUIRED = 1335
EXPECTED_ORIGINAL_HIT = 210
SOURCE_START = "2024-07-19"
SOURCE_END = "2025-01-14"
SOURCE_ROWS = 180
FORBIDDEN_TRAIN_STORE = 166


class StageAStop(RuntimeError):
    def __init__(self, token: str, detail: str = "") -> None:
        super().__init__(token if not detail else f"{token}: {detail}")
        self.token = token
        self.detail = detail


class _Log:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_requirements() -> pd.DataFrame:
    if not REQ_PATH.is_file():
        raise StageAStop("HOLD_STAGE_A_REQUIREMENT_COUNT_MISMATCH", "missing stage_a_requirements.csv")
    frame = pd.read_csv(REQ_PATH)
    frame["store_id"] = frame["store_id"].astype(int)
    frame["horizon"] = frame["horizon"].astype(int)
    frame["seed"] = frame["seed"].astype(int)
    return frame


def _assert_requirement_counts(req: pd.DataFrame) -> pd.DataFrame:
    n_miss = int((req["cache_status"] == "CACHE_MISS").sum())
    if n_miss != EXPECTED_MISS:
        raise StageAStop(
            "HOLD_STAGE_A_REQUIREMENT_COUNT_MISMATCH",
            f"n_cache_miss={n_miss} expected={EXPECTED_MISS}",
        )
    miss = req.loc[req["cache_status"] == "CACHE_MISS"].copy()
    if (miss["store_id"] == FORBIDDEN_TRAIN_STORE).any():
        raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", "store 166 appears in CACHE_MISS")
    identities = set(miss["source_identity"].astype(str))
    extra_stores = set(int(item.split("/")[0]) for item in identities) - set(TRAIN_STORES)
    if extra_stores:
        raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", f"unexpected miss stores {sorted(extra_stores)}")
    return miss.reset_index(drop=True)


def _canary_sources(miss: pd.DataFrame) -> pd.DataFrame:
    usage = pd.read_csv(USAGE_PATH)
    usage["store_id"] = usage["store_id"].astype(int)
    miss_identities = set(miss["source_identity"].astype(str))
    rows = []
    for store in CANARY_STORES:
        store_usage = usage.loc[
            (usage["store_id"] == store) & (usage["source_identity"].isin(miss_identities))
        ].copy()
        if store_usage.empty:
            raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", f"no CACHE_MISS source for store {store}")
        store_usage = store_usage.sort_values(
            ["selected_by_n_targets", "source_identity"],
            ascending=[False, True],
            kind="mergesort",
        )
        chosen = store_usage.iloc[0]
        identity = str(chosen["source_identity"])
        key_rows = miss.loc[
            (miss["store_id"] == store)
            & (miss["source_identity"] == identity)
            & (miss["horizon"] == 1)
            & (miss["seed"] == 42)
        ]
        if len(key_rows) != 1:
            raise StageAStop(
                "HOLD_STAGE_A_REQUIREMENT_COUNT_MISMATCH",
                f"canary key not unique for {identity} h=1 seed=42",
            )
        row = key_rows.iloc[0].to_dict()
        row["selected_by_n_targets"] = int(chosen["selected_by_n_targets"])
        rows.append(row)
    return pd.DataFrame(rows)


def _load_store_histories(stores: tuple[int, ...], log: _Log) -> dict[int, pd.DataFrame]:
    calendar = calendar_for("D4")
    loaded: dict[int, pd.DataFrame] = {}
    for store in stores:
        log.write(f"loading sealed source history for store {store}")
        dummy = pd.DataFrame(columns=["date", *calendar.key_fields])
        frame = load_source_histories("D4", target_identity=f"{store}/0", target_frame=dummy)
        loaded[store] = frame
    return loaded


def _assert_source_protocol(series: pd.DataFrame, features: tuple[str, ...], identity: str) -> None:
    dates = pd.to_datetime(series["date"]).dt.normalize()
    if len(series) != SOURCE_ROWS:
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{identity} n_rows={len(series)}")
    if dates.nunique() != SOURCE_ROWS or bool(dates.duplicated().any()):
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{identity} dates are not 180 unique days")
    first = dates.min().strftime("%Y-%m-%d")
    last = dates.max().strftime("%Y-%m-%d")
    if first != SOURCE_START or last != SOURCE_END:
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{identity} span {first}..{last}")
    expected = pd.date_range(SOURCE_START, SOURCE_END, freq="D")
    if not dates.reset_index(drop=True).equals(pd.Series(expected)):
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{identity} is not 180 consecutive calendar days")
    if (dates > pd.Timestamp(SOURCE_END)).any():
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{identity} has a row after source cutoff")
    sales = pd.to_numeric(series["sales"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(sales).all():
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{identity} sales is not finite")
    for column in features:
        values = pd.to_numeric(series[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{identity} feature {column} is not finite")


def _expected_train_windows(horizon: int) -> int:
    n_train = max(1, int(SOURCE_ROWS * 0.8))
    return n_train - WINDOW_SIZE - int(horizon) + 1


def _audit_windows(series: pd.DataFrame, *, source_key: tuple[str, ...], features: tuple[str, ...], horizon: int) -> dict:
    split = split_source_frame(series, features)
    scaler = fit_source_model_scaler(split.train, features)
    transformed = transform_model_frame(scaler, split.train)
    windows = build_partition_windows(
        transformed,
        key_fields=("store_id", "product_id"),
        key=source_key,
        feature_columns=features,
        horizon=int(horizon),
        partition="train",
    )
    n_windows = int(windows.y_scaled.shape[0])
    expected = _expected_train_windows(horizon)
    if n_windows != expected:
        raise StageAStop(
            "BLOCKED_SOURCE_DATA_PROTOCOL",
            f"{'/'.join(source_key)} h={horizon} n_train_windows={n_windows} expected={expected}",
        )
    train_dates = set(pd.to_datetime(split.train["date"]).dt.normalize())
    val_dates = set(pd.to_datetime(split.validation["date"]).dt.normalize())
    test_dates = set(pd.to_datetime(split.test["date"]).dt.normalize())
    sales_scaled = transformed["sales"].to_numpy(dtype=np.float32)
    dates = pd.to_datetime(transformed["date"]).dt.normalize().to_list()
    by_date = {day: index for index, day in enumerate(dates)}
    for index in range(n_windows):
        origin = pd.Timestamp(windows.input_end[index]).normalize()
        label = pd.Timestamp(windows.label_date[index]).normalize()
        if origin not in by_date:
            raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", "window origin is outside the train split")
        origin_idx = by_date[origin]
        label_idx = origin_idx + int(horizon)
        if label_idx >= len(dates):
            raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", "direct-h label index overflow")
        if dates[label_idx] != label:
            raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", "label date is not y[o+h]")
        if abs(float(windows.y_scaled[index]) - float(sales_scaled[label_idx])) > 1e-6:
            raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", "direct-h scaled label mismatch")
        if origin in val_dates or origin in test_dates or label in val_dates or label in test_dates:
            raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", "window crossed the source split")
        if origin not in train_dates or label not in train_dates:
            raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", "window left the source train split")
    return {
        "n_train_windows": n_windows,
        "n_train_days": int(len(split.train)),
        "source_start_date": pd.Timestamp(series["date"].min()).strftime("%Y-%m-%d"),
        "source_end_date": pd.Timestamp(series["date"].max()).strftime("%Y-%m-%d"),
        "n_source_rows": int(len(series)),
    }


def _inspect_cache(row: pd.Series, *, load_weights: bool = True) -> str:
    directory = SOURCE_MODEL_CACHE / str(row["cache_key"])
    done_path = directory / "DONE.json"
    key_path = directory / "key.json"
    weights_path = directory / "weights.npz"
    if not directory.exists():
        return "CACHE_MISS"
    if not (done_path.is_file() and key_path.is_file() and weights_path.is_file()):
        return "CACHE_MISS"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    expected_payload = {
        "dataset": str(row["dataset"]),
        "source_identity": str(row["source_identity"]),
        "horizon": int(row["horizon"]),
        "seed": int(row["seed"]),
        "architecture": str(row["architecture"]),
        "source_data_sha256": str(row["source_data_hash"]),
        "protocol_sha256": str(row["protocol_hash"]),
    }
    digest = _payload_digest(expected_payload)
    if str(row["cache_key"]) != digest:
        return "CACHE_HASH_MISMATCH"
    if done.get("cache_key_sha256") != digest or done.get("status") != "COMPLETED":
        return "CACHE_HASH_MISMATCH"
    for field, value in expected_payload.items():
        if key.get(field) != value:
            return "CACHE_HASH_MISMATCH"
    if load_weights:
        try:
            record = _read_source_record(directory, tuple(str(row["source_identity"]).split("/")))
            if not record.weights:
                return "LOAD_FAILURE"
        except Exception:
            return "LOAD_FAILURE"
    return "CACHE_HIT"


def _snapshot_store166(req: pd.DataFrame) -> dict:
    hit = req.loc[(req["store_id"] == FORBIDDEN_TRAIN_STORE) & (req["cache_status"] == "CACHE_HIT")]
    items = []
    for _, row in hit.iterrows():
        directory = SOURCE_MODEL_CACHE / str(row["cache_key"])
        weights = directory / "weights.npz"
        done = directory / "DONE.json"
        items.append(
            {
                "cache_key": str(row["cache_key"]),
                "done_sha256": _file_sha256(done) if done.is_file() else None,
                "weights_sha256": _file_sha256(weights) if weights.is_file() else None,
            }
        )
    return {
        "n_keys": int(len(items)),
        "items": items,
    }


def _assert_store166_unchanged(before: dict, after: dict) -> None:
    if before["n_keys"] != 210 or after["n_keys"] != 210:
        raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", "store 166 cache key count changed")
    before_map = {item["cache_key"]: item for item in before["items"]}
    after_map = {item["cache_key"]: item for item in after["items"]}
    if set(before_map) != set(after_map):
        raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", "store 166 cache key set changed")
    for key, item in before_map.items():
        other = after_map[key]
        if item["done_sha256"] != other["done_sha256"] or item["weights_sha256"] != other["weights_sha256"]:
            raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", f"store 166 cache mutated: {key}")


_TF_RESETS = {"n": 0}
RECYCLE_AFTER = 40


def _reset_tf(*, collect: bool = False) -> None:
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    _TF_RESETS["n"] += 1
    if collect or _TF_RESETS["n"] % 20 == 0:
        import gc

        gc.collect()


def _is_retryable_tf(exc: BaseException) -> bool:
    text = str(exc)
    name = type(exc).__name__
    if "AssignSubVariableOp" in text or "shapes must be equal" in text:
        return True
    return name in {"InvalidArgumentError", "OpError", "FailedPreconditionError"}


def _in_child_process() -> bool:
    return os.environ.get("D4_STAGE_A_TRAIN_CHILD") == "1"


def _spawn_train_one(cache_key: str, log: _Log) -> None:
    env = os.environ.copy()
    env["D4_STAGE_A_TRAIN_CHILD"] = "1"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        "train-one",
        "--cache-key",
        cache_key,
    ]
    log.write(f"retrying {cache_key} in an isolated TensorFlow process")
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode != 0:
        raise StageAStop("BLOCKED_STAGE_A_TRAINING", f"isolated retry failed for {cache_key}")


def _progress_path() -> Path:
    return OUT_DIR / "stage_a_training_progress.csv"


def _failures_path() -> Path:
    return OUT_DIR / "stage_a_training_failures.csv"


PROGRESS_COLUMNS = [
    "key",
    "store",
    "source",
    "horizon",
    "seed",
    "original_cache_status",
    "final_status",
    "checkpoint",
    "error",
    "training_loss_final",
    "n_train_windows",
    "checkpoint_sha256",
    "trained_in_canary",
    "trained_at",
    "source_data_hash",
    "protocol_hash",
    "source_start_date",
    "source_end_date",
    "n_source_rows",
    "architecture",
    "dataset",
    "training_status",
]


def _load_progress() -> pd.DataFrame:
    path = _progress_path()
    if not path.is_file():
        return pd.DataFrame(columns=PROGRESS_COLUMNS)
    frame = pd.read_csv(path)
    for column in PROGRESS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[PROGRESS_COLUMNS]


def _progress_row(row: dict) -> dict:
    return {column: row.get(column, "") for column in PROGRESS_COLUMNS}


def _upsert_progress(progress: pd.DataFrame, row: dict) -> pd.DataFrame:
    addition = pd.DataFrame([_progress_row(row)])
    frame = progress.copy()
    if not frame.empty:
        frame = frame.loc[frame["key"].astype(str) != str(row["key"])]
    out = pd.concat([frame, addition], ignore_index=True)
    _atomic_csv(_progress_path(), out)
    failures = out.loc[
        out["final_status"].isin(
            ["FAILED", "CACHE_HASH_MISMATCH", "LOAD_FAILURE", "BLOCKED_SOURCE_DATA_PROTOCOL"]
        )
    ].copy()
    _atomic_csv(_failures_path(), failures)
    return out


def _prepare_key(
    row: pd.Series,
    histories: dict[int, pd.DataFrame],
    allowed_identities: set[str],
    *,
    strict_windows: bool = False,
) -> tuple[pd.DataFrame, tuple[str, ...], dict]:
    identity = str(row["source_identity"])
    if identity not in allowed_identities:
        raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", identity)
    store = int(row["store_id"])
    if store == FORBIDDEN_TRAIN_STORE:
        raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", "attempted to train store 166")
    calendar = calendar_for("D4")
    source_key = tuple(identity.split("/"))
    series = _one_source(histories[store], source_key, calendar.key_fields)
    _assert_source_protocol(series, calendar.model_features, identity)
    computed = _series_sha256(series, calendar.model_features)
    if computed != str(row["source_data_hash"]):
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{identity} source_data_hash drifted")
    if strict_windows:
        audit = _audit_windows(
            series,
            source_key=source_key,
            features=calendar.model_features,
            horizon=int(row["horizon"]),
        )
    else:
        n_train = max(1, int(SOURCE_ROWS * 0.8))
        audit = {
            "n_train_windows": n_train - WINDOW_SIZE - int(row["horizon"]) + 1,
            "n_train_days": n_train,
            "source_start_date": SOURCE_START,
            "source_end_date": SOURCE_END,
            "n_source_rows": SOURCE_ROWS,
        }
    return series, source_key, audit


def _train_one(
    row: pd.Series,
    histories: dict[int, pd.DataFrame],
    allowed_identities: set[str],
    *,
    trained_in_canary: bool,
    trained_166: list[int],
    log: _Log,
    strict_windows: bool = False,
) -> dict:
    store = int(row["store_id"])
    if store == FORBIDDEN_TRAIN_STORE:
        trained_166[0] += 1
        raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", "store 166 train attempt")
    inspect = _inspect_cache(row, load_weights=False)
    if inspect == "CACHE_HIT":
        directory = SOURCE_MODEL_CACHE / str(row["cache_key"])
        return {
            "key": str(row["cache_key"]),
            "store": store,
            "source": str(row["source_identity"]),
            "horizon": int(row["horizon"]),
            "seed": int(row["seed"]),
            "original_cache_status": str(row["cache_status"]),
            "final_status": "CACHE_HIT_AFTER_RESUME",
            "checkpoint": str(directory / "weights.npz"),
            "error": "",
            "training_loss_final": "",
            "n_train_windows": "",
            "checkpoint_sha256": _file_sha256(directory / "weights.npz"),
            "trained_in_canary": bool(trained_in_canary),
            "trained_at": _utc_now(),
            "source_data_hash": str(row["source_data_hash"]),
            "protocol_hash": str(row["protocol_hash"]),
        }
    if inspect == "CACHE_HASH_MISMATCH":
        return {
            "key": str(row["cache_key"]),
            "store": store,
            "source": str(row["source_identity"]),
            "horizon": int(row["horizon"]),
            "seed": int(row["seed"]),
            "original_cache_status": str(row["cache_status"]),
            "final_status": "CACHE_HASH_MISMATCH",
            "checkpoint": str(SOURCE_MODEL_CACHE / str(row["cache_key"])),
            "error": "CACHE_HASH_MISMATCH",
            "training_loss_final": "",
            "n_train_windows": "",
            "checkpoint_sha256": "",
            "trained_in_canary": bool(trained_in_canary),
            "trained_at": _utc_now(),
            "source_data_hash": str(row["source_data_hash"]),
            "protocol_hash": str(row["protocol_hash"]),
        }
    series, source_key, audit = _prepare_key(
        row, histories, allowed_identities, strict_windows=strict_windows
    )
    calendar = calendar_for("D4")
    cache_key = str(row["cache_key"])
    last_error: BaseException | None = None
    status = None
    for attempt in range(1, 4):
        _reset_tf()
        try:
            _record, status = _load_or_train_source(
                histories[store],
                source_key=source_key,
                features=calendar.model_features,
                horizon=int(row["horizon"]),
                seed=int(row["seed"]),
                dataset="D4",
                key_fields=calendar.key_fields,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            _reset_tf()
            if _is_retryable_tf(exc) and attempt < 3:
                log.write(
                    f"retryable TF error for {row['source_identity']} h={row['horizon']} "
                    f"seed={row['seed']} attempt={attempt}: {type(exc).__name__}"
                )
                continue
            if _is_retryable_tf(exc) and not _in_child_process():
                _spawn_train_one(cache_key, log)
                last_error = None
                status = "TRAINED"
                break
            raise
        finally:
            _reset_tf()
    if last_error is not None:
        raise last_error
    directory = SOURCE_MODEL_CACHE / cache_key
    inspect_after = _inspect_cache(row, load_weights=True)
    if inspect_after != "CACHE_HIT":
        raise StageAStop("CACHE_HASH_MISMATCH", f"post-train inspect={inspect_after} status={status}")
    log.write(
        f"trained {row['source_identity']} h={row['horizon']} seed={row['seed']} "
        f"status={status} windows={audit['n_train_windows']}"
    )
    return {
        "key": cache_key,
        "store": store,
        "source": str(row["source_identity"]),
        "horizon": int(row["horizon"]),
        "seed": int(row["seed"]),
        "original_cache_status": str(row["cache_status"]),
        "final_status": "TRAINED_IN_CANARY" if trained_in_canary else "TRAINED",
        "checkpoint": str(directory / "weights.npz"),
        "error": "",
        "training_loss_final": "",
        "n_train_windows": audit["n_train_windows"],
        "checkpoint_sha256": _file_sha256(directory / "weights.npz"),
        "trained_in_canary": bool(trained_in_canary),
        "trained_at": _utc_now(),
        "source_data_hash": str(row["source_data_hash"]),
        "protocol_hash": str(row["protocol_hash"]),
        "source_start_date": audit["source_start_date"],
        "source_end_date": audit["source_end_date"],
        "n_source_rows": audit["n_source_rows"],
        "architecture": str(row["architecture"]),
        "dataset": "D4",
        "training_status": status,
    }


def _write_plan(miss: pd.DataFrame, canary: pd.DataFrame) -> None:
    plan = miss.copy()
    canary_keys = set(canary["cache_key"].astype(str))
    plan["trained_in_canary_planned"] = plan["cache_key"].astype(str).isin(canary_keys)
    plan["phase"] = np.where(plan["trained_in_canary_planned"], "canary", "remaining")
    _atomic_csv(OUT_DIR / "stage_a_training_plan.csv", plan)


def run_canary(req: pd.DataFrame, miss: pd.DataFrame, log: _Log) -> pd.DataFrame:
    CANARY_DIR.mkdir(parents=True, exist_ok=True)
    canary_log = _Log(CANARY_DIR / "canary_run_log.txt")
    canary = _canary_sources(miss)
    _atomic_csv(CANARY_DIR / "canary_plan.csv", canary)
    allowed = set(miss["source_identity"].astype(str))
    before166 = _snapshot_store166(req)
    histories = _load_store_histories(tuple(int(s) for s in canary["store_id"].tolist()), canary_log)
    trained_166 = [0]
    progress = _load_progress()
    results = []
    roundtrip = []
    all_ok = True
    for _, row in canary.iterrows():
        canary_log.write(f"canary start store={row['store_id']} source={row['source_identity']} h=1 seed=42")
        try:
            result = _train_one(
                row,
                histories,
                allowed,
                trained_in_canary=True,
                trained_166=trained_166,
                log=canary_log,
                strict_windows=True,
            )
            directory = SOURCE_MODEL_CACHE / str(row["cache_key"])
            record = _read_source_record(directory, tuple(str(row["source_identity"]).split("/")))
            reload_ok = bool(record.weights) and _inspect_cache(row) == "CACHE_HIT"
            result["cache_reload"] = "PASS" if reload_ok else "FAIL"
            if not reload_ok:
                all_ok = False
                result["final_status"] = "LOAD_FAILURE"
            loss_ok = True
            if result["training_loss_final"] not in {"", None}:
                loss_ok = np.isfinite(float(result["training_loss_final"]))
            if not loss_ok:
                all_ok = False
                result["final_status"] = "FAILED"
                result["error"] = "non-finite loss"
        except StageAStop as exc:
            all_ok = False
            result = {
                "key": str(row["cache_key"]),
                "store": int(row["store_id"]),
                "source": str(row["source_identity"]),
                "horizon": 1,
                "seed": 42,
                "original_cache_status": "CACHE_MISS",
                "final_status": exc.token,
                "checkpoint": "",
                "error": exc.detail,
                "training_loss_final": "",
                "n_train_windows": "",
                "checkpoint_sha256": "",
                "trained_in_canary": True,
                "trained_at": _utc_now(),
                "source_data_hash": str(row["source_data_hash"]),
                "protocol_hash": str(row["protocol_hash"]),
                "cache_reload": "FAIL",
            }
            canary_log.write(f"canary failed {exc.token}: {exc.detail}")
        except Exception as exc:
            all_ok = False
            result = {
                "key": str(row["cache_key"]),
                "store": int(row["store_id"]),
                "source": str(row["source_identity"]),
                "horizon": 1,
                "seed": 42,
                "original_cache_status": "CACHE_MISS",
                "final_status": "FAILED",
                "checkpoint": "",
                "error": f"{type(exc).__name__}: {exc}",
                "training_loss_final": "",
                "n_train_windows": "",
                "checkpoint_sha256": "",
                "trained_in_canary": True,
                "trained_at": _utc_now(),
                "source_data_hash": str(row["source_data_hash"]),
                "protocol_hash": str(row["protocol_hash"]),
                "cache_reload": "FAIL",
            }
            canary_log.write(traceback.format_exc())
        progress = _upsert_progress(progress, result)
        results.append(result)
        roundtrip.append(
            {
                "cache_key": str(row["cache_key"]),
                "store_id": int(row["store_id"]),
                "source_identity": str(row["source_identity"]),
                "inspect": _inspect_cache(row),
                "cache_reload": result.get("cache_reload"),
                "trained_in_canary": True,
            }
        )
    after166 = _snapshot_store166(req)
    try:
        _assert_store166_unchanged(before166, after166)
        store166_ok = True
        store166_error = ""
    except StageAStop as exc:
        all_ok = False
        store166_ok = False
        store166_error = str(exc)
        canary_log.write(str(exc))
    if trained_166[0] != 0:
        all_ok = False
        store166_ok = False
        store166_error = f"trained_keys_for_store_166={trained_166[0]}"
    result_frame = pd.DataFrame(results)
    _atomic_csv(CANARY_DIR / "canary_results.csv", result_frame)
    _atomic_json(
        CANARY_DIR / "cache_roundtrip_audit.json",
        {
            "canary_keys": roundtrip,
            "store_166_unchanged": store166_ok,
            "store_166_error": store166_error,
            "trained_keys_for_store_166": trained_166[0],
            "official_cache_root": str(SOURCE_MODEL_CACHE),
            "schema": "additional_target_robustness source_model_cache key.json+DONE.json+weights.npz",
        },
    )
    _atomic_json(
        CANARY_DIR / "DONE.json",
        {
            "status": "CANARY_PASSED" if all_ok else "CANARY_FAILED",
            "n_canary": int(len(result_frame)),
            "n_passed": int((result_frame["cache_reload"] == "PASS").sum()) if not result_frame.empty else 0,
            "trained_in_canary": True,
            "store_166_unchanged": store166_ok,
            "finished_at": _utc_now(),
        },
    )
    log.write(f"canary complete all_ok={all_ok}")
    if not all_ok:
        raise StageAStop("BLOCKED_STAGE_A_TRAINING", "canary failed")
    return result_frame


def run_remaining(req: pd.DataFrame, miss: pd.DataFrame, log: _Log) -> pd.DataFrame:
    allowed = set(miss["source_identity"].astype(str))
    canary_done = CANARY_DIR / "DONE.json"
    if not canary_done.is_file() or json.loads(canary_done.read_text(encoding="utf-8")).get("status") != "CANARY_PASSED":
        raise StageAStop("BLOCKED_STAGE_A_TRAINING", "canary has not passed")
    histories = _load_store_histories(TRAIN_STORES, log)
    trained_166 = [0]
    progress = _load_progress()
    done_keys = set(progress.loc[progress["final_status"].isin(["TRAINED", "TRAINED_IN_CANARY", "CACHE_HIT_AFTER_RESUME"]), "key"].astype(str))
    newly_trained = 0
    for _, row in miss.iterrows():
        key = str(row["cache_key"])
        if key in done_keys and _inspect_cache(row, load_weights=False) == "CACHE_HIT":
            continue
        try:
            result = _train_one(
                row,
                histories,
                allowed,
                trained_in_canary=False,
                trained_166=trained_166,
                log=log,
            )
            if result["final_status"] == "CACHE_HASH_MISMATCH":
                progress = _upsert_progress(progress, result)
                raise StageAStop("CACHE_HASH_MISMATCH", key)
        except StageAStop as exc:
            result = {
                "key": key,
                "store": int(row["store_id"]),
                "source": str(row["source_identity"]),
                "horizon": int(row["horizon"]),
                "seed": int(row["seed"]),
                "original_cache_status": str(row["cache_status"]),
                "final_status": exc.token,
                "checkpoint": "",
                "error": exc.detail,
                "training_loss_final": "",
                "n_train_windows": "",
                "checkpoint_sha256": "",
                "trained_in_canary": False,
                "trained_at": _utc_now(),
                "source_data_hash": str(row["source_data_hash"]),
                "protocol_hash": str(row["protocol_hash"]),
            }
            progress = _upsert_progress(progress, result)
            raise
        except Exception as exc:
            result = {
                "key": key,
                "store": int(row["store_id"]),
                "source": str(row["source_identity"]),
                "horizon": int(row["horizon"]),
                "seed": int(row["seed"]),
                "original_cache_status": str(row["cache_status"]),
                "final_status": "FAILED",
                "checkpoint": "",
                "error": f"{type(exc).__name__}: {exc}",
                "training_loss_final": "",
                "n_train_windows": "",
                "checkpoint_sha256": "",
                "trained_in_canary": False,
                "trained_at": _utc_now(),
                "source_data_hash": str(row["source_data_hash"]),
                "protocol_hash": str(row["protocol_hash"]),
            }
            progress = _upsert_progress(progress, result)
            log.write(traceback.format_exc())
            raise StageAStop("BLOCKED_STAGE_A_TRAINING", f"key {key} failed") from exc
        progress = _upsert_progress(progress, result)
        done_keys.add(key)
        if result["final_status"] in {"TRAINED", "TRAINED_IN_CANARY"}:
            newly_trained += 1
        if newly_trained >= RECYCLE_AFTER and not _in_child_process():
            leftover = int(len(miss) - len(done_keys))
            if leftover > 0:
                log.write(f"recycling TensorFlow process after {newly_trained} keys, leftover={leftover}")
                os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), "--phase", "train"])
    if trained_166[0] != 0:
        raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", f"trained_keys_for_store_166={trained_166[0]}")
    return progress


def run_one_key(req: pd.DataFrame, miss: pd.DataFrame, cache_key: str, log: _Log) -> dict:
    rows = miss.loc[miss["cache_key"].astype(str) == str(cache_key)]
    if len(rows) != 1:
        raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", f"unrequested cache key {cache_key}")
    row = rows.iloc[0]
    store = int(row["store_id"])
    histories = _load_store_histories((store,), log)
    trained_166 = [0]
    progress = _load_progress()
    result = _train_one(
        row,
        histories,
        set(miss["source_identity"].astype(str)),
        trained_in_canary=False,
        trained_166=trained_166,
        log=log,
    )
    if trained_166[0] != 0:
        raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", "store 166 train attempt")
    _upsert_progress(progress, result)
    if result["final_status"] not in {"TRAINED", "TRAINED_IN_CANARY", "CACHE_HIT_AFTER_RESUME"}:
        raise StageAStop("BLOCKED_STAGE_A_TRAINING", f"train-one status={result['final_status']}")
    return result


def final_audit(req: pd.DataFrame, log: _Log) -> dict:
    rows = []
    n_hit = 0
    n_miss = 0
    n_mismatch = 0
    n_load = 0
    for _, row in req.iterrows():
        status = _inspect_cache(row)
        rows.append(
            {
                "cache_key": str(row["cache_key"]),
                "store_id": int(row["store_id"]),
                "source_identity": str(row["source_identity"]),
                "horizon": int(row["horizon"]),
                "seed": int(row["seed"]),
                "original_cache_status": str(row["cache_status"]),
                "final_inspect": status,
            }
        )
        if status == "CACHE_HIT":
            n_hit += 1
        elif status == "CACHE_HASH_MISMATCH":
            n_mismatch += 1
        elif status == "LOAD_FAILURE":
            n_load += 1
        else:
            n_miss += 1
    audit = pd.DataFrame(rows)
    _atomic_csv(OUT_DIR / "stage_a_final_cache_audit.csv", audit)
    missing_by_source = []
    for identity, group in req.groupby("source_identity"):
        available = 0
        missing = []
        for _, row in group.iterrows():
            if _inspect_cache(row) == "CACHE_HIT":
                available += 1
            else:
                missing.append(
                    {
                        "source_identity": str(identity),
                        "horizon": int(row["horizon"]),
                        "seed": int(row["seed"]),
                    }
                )
        if available != 15:
            missing_by_source.extend(missing)
    payload = {
        "required_keys": int(len(req)),
        "available_keys": n_hit,
        "missing": n_miss,
        "hash_mismatch": n_mismatch,
        "load_failure": n_load,
        "original_cache_hit": EXPECTED_ORIGINAL_HIT,
        "expected_required": EXPECTED_REQUIRED,
        "n_required_sources": int(req["source_identity"].nunique()),
        "sources_complete_15": int(req["source_identity"].nunique() - len({item["source_identity"] for item in missing_by_source})),
        "source_missing_keys": missing_by_source,
        "ready": n_hit == EXPECTED_REQUIRED and n_miss == 0 and n_mismatch == 0 and n_load == 0 and not missing_by_source,
        "audited_at": _utc_now(),
    }
    _atomic_json(OUT_DIR / "stage_a_final_cache_audit.json", payload)
    log.write(
        f"final audit required={payload['required_keys']} available={n_hit} "
        f"missing={n_miss} mismatch={n_mismatch} load_failure={n_load}"
    )
    return payload


def _write_fatal(token: str, detail: str) -> None:
    _atomic_json(
        OUT_DIR / "HOLD.json",
        {"status": token, "detail": detail, "written_at": _utc_now()},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("canary", "train", "train-one", "audit", "all"), default="all")
    parser.add_argument("--cache-key", default="")
    args = parser.parse_args()
    log = _Log(OUT_DIR / "stage_a_run_log.txt")
    try:
        req = _load_requirements()
        miss = _assert_requirement_counts(req)
        log.write(f"n_cache_miss={len(miss)} phase={args.phase}")
        hold_path = OUT_DIR / "HOLD.json"
        if args.phase in {"train", "train-one"} and hold_path.is_file():
            hold_path.unlink()
        if args.phase == "train-one":
            if not args.cache_key:
                raise StageAStop("BLOCKED_STAGE_A_TRAINING", "train-one requires --cache-key")
            run_one_key(req, miss, args.cache_key, log)
            return 0
        canary = _canary_sources(miss)
        _write_plan(miss, canary)
        if args.phase in {"canary", "all"}:
            run_canary(req, miss, log)
        if args.phase in {"train", "all"}:
            run_remaining(req, miss, log)
        if args.phase in {"audit", "all", "train"}:
            payload = final_audit(req, log)
            status = "READY_FOR_D4_60TARGET_CANARY" if payload["ready"] else "BLOCKED_STAGE_A_TRAINING"
            _atomic_json(OUT_DIR / "stage_a_training_status.json", {"status": status, **payload})
            log.write(status)
            if status != "READY_FOR_D4_60TARGET_CANARY" and args.phase in {"audit", "all"}:
                return 1
        return 0
    except StageAStop as exc:
        log.write(f"{exc.token}: {exc.detail}")
        _write_fatal(exc.token, exc.detail)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
