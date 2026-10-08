#!/usr/bin/env python3
"""Train CACHE_MISS Stage A source CNNs for D5/D6 60-target robustness.

Does not run SITA, baselines, or overwrite exact-match cache hits.
"""

from __future__ import annotations

import argparse
import fcntl
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
from additional_target_robustness.runner import _load_or_train_source, _one_source, _read_source_record, _series_sha256
from additional_target_robustness.source_selection import load_source_histories

ensure_ablation_path()
from sita_ablation.data import split_source_frame
from sita_ablation.scaling import fit_source_model_scaler, transform_model_frame
from sita_ablation.windows import WINDOW_SIZE, build_partition_windows

OUT = {
    "D5": REPO_ROOT / "outputs" / "d5_60target_robustness_20261004",
    "D6": REPO_ROOT / "outputs" / "d6_60target_robustness_20261004",
}
EXPECTED = {
    "D5": {"required": 1725, "miss": 1500, "hit": 225, "sources": 115},
    "D6": {"required": 1845, "miss": 1740, "hit": 105, "sources": 123},
}
EXPECTED_TOTAL_MISS = 3240
EXPECTED_TOTAL_REQUIRED = 3570
SOURCE_SPAN = {
    "D5": ("2016-08-20", "2017-02-15"),
    "D6": ("2015-05-29", "2015-11-24"),
}
GROUP_VALUES = {"D5": "GROCERY I", "D6": "FOODS_3"}
STORE_VALUES = {"D5": "48", "D6": "CA_1"}
SOURCE_ROWS = 180
RECYCLE_AFTER = 40
LOCK_PATH = OUT["D5"] / "stage_a.train.lock"
_LOCK_HANDLE = None


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


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _combined_log() -> _Log:
    return _Log(OUT["D5"] / "stage_a_run_log.txt")


def _acquire_train_lock() -> None:
    global _LOCK_HANDLE
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise StageAStop(
            "BLOCKED_STAGE_A_TRAINING",
            "another Stage A --phase train process is already running; do not start a second copy",
        ) from exc
    handle.write(f"pid={os.getpid()} started={_utc_now()}\n")
    handle.flush()
    _LOCK_HANDLE = handle


PROGRESS_COLUMNS = [
    "key",
    "dataset",
    "store",
    "source",
    "horizon",
    "seed",
    "original_cache_status",
    "final_status",
    "checkpoint",
    "error",
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
    "training_status",
]


def _progress_path(dataset: str) -> Path:
    return OUT[dataset] / "stage_a_training_progress.csv"


def _failures_path(dataset: str) -> Path:
    return OUT[dataset] / "stage_a_training_failures.csv"


def _canary_dir(dataset: str) -> Path:
    return OUT[dataset] / "stage_a_canary"


def _load_progress(dataset: str) -> pd.DataFrame:
    path = _progress_path(dataset)
    if not path.is_file():
        return pd.DataFrame(columns=PROGRESS_COLUMNS)
    frame = pd.read_csv(path)
    for column in PROGRESS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[PROGRESS_COLUMNS]


def _progress_row(row: dict) -> dict:
    return {column: row.get(column, "") for column in PROGRESS_COLUMNS}


def _upsert_progress(dataset: str, progress: pd.DataFrame, row: dict) -> pd.DataFrame:
    addition = pd.DataFrame([_progress_row(row)])
    frame = progress.copy()
    if not frame.empty:
        frame = frame.loc[frame["key"].astype(str) != str(row["key"])]
    out = pd.concat([frame, addition], ignore_index=True)
    _atomic_csv(_progress_path(dataset), out)
    failures = out.loc[
        out["final_status"].isin(["FAILED", "CACHE_HASH_MISMATCH", "LOAD_FAILURE", "BLOCKED_SOURCE_DATA_PROTOCOL"])
    ].copy()
    _atomic_csv(_failures_path(dataset), failures)
    return out


def _load_requirements(dataset: str) -> pd.DataFrame:
    path = OUT[dataset] / "stage_a_requirements.csv"
    if not path.is_file():
        raise StageAStop("HOLD_STAGE_A_REQUIREMENT_COUNT_MISMATCH", f"missing {path}")
    frame = pd.read_csv(path)
    frame["dataset"] = frame["dataset"].astype(str)
    frame["store_id"] = frame["store_id"].astype(str)
    frame["source_identity"] = frame["source_identity"].astype(str)
    frame["horizon"] = frame["horizon"].astype(int)
    frame["seed"] = frame["seed"].astype(int)
    frame["cache_status"] = frame["cache_status"].astype(str)
    return frame


def _assert_requirement_counts(req: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    misses = {}
    total_miss = 0
    for dataset, expected in EXPECTED.items():
        frame = req[dataset]
        if len(frame) != expected["required"]:
            raise StageAStop(
                "HOLD_STAGE_A_REQUIREMENT_COUNT_MISMATCH",
                f"{dataset} required={len(frame)} expected={expected['required']}",
            )
        n_hit = int((frame["cache_status"] == "CACHE_HIT").sum())
        n_miss = int((frame["cache_status"] == "CACHE_MISS").sum())
        n_mismatch = int((frame["cache_status"] == "CACHE_HASH_MISMATCH").sum())
        if n_hit != expected["hit"] or n_miss != expected["miss"] or n_mismatch != 0:
            raise StageAStop(
                "HOLD_STAGE_A_REQUIREMENT_COUNT_MISMATCH",
                f"{dataset} hit={n_hit} miss={n_miss} mismatch={n_mismatch}",
            )
        n_sources = int(frame["source_identity"].nunique())
        if n_sources != expected["sources"]:
            raise StageAStop(
                "HOLD_STAGE_A_REQUIREMENT_COUNT_MISMATCH",
                f"{dataset} unique sources={n_sources} expected={expected['sources']}",
            )
        usage = pd.read_csv(OUT[dataset] / "source_usage_summary.csv")
        allowed = set(usage["source_identity"].astype(str))
        miss = frame.loc[frame["cache_status"] == "CACHE_MISS"].copy()
        extra = sorted(set(miss["source_identity"].astype(str)) - allowed)
        if extra:
            raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", f"{dataset} extra {extra[:5]}")
        misses[dataset] = miss.reset_index(drop=True)
        total_miss += n_miss
    if total_miss != EXPECTED_TOTAL_MISS:
        raise StageAStop("HOLD_STAGE_A_REQUIREMENT_COUNT_MISMATCH", f"total miss={total_miss}")
    return misses


def _assert_original_hits_untouched(req: dict[str, pd.DataFrame]) -> None:
    for dataset, frame in req.items():
        hits = frame.loc[frame["cache_status"] == "CACHE_HIT"]
        for _, row in hits.iterrows():
            status = _inspect_cache(row, load_weights=False)
            if status != "CACHE_HIT":
                raise StageAStop("CACHE_HASH_MISMATCH", f"{dataset} original hit became {status}")


def _dummy_target(dataset: str) -> pd.DataFrame:
    calendar = calendar_for(dataset)
    dummy = pd.DataFrame(columns=["date", *calendar.key_fields, calendar.group_field])
    dummy.loc[0, "date"] = pd.Timestamp(calendar.train_start)
    dummy.loc[0, calendar.key_fields[0]] = STORE_VALUES[dataset]
    dummy.loc[0, calendar.key_fields[1]] = "0"
    dummy.loc[0, calendar.group_field] = GROUP_VALUES[dataset]
    return dummy


def _load_histories(dataset: str, log: _Log) -> pd.DataFrame:
    log.write(f"loading frozen {dataset} source histories")
    dummy = _dummy_target(dataset)
    return load_source_histories(dataset, target_identity=f"{STORE_VALUES[dataset]}/0", target_frame=dummy)


def _assert_source_protocol(dataset: str, series: pd.DataFrame, features: tuple[str, ...], identity: str) -> None:
    start, end = SOURCE_SPAN[dataset]
    dates = pd.to_datetime(series["date"]).dt.normalize()
    if len(series) != SOURCE_ROWS:
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{dataset} {identity} n_rows={len(series)}")
    if dates.nunique() != SOURCE_ROWS or bool(dates.duplicated().any()):
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{dataset} {identity} dates are not 180 unique days")
    first = dates.min().strftime("%Y-%m-%d")
    last = dates.max().strftime("%Y-%m-%d")
    if first != start or last != end:
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{dataset} {identity} span {first}..{last}")
    expected = pd.date_range(start, end, freq="D")
    if not dates.reset_index(drop=True).equals(pd.Series(expected)):
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{dataset} {identity} is not 180 consecutive calendar days")
    if (dates > pd.Timestamp(end)).any():
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{dataset} {identity} has a row after source cutoff")
    sales = pd.to_numeric(series["sales"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(sales).all():
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{dataset} {identity} sales is not finite")
    for column in features:
        values = pd.to_numeric(series[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{dataset} {identity} feature {column} is not finite")


def _expected_train_windows(horizon: int) -> int:
    n_train = max(1, int(SOURCE_ROWS * 0.8))
    return n_train - WINDOW_SIZE - int(horizon) + 1


def _audit_windows(dataset: str, series: pd.DataFrame, *, source_key: tuple[str, ...], features: tuple[str, ...], horizon: int) -> dict:
    calendar = calendar_for(dataset)
    split = split_source_frame(series, features)
    scaler = fit_source_model_scaler(split.train, features)
    transformed = transform_model_frame(scaler, split.train)
    windows = build_partition_windows(
        transformed,
        key_fields=calendar.key_fields,
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
            f"{dataset} {'/'.join(source_key)} h={horizon} n_train_windows={n_windows} expected={expected}",
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
    start, end = SOURCE_SPAN[dataset]
    return {
        "n_train_windows": n_windows,
        "n_train_days": int(len(split.train)),
        "source_start_date": start,
        "source_end_date": end,
        "n_source_rows": SOURCE_ROWS,
    }


def _inspect_cache(row: pd.Series, *, load_weights: bool = True) -> str:
    directory = SOURCE_MODEL_CACHE / str(row["cache_key"])
    done_path = directory / "DONE.json"
    key_path = directory / "key.json"
    weights_path = directory / "weights.npz"
    if not directory.exists() or not (done_path.is_file() and key_path.is_file() and weights_path.is_file()):
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


def _canary_row(dataset: str, miss: pd.DataFrame) -> pd.Series:
    usage = pd.read_csv(OUT[dataset] / "source_usage_summary.csv")
    usage["source_identity"] = usage["source_identity"].astype(str)
    miss_ids = set(miss["source_identity"].astype(str))
    ranked = usage.loc[usage["source_identity"].isin(miss_ids)].sort_values(
        ["selected_by_n_targets", "source_identity"],
        ascending=[False, True],
        kind="mergesort",
    )
    if ranked.empty:
        raise StageAStop("HOLD_STAGE_A_REQUIREMENT_COUNT_MISMATCH", f"{dataset} no CACHE_MISS source")
    identity = str(ranked.iloc[0]["source_identity"])
    key_rows = miss.loc[(miss["source_identity"] == identity) & (miss["horizon"] == 1) & (miss["seed"] == 42)]
    if len(key_rows) != 1:
        raise StageAStop("HOLD_STAGE_A_REQUIREMENT_COUNT_MISMATCH", f"{dataset} canary key not unique for {identity}")
    return key_rows.iloc[0]


_TF_RESETS = {"n": 0}


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
    return os.environ.get("D5D6_STAGE_A_TRAIN_CHILD") == "1"


def _spawn_train_one(cache_key: str, log: _Log) -> None:
    env = os.environ.copy()
    env["D5D6_STAGE_A_TRAIN_CHILD"] = "1"
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", "train-one", "--cache-key", cache_key]
    log.write(f"retrying {cache_key} in an isolated TensorFlow process")
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode != 0:
        raise StageAStop("BLOCKED_STAGE_A_TRAINING", f"isolated retry failed for {cache_key}")


def _prepare_key(row: pd.Series, histories: pd.DataFrame, allowed: set[str], *, strict_windows: bool) -> tuple[pd.DataFrame, tuple[str, ...], dict]:
    dataset = str(row["dataset"])
    identity = str(row["source_identity"])
    if identity not in allowed:
        raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", identity)
    calendar = calendar_for(dataset)
    source_key = tuple(identity.split("/"))
    series = _one_source(histories, source_key, calendar.key_fields)
    _assert_source_protocol(dataset, series, calendar.model_features, identity)
    computed = _series_sha256(series, calendar.model_features)
    if computed != str(row["source_data_hash"]):
        raise StageAStop("BLOCKED_SOURCE_DATA_PROTOCOL", f"{identity} source_data_hash drifted")
    if strict_windows:
        audit = _audit_windows(
            dataset,
            series,
            source_key=source_key,
            features=calendar.model_features,
            horizon=int(row["horizon"]),
        )
    else:
        start, end = SOURCE_SPAN[dataset]
        n_train = max(1, int(SOURCE_ROWS * 0.8))
        audit = {
            "n_train_windows": n_train - WINDOW_SIZE - int(row["horizon"]) + 1,
            "n_train_days": n_train,
            "source_start_date": start,
            "source_end_date": end,
            "n_source_rows": SOURCE_ROWS,
        }
    return series, source_key, audit


def _train_one(row: pd.Series, histories: pd.DataFrame, allowed: set[str], *, trained_in_canary: bool, log: _Log, strict_windows: bool = False) -> dict:
    dataset = str(row["dataset"])
    inspect = _inspect_cache(row, load_weights=False)
    if inspect == "CACHE_HIT":
        directory = SOURCE_MODEL_CACHE / str(row["cache_key"])
        return {
            "key": str(row["cache_key"]),
            "dataset": dataset,
            "store": str(row["store_id"]),
            "source": str(row["source_identity"]),
            "horizon": int(row["horizon"]),
            "seed": int(row["seed"]),
            "original_cache_status": str(row["cache_status"]),
            "final_status": "CACHE_HIT_AFTER_RESUME",
            "checkpoint": str(directory / "weights.npz"),
            "error": "",
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
            "dataset": dataset,
            "store": str(row["store_id"]),
            "source": str(row["source_identity"]),
            "horizon": int(row["horizon"]),
            "seed": int(row["seed"]),
            "original_cache_status": str(row["cache_status"]),
            "final_status": "CACHE_HASH_MISMATCH",
            "checkpoint": str(SOURCE_MODEL_CACHE / str(row["cache_key"])),
            "error": "CACHE_HASH_MISMATCH",
            "n_train_windows": "",
            "checkpoint_sha256": "",
            "trained_in_canary": bool(trained_in_canary),
            "trained_at": _utc_now(),
            "source_data_hash": str(row["source_data_hash"]),
            "protocol_hash": str(row["protocol_hash"]),
        }
    _series, source_key, audit = _prepare_key(row, histories, allowed, strict_windows=strict_windows)
    calendar = calendar_for(dataset)
    cache_key = str(row["cache_key"])
    last_error: BaseException | None = None
    status = None
    for attempt in range(1, 4):
        _reset_tf()
        try:
            _record, status = _load_or_train_source(
                histories,
                source_key=source_key,
                features=calendar.model_features,
                horizon=int(row["horizon"]),
                seed=int(row["seed"]),
                dataset=dataset,
                key_fields=calendar.key_fields,
            )
            last_error = None
            break
        except FileNotFoundError as exc:
            inspect_race = _inspect_cache(row, load_weights=True)
            if inspect_race == "CACHE_HIT":
                directory = SOURCE_MODEL_CACHE / str(row["cache_key"])
                return {
                    "key": str(row["cache_key"]),
                    "dataset": dataset,
                    "store": str(row["store_id"]),
                    "source": str(row["source_identity"]),
                    "horizon": int(row["horizon"]),
                    "seed": int(row["seed"]),
                    "original_cache_status": str(row["cache_status"]),
                    "final_status": "CACHE_HIT_AFTER_RESUME",
                    "checkpoint": str(directory / "weights.npz"),
                    "error": "",
                    "n_train_windows": "",
                    "checkpoint_sha256": _file_sha256(directory / "weights.npz"),
                    "trained_in_canary": bool(trained_in_canary),
                    "trained_at": _utc_now(),
                    "source_data_hash": str(row["source_data_hash"]),
                    "protocol_hash": str(row["protocol_hash"]),
                }
            last_error = exc
            _reset_tf()
        except Exception as exc:
            last_error = exc
            _reset_tf()
            if _is_retryable_tf(exc) and attempt < 3:
                log.write(
                    f"retryable TF error for {dataset} {row['source_identity']} h={row['horizon']} "
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
        f"trained {dataset} {row['source_identity']} h={row['horizon']} seed={row['seed']} "
        f"status={status} windows={audit['n_train_windows']}"
    )
    return {
        "key": cache_key,
        "dataset": dataset,
        "store": str(row["store_id"]),
        "source": str(row["source_identity"]),
        "horizon": int(row["horizon"]),
        "seed": int(row["seed"]),
        "original_cache_status": str(row["cache_status"]),
        "final_status": "TRAINED_IN_CANARY" if trained_in_canary else "TRAINED",
        "checkpoint": str(directory / "weights.npz"),
        "error": "",
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
        "training_status": status,
    }


def run_canary(req: dict[str, pd.DataFrame], misses: dict[str, pd.DataFrame], log: _Log) -> None:
    all_ok = True
    for dataset in ("D5", "D6"):
        canary_root = _canary_dir(dataset)
        canary_root.mkdir(parents=True, exist_ok=True)
        canary_log = _Log(canary_root / "canary_run_log.txt")
        row = _canary_row(dataset, misses[dataset])
        _atomic_csv(canary_root / "canary_plan.csv", pd.DataFrame([row]))
        canary_log.write(f"canary start {dataset} source={row['source_identity']} h=1 seed=42")
        allowed = set(misses[dataset]["source_identity"].astype(str))
        histories = _load_histories(dataset, canary_log)
        progress = _load_progress(dataset)
        try:
            result = _train_one(row, histories, allowed, trained_in_canary=True, log=canary_log, strict_windows=True)
            directory = SOURCE_MODEL_CACHE / str(row["cache_key"])
            record = _read_source_record(directory, tuple(str(row["source_identity"]).split("/")))
            reload_ok = bool(record.weights) and _inspect_cache(row) == "CACHE_HIT"
            result["cache_reload"] = "PASS" if reload_ok else "FAIL"
            if not reload_ok:
                all_ok = False
                result["final_status"] = "LOAD_FAILURE"
        except Exception as exc:
            all_ok = False
            result = {
                "key": str(row["cache_key"]),
                "dataset": dataset,
                "store": str(row["store_id"]),
                "source": str(row["source_identity"]),
                "horizon": 1,
                "seed": 42,
                "original_cache_status": "CACHE_MISS",
                "final_status": getattr(exc, "token", "FAILED"),
                "checkpoint": "",
                "error": getattr(exc, "detail", f"{type(exc).__name__}: {exc}"),
                "n_train_windows": "",
                "checkpoint_sha256": "",
                "trained_in_canary": True,
                "trained_at": _utc_now(),
                "source_data_hash": str(row["source_data_hash"]),
                "protocol_hash": str(row["protocol_hash"]),
                "cache_reload": "FAIL",
            }
            canary_log.write(traceback.format_exc())
        progress = _upsert_progress(dataset, progress, result)
        _atomic_csv(canary_root / "canary_results.csv", pd.DataFrame([result]))
        _atomic_json(
            canary_root / "cache_roundtrip_audit.json",
            {
                "dataset": dataset,
                "cache_key": str(row["cache_key"]),
                "source_identity": str(row["source_identity"]),
                "inspect": _inspect_cache(row),
                "cache_reload": result.get("cache_reload"),
                "official_cache_root": str(SOURCE_MODEL_CACHE),
                "sita_ran": False,
            },
        )
        passed = result.get("cache_reload") == "PASS" and result["final_status"] in {"TRAINED_IN_CANARY", "CACHE_HIT_AFTER_RESUME"}
        _atomic_json(
            canary_root / "DONE.json",
            {
                "status": "CANARY_PASSED" if passed else "CANARY_FAILED",
                "dataset": dataset,
                "n_canary": 1,
                "source_identity": str(row["source_identity"]),
                "finished_at": _utc_now(),
            },
        )
        if not passed:
            all_ok = False
        log.write(f"{dataset} canary passed={passed} source={row['source_identity']}")
    if not all_ok:
        raise StageAStop("BLOCKED_STAGE_A_TRAINING", "D5/D6 Stage A canary failed")


def _canary_passed() -> bool:
    for dataset in ("D5", "D6"):
        path = _canary_dir(dataset) / "DONE.json"
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")).get("status") != "CANARY_PASSED":
            return False
    return True


def run_remaining(misses: dict[str, pd.DataFrame], log: _Log) -> None:
    if not _canary_passed():
        raise StageAStop("BLOCKED_STAGE_A_TRAINING", "canary has not passed")
    histories = {dataset: _load_histories(dataset, log) for dataset in ("D5", "D6")}
    newly_trained = 0
    for dataset in ("D5", "D6"):
        miss = misses[dataset]
        allowed = set(miss["source_identity"].astype(str))
        progress = _load_progress(dataset)
        done_keys = set(
            progress.loc[
                progress["final_status"].isin(["TRAINED", "TRAINED_IN_CANARY", "CACHE_HIT_AFTER_RESUME"]),
                "key",
            ].astype(str)
        )
        for _, row in miss.iterrows():
            key = str(row["cache_key"])
            if key in done_keys and _inspect_cache(row, load_weights=False) == "CACHE_HIT":
                continue
            try:
                result = _train_one(row, histories[dataset], allowed, trained_in_canary=False, log=log)
                if result["final_status"] == "CACHE_HASH_MISMATCH":
                    _upsert_progress(dataset, progress, result)
                    raise StageAStop("CACHE_HASH_MISMATCH", key)
            except StageAStop:
                raise
            except Exception as exc:
                result = {
                    "key": key,
                    "dataset": dataset,
                    "store": str(row["store_id"]),
                    "source": str(row["source_identity"]),
                    "horizon": int(row["horizon"]),
                    "seed": int(row["seed"]),
                    "original_cache_status": str(row["cache_status"]),
                    "final_status": "FAILED",
                    "checkpoint": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_train_windows": "",
                    "checkpoint_sha256": "",
                    "trained_in_canary": False,
                    "trained_at": _utc_now(),
                    "source_data_hash": str(row["source_data_hash"]),
                    "protocol_hash": str(row["protocol_hash"]),
                }
                _upsert_progress(dataset, progress, result)
                log.write(traceback.format_exc())
                raise StageAStop("BLOCKED_STAGE_A_TRAINING", f"key {key} failed") from exc
            progress = _upsert_progress(dataset, progress, result)
            done_keys.add(key)
            if result["final_status"] in {"TRAINED", "TRAINED_IN_CANARY"}:
                newly_trained += 1
            if newly_trained >= RECYCLE_AFTER and not _in_child_process():
                log.write(f"recycling TensorFlow process after {newly_trained} keys")
                os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), "--phase", "train"])


def run_one_key(req: dict[str, pd.DataFrame], misses: dict[str, pd.DataFrame], cache_key: str, log: _Log) -> dict:
    row = None
    dataset = None
    for name, miss in misses.items():
        rows = miss.loc[miss["cache_key"].astype(str) == str(cache_key)]
        if len(rows) == 1:
            row = rows.iloc[0]
            dataset = name
            break
        if len(rows) > 1:
            raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", f"duplicate cache key {cache_key}")
    if row is None or dataset is None:
        raise StageAStop("HOLD_UNREQUESTED_SOURCE_MODEL", f"unrequested cache key {cache_key}")
    histories = _load_histories(dataset, log)
    progress = _load_progress(dataset)
    result = _train_one(row, histories, set(misses[dataset]["source_identity"].astype(str)), trained_in_canary=False, log=log)
    _upsert_progress(dataset, progress, result)
    if result["final_status"] not in {"TRAINED", "TRAINED_IN_CANARY", "CACHE_HIT_AFTER_RESUME"}:
        raise StageAStop("BLOCKED_STAGE_A_TRAINING", f"train-one status={result['final_status']}")
    return result


def final_audit(req: dict[str, pd.DataFrame], log: _Log) -> dict:
    by_dataset = {}
    total_available = 0
    total_missing = 0
    total_mismatch = 0
    total_load = 0
    ready = True
    for dataset, frame in req.items():
        rows = []
        n_hit = n_miss = n_mismatch = n_load = 0
        missing_by_source = []
        for identity, group in frame.groupby("source_identity"):
            available = 0
            missing = []
            for _, row in group.iterrows():
                status = _inspect_cache(row)
                rows.append(
                    {
                        "cache_key": str(row["cache_key"]),
                        "dataset": dataset,
                        "source_identity": str(row["source_identity"]),
                        "horizon": int(row["horizon"]),
                        "seed": int(row["seed"]),
                        "original_cache_status": str(row["cache_status"]),
                        "final_inspect": status,
                    }
                )
                if status == "CACHE_HIT":
                    n_hit += 1
                    available += 1
                elif status == "CACHE_HASH_MISMATCH":
                    n_mismatch += 1
                    missing.append({"source_identity": str(identity), "horizon": int(row["horizon"]), "seed": int(row["seed"])})
                elif status == "LOAD_FAILURE":
                    n_load += 1
                    missing.append({"source_identity": str(identity), "horizon": int(row["horizon"]), "seed": int(row["seed"])})
                else:
                    n_miss += 1
                    missing.append({"source_identity": str(identity), "horizon": int(row["horizon"]), "seed": int(row["seed"])})
            if available != 15:
                missing_by_source.extend(missing)
        audit = pd.DataFrame(rows)
        _atomic_csv(OUT[dataset] / "stage_a_final_cache_audit.csv", audit)
        expected = EXPECTED[dataset]
        dataset_ready = (
            n_hit == expected["required"]
            and n_miss == 0
            and n_mismatch == 0
            and n_load == 0
            and not missing_by_source
            and int(frame["source_identity"].nunique()) == expected["sources"]
        )
        payload = {
            "dataset": dataset,
            "required_keys": int(len(frame)),
            "available_keys": n_hit,
            "missing": n_miss,
            "hash_mismatch": n_mismatch,
            "load_failure": n_load,
            "original_cache_hit": expected["hit"],
            "n_required_sources": expected["sources"],
            "sources_complete_15": expected["sources"] - len({item["source_identity"] for item in missing_by_source}),
            "source_missing_keys": missing_by_source,
            "ready": dataset_ready,
            "sita_ran": False,
            "audited_at": _utc_now(),
        }
        _atomic_json(OUT[dataset] / "stage_a_final_cache_audit.json", payload)
        by_dataset[dataset] = payload
        total_available += n_hit
        total_missing += n_miss
        total_mismatch += n_mismatch
        total_load += n_load
        ready = ready and dataset_ready
        log.write(
            f"{dataset} audit required={len(frame)} available={n_hit} missing={n_miss} "
            f"mismatch={n_mismatch} load_failure={n_load} complete={payload['sources_complete_15']}/{expected['sources']}"
        )
    status = "READY_FOR_D5_D6_SITA_CANARY" if ready else "BLOCKED_STAGE_A_TRAINING"
    combined = {
        "status": status,
        "required": EXPECTED_TOTAL_REQUIRED,
        "available": total_available,
        "missing": total_missing,
        "hash_mismatch": total_mismatch,
        "load_failure": total_load,
        "datasets": by_dataset,
        "sita_ran": False,
        "audited_at": _utc_now(),
    }
    for dataset in ("D5", "D6"):
        _atomic_json(OUT[dataset] / "stage_a_training_status.json", combined)
    log.write(status)
    return combined


def _write_fatal(token: str, detail: str) -> None:
    payload = {"status": token, "detail": detail, "written_at": _utc_now()}
    for dataset in ("D5", "D6"):
        _atomic_json(OUT[dataset] / "STAGE_A_HOLD.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("canary", "train", "train-one", "audit", "all"), default="all")
    parser.add_argument("--cache-key", default="")
    args = parser.parse_args()
    for path in OUT.values():
        path.mkdir(parents=True, exist_ok=True)
    log = _combined_log()
    try:
        if args.phase in {"canary", "train", "all"}:
            _acquire_train_lock()
        req = {dataset: _load_requirements(dataset) for dataset in ("D5", "D6")}
        misses = _assert_requirement_counts(req)
        _assert_original_hits_untouched(req)
        log.write(f"n_cache_miss={sum(len(item) for item in misses.values())} phase={args.phase}")
        for dataset in ("D5", "D6"):
            hold = OUT[dataset] / "STAGE_A_HOLD.json"
            if args.phase in {"train", "train-one"} and hold.is_file():
                hold.unlink()
        if args.phase == "train-one":
            if not args.cache_key:
                raise StageAStop("BLOCKED_STAGE_A_TRAINING", "train-one requires --cache-key")
            run_one_key(req, misses, args.cache_key, log)
            return 0
        if args.phase in {"canary", "all"}:
            run_canary(req, misses, log)
        if args.phase in {"train", "all"}:
            run_remaining(misses, log)
        if args.phase in {"audit", "all", "train"}:
            payload = final_audit(req, log)
            print(payload["status"], flush=True)
            if payload["status"] != "READY_FOR_D5_D6_SITA_CANARY" and args.phase in {"audit", "all"}:
                return 1
        return 0
    except (StageAStop, ProtocolHold) as exc:
        token = getattr(exc, "token", type(exc).__name__)
        detail = getattr(exc, "detail", str(exc))
        log.write(f"{token}: {detail}")
        _write_fatal(token, detail)
        print(token, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
