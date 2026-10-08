#!/usr/bin/env python3
"""D5/D6 60-target SITA canary and full run.

Freezes the existing 60-target manifests and Stage A source_model_cache.
Does not train new source CNNs. Does not rewrite frozen selection or calendars.
"""

from __future__ import annotations

import argparse
import hashlib
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

from additional_target_robustness.horizon import verify_direct_horizon
from additional_target_robustness.paths import SOURCE_MODEL_CACHE, ensure_ablation_path
from additional_target_robustness.protocol import (
    HORIZONS,
    SEEDS,
    TEST_DAYS,
    ProtocolError,
    ProtocolHold,
    build_evaluation_index,
    calendar_for,
    frozen_protocol_sha256,
    sha256_file,
)
from additional_target_robustness.runner import _one_source, _predict_sita, _read_source_record, _series_sha256
from additional_target_robustness.source_selection import load_source_histories, select_top3
from additional_target_robustness.target_data import load_target_series, sales_by_date, split_target_frame
import additional_target_robustness.runner as sita_runner

ensure_ablation_path()
from sita_ablation.metrics import evaluate_original_sales

DATASETS = {
    "D5": {
        "out": REPO_ROOT / "outputs" / "d5_60target_robustness_20261004",
        "manifest_name": "d5_60target_manifest.json",
        "manifest_sha256": "613f992d448a2717f6ab119bd10a3d2ebc7a708e72aca28f35020148f86c63b8",
        "source_sha256": "0cbbda5ae42e2e962e7fd83a4b57a10d4fc0fdafc2687177bee0841e6d40e506",
        "required_keys": 1725,
        "canary_target": "48/261700",
        "store": "48",
        "ready_canary": "READY_FOR_D5_SITA_FULL",
        "ready_full": "D5_60TARGET_SITA_FULL_COMPLETE",
    },
    "D6": {
        "out": REPO_ROOT / "outputs" / "d6_60target_robustness_20261004",
        "manifest_name": "d6_60target_manifest.json",
        "manifest_sha256": "6fb0b8045ec912a661fbc2d705ee38ae6429b3d2af3423020b3a09447d30b0f7",
        "source_sha256": "12297d9b00a8e40fd9f13966e41aa558fe15e00079f8d0b23477bf4e7c288458",
        "required_keys": 1845,
        "canary_target": "CA_1/FOODS_3_726",
        "store": "CA_1",
        "ready_canary": "READY_FOR_D6_SITA_FULL",
        "ready_full": "D6_60TARGET_SITA_FULL_COMPLETE",
    },
}
PROTOCOL_SHA256 = "405beca0c84a85d3bdd77016cda37d5c2744ca7d4bc863e5c9d77ab17cd7788c"
EXPECTED_TARGETS = 60
EXPECTED_FULL = 900
EXPECTED_CANARY = 3
RECYCLE_AFTER = 15
CANARY_SEED = 42


class SitaStop(RuntimeError):
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


def _payload_digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _cfg(dataset: str) -> dict:
    return DATASETS[dataset]


def _sita_root(dataset: str) -> Path:
    return _cfg(dataset)["out"] / "sita"


def load_frozen_manifest(dataset: str) -> list[dict]:
    cfg = _cfg(dataset)
    path = cfg["out"] / cfg["manifest_name"]
    actual = sha256_file(path)
    if actual != cfg["manifest_sha256"]:
        raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", f"{dataset} manifest sha256 {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload["records"])
    if len(records) != EXPECTED_TARGETS:
        raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", f"{dataset} n_targets={len(records)}")
    if str(payload.get("source_data_sha256") or payload.get("source_calendar_sha256")) != cfg["source_sha256"]:
        raise SitaStop("HOLD_SOURCE_CACHE_HASH_MISMATCH", f"{dataset} source sha256 drifted")
    identities = [str(item["target_identity"]) for item in records]
    if len(set(identities)) != EXPECTED_TARGETS:
        raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", f"{dataset} duplicate targets")
    if cfg["canary_target"] not in identities:
        raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", f"{dataset} missing canary {cfg['canary_target']}")
    return records


def load_frozen_selection(dataset: str) -> pd.DataFrame:
    path = _cfg(dataset)["out"] / "source_selection_results.csv"
    if not path.is_file():
        raise SitaStop("HOLD_SOURCE_SELECTION_DRIFT", f"{dataset} missing source_selection_results.csv")
    frame = pd.read_csv(path)
    if len(frame) != EXPECTED_TARGETS:
        raise SitaStop("HOLD_SOURCE_SELECTION_DRIFT", f"{dataset} n_selection={len(frame)}")
    return frame


def load_requirements(dataset: str) -> pd.DataFrame:
    path = _cfg(dataset)["out"] / "stage_a_requirements.csv"
    if not path.is_file():
        raise SitaStop("HOLD_STAGE_A_CACHE_MISS", f"{dataset} missing stage_a_requirements.csv")
    frame = pd.read_csv(path)
    if len(frame) != _cfg(dataset)["required_keys"]:
        raise SitaStop("HOLD_STAGE_A_CACHE_MISS", f"{dataset} required_keys={len(frame)}")
    return frame


def _snapshot_required(req: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for _, row in req.sort_values("cache_key").iterrows():
        key = str(row["cache_key"])
        directory = SOURCE_MODEL_CACHE / key
        done = directory / "DONE.json"
        key_path = directory / "key.json"
        weights = directory / "weights.npz"
        if not (done.is_file() and key_path.is_file() and weights.is_file()):
            raise SitaStop("HOLD_STAGE_A_CACHE_MISS", f"{dataset} {key}")
        rows.append(
            {
                "cache_key": key,
                "source_identity": str(row["source_identity"]),
                "horizon": int(row["horizon"]),
                "seed": int(row["seed"]),
                "done_sha256": sha256_file(done),
                "key_sha256": sha256_file(key_path),
                "weights_sha256": sha256_file(weights),
            }
        )
    return pd.DataFrame(rows)


def _merkle(frame: pd.DataFrame) -> dict:
    hasher = hashlib.sha256()
    for _, row in frame.sort_values("cache_key").iterrows():
        hasher.update(
            f"{row['cache_key']}\t{row['done_sha256']}\t{row['key_sha256']}\t{row['weights_sha256']}\n".encode("utf-8")
        )
    return {"required_files_sha256": hasher.hexdigest(), "n_required": int(len(frame))}


def freeze_artifacts(dataset: str, records: list[dict], req: pd.DataFrame, log: _Log) -> dict:
    root = _sita_root(dataset)
    root.mkdir(parents=True, exist_ok=True)
    freeze_path = root / "freeze_lock.json"
    protocol_sha = frozen_protocol_sha256()
    if protocol_sha != PROTOCOL_SHA256:
        raise SitaStop("HOLD_HASH_MISMATCH", f"protocol sha256 {protocol_sha}")
    index = _snapshot_required(req, dataset)
    merkle = _merkle(index)
    payload = {
        "frozen_at": _utc_now(),
        "dataset": dataset,
        "manifest_sha256": _cfg(dataset)["manifest_sha256"],
        "source_data_sha256": _cfg(dataset)["source_sha256"],
        "protocol_sha256": PROTOCOL_SHA256,
        "source_model_cache": str(SOURCE_MODEL_CACHE.relative_to(REPO_ROOT)),
        "n_targets": len(records),
        "n_required_stage_a_keys": int(len(req)),
        "no_source_identity_change": True,
        "no_cache_key_change": True,
        "no_training_protocol_change": True,
        "hit_only": True,
        **merkle,
    }
    if freeze_path.is_file():
        stored = json.loads(freeze_path.read_text(encoding="utf-8"))
        for field in ("manifest_sha256", "protocol_sha256", "required_files_sha256", "n_required"):
            if stored.get(field) != payload[field]:
                raise SitaStop("HOLD_SOURCE_CACHE_MUTATED", f"{dataset} {field} drifted")
        log.write(f"{dataset} freeze lock already present and unchanged")
        return stored
    _atomic_csv(root / "freeze_cache_index.csv", index)
    _atomic_json(freeze_path, payload)
    log.write(f"{dataset} froze required={merkle['n_required']} files={merkle['required_files_sha256'][:12]}")
    return payload


def assert_freeze_unchanged(dataset: str, req: pd.DataFrame) -> dict:
    freeze_path = _sita_root(dataset) / "freeze_lock.json"
    if not freeze_path.is_file():
        raise SitaStop("HOLD_SOURCE_CACHE_MUTATED", f"{dataset} freeze_lock.json missing")
    stored = json.loads(freeze_path.read_text(encoding="utf-8"))
    merkle = _merkle(_snapshot_required(req, dataset))
    for field in ("required_files_sha256", "n_required"):
        if stored.get(field) != merkle[field]:
            raise SitaStop("HOLD_SOURCE_CACHE_MUTATED", f"{dataset} {field} drifted")
    if stored.get("manifest_sha256") != _cfg(dataset)["manifest_sha256"]:
        raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", f"{dataset} freeze manifest drifted")
    return stored


def _selection_row(frame: pd.DataFrame, identity: str) -> pd.Series:
    rows = frame.loc[frame["target_identity"].astype(str) == identity]
    if len(rows) != 1:
        raise SitaStop("HOLD_SOURCE_SELECTION_DRIFT", identity)
    return rows.iloc[0]


def _live_select(dataset: str, identity: str, *, histories: pd.DataFrame, frozen: pd.Series):
    calendar = calendar_for(dataset)
    target = load_target_series(dataset, identity)
    train, validation, test = split_target_frame(target, calendar)
    if len(train) != 15 or len(validation) != 15 or len(test) != TEST_DAYS:
        raise SitaStop("BLOCKED_SITA_PROTOCOL", f"{dataset} {identity} partition lengths")
    if dataset == "D5":
        cutoff = pd.Timestamp("2017-08-14")
        if (target["date"] > cutoff).any() or (target["date"] == pd.Timestamp("2017-08-15")).any():
            raise SitaStop("BLOCKED_SITA_PROTOCOL", f"{identity} has dates after 2017-08-14")
    store = identity.split("/")[0]
    selection = select_top3(
        dataset,
        identity,
        target,
        histories,
        source_universe_id=str(frozen["source_universe_id"]),
    )
    if selection.status != "COMPLETED" or len(selection.selected) != 3:
        raise SitaStop("HOLD_SOURCE_SELECTION_DRIFT", f"{identity} status={selection.status}")
    live = [item.identity for item in selection.selected]
    expected = [str(frozen["source_1"]), str(frozen["source_2"]), str(frozen["source_3"])]
    if live != expected:
        raise SitaStop("HOLD_SOURCE_SELECTION_DRIFT", f"{identity} live={live} frozen={expected}")
    if identity in live:
        raise SitaStop("HOLD_TARGET_NOT_EXCLUDED", identity)
    if any(item.split("/")[0] != store for item in live):
        raise SitaStop("HOLD_SOURCE_SELECTION_DRIFT", f"{identity} left store {live}")
    for item, rank in zip(selection.selected, (1, 2, 3)):
        if abs(float(item.distance) - float(frozen[f"distance_{rank}"])) > 1e-12:
            raise SitaStop("HOLD_SOURCE_SELECTION_DRIFT", f"{identity} distance rank {rank}")
        if abs(float(item.weight) - float(frozen[f"weight_{rank}"])) > 1e-12:
            raise SitaStop("HOLD_SOURCE_SELECTION_DRIFT", f"{identity} weight rank {rank}")
    return target, train, validation, selection


def _cache_lookup(req: pd.DataFrame, identity: str, horizon: int, seed: int) -> pd.Series:
    rows = req.loc[
        (req["source_identity"].astype(str) == identity)
        & (req["horizon"].astype(int) == int(horizon))
        & (req["seed"].astype(int) == int(seed))
    ]
    if len(rows) != 1:
        raise SitaStop("HOLD_STAGE_A_CACHE_MISS", f"{identity} h={horizon} seed={seed}")
    return rows.iloc[0]


def _inspect_requirement(dataset: str, row: pd.Series) -> str:
    directory = SOURCE_MODEL_CACHE / str(row["cache_key"])
    done_path = directory / "DONE.json"
    key_path = directory / "key.json"
    weights_path = directory / "weights.npz"
    if not (done_path.is_file() and key_path.is_file() and weights_path.is_file()):
        return "CACHE_MISS"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": dataset,
        "source_identity": str(row["source_identity"]),
        "horizon": int(row["horizon"]),
        "seed": int(row["seed"]),
        "architecture": "cnn",
        "source_data_sha256": str(row["source_data_hash"]),
        "protocol_sha256": str(row["protocol_hash"]),
    }
    digest = _payload_digest(expected)
    if str(row["cache_key"]) != digest:
        return "CACHE_HASH_MISMATCH"
    if done.get("cache_key_sha256") != digest or done.get("status") != "COMPLETED":
        return "CACHE_HASH_MISMATCH"
    for field, value in expected.items():
        if key.get(field) != value:
            return "CACHE_HASH_MISMATCH"
    try:
        record = _read_source_record(directory, tuple(str(row["source_identity"]).split("/")))
        if not record.weights:
            return "LOAD_FAILURE"
    except Exception:
        return "LOAD_FAILURE"
    return "CACHE_HIT"


def _load_source_hit_only(histories, *, source_key, features, horizon, seed, dataset, key_fields):
    if dataset not in DATASETS:
        raise SitaStop("BLOCKED_SITA_PROTOCOL", f"refusing dataset {dataset}")
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
    digest = _payload_digest(payload)
    directory = SOURCE_MODEL_CACHE / digest
    done_path = directory / "DONE.json"
    if not done_path.is_file():
        raise SitaStop("HOLD_STAGE_A_CACHE_MISS", f"{dataset} {identity} h={horizon} seed={seed} key={digest}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("cache_key_sha256") != digest or done.get("status") != "COMPLETED":
        raise SitaStop("HOLD_HASH_MISMATCH", digest)
    stored = json.loads((directory / "key.json").read_text(encoding="utf-8"))
    if stored != payload:
        raise SitaStop("HOLD_HASH_MISMATCH", f"{digest} key.json drifted")
    return _read_source_record(directory, source_key), "CACHE_HIT"


sita_runner._load_or_train_source = _load_source_hit_only


def _load_histories(dataset: str, records: list[dict], log: _Log) -> pd.DataFrame:
    identity = str(records[0]["target_identity"])
    log.write(f"{dataset} loading frozen source histories via {identity}")
    target = load_target_series(dataset, identity)
    return load_source_histories(dataset, target_identity=identity, target_frame=target)


def run_preflight(dataset: str, records: list[dict], req: pd.DataFrame, log: _Log) -> dict:
    root = _sita_root(dataset)
    existing = root / "preflight.json"
    if existing.is_file():
        payload = json.loads(existing.read_text(encoding="utf-8"))
        if payload.get("status") == f"READY_FOR_{dataset}_SITA_CANARY" and int(payload.get("n_targets_readable", 0)) == EXPECTED_TARGETS:
            log.write(f"{dataset} preflight already complete, skipping")
            return payload
    contract = verify_direct_horizon()
    log.write(f"{dataset} horizon contract {json.dumps(contract, sort_keys=True)}")
    calendar = calendar_for(dataset)
    frozen_sel = load_frozen_selection(dataset)
    histories = _load_histories(dataset, records, log)
    for horizon in HORIZONS:
        dates = build_evaluation_index(dataset, horizon).label_date_text()
        if len(dates) != TEST_DAYS or dates[0] != calendar.test_start or dates[-1] != calendar.test_end:
            raise SitaStop("BLOCKED_SITA_PROTOCOL", f"{dataset} evaluation index h={horizon}")
    readable = 0
    cache_hits = 0
    selection_rows = []
    for record in records:
        identity = str(record["target_identity"])
        frozen = _selection_row(frozen_sel, identity)
        target, _train, _validation, selection = _live_select(dataset, identity, histories=histories, frozen=frozen)
        del target
        readable += 1
        live = [item.identity for item in selection.selected]
        for source in live:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    row = _cache_lookup(req, source, horizon, seed)
                    status = _inspect_requirement(dataset, row)
                    if status != "CACHE_HIT":
                        raise SitaStop("HOLD_STAGE_A_CACHE_MISS", f"{dataset} {source} h={horizon} seed={seed} {status}")
                    cache_hits += 1
        selection_rows.append(
            {
                "target_identity": identity,
                "selected": live,
                "target_excluded": identity not in live,
                "matches_frozen_top3": True,
                "readable": True,
            }
        )
        log.write(f"{dataset} preflight {identity} top3={live} cache=45/45")
    payload = {
        "status": f"READY_FOR_{dataset}_SITA_CANARY",
        "dataset": dataset,
        "n_targets_readable": readable,
        "n_targets": EXPECTED_TARGETS,
        "cache_hits": cache_hits,
        "cache_miss": 0,
        "new_source_training": 0,
        "finished_at": _utc_now(),
    }
    if readable != EXPECTED_TARGETS or cache_hits != EXPECTED_TARGETS * 3 * 3 * 5:
        raise SitaStop("BLOCKED_SITA_CANARY", json.dumps(payload, sort_keys=True))
    _atomic_json(root / "preflight.json", payload)
    _atomic_csv(root / "preflight_selection.csv", pd.DataFrame(selection_rows))
    log.write(f"{dataset} preflight complete targets={readable} cache_hits={cache_hits}")
    return payload


def _cell_dir(dataset: str, identity: str, horizon: int, seed: int) -> Path:
    return _sita_root(dataset) / "cells" / identity.replace("/", "__") / f"h{int(horizon)}" / f"s{int(seed)}"


def _cell_key(dataset: str, identity: str, horizon: int, seed: int, protocol_sha: str, selected: list[str]) -> str:
    payload = {
        "dataset": dataset,
        "method": "SITA",
        "target_identity": identity,
        "horizon": int(horizon),
        "seed": int(seed),
        "selected_sources": selected,
        "manifest_sha256": _cfg(dataset)["manifest_sha256"],
        "protocol_sha256": protocol_sha,
        "n_test": TEST_DAYS,
        "source_training": "CACHE_HIT_ONLY",
    }
    return _payload_digest(payload)


def _inspect_cell(directory: Path, expected_key: str) -> str:
    done = directory / "DONE.json"
    pred = directory / "predictions.csv"
    if not done.is_file() or not pred.is_file():
        return "MISS"
    payload = json.loads(done.read_text(encoding="utf-8"))
    if payload.get("cell_key") != expected_key or payload.get("status") != "COMPLETED":
        return "HASH_MISMATCH"
    frame = pd.read_csv(pred)
    if len(frame) != TEST_DAYS or int(payload.get("n_test", 0)) != TEST_DAYS:
        return "N_TEST_MISMATCH"
    return "HIT"


def _progress_tables(dataset: str, records: list[dict]) -> None:
    root = _sita_root(dataset)
    frame = pd.DataFrame(records)
    _atomic_csv(root / "sita_progress.csv", frame)
    if frame.empty:
        _atomic_csv(root / "sita_metrics.csv", frame)
        return
    cols = [
        column
        for column in (
            "target_identity",
            "horizon",
            "seed",
            "n_test",
            "rmse",
            "smape",
            "status",
            "selected_sources",
            "cache_status",
            "canary_cell",
        )
        if column in frame.columns
    ]
    _atomic_csv(root / "sita_metrics.csv", frame.loc[:, cols])


def _load_progress(dataset: str) -> list[dict]:
    path = _sita_root(dataset) / "sita_progress.csv"
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict("records")


def _write_cell(
    dataset: str,
    directory: Path,
    *,
    cell_key: str,
    identity: str,
    horizon: int,
    seed: int,
    dates: tuple[str, ...],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    selected: list[str],
    usage: list[dict],
    protocol_sha: str,
    canary_cell: bool,
) -> dict:
    if y_true.shape != (TEST_DAYS,) or y_pred.shape != (TEST_DAYS,) or len(dates) != TEST_DAYS:
        raise SitaStop("BLOCKED_SITA_PROTOCOL", f"{dataset} {identity} h={horizon} n_test is not 180")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise SitaStop("BLOCKED_SITA_PROTOCOL", f"{dataset} {identity} non-finite forecast")
    cache_status = sorted({str(row.get("cache_status")) for row in usage})
    if cache_status != ["CACHE_HIT"]:
        raise SitaStop("HOLD_NEW_SOURCE_TRAINING", f"{identity} cache_status={cache_status}")
    pair = evaluate_original_sales(y_true, y_pred)
    tmp = directory.with_name(directory.name + ".partial")
    if tmp.exists():
        for child in sorted(tmp.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        tmp.rmdir()
    tmp.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(
        {
            "dataset": dataset,
            "target_identity": identity,
            "method": "SITA",
            "horizon": int(horizon),
            "seed": int(seed),
            "target_date": list(dates),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).to_csv(tmp / "predictions.csv", index=False)
    record = {
        "cell_key": cell_key,
        "status": "COMPLETED",
        "dataset": dataset,
        "method": "SITA",
        "target_identity": identity,
        "horizon": int(horizon),
        "seed": int(seed),
        "n_test": TEST_DAYS,
        "rmse": pair.rmse,
        "smape": pair.smape,
        "prediction_finite_count": int(np.isfinite(y_pred).sum()),
        "target_finite_count": int(np.isfinite(y_true).sum()),
        "first_target_date": dates[0],
        "last_target_date": dates[-1],
        "selected_sources": ",".join(selected),
        "cache_status": "CACHE_HIT",
        "new_source_training": False,
        "canary_cell": bool(canary_cell),
        "manifest_sha256": _cfg(dataset)["manifest_sha256"],
        "protocol_sha256": protocol_sha,
        "trained_at": _utc_now(),
    }
    (tmp / "DONE.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    (tmp / "source_usage.json").write_text(json.dumps(usage, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(directory)
    return record


def _recycle(dataset: str, phase: str, log: _Log, newly: int, leftover: int) -> None:
    if newly < RECYCLE_AFTER or leftover <= 0:
        return
    log.write(f"{dataset} recycling TensorFlow process after {newly} cells leftover={leftover} phase={phase}")
    os.execv(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), "--phase", phase, "--dataset", dataset],
    )


def run_cells(
    dataset: str,
    *,
    identities: list[str],
    records: list[dict],
    req: pd.DataFrame,
    seeds: tuple[int, ...],
    log: _Log,
    canary_cell: bool,
    phase: str,
) -> list[dict]:
    protocol_sha = frozen_protocol_sha256()
    frozen_sel = load_frozen_selection(dataset)
    histories = _load_histories(dataset, records, log)
    prepared: dict[str, dict] = {}
    progress = _load_progress(dataset)
    done = {
        (str(row.get("target_identity")), int(row.get("horizon")), int(row.get("seed")))
        for row in progress
        if str(row.get("status")) in {"COMPLETED", "SKIP"}
    }
    newly = 0
    expected = EXPECTED_CANARY if canary_cell else EXPECTED_FULL
    for identity in identities:
        frozen = _selection_row(frozen_sel, identity)
        if identity not in prepared:
            target, train, validation, selection = _live_select(dataset, identity, histories=histories, frozen=frozen)
            prepared[identity] = {
                "target": target,
                "train": train,
                "validation": validation,
                "selection": selection,
                "sales": sales_by_date(target),
                "selected": [item.identity for item in selection.selected],
            }
        item = prepared[identity]
        selected = item["selected"]
        for horizon in HORIZONS:
            index = build_evaluation_index(dataset, horizon)
            dates = index.label_date_text()
            y_true = np.asarray([item["sales"][day] for day in dates], dtype=np.float64)
            for seed in seeds:
                key = _cell_key(dataset, identity, horizon, seed, protocol_sha, selected)
                directory = _cell_dir(dataset, identity, horizon, seed)
                inspect = _inspect_cell(directory, key)
                tuple_key = (identity, int(horizon), int(seed))
                if inspect == "HIT" or tuple_key in done:
                    if inspect == "HIT" and tuple_key not in done:
                        payload = json.loads((directory / "DONE.json").read_text(encoding="utf-8"))
                        payload["status"] = "SKIP"
                        progress.append(payload)
                        done.add(tuple_key)
                    continue
                if inspect == "HASH_MISMATCH":
                    raise SitaStop("HOLD_HASH_MISMATCH", str(directory))
                if inspect == "N_TEST_MISMATCH":
                    raise SitaStop("BLOCKED_SITA_PROTOCOL", f"{directory} n_test")
                log.write(f"{dataset} adapt {identity} h={horizon} seed={seed} sources={selected}")
                y_pred, usage = _predict_sita(
                    item["target"],
                    item["train"],
                    item["validation"],
                    item["selection"],
                    index,
                    horizon=int(horizon),
                    seed=int(seed),
                    dataset=dataset,
                    identity=identity,
                    histories=histories,
                )
                record = _write_cell(
                    dataset,
                    directory,
                    cell_key=key,
                    identity=identity,
                    horizon=horizon,
                    seed=seed,
                    dates=dates,
                    y_true=y_true,
                    y_pred=np.asarray(y_pred, dtype=np.float64),
                    selected=selected,
                    usage=usage,
                    protocol_sha=protocol_sha,
                    canary_cell=canary_cell,
                )
                progress.append(record)
                done.add(tuple_key)
                newly += 1
                _progress_tables(dataset, progress)
                log.write(
                    f"{dataset} SITA {identity} h={horizon} seed={seed} n_test=180 "
                    f"rmse={record['rmse']:.6g} cache=CACHE_HIT"
                )
                leftover = expected - len([item_key for item_key in done if (not canary_cell) or item_key[0] in identities])
                _recycle(dataset, phase, log, newly, leftover)
    _progress_tables(dataset, progress)
    return progress


def run_canary(dataset: str, records: list[dict], req: pd.DataFrame, log: _Log) -> dict:
    canary_dir = _sita_root(dataset) / "canary"
    canary_dir.mkdir(parents=True, exist_ok=True)
    identity = _cfg(dataset)["canary_target"]
    log.write(f"{dataset} canary identity {identity} horizons={list(HORIZONS)} seed={CANARY_SEED}")
    progress = run_cells(
        dataset,
        identities=[identity],
        records=records,
        req=req,
        seeds=(CANARY_SEED,),
        log=log,
        canary_cell=True,
        phase="canary",
    )
    freeze = assert_freeze_unchanged(dataset, req)
    canary_keys = {(identity, int(horizon), CANARY_SEED) for horizon in HORIZONS}
    completed = [
        row
        for row in progress
        if (str(row.get("target_identity")), int(row.get("horizon")), int(row.get("seed"))) in canary_keys
        and str(row.get("status")) in {"COMPLETED", "SKIP"}
        and int(row.get("n_test", 0)) == TEST_DAYS
    ]
    n_test_ok = all(int(row.get("n_test", 0)) == TEST_DAYS for row in completed)
    cache_ok = all(str(row.get("cache_status")) == "CACHE_HIT" for row in completed)
    status = _cfg(dataset)["ready_canary"] if len(completed) == EXPECTED_CANARY and n_test_ok and cache_ok else "BLOCKED_SITA_CANARY"
    payload = {
        "status": status,
        "dataset": dataset,
        "n_canary_cells": len(completed),
        "expected_canary_cells": EXPECTED_CANARY,
        "identity": identity,
        "n_test_all_180": n_test_ok,
        "cache_hit_only": cache_ok,
        "new_source_training": 0,
        "freeze_required_files_sha256": freeze["required_files_sha256"],
        "finished_at": _utc_now(),
    }
    _atomic_csv(canary_dir / "canary_results.csv", pd.DataFrame(completed))
    _atomic_json(canary_dir / "DONE.json", payload)
    log.write(payload["status"])
    if payload["status"] != _cfg(dataset)["ready_canary"]:
        raise SitaStop("BLOCKED_SITA_CANARY", json.dumps(payload, sort_keys=True))
    return payload


def run_full(dataset: str, records: list[dict], req: pd.DataFrame, log: _Log) -> dict:
    canary_done = _sita_root(dataset) / "canary" / "DONE.json"
    if not canary_done.is_file():
        raise SitaStop("BLOCKED_SITA_CANARY", f"{dataset} canary DONE.json missing")
    canary = json.loads(canary_done.read_text(encoding="utf-8"))
    if canary.get("status") != _cfg(dataset)["ready_canary"]:
        raise SitaStop("BLOCKED_SITA_CANARY", str(canary.get("status")))
    identities = [str(item["target_identity"]) for item in records]
    progress = run_cells(
        dataset,
        identities=identities,
        records=records,
        req=req,
        seeds=SEEDS,
        log=log,
        canary_cell=False,
        phase="run",
    )
    freeze = assert_freeze_unchanged(dataset, req)
    ok = [
        row
        for row in progress
        if str(row.get("status")) in {"COMPLETED", "SKIP"} and int(row.get("n_test", 0)) == TEST_DAYS
    ]
    status = _cfg(dataset)["ready_full"] if len(ok) == EXPECTED_FULL else "SITA_PARTIAL_COMPLETE"
    payload = {
        "status": status,
        "dataset": dataset,
        "n_completed": len(ok),
        "expected": EXPECTED_FULL,
        "n_test": TEST_DAYS,
        "new_source_training": 0,
        "freeze_required_files_sha256": freeze["required_files_sha256"],
        "finished_at": _utc_now(),
    }
    _atomic_json(_sita_root(dataset) / "sita_full_status.json", payload)
    log.write(f"{payload['status']} completed={payload['n_completed']}/{EXPECTED_FULL}")
    return payload


def run_dataset(dataset: str, phase: str) -> int:
    root = _sita_root(dataset)
    root.mkdir(parents=True, exist_ok=True)
    log = _Log(root / "sita_run_log.txt")
    records = load_frozen_manifest(dataset)
    req = load_requirements(dataset)
    log.write(f"{dataset} phase={phase} n_targets={len(records)} required_keys={len(req)}")
    hold = root / "HOLD.json"
    if phase in {"canary", "run"} and hold.is_file():
        hold.unlink()
    if phase in {"freeze", "preflight", "canary", "all"}:
        freeze_artifacts(dataset, records, req, log)
    if phase in {"preflight", "canary", "all"}:
        run_preflight(dataset, records, req, log)
    if phase in {"canary", "all"}:
        run_canary(dataset, records, req, log)
    if phase == "run":
        run_full(dataset, records, req, log)
    if phase in {"audit", "run"}:
        freeze = assert_freeze_unchanged(dataset, req)
        log.write(f"{dataset} audit freeze unchanged files={freeze['required_files_sha256'][:12]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze", "preflight", "canary", "run", "audit", "all"), default="canary")
    parser.add_argument("--dataset", choices=("D5", "D6", "both"), default="D5")
    args = parser.parse_args()
    datasets = ("D5", "D6") if args.dataset == "both" else (args.dataset,)
    try:
        for dataset in datasets:
            run_dataset(dataset, args.phase)
        return 0
    except (SitaStop, ProtocolHold, ProtocolError) as exc:
        token = getattr(exc, "token", type(exc).__name__)
        detail = getattr(exc, "detail", str(exc))
        print(token, flush=True)
        for dataset in datasets:
            root = _sita_root(dataset)
            root.mkdir(parents=True, exist_ok=True)
            _Log(root / "sita_run_log.txt").write(f"{token}: {detail}")
            _atomic_json(root / "HOLD.json", {"status": token, "detail": detail, "written_at": _utc_now()})
        return 1
    except Exception as exc:
        print("BLOCKED_SITA_CANARY", flush=True)
        for dataset in datasets:
            root = _sita_root(dataset)
            root.mkdir(parents=True, exist_ok=True)
            _Log(root / "sita_run_log.txt").write(traceback.format_exc())
            _atomic_json(
                root / "HOLD.json",
                {"status": "BLOCKED_SITA_CANARY", "detail": f"{type(exc).__name__}: {exc}", "written_at": _utc_now()},
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
