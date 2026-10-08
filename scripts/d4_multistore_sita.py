#!/usr/bin/env python3
"""D4 60-target SITA canary and full run.

Freezes the existing 60-target manifest and Stage A source_model_cache.
Does not change source identities, cache keys, hashes, or the training protocol.
Does not train new source CNNs. Does not rewrite additional_target_robustness
runners, additional_targets_manifest, or source.parquet.
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
from additional_target_robustness.runner import (
    _one_source,
    _predict_sita,
    _read_source_record,
    _series_sha256,
)
from additional_target_robustness.source_selection import load_source_histories, select_top3
from additional_target_robustness.target_data import load_target_series, sales_by_date, split_target_frame
import additional_target_robustness.runner as sita_runner

ensure_ablation_path()
from sita_ablation.metrics import evaluate_original_sales

OUT_DIR = REPO_ROOT / "outputs" / "d4_multistore_60target_robustness_20261004"
MANIFEST_PATH = OUT_DIR / "d4_60target_manifest.json"
MANIFEST_SHA256 = "1a4ad685b60563d3e778c3d656535fa5c101e4cd1c041c0dc9b4e569468fea84"
PROTOCOL_SHA256 = "405beca0c84a85d3bdd77016cda37d5c2744ca7d4bc863e5c9d77ab17cd7788c"
SOURCE_CALENDAR_SHA256 = "17a1fa5bd1dddfd46bda2a6922ff7821aee2a7e79deca58a94ff7bf20821f7ef"
SELECTION_PATH = OUT_DIR / "source_selection_results.csv"
REQ_PATH = OUT_DIR / "stage_a_requirements.csv"
SITA_ROOT = OUT_DIR / "sita"
FREEZE_PATH = SITA_ROOT / "freeze_lock.json"
FREEZE_INDEX_PATH = SITA_ROOT / "freeze_cache_index.csv"
CANARY_DIR = SITA_ROOT / "canary"
STORES = ("155", "274", "576", "166", "139", "161")
EXPECTED_TARGETS = 60
EXPECTED_REQUIRED_KEYS = 1335
EXPECTED_CANARY = 18
EXPECTED_FULL = 900
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
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _in_child_process() -> bool:
    return os.environ.get("D4_SITA_CHILD") == "1"


def load_frozen_manifest() -> list[dict]:
    actual = sha256_file(MANIFEST_PATH)
    if actual != MANIFEST_SHA256:
        raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", f"manifest sha256 {actual}")
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records = list(payload["records"])
    if len(records) != EXPECTED_TARGETS:
        raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", f"n_targets={len(records)}")
    if str(payload.get("source_calendar_sha256")) != SOURCE_CALENDAR_SHA256:
        raise SitaStop("HOLD_SOURCE_CACHE_HASH_MISMATCH", "source calendar sha256 drifted")
    identities = [str(item["target_identity"]) for item in records]
    if len(set(identities)) != EXPECTED_TARGETS:
        raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", "duplicate targets")
    stores = [str(item["store_id"]) for item in records]
    if set(stores) != set(STORES):
        raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", f"stores={sorted(set(stores))}")
    counts = {store: stores.count(store) for store in STORES}
    if any(count != 10 for count in counts.values()):
        raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", f"per-store counts {counts}")
    return records


def load_frozen_selection() -> pd.DataFrame:
    if not SELECTION_PATH.is_file():
        raise SitaStop("HOLD_SOURCE_SELECTION_DRIFT", "missing source_selection_results.csv")
    frame = pd.read_csv(SELECTION_PATH)
    if len(frame) != EXPECTED_TARGETS:
        raise SitaStop("HOLD_SOURCE_SELECTION_DRIFT", f"n_selection={len(frame)}")
    return frame


def load_requirements() -> pd.DataFrame:
    if not REQ_PATH.is_file():
        raise SitaStop("HOLD_STAGE_A_CACHE_MISS", "missing stage_a_requirements.csv")
    frame = pd.read_csv(REQ_PATH)
    if len(frame) != EXPECTED_REQUIRED_KEYS:
        raise SitaStop("HOLD_STAGE_A_CACHE_MISS", f"required_keys={len(frame)}")
    return frame


def canary_identities(records: list[dict]) -> list[str]:
    chosen = []
    for store in STORES:
        matches = [
            item for item in records if str(item["store_id"]) == store and int(item["target_rank_within_store"]) == 1
        ]
        if len(matches) != 1:
            raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", f"rank-1 missing for store {store}")
        chosen.append(str(matches[0]["target_identity"]))
    if len(chosen) != 6:
        raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", f"canary identities {chosen}")
    return chosen


def _cache_children() -> list[str]:
    if not SOURCE_MODEL_CACHE.is_dir():
        raise SitaStop("HOLD_STAGE_A_CACHE_MISS", "source_model_cache missing")
    return sorted(path.name for path in SOURCE_MODEL_CACHE.iterdir() if path.is_dir())


def _snapshot_required(req: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in req.sort_values("cache_key").iterrows():
        key = str(row["cache_key"])
        directory = SOURCE_MODEL_CACHE / key
        done = directory / "DONE.json"
        key_path = directory / "key.json"
        weights = directory / "weights.npz"
        if not (done.is_file() and key_path.is_file() and weights.is_file()):
            raise SitaStop("HOLD_STAGE_A_CACHE_MISS", key)
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


def _merkle(frame: pd.DataFrame, children: list[str]) -> dict[str, str]:
    hasher = hashlib.sha256()
    for _, row in frame.sort_values("cache_key").iterrows():
        hasher.update(
            f"{row['cache_key']}\t{row['done_sha256']}\t{row['key_sha256']}\t{row['weights_sha256']}\n".encode("utf-8")
        )
    return {
        "n_cache_children": len(children),
        "cache_children_sha256": hashlib.sha256("\n".join(children).encode("utf-8")).hexdigest(),
        "required_files_sha256": hasher.hexdigest(),
        "n_required": int(len(frame)),
    }


def freeze_artifacts(records: list[dict], req: pd.DataFrame, log: _Log) -> dict:
    SITA_ROOT.mkdir(parents=True, exist_ok=True)
    protocol_sha = frozen_protocol_sha256()
    if protocol_sha != PROTOCOL_SHA256:
        raise SitaStop("HOLD_HASH_MISMATCH", f"protocol sha256 {protocol_sha}")
    children = _cache_children()
    index = _snapshot_required(req)
    merkle = _merkle(index, children)
    payload = {
        "frozen_at": _utc_now(),
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": MANIFEST_SHA256,
        "source_calendar_sha256": SOURCE_CALENDAR_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_model_cache": str(SOURCE_MODEL_CACHE.relative_to(REPO_ROOT)),
        "n_targets": len(records),
        "n_required_stage_a_keys": int(len(req)),
        "no_source_identity_change": True,
        "no_cache_key_change": True,
        "no_hash_change": True,
        "no_training_protocol_change": True,
        **merkle,
    }
    if FREEZE_PATH.is_file():
        stored = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        comparable = {key: stored.get(key) for key in ("manifest_sha256", "protocol_sha256", "required_files_sha256", "cache_children_sha256", "n_required", "n_cache_children")}
        expected = {key: payload[key] for key in comparable}
        if comparable != expected:
            raise SitaStop("HOLD_SOURCE_CACHE_MUTATED", json.dumps({"stored": comparable, "live": expected}, sort_keys=True))
        log.write("freeze lock already present and unchanged")
        return stored
    _atomic_csv(FREEZE_INDEX_PATH, index)
    _atomic_json(FREEZE_PATH, payload)
    log.write(
        f"froze manifest={MANIFEST_SHA256[:12]} cache_children={merkle['n_cache_children']} "
        f"required={merkle['n_required']} files={merkle['required_files_sha256'][:12]}"
    )
    return payload


def assert_freeze_unchanged(req: pd.DataFrame) -> dict:
    if not FREEZE_PATH.is_file():
        raise SitaStop("HOLD_SOURCE_CACHE_MUTATED", "freeze_lock.json missing")
    stored = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    children = _cache_children()
    index = _snapshot_required(req)
    merkle = _merkle(index, children)
    for field in ("required_files_sha256", "cache_children_sha256", "n_required", "n_cache_children"):
        if stored.get(field) != merkle[field]:
            raise SitaStop("HOLD_SOURCE_CACHE_MUTATED", f"{field} drifted")
    if stored.get("manifest_sha256") != MANIFEST_SHA256:
        raise SitaStop("HOLD_TARGET_MANIFEST_MISMATCH", "freeze manifest drifted")
    return stored


def _selection_row(frame: pd.DataFrame, identity: str) -> pd.Series:
    rows = frame.loc[frame["target_identity"].astype(str) == identity]
    if len(rows) != 1:
        raise SitaStop("HOLD_SOURCE_SELECTION_DRIFT", identity)
    return rows.iloc[0]


def _live_select(identity: str, *, histories_by_store: dict[str, pd.DataFrame], frozen: pd.Series):
    calendar = calendar_for("D4")
    target = load_target_series("D4", identity)
    train, validation, test = split_target_frame(target, calendar)
    if len(train) != 15 or len(validation) != 15 or len(test) != TEST_DAYS:
        raise SitaStop("BLOCKED_SITA_PROTOCOL", f"{identity} partition lengths")
    store = identity.split("/")[0]
    histories = histories_by_store[store]
    selection = select_top3(
        "D4",
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
        raise SitaStop("HOLD_SOURCE_SELECTION_DRIFT", f"{identity} cross-store {live}")
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


def _inspect_requirement(row: pd.Series) -> str:
    directory = SOURCE_MODEL_CACHE / str(row["cache_key"])
    done_path = directory / "DONE.json"
    key_path = directory / "key.json"
    weights_path = directory / "weights.npz"
    if not (done_path.is_file() and key_path.is_file() and weights_path.is_file()):
        return "CACHE_MISS"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": "D4",
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
    if dataset != "D4":
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
        raise SitaStop("HOLD_STAGE_A_CACHE_MISS", f"{identity} h={horizon} seed={seed} key={digest}")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("cache_key_sha256") != digest or done.get("status") != "COMPLETED":
        raise SitaStop("HOLD_HASH_MISMATCH", digest)
    key_path = directory / "key.json"
    stored = json.loads(key_path.read_text(encoding="utf-8"))
    if stored != payload:
        raise SitaStop("HOLD_HASH_MISMATCH", f"{digest} key.json drifted")
    return _read_source_record(directory, source_key), "CACHE_HIT"


sita_runner._load_or_train_source = _load_source_hit_only


def _histories_by_store(records: list[dict], log: _Log) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for store in STORES:
        identity = next(str(item["target_identity"]) for item in records if str(item["store_id"]) == store)
        log.write(f"loading sealed source history for store {store}")
        target = load_target_series("D4", identity)
        out[store] = load_source_histories("D4", target_identity=identity, target_frame=target)
    return out


def run_preflight(records: list[dict], req: pd.DataFrame, log: _Log) -> dict:
    existing = SITA_ROOT / "preflight.json"
    if existing.is_file():
        payload = json.loads(existing.read_text(encoding="utf-8"))
        if payload.get("status") == "READY_FOR_D4_60TARGET_SITA_CANARY" and int(payload.get("n_targets_readable", 0)) == EXPECTED_TARGETS:
            log.write("preflight already complete, skipping")
            return payload
    contract = verify_direct_horizon()
    log.write(f"horizon contract {json.dumps(contract, sort_keys=True)}")
    calendar = calendar_for("D4")
    frozen_sel = load_frozen_selection()
    histories = _histories_by_store(records, log)
    indexes = {horizon: build_evaluation_index("D4", horizon) for horizon in HORIZONS}
    for horizon, index in indexes.items():
        dates = index.label_date_text()
        if len(dates) != TEST_DAYS or dates[0] != calendar.test_start or dates[-1] != calendar.test_end:
            raise SitaStop("BLOCKED_SITA_PROTOCOL", f"evaluation index h={horizon}")
    readable = 0
    cache_hits = 0
    cache_lookups = 0
    selection_rows = []
    for record in records:
        identity = str(record["target_identity"])
        frozen = _selection_row(frozen_sel, identity)
        target, _train, _validation, selection = _live_select(
            identity, histories_by_store=histories, frozen=frozen
        )
        del target
        readable += 1
        live = [item.identity for item in selection.selected]
        for source in live:
            for horizon in HORIZONS:
                for seed in SEEDS:
                    cache_lookups += 1
                    row = _cache_lookup(req, source, horizon, seed)
                    status = _inspect_requirement(row)
                    if status != "CACHE_HIT":
                        raise SitaStop("HOLD_STAGE_A_CACHE_MISS", f"{source} h={horizon} seed={seed} {status}")
                    cache_hits += 1
        selection_rows.append(
            {
                "target_identity": identity,
                "store_id": str(record["store_id"]),
                "selected": live,
                "target_excluded": identity not in live,
                "matches_frozen_top3": True,
                "readable": True,
            }
        )
        log.write(f"preflight {identity} top3={live} target_excluded=True cache=45/45")
    payload = {
        "status": "READY_FOR_D4_60TARGET_SITA_CANARY",
        "n_targets_readable": readable,
        "n_targets": EXPECTED_TARGETS,
        "n_top3_matched": readable,
        "target_self_excluded": True,
        "cache_lookups": cache_lookups,
        "cache_hits": cache_hits,
        "cache_miss": 0,
        "new_source_training": 0,
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "finished_at": _utc_now(),
    }
    if readable != EXPECTED_TARGETS or cache_hits != EXPECTED_TARGETS * 3 * 3 * 5:
        raise SitaStop("BLOCKED_SITA_CANARY", json.dumps(payload, sort_keys=True))
    _atomic_json(SITA_ROOT / "preflight.json", payload)
    _atomic_csv(SITA_ROOT / "preflight_selection.csv", pd.DataFrame(selection_rows))
    log.write(f"preflight complete targets={readable} cache_hits={cache_hits}")
    return payload


def _cell_dir(identity: str, horizon: int, seed: int) -> Path:
    return SITA_ROOT / "cells" / identity.replace("/", "__") / f"h{int(horizon)}" / f"s{int(seed)}"


def _cell_key(identity: str, horizon: int, seed: int, protocol_sha: str, selected: list[str]) -> str:
    payload = {
        "dataset": "D4",
        "method": "SITA",
        "target_identity": identity,
        "horizon": int(horizon),
        "seed": int(seed),
        "selected_sources": selected,
        "manifest_sha256": MANIFEST_SHA256,
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
    if len(frame) != TEST_DAYS:
        return "N_TEST_MISMATCH"
    if int(payload.get("n_test", 0)) != TEST_DAYS:
        return "N_TEST_MISMATCH"
    return "HIT"


def _progress_tables(records: list[dict]) -> None:
    frame = pd.DataFrame(records)
    if frame.empty:
        _atomic_csv(SITA_ROOT / "sita_progress.csv", frame)
        _atomic_csv(SITA_ROOT / "sita_metrics.csv", frame)
        return
    _atomic_csv(SITA_ROOT / "sita_progress.csv", frame)
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
    _atomic_csv(SITA_ROOT / "sita_metrics.csv", frame.loc[:, cols])


def _load_progress() -> list[dict]:
    path = SITA_ROOT / "sita_progress.csv"
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict("records")


def _write_cell(
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
        raise SitaStop("BLOCKED_SITA_PROTOCOL", f"{identity} h={horizon} n_test is not 180")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise SitaStop("BLOCKED_SITA_PROTOCOL", f"{identity} non-finite forecast")
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
            "dataset": "D4",
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
        "dataset": "D4",
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
        "manifest_sha256": MANIFEST_SHA256,
        "protocol_sha256": protocol_sha,
        "trained_at": _utc_now(),
    }
    (tmp / "DONE.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    (tmp / "source_usage.json").write_text(json.dumps(usage, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(directory)
    return record


def _recycle(phase: str, log: _Log, newly: int, leftover: int) -> None:
    if newly < RECYCLE_AFTER or leftover <= 0:
        return
    log.write(f"recycling TensorFlow process after {newly} cells, leftover={leftover} phase={phase}")
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), "--phase", phase])


def run_cells(
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
    frozen_sel = load_frozen_selection()
    histories = _histories_by_store(records, log)
    prepared: dict[str, dict] = {}
    progress = _load_progress()
    done = {
        (str(row.get("target_identity")), int(row.get("horizon")), int(row.get("seed")))
        for row in progress
        if str(row.get("status")) in {"COMPLETED", "SKIP"}
    }
    newly = 0
    remaining_idents = list(identities)
    for identity in remaining_idents:
        frozen = _selection_row(frozen_sel, identity)
        if identity not in prepared:
            target, train, validation, selection = _live_select(
                identity, histories_by_store=histories, frozen=frozen
            )
            prepared[identity] = {
                "target": target,
                "train": train,
                "validation": validation,
                "selection": selection,
                "sales": sales_by_date(target),
                "histories": histories[identity.split("/")[0]],
                "selected": [item.identity for item in selection.selected],
            }
        item = prepared[identity]
        selected = item["selected"]
        for horizon in HORIZONS:
            index = build_evaluation_index("D4", horizon)
            dates = index.label_date_text()
            y_true = np.asarray([item["sales"][day] for day in dates], dtype=np.float64)
            for seed in seeds:
                key = _cell_key(identity, horizon, seed, protocol_sha, selected)
                directory = _cell_dir(identity, horizon, seed)
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
                log.write(f"adapt {identity} h={horizon} seed={seed} sources={selected}")
                y_pred, usage = _predict_sita(
                    item["target"],
                    item["train"],
                    item["validation"],
                    item["selection"],
                    index,
                    horizon=int(horizon),
                    seed=int(seed),
                    dataset="D4",
                    identity=identity,
                    histories=item["histories"],
                )
                record = _write_cell(
                    directory,
                    cell_key=key,
                    identity=identity,
                    horizon=int(horizon),
                    seed=int(seed),
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
                _progress_tables(progress)
                log.write(
                    f"SITA {identity} h={horizon} seed={seed} n_test=180 "
                    f"rmse={record['rmse']:.6g} cache=CACHE_HIT"
                )
                leftover = EXPECTED_CANARY - len(done) if canary_cell else EXPECTED_FULL - len(done)
                _recycle(phase, log, newly, leftover)
    _progress_tables(progress)
    return progress


def run_canary(records: list[dict], req: pd.DataFrame, log: _Log) -> dict:
    CANARY_DIR.mkdir(parents=True, exist_ok=True)
    identities = canary_identities(records)
    log.write(f"canary identities {identities} horizons={list(HORIZONS)} seed={CANARY_SEED}")
    progress = run_cells(
        identities=identities,
        records=records,
        req=req,
        seeds=(CANARY_SEED,),
        log=log,
        canary_cell=True,
        phase="canary",
    )
    freeze = assert_freeze_unchanged(req)
    canary_keys = {
        (identity, int(horizon), CANARY_SEED) for identity in identities for horizon in HORIZONS
    }
    completed = [
        row
        for row in progress
        if (str(row.get("target_identity")), int(row.get("horizon")), int(row.get("seed"))) in canary_keys
        and str(row.get("status")) in {"COMPLETED", "SKIP"}
        and int(row.get("n_test", 0)) == TEST_DAYS
    ]
    n_test_ok = all(int(row.get("n_test", 0)) == TEST_DAYS for row in completed)
    cache_ok = all(str(row.get("cache_status")) == "CACHE_HIT" for row in completed)
    payload = {
        "status": "READY_FOR_D4_60TARGET_SITA_FULL" if len(completed) == EXPECTED_CANARY and n_test_ok and cache_ok else "BLOCKED_SITA_CANARY",
        "n_canary_cells": len(completed),
        "expected_canary_cells": EXPECTED_CANARY,
        "identities": identities,
        "horizons": list(HORIZONS),
        "seed": CANARY_SEED,
        "n_test": TEST_DAYS,
        "n_test_all_180": n_test_ok,
        "cache_hit_only": cache_ok,
        "new_source_training": 0,
        "freeze_required_files_sha256": freeze["required_files_sha256"],
        "finished_at": _utc_now(),
    }
    _atomic_csv(CANARY_DIR / "canary_results.csv", pd.DataFrame(completed))
    _atomic_json(CANARY_DIR / "DONE.json", payload)
    log.write(payload["status"])
    if payload["status"] != "READY_FOR_D4_60TARGET_SITA_FULL":
        raise SitaStop("BLOCKED_SITA_CANARY", json.dumps(payload, sort_keys=True))
    return payload


def run_full(records: list[dict], req: pd.DataFrame, log: _Log) -> dict:
    canary_done = CANARY_DIR / "DONE.json"
    if not canary_done.is_file():
        raise SitaStop("BLOCKED_SITA_CANARY", "canary DONE.json missing")
    canary = json.loads(canary_done.read_text(encoding="utf-8"))
    if canary.get("status") != "READY_FOR_D4_60TARGET_SITA_FULL":
        raise SitaStop("BLOCKED_SITA_CANARY", str(canary.get("status")))
    identities = [str(item["target_identity"]) for item in records]
    progress = run_cells(
        identities=identities,
        records=records,
        req=req,
        seeds=SEEDS,
        log=log,
        canary_cell=False,
        phase="run",
    )
    freeze = assert_freeze_unchanged(req)
    ok = [
        row
        for row in progress
        if str(row.get("status")) in {"COMPLETED", "SKIP"} and int(row.get("n_test", 0)) == TEST_DAYS
    ]
    payload = {
        "status": "D4_60TARGET_SITA_FULL_COMPLETE" if len(ok) == EXPECTED_FULL else "SITA_PARTIAL_COMPLETE",
        "n_completed": len(ok),
        "expected": EXPECTED_FULL,
        "n_test": TEST_DAYS,
        "new_source_training": 0,
        "freeze_required_files_sha256": freeze["required_files_sha256"],
        "finished_at": _utc_now(),
    }
    _atomic_json(SITA_ROOT / "sita_full_status.json", payload)
    log.write(f"{payload['status']} completed={payload['n_completed']}/{EXPECTED_FULL}")
    return payload


def _write_fatal(token: str, detail: str) -> None:
    _atomic_json(SITA_ROOT / "HOLD.json", {"status": token, "detail": detail, "written_at": _utc_now()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze", "preflight", "canary", "run", "audit", "all"), default="canary")
    args = parser.parse_args()
    SITA_ROOT.mkdir(parents=True, exist_ok=True)
    log = _Log(SITA_ROOT / "sita_run_log.txt")
    try:
        records = load_frozen_manifest()
        req = load_requirements()
        log.write(f"phase={args.phase} n_targets={len(records)} required_keys={len(req)}")
        hold = SITA_ROOT / "HOLD.json"
        if args.phase in {"canary", "run"} and hold.is_file():
            hold.unlink()
        if args.phase in {"freeze", "preflight", "canary", "all"}:
            freeze_artifacts(records, req, log)
        if args.phase in {"preflight", "canary", "all"}:
            run_preflight(records, req, log)
        if args.phase in {"canary", "all"}:
            run_canary(records, req, log)
        if args.phase == "run":
            run_full(records, req, log)
        if args.phase in {"audit", "run"}:
            freeze = assert_freeze_unchanged(req)
            log.write(f"audit freeze unchanged files={freeze['required_files_sha256'][:12]}")
        return 0
    except (SitaStop, ProtocolHold, ProtocolError) as exc:
        token = getattr(exc, "token", type(exc).__name__)
        detail = getattr(exc, "detail", str(exc))
        log.write(f"{token}: {detail}")
        _write_fatal(token, detail)
        print(token, flush=True)
        return 1
    except Exception as exc:
        log.write(traceback.format_exc())
        _write_fatal("BLOCKED_SITA_CANARY", f"{type(exc).__name__}: {exc}")
        print("BLOCKED_SITA_CANARY", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
