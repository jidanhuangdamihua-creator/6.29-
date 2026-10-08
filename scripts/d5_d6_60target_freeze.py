#!/usr/bin/env python3
"""Freeze D5 and D6 new 60-target robustness manifests and audit Stage A cache.

This round samples 60 brand-new D5 targets and 60 brand-new D6 targets,
runs Top-3 source selection, and writes Stage A requirements. It does not
train source CNNs, run SITA, or run baselines.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from additional_target_robustness.paths import SOURCE_MODEL_CACHE, ensure_ablation_path
from additional_target_robustness.protocol import (
    HORIZONS,
    PAPER_TARGETS,
    SEEDS,
    TEST_DAYS,
    TRAIN_DAYS,
    VAL_DAYS,
    ProtocolError,
    ProtocolHold,
    build_evaluation_index,
    calendar_for,
    frozen_protocol_sha256,
    sha256_file,
)
from additional_target_robustness.runner import _series_sha256
from additional_target_robustness.source_selection import (
    _comparison_matrix,
    _component,
    _exact_history,
    _identity_key,
    load_source_histories,
    select_top3,
)
from additional_target_robustness.target_data import (
    load_target_series,
    source_cache_path,
    split_target_frame,
    target_authority_path,
)

ensure_ablation_path()
from sita_ablation.scaling import fit_selection_scaler, transform_selection
from sita_ablation.source_selection import STATUS_COMPLETED, STATUS_SKIPPED, anchored_rank, inverse_distance_weights

BASE_SEED = 20261004
PROTOCOL_VERSION = {
    "D5": "d5_60target_v1",
    "D6": "d6_60target_v1",
}
OUT_DIRS = {
    "D5": REPO_ROOT / "outputs" / "d5_60target_robustness_20261004",
    "D6": REPO_ROOT / "outputs" / "d6_60target_robustness_20261004",
}
GROUP_VALUES = {"D5": "GROCERY I", "D6": "FOODS_3"}
STORE_VALUES = {"D5": "48", "D6": "CA_1"}
EXPECTED_SOURCE_SHA = {
    "D5": "0cbbda5ae42e2e962e7fd83a4b57a10d4fc0fdafc2687177bee0841e6d40e506",
    "D6": "12297d9b00a8e40fd9f13966e41aa558fe15e00079f8d0b23477bf4e7c288458",
}
EXPECTED_UNIVERSE = {
    "D5": {
        "id": "755448b5dc59eb89",
        "hash": "755448b5dc59eb899a3baab0fb1a91812ebb8aac1535dce9d54b0508c4546808",
        "legal_sources": 1054,
        "eligible_targets": 115,
    },
    "D6": {
        "id": "832c50ec32b11c8f",
        "hash": "832c50ec32b11c8fadd56fca6ef9bc4e9035f8727b7778680bfb10f0db8a4ea4",
        "legal_sources": 818,
        "eligible_targets": 816,
    },
}
OLD_ADDITIONAL = {
    "D5": {
        "48/129296",
        "48/1091365",
        "48/575863",
        "48/1936589",
        "48/2010902",
        "48/162066",
        "48/261052",
        "48/1354380",
        "48/864510",
        "48/567623",
    },
    "D6": {
        "CA_1/FOODS_3_181",
        "CA_1/FOODS_3_057",
        "CA_1/FOODS_3_784",
        "CA_1/FOODS_3_314",
        "CA_1/FOODS_3_407",
        "CA_1/FOODS_3_284",
        "CA_1/FOODS_3_540",
        "CA_1/FOODS_3_198",
        "CA_1/FOODS_3_773",
        "CA_1/FOODS_3_733",
    },
}


class FreezeStop(RuntimeError):
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


def derived_seed(dataset: str) -> int:
    text = f"target_sampling_seed={BASE_SEED}|dataset={dataset}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % (2**31)


def _identity_sort_key(dataset: str, identity: str) -> tuple:
    store, item = identity.split("/", 1)
    if dataset == "D5":
        return (int(store), int(item))
    return (store, item)


def _ranking_name(identity: str) -> str:
    return identity.replace("/", "_") + ".parquet"


def _blocked_dataset(dataset: str) -> str:
    return "BLOCKED_D5_TARGET_OR_SELECTION" if dataset == "D5" else "BLOCKED_D6_TARGET_OR_SELECTION"


def _assert_horizon_independent() -> None:
    parameters = inspect.signature(select_top3).parameters
    if "horizon" in parameters:
        raise FreezeStop("HOLD_HORIZON_DEPENDENT_SELECTION_FOUND", "select_top3 accepts horizon")
    names = set(select_top3.__code__.co_varnames)
    if "horizon" in names:
        raise FreezeStop("HOLD_HORIZON_DEPENDENT_SELECTION_FOUND", "select_top3 uses horizon")


def _read_scoped_table(dataset: str, *, source: bool) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq

    calendar = calendar_for(dataset)
    path = source_cache_path(dataset) if source else target_authority_path(dataset)
    schema = pq.read_schema(path)
    columns = ["date", *calendar.key_fields, *calendar.model_features, *calendar.similarity_variables]
    if calendar.group_field is not None:
        columns.append(calendar.group_field)
    columns = list(dict.fromkeys(columns))
    missing = [column for column in columns if column not in schema.names]
    if missing:
        raise FreezeStop("BLOCKED_PROTOCOL_MISMATCH", f"{path.name} missing {missing}")
    store_field = calendar.key_fields[0]
    store_value = STORE_VALUES[dataset]
    arrow_type = schema.field(store_field).type
    store_scalar = pa.scalar(int(store_value), type=arrow_type) if pa.types.is_integer(arrow_type) else store_value
    date_min = calendar.history_start if source else calendar.train_start
    date_max = calendar.history_end if source else ("2017-08-14" if dataset == "D5" else calendar.test_end)
    filters = [
        (store_field, "==", store_scalar),
        (calendar.group_field, "==", GROUP_VALUES[dataset]),
        ("date", ">=", pa.scalar(np.datetime64(date_min, "ns"))),
        ("date", "<=", pa.scalar(np.datetime64(date_max, "ns"))),
    ]
    frame = pq.read_table(path, columns=columns, filters=filters).to_pandas()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if dataset == "D5" and not source:
        cutoff = pd.Timestamp("2017-08-14")
        banned = pd.Timestamp("2017-08-15")
        if (frame["date"] > cutoff).any() or (frame["date"] == banned).any():
            raise FreezeStop("HOLD_TARGET_ELIGIBILITY_CHANGED", "D5 target table contains dates after 2017-08-14")
        frame = frame.loc[frame["date"] <= cutoff].copy()
    return frame.reset_index(drop=True)


def _identity_of(key: tuple[str, ...]) -> str:
    return "/".join(key)


def _key_of_rowgroup(key_values: object, key_fields: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(key_values, tuple):
        key_values = (key_values,)
    if len(key_values) != len(key_fields):
        raise FreezeStop("BLOCKED_PROTOCOL_MISMATCH", f"key {key_values!r} != {key_fields!r}")
    return tuple(_component(value) for value in key_values)


def _legal_source_catalog(dataset: str, source_frame: pd.DataFrame) -> dict:
    calendar = calendar_for(dataset)
    source = source_frame.copy()
    source["date"] = pd.to_datetime(source["date"]).dt.normalize()
    history_start = pd.Timestamp(calendar.history_start)
    history_end = pd.Timestamp(calendar.history_end)
    catalog: dict[str, dict] = {}
    for key_values, group in source.groupby(list(calendar.key_fields), sort=False, dropna=False):
        key = _key_of_rowgroup(key_values, calendar.key_fields)
        identity = _identity_of(key)
        ordered = group.sort_values("date", kind="mergesort")
        ordered = ordered.loc[(ordered["date"] >= history_start) & (ordered["date"] <= history_end)].reset_index(drop=True)
        if ordered.empty or ordered["date"].duplicated().any():
            continue
        exact = bool(_exact_history(ordered, calendar))
        window = _comparison_matrix(dataset, ordered, calendar, required=False) if exact else None
        catalog[identity] = {
            "key": key,
            "series": ordered,
            "exact_180": exact,
            "comparison": window,
            "legal": window is not None and exact,
        }
    identities = sorted(catalog, key=lambda item: catalog[item]["key"])
    legal_identities = [item for item in identities if catalog[item]["legal"]]
    return {
        "catalog": catalog,
        "identities": identities,
        "legal_identities": legal_identities,
        "legal_keys": [catalog[item]["key"] for item in legal_identities],
        "legal_windows": [catalog[item]["comparison"] for item in legal_identities],
    }


def _index_targets(dataset: str, target_frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    calendar = calendar_for(dataset)
    indexed: dict[str, pd.DataFrame] = {}
    for key_values, group in target_frame.groupby(list(calendar.key_fields), sort=False, dropna=False):
        key = _key_of_rowgroup(key_values, calendar.key_fields)
        ordered = group.sort_values("date", kind="mergesort").drop_duplicates("date").reset_index(drop=True)
        indexed[_identity_of(key)] = ordered
    return indexed


def _target_is_eligible(dataset: str, identity: str, series: pd.DataFrame, catalog: dict) -> bool:
    calendar = calendar_for(dataset)
    expected = pd.date_range(calendar.train_start, calendar.test_end, freq="D")
    dates = series["date"].reset_index(drop=True)
    if len(dates) != 210 or not dates.equals(pd.Series(expected)):
        return False
    if series["date"].duplicated().any():
        return False
    for column in calendar.model_features:
        values = pd.to_numeric(series[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            return False
    window = _comparison_matrix(dataset, series, calendar, required=False)
    if window is None:
        return False
    legal_after = sum(1 for item in catalog["legal_identities"] if item != identity)
    if legal_after < 3:
        return False
    group_values = sorted({_component(value) for value in series[calendar.group_field].tolist() if _component(value)})
    if group_values != [GROUP_VALUES[dataset]]:
        return False
    return True


def _assert_target(dataset: str, identity: str, series: pd.DataFrame, catalog: dict) -> pd.DataFrame:
    calendar = calendar_for(dataset)
    frame = series.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if dataset == "D5":
        cutoff = pd.Timestamp("2017-08-14")
        banned = pd.Timestamp("2017-08-15")
        frame = frame.loc[frame["date"] <= cutoff].copy()
        if (frame["date"] == banned).any() or (frame["date"] > cutoff).any():
            raise FreezeStop("HOLD_TARGET_ELIGIBILITY_CHANGED", f"{dataset} {identity} has dates after 2017-08-14")
    for column in calendar.model_features:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise FreezeStop("HOLD_TARGET_ELIGIBILITY_CHANGED", f"{dataset} {identity} non-finite {column}")
    try:
        train, validation, test = split_target_frame(frame, calendar)
    except ProtocolError as exc:
        raise FreezeStop("HOLD_TARGET_ELIGIBILITY_CHANGED", f"{dataset} {identity}: {exc}") from exc
    if len(train) != TRAIN_DAYS or len(validation) != VAL_DAYS or len(test) != TEST_DAYS:
        raise FreezeStop("HOLD_TARGET_ELIGIBILITY_CHANGED", f"{dataset} {identity} partition lengths")
    if frame["date"].duplicated().any():
        raise FreezeStop("HOLD_TARGET_ELIGIBILITY_CHANGED", f"{dataset} {identity} duplicate dates")
    for horizon in HORIZONS:
        index = build_evaluation_index(dataset, horizon)
        dates = index.label_date_text()
        if len(dates) != TEST_DAYS or dates[0] != calendar.test_start or dates[-1] != calendar.test_end:
            raise FreezeStop("HOLD_TARGET_ELIGIBILITY_CHANGED", f"{dataset} {identity} h={horizon}")
    legal_after = sum(1 for item in catalog["legal_identities"] if item != identity)
    if legal_after < 3:
        raise FreezeStop("HOLD_TARGET_ELIGIBILITY_CHANGED", f"{dataset} {identity} <3 legal sources")
    group_values = sorted({_component(value) for value in frame[calendar.group_field].tolist() if _component(value)})
    if group_values != [GROUP_VALUES[dataset]]:
        raise FreezeStop("HOLD_TARGET_ELIGIBILITY_CHANGED", f"{dataset} {identity} group={group_values}")
    store, item = identity.split("/", 1)
    found_store = {_component(value) for value in frame[calendar.key_fields[0]].tolist()}
    found_item = {_component(value) for value in frame[calendar.key_fields[1]].tolist()}
    if found_store != {store} or found_item != {item}:
        raise FreezeStop("HOLD_TARGET_ELIGIBILITY_CHANGED", f"{dataset} {identity} key mismatch")
    return frame.sort_values("date", kind="mergesort").reset_index(drop=True)


def _rank_sources(dataset: str, identity: str, target: pd.DataFrame, catalog: dict) -> tuple[dict, pd.DataFrame]:
    calendar = calendar_for(dataset)
    target_key = _identity_key(identity)
    target_window = _comparison_matrix(dataset, target, calendar, required=True)
    legal_keys = []
    legal_windows = []
    for item in catalog["legal_identities"]:
        if catalog["catalog"][item]["key"] == target_key:
            continue
        legal_keys.append(catalog["catalog"][item]["key"])
        legal_windows.append(catalog["catalog"][item]["comparison"])
    after = len(legal_keys)
    before = len(catalog["legal_identities"])
    target_in = identity in catalog["catalog"]
    if after < 3:
        raise FreezeStop("SKIPPED_INSUFFICIENT_CANDIDATES", f"{dataset} {identity} after={after}")
    candidates = np.stack(legal_windows).astype(np.float64, copy=False)
    scaler = fit_selection_scaler(target_window, candidates, calendar.similarity_variables)
    scaled_target = transform_selection(scaler, target_window)
    scaled_candidates = transform_selection(scaler, candidates)
    distances = np.linalg.norm(scaled_candidates - scaled_target, axis=1).astype(np.float64, copy=False)
    ranked = anchored_rank(tuple(legal_keys), distances)
    chosen = ranked[:3]
    weights = inverse_distance_weights(np.array([item.distance for item in chosen], dtype=np.float64))
    live = ["/".join(item.source_key) for item in chosen]
    if identity in live:
        raise FreezeStop("HOLD_TARGET_OVERLAP", f"{dataset} selected itself {identity}")
    if abs(float(weights.sum()) - 1.0) > 1e-8:
        raise FreezeStop(_blocked_dataset(dataset), f"{dataset} {identity} weight_sum={float(weights.sum())}")
    scaler_hash = _payload_digest(
        {
            "variables": list(scaler.variables),
            "minimum": np.asarray(scaler.minimum, dtype=np.float64).tolist(),
            "maximum": np.asarray(scaler.maximum, dtype=np.float64).tolist(),
        }
    )
    distance_by = {"/".join(item.source_key): float(item.distance) for item in ranked}
    rank_by = {"/".join(item.source_key): index + 1 for index, item in enumerate(ranked)}
    selected_set = set(live)
    rank_rows = []
    for source_identity, info in catalog["catalog"].items():
        rank_rows.append(
            {
                "target_identity": identity,
                "source_identity": source_identity,
                "self_excluded": source_identity == identity,
                "raw_comparison_eligibility": info["comparison"] is not None,
                "exact_180": bool(info["exact_180"]),
                "in_source_universe_before_self_exclusion": True,
                "distance": distance_by.get(source_identity, np.nan),
                "rank": rank_by.get(source_identity, np.nan),
                "selected_top3": source_identity in selected_set,
            }
        )
    result = {
        "dataset": dataset,
        "store_id": STORE_VALUES[dataset],
        "target_identity": identity,
        "candidate_count_before_self_exclusion": int(before),
        "target_in_source_universe": bool(target_in),
        "candidate_count_after_self_exclusion": int(after),
        "source_1": live[0],
        "distance_1": float(chosen[0].distance),
        "weight_1": float(weights[0]),
        "source_2": live[1],
        "distance_2": float(chosen[1].distance),
        "weight_2": float(weights[1]),
        "source_3": live[2],
        "distance_3": float(chosen[2].distance),
        "weight_3": float(weights[2]),
        "weight_sum": float(weights.sum()),
        "comparison_start": calendar.comparison_start,
        "comparison_end": calendar.comparison_end,
        "selector_scaler_hash": scaler_hash,
        "protocol_hash": frozen_protocol_sha256(),
        "selection_status": STATUS_COMPLETED,
        "source_universe_id": EXPECTED_UNIVERSE[dataset]["id"],
        "horizon_independent": True,
        "selected_sources_h1": live,
        "selected_sources_h3": list(live),
        "selected_sources_h5": list(live),
    }
    return result, pd.DataFrame(rank_rows)


def _verify_official_top3(dataset: str, identity: str, target: pd.DataFrame, source_frame: pd.DataFrame, live: list[str]) -> None:
    official = select_top3(
        dataset,
        identity,
        target,
        source_frame,
        source_universe_id=EXPECTED_UNIVERSE[dataset]["id"],
    )
    if official.status != STATUS_COMPLETED:
        raise FreezeStop("SKIPPED_INSUFFICIENT_CANDIDATES", f"{dataset} {identity} official={official.status}")
    official_ids = [item.identity for item in official.selected]
    if official_ids != live:
        raise FreezeStop(
            "HOLD_HORIZON_DEPENDENT_SELECTION_FOUND",
            f"{dataset} {identity} official {official_ids} != precomputed {live}",
        )


def _inspect_cache(cache_key: str, expected: dict) -> str:
    directory = SOURCE_MODEL_CACHE / cache_key
    done_path = directory / "DONE.json"
    key_path = directory / "key.json"
    weights_path = directory / "weights.npz"
    if not done_path.is_file():
        return "CACHE_MISS"
    if not (key_path.is_file() and weights_path.is_file()):
        return "CACHE_HASH_MISMATCH"
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        key = json.loads(key_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "CACHE_HASH_MISMATCH"
    digest = _payload_digest(expected)
    if cache_key != digest:
        return "CACHE_HASH_MISMATCH"
    if done.get("cache_key_sha256") != digest or done.get("status") != "COMPLETED":
        return "CACHE_HASH_MISMATCH"
    for field, value in expected.items():
        if key.get(field) != value:
            return "CACHE_HASH_MISMATCH"
    return "CACHE_HIT"


def freeze_dataset(dataset: str, log: _Log) -> dict:
    out = OUT_DIRS[dataset]
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_rankings").mkdir(parents=True, exist_ok=True)
    calendar = calendar_for(dataset)
    source_path = source_cache_path(dataset)
    source_sha = sha256_file(source_path)
    if source_sha != EXPECTED_SOURCE_SHA[dataset]:
        raise FreezeStop("HOLD_SOURCE_CACHE_HASH_MISMATCH", f"{dataset} source sha {source_sha}")
    protocol_sha = frozen_protocol_sha256()
    seed = derived_seed(dataset)
    log.write(f"{dataset} derived_seed={seed} source_sha={source_sha[:12]} protocol={protocol_sha[:12]}")
    log.write(f"{dataset} loading source universe")
    source_frame = _read_scoped_table(dataset, source=True)
    catalog = _legal_source_catalog(dataset, source_frame)
    expected_legal = EXPECTED_UNIVERSE[dataset]["legal_sources"]
    log.write(f"{dataset} source identities={len(catalog['identities'])} legal={len(catalog['legal_identities'])}")
    if len(catalog["legal_identities"]) != expected_legal:
        raise FreezeStop(
            _blocked_dataset(dataset),
            f"{dataset} legal sources {len(catalog['legal_identities'])} != frozen {expected_legal}",
        )
    log.write(f"{dataset} loading target pool")
    target_frame = _read_scoped_table(dataset, source=False)
    by_identity = _index_targets(dataset, target_frame)
    eligible = [
        identity
        for identity in sorted(by_identity, key=lambda item: _identity_sort_key(dataset, item))
        if identity != PAPER_TARGETS[dataset] and _target_is_eligible(dataset, identity, by_identity[identity], catalog)
    ]
    excluded = set(OLD_ADDITIONAL[dataset]) | {PAPER_TARGETS[dataset]}
    if PAPER_TARGETS[dataset] in eligible:
        raise FreezeStop("HOLD_TARGET_OVERLAP", f"{dataset} paper target remained eligible")
    missing_old = sorted(OLD_ADDITIONAL[dataset] - set(eligible), key=lambda item: _identity_sort_key(dataset, item))
    if missing_old:
        raise FreezeStop("HOLD_TARGET_ELIGIBILITY_CHANGED", f"{dataset} old additional missing {missing_old}")
    expected_eligible = EXPECTED_UNIVERSE[dataset]["eligible_targets"]
    if len(eligible) != expected_eligible:
        raise FreezeStop(
            "HOLD_TARGET_ELIGIBILITY_CHANGED",
            f"{dataset} eligible {len(eligible)} != frozen {expected_eligible}",
        )
    pool = [item for item in eligible if item not in OLD_ADDITIONAL[dataset]]
    log.write(f"{dataset} eligible={len(eligible)} remaining_after_old10={len(pool)}")
    if len(pool) < 60:
        token = "BLOCKED_D5_INSUFFICIENT_NEW_TARGETS" if dataset == "D5" else "BLOCKED_D6_INSUFFICIENT_NEW_TARGETS"
        raise FreezeStop(token, f"remaining={len(pool)}")
    sampled = random.Random(seed).sample(pool, 60)
    overlap_old = sorted(set(sampled) & OLD_ADDITIONAL[dataset], key=lambda item: _identity_sort_key(dataset, item))
    overlap_formal = sorted(set(sampled) & {PAPER_TARGETS[dataset]}, key=lambda item: _identity_sort_key(dataset, item))
    if overlap_old or overlap_formal:
        raise FreezeStop("HOLD_TARGET_OVERLAP", f"{dataset} old={overlap_old} formal={overlap_formal}")
    if len(set(sampled)) != 60:
        raise FreezeStop("HOLD_TARGET_OVERLAP", f"{dataset} duplicate sample")
    records = []
    selection_rows = []
    usage: dict[str, dict] = {}
    for rank, identity in enumerate(sampled, start=1):
        log.write(f"{dataset} assert+select {rank}/60 {identity}")
        target = _assert_target(dataset, identity, by_identity[identity], catalog)
        if rank == 1:
            official_target = load_target_series(dataset, identity)
            if list(official_target["date"]) != list(target["date"]):
                raise FreezeStop("HOLD_TARGET_ELIGIBILITY_CHANGED", f"{dataset} {identity} loader date mismatch")
        row, ranking = _rank_sources(dataset, identity, target, catalog)
        if rank == 1:
            log.write(f"{dataset} verifying official select_top3 on {identity}")
            _verify_official_top3(dataset, identity, target, source_frame, [row["source_1"], row["source_2"], row["source_3"]])
        ranking.to_parquet(out / "source_rankings" / _ranking_name(identity), index=False)
        selection_rows.append(row)
        records.append(
            {
                "dataset": dataset,
                "target_identity": identity,
                "target_rank": rank,
                "selection_origin": "NEW_RANDOM_60",
                "base_sampling_seed": BASE_SEED,
                "derived_sampling_seed": seed,
                "eligibility_status": "eligible",
                "train_start": calendar.train_start,
                "train_end": calendar.train_end,
                "val_start": calendar.val_start,
                "val_end": calendar.val_end,
                "test_start": calendar.test_start,
                "test_end": calendar.test_end,
                "n_train_days": TRAIN_DAYS,
                "n_val_days": VAL_DAYS,
                "n_test_days": TEST_DAYS,
                "source_universe_id": EXPECTED_UNIVERSE[dataset]["id"],
                "source_data_path": str(source_path.relative_to(REPO_ROOT)),
                "source_data_sha256": source_sha,
                "protocol_version": PROTOCOL_VERSION[dataset],
            }
        )
        for position, source_identity in enumerate((row["source_1"], row["source_2"], row["source_3"]), start=1):
            bucket = usage.setdefault(
                source_identity,
                {
                    "source_identity": source_identity,
                    "selected_by_n_targets": 0,
                    "selected_as_rank1_n": 0,
                    "selected_as_rank2_n": 0,
                    "selected_as_rank3_n": 0,
                    "distances": [],
                    "weights": [],
                },
            )
            bucket["selected_by_n_targets"] += 1
            bucket[f"selected_as_rank{position}_n"] += 1
            bucket["distances"].append(float(row[f"distance_{position}"]))
            bucket["weights"].append(float(row[f"weight_{position}"]))
    if len(records) != 60 or len({row["target_identity"] for row in records}) != 60:
        raise FreezeStop("HOLD_TARGET_OVERLAP", f"{dataset} n_targets={len(records)}")
    manifest = {
        "protocol_version": PROTOCOL_VERSION[dataset],
        "dataset": dataset,
        "target_sampling_seed": BASE_SEED,
        "derived_sampling_seed": seed,
        "derived_seed_algorithm": f'sha256("target_sampling_seed={BASE_SEED}|dataset={dataset}")[:8] as int % 2**31',
        "n_targets": 60,
        "n_unique_targets": 60,
        "selection_origin": "NEW_RANDOM_60",
        "excluded_formal_target": PAPER_TARGETS[dataset],
        "excluded_old_additional_targets": sorted(OLD_ADDITIONAL[dataset], key=lambda item: _identity_sort_key(dataset, item)),
        "store": STORE_VALUES[dataset],
        "group_field": calendar.group_field,
        "group_value": GROUP_VALUES[dataset],
        "source_data_path": str(source_path.relative_to(REPO_ROOT)),
        "source_data_sha256": source_sha,
        "records": records,
        "training_this_round": False,
    }
    manifest_path = out / f"{dataset.lower()}_60target_manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    (out / f"{dataset.lower()}_60target_manifest.sha256").write_text(f"{manifest_sha}  {manifest_path.name}\n", encoding="utf-8")
    _atomic_csv(out / f"{dataset.lower()}_60target_manifest.csv", pd.DataFrame(records))
    _atomic_json(
        out / "sampling_audit.json",
        {
            "dataset": dataset,
            "base_sampling_seed": BASE_SEED,
            "derived_sampling_seed": seed,
            "derived_seed_algorithm": f'sha256("target_sampling_seed={BASE_SEED}|dataset={dataset}")[:8] as int % 2**31',
            "eligible_pool_including_old_additional": len(eligible),
            "eligible_pool_after_exclusions": len(pool),
            "old_additional_in_eligible": sorted(OLD_ADDITIONAL[dataset], key=lambda item: _identity_sort_key(dataset, item)),
            "formal_excluded": PAPER_TARGETS[dataset],
            "sampled": sampled,
            "n_sampled": 60,
            "n_unique_targets": 60,
            "overlap_old_10": overlap_old,
            "overlap_formal": overlap_formal,
            "python_builtin_hash": False,
            "parquet_row_order": False,
            "replacement": False,
            "sort": "canonical identity",
            "source_legal_universe": len(catalog["legal_identities"]),
            "source_universe_id": EXPECTED_UNIVERSE[dataset]["id"],
            "source_data_sha256": source_sha,
            "manifest_sha256": manifest_sha,
            "exclusion_check": "PASS",
            "eligibility_check": "PASS",
        },
    )
    selection_frame = pd.DataFrame(selection_rows)
    if not np.allclose(selection_frame["weight_sum"].to_numpy(dtype=np.float64), 1.0, atol=1e-8):
        raise FreezeStop(_blocked_dataset(dataset), f"{dataset} weight_sum not 1")
    _atomic_csv(out / "source_selection_results.csv", selection_frame)
    _atomic_json(out / "source_selection_results.json", {"records": selection_rows, "horizon_independent": True})
    usage_rows = []
    for source_identity, bucket in usage.items():
        usage_rows.append(
            {
                "dataset": dataset,
                "store_id": STORE_VALUES[dataset],
                "source_identity": source_identity,
                "selected_by_n_targets": bucket["selected_by_n_targets"],
                "selected_as_rank1_n": bucket["selected_as_rank1_n"],
                "selected_as_rank2_n": bucket["selected_as_rank2_n"],
                "selected_as_rank3_n": bucket["selected_as_rank3_n"],
                "mean_distance": float(np.mean(bucket["distances"])),
                "mean_weight": float(np.mean(bucket["weights"])),
            }
        )
    usage_frame = pd.DataFrame(usage_rows).sort_values(
        ["selected_by_n_targets", "source_identity"],
        ascending=[False, True],
        kind="mergesort",
    )
    _atomic_csv(out / "source_usage_summary.csv", usage_frame)
    unique_sources = sorted(usage, key=lambda item: _identity_sort_key(dataset, item))
    req_rows = []
    hits = misses = mismatches = 0
    for source_identity in unique_sources:
        info = catalog["catalog"][source_identity]
        series_hash = _series_sha256(info["series"], calendar.model_features)
        n_targets = int(usage[source_identity]["selected_by_n_targets"])
        for horizon in HORIZONS:
            for seed_value in SEEDS:
                payload = {
                    "dataset": dataset,
                    "source_identity": source_identity,
                    "horizon": int(horizon),
                    "seed": int(seed_value),
                    "architecture": "cnn",
                    "source_data_sha256": series_hash,
                    "protocol_sha256": protocol_sha,
                }
                cache_key = _payload_digest(payload)
                status = _inspect_cache(cache_key, payload)
                if status == "CACHE_HIT":
                    hits += 1
                elif status == "CACHE_MISS":
                    misses += 1
                else:
                    mismatches += 1
                req_rows.append(
                    {
                        "dataset": dataset,
                        "store_id": STORE_VALUES[dataset],
                        "source_identity": source_identity,
                        "horizon": int(horizon),
                        "seed": int(seed_value),
                        "architecture": "cnn",
                        "source_data_hash": series_hash,
                        "protocol_hash": protocol_sha,
                        "required_by_n_targets": n_targets,
                        "cache_key": cache_key,
                        "cache_status": status,
                    }
                )
    req = pd.DataFrame(req_rows)
    expected_keys = len(unique_sources) * 15
    if len(req) != expected_keys:
        raise FreezeStop("BLOCKED_STAGE_A_CACHE_AUDIT", f"{dataset} keys {len(req)} != {expected_keys}")
    _atomic_csv(out / "stage_a_requirements.csv", req)
    reuse = 1.0 - (len(unique_sources) / 180.0)
    summary = {
        "dataset": dataset,
        "eligible_pool_after_exclusions": len(pool),
        "sampled_new_targets": 60,
        "top3_slots": 180,
        "unique_selected_sources": len(unique_sources),
        "source_reuse_rate": reuse,
        "required_stage_a_keys": int(len(req)),
        "CACHE_HIT": hits,
        "CACHE_MISS": misses,
        "HASH_MISMATCH": mismatches,
        "legal_source_universe": len(catalog["legal_identities"]),
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "manifest_sha256": manifest_sha,
        "derived_sampling_seed": seed,
        "sampled": sampled,
        "exclusion_check": "PASS",
        "eligibility_check": "PASS",
        "training_this_round": False,
        "fit_source_cnn": False,
    }
    _atomic_json(out / "round_summary.json", summary)
    _atomic_json(
        out / "protocol_snapshot.json",
        {
            "protocol_version": PROTOCOL_VERSION[dataset],
            "horizon_independent": True,
            "seeds_share_selection": True,
            "horizons_share_selection": True,
            "comparison_window": [calendar.comparison_start, calendar.comparison_end],
            "source_history": [calendar.history_start, calendar.history_end],
            "similarity_variables": list(calendar.similarity_variables),
            "model_features": list(calendar.model_features),
            "top_k": 3,
            "source_data_sha256": source_sha,
            "protocol_sha256": protocol_sha,
            "manifest_sha256": manifest_sha,
            "training_this_round": False,
        },
    )
    log.write(
        f"{dataset} unique_sources={len(unique_sources)} keys={len(req)} hit={hits} miss={misses} mismatch={mismatches}"
    )
    return summary


def main() -> int:
    for path in OUT_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    log = _Log(OUT_DIRS["D5"] / "freeze_run_log.txt")
    d6_log = _Log(OUT_DIRS["D6"] / "freeze_run_log.txt")
    try:
        _assert_horizon_independent()
        d5 = freeze_dataset("D5", log)
        d6_log.write("D5 freeze complete; starting D6")
        d6 = freeze_dataset("D6", d6_log)
        if d5["sampled_new_targets"] != 60 or d6["sampled_new_targets"] != 60:
            raise FreezeStop("HOLD_TARGET_OVERLAP", "count")
        if d5["HASH_MISMATCH"] or d6["HASH_MISMATCH"]:
            raise FreezeStop(
                "BLOCKED_STAGE_A_CACHE_AUDIT",
                f"D5 mismatch={d5['HASH_MISMATCH']} D6 mismatch={d6['HASH_MISMATCH']}",
            )
        combined = {
            "status": "READY_FOR_D5_D6_STAGE_A_MISSING_MODEL_TRAINING",
            "targets": {"D5": 60, "D6": 60, "TOTAL": 120},
            "top3_slots": {"D5": 180, "D6": 180, "TOTAL": 360},
            "unique_selected_sources": {
                "D5": d5["unique_selected_sources"],
                "D6": d6["unique_selected_sources"],
                "TOTAL": d5["unique_selected_sources"] + d6["unique_selected_sources"],
            },
            "stage_a_keys": {
                "D5": d5["required_stage_a_keys"],
                "D6": d6["required_stage_a_keys"],
                "TOTAL": d5["required_stage_a_keys"] + d6["required_stage_a_keys"],
            },
            "CACHE_HIT": {"D5": d5["CACHE_HIT"], "D6": d6["CACHE_HIT"], "TOTAL": d5["CACHE_HIT"] + d6["CACHE_HIT"]},
            "CACHE_MISS": {"D5": d5["CACHE_MISS"], "D6": d6["CACHE_MISS"], "TOTAL": d5["CACHE_MISS"] + d6["CACHE_MISS"]},
            "HASH_MISMATCH": {
                "D5": d5["HASH_MISMATCH"],
                "D6": d6["HASH_MISMATCH"],
                "TOTAL": d5["HASH_MISMATCH"] + d6["HASH_MISMATCH"],
            },
            "d5": d5,
            "d6": d6,
            "training_this_round": False,
            "finished_at": _utc_now(),
        }
        _atomic_json(OUT_DIRS["D5"] / "combined_d5_d6_summary.json", combined)
        _atomic_json(OUT_DIRS["D6"] / "combined_d5_d6_summary.json", combined)
        print(combined["status"], flush=True)
        log.write(combined["status"])
        d6_log.write(combined["status"])
        return 0
    except FreezeStop as exc:
        log.write(f"{exc.token}: {exc.detail}")
        d6_log.write(f"{exc.token}: {exc.detail}")
        print(exc.token, flush=True)
        _atomic_json(OUT_DIRS["D5"] / "HOLD.json", {"status": exc.token, "detail": exc.detail, "written_at": _utc_now()})
        _atomic_json(OUT_DIRS["D6"] / "HOLD.json", {"status": exc.token, "detail": exc.detail, "written_at": _utc_now()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
