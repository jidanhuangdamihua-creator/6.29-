"""Read the frozen additional-target manifest. This module does not sample."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from additional_target_robustness.paths import (
    MANIFEST_JSON,
    OUTPUT_ROOT,
    REPO_ROOT,
    SEALED_ROOT,
    SOURCE_AUDIT_JSON,
)
from additional_target_robustness.protocol import (
    CALENDARS,
    HOLD_SOURCE_CACHE_HASH_MISMATCH,
    HOLD_TARGET_MANIFEST_MISMATCH,
    MANIFEST_SHA256,
    PAPER_TARGETS,
    SAMPLING_SEED,
    SOURCE_AUDIT_SHA256,
    ProtocolError,
    ProtocolHold,
    assert_file_hash,
    calendar_for,
)


@dataclass(frozen=True)
class ManifestRecord:
    dataset: str
    target_identity: str
    target_sampling_seed: int
    eligibility_status: str
    target_train_start: str
    target_train_end: str
    target_val_start: str
    target_val_end: str
    target_test_start: str
    target_test_end: str
    target_total_days: int
    source_candidate_pool_id: str
    source_candidate_pool_hash: str
    selection_pool_size: int
    selected_rank: int

    @property
    def source_universe_id(self) -> str:
        return self.source_candidate_pool_id


def load_manifest(path: Path = MANIFEST_JSON, *, expected_sha256: str = MANIFEST_SHA256) -> tuple[ManifestRecord, ...]:
    path = Path(path)
    if not path.is_file():
        raise ProtocolHold(HOLD_TARGET_MANIFEST_MISMATCH)
    assert_file_hash(path, expected_sha256, token=HOLD_TARGET_MANIFEST_MISMATCH)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(raw_records, list):
        raise ProtocolHold(HOLD_TARGET_MANIFEST_MISMATCH)
    records = tuple(_record(item) for item in raw_records)
    _require_frozen_shape(records)
    return records


def require_manifest_target(records: tuple[ManifestRecord, ...], dataset: str, identity: str) -> ManifestRecord:
    matches = [record for record in records if record.dataset == dataset and record.target_identity == identity]
    if len(matches) != 1:
        raise ProtocolError(f"{dataset} {identity} is not a single frozen manifest target")
    return matches[0]


def verify_source_caches(audit_path: Path = SOURCE_AUDIT_JSON) -> dict[str, str]:
    """Hash the frozen source files and refuse to continue when any hash moved."""
    audit_path = Path(audit_path)
    assert_file_hash(audit_path, SOURCE_AUDIT_SHA256, token=HOLD_SOURCE_CACHE_HASH_MISMATCH)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    datasets = audit.get("datasets")
    if not isinstance(datasets, dict):
        raise ProtocolHold(HOLD_SOURCE_CACHE_HASH_MISMATCH)
    specs = (
        ("D1", "source_cache_path", "source_sha256", "source"),
        ("D4", "source_cache_path", "source_sha256", "source"),
        ("D5", "source_cache_path", "prepared_sha256", "source"),
        ("D5", "target_side_path", "sealed_source_sha256", "target"),
        ("D6", "source_cache_path", "source_sha256", "source"),
    )
    found: dict[str, str] = {}
    for dataset, path_field, hash_field, role in specs:
        block = datasets.get(dataset)
        if not isinstance(block, dict) or path_field not in block or hash_field not in block:
            raise ProtocolHold(HOLD_SOURCE_CACHE_HASH_MISMATCH)
        relative = str(block[path_field])
        expected = str(block[hash_field])
        path = (REPO_ROOT / relative).resolve()
        calendar = calendar_for(dataset)
        filename = calendar.source_filename if role == "source" else calendar.target_filename
        if path != (SEALED_ROOT / filename).resolve():
            raise ProtocolHold(HOLD_SOURCE_CACHE_HASH_MISMATCH)
        actual = assert_file_hash(path, expected, token=HOLD_SOURCE_CACHE_HASH_MISMATCH)
        found[f"{dataset}:{role}"] = actual
    if audit.get("overall_source_calendar_rebuild") != "NO_REBUILD_REQUIRED":
        raise ProtocolHold(HOLD_SOURCE_CACHE_HASH_MISMATCH)
    return found


def _record(item: object) -> ManifestRecord:
    if not isinstance(item, dict):
        raise ProtocolHold(HOLD_TARGET_MANIFEST_MISMATCH)
    try:
        return ManifestRecord(
            dataset=str(item["dataset"]),
            target_identity=str(item["target_identity"]),
            target_sampling_seed=int(item["target_sampling_seed"]),
            eligibility_status=str(item["eligibility_status"]),
            target_train_start=str(item["target_train_start"]),
            target_train_end=str(item["target_train_end"]),
            target_val_start=str(item["target_val_start"]),
            target_val_end=str(item["target_val_end"]),
            target_test_start=str(item["target_test_start"]),
            target_test_end=str(item["target_test_end"]),
            target_total_days=int(item["target_total_days"]),
            source_candidate_pool_id=str(item["source_candidate_pool_id"]),
            source_candidate_pool_hash=str(item["source_candidate_pool_hash"]),
            selection_pool_size=int(item["selection_pool_size"]),
            selected_rank=int(item["selected_rank"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolHold(HOLD_TARGET_MANIFEST_MISMATCH) from exc


def _require_frozen_shape(records: tuple[ManifestRecord, ...]) -> None:
    if len(records) != 40:
        raise ProtocolHold(HOLD_TARGET_MANIFEST_MISMATCH)
    by_dataset: dict[str, list[ManifestRecord]] = {name: [] for name in CALENDARS}
    for record in records:
        if record.dataset not in CALENDARS or record.eligibility_status != "eligible":
            raise ProtocolHold(HOLD_TARGET_MANIFEST_MISMATCH)
        if record.target_sampling_seed != SAMPLING_SEED or record.target_total_days != 210:
            raise ProtocolHold(HOLD_TARGET_MANIFEST_MISMATCH)
        if record.target_identity == PAPER_TARGETS[record.dataset]:
            raise ProtocolHold(HOLD_TARGET_MANIFEST_MISMATCH)
        calendar = calendar_for(record.dataset)
        expected = (
            calendar.train_start,
            calendar.train_end,
            calendar.val_start,
            calendar.val_end,
            calendar.test_start,
            calendar.test_end,
        )
        got = (
            record.target_train_start,
            record.target_train_end,
            record.target_val_start,
            record.target_val_end,
            record.target_test_start,
            record.target_test_end,
        )
        if got != expected:
            raise ProtocolHold(HOLD_TARGET_MANIFEST_MISMATCH)
        by_dataset[record.dataset].append(record)
    for dataset, rows in by_dataset.items():
        ranks = [row.selected_rank for row in rows]
        identities = [row.target_identity for row in rows]
        if len(rows) != 10 or sorted(ranks) != list(range(1, 11)) or len(set(identities)) != 10:
            raise ProtocolHold(HOLD_TARGET_MANIFEST_MISMATCH)
        pool_ids = {row.source_candidate_pool_id for row in rows}
        if len(pool_ids) != 1:
            raise ProtocolHold(HOLD_TARGET_MANIFEST_MISMATCH)
        del dataset
    if OUTPUT_ROOT.name != "additional_target_robustness_20261004":
        raise ProtocolHold(HOLD_TARGET_MANIFEST_MISMATCH)
