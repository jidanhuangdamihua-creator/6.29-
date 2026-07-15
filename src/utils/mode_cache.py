"""Separate worker/evaluator cache contracts for formal runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import secrets
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pandas as pd

from src.protocols.artifact_schemas import get_worker_manifest_schema
from src.protocols.experiment_protocol import FORMAL_METHODS, normalize_scenario
from src.protocols.sealing_protocol import normalize_dataset_id


class CacheIsolationError(ValueError):
    """A cache or manifest crossed the worker/evaluator boundary."""


def _cache_digest(names: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(str(name) for name in names), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class WorkerRunLayout:
    """The only filesystem layout object a worker is allowed to receive."""

    run_root: Path
    run_id: str
    dataset_id: str
    scenario: str
    cell_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_root", Path(self.run_root))
        object.__setattr__(self, "run_id", str(self.run_id))
        object.__setattr__(self, "dataset_id", normalize_dataset_id(self.dataset_id))
        object.__setattr__(self, "scenario", normalize_scenario(self.scenario))
        if not str(self.cell_id).strip() or not str(self.attempt_id).strip():
            raise ValueError("worker layout requires cell_id and attempt_id")

    @property
    def worker_cache_dir(self) -> Path:
        return self.run_root / "worker_cache" / self.run_id / self.dataset_id.lower() / self.scenario / self.cell_id.replace("/", "_") / self.attempt_id

    @property
    def worker_manifest_path(self) -> Path:
        return self.worker_cache_dir / "worker_manifest.json"

    def to_worker_dict(self) -> dict[str, str]:
        """Return only non-sensitive worker routing values."""

        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "scenario": self.scenario,
            "cell_id": str(self.cell_id),
            "attempt_id": str(self.attempt_id),
            "worker_cache_dir": str(self.worker_cache_dir),
        }


@dataclass(frozen=True)
class WorkerCache:
    root: Path
    views: Mapping[str, pd.DataFrame]
    cache_identity: str

    def get_view(self, name: str) -> pd.DataFrame:
        key = str(name)
        if key == "evaluator_truth_frame" or "truth" in key.lower():
            raise CacheIsolationError("worker cache cannot expose evaluator truth")
        try:
            return self.views[key].copy()
        except KeyError as exc:
            raise KeyError(f"worker view is not cached: {key}") from exc


@dataclass(frozen=True)
class EvaluatorCache:
    root: Path
    truth_frame: pd.DataFrame
    capability_id: str

    def get_truth(self, capability_id: str) -> pd.DataFrame:
        if str(capability_id) != self.capability_id:
            raise CacheIsolationError("invalid evaluator capability")
        return self.truth_frame.copy()


class EvaluatorControlPlane:
    """Private capability-to-cache mapping; never serialized into worker data."""

    def __init__(self) -> None:
        self._caches: dict[str, EvaluatorCache] = {}

    def register(self, cache: EvaluatorCache) -> str:
        self._caches[cache.capability_id] = cache
        return cache.capability_id

    def resolve(self, capability_id: str) -> EvaluatorCache:
        try:
            return self._caches[str(capability_id)]
        except KeyError as exc:
            raise CacheIsolationError("unknown evaluator capability") from exc


def _reject_truth_views(views: Mapping[str, pd.DataFrame]) -> None:
    forbidden = {"evaluator_truth_frame", "y_true", "truth_key", "is_synthetic_date"}
    leaked = sorted(
        name for name, frame in views.items()
        if str(name).lower() in forbidden
        or forbidden.intersection(str(column).lower() for column in frame.columns)
    )
    if leaked:
        raise CacheIsolationError(f"worker cache contains evaluator-only view(s): {leaked}")


def create_worker_cache(
    root: str | Path,
    views: Mapping[str, pd.DataFrame],
    *,
    cache_id: str | None = None,
) -> WorkerCache:
    """Create a worker-only cache in a directory disjoint from evaluator data."""

    _reject_truth_views(views)
    identity = str(cache_id or _cache_digest(tuple(views)))
    cache_root = Path(root) / "worker_cache" / identity
    cache_root.mkdir(parents=True, exist_ok=True)
    copied = {str(name): frame.copy() for name, frame in views.items()}
    for name, frame in copied.items():
        if not name or "/" in name or "truth" in name.lower():
            raise CacheIsolationError(f"unsafe worker cache view name: {name!r}")
        frame.to_parquet(cache_root / f"{name}.parquet", index=False)
    return WorkerCache(cache_root, MappingProxyType(copied), identity)


def create_evaluator_cache(
    root: str | Path,
    truth_frame: pd.DataFrame,
    *,
    capability_id: str | None = None,
    return_control_plane: bool = False,
) -> EvaluatorCache | tuple[EvaluatorCache, EvaluatorControlPlane]:
    """Create evaluator truth under a high-entropy capability held by the control plane."""

    if not isinstance(truth_frame, pd.DataFrame):
        raise TypeError("evaluator cache requires a pandas DataFrame")
    if "y_true" not in truth_frame.columns:
        raise CacheIsolationError("evaluator cache requires y_true")
    capability = str(capability_id or secrets.token_urlsafe(32))
    if len(capability) < 32:
        raise CacheIsolationError("evaluator capability must be high entropy")
    cache_root = Path(root) / "evaluator_cache" / capability
    cache_root.mkdir(parents=True, exist_ok=True)
    copied = truth_frame.copy()
    copied.to_parquet(cache_root / "evaluator_truth.parquet", index=False)
    cache = EvaluatorCache(cache_root, copied, capability)
    if return_control_plane:
        control = EvaluatorControlPlane()
        control.register(cache)
        return cache, control
    return cache


def build_worker_manifest(
    layout: WorkerRunLayout,
    *,
    method: str,
    seed: int,
    schema_name: str,
    schema_version: str,
    schema_digest: str,
    artifact_path: str,
    row_count: int,
    canonical_content_sha256: str,
    artifact_sha256: str,
    semantic_prediction_digest: str,
    status: str,
) -> dict[str, Any]:
    """Build and validate the fixed WorkerManifestSchemaV1 row."""

    if str(method) not in FORMAL_METHODS:
        raise ValueError(f"unsupported formal method: {method!r}")
    row = {
        "run_id": layout.run_id,
        "cell_id": str(layout.cell_id),
        "attempt_id": str(layout.attempt_id),
        "dataset_id": layout.dataset_id,
        "scenario": layout.scenario,
        "target_entity_key": str(str(layout.cell_id).split("/")[-2] if "/" in str(layout.cell_id) else layout.cell_id),
        "method": str(method),
        "seed": int(seed),
        "schema_name": str(schema_name),
        "schema_version": str(schema_version),
        "schema_digest": str(schema_digest),
        "artifact_path": str(artifact_path),
        "row_count": int(row_count),
        "canonical_content_sha256": str(canonical_content_sha256),
        "artifact_sha256": str(artifact_sha256),
        "semantic_prediction_digest": str(semantic_prediction_digest),
        "status": str(status),
    }
    return get_worker_manifest_schema().validate_record(row)


__all__ = [
    "CacheIsolationError",
    "EvaluatorCache",
    "EvaluatorControlPlane",
    "WorkerCache",
    "WorkerRunLayout",
    "build_worker_manifest",
    "create_evaluator_cache",
    "create_worker_cache",
]
