"""Immutable experiment definitions shared by every D1-D6 runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Sequence, Tuple, Union


PROTOCOL_VERSION = "d1_d6_protocol_v1"
FORMAL_HORIZONS = (1, 2, 3, 4, 5)
FORMAL_SEEDS = (42, 43, 44, 45, 46)
STRICT_PAPER_TRACK = "strict_paper"
EXTENDED_TRACK = "extended"

SourceKey = Tuple[str, ...]


class ProtocolViolation(ValueError):
    """Raised when data or configuration violates the strict protocol."""


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "to_pydatetime"):
        converted = value.to_pydatetime()
        return converted.date() if isinstance(converted, datetime) else converted
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation(f"invalid protocol date: {value!r}") from exc


def _normalize_component(value: object) -> str:
    if value is None:
        raise ProtocolViolation("source key components may not be null")
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not value.is_integer():
            return format(value, ".17g")
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise ProtocolViolation("source key components may not be empty")
    return text


def normalize_source_key(key: Sequence[object]) -> SourceKey:
    normalized = tuple(_normalize_component(part) for part in key)
    if not normalized:
        raise ProtocolViolation("source key may not be empty")
    return normalized


def normalize_scenario(scenario: object) -> str:
    text = "_".join(
        str(scenario).strip().lower().replace("-", "_").replace(" ", "_").split("_")
    )
    aliases = {
        "with": "with",
        "with_info_sharing": "with",
        "with_information_sharing": "with",
        "without": "without",
        "without_info_sharing": "without",
        "without_information_sharing": "without",
    }
    if text not in aliases:
        raise ProtocolViolation(f"unsupported information-sharing scenario: {scenario!r}")
    return aliases[text]


@dataclass(frozen=True)
class ObservationWindow:
    knn_observed_start: date
    knn_observed_end: date
    source_observation_cutoff: date

    @classmethod
    def from_start(cls, observed_start: object) -> "ObservationWindow":
        start = _as_date(observed_start)
        end = start + timedelta(days=29)
        return cls(start, end, end)

    def is_test_date(self, value: object) -> bool:
        return _as_date(value) > self.knn_observed_end


@dataclass(frozen=True)
class SourcePoolRule:
    key_fields: Tuple[str, ...]
    target_key: Optional[SourceKey]
    grouping_field: Optional[str] = None


@dataclass(frozen=True)
class SourceIdentity:
    key: SourceKey
    group_value: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", normalize_source_key(self.key))
        if self.group_value is not None:
            object.__setattr__(self, "group_value", _normalize_component(self.group_value))


@dataclass(frozen=True)
class ExperimentProtocol:
    dataset_id: str
    track: str
    source_pool_rule: SourcePoolRule
    protocol_version: str = PROTOCOL_VERSION
    horizons: Tuple[int, ...] = FORMAL_HORIZONS
    seeds: Tuple[int, ...] = FORMAL_SEEDS
    knn_representation: str = "daily_sales_flattened_30d"
    primary_metric_space: str = "original_sales"
    tie_tolerance: float = 1e-12
    weight_mode: str = "inverse_distance"
    weight_epsilon: float = 1e-8

    def observation_window(self, observed_start: object) -> ObservationWindow:
        return ObservationWindow.from_start(observed_start)


_PROTOCOLS = {
    "D1": ExperimentProtocol(
        "D1",
        STRICT_PAPER_TRACK,
        SourcePoolRule(("store", "item"), ("Store1", "Item10")),
    ),
    "D2": ExperimentProtocol(
        "D2",
        STRICT_PAPER_TRACK,
        SourcePoolRule(("brand", "item"), ("Brand1", "Item10")),
    ),
    "D3": ExperimentProtocol(
        "D3",
        STRICT_PAPER_TRACK,
        SourcePoolRule(("store",), ("Store10",)),
    ),
    "D4": ExperimentProtocol(
        "D4",
        EXTENDED_TRACK,
        SourcePoolRule(("store", "item"), None, "category"),
    ),
    "D5": ExperimentProtocol(
        "D5",
        EXTENDED_TRACK,
        SourcePoolRule(("store", "item"), None, "family"),
    ),
    "D6": ExperimentProtocol(
        "D6",
        EXTENDED_TRACK,
        SourcePoolRule(("store", "item"), None, "department"),
    ),
}


def _normalize_dataset_id(dataset_id: object) -> str:
    text = str(dataset_id).strip().upper().replace("DATASET", "").replace("D", "")
    if not text.isdigit() or int(text) not in range(1, 7):
        raise ProtocolViolation(f"unsupported dataset id: {dataset_id!r}")
    return f"D{int(text)}"


def get_experiment_protocol(dataset_id: object) -> ExperimentProtocol:
    return _PROTOCOLS[_normalize_dataset_id(dataset_id)]


def _strict_expected_keys(dataset_id: str, scenario: str) -> Tuple[SourceKey, ...]:
    if dataset_id == "D1":
        stores = range(1, 4) if scenario == "with" else range(1, 2)
        return tuple(
            (f"Store{store}", f"Item{item}")
            for store in stores
            for item in range(1, 10)
        )
    if dataset_id == "D2":
        brands = range(1, 4) if scenario == "with" else range(1, 2)
        return tuple(
            (f"Brand{brand}", f"Item{item}")
            for brand in brands
            for item in range(1, 10)
        )
    stores = range(1, 31) if scenario == "with" else range(1, 10)
    return tuple((f"Store{store}",) for store in stores if store != 10)


AvailableIdentity = Union[SourceIdentity, Sequence[object]]


def _materialize_identities(
    available_keys: Iterable[AvailableIdentity],
) -> Tuple[SourceIdentity, ...]:
    identities = tuple(
        entry if isinstance(entry, SourceIdentity) else SourceIdentity(tuple(entry))
        for entry in available_keys
    )
    seen = set()
    duplicates = []
    for identity in identities:
        if identity.key in seen:
            duplicates.append(identity.key)
        seen.add(identity.key)
    if duplicates:
        raise ProtocolViolation(f"duplicate source key: {duplicates[0]!r}")
    return identities


def build_candidate_keys(
    protocol: ExperimentProtocol,
    scenario: object,
    target_key: Sequence[object],
    available_keys: Iterable[AvailableIdentity],
) -> Tuple[SourceKey, ...]:
    """Return the exact legal candidate keys or fail before training."""

    normalized_scenario = normalize_scenario(scenario)
    normalized_target = normalize_source_key(target_key)
    identities = _materialize_identities(available_keys)
    available_by_key = {identity.key: identity for identity in identities}

    if protocol.track == STRICT_PAPER_TRACK:
        expected_target = protocol.source_pool_rule.target_key
        if normalized_target != expected_target:
            raise ProtocolViolation(
                f"{protocol.dataset_id} target must be {expected_target!r}, got {normalized_target!r}"
            )
        expected = _strict_expected_keys(protocol.dataset_id, normalized_scenario)
        missing = tuple(key for key in expected if key not in available_by_key)
        if missing:
            raise ProtocolViolation(
                f"{protocol.dataset_id}/{normalized_scenario} missing required candidate keys: {missing!r}"
            )
        if normalized_target in expected:
            raise ProtocolViolation("target key entered strict candidate pool")
        return expected

    target_identity = available_by_key.get(normalized_target)
    if target_identity is None:
        raise ProtocolViolation(
            f"extended target key missing from available identities: {normalized_target!r}"
        )
    if target_identity.group_value is None:
        raise ProtocolViolation(
            f"extended target {normalized_target!r} has no {protocol.source_pool_rule.grouping_field}"
        )

    target_store = normalized_target[0]
    candidates = []
    for identity in identities:
        if identity.key == normalized_target:
            continue
        if identity.group_value != target_identity.group_value:
            continue
        if normalized_scenario == "without" and identity.key[0] != target_store:
            continue
        candidates.append(identity.key)

    candidates.sort()
    if not candidates:
        raise ProtocolViolation(
            f"{protocol.dataset_id}/{normalized_scenario} candidate pool is empty for {normalized_target!r}"
        )
    return tuple(candidates)
