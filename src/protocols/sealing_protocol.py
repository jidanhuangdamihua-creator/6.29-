"""Immutable calendar contract for the sealed D1-D6 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Tuple

from .experiment_protocol import (
    FORMAL_HORIZONS,
    FORMAL_METHODS,
    FORMAL_SEEDS,
    ProtocolViolation,
)


SEALING_PROTOCOL_VERSION = "d1_d6_sealed_v1"
TARGET_TRAIN_DAYS = 15
TARGET_VALIDATION_DAYS = 15
TARGET_BLIND_DAYS = 180
SOURCE_PRETRAIN_DAYS = 180
SOURCE_KNN_DAYS = 30
FORMAL_SAMPLE_COUNTS = (180, 179, 178, 177, 176)


def normalize_dataset_id(dataset_id: object) -> str:
    """Normalize only the closed D1-D6 identifier vocabulary."""

    if isinstance(dataset_id, bool):
        raise ProtocolViolation(f"unsupported dataset id: {dataset_id!r}")
    text = str(dataset_id).strip().upper()
    if text.startswith("DATASET"):
        text = text[len("DATASET") :].strip()
    if text.startswith("D"):
        text = text[1:]
    if not text.isdigit() or int(text) not in range(1, 7):
        raise ProtocolViolation(f"unsupported dataset id: {dataset_id!r}")
    return f"D{int(text)}"


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError) as exc:
        raise ProtocolViolation(f"invalid protocol date: {value!r}") from exc


def _inclusive_days(start: date, end: date) -> int:
    return (end - start).days + 1


@dataclass(frozen=True)
class TargetWindow:
    dataset_id: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    blind_start: date
    blind_end: date
    protocol_version: str = SEALING_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        normalized = normalize_dataset_id(self.dataset_id)
        object.__setattr__(self, "dataset_id", normalized)
        for field_name in (
            "train_start",
            "train_end",
            "validation_start",
            "validation_end",
            "blind_start",
            "blind_end",
        ):
            object.__setattr__(self, field_name, _as_date(getattr(self, field_name)))
        if self.train_days != TARGET_TRAIN_DAYS:
            raise ProtocolViolation(f"{normalized} target train window must contain 15 natural days")
        if self.validation_days != TARGET_VALIDATION_DAYS:
            raise ProtocolViolation(
                f"{normalized} target validation window must contain 15 natural days"
            )
        if self.blind_days != TARGET_BLIND_DAYS:
            raise ProtocolViolation(f"{normalized} target blind window must contain 180 natural days")
        if self.validation_start != self.train_end + timedelta(days=1):
            raise ProtocolViolation(f"{normalized} target train/validation windows are not contiguous")
        if self.blind_start != self.validation_end + timedelta(days=1):
            raise ProtocolViolation(f"{normalized} target validation/blind windows are not contiguous")

    @property
    def target_start(self) -> date:
        return self.train_start

    @property
    def observed_start(self) -> date:
        return self.train_start

    @property
    def observed_end(self) -> date:
        return self.validation_end

    @property
    def train_days(self) -> int:
        return _inclusive_days(self.train_start, self.train_end)

    @property
    def validation_days(self) -> int:
        return _inclusive_days(self.validation_start, self.validation_end)

    @property
    def observed_days(self) -> int:
        return _inclusive_days(self.observed_start, self.observed_end)

    @property
    def blind_days(self) -> int:
        return _inclusive_days(self.blind_start, self.blind_end)


@dataclass(frozen=True)
class SourcePretrainWindow:
    dataset_id: str
    pretrain_start: date
    pretrain_end: date
    knn_start: date
    knn_end: date
    protocol_version: str = SEALING_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        normalized = normalize_dataset_id(self.dataset_id)
        object.__setattr__(self, "dataset_id", normalized)
        for field_name in ("pretrain_start", "pretrain_end", "knn_start", "knn_end"):
            object.__setattr__(self, field_name, _as_date(getattr(self, field_name)))
        if self.pretrain_days != SOURCE_PRETRAIN_DAYS:
            raise ProtocolViolation(f"{normalized} source pretrain window must contain 180 natural days")
        if self.knn_days != SOURCE_KNN_DAYS:
            raise ProtocolViolation(f"{normalized} source KNN window must contain 30 natural days")
        if self.knn_end != self.pretrain_end:
            raise ProtocolViolation(f"{normalized} source KNN window must end at pretrain end")
        if self.knn_start != self.pretrain_end - timedelta(days=SOURCE_KNN_DAYS - 1):
            raise ProtocolViolation(f"{normalized} source KNN window must be the final 30 pretrain days")

    @classmethod
    def ending_at(cls, dataset_id: object, observed_end: object) -> "SourcePretrainWindow":
        end = _as_date(observed_end)
        return cls(
            dataset_id=normalize_dataset_id(dataset_id),
            pretrain_start=end - timedelta(days=SOURCE_PRETRAIN_DAYS - 1),
            pretrain_end=end,
            knn_start=end - timedelta(days=SOURCE_KNN_DAYS - 1),
            knn_end=end,
        )

    @property
    def source_observation_cutoff(self) -> date:
        return self.pretrain_end

    @property
    def pretrain_days(self) -> int:
        return _inclusive_days(self.pretrain_start, self.pretrain_end)

    @property
    def knn_days(self) -> int:
        return _inclusive_days(self.knn_start, self.knn_end)


def _target_window(dataset_id: str, start: str, observed_end: str, blind_end: str) -> TargetWindow:
    train_start = _as_date(start)
    observed_end_date = _as_date(observed_end)
    return TargetWindow(
        dataset_id=dataset_id,
        train_start=train_start,
        train_end=train_start + timedelta(days=TARGET_TRAIN_DAYS - 1),
        validation_start=train_start + timedelta(days=TARGET_TRAIN_DAYS),
        validation_end=observed_end_date,
        blind_start=observed_end_date + timedelta(days=1),
        blind_end=_as_date(blind_end),
    )


_TARGET_WINDOWS: Dict[str, TargetWindow] = {
    "D1": _target_window("D1", "2017-06-01", "2017-06-30", "2017-12-27"),
    "D2": _target_window("D2", "2018-06-01", "2018-06-30", "2018-12-27"),
    "D3": _target_window("D3", "2015-01-03", "2015-02-01", "2015-07-31"),
    "D4": _target_window("D4", "2024-12-16", "2025-01-14", "2025-07-13"),
    "D5": _target_window("D5", "2017-01-17", "2017-02-15", "2017-08-14"),
    "D6": _target_window("D6", "2015-10-26", "2015-11-24", "2016-05-22"),
}

_SOURCE_WINDOWS: Dict[str, SourcePretrainWindow] = {
    dataset_id: SourcePretrainWindow.ending_at(dataset_id, window.observed_end)
    for dataset_id, window in _TARGET_WINDOWS.items()
}


def get_target_window(dataset_id: object) -> TargetWindow:
    return _TARGET_WINDOWS[normalize_dataset_id(dataset_id)]


def get_source_pretrain_window(dataset_id: object) -> SourcePretrainWindow:
    return _SOURCE_WINDOWS[normalize_dataset_id(dataset_id)]


def formal_sample_count(horizon: int) -> int:
    try:
        index = FORMAL_HORIZONS.index(int(horizon))
    except (ValueError, TypeError) as exc:
        raise ProtocolViolation(f"unsupported formal horizon: {horizon!r}") from exc
    return FORMAL_SAMPLE_COUNTS[index]


__all__ = [
    "FORMAL_HORIZONS",
    "FORMAL_METHODS",
    "FORMAL_SAMPLE_COUNTS",
    "FORMAL_SEEDS",
    "SEALING_PROTOCOL_VERSION",
    "SOURCE_KNN_DAYS",
    "SOURCE_PRETRAIN_DAYS",
    "TARGET_BLIND_DAYS",
    "TARGET_TRAIN_DAYS",
    "TARGET_VALIDATION_DAYS",
    "ProtocolViolation",
    "SourcePretrainWindow",
    "TargetWindow",
    "formal_sample_count",
    "get_source_pretrain_window",
    "get_target_window",
    "normalize_dataset_id",
]
