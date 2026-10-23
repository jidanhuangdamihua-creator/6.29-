"""Immutable date windows and feature roles for sealed D1-D6 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

from .experiment_protocol import ProtocolViolation


SEALED_PROTOCOL_VERSION = "d1_d6_sealed_blind_v1"
SOURCE_PRETRAIN_DAYS = 180

IDENTIFIER_COLUMNS = frozenset(
    {
        "entity_id",
        "item_id",
        "store_id",
        "store_nbr",
        "product_id",
        "brand_id",
        "brand_code",
        "region_id",
        "region_code",
        "category",
        "family",
        "department",
    }
)
CALENDAR_COLUMNS = frozenset({"year", "month", "week", "day", "day_of_week"})
DECLARED_FUTURE_KNOWN = {
    1: CALENDAR_COLUMNS,
    2: CALENDAR_COLUMNS,
    3: CALENDAR_COLUMNS | {"open", "promo", "state_holiday", "school_holiday"},
    4: CALENDAR_COLUMNS,
    5: CALENDAR_COLUMNS,
    6: CALENDAR_COLUMNS,
}


@dataclass(frozen=True)
class TargetWindow:
    dataset_id: int
    start: date
    train_days: int = 15
    validation_days: int = 15
    test_days: int = 180

    @property
    def observed_days(self) -> int:
        return self.train_days + self.validation_days

    @property
    def observed_end(self) -> date:
        return self.start + timedelta(days=self.observed_days - 1)

    @property
    def test_start(self) -> date:
        return self.observed_end + timedelta(days=1)

    @property
    def end(self) -> date:
        return self.start + timedelta(days=self.observed_days + self.test_days - 1)


@dataclass(frozen=True)
class SourcePretrainWindow:
    dataset_id: int
    start: date
    end: date
    days: int = 180


@dataclass(frozen=True)
class FeatureRoles:
    knn_observed: tuple[str, ...]
    model_historical: tuple[str, ...]
    future_known: tuple[str, ...]
    evaluation_only: tuple[str, ...]


_TARGET_STARTS = {
    1: date(2017, 6, 1),
    2: date(2018, 6, 1),
    3: date(2015, 1, 3),
    4: date(2024, 12, 16),
    5: date(2017, 1, 17),
    6: date(2015, 10, 26),
}


def _require_dataset_id(dataset_id: int) -> int:
    if type(dataset_id) is not int or dataset_id not in _TARGET_STARTS:
        raise ProtocolViolation(f"unsupported sealed dataset_id: {dataset_id!r}")
    return dataset_id


def _normalize_columns(columns: Iterable[str], declaration: str) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_column in columns:
        if not isinstance(raw_column, str) or not raw_column.strip():
            raise ProtocolViolation(f"{declaration} column names may not be empty")
        column = raw_column.strip()
        if column in seen:
            raise ProtocolViolation(f"duplicate {declaration} column: {column}")
        normalized.append(column)
        seen.add(column)
    return tuple(normalized)


def get_target_window(dataset_id: int) -> TargetWindow:
    dataset_id = _require_dataset_id(dataset_id)
    return TargetWindow(dataset_id=dataset_id, start=_TARGET_STARTS[dataset_id])


def get_source_pretrain_window(dataset_id: int) -> SourcePretrainWindow:
    target = get_target_window(dataset_id)
    end = target.observed_end
    return SourcePretrainWindow(
        dataset_id=target.dataset_id,
        start=end - timedelta(days=SOURCE_PRETRAIN_DAYS - 1),
        end=end,
    )


def classify_feature_roles(
    dataset_id: int,
    observed_numeric_columns: Iterable[str],
    blind_available_columns: Iterable[str] = (),
) -> FeatureRoles:
    """Classify observed numeric and blind-available fields without inference."""

    dataset_id = _require_dataset_id(dataset_id)
    observed = _normalize_columns(observed_numeric_columns, "observed")
    blind = _normalize_columns(blind_available_columns, "blind")
    observed_set = set(observed)
    blind_set = set(blind)
    future_allowed = DECLARED_FUTURE_KNOWN[dataset_id]

    for column in blind:
        if (
            column not in observed_set
            and column not in IDENTIFIER_COLUMNS
            and column != "sales"
            and column not in future_allowed
        ):
            raise ProtocolViolation(f"unclassified blind feature: {column}")

    ordered_columns = (*observed, *(column for column in blind if column not in observed_set))
    historical = tuple(column for column in observed if column not in IDENTIFIER_COLUMNS)
    future_known = tuple(column for column in ordered_columns if column in future_allowed)
    evaluation_only = tuple(
        column
        for column in ordered_columns
        if column == "sales"
        or (
            column in observed_set
            and column in blind_set
            and column not in future_allowed
            and column not in IDENTIFIER_COLUMNS
        )
    )

    return FeatureRoles(
        knn_observed=historical,
        model_historical=historical,
        future_known=future_known,
        evaluation_only=evaluation_only,
    )
