"""Frozen calendars, evaluation index, and hold tokens.

The evaluation index follows ``sita_ablation.windows.build_test_windows``:
every test day is a label, and the 10-day input ends ``horizon`` days earlier.
That is the opposite of scoring only windows that lie entirely inside the test
slice, which drops the first 10, 12, or 14 test days (170 / 168 / 166).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd

from additional_target_robustness.paths import ABLATION_ROOT

HOLD_TARGET_MANIFEST_MISMATCH = "HOLD_TARGET_MANIFEST_MISMATCH"
HOLD_SOURCE_CACHE_HASH_MISMATCH = "HOLD_SOURCE_CACHE_HASH_MISMATCH"
HOLD_HASH_MISMATCH = "HOLD_HASH_MISMATCH"
BLOCKED_HORIZON_MISMATCH = "BLOCKED_HORIZON_MISMATCH"
HOLD_HASH_MISMATCH = "HOLD_HASH_MISMATCH"
BLOCKED_HORIZON_MISMATCH = "BLOCKED_HORIZON_MISMATCH"
SAMPLING_SEED = 20261004
MANIFEST_SHA256 = "c83d03d337ef2b8c31d1e6b8aa0ea94f2b4ae8f7c6b312c39aedfdaf4035fec9"
SOURCE_AUDIT_SHA256 = "15ce7801b30bdc3c821f4be292e3864776c28df0b829294982435824c659a5e8"
WINDOW_SIZE = 10
TEST_DAYS = 180
TRAIN_DAYS = 15
VAL_DAYS = 15
TOP_K = 3
SEEDS = (42, 43, 44, 45, 46)
HORIZONS = (1, 3, 5)
METHODS = ("SITA", "SeasonalNaive", "PatchTST", "TimesNet")
D1_SOURCE_STORES = ("1",)
D1_SOURCE_ITEMS = tuple(str(item) for item in range(1, 10))
LEARNING_RATES = {"D1": 3e-5, "D4": 3e-5, "D5": 3e-4, "D6": 3e-5}


class ProtocolHold(RuntimeError):
    def __init__(self, token: str) -> None:
        super().__init__(token)
        self.token = token


class ProtocolError(RuntimeError):
    """A protocol check failed after the frozen files themselves were read."""


@dataclass(frozen=True)
class Calendar:
    dataset: str
    key_fields: tuple[str, ...]
    history_start: str
    history_end: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    model_features: tuple[str, ...]
    similarity_variables: tuple[str, ...]
    group_field: str | None
    source_filename: str
    target_filename: str

    @property
    def comparison_start(self) -> str:
        return self.train_start

    @property
    def comparison_end(self) -> str:
        return self.val_end


CALENDARS = {
    "D1": Calendar(
        "D1", ("store_id", "item_id"),
        "2017-01-02", "2017-06-30",
        "2017-06-01", "2017-06-15",
        "2017-06-16", "2017-06-30",
        "2017-07-01", "2017-12-27",
        ("sales", "year", "month", "week", "day"),
        ("sales",),
        None,
        "dataset1/source.parquet",
        "dataset1/source.parquet",
    ),
    "D4": Calendar(
        "D4", ("store_id", "product_id"),
        "2024-07-19", "2025-01-14",
        "2024-12-16", "2024-12-30",
        "2024-12-31", "2025-01-14",
        "2025-01-15", "2025-07-13",
        (
            "sales", "stock_hour6_22_cnt", "activity_flag", "discount", "holiday_flag",
            "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
            "year", "month", "week", "day",
        ),
        ("sales",),
        None,
        "dataset4/source.parquet",
        "dataset4/source.parquet",
    ),
    "D5": Calendar(
        "D5", ("store_nbr", "item_nbr"),
        "2016-08-20", "2017-02-15",
        "2017-01-17", "2017-01-31",
        "2017-02-01", "2017-02-15",
        "2017-02-16", "2017-08-14",
        (
            "sales", "year", "month", "week", "day", "class", "perishable",
            "cluster", "transactions", "oil_price", "is_holiday",
        ),
        ("sales", "onpromotion", "oil_price"),
        "family",
        "dataset5/prepared_180day_source.parquet",
        "dataset5/source.parquet",
    ),
    "D6": Calendar(
        "D6", ("store_id", "item_id"),
        "2015-05-29", "2015-11-24",
        "2015-10-26", "2015-11-09",
        "2015-11-10", "2015-11-24",
        "2015-11-25", "2016-05-22",
        (
            "sales", "year", "month", "week", "day", "wm_yr_wk", "weekday",
            "is_event_1", "is_event_2", "snap", "sell_price",
        ),
        ("sales",),
        "dept_id",
        "dataset6/source.parquet",
        "dataset6/source.parquet",
    ),
}
D5_TARGET_CUTOFF = "2017-08-14"
PAPER_TARGETS = {
    "D1": "1/10",
    "D4": "166/258",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}


@dataclass(frozen=True)
class ForecastRow:
    label_date: pd.Timestamp
    forecast_origin: pd.Timestamp
    input_start: pd.Timestamp
    input_end: pd.Timestamp
    input_dates: tuple[pd.Timestamp, ...]


@dataclass(frozen=True)
class EvaluationIndex:
    dataset: str
    horizon: int
    rows: tuple[ForecastRow, ...]

    @property
    def label_dates(self) -> tuple[pd.Timestamp, ...]:
        return tuple(row.label_date for row in self.rows)

    def label_date_text(self) -> tuple[str, ...]:
        return tuple(day.strftime("%Y-%m-%d") for day in self.label_dates)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_protocol_sha256() -> str:
    digest = hashlib.sha256()
    config_dir = ABLATION_ROOT / "configs"
    for name in ("features_frozen.json", "protocol_frozen.json", "tcn_frozen.json"):
        digest.update((config_dir / name).read_bytes())
    return digest.hexdigest()


def calendar_for(dataset: str) -> Calendar:
    try:
        return CALENDARS[dataset]
    except KeyError as exc:
        raise ProtocolHold("BLOCKED_PROTOCOL_MISMATCH") from exc


def build_evaluation_index(dataset: str, horizon: int) -> EvaluationIndex:
    if horizon not in HORIZONS:
        raise ProtocolHold("BLOCKED_PROTOCOL_MISMATCH")
    calendar = calendar_for(dataset)
    labels = tuple(pd.date_range(calendar.test_start, calendar.test_end, freq="D"))
    if len(labels) != TEST_DAYS:
        raise ProtocolHold("BLOCKED_PROTOCOL_MISMATCH")
    rows = []
    for label in labels:
        label = pd.Timestamp(label).normalize()
        input_end = label - pd.Timedelta(days=int(horizon))
        input_start = input_end - pd.Timedelta(days=WINDOW_SIZE - 1)
        input_dates = tuple(pd.date_range(input_start, input_end, freq="D"))
        if len(input_dates) != WINDOW_SIZE:
            raise ProtocolHold("BLOCKED_PROTOCOL_MISMATCH")
        if input_dates[-1] != input_end or input_end >= label:
            raise ProtocolHold("BLOCKED_PROTOCOL_MISMATCH")
        if any(day > input_end or day >= label for day in input_dates):
            raise ProtocolHold("BLOCKED_PROTOCOL_MISMATCH")
        rows.append(
            ForecastRow(
                label_date=label,
                forecast_origin=input_end,
                input_start=input_start,
                input_end=input_end,
                input_dates=input_dates,
            )
        )
    if rows[0].label_date != pd.Timestamp(calendar.test_start):
        raise ProtocolHold("BLOCKED_PROTOCOL_MISMATCH")
    if rows[-1].label_date != pd.Timestamp(calendar.test_end):
        raise ProtocolHold("BLOCKED_PROTOCOL_MISMATCH")
    return EvaluationIndex(dataset, int(horizon), tuple(rows))


def method_label_dates(method: str, index: EvaluationIndex) -> tuple[str, ...]:
    """Every method scores the same frozen index. None keeps a private date list."""
    if method not in METHODS:
        raise ProtocolHold("BLOCKED_PROTOCOL_MISMATCH")
    return index.label_date_text()


def assert_file_hash(path: Path, expected: str, *, token: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ProtocolHold(token)
    return actual
