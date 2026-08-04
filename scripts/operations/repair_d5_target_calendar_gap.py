"""Fail-closed repair of the sealed D5 target calendar gap.

This operator only repairs the five formal D5 target entities and only the
fifteen pre-declared entity-date keys.  It never changes runner behavior or
calendarizes a frame at runtime.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_RELATIVE = Path("数据集/固化数据/d1_d6_sealed_v1/dataset5/target.parquet")
DEFAULT_AUTHORITY_PATH = ROOT / AUTHORITY_RELATIVE
DEFAULT_REPAIR_ROOT = ROOT / "outputs" / "repairs"

OLD_SHA256 = "89df965859b3b563d178c0341039acc44ad6192a53196c8974f256ebd400edff"
OLD_TOTAL_ROWS = 7323
FORMAL_START = pd.Timestamp("2017-01-17")
FORMAL_END = pd.Timestamp("2017-08-14")
BLIND_START = pd.Timestamp("2017-02-16")
BLIND_END = pd.Timestamp("2017-08-14")
TARGET_KEYS = (
    (48, 364606),
    (48, 1159415),
    (48, 1159414),
    (48, 1349808),
    (48, 320682),
)
MISSING_DATES = tuple(
    pd.Timestamp(value)
    for value in (
        "2017-07-15",
        "2017-07-16",
        "2017-07-17",
        "2017-07-18",
        "2017-07-19",
        "2017-07-20",
        "2017-07-21",
        "2017-07-22",
        "2017-07-23",
        "2017-07-24",
        "2017-07-25",
        "2017-07-26",
        "2017-07-27",
        "2017-07-28",
        "2017-07-31",
    )
)

ITEM_STATIC_FIELDS = ("family", "class", "perishable")
ENTITY_STATIC_FIELDS = ("entity_id", "item_id")
DONOR_FIELDS = (
    "year",
    "month",
    "week",
    "day",
    "city",
    "state",
    "type",
    "cluster",
    "transactions",
    "oil_price",
    "is_holiday",
)
SUPPORTED_FIELDS = {
    "date",
    "store_nbr",
    "item_nbr",
    "sales",
    "onpromotion",
    *ITEM_STATIC_FIELDS,
    *ENTITY_STATIC_FIELDS,
    *DONOR_FIELDS,
}
REPAIR_KEY = (48, 1159415)


class RepairBlocked(RuntimeError):
    """Raised when the authority is not exactly in the declared repair state."""


@dataclass(frozen=True)
class RepairSpec:
    authority_path: Path
    expected_sha256: str = OLD_SHA256
    expected_rows: int = OLD_TOTAL_ROWS
    target_keys: tuple[tuple[int, int], ...] = TARGET_KEYS
    missing_dates: tuple[pd.Timestamp, ...] = MISSING_DATES
    formal_start: pd.Timestamp = FORMAL_START
    formal_end: pd.Timestamp = FORMAL_END
    blind_start: pd.Timestamp = BLIND_START
    blind_end: pd.Timestamp = BLIND_END

    def __post_init__(self) -> None:
        object.__setattr__(self, "authority_path", Path(self.authority_path))
        for name in ("formal_start", "formal_end", "blind_start", "blind_end"):
            value = pd.Timestamp(getattr(self, name)).normalize()
            if value.tz is not None:
                raise ValueError(f"{name} must be timezone-naive")
            object.__setattr__(self, name, value)
        dates = tuple(pd.Timestamp(value).normalize() for value in self.missing_dates)
        if not dates or len(set(dates)) != len(dates) or tuple(sorted(dates)) != dates:
            raise ValueError("missing_dates must be unique and sorted")
        object.__setattr__(self, "missing_dates", dates)


@dataclass(frozen=True)
class RepairPlan:
    spec: RepairSpec
    mode: str
    source_sha256: str
    source_rows: int
    missing_exact_keys: tuple[tuple[int, int, pd.Timestamp], ...]
    added_rows: pd.DataFrame
    old_formal_rows: int
    old_blind_rows: int


DEFAULT_SPEC = RepairSpec(authority_path=DEFAULT_AUTHORITY_PATH)


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _value_token(value: object) -> tuple[str, str]:
    if _is_missing(value):
        return ("missing", "")
    if isinstance(value, pd.Timestamp):
        return ("timestamp", value.isoformat())
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    return (type(value).__name__, repr(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key_set(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (int(store_nbr), int(item_nbr))
        for store_nbr, item_nbr in frame[["store_nbr", "item_nbr"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }


def _normalize_dates(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise RepairBlocked("target contains invalid dates")
    if getattr(dates.dt, "tz", None) is not None:
        raise RepairBlocked("target dates must be timezone-naive")
    normalized = dates.dt.normalize()
    if not dates.equals(normalized):
        raise RepairBlocked("target dates are not normalized natural days")
    return normalized


def _unique_non_null(frame: pd.DataFrame, column: str, context: str) -> object:
    values = frame[column].tolist()
    non_null = [value for value in values if not _is_missing(value)]
    tokens = {_value_token(value) for value in non_null}
    if not non_null:
        raise RepairBlocked(f"donor {column} is missing for {context}")
    if len(tokens) != 1:
        rendered = [repr(value) for value in non_null]
        raise RepairBlocked(f"donor {column} is inconsistent for {context}: {rendered}")
    return non_null[0]


def _entity_value(frame: pd.DataFrame, column: str, entity: tuple[int, int]) -> object:
    return _unique_non_null(frame, column, f"entity {entity[0]}/{entity[1]}")


def _logical_zero_value(
    table: pa.Table,
    donors: pd.DataFrame,
    context: str,
) -> object:
    value = _unique_non_null(donors, "onpromotion", context)
    field_type = table.schema.field("onpromotion").type
    if pa.types.is_boolean(field_type):
        if value is not False:
            raise RepairBlocked(f"onpromotion donor is not false for {context}: {value!r}")
        return False
    if pa.types.is_integer(field_type) or pa.types.is_floating(field_type):
        if float(value) != 0.0:
            raise RepairBlocked(f"onpromotion donor is not zero for {context}: {value!r}")
        return 0
    if pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        if str(value).strip().lower() not in {"false", "0"}:
            raise RepairBlocked(f"onpromotion donor is not false for {context}: {value!r}")
        return value
    raise RepairBlocked(f"unsupported onpromotion schema: {field_type}")


def _calendar_stats(frame: pd.DataFrame, spec: RepairSpec) -> dict[str, Any]:
    dates = _normalize_dates(frame)
    expected_dates = pd.date_range(spec.formal_start, spec.formal_end, freq="D")
    formal_mask = dates.between(spec.formal_start, spec.formal_end, inclusive="both")
    formal = frame.loc[formal_mask]
    formal_dates = dates.loc[formal_mask]
    formal_key_dates = {
        (int(store_nbr), int(item_nbr), date): None
        for store_nbr, item_nbr, date in zip(
            formal["store_nbr"], formal["item_nbr"], formal_dates
        )
    }
    expected_key_dates = {
        (store_nbr, item_nbr, date): None
        for store_nbr, item_nbr in spec.target_keys
        for date in expected_dates
    }
    missing = tuple(
        (store_nbr, item_nbr, date)
        for store_nbr, item_nbr, date in sorted(
            set(expected_key_dates).difference(formal_key_dates)
        )
    )
    unexpected_keys = sorted(
        {
            (int(store_nbr), int(item_nbr))
            for store_nbr, item_nbr in formal[["store_nbr", "item_nbr"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        }.difference(set(spec.target_keys))
    )
    duplicate_count = int(frame.duplicated(["store_nbr", "item_nbr", "date"]).sum())
    return {
        "formal_rows": int(len(formal)),
        "blind_rows": int(
            dates.between(spec.blind_start, spec.blind_end, inclusive="both").sum()
        ),
        "missing_exact_keys": missing,
        "unexpected_keys": unexpected_keys,
        "duplicate_exact_keys": duplicate_count,
        "expected_formal_rows": int(len(expected_dates) * len(spec.target_keys)),
        "expected_blind_rows": int(
            (spec.blind_end - spec.blind_start).days + 1
        )
        * len(spec.target_keys),
    }


def _validate_shape(frame: pd.DataFrame, table: pa.Table) -> None:
    missing = sorted(SUPPORTED_FIELDS.difference(frame.columns))
    unsupported = sorted(set(frame.columns).difference(SUPPORTED_FIELDS))
    if missing:
        raise RepairBlocked(f"target is missing required columns: {missing}")
    if unsupported:
        raise RepairBlocked(f"unsupported target columns: {unsupported}")
    if list(frame.columns) != list(table.column_names):
        raise RepairBlocked("target column order does not match Arrow schema")


def _build_added_rows(
    frame: pd.DataFrame,
    table: pa.Table,
    spec: RepairSpec,
) -> pd.DataFrame:
    entity_mask = (frame["store_nbr"] == REPAIR_KEY[0]) & frame["item_nbr"].eq(REPAIR_KEY[1])
    entity_rows = frame.loc[entity_mask]
    if entity_rows.empty:
        raise RepairBlocked("repair entity 48/1159415 is absent")
    static = {
        column: _entity_value(entity_rows, column, REPAIR_KEY)
        for column in (*ITEM_STATIC_FIELDS, *ENTITY_STATIC_FIELDS)
    }
    rows: list[dict[str, object]] = []
    for date in spec.missing_dates:
        donor_mask = (
            frame["store_nbr"].eq(REPAIR_KEY[0])
            & frame["date"].eq(date)
            & frame["item_nbr"].ne(REPAIR_KEY[1])
        )
        donors = frame.loc[donor_mask]
        context = f"store 48 date {date.date().isoformat()}"
        if donors.empty:
            raise RepairBlocked(f"donor rows are absent for {context}")
        donor_values = {
            column: _unique_non_null(donors, column, context)
            for column in DONOR_FIELDS
        }
        row = {column: pd.NA for column in table.column_names}
        row.update(
            {
                "date": date,
                "store_nbr": REPAIR_KEY[0],
                "item_nbr": REPAIR_KEY[1],
                "sales": 0.0,
                "onpromotion": _logical_zero_value(table, donors, context),
                **static,
                **donor_values,
            }
        )
        if any(_is_missing(row[column]) for column in table.column_names):
            missing = [column for column in table.column_names if _is_missing(row[column])]
            raise RepairBlocked(f"repair row has unresolved fields for {context}: {missing}")
        rows.append(row)
    additions = pd.DataFrame(rows, columns=table.column_names)
    for column, dtype in frame.dtypes.items():
        try:
            additions[column] = additions[column].astype(dtype)
        except (TypeError, ValueError) as exc:
            raise RepairBlocked(
                f"repair additions changed dtype for {column}: expected {dtype}"
            ) from exc
    return additions


def inspect_authority(
    table: pa.Table,
    *,
    spec: RepairSpec = DEFAULT_SPEC,
) -> RepairPlan:
    """Validate the exact pre-repair state and construct only the 15 additions."""

    frame = table.to_pandas()
    _validate_shape(frame, table)
    dates = _normalize_dates(frame)
    frame = frame.copy()
    frame["date"] = dates
    actual_keys = _key_set(frame)
    expected_keys = set(spec.target_keys)
    if actual_keys != expected_keys:
        raise RepairBlocked(
            f"target entity set mismatch: actual={sorted(actual_keys)} expected={sorted(expected_keys)}"
        )
    stats = _calendar_stats(frame, spec)
    if stats["unexpected_keys"]:
        raise RepairBlocked(f"unexpected formal target keys: {stats['unexpected_keys']}")
    if stats["duplicate_exact_keys"]:
        raise RepairBlocked(
            f"duplicate exact keys: {stats['duplicate_exact_keys']}"
        )

    source_sha = _sha256(spec.authority_path)
    complete = (
        not stats["missing_exact_keys"]
        and stats["formal_rows"] == stats["expected_formal_rows"]
        and stats["blind_rows"] == stats["expected_blind_rows"]
    )
    if complete:
        empty = pd.DataFrame(columns=table.column_names)
        return RepairPlan(
            spec=spec,
            mode="already_complete",
            source_sha256=source_sha,
            source_rows=len(table),
            missing_exact_keys=(),
            added_rows=empty,
            old_formal_rows=stats["formal_rows"],
            old_blind_rows=stats["blind_rows"],
        )
    if source_sha != spec.expected_sha256:
        raise RepairBlocked(
            f"old SHA-256 mismatch: actual={source_sha} expected={spec.expected_sha256}"
        )
    if len(table) != spec.expected_rows:
        raise RepairBlocked(
            f"old row count mismatch: actual={len(table)} expected={spec.expected_rows}"
        )
    expected_missing = tuple(
        (REPAIR_KEY[0], REPAIR_KEY[1], date) for date in spec.missing_dates
    )
    if stats["missing_exact_keys"] != expected_missing:
        rendered = [
            [store_nbr, item_nbr, date.strftime("%Y-%m-%d")]
            for store_nbr, item_nbr, date in stats["missing_exact_keys"]
        ]
        raise RepairBlocked(f"missing exact key set does not match declaration: {rendered}")
    added_rows = _build_added_rows(frame, table, spec)
    return RepairPlan(
        spec=spec,
        mode="repair",
        source_sha256=source_sha,
        source_rows=len(table),
        missing_exact_keys=stats["missing_exact_keys"],
        added_rows=added_rows,
        old_formal_rows=stats["formal_rows"],
        old_blind_rows=stats["blind_rows"],
    )


def build_candidate(table: pa.Table, plan: RepairPlan) -> pa.Table:
    """Append the planned rows using the original Arrow schema and metadata."""

    if plan.mode == "already_complete":
        return table
    additions = pa.Table.from_pandas(
        plan.added_rows,
        schema=table.schema,
        preserve_index=False,
    )
    if not additions.schema.equals(table.schema, check_metadata=True):
        raise RepairBlocked("repair additions changed the authority schema")
    return pa.concat_tables([table, additions])


def verify_candidate(
    original: pa.Table,
    candidate: pa.Table,
    plan: RepairPlan,
) -> dict[str, Any]:
    """Verify semantic counts, exact keys, old-row identity, and schema preservation."""

    if not candidate.schema.equals(original.schema, check_metadata=True):
        raise RepairBlocked("candidate schema or metadata differs from original")
    if not candidate.slice(0, len(original)).equals(original):
        raise RepairBlocked("old rows changed")
    old_frame = original.to_pandas()
    new_frame = candidate.to_pandas()
    old_nulls = int(old_frame.isna().sum().sum())
    new_nulls = int(new_frame.isna().sum().sum())
    if new_nulls != old_nulls:
        raise RepairBlocked(f"candidate introduced nulls: old={old_nulls} new={new_nulls}")
    stats = _calendar_stats(new_frame, plan.spec)
    if stats["missing_exact_keys"] or stats["unexpected_keys"]:
        raise RepairBlocked(
            f"candidate formal calendar is incomplete: missing={stats['missing_exact_keys']} "
            f"unexpected={stats['unexpected_keys']}"
        )
    if stats["duplicate_exact_keys"]:
        raise RepairBlocked(
            f"candidate has duplicate exact keys: {stats['duplicate_exact_keys']}"
        )
    if len(candidate) != plan.source_rows + len(plan.added_rows):
        raise RepairBlocked("candidate row count does not equal old rows plus 15 additions")
    added_frame = candidate.slice(len(original)).to_pandas()
    if len(added_frame) != len(plan.added_rows):
        raise RepairBlocked("candidate addition count mismatch")
    if len(added_frame) and not added_frame["sales"].eq(0).all():
        raise RepairBlocked("candidate additions have nonzero sales")
    if len(added_frame) and not added_frame["onpromotion"].map(
        lambda value: str(value).strip().lower() in {"false", "0"}
    ).all():
        raise RepairBlocked("candidate additions have nonzero onpromotion")
    if set(
        zip(
            added_frame["store_nbr"],
            added_frame["item_nbr"],
            added_frame["date"],
        )
    ) != set(plan.missing_exact_keys):
        raise RepairBlocked("candidate additions do not equal the declared missing keys")
    return {
        "new_total_rows": int(len(candidate)),
        "new_formal_rows": stats["formal_rows"],
        "new_blind_rows": stats["blind_rows"],
        "new_target_entities": len(_key_set(new_frame)),
        "target_days_per_entity": {
            f"{store_nbr}/{item_nbr}": int(
                new_frame.loc[
                    (new_frame["store_nbr"] == store_nbr)
                    & (new_frame["item_nbr"] == item_nbr)
                    & new_frame["date"].between(
                        plan.spec.formal_start,
                        plan.spec.formal_end,
                        inclusive="both",
                    ),
                    "date",
                ].nunique()
            )
            for store_nbr, item_nbr in plan.spec.target_keys
        },
        "missing_exact_keys_after": len(stats["missing_exact_keys"]),
        "duplicate_exact_keys_after": stats["duplicate_exact_keys"],
        "unexpected_target_keys_after": len(stats["unexpected_keys"]),
        "formal_window_outside_added_rows": 0,
        "added_rows": len(added_frame),
        "old_rows_changed": 0,
        "schema_preserved": True,
    }


def _report_payload(
    plan: RepairPlan,
    report: dict[str, Any],
    *,
    status: str,
    new_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "authority_path": str(plan.spec.authority_path),
        "old_sha256": plan.source_sha256,
        "new_sha256": new_sha256,
        "old_total_rows": plan.source_rows,
        "new_total_rows": report.get("new_total_rows", plan.source_rows),
        "old_formal_rows": plan.old_formal_rows,
        "new_formal_rows": report.get("new_formal_rows", plan.old_formal_rows),
        "old_blind_rows": plan.old_blind_rows,
        "new_blind_rows": report.get("new_blind_rows", plan.old_blind_rows),
        "added_rows": report.get("added_rows", 0),
        "store_nbr": REPAIR_KEY[0],
        "item_nbr": REPAIR_KEY[1],
        "missing_dates": [date.strftime("%Y-%m-%d") for date in MISSING_DATES],
        "sales_fill": 0,
        "onpromotion_fill": 0,
        "onpromotion_storage_value": "False",
        "donor_source": "same_date_store_48_other_existing_item_rows",
        "missing_exact_keys_after": report.get("missing_exact_keys_after", 0),
        "duplicate_exact_keys_after": report.get("duplicate_exact_keys_after", 0),
        "old_rows_changed": report.get("old_rows_changed", 0),
        "schema_preserved": report.get("schema_preserved", True),
        "verification": report,
    }


def _write_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="snappy")


def _unique_repair_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / f"d5_target_calendar_{stamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=False, exist_ok=False)
    return path


def apply_repair(
    *,
    spec: RepairSpec = DEFAULT_SPEC,
    repair_root: Path = DEFAULT_REPAIR_ROOT,
) -> dict[str, Any]:
    """Revalidate, write a candidate/backup, then atomically replace authority."""

    table = pq.read_table(spec.authority_path)
    plan = inspect_authority(table, spec=spec)
    if plan.mode == "already_complete":
        report = _report_payload(
            plan,
            {
                "new_total_rows": len(table),
                "new_formal_rows": plan.old_formal_rows,
                "new_blind_rows": plan.old_blind_rows,
                "added_rows": 0,
                "missing_exact_keys_after": 0,
                "duplicate_exact_keys_after": 0,
                "old_rows_changed": 0,
                "schema_preserved": True,
            },
            status="D5_TARGET_CALENDAR_REPAIR_ALREADY_APPLIED",
            new_sha256=plan.source_sha256,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return report

    candidate = build_candidate(table, plan)
    report = verify_candidate(table, candidate, plan)
    repair_dir = _unique_repair_dir(Path(repair_root))
    before_path = repair_dir / "target.parquet.before_repair"
    candidate_path = repair_dir / "target.parquet.repaired"
    report_path = repair_dir / "repair-report.json"
    shutil.copy2(spec.authority_path, before_path)
    _write_parquet(candidate, candidate_path)
    candidate_from_disk = pq.read_table(candidate_path)
    report = verify_candidate(table, candidate_from_disk, plan)
    new_sha256 = _sha256(candidate_path)

    temporary = spec.authority_path.with_name(
        f".{spec.authority_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        shutil.copyfile(candidate_path, temporary)
        os.replace(temporary, spec.authority_path)
        if _sha256(spec.authority_path) != new_sha256:
            raise RepairBlocked("atomic authority replacement SHA-256 mismatch")
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    final_report = _report_payload(
        plan,
        report,
        status="D5_TARGET_CALENDAR_REPAIR_APPLIED",
        new_sha256=new_sha256,
    )
    report_path.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True))
    return final_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--authority-path", type=Path, default=DEFAULT_AUTHORITY_PATH)
    parser.add_argument("--repair-root", type=Path, default=DEFAULT_REPAIR_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = RepairSpec(authority_path=args.authority_path)
    try:
        table = pq.read_table(spec.authority_path)
        plan = inspect_authority(table, spec=spec)
        if args.check_only:
            candidate = build_candidate(table, plan)
            report = verify_candidate(table, candidate, plan)
            payload = _report_payload(
                plan,
                report,
                status=(
                    "D5_TARGET_CALENDAR_REPAIR_READY"
                    if plan.mode == "repair"
                    else "D5_TARGET_CALENDAR_REPAIR_ALREADY_APPLIED"
                ),
                new_sha256=(plan.source_sha256 if plan.mode == "already_complete" else None),
            )
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        apply_repair(spec=spec, repair_root=args.repair_root)
        return 0
    except (RepairBlocked, FileNotFoundError, OSError, pa.ArrowException) as exc:
        print(json.dumps({"status": "D5_TARGET_CALENDAR_REPAIR_BLOCKED", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
