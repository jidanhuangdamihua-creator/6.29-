"""Authoritative reconstruction of missing natural days in D5 targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Mapping

import pandas as pd


ITEM_STATIC_COLUMNS = ("family", "class", "perishable")
STORE_STATIC_COLUMNS = ("city", "state", "type", "cluster")
DATE_COLUMNS = ("year", "month", "week", "day")
HISTORICAL_DERIVED_PATTERN = re.compile(r"(?:^|_)(?:lag|lags|rolling|roll)(?:_|$)", re.I)


@dataclass(frozen=True)
class AuthorityFileEvidence:
    path: str
    sha256: str
    size_bytes: int
    used: bool


@dataclass(frozen=True)
class D5AuthorityBundle:
    oil_by_date: pd.DataFrame
    transactions_by_store_date: pd.DataFrame
    items_by_item: pd.DataFrame
    stores_by_store: pd.DataFrame
    holidays_by_store_date: pd.DataFrame | None
    files: Mapping[str, AuthorityFileEvidence]


@dataclass(frozen=True)
class D5FieldReconstructionStats:
    field: str
    authority: str
    lookup_keys: tuple[str, ...]
    restored_count: int
    zero_fill_count: int
    missing_lookup_count: int


@dataclass(frozen=True)
class D5ReconstructionReport:
    synthetic_entity_date_keys: tuple[tuple[str, str], ...]
    synthetic_row_count: int
    field_stats: tuple[D5FieldReconstructionStats, ...]
    original_row_count: int
    original_rows_unchanged: bool
    missing_lookups: tuple[str, ...]
    authority_files: Mapping[str, AuthorityFileEvidence]

    def to_dict(self) -> dict[str, object]:
        return {
            "synthetic_entity_date_keys": [list(key) for key in self.synthetic_entity_date_keys],
            "synthetic_row_count": self.synthetic_row_count,
            "field_stats": [
                {
                    "field": item.field,
                    "authority": item.authority,
                    "lookup_keys": list(item.lookup_keys),
                    "restored_count": item.restored_count,
                    "zero_fill_count": item.zero_fill_count,
                    "missing_lookup_count": item.missing_lookup_count,
                }
                for item in self.field_stats
            ],
            "original_row_count": self.original_row_count,
            "original_rows_unchanged": self.original_rows_unchanged,
            "missing_lookups": list(self.missing_lookups),
            "authority_files": {
                name: {
                    "path": evidence.path,
                    "sha256": evidence.sha256,
                    "size_bytes": evidence.size_bytes,
                    "used": evidence.used,
                }
                for name, evidence in sorted(self.authority_files.items())
            },
        }


def _validate_expected_dates(expected_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(expected_dates, errors="coerce"))
    if dates.empty:
        raise ValueError("expected_dates must be nonempty")
    if dates.isna().any():
        raise ValueError("expected_dates must not contain invalid dates")
    if dates.tz is not None:
        raise ValueError("expected_dates must be timezone-naive")
    if not dates.equals(dates.normalize()):
        raise ValueError("expected_dates must contain normalized natural days")
    if not dates.is_unique:
        raise ValueError("expected_dates must be unique")
    if not dates.is_monotonic_increasing:
        raise ValueError("expected_dates must be strictly increasing")
    return dates


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], authority: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{authority} authority missing columns: {sorted(missing)}")


def _reject_duplicate_keys(frame: pd.DataFrame, keys: list[str], authority: str) -> None:
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, keys].head(5).to_dict(orient="records")
        raise ValueError(f"duplicate authority key in {authority}: {examples}")


def _normalize_integer_key(series: pd.Series, authority: str, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.map(lambda value: float(value).is_integer()).all():
        raise ValueError(f"{authority} authority has invalid {column}")
    return values.astype("int64")


def build_authoritative_d5_oil_by_date(
    oil: pd.DataFrame,
    *,
    expected_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Align raw oil to natural days, forward-fill, then lag exactly one day."""
    dates = _validate_expected_dates(expected_dates)
    _require_columns(oil, ("date", "dcoilwtico"), "oil")
    normalized = oil.loc[:, ["date", "dcoilwtico"]].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    if normalized["date"].isna().any():
        raise ValueError("oil authority has invalid date")
    _reject_duplicate_keys(normalized, ["date"], "oil")
    normalized["dcoilwtico"] = pd.to_numeric(normalized["dcoilwtico"], errors="coerce")
    aligned = normalized.set_index("date").reindex(dates)
    aligned["dcoilwtico"] = aligned["dcoilwtico"].ffill()
    aligned["oil_price"] = aligned["dcoilwtico"].shift(1)
    return aligned[["oil_price"]].rename_axis("date").reset_index()


def _file_evidence(path: Path, *, used: bool) -> AuthorityFileEvidence:
    if not used:
        return AuthorityFileEvidence(str(path), "", 0, False)
    payload = path.read_bytes()
    return AuthorityFileEvidence(
        path=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        used=True,
    )


def _normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("date", "store_nbr", "transactions"), "transactions")
    out = frame.loc[:, ["date", "store_nbr", "transactions"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if out["date"].isna().any():
        raise ValueError("transactions authority has invalid date")
    out["store_nbr"] = _normalize_integer_key(out["store_nbr"], "transactions", "store_nbr")
    out["transactions"] = pd.to_numeric(out["transactions"], errors="coerce")
    if out["transactions"].isna().any():
        raise ValueError("transactions authority has invalid values")
    _reject_duplicate_keys(out, ["store_nbr", "date"], "transactions")
    return out.sort_values(["store_nbr", "date"], kind="stable").reset_index(drop=True)


def _normalize_items(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("item_nbr",) + ITEM_STATIC_COLUMNS, "items")
    out = frame.loc[:, ["item_nbr", *ITEM_STATIC_COLUMNS]].copy()
    out["item_nbr"] = _normalize_integer_key(out["item_nbr"], "items", "item_nbr")
    _reject_duplicate_keys(out, ["item_nbr"], "items")
    return out.sort_values("item_nbr", kind="stable").reset_index(drop=True)


def _normalize_stores(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("store_nbr",) + STORE_STATIC_COLUMNS, "stores")
    out = frame.loc[:, ["store_nbr", *STORE_STATIC_COLUMNS]].copy()
    out["store_nbr"] = _normalize_integer_key(out["store_nbr"], "stores", "store_nbr")
    _reject_duplicate_keys(out, ["store_nbr"], "stores")
    return out.sort_values("store_nbr", kind="stable").reset_index(drop=True)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    values = series.astype("string").str.strip().str.lower()
    mapped = values.map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        raise ValueError("holidays authority has invalid transferred values")
    return mapped.astype(bool)


def _normalize_holidays(frame: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        frame,
        ("date", "type", "locale", "locale_name", "transferred"),
        "holidays",
    )
    holidays = frame.copy()
    holidays["date"] = pd.to_datetime(holidays["date"], errors="coerce").dt.normalize()
    if holidays["date"].isna().any():
        raise ValueError("holidays authority has invalid date")
    holiday_type = holidays["type"].astype("string")
    transferred = _as_bool(holidays["transferred"])
    effective = (
        holiday_type.eq("Transfer")
        | (holiday_type.isin(["Holiday", "Additional", "Bridge"]) & ~transferred)
    ) & ~holiday_type.eq("Work Day")
    holidays = holidays.loc[effective].copy()

    pieces: list[pd.DataFrame] = []
    national = holidays[holidays["locale"].eq("National")]
    if not national.empty:
        pieces.append(stores[["store_nbr"]].merge(national[["date"]], how="cross"))
    regional = holidays[holidays["locale"].eq("Regional")]
    if not regional.empty:
        pieces.append(
            stores[["store_nbr", "state"]]
            .merge(
                regional[["date", "locale_name"]],
                left_on="state",
                right_on="locale_name",
                how="inner",
            )[["store_nbr", "date"]]
        )
    local = holidays[holidays["locale"].eq("Local")]
    if not local.empty:
        pieces.append(
            stores[["store_nbr", "city"]]
            .merge(
                local[["date", "locale_name"]],
                left_on="city",
                right_on="locale_name",
                how="inner",
            )[["store_nbr", "date"]]
        )
    if not pieces:
        return pd.DataFrame(columns=["store_nbr", "date", "is_holiday"])
    out = pd.concat(pieces, ignore_index=True).drop_duplicates(["store_nbr", "date"])
    out["is_holiday"] = 1
    return out.sort_values(["store_nbr", "date"], kind="stable").reset_index(drop=True)


def load_d5_authorities(
    raw_dir: Path,
    *,
    use_holidays: bool,
) -> D5AuthorityBundle:
    """Load and normalize each D5 authority once for one dataset process."""
    root = Path(raw_dir)
    paths = {
        "oil": root / "oil.csv",
        "transactions": root / "transactions.csv",
        "items": root / "items.csv",
        "stores": root / "stores.csv",
        "holidays": root / "holidays_events.csv",
    }
    required = [paths[name] for name in ("oil", "transactions", "items", "stores")]
    if use_holidays:
        required.append(paths["holidays"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing D5 authority files: {missing}")

    raw_oil = pd.read_csv(paths["oil"])
    _require_columns(raw_oil, ("date", "dcoilwtico"), "oil")
    oil_dates = pd.to_datetime(raw_oil["date"], errors="coerce").dropna().dt.normalize()
    if oil_dates.empty:
        raise ValueError("oil authority has no valid dates")
    oil_by_date = build_authoritative_d5_oil_by_date(
        raw_oil,
        expected_dates=pd.date_range(oil_dates.min(), oil_dates.max(), freq="D"),
    )
    transactions = _normalize_transactions(pd.read_csv(paths["transactions"]))
    items = _normalize_items(pd.read_csv(paths["items"]))
    stores = _normalize_stores(pd.read_csv(paths["stores"]))
    holidays = (
        _normalize_holidays(pd.read_csv(paths["holidays"]), stores)
        if use_holidays
        else None
    )
    evidence = {
        name: _file_evidence(path, used=(name != "holidays" or use_holidays))
        for name, path in paths.items()
    }
    return D5AuthorityBundle(
        oil_by_date=oil_by_date,
        transactions_by_store_date=transactions,
        items_by_item=items,
        stores_by_store=stores,
        holidays_by_store_date=holidays,
        files=evidence,
    )


def _single_key(frame: pd.DataFrame, column: str, entity: object) -> object:
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"D5 entity {entity!r} must have exactly one {column}: {values}")
    return values[0]


def _lookup_one(
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    authority: str,
    identity: str,
) -> pd.Series:
    matches = frame.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"missing {authority} authority lookup for {identity}")
    return matches.iloc[0]


def _cast_like_original(frame: pd.DataFrame, dtypes: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    for column, dtype in dtypes.items():
        try:
            out[column] = out[column].astype(dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"D5 reconstruction changed dtype contract for {column}: {dtype}") from exc
    return out


def reconstruct_d5_target_calendar(
    observed: pd.DataFrame,
    *,
    date_col: str,
    entity_col: str,
    expected_dates: pd.DatetimeIndex,
    authorities: D5AuthorityBundle,
) -> tuple[pd.DataFrame, D5ReconstructionReport]:
    """Add only missing entity-days using field-specific D5 authorities."""
    dates = _validate_expected_dates(expected_dates)
    required = {date_col, entity_col, "store_nbr", "item_nbr", "sales"}
    missing = required.difference(observed.columns)
    if missing:
        raise ValueError(f"D5 target missing reconstruction columns: {sorted(missing)}")
    if observed.empty:
        raise ValueError("D5 target has no observed rows to reconstruct")
    historical = [col for col in observed.columns if HISTORICAL_DERIVED_PATTERN.search(str(col))]
    if historical:
        raise ValueError(
            f"D5 historical derived fields cannot be synthesized: {sorted(historical)}"
        )

    original = observed.copy(deep=True)
    original_attrs = observed.attrs.copy()
    working = observed.copy(deep=True)
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce").dt.normalize()
    if working[date_col].isna().any():
        raise ValueError("D5 target contains invalid dates")
    if working.duplicated([entity_col, date_col]).any():
        raise ValueError("D5 target contains duplicate entity-date keys")
    outside = working.loc[~working[date_col].isin(dates), [entity_col, date_col]]
    if not outside.empty:
        raise ValueError("D5 target contains dates outside expected_dates")

    known_columns = {
        date_col,
        entity_col,
        "item_id",
        "store_nbr",
        "item_nbr",
        "sales",
        "onpromotion",
        *ITEM_STATIC_COLUMNS,
        *STORE_STATIC_COLUMNS,
        "transactions",
        "oil_price",
        "is_holiday",
        *DATE_COLUMNS,
    }
    unsupported = set(working.columns).difference(known_columns)
    if unsupported:
        raise ValueError(f"D5 target has fields without reconstruction authority: {sorted(unsupported)}")

    oil = authorities.oil_by_date
    tx = authorities.transactions_by_store_date
    items = authorities.items_by_item
    stores = authorities.stores_by_store
    synthetic_rows: list[dict[str, object]] = []
    synthetic_keys: list[tuple[str, str]] = []
    entity_order = working[entity_col].drop_duplicates().tolist()

    for entity in entity_order:
        entity_rows = working.loc[working[entity_col].eq(entity)]
        store_nbr = int(_single_key(entity_rows, "store_nbr", entity))
        item_nbr = int(_single_key(entity_rows, "item_nbr", entity))
        item_authority = _lookup_one(
            items,
            items["item_nbr"].eq(item_nbr),
            authority="items",
            identity=f"item_nbr={item_nbr}",
        )
        store_authority = _lookup_one(
            stores,
            stores["store_nbr"].eq(store_nbr),
            authority="stores",
            identity=f"store_nbr={store_nbr}",
        )
        missing_dates = dates.difference(pd.DatetimeIndex(entity_rows[date_col]))
        for day in missing_dates:
            identity = f"store_nbr={store_nbr} date={day.date().isoformat()}"
            tx_authority = _lookup_one(
                tx,
                tx["store_nbr"].eq(store_nbr) & tx["date"].eq(day),
                authority="transactions",
                identity=identity,
            )
            oil_authority = _lookup_one(
                oil,
                oil["date"].eq(day),
                authority="oil",
                identity=f"date={day.date().isoformat()}",
            )
            if pd.isna(oil_authority["oil_price"]):
                raise ValueError(f"missing oil authority lookup for date={day.date().isoformat()}")

            row: dict[str, object] = {column: pd.NA for column in working.columns}
            row[date_col] = day
            row[entity_col] = entity
            row["store_nbr"] = store_nbr
            row["item_nbr"] = item_nbr
            if "item_id" in row:
                row["item_id"] = _single_key(entity_rows, "item_id", entity)
            row["sales"] = 0
            if "onpromotion" in row:
                row["onpromotion"] = 0
            for column in ITEM_STATIC_COLUMNS:
                if column in row:
                    row[column] = item_authority[column]
            for column in STORE_STATIC_COLUMNS:
                if column in row:
                    row[column] = store_authority[column]
            if "transactions" in row:
                row["transactions"] = tx_authority["transactions"]
            if "oil_price" in row:
                row["oil_price"] = oil_authority["oil_price"]
            if "is_holiday" in row:
                if authorities.holidays_by_store_date is None:
                    raise ValueError("missing holidays authority for is_holiday reconstruction")
                holiday = authorities.holidays_by_store_date
                row["is_holiday"] = int(
                    (
                        holiday["store_nbr"].eq(store_nbr)
                        & holiday["date"].eq(day)
                    ).any()
                )
            iso = day.isocalendar()
            generated = {
                "year": day.year,
                "month": day.month,
                "week": int(iso.week),
                "day": day.day,
            }
            for column, value in generated.items():
                if column in row:
                    row[column] = value
            synthetic_rows.append(row)
            synthetic_keys.append((str(entity), day.date().isoformat()))

    if synthetic_rows:
        combined = pd.concat([working, pd.DataFrame(synthetic_rows)], ignore_index=True)
    else:
        combined = working.copy()
    entity_rank = {value: index for index, value in enumerate(entity_order)}
    combined["__entity_rank"] = combined[entity_col].map(entity_rank)
    combined = combined.sort_values(["__entity_rank", date_col], kind="stable").drop(
        columns="__entity_rank"
    )
    combined = combined.loc[:, working.columns].reset_index(drop=True)
    combined = _cast_like_original(combined, original.dtypes)

    original_for_compare = original.copy()
    original_for_compare[date_col] = pd.to_datetime(
        original_for_compare[date_col], errors="coerce"
    ).dt.normalize()
    original_after = combined.merge(
        original_for_compare[[entity_col, date_col]],
        on=[entity_col, date_col],
        how="inner",
    )
    original_sorted = original_for_compare.copy()
    original_sorted["__entity_rank"] = original_sorted[entity_col].map(entity_rank)
    original_sorted = original_sorted.sort_values(
        ["__entity_rank", date_col], kind="stable"
    ).drop(columns="__entity_rank").reset_index(drop=True)
    original_unchanged = original_after.reset_index(drop=True).equals(original_sorted)
    if not original_unchanged:
        raise AssertionError("D5 reconstruction modified original observed rows")

    count = len(synthetic_rows)
    stats = (
        D5FieldReconstructionStats("sales", "natural_day_zero_demand", (), 0, count, 0),
        D5FieldReconstructionStats("onpromotion", "missing_promotion_contract", (), 0, count, 0),
        D5FieldReconstructionStats("oil_price", "oil_by_date", ("date",), count, 0, 0),
        D5FieldReconstructionStats(
            "transactions", "transactions_by_store_date", ("store_nbr", "date"), count, 0, 0
        ),
        D5FieldReconstructionStats("item_static", "items_by_item", ("item_nbr",), count, 0, 0),
        D5FieldReconstructionStats("store_static", "stores_by_store", ("store_nbr",), count, 0, 0),
        D5FieldReconstructionStats("date_fields", "expected_dates", ("date",), count, 0, 0),
    )
    report = D5ReconstructionReport(
        synthetic_entity_date_keys=tuple(synthetic_keys),
        synthetic_row_count=count,
        field_stats=stats,
        original_row_count=len(original),
        original_rows_unchanged=True,
        missing_lookups=(),
        authority_files=authorities.files,
    )
    combined.attrs = original_attrs
    combined.attrs["d5_calendar_reconstruction"] = {
        "synthetic_row_count": count,
        "original_row_count": len(original),
        "original_rows_unchanged": True,
    }
    return combined, report
