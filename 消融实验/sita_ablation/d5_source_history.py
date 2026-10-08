"""D5 source-history reconstruction for the ablation.

The behavior follows the frozen source-history calendarizer. This module does
not import ``src`` and does not read or write sealed parquet files. Authority
CSVs are read only to fill synthetic covariate cells and to check the manifest
SHA256. They are not a catalog of candidate identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from sita_ablation.data import ContractError

D5_HISTORY_START = "2016-08-20"
D5_HISTORY_END = "2017-02-15"
D5_HISTORY_DAYS = 180
ITEM_STATIC_COLUMNS = ("family", "class", "perishable")
STORE_STATIC_COLUMNS = ("city", "state", "type", "cluster")
DATE_COLUMNS = ("year", "month", "week", "day")
_HISTORICAL_DERIVED = re.compile(r"(?:^|_)(?:lag|lags|rolling|roll)(?:_|$)", re.I)
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
_MANIFEST_PATH = (
    _REPO_ROOT
    / "数据集"
    / "固化数据"
    / "d1_d6_sealed_v1"
    / "dataset5"
    / "prepared_180day_source_manifest.json"
)
_PREPARED_PARQUET_NAME = "prepared_180day_source.parquet"


@dataclass(frozen=True)
class ReconstructionSummary:
    applied: bool
    source_history_start: str
    source_history_end: str
    expected_day_count: int
    input_entity_count: int
    synthetic_row_count: int
    original_row_count: int
    original_rows_unchanged: bool
    rows_outside_window: int
    identities_only_after_cutoff: tuple[tuple[str, ...], ...]
    authority_sha256: tuple[tuple[str, str], ...]


def expected_source_history_dates(origin: object, *, days: int = D5_HISTORY_DAYS) -> pd.DatetimeIndex:
    """Inclusive Gregorian grid ending on ``origin``, matching the formal date builder."""
    count = int(days)
    if isinstance(days, bool) or count <= 0:
        raise ContractError("source history length must be a positive integer")
    end = pd.Timestamp(origin)
    if pd.isna(end):
        raise ContractError(f"invalid source-history origin: {origin!r}")
    dates = pd.DatetimeIndex(pd.date_range(end=end.normalize(), periods=count, freq="D")).normalize()
    if len(dates) != count or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ContractError("source-history date grid is not a unique ascending calendar")
    return dates


def inactive_reconstruction_summary(start: str, end: str) -> ReconstructionSummary:
    return ReconstructionSummary(
        applied=False,
        source_history_start=start,
        source_history_end=end,
        expected_day_count=D5_HISTORY_DAYS,
        input_entity_count=0,
        synthetic_row_count=0,
        original_row_count=0,
        original_rows_unchanged=True,
        rows_outside_window=0,
        identities_only_after_cutoff=(),
        authority_sha256=(),
    )


def reconstruct_d5_source_history(
    observed: pd.DataFrame,
    *,
    source_history_start: str = D5_HISTORY_START,
    source_history_end: str = D5_HISTORY_END,
) -> tuple[pd.DataFrame, ReconstructionSummary]:
    """Clip to the frozen cutoff, then fill missing natural days for identities inside it.

    Rows after ``source_history_end`` and identities that appear only after that
    cutoff are removed before any synthetic row is built. A missing authority
    value is not replaced with a guessed default: transactions and ``is_holiday``
    use the frozen zero fills, and a missing oil price fails.
    """
    expected = expected_source_history_dates(source_history_end)
    if str(expected[0].date()) != source_history_start or str(expected[-1].date()) != source_history_end:
        raise ContractError(
            f"D5 source-history grid {expected[0].date()}..{expected[-1].date()} "
            f"does not match {source_history_start}..{source_history_end}"
        )
    if source_history_end != D5_HISTORY_END or source_history_start != D5_HISTORY_START:
        raise ContractError("D5 source_observation_cutoff is fixed at 2017-02-15")
    prepared = _require_reconstruction_frame(observed)
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce").dt.normalize()
    if prepared["date"].isna().any():
        raise ContractError("D5 source history contains an invalid date")
    start = expected[0]
    end = expected[-1]
    inside = prepared["date"].between(start, end, inclusive="both")
    outside_count = int((~inside).sum())
    only_after = _identities_only_after(prepared, start, end)
    clipped = prepared.loc[inside].copy()
    if clipped.empty:
        return clipped.reset_index(drop=True), ReconstructionSummary(
            applied=True,
            source_history_start=source_history_start,
            source_history_end=source_history_end,
            expected_day_count=len(expected),
            input_entity_count=0,
            synthetic_row_count=0,
            original_row_count=0,
            original_rows_unchanged=True,
            rows_outside_window=outside_count,
            identities_only_after_cutoff=only_after,
            authority_sha256=(),
        )
    if clipped.duplicated(["entity_id", "date"]).any():
        raise ContractError("D5 source history contains duplicate entity-date keys")
    _reject_inconsistent_entity_fields(clipped)
    reconstructed, synthetic_count, original_count, hashes = _fill_missing_days(clipped, expected)
    _assert_original_rows_unchanged(clipped, reconstructed)
    if (reconstructed["date"] > end).any() or (reconstructed["date"] < start).any():
        raise ContractError("D5 reconstruction emitted a date outside the frozen source history")
    return reconstructed, ReconstructionSummary(
        applied=True,
        source_history_start=source_history_start,
        source_history_end=source_history_end,
        expected_day_count=len(expected),
        input_entity_count=int(clipped["entity_id"].nunique(dropna=False)),
        synthetic_row_count=synthetic_count,
        original_row_count=original_count,
        original_rows_unchanged=True,
        rows_outside_window=outside_count,
        identities_only_after_cutoff=only_after,
        authority_sha256=hashes,
    )


def authoritative_oil_by_date(oil: pd.DataFrame, *, expected_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Align ``dcoilwtico`` to natural days, forward-fill, then lag one day."""
    dates = _validate_dates(expected_dates)
    _require_columns(oil, ("date", "dcoilwtico"), "oil")
    normalized = oil.loc[:, ["date", "dcoilwtico"]].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    if normalized["date"].isna().any():
        raise ContractError("oil authority has an invalid date")
    _reject_duplicate_keys(normalized, ["date"], "oil")
    normalized["dcoilwtico"] = pd.to_numeric(normalized["dcoilwtico"], errors="coerce")
    aligned = normalized.set_index("date").reindex(dates)
    aligned["dcoilwtico"] = aligned["dcoilwtico"].ffill()
    aligned["oil_price"] = aligned["dcoilwtico"].shift(1)
    return aligned[["oil_price"]].rename_axis("date").reset_index()


def _fill_missing_days(
    observed: pd.DataFrame,
    expected: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, int, int, tuple[tuple[str, str], ...]]:
    entity_col = "entity_id"
    date_col = "date"
    historical = [column for column in observed.columns if _HISTORICAL_DERIVED.search(str(column))]
    if historical:
        raise ContractError(f"D5 historical derived fields cannot be synthesized: {sorted(historical)}")
    known = {
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
    unsupported = set(observed.columns).difference(known)
    if unsupported:
        raise ContractError(f"D5 source has fields without reconstruction authority: {sorted(unsupported)}")

    entity_fields = ["store_nbr", "item_nbr"]
    if "item_id" in observed.columns:
        entity_fields.append("item_id")
    grouped = observed.groupby(entity_col, sort=False, dropna=False)
    entity_meta = grouped[entity_fields].first().reset_index()
    entity_meta["store_nbr"] = _normalize_integer_key(entity_meta["store_nbr"], "D5 source", "store_nbr")
    entity_meta["item_nbr"] = _normalize_integer_key(entity_meta["item_nbr"], "D5 source", "item_nbr")
    full_index = pd.MultiIndex.from_product(
        [entity_meta[entity_col].tolist(), expected],
        names=[entity_col, date_col],
    )
    observed_index = pd.MultiIndex.from_frame(observed[[entity_col, date_col]])
    missing_index = full_index.difference(observed_index, sort=False)
    missing_frame = missing_index.to_frame(index=False)
    if missing_frame.empty:
        combined = observed.sort_values([entity_col, date_col], kind="mergesort").reset_index(drop=True)
        return combined, 0, len(observed), ()

    authorities, hashes = _load_verified_authorities(use_holidays="is_holiday" in observed.columns)
    rows = missing_frame.merge(entity_meta, on=entity_col, how="left", validate="many_to_one")
    rows = _merge_item_static(rows, authorities["items"])
    rows = _merge_store_static(rows, authorities["stores"])
    rows, _zero_fills = _merge_transactions(rows, authorities["transactions"])
    rows = _merge_oil(rows, authorities["oil"])
    if "is_holiday" in observed.columns:
        rows = _merge_holidays(rows, authorities["holidays"])
    synthetic = rows.reindex(columns=list(observed.columns))
    synthetic["sales"] = 0
    if "onpromotion" in synthetic.columns:
        synthetic["onpromotion"] = 0
    synthetic_dates = pd.to_datetime(synthetic[date_col], errors="raise")
    iso = synthetic_dates.dt.isocalendar()
    generated = {
        "year": synthetic_dates.dt.year,
        "month": synthetic_dates.dt.month,
        "week": iso["week"].astype("int64"),
        "day": synthetic_dates.dt.day,
    }
    for column, values in generated.items():
        if column in synthetic.columns:
            synthetic[column] = values
    combined = pd.concat([observed, synthetic], ignore_index=True)
    combined = combined.sort_values([entity_col, date_col], kind="mergesort").reset_index(drop=True)
    combined = combined.loc[:, list(observed.columns)]
    combined = _cast_like_original(combined, observed.dtypes)
    return combined, len(synthetic), len(observed), hashes


def _load_verified_authorities(*, use_holidays: bool) -> tuple[dict[str, pd.DataFrame], tuple[tuple[str, str], ...]]:
    if not _MANIFEST_PATH.is_file():
        raise ContractError(f"D5 authority manifest is missing: {_MANIFEST_PATH}")
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    auxiliary = manifest.get("auxiliary_csvs")
    if not isinstance(auxiliary, dict):
        raise ContractError("D5 authority manifest has no auxiliary_csvs")
    required = ("oil", "transactions", "items", "stores", "holidays")
    missing_names = [name for name in required if name not in auxiliary]
    if missing_names:
        raise ContractError(f"D5 authority manifest is missing {missing_names}")
    verified: list[tuple[str, str]] = []
    frames: dict[str, pd.DataFrame] = {}
    for name in required:
        entry = auxiliary[name]
        relative = entry.get("path") if isinstance(entry, dict) else None
        expected_hash = entry.get("sha256") if isinstance(entry, dict) else None
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise ContractError(f"D5 authority manifest entry {name!r} has no path and sha256")
        if Path(relative).name == _PREPARED_PARQUET_NAME or relative.endswith(_PREPARED_PARQUET_NAME):
            raise ContractError("prepared_180day_source.parquet cannot be a reconstruction input")
        path = (_REPO_ROOT / relative).resolve()
        if not path.is_file():
            raise ContractError(f"D5 authority file is missing: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ContractError(
                f"D5 authority SHA256 mismatch for {name}: expected {expected_hash}, got {actual_hash}"
            )
        verified.append((name, actual_hash))
        if name == "holidays" and not use_holidays:
            continue
        frames[name] = pd.read_csv(path)
    oil_dates = pd.to_datetime(frames["oil"]["date"], errors="coerce").dropna().dt.normalize()
    if oil_dates.empty:
        raise ContractError("oil authority has no valid dates")
    frames["oil"] = authoritative_oil_by_date(
        frames["oil"],
        expected_dates=pd.date_range(oil_dates.min(), oil_dates.max(), freq="D"),
    )
    frames["transactions"] = _normalize_transactions(frames["transactions"])
    frames["items"] = _normalize_items(frames["items"])
    frames["stores"] = _normalize_stores(frames["stores"])
    if use_holidays:
        frames["holidays"] = _normalize_holidays(frames["holidays"], frames["stores"])
    return frames, tuple(verified)


def _merge_item_static(rows: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    merged = rows.merge(
        items.loc[:, ["item_nbr", *ITEM_STATIC_COLUMNS]],
        on="item_nbr",
        how="left",
        validate="many_to_one",
        indicator="_item_authority",
    )
    missing = merged.loc[merged["_item_authority"].ne("both"), "item_nbr"]
    if not missing.empty:
        raise ContractError(f"missing items authority lookup for item_nbr={int(missing.iloc[0])}")
    return merged.drop(columns="_item_authority")


def _merge_store_static(rows: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    merged = rows.merge(
        stores.loc[:, ["store_nbr", *STORE_STATIC_COLUMNS]],
        on="store_nbr",
        how="left",
        validate="many_to_one",
        indicator="_store_authority",
    )
    missing = merged.loc[merged["_store_authority"].ne("both"), "store_nbr"]
    if not missing.empty:
        raise ContractError(f"missing stores authority lookup for store_nbr={int(missing.iloc[0])}")
    return merged.drop(columns="_store_authority")


def _merge_transactions(rows: pd.DataFrame, transactions: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    merged = rows.merge(
        transactions.loc[:, ["store_nbr", "date", "transactions"]],
        on=["store_nbr", "date"],
        how="left",
        validate="many_to_one",
        indicator="_transactions_authority",
    )
    missing = merged["_transactions_authority"].ne("both")
    zero_fills = int(missing.sum())
    if missing.any():
        merged.loc[missing, "transactions"] = 0
    return merged.drop(columns="_transactions_authority"), zero_fills


def _merge_oil(rows: pd.DataFrame, oil: pd.DataFrame) -> pd.DataFrame:
    merged = rows.merge(
        oil.loc[:, ["date", "oil_price"]],
        on="date",
        how="left",
        validate="many_to_one",
        indicator="_oil_authority",
    )
    missing = merged["_oil_authority"].ne("both") | merged["oil_price"].isna()
    if missing.any():
        identity = pd.Timestamp(merged.loc[missing, "date"].iloc[0]).date().isoformat()
        raise ContractError(f"missing oil authority lookup for date={identity}")
    return merged.drop(columns="_oil_authority")


def _merge_holidays(rows: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    merged = rows.merge(
        holidays.loc[:, ["store_nbr", "date", "is_holiday"]],
        on=["store_nbr", "date"],
        how="left",
        validate="many_to_one",
    )
    merged["is_holiday"] = pd.to_numeric(merged["is_holiday"], errors="coerce").fillna(0).astype("int64")
    return merged


def _normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("date", "store_nbr", "transactions"), "transactions")
    out = frame.loc[:, ["date", "store_nbr", "transactions"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if out["date"].isna().any():
        raise ContractError("transactions authority has an invalid date")
    out["store_nbr"] = _normalize_integer_key(out["store_nbr"], "transactions", "store_nbr")
    out["transactions"] = pd.to_numeric(out["transactions"], errors="coerce")
    if out["transactions"].isna().any():
        raise ContractError("transactions authority has an invalid value")
    _reject_duplicate_keys(out, ["store_nbr", "date"], "transactions")
    return out.sort_values(["store_nbr", "date"], kind="mergesort").reset_index(drop=True)


def _normalize_items(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("item_nbr",) + ITEM_STATIC_COLUMNS, "items")
    out = frame.loc[:, ["item_nbr", *ITEM_STATIC_COLUMNS]].copy()
    out["item_nbr"] = _normalize_integer_key(out["item_nbr"], "items", "item_nbr")
    _reject_duplicate_keys(out, ["item_nbr"], "items")
    return out.sort_values("item_nbr", kind="mergesort").reset_index(drop=True)


def _normalize_stores(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("store_nbr",) + STORE_STATIC_COLUMNS, "stores")
    out = frame.loc[:, ["store_nbr", *STORE_STATIC_COLUMNS]].copy()
    out["store_nbr"] = _normalize_integer_key(out["store_nbr"], "stores", "store_nbr")
    _reject_duplicate_keys(out, ["store_nbr"], "stores")
    return out.sort_values("store_nbr", kind="mergesort").reset_index(drop=True)


def _normalize_holidays(frame: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, ("date", "type", "locale", "locale_name", "transferred"), "holidays")
    holidays = frame.copy()
    holidays["date"] = pd.to_datetime(holidays["date"], errors="coerce").dt.normalize()
    if holidays["date"].isna().any():
        raise ContractError("holidays authority has an invalid date")
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
            .merge(regional[["date", "locale_name"]], left_on="state", right_on="locale_name", how="inner")
            [["store_nbr", "date"]]
        )
    local = holidays[holidays["locale"].eq("Local")]
    if not local.empty:
        pieces.append(
            stores[["store_nbr", "city"]]
            .merge(local[["date", "locale_name"]], left_on="city", right_on="locale_name", how="inner")
            [["store_nbr", "date"]]
        )
    if not pieces:
        return pd.DataFrame(columns=["store_nbr", "date", "is_holiday"])
    out = pd.concat(pieces, ignore_index=True).drop_duplicates(["store_nbr", "date"])
    out["is_holiday"] = 1
    return out.sort_values(["store_nbr", "date"], kind="mergesort").reset_index(drop=True)


def _require_reconstruction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "entity_id", "store_nbr", "item_nbr", "sales"}
    missing = required.difference(frame.columns)
    if missing:
        raise ContractError(
            "D5 source reconstruction requires "
            + ", ".join(sorted(missing))
            + "; entity_id is the formal aggregation column and is not synthesized"
        )
    return frame.copy()


def _reject_inconsistent_entity_fields(frame: pd.DataFrame) -> None:
    fields = ["store_nbr", "item_nbr"]
    if "item_id" in frame.columns:
        fields.append("item_id")
    grouped = frame.groupby("entity_id", sort=False, dropna=False)
    for column in fields:
        counts = grouped[column].nunique(dropna=True)
        invalid = counts.loc[counts.ne(1)]
        if not invalid.empty:
            entity = invalid.index[0]
            values = frame.loc[frame["entity_id"].eq(entity), column].drop_duplicates().tolist()
            raise ContractError(f"D5 entity {entity!r} must have exactly one {column}: {values}")


def _identities_only_after(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> tuple[tuple[str, ...], ...]:
    after = frame.loc[frame["date"] > cutoff]
    inside = frame.loc[frame["date"].between(start, cutoff, inclusive="both")]
    if after.empty:
        return ()
    after_keys = {_pair_key(row) for row in after[["store_nbr", "item_nbr"]].drop_duplicates().itertuples(index=False)}
    inside_keys = {_pair_key(row) for row in inside[["store_nbr", "item_nbr"]].drop_duplicates().itertuples(index=False)}
    return tuple(sorted(after_keys - inside_keys))


def _pair_key(row: object) -> tuple[str, ...]:
    store, item = row[0], row[1]
    return (_text(store), _text(item))


def _assert_original_rows_unchanged(before: pd.DataFrame, after: pd.DataFrame) -> None:
    keys = ["entity_id", "date"]
    left = before.sort_values(keys, kind="mergesort").reset_index(drop=True).loc[:, list(before.columns)]
    right = (
        after.merge(left[keys], on=keys, how="inner")
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
        .loc[:, list(before.columns)]
    )
    if len(right) != len(left) or not right.equals(left):
        raise ContractError("D5 reconstruction modified an original observed row")


def _cast_like_original(frame: pd.DataFrame, dtypes: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    for column, dtype in dtypes.items():
        try:
            out[column] = out[column].astype(dtype)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"D5 reconstruction changed the dtype contract for {column}: {dtype}") from exc
    return out


def _validate_dates(expected_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(expected_dates, errors="coerce"))
    if dates.empty or dates.isna().any() or dates.tz is not None:
        raise ContractError("expected_dates must be a nonempty timezone-naive calendar")
    if not dates.equals(dates.normalize()) or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ContractError("expected_dates must be unique normalized ascending days")
    return dates


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], authority: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ContractError(f"{authority} authority is missing columns {missing}")


def _reject_duplicate_keys(frame: pd.DataFrame, keys: list[str], authority: str) -> None:
    if frame.duplicated(keys).any():
        raise ContractError(f"duplicate authority key in {authority}")


def _normalize_integer_key(series: pd.Series, authority: str, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any() or not values.map(lambda value: float(value).is_integer()).all():
        raise ContractError(f"{authority} authority has an invalid {column}")
    return values.astype("int64")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    mapped = series.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ContractError("holidays authority has an invalid transferred value")
    return mapped.astype(bool)


def _text(value: object) -> str:
    if value is None or value is pd.NA:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
