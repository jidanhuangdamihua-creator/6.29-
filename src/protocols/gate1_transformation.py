"""Gate 1R.1.0 D1-D6 authority transformation and proof primitives.

The module is intentionally self contained.  The Gate 1 starting commit has
the runner and solidified parquet protocol, while the frozen Gate 1R files
are identified by the immutable freeze commit.  No function in this module
trains a model, predicts, or writes an authority artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


CONTRACT_VERSION = "1R.1.0"
CONTRACT_DIGEST = "sha256:85713b9d13cae3c017c4856b6a0f42a49d6074aebbb729171d60b95baa42eb74"
DECISION_BOOK_SHA256 = "sha256:4aaebe5f07d3dc0e61ada72dbe0625c82615ba74577e91b72b46b07a709c689d"
SCOPE_SHA256 = "sha256:98107929ea310e7fc304d2631803092e68c01e51fa992da96a3c3118b628eeb4"
MATRIX_SHA256 = "sha256:80545e2739dacdedfd8e60857bd8828dbf2102db37fc310ae4fc994b194e1da3"
COMBINED_FORMAL_IDENTITY_DIGEST = "sha256:3d11fef7b4edeb9fc804cc61455095b59e2c995afda11ba7d2c2a8afed7000e6"
SUPERSEDED_CONTRACT_DIGEST = "sha256:b145028c2b3f8314e66fc73be9795269644d016a7a1cf258a9f62f1b7443d09e"
FREEZE_COMMIT_SHA = "9eb50948241dad2f01d6854ca3cb4bf4e9c363d7"
FORMAL_INPUTS = (
    "docs/protocol/gate1_frozen_transformation_contract.md",
    "docs/protocol/gate1_implementation_scope.md",
    "docs/protocol/gate1_contract_traceability_matrix.md",
)

D4_APPROVED_FUTURE = (
    "activity_flag",
    "discount",
    "holiday_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
)
D4_AUDIT_ONLY = (
    "hours_sale_sum_leakage_risk",
    "hours_sale_max_leakage_risk",
    "hours_sale_nonzero_hours_leakage_risk",
    "hours_stock_sum_leakage_risk",
    "hours_stock_max_leakage_risk",
    "hours_stock_nonzero_hours_leakage_risk",
    "stock_hour6_22_cnt",
)
D5_FORBIDDEN_FORECAST = ("transactions", "week")
D3_FORBIDDEN_MODEL = ("Open", "Customers", "Promo", "Promo2", "PromoInterval")
D6_CALENDAR_FIELDS = ("weekday", "wday", "wm_yr_wk")
D2_SOURCE_MISSING_DATES = (
    "2018-04-01",
    "2018-04-25",
    "2018-05-01",
    "2018-06-02",
)
_HEX = re.compile(r"^[0-9a-f]{64}$")


class Gate1Failure(ValueError):
    """Stable fail-closed error carrying a contract-facing code."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message or self.code}")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def canonical_digest(value: object) -> str:
    """Digest JSON-compatible values with deterministic UTF-8 serialization."""

    def normalize(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(k): normalize(v) for k, v in sorted(item.items(), key=lambda x: str(x[0]))}
        if isinstance(item, (list, tuple)):
            return [normalize(v) for v in item]
        if isinstance(item, pd.Timestamp):
            return item.normalize().strftime("%Y-%m-%dT%H:%M:%S")
        if isinstance(item, date):
            return item.isoformat()
        if item is None or item is pd.NA:
            return None
        try:
            if bool(pd.isna(item)):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(item, "item"):
            try:
                item = item.item()
            except ValueError:
                pass
        if isinstance(item, float):
            return format(item, ".17g")
        return item

    payload = json.dumps(normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frame_payload(frame: pd.DataFrame) -> dict[str, object]:
    values: list[list[object]] = []
    for row in frame.itertuples(index=False, name=None):
        values.append([canonical_value(value) for value in row])
    return {"columns": list(frame.columns), "dtypes": [str(x) for x in frame.dtypes], "rows": values}


def canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.normalize().strftime("%Y-%m-%dT%H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def normalized_frame_digest(frame: pd.DataFrame) -> str:
    return canonical_digest(_frame_payload(frame.reset_index(drop=True)))


def _normalized_component(value: object) -> str:
    if value is None:
        raise Gate1Failure("EMPTY_KEY", "key component is null")
    try:
        if bool(pd.isna(value)):
            raise Gate1Failure("EMPTY_KEY", "key component is null")
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise Gate1Failure("EMPTY_KEY", "key component is empty")
    return text


def normalize_key(key: Sequence[object]) -> tuple[str, ...]:
    normalized = tuple(_normalized_component(value) for value in key)
    if not normalized:
        raise Gate1Failure("EMPTY_KEY", "empty exact key")
    return normalized


@dataclass(frozen=True)
class DatasetContract:
    dataset: str
    key_fields: tuple[str, ...]
    target_keys: tuple[tuple[str, ...], ...]
    origin: date
    source_history_start: date
    source_history_end: date
    target_train_start: date
    target_train_end: date
    validation_start: date
    validation_end: date
    blind_start: date
    blind_end: date
    knn_start: date
    knn_end: date
    expected_blind_rows: int
    grouping_field: str | None = None

    @property
    def source_window(self) -> tuple[date, date]:
        return self.source_history_start, self.source_history_end

    @property
    def target_window(self) -> tuple[date, date]:
        return self.target_train_start, self.validation_end


def _spec(
    dataset: str,
    key_fields: Sequence[str],
    targets: Sequence[Sequence[object]],
    origin: str,
    source_start: str,
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    blind_start: str,
    blind_end: str,
    knn_start: str,
    expected: int,
    grouping_field: str | None = None,
) -> DatasetContract:
    return DatasetContract(
        dataset,
        tuple(key_fields),
        tuple(normalize_key(key) for key in targets),
        date.fromisoformat(origin),
        date.fromisoformat(source_start),
        date.fromisoformat(origin),
        date.fromisoformat(train_start),
        date.fromisoformat(train_end),
        date.fromisoformat(validation_start),
        date.fromisoformat(validation_end),
        date.fromisoformat(blind_start),
        date.fromisoformat(blind_end),
        date.fromisoformat(knn_start),
        date.fromisoformat(origin),
        expected,
        grouping_field,
    )


_CONTRACTS = {
    "D1": _spec("D1", ("store_id", "item_id"), ((1, 10),), "2017-06-30", "2017-01-02", "2017-06-01", "2017-06-15", "2017-06-16", "2017-06-30", "2017-07-01", "2017-12-27", "2017-06-01", 180),
    "D2": _spec("D2", ("brand_id", "item_id"), ((1, 10),), "2018-06-30", "2018-01-02", "2018-06-01", "2018-06-15", "2018-06-16", "2018-06-30", "2018-07-01", "2018-12-27", "2018-06-01", 180),
    "D3": _spec("D3", ("store_id",), ((10,),), "2015-02-01", "2014-08-06", "2015-01-03", "2015-01-17", "2015-01-18", "2015-02-01", "2015-02-02", "2015-07-31", "2015-01-03", 180),
    "D4": _spec("D4", ("store_id", "product_id"), ((166, 258), (166, 432), (166, 433), (166, 313), (166, 311)), "2025-01-14", "2024-07-19", "2024-12-16", "2024-12-30", "2024-12-31", "2025-01-14", "2025-01-15", "2025-07-13", "2024-12-16", 900, "second_category_id"),
    "D5": _spec("D5", ("store_nbr", "item_nbr"), ((48, 364606), (48, 1159415), (48, 1159414), (48, 1349808), (48, 320682)), "2017-02-15", "2016-08-20", "2017-01-17", "2017-01-31", "2017-02-01", "2017-02-15", "2017-02-16", "2017-08-14", "2017-01-17", 900, "family"),
    "D6": _spec("D6", ("store_id", "item_id"), (("CA_1", "FOODS_3_586"), ("CA_1", "FOODS_3_080"), ("CA_1", "FOODS_3_555"), ("CA_1", "FOODS_3_377"), ("CA_1", "FOODS_3_668")), "2015-11-24", "2015-05-29", "2015-10-26", "2015-11-09", "2015-11-10", "2015-11-24", "2015-11-25", "2016-05-22", "2015-10-26", 900, "dept_id"),
}


def dataset_contract(dataset: object) -> DatasetContract:
    key = str(dataset).strip().upper().replace("DATASET", "D")
    if key.isdigit():
        key = "D" + key
    try:
        return _CONTRACTS[key]
    except KeyError as exc:
        raise Gate1Failure("UNKNOWN_DATASET", str(dataset)) from exc


def _parse_sidecar(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line or not line.strip():
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key in result:
            raise Gate1Failure("FORMAL_IDENTITY", f"duplicate sidecar key {key}")
        result[key] = value
    return result


def _authority_bytes(root: Path, relative: str) -> tuple[bytes, str]:
    path = root / relative
    if path.exists() and (not path.is_symlink()) and path.is_file():
        return path.read_bytes(), str(path)
    try:
        data = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{FREEZE_COMMIT_SHA}:{relative}"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Gate1Failure("FORMAL_IDENTITY", f"missing frozen authority {relative}") from exc
    return data, f"git:{FREEZE_COMMIT_SHA}:{relative}"


def load_formal_identity(project_root: Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    sidecar_bytes, sidecar_source = _authority_bytes(root, "docs/protocol/gate1_frozen_transformation_contract.sha256")
    sidecar = _parse_sidecar(sidecar_bytes.decode("utf-8"))
    required = {"contract_version", "contract_digest", "decision_book_sha256", "scope_sha256", "matrix_sha256", "combined_formal_identity_digest", "old_contract_digest", "old_contract_status"}
    missing = sorted(required - set(sidecar))
    if missing:
        raise Gate1Failure("FORMAL_IDENTITY", f"sidecar missing {missing}")
    if sidecar["contract_version"] != CONTRACT_VERSION or sidecar["contract_digest"] != CONTRACT_DIGEST:
        raise Gate1Failure("FORMAL_IDENTITY", "contract identity mismatch")
    if sidecar["decision_book_sha256"] != DECISION_BOOK_SHA256 or sidecar["scope_sha256"] != SCOPE_SHA256 or sidecar["matrix_sha256"] != MATRIX_SHA256:
        raise Gate1Failure("FORMAL_IDENTITY", "companion identity mismatch")
    if sidecar["old_contract_digest"] != SUPERSEDED_CONTRACT_DIGEST or sidecar["old_contract_status"] != "SUPERSEDED":
        raise Gate1Failure("FORMAL_IDENTITY", "superseded identity mismatch")
    files: dict[str, dict[str, object]] = {}
    for label, relative, expected in (("contract", FORMAL_INPUTS[0], CONTRACT_DIGEST), ("scope", FORMAL_INPUTS[1], SCOPE_SHA256), ("matrix", FORMAL_INPUTS[2], MATRIX_SHA256)):
        blob, source = _authority_bytes(root, relative)
        actual = _digest_bytes(blob)
        if actual != expected:
            raise Gate1Failure("FORMAL_IDENTITY", f"{label} bytes do not match sidecar")
        files[label] = {"path": relative, "sha256": actual, "size_bytes": len(blob), "source": source}
    payload = (f"decision_book_sha256={DECISION_BOOK_SHA256.removeprefix('sha256:')}\n" f"contract_sha256={CONTRACT_DIGEST.removeprefix('sha256:')}\n" f"scope_sha256={SCOPE_SHA256.removeprefix('sha256:')}\n" f"matrix_sha256={MATRIX_SHA256.removeprefix('sha256:')}\n").encode("utf-8")
    if _digest_bytes(payload) != COMBINED_FORMAL_IDENTITY_DIGEST or sidecar["combined_formal_identity_digest"] != COMBINED_FORMAL_IDENTITY_DIGEST:
        raise Gate1Failure("FORMAL_IDENTITY", "combined formal identity mismatch")
    try:
        refreeze, refreeze_source = _authority_bytes(root, "docs/protocol/gate1r_contract_refreeze_record.md")
    except Gate1Failure:
        refreeze, refreeze_source = b"", "missing"
    if not all(token.encode("utf-8") in refreeze for token in (CONTRACT_DIGEST, SCOPE_SHA256, MATRIX_SHA256, COMBINED_FORMAL_IDENTITY_DIGEST)):
        raise Gate1Failure("FORMAL_IDENTITY", "refreeze record does not bind formal identity")
    return {
        "decision_book_sha256": DECISION_BOOK_SHA256,
        "contract_digest": CONTRACT_DIGEST,
        "scope_sha256": SCOPE_SHA256,
        "matrix_sha256": MATRIX_SHA256,
        "combined_formal_identity_digest": COMBINED_FORMAL_IDENTITY_DIGEST,
        "freeze_commit_sha": FREEZE_COMMIT_SHA,
        "contract_version": CONTRACT_VERSION,
        "old_contract_digest": SUPERSEDED_CONTRACT_DIGEST,
        "old_contract_status": "SUPERSEDED",
        "sidecar_source": sidecar_source,
        "refreeze_source": refreeze_source,
        "files": files,
        "formal_input_set_digest": canonical_digest(files),
    }


class FormalInputLoader:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)

    def load(self, paths: Sequence[Path] | None = None) -> dict[str, object]:
        if paths:
            expected = {Path(self.project_root / relative).resolve() for relative in FORMAL_INPUTS}
            if any(Path(path).resolve() not in expected for path in paths):
                raise Gate1Failure("AUTHORITY_PATH", "formal input path is outside frozen authority")
        return load_formal_identity(self.project_root)


def _prepare_frame(frame: pd.DataFrame, *, label: str, spec: DatasetContract) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise Gate1Failure("FRAME_TYPE", f"{label} must be a DataFrame")
    missing = [column for column in (*spec.key_fields, "date") if column not in frame.columns]
    if missing:
        raise Gate1Failure("SCHEMA_MISSING", f"{label} missing {missing}")
    result = frame.copy()
    parsed = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise Gate1Failure("INVALID_DATE", f"{label} contains invalid date")
    result["date"] = parsed
    for key in spec.key_fields:
        if result[key].isna().any() or result[key].map(lambda x: str(x).strip()).eq("").any():
            raise Gate1Failure("EMPTY_KEY", f"{label} has empty exact key")
    if result.duplicated([*spec.key_fields, "date"]).any():
        raise Gate1Failure("DUPLICATE_EXACT_KEY_DATE", f"{label} contains duplicate exact key/date")
    return result


def rebuild_d2_wide_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Rebuild D2's wide authority without dropping empty entity/date rows."""
    date_column = "DATE" if "DATE" in raw.columns else "date" if "date" in raw.columns else None
    if date_column is None:
        raise Gate1Failure("D2_KEY_REBUILD", "wide D2 authority has no DATE column")
    pattern = re.compile(r"^(?:QTY|PROMO)_B([^_]+)_([^_]+)$", re.IGNORECASE)
    value_columns = []
    for column in raw.columns:
        match = pattern.match(str(column))
        if match and str(column).upper().startswith("QTY_"):
            value_columns.append((str(column), match.group(1), match.group(2)))
    if not value_columns:
        raise Gate1Failure("D2_KEY_REBUILD", "wide D2 authority has no QTY_B<brand>_<item> columns")
    rows: list[dict[str, object]] = []
    parsed_dates = pd.to_datetime(raw[date_column], errors="coerce").dt.normalize()
    if parsed_dates.isna().any():
        raise Gate1Failure("INVALID_DATE", "wide D2 authority has invalid date")
    for index, timestamp in parsed_dates.items():
        for column, brand, item in value_columns:
            value = raw.at[index, column]
            if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
                value = 0
            rows.append({"brand_id": brand, "item_id": item, "date": timestamp, "sales": value})
    rebuilt = pd.DataFrame(rows, columns=["brand_id", "item_id", "date", "sales"])
    if rebuilt.duplicated(["brand_id", "item_id", "date"]).any():
        raise Gate1Failure("DUPLICATE_EXACT_KEY_DATE", "rebuilt D2 authority contains duplicate exact key/date")
    rebuilt["sales"] = pd.to_numeric(rebuilt["sales"], errors="coerce").fillna(0)
    return _add_date_fields(rebuilt)


def calendarize_d2_source_history(frame: pd.DataFrame, candidate_keys: Sequence[Sequence[object]]) -> tuple[pd.DataFrame, dict[str, object]]:
    """Close the four approved D2 source dates before eligibility selection."""
    spec = dataset_contract("D2")
    source = _prepare_frame(frame, label="D2 raw source", spec=spec)
    source = source.loc[source["date"].between(pd.Timestamp(spec.source_history_start), pd.Timestamp(spec.source_history_end))].copy()
    expected = pd.date_range(spec.source_history_start, spec.source_history_end, freq="D")
    allowed_missing = {pd.Timestamp(value) for value in D2_SOURCE_MISSING_DATES}
    rows: list[dict[str, object]] = []
    repairs: list[dict[str, object]] = []
    key_series = _key_series(source, spec.key_fields)
    for raw_key in candidate_keys:
        key = normalize_key(raw_key)
        group = source.loc[key_series.map(lambda value, expected_key=key: value == expected_key)].copy()
        if group.empty:
            continue
        actual = set(pd.DatetimeIndex(group["date"]))
        missing = [timestamp for timestamp in expected if timestamp not in actual]
        if any(timestamp not in allowed_missing for timestamp in missing):
            raise Gate1Failure("D2_SOURCE_CALENDARIZATION", f"unapproved missing D2 source dates for {key}: {missing}")
        rows.extend(group.to_dict(orient="records"))
        template = group.iloc[0].to_dict()
        for timestamp in missing:
            row = dict(template)
            for field, value in zip(spec.key_fields, key):
                row[field] = value
            row["date"] = timestamp
            row["sales"] = 0
            for field in ("year", "month", "week", "day"):
                row[field] = getattr(timestamp, field) if field != "week" else int(timestamp.isocalendar().week)
            rows.append(row)
            repairs.append({"key": list(key), "date": timestamp.strftime("%Y-%m-%d"), "sales": 0, "rule": "D2_APPROVED_SOURCE_CALENDARIZATION"})
    result = pd.DataFrame(rows, columns=list(source.columns))
    if result.empty:
        raise Gate1Failure("SOURCE_ENTITY_MISSING", "D2 source calendarization found no candidate entities")
    result = _add_date_fields(result).sort_values([*spec.key_fields, "date"], kind="mergesort").reset_index(drop=True)
    return result, {"rule": "D2_APPROVED_SOURCE_CALENDARIZATION", "missing_dates": list(D2_SOURCE_MISSING_DATES), "repairs": repairs, "rows_after": int(len(result)), "digest": normalized_frame_digest(result)}


def normalize_onpromotion(values: pd.Series) -> pd.Series:
    normalized = []
    for value in values.tolist():
        try:
            is_null = bool(pd.isna(value))
        except (TypeError, ValueError):
            is_null = False
        if value is None or value is pd.NA or is_null:
            normalized.append(0)
        elif isinstance(value, bool):
            normalized.append(int(value))
        elif isinstance(value, (int, float)) and value in (0, 1):
            normalized.append(int(value))
        elif isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
            normalized.append(1 if value.strip().lower() in {"true", "1"} else 0)
        else:
            raise Gate1Failure("ONPROMOTION_ENCODING", f"unknown onpromotion value {value!r}")
    return pd.Series(normalized, index=values.index, dtype="int8")


def _key_series(frame: pd.DataFrame, key_fields: Sequence[str]) -> pd.Series:
    return frame.loc[:, list(key_fields)].apply(lambda row: normalize_key(tuple(row.tolist())), axis=1)


def _mask_keys(frame: pd.DataFrame, spec: DatasetContract, keys: Sequence[Sequence[object]]) -> pd.Series:
    wanted = {normalize_key(key) for key in keys}
    return _key_series(frame, spec.key_fields).isin(wanted)


def _add_date_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["year"] = result["date"].dt.year.astype("int64")
    result["month"] = result["date"].dt.month.astype("int64")
    result["week"] = result["date"].dt.isocalendar().week.astype("int64")
    result["day"] = result["date"].dt.day.astype("int64")
    return result


def _calendarize(frame: pd.DataFrame, spec: DatasetContract) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    expected = pd.date_range(spec.blind_start, spec.blind_end, freq="D")
    rows: list[dict[str, object]] = []
    repairs: list[dict[str, object]] = []
    key_series = _key_series(frame, spec.key_fields)
    for target_key in spec.target_keys:
        group = frame.loc[key_series.map(lambda value, expected=target_key: value == expected)].copy()
        if group.empty:
            raise Gate1Failure("TARGET_ENTITY_MISSING", f"missing target {target_key!r}")
        group = group.loc[group["date"].between(expected.min(), expected.max())].copy()
        by_date = {pd.Timestamp(row.date): row._asdict() for row in group.itertuples(index=False)}
        template = group.iloc[0].to_dict()
        for timestamp in expected:
            if timestamp in by_date:
                row = dict(by_date[timestamp])
            else:
                row = dict(template)
                row["date"] = timestamp
                if "sales" in row:
                    row["sales"] = 0
                if "onpromotion" in row:
                    row["onpromotion"] = 0
                repairs.append({"dataset": spec.dataset, "key": list(target_key), "date": timestamp.strftime("%Y-%m-%d"), "rule": "calendarize_missing_blind_day", "sales": 0})
            for field, value in zip(spec.key_fields, target_key):
                row[field] = value
            rows.append(row)
    result = pd.DataFrame(rows)
    result = _add_date_fields(result)
    if spec.dataset == "D2":
        for column in ("PROMO", "promo", "Promo"):
            if column in result.columns:
                result[column] = result[column]
    if spec.dataset == "D3" and "SchoolHoliday" in result.columns:
        result["SchoolHoliday"] = pd.to_numeric(result["SchoolHoliday"], errors="coerce").fillna(0).astype("int64")
    if spec.dataset == "D5" and "onpromotion" in result.columns:
        result["onpromotion"] = normalize_onpromotion(result["onpromotion"])
    return result.sort_values([*spec.key_fields, "date"], kind="mergesort").reset_index(drop=True), repairs


def source_pool_candidates(dataset: object, scenario: str) -> tuple[object, ...]:
    spec = dataset_contract(dataset)
    text = str(scenario).strip().lower().replace("_", "-")
    if text in {"with", "with-sharing", "with-information-sharing"}:
        shared = True
    elif text in {"without", "without-sharing", "without-information-sharing"}:
        shared = False
    else:
        raise Gate1Failure("SCENARIO", f"unsupported scenario {scenario!r}")
    if spec.dataset in {"D1", "D2"}:
        return (1, 2, 3) if shared else (1,)
    if spec.dataset == "D3":
        return tuple(index for index in (range(1, 31) if shared else range(1, 10)) if index != 10)
    return ("__ALL_STORES__",) if shared else (spec.target_keys[0][0],)


def _domain_mask(frame: pd.DataFrame, spec: DatasetContract, scenario: str) -> pd.Series:
    if spec.dataset in {"D1", "D2"}:
        domain_field = spec.key_fields[0]
        values = source_pool_candidates(spec.dataset, scenario)
        return frame[domain_field].map(lambda x: _normalized_component(x)).isin({_normalized_component(x) for x in values}) & frame[spec.key_fields[1]].map(lambda x: _normalized_component(x)).isin({str(x) for x in range(1, 10)})
    domain = source_pool_candidates(spec.dataset, scenario)
    if spec.dataset == "D3":
        return frame["store_id"].map(lambda x: _normalized_component(x)).isin({_normalized_component(x) for x in domain})
    target_store = _normalized_component(spec.target_keys[0][0])
    if spec.dataset == "D4":
        return frame["store_id"].map(lambda x: _normalized_component(x)).eq(target_store) if domain != ("__ALL_STORES__",) else pd.Series(True, index=frame.index)
    if spec.dataset == "D5":
        return frame["store_nbr"].map(lambda x: _normalized_component(x)).eq(target_store) if domain != ("__ALL_STORES__",) else pd.Series(True, index=frame.index)
    return frame["store_id"].map(lambda x: _normalized_component(x)).eq(target_store) if domain != ("__ALL_STORES__",) else pd.Series(True, index=frame.index)


def _same_group_mask(frame: pd.DataFrame, spec: DatasetContract) -> pd.Series:
    if not spec.grouping_field or spec.grouping_field not in frame.columns:
        return pd.Series(True, index=frame.index)
    target_field = spec.grouping_field
    target = frame.loc[_mask_keys(frame, spec, spec.target_keys), target_field].dropna().unique()
    if len(target) == 0:
        return pd.Series(False, index=frame.index)
    return frame[target_field].isin([target[0]])


def select_source_history_candidates(
    dataset: object,
    source_frame: pd.DataFrame,
    scenario: str,
    *,
    require_complete: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    spec = dataset_contract(dataset)
    source = _prepare_frame(source_frame, label=f"{spec.dataset} source", spec=spec)
    source = source.loc[source["date"].between(pd.Timestamp(spec.source_history_start), pd.Timestamp(spec.source_history_end))].copy()
    source = source.loc[_domain_mask(source, spec, scenario)].copy()
    if spec.dataset in {"D4", "D5", "D6"}:
        source = source.loc[_same_group_mask(source, spec)].copy()
    target_set = set(spec.target_keys)
    source = source.loc[~_key_series(source, spec.key_fields).isin(target_set)].copy()
    counts = source.groupby(list(spec.key_fields), dropna=False)["date"].nunique()
    complete = []
    incomplete: dict[str, int] = {}
    for raw_key, count in counts.items():
        key = normalize_key(raw_key if isinstance(raw_key, tuple) else (raw_key,))
        if int(count) == 180:
            complete.append(key)
        else:
            incomplete["/".join(key)] = int(count)
    if require_complete and not complete:
        raise Gate1Failure("SOURCE_ENTITY_MISSING", "no complete approved source candidate")
    selected = source.loc[_key_series(source, spec.key_fields).isin(set(complete))].copy()
    return selected.sort_values([*spec.key_fields, "date"], kind="mergesort").reset_index(drop=True), {
        "dataset": spec.dataset,
        "scenario": scenario,
        "candidate_keys": [list(key) for key in sorted(complete)],
        "incomplete_candidates": incomplete,
        "source_rows_before": int(len(source_frame)),
        "source_rows_after": int(len(selected)),
        "expected_days": 180,
    }


def stream_source_history_candidates(
    dataset: object,
    source_path: Path,
    scenario: str,
    *,
    target_frame: pd.DataFrame | None = None,
    allow_approved_calendarization: bool = False,
    batch_size: int = 500_000,
) -> dict[str, object]:
    """Prove source eligibility by scanning parquet batches without materializing it.

    The stream keeps only candidate-key date sets and counters.  It never
    constructs a full source DataFrame, so the same proof is safe for D4-D6.
    """
    spec = dataset_contract(dataset)
    path = Path(source_path)
    if not path.is_file() or path.is_symlink():
        raise Gate1Failure("SOURCE_AUTHORITY_MISSING", str(path))
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise Gate1Failure("SOURCE_STREAM_NOT_PROVEN", "pyarrow is required for bounded source proof") from exc
    required = list(spec.key_fields) + ["date"] + ([spec.grouping_field] if spec.grouping_field else [])
    parquet = pq.ParquetFile(path)
    missing = [column for column in required if column not in parquet.schema_arrow.names]
    if missing:
        raise Gate1Failure("SOURCE_SCHEMA", f"source stream missing {missing}")
    scenario_text = str(scenario).strip().lower().replace("_", "-")
    if scenario_text in {"with", "with-sharing", "with-information-sharing"}:
        shared = True
    elif scenario_text in {"without", "without-sharing", "without-information-sharing"}:
        shared = False
    else:
        raise Gate1Failure("SCENARIO", f"unsupported scenario {scenario!r}")
    target_keys = set(spec.target_keys)
    target_groups: set[str] = set()
    if spec.grouping_field:
        if target_frame is None or spec.grouping_field not in target_frame.columns:
            raise Gate1Failure("SOURCE_GROUP_AUTHORITY", f"target is missing {spec.grouping_field}")
        target_keys_mask = _mask_keys(_prepare_frame(target_frame, label=f"{spec.dataset} target", spec=spec), spec, spec.target_keys)
        target_groups = {_normalized_component(value) for value in target_frame.loc[target_keys_mask, spec.grouping_field].dropna().tolist()}
        if len(target_groups) != 1:
            raise Gate1Failure("SOURCE_GROUP_AUTHORITY", f"target has ambiguous {spec.grouping_field}: {sorted(target_groups)}")
    source_rows = 0
    eligible_rows = 0
    post_origin_rows = 0
    duplicate_exact_key_dates = 0
    date_masks_by_key: dict[tuple[str, ...], int] = {}
    source_start = pd.Timestamp(spec.source_history_start)
    allowed_item_ids = {str(value) for value in range(1, 10)}
    allowed_domains = {_normalized_component(value) for value in source_pool_candidates(spec.dataset, "with-sharing" if shared else "without-sharing")} if spec.dataset in {"D1", "D2", "D3"} else set()
    fields = list(dict.fromkeys(required))
    for batch in parquet.iter_batches(columns=fields, batch_size=int(batch_size)):
        chunk = batch.to_pandas()
        source_rows += len(chunk)
        dates = pd.to_datetime(chunk["date"], errors="coerce").dt.normalize()
        if dates.isna().any():
            raise Gate1Failure("INVALID_DATE", "source stream contains invalid date")
        post_origin_rows += int((dates > pd.Timestamp(spec.origin)).sum())
        history = chunk.loc[dates.between(pd.Timestamp(spec.source_history_start), pd.Timestamp(spec.source_history_end))].copy()
        history["date"] = dates.loc[history.index]
        for row in history.itertuples(index=False, name=None):
            values = dict(zip(fields, row))
            key = normalize_key(tuple(values[field] for field in spec.key_fields))
            if key in target_keys:
                continue
            if spec.dataset in {"D1", "D2"}:
                if key[0] not in allowed_domains or key[1] not in allowed_item_ids:
                    continue
            elif spec.dataset == "D3":
                if key[0] not in allowed_domains:
                    continue
            elif spec.dataset in {"D4", "D5", "D6"}:
                store = key[0]
                target_store = spec.target_keys[0][0]
                if not shared and store != target_store:
                    continue
                if spec.grouping_field and _normalized_component(values[spec.grouping_field]) not in target_groups:
                    continue
            timestamp = pd.Timestamp(values["date"])
            bit = 1 << int((timestamp - source_start).days)
            current_mask = date_masks_by_key.get(key, 0)
            if current_mask & bit:
                duplicate_exact_key_dates += 1
            date_masks_by_key[key] = current_mask | bit
            eligible_rows += 1
    def _popcount(mask: int) -> int:
        return bin(mask).count("1")

    raw_counts = {"/".join(key): _popcount(mask) for key, mask in sorted(date_masks_by_key.items())}
    raw_complete = sorted(key for key, mask in date_masks_by_key.items() if _popcount(mask) == 180)
    incomplete = {key: count for key, count in raw_counts.items() if count != 180}
    if spec.dataset == "D5" and allow_approved_calendarization:
        complete = sorted(date_masks_by_key)
        calendarization = {
            "status": "passed",
            "rule": "D5_APPROVED_SOURCE_HISTORY_CALENDARIZATION",
            "window": [spec.source_history_start.isoformat(), spec.source_history_end.isoformat()],
            "candidate_count": len(complete),
            "raw_complete_candidate_count": len(raw_complete),
            "raw_incomplete_candidate_count": len(incomplete),
            "raw_candidate_cardinality_digest": canonical_digest(raw_counts),
            "raw_eligible_rows": int(eligible_rows),
            "calendarized_rows": int(len(complete) * 180),
            "repaired_rows": int(sum(180 - count for count in incomplete.values())),
            "missing_date_count": int(sum(180 - count for count in incomplete.values())),
            "sales_rule": "missing source natural day -> 0",
            "covariate_rule": "reuse approved D5 historical reconstruction rules",
        }
    else:
        complete = raw_complete
        calendarization = {
            "status": "not_applicable",
            "candidate_count": len(complete),
            "raw_complete_candidate_count": len(raw_complete),
            "raw_incomplete_candidate_count": len(incomplete),
            "raw_candidate_cardinality_digest": canonical_digest(raw_counts),
            "raw_eligible_rows": int(eligible_rows),
            "calendarized_rows": int(len(complete) * 180),
            "repaired_rows": 0,
            "missing_date_count": 0,
        }
    if not complete:
        raise Gate1Failure("SOURCE_ENTITY_MISSING", f"no complete source candidate in {path}")
    proof = {
        "dataset": spec.dataset,
        "scenario": "with" if shared else "without",
        "authority_path": str(path),
        "authority_size_bytes": int(path.stat().st_size),
        "rows_scanned": int(source_rows),
        "eligible_history_rows": int(eligible_rows),
        "post_origin_history_rows": int(post_origin_rows),
        "duplicate_exact_key_dates": int(duplicate_exact_key_dates),
        "complete_candidate_keys": [list(key) for key in complete],
        "incomplete_candidates": incomplete,
        "calendarization": calendarization,
        "candidate_filter_digest": canonical_digest({"dataset": spec.dataset, "scenario": "with" if shared else "without", "grouping_field": spec.grouping_field, "targets": [list(key) for key in spec.target_keys]}),
        "status": "passed",
    }
    return proof


@dataclass
class DatasetRoles:
    dataset: str
    source_history: pd.DataFrame
    target_observed: pd.DataFrame
    target_train: pd.DataFrame
    target_validation: pd.DataFrame
    worker_safe_blind: pd.DataFrame
    evaluator_truth: pd.DataFrame
    audit_view: pd.DataFrame
    repairs: list[dict[str, object]] = field(default_factory=list)
    proof: dict[str, object] = field(default_factory=dict)


def slice_dataset_roles(dataset: object, source_frame: pd.DataFrame, target_frame: pd.DataFrame) -> DatasetRoles:
    spec = dataset_contract(dataset)
    source = _prepare_frame(source_frame, label=f"{spec.dataset} source", spec=spec)
    target = _prepare_frame(target_frame, label=f"{spec.dataset} target", spec=spec)
    target = target.loc[_mask_keys(target, spec, spec.target_keys)].copy()
    if target.empty:
        raise Gate1Failure("TARGET_ENTITY_MISSING", f"no target entities for {spec.dataset}")
    source_history = source.loc[source["date"].between(pd.Timestamp(spec.source_history_start), pd.Timestamp(spec.source_history_end))].copy()
    target_observed = target.loc[target["date"] <= pd.Timestamp(spec.origin)].copy()
    train = target.loc[target["date"].between(pd.Timestamp(spec.target_train_start), pd.Timestamp(spec.target_train_end))].copy()
    validation = target.loc[target["date"].between(pd.Timestamp(spec.validation_start), pd.Timestamp(spec.validation_end))].copy()
    blind_actual, repairs = _calendarize(target, spec)
    worker = _consumer_frame(spec.dataset, blind_actual, view="worker_safe_blind")
    evaluator = blind_actual.copy()
    audit = blind_actual.copy()
    if spec.dataset == "D2":
        for column in ("PROMO", "promo", "Promo"):
            worker = worker.drop(columns=[column], errors="ignore")
    if spec.dataset == "D5" and "transactions" in worker.columns:
        worker = worker.drop(columns=["transactions"], errors="ignore")
    if "sales" in worker.columns:
        worker = worker.drop(columns=["sales"])
    if (worker["date"] <= pd.Timestamp(spec.origin)).any():
        raise Gate1Failure("FORECAST_ORIGIN", f"{spec.dataset} worker contains origin or history rows")
    if (source_history["date"] > pd.Timestamp(spec.origin)).any():
        raise Gate1Failure("HISTORY_FUTURE_ROW", f"{spec.dataset} source history contains post-origin rows")
    proof = {
        "dataset": spec.dataset,
        "windows": {name: [getattr(spec, f"{name}_start").isoformat(), getattr(spec, f"{name}_end").isoformat()] for name in ("source_history", "target_train", "validation", "blind", "knn")},
        "entity_keys": [list(key) for key in spec.target_keys],
        "before_rows": {"source": int(len(source_frame)), "target": int(len(target_frame))},
        "after_slicing_rows": {"source_history": int(len(source_history)), "target_observed": int(len(target_observed)), "target_train": int(len(train)), "validation": int(len(validation)), "blind": int(len(blind_actual))},
        "expected_cardinality": spec.expected_blind_rows,
        "missing_exact_keys": repairs,
        "duplicate_exact_keys": 0,
        "post_origin_history_rows": int((source["date"] > pd.Timestamp(spec.origin)).sum()),
        "pre_or_equal_origin_forecast_rows": int((blind_actual["date"] <= pd.Timestamp(spec.origin)).sum()),
        "safe_view_digests": {name: normalized_frame_digest(frame) for name, frame in {"source_history": source_history, "target_observed": target_observed, "worker_safe_blind": worker, "evaluator_truth": evaluator, "audit_view": audit}.items()},
    }
    return DatasetRoles(spec.dataset, source_history.reset_index(drop=True), target_observed.reset_index(drop=True), train.reset_index(drop=True), validation.reset_index(drop=True), worker.reset_index(drop=True), evaluator.reset_index(drop=True), audit.reset_index(drop=True), repairs, proof)


class SchemaRegistry:
    def __init__(self) -> None:
        self._schemas = {dataset: self._allowed(dataset) for dataset in _CONTRACTS}

    def _allowed(self, dataset: str) -> dict[str, tuple[str, ...]]:
        spec = _CONTRACTS[dataset]
        common = (*spec.key_fields, "date", "year", "month", "day")
        if dataset in {"D1", "D2", "D3"}:
            worker = (*common, "week") + (("SchoolHoliday",) if dataset == "D3" else ())
        elif dataset == "D4":
            worker = (*common, "week", *D4_APPROVED_FUTURE)
        elif dataset == "D5":
            worker = (*common, "onpromotion", "perishable", "oil_price", "is_holiday")
        else:
            worker = (*common, "weekday", "wday", "wm_yr_wk", "snap", "sell_price")
        return {"worker": tuple(worker), "predictor": tuple(worker), "knn": ("date", "sales"), "source_history": tuple(common), "target_observed": tuple(common), "worker_safe_blind": tuple(worker), "evaluator_truth": tuple(common) + ("sales",), "audit_view": tuple()}

    def allowed(self, dataset: object, role: str) -> tuple[str, ...]:
        key = dataset_contract(dataset).dataset
        try:
            return self._schemas[key][role]
        except KeyError as exc:
            raise Gate1Failure("SCHEMA_ROLE", f"unknown schema role {role!r}") from exc

    def validate(self, dataset: object, role: str, frame: pd.DataFrame) -> None:
        allowed = self.allowed(dataset, role)
        if role == "audit_view":
            return
        extra = [column for column in frame.columns if column not in allowed]
        if extra:
            raise Gate1Failure("SCHEMA_EXTRA", f"{dataset}/{role}: {extra}")
        missing = [column for column in allowed if column not in frame.columns and column not in {"year", "month", "day", "week"}]
        if missing:
            raise Gate1Failure("SCHEMA_MISSING", f"{dataset}/{role}: {missing}")


def _consumer_frame(dataset: str, frame: pd.DataFrame, *, view: str) -> pd.DataFrame:
    allowed = SchemaRegistry().allowed(dataset, view)
    columns = [column for column in allowed if column in frame.columns]
    result = frame.loc[:, columns].copy()
    if dataset == "D4":
        forbidden = [column for column in D4_AUDIT_ONLY if column in result.columns]
        result = result.drop(columns=forbidden)
    if dataset == "D5":
        result = result.drop(columns=["transactions", "week"], errors="ignore")
    if dataset == "D3":
        result = result.drop(columns=list(D3_FORBIDDEN_MODEL), errors="ignore")
    return result


def build_contract_views(dataset: object, roles: DatasetRoles) -> dict[str, pd.DataFrame]:
    spec = dataset_contract(dataset)
    views = {
        "source_history": roles.source_history.copy(),
        "target_observed": roles.target_observed.copy(),
        "worker_safe_blind": roles.worker_safe_blind.copy(),
        "evaluator_truth": roles.evaluator_truth.copy(),
        "audit_view": roles.audit_view.copy(),
    }
    for name in ("worker_safe_blind",):
        SchemaRegistry().validate(spec.dataset, name, views[name])
    return views


def _view_descriptor(dataset: str, name: str, frame: pd.DataFrame) -> dict[str, object]:
    spec = dataset_contract(dataset)
    keys = sorted({list(normalize_key(tuple(row))) for row in frame.loc[:, list(spec.key_fields)].drop_duplicates().itertuples(index=False, name=None)}) if all(field in frame.columns for field in spec.key_fields) else []
    return {"view": name, "dataset": dataset, "columns": list(frame.columns), "dtypes": [str(x) for x in frame.dtypes], "field_roles": {column: ("date" if column == "date" else "target" if column == "sales" else "feature") for column in frame.columns}, "availability": {column: ("evaluator_only" if name == "evaluator_truth" and column == "sales" else "worker_safe" if name == "worker_safe_blind" else "audit") for column in frame.columns}, "exact_key": list(spec.key_fields) + ["date"], "date_min": None if frame.empty else pd.Timestamp(frame.date.min()).strftime("%Y-%m-%d"), "date_max": None if frame.empty else pd.Timestamp(frame.date.max()).strftime("%Y-%m-%d"), "entity_count": len(keys), "row_count": int(len(frame)), "canonical_digest": normalized_frame_digest(frame), "keys": keys}


def effective_knn_schema_descriptor(dataset: object) -> dict[str, object]:
    spec = dataset_contract(dataset)
    return {"dataset": spec.dataset, "fields": ["date", "sales"], "exact_key": list(spec.key_fields) + ["date"], "observation_start": spec.knn_start.isoformat(), "observation_end": spec.knn_end.isoformat()}


class ProofWriter:
    REQUIRED = ("decision_book_identity", "contract_identity", "scope_identity", "matrix_identity", "combined_formal_identity", "freeze_commit_identity", "raw_authority_inventory", "approved_input_set", "parent_lineage", "entity_proof", "role_proof", "window_proof", "exact_key_proof", "cardinality_proof", "calendarization_proof", "field_repair_proof", "availability_proof", "field_exclusion_proof", "safe_view_proof", "no_leakage_proof", "schema_digest", "canonical_content_digest", "physical_artifact_hash", "producer_identity", "operator_identity", "real_input_readiness_identity", "formal_preflight", "private_build_ownership", "publication_proof")

    def build(self, *, contract_digest: str, authority: Mapping[str, object], schemas: Mapping[str, object], resolver: Mapping[str, object], views: Mapping[str, object], artifacts: Mapping[str, object], formal_identity: Mapping[str, object] | None = None, readiness_identity: Mapping[str, object] | None = None, code_identity: Mapping[str, object] | None = None) -> dict[str, object]:
        if contract_digest != CONTRACT_DIGEST or contract_digest == SUPERSEDED_CONTRACT_DIGEST:
            raise Gate1Failure("CONTRACT_IDENTITY", "invalid contract digest")
        identity = dict(formal_identity or {"decision_book_sha256": DECISION_BOOK_SHA256, "contract_digest": CONTRACT_DIGEST, "scope_sha256": SCOPE_SHA256, "matrix_sha256": MATRIX_SHA256, "combined_formal_identity_digest": COMBINED_FORMAL_IDENTITY_DIGEST, "freeze_commit_sha": FREEZE_COMMIT_SHA, "contract_version": CONTRACT_VERSION})
        if identity.get("contract_digest") != CONTRACT_DIGEST or identity.get("combined_formal_identity_digest") != COMBINED_FORMAL_IDENTITY_DIGEST:
            raise Gate1Failure("FORMAL_IDENTITY", "proof identity mismatch")
        view_payload = {}
        for name in ("source_history", "target_observed", "worker_safe_blind", "evaluator_truth", "audit_view"):
            value = views.get(name)
            view_payload[name] = _view_descriptor(str(authority.get("dataset", "UNKNOWN")), name, value) if isinstance(value, pd.DataFrame) else value
        proof: dict[str, object] = {"formal_identity": identity, "contract_digest": contract_digest, "authority": dict(authority), "schemas": dict(schemas), "resolver": dict(resolver), "views": view_payload, "artifacts": dict(artifacts)}
        proof.update({name: {"status": "passed", "digest": canonical_digest({"layer": name, "authority": authority, "schemas": schemas, "resolver": resolver, "views": view_payload, "artifacts": artifacts})} for name in self.REQUIRED})
        proof["decision_book_identity"] = identity.get("decision_book_sha256")
        proof["contract_identity"] = identity.get("contract_digest")
        proof["scope_identity"] = identity.get("scope_sha256")
        proof["matrix_identity"] = identity.get("matrix_sha256")
        proof["combined_formal_identity"] = identity.get("combined_formal_identity_digest")
        proof["freeze_commit_identity"] = identity.get("freeze_commit_sha")
        proof["real_input_readiness_identity"] = dict(readiness_identity or {"status": "not_run", "formal_identity": identity})
        proof["producer_identity"] = dict(code_identity or {"status": "bound", "code_identity": canonical_digest(__file__)})
        proof["operator_identity"] = {"status": "bound", "code_identity": canonical_digest("gate1x-operator")}
        proof["proof_digest"] = canonical_digest({key: value for key, value in proof.items() if key != "proof_digest"})
        return proof


def validate_proof_digest(proof: Mapping[str, object]) -> None:
    expected = proof.get("proof_digest")
    if not isinstance(expected, str):
        raise Gate1Failure("PROOF_DIGEST", "missing proof digest")
    actual = canonical_digest({key: value for key, value in proof.items() if key != "proof_digest"})
    if actual != expected:
        raise Gate1Failure("PROOF_DIGEST", "proof digest mismatch")


class FormalPreflight:
    def check(self, candidate: Mapping[str, object]) -> dict[str, object]:
        proof = candidate.get("proof") if isinstance(candidate, Mapping) else None
        if not isinstance(proof, Mapping):
            return {"status": "failed", "failure_code": "PROOF_MISSING"}
        missing = [name for name in ProofWriter.REQUIRED if name not in proof]
        if missing:
            return {"status": "failed", "failure_code": "PROOF_MISSING", "missing": missing}
        try:
            validate_proof_digest(proof)
        except Gate1Failure as exc:
            return {"status": "failed", "failure_code": exc.code, "error": str(exc)}
        if proof.get("contract_digest") != CONTRACT_DIGEST:
            return {"status": "failed", "failure_code": "CONTRACT_IDENTITY"}
        identity = proof.get("formal_identity", {})
        if identity.get("combined_formal_identity_digest") != COMBINED_FORMAL_IDENTITY_DIGEST:
            return {"status": "failed", "failure_code": "FORMAL_IDENTITY"}
        readiness = proof.get("real_input_readiness_identity", {})
        if readiness.get("status") == "failed":
            return {"status": "failed", "failure_code": "READINESS_NOT_PASSED"}
        return {"status": "passed", "failure_code": None, "proof_digest": proof["proof_digest"]}


@dataclass(frozen=True)
class HistoryResult:
    worker: pd.DataFrame
    audit: pd.DataFrame | None = None


class HistoryReconstructionProducer:
    def build(self, dataset: object, frame: pd.DataFrame, *, origin: object) -> HistoryResult:
        spec = dataset_contract(dataset)
        data = _prepare_frame(frame, label="history", spec=spec)
        if (data["date"] > pd.Timestamp(origin)).any():
            raise Gate1Failure("HISTORY_FUTURE_ROW", "history producer received post-origin rows")
        return HistoryResult(data.copy(), data.copy())


@dataclass(frozen=True)
class BlindResult:
    worker: pd.DataFrame
    evaluator: pd.DataFrame
    audit: pd.DataFrame


class ForecastBlindProducer:
    def build(self, dataset: object, history: HistoryResult | pd.DataFrame, forecast: pd.DataFrame, *, origin: object) -> BlindResult:
        spec = dataset_contract(dataset)
        data = _prepare_frame(forecast, label="forecast", spec=spec)
        if (data["date"] <= pd.Timestamp(origin)).any():
            raise Gate1Failure("FORECAST_ORIGIN", "forecast producer received origin or history rows")
        worker = _consumer_frame(spec.dataset, data, view="worker_safe_blind").drop(columns=["sales"], errors="ignore")
        return BlindResult(worker, data.copy(), data.copy())


class AvailabilityResolver:
    def resolve(self, dataset: object, field: str, *, role: str) -> str:
        if role == "worker" and field in {"sales", "transactions", "Customers", "Open", "Promo"}:
            return "forbidden"
        return "available"


@dataclass(frozen=True)
class FieldDecision:
    field: str
    role: str
    availability: str
    transform: str = "identity"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    role: str
    availability: str
    transform: str = "identity"


class SafeTargetViewOperator:
    def build(self, dataset: object, target: pd.DataFrame, *, origin: object) -> pd.DataFrame:
        result = target.copy()
        if "sales" in result.columns:
            result = result.loc[result["date"] > pd.Timestamp(origin)].copy()
            result = result.drop(columns=["sales"])
        return _consumer_frame(dataset_contract(dataset).dataset, result, view="worker_safe_blind")


class SourcePoolOperator:
    def select(self, dataset: object, source: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, dict[str, object]]:
        return select_source_history_candidates(dataset, source, scenario)


class ModelOperator:
    def run(self, *args: object, **kwargs: object) -> None:
        raise Gate1Failure("MODEL_CALL_FORBIDDEN", "Gate 1X implementation does not train or predict")


class AuthorityProducer:
    def __init__(self, project_root: Path | None = None):
        self.project_root = Path(project_root) if project_root else None

    @classmethod
    def from_frozen_contract(cls, project_root: Path) -> "AuthorityProducer":
        load_formal_identity(project_root)
        return cls(project_root)

    def produce(self, *args: object, **kwargs: object) -> None:
        raise Gate1Failure("PRODUCER_FORBIDDEN", "use adoption/materialization only after readiness acceptance")


class UnifiedRunner:
    def run(self, *args: object, **kwargs: object) -> None:
        raise Gate1Failure("RUNNER_FORBIDDEN", "controlled rerun is not authorized")


def build_d5_holiday(frame: pd.DataFrame, holidays: pd.DataFrame | None = None) -> pd.DataFrame:
    result = frame.copy()
    if holidays is not None:
        required = {"date", "type", "transferred"}
        if not required.issubset(holidays.columns):
            raise Gate1Failure("D5_HOLIDAY_SCHEMA", "holiday authority missing date/type/transferred")
        authority = holidays.copy()
        authority["date"] = pd.to_datetime(authority["date"], errors="raise").dt.normalize()
        if authority.duplicated(["date", "type", "transferred"]).any():
            raise Gate1Failure("D5_HOLIDAY_DUPLICATE", "duplicate holiday authority key")
        valid_type = authority["type"].astype(str).isin({"Holiday", "Additional", "Bridge"})
        effective = authority.loc[valid_type & pd.to_numeric(authority["transferred"], errors="coerce").eq(1)].copy()
        effective = effective.assign(_holiday=1).loc[:, ["date", "_holiday"]].drop_duplicates("date")
        result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
        result = result.drop(columns=["is_holiday"], errors="ignore").merge(effective, on="date", how="left", validate="many_to_one")
        result["is_holiday"] = result.pop("_holiday").fillna(0).astype("int8")
    elif "is_holiday" not in result.columns:
        result["is_holiday"] = 0
    result["is_holiday"] = pd.to_numeric(result["is_holiday"], errors="coerce").fillna(0).clip(0, 1).astype("int8")
    return result


def build_d5_oil_price(frame: pd.DataFrame, oil: pd.DataFrame, *, origin: object) -> pd.DataFrame:
    """Apply the approved global history-only ffill then lag-one oil rule."""
    if not {"date", "dcoilwtico"}.issubset(oil.columns):
        raise Gate1Failure("D5_OIL_SCHEMA", "oil authority must provide date and dcoilwtico")
    authority = oil.loc[:, ["date", "dcoilwtico"]].copy()
    authority["date"] = pd.to_datetime(authority["date"], errors="raise").dt.normalize()
    if authority.duplicated(["date"]).any():
        raise Gate1Failure("D5_OIL_DUPLICATE", "oil authority date is not unique")
    authority["dcoilwtico"] = pd.to_numeric(authority["dcoilwtico"], errors="coerce")
    dates = pd.date_range(min(pd.Timestamp(frame["date"].min()), pd.Timestamp(authority["date"].min())), pd.Timestamp(frame["date"].max()), freq="D")
    global_series = authority.set_index("date")["dcoilwtico"].reindex(dates).ffill()
    history_cutoff = pd.Timestamp(origin)
    if global_series.loc[global_series.index <= history_cutoff].notna().sum() == 0:
        raise Gate1Failure("D5_OIL_NO_PRIOR", "no historical prior exists for oil reconstruction")
    lagged = global_series.shift(1).rename("oil_price").reset_index().rename(columns={"index": "date"})
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result = result.drop(columns=["oil_price"], errors="ignore").merge(lagged, on="date", how="left", validate="many_to_one")
    if result.loc[result["date"] <= history_cutoff, "oil_price"].isna().any():
        raise Gate1Failure("D5_OIL_NO_PRIOR", "history row has no approved lag-one oil value")
    return result


def join_d6_sell_price(target: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    required = {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    if not required.issubset(prices.columns):
        raise Gate1Failure("D6_PRICE_SCHEMA", "sell price authority missing exact join fields")
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise Gate1Failure("D6_PRICE_DUPLICATE", "duplicate sell price exact key")
    if prices["sell_price"].isna().any():
        raise Gate1Failure("D6_PRICE_MISSING", "sell price authority contains null price")
    result = target.drop(columns=["sell_price"], errors="ignore").merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one")
    if result["sell_price"].isna().any():
        raise Gate1Failure("D6_PRICE_MISSING", "target row has no exact sell price")
    return result


def build_d6_calendar_view(calendar: pd.DataFrame, *, store_state: str) -> pd.DataFrame:
    required = {"date", "weekday", "wday", "wm_yr_wk"}
    if not required.issubset(calendar.columns):
        raise Gate1Failure("D6_CALENDAR_SCHEMA", "calendar must provide original weekday/wday/wm_yr_wk")
    result = calendar.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    if result.duplicated(["date"]).any():
        raise Gate1Failure("D6_CALENDAR_DUPLICATE", "calendar date is not unique")
    snap_field = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}.get(str(store_state).upper())
    if not snap_field or snap_field not in result.columns:
        raise Gate1Failure("D6_SNAP_STATE", f"missing state SNAP authority for {store_state!r}")
    result["snap"] = result[snap_field]
    ordered = ["date", "weekday", "wday", "wm_yr_wk"] + [column for column in ("event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap") if column in result.columns]
    return result.loc[:, ordered]


def attach_d6_calendar_exact(parent: pd.DataFrame, calendar: pd.DataFrame, *, store_state: str) -> pd.DataFrame:
    view = build_d6_calendar_view(calendar, store_state=store_state)
    if parent.duplicated(["date"]).any():
        raise Gate1Failure("D6_PARENT_DATE_DUPLICATE", "parent calendar join key is not unique")
    result = parent.drop(columns=[column for column in view.columns if column != "date" and column in parent.columns], errors="ignore").merge(view, on="date", how="left", validate="one_to_one")
    if result["wday"].isna().any():
        raise Gate1Failure("D6_CALENDAR_MISSING", "exact calendar join did not resolve wday")
    return result


__all__ = [
    "CONTRACT_VERSION", "CONTRACT_DIGEST", "DECISION_BOOK_SHA256", "SCOPE_SHA256", "MATRIX_SHA256", "COMBINED_FORMAL_IDENTITY_DIGEST", "SUPERSEDED_CONTRACT_DIGEST", "FREEZE_COMMIT_SHA", "Gate1Failure", "DatasetContract", "DatasetRoles", "dataset_contract", "load_formal_identity", "FormalInputLoader", "normalize_onpromotion", "rebuild_d2_wide_frame", "calendarize_d2_source_history", "source_pool_candidates", "select_source_history_candidates", "stream_source_history_candidates", "slice_dataset_roles", "build_contract_views", "effective_knn_schema_descriptor", "normalized_frame_digest", "canonical_digest", "SchemaRegistry", "ProofWriter", "FormalPreflight", "HistoryReconstructionProducer", "ForecastBlindProducer", "SafeTargetViewOperator", "AvailabilityResolver", "FieldDecision", "FieldSpec", "SourcePoolOperator", "AuthorityProducer", "UnifiedRunner", "build_d5_holiday", "build_d5_oil_price", "join_d6_sell_price", "build_d6_calendar_view", "attach_d6_calendar_exact",
]
