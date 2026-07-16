"""Gate 1I implementation of the frozen transformation contract.

This module is deliberately closed-world: fields enter a formal view only when
the Gate 1 contract names them.  The implementation is fixture-friendly and
does not read, materialize, train, or predict on formal datasets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from .feature_schema import get_knn_schema, get_predictor_schema


CONTRACT_DIGEST = "sha256:b145028c2b3f8314e66fc73be9795269644d016a7a1cf258a9f62f1b7443d09e"
CONTRACT_VERSION = "1.0.0"
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
D6_CALENDAR_INPUTS = (
    "weekday",
    "wday",
    "wm_yr_wk",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
)
D6_CALENDAR_OUTPUTS = (
    "weekday",
    "wday",
    "wm_yr_wk",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap",
)
D4_AUDIT_ONLY = (
    "hours_sale",
    "hours_stock_status",
    "stock_hour6_22_cnt",
)
D5_FORBIDDEN_FORECAST = ("transactions", "week")
D3_FORBIDDEN_MODEL = ("Open", "Customers", "Promo", "Promo2", "PromoInterval")
_DATE_FIELDS = ("date", "year", "month", "day")
_D6_STATES = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
_GENERIC_FILL_TOKENS = {"bfill", "ffill", "backfill", "forward_fill", "generic_fill", "mean"}


class Gate1Failure(ValueError):
    """Stable fail-closed error carrying a contract-facing failure code."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message or self.code}")


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frame_digest(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    rows = []
    for row in normalized.itertuples(index=False, name=None):
        rows.append([None if pd.isna(value) else value for value in row])
    payload = {
        "columns": list(normalized.columns),
        "dtypes": [str(dtype) for dtype in normalized.dtypes],
        "rows": rows,
    }
    return _digest(payload)


def canonical_digest(value: object) -> str:
    """Return the Gate 1 canonical JSON digest for an identity payload."""

    return _digest(value)


def normalized_frame_digest(frame: pd.DataFrame) -> str:
    """Digest a frame by logical values, column order, and dtypes, not parquet bytes."""

    return _frame_digest(_as_frame(frame, label="canonical content"))


def validate_proof_digest(proof: Mapping[str, object]) -> None:
    """Fail closed unless ``proof_digest`` binds every other proof field."""

    payload = dict(proof)
    declared = payload.pop("proof_digest", None)
    if not isinstance(declared, str) or declared != _digest(payload):
        raise Gate1Failure("PROOF_DIGEST", "proof digest does not bind its payload")


def _as_frame(value: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise Gate1Failure("FRAME_TYPE", f"{label} must be a pandas DataFrame")
    frame = value.copy(deep=True)
    if "date" in frame.columns:
        parsed = pd.to_datetime(frame["date"], errors="coerce")
        if parsed.isna().any():
            raise Gate1Failure("DATE_INVALID", f"{label} contains an invalid date")
        frame["date"] = parsed
    return frame


def _add_date_fields(frame: pd.DataFrame) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise Gate1Failure("DATE_MISSING", "date is required for a formal view")
    out = frame.copy()
    out["year"] = out["date"].dt.year.astype("int64")
    out["month"] = out["date"].dt.month.astype("int64")
    out["day"] = out["date"].dt.day.astype("int64")
    return out


@dataclass(frozen=True)
class AuthorityManifest:
    files: Mapping[str, Mapping[str, object]]
    expected_hashes: Mapping[str, str]
    snapshot_id: str
    digest: str


class FormalInputLoader:
    """Load only the three version-controlled Gate 1 formal authority files."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()

    def load(self, paths: Sequence[Path] | None = None) -> dict[str, bytes]:
        selected = FORMAL_INPUTS if paths is None else tuple(str(Path(path)) for path in paths)
        if selected != FORMAL_INPUTS:
            raise Gate1Failure("AUTHORITY_PATH", "formal input path set differs from the three frozen files")
        loaded: dict[str, bytes] = {}
        for relative in FORMAL_INPUTS:
            path = (self.project_root / relative).resolve()
            if not path.is_file() or path.parent != (self.project_root / Path(relative).parent).resolve():
                raise Gate1Failure("AUTHORITY_PATH", f"formal input is missing or redirected: {relative}")
            loaded[relative] = path.read_bytes()
        actual_digest = "sha256:" + hashlib.sha256(loaded[FORMAL_INPUTS[0]]).hexdigest()
        if actual_digest != CONTRACT_DIGEST:
            raise Gate1Failure("CONTRACT_DIGEST", "frozen contract digest mismatch")
        sidecar = self.project_root / "docs/protocol/gate1_frozen_transformation_contract.sha256"
        if not sidecar.is_file() or CONTRACT_DIGEST not in sidecar.read_text(encoding="utf-8"):
            raise Gate1Failure("CONTRACT_DIGEST", "contract digest sidecar mismatch")
        return loaded


class AuthorityProducer:
    """Read a caller-declared raw authority set and bind bytes to their hashes."""

    def __init__(
        self,
        *,
        root: Path,
        files: Mapping[str, Path | str],
        expected_hashes: Mapping[str, str],
    ) -> None:
        self.root = Path(root).resolve()
        self.files = {str(key): Path(value) for key, value in files.items()}
        self.expected_hashes = {str(key): str(value).lower() for key, value in expected_hashes.items()}
        if set(self.files) != set(self.expected_hashes):
            raise Gate1Failure("AUTHORITY_MANIFEST", "authority files and expected hashes must match exactly")

    def load(self) -> AuthorityManifest:
        records: dict[str, Mapping[str, object]] = {}
        for name, relative in self.files.items():
            if relative.is_absolute():
                raise Gate1Failure("AUTHORITY_PATH", f"absolute authority path is not allowed: {relative}")
            path = (self.root / relative).resolve()
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise Gate1Failure("AUTHORITY_PATH", f"authority path escapes root: {relative}") from exc
            if any(token in str(path).lower() for token in ("/tmp/", ".sealed", "private_build", "legacy")):
                raise Gate1Failure("AUTHORITY_PATH", f"temporary or private authority rejected: {relative}")
            if not path.is_file():
                raise Gate1Failure("AUTHORITY_MISSING", f"authority file is missing: {relative}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            expected = self.expected_hashes[name]
            if actual != expected:
                raise Gate1Failure("RAW_HASH_DRIFT", f"hash mismatch for {name}")
            stat = path.stat()
            records[name] = {
                "path": str(relative),
                "size_bytes": int(stat.st_size),
                "sha256": actual,
                "read_at": datetime.now(timezone.utc).isoformat(),
            }
        snapshot_id = _digest({name: records[name]["sha256"] for name in sorted(records)})
        return AuthorityManifest(
            files=MappingProxyType(records),
            expected_hashes=MappingProxyType(dict(self.expected_hashes)),
            snapshot_id=snapshot_id,
            digest=_digest({"snapshot_id": snapshot_id, "files": records}),
        )

    @classmethod
    def from_frozen_contract(cls, project_root: Path) -> "AuthorityProducer":
        """Bind exactly the raw paths and hashes listed in the frozen contract."""
        root = Path(project_root).resolve()
        contract = root / "docs/protocol/gate1_frozen_transformation_contract.md"
        if not contract.is_file():
            raise Gate1Failure("CONTRACT_MISSING", "frozen contract is missing")
        if "sha256:" + hashlib.sha256(contract.read_bytes()).hexdigest() != CONTRACT_DIGEST:
            raise Gate1Failure("CONTRACT_DIGEST", "frozen contract digest mismatch")
        files: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        for line in contract.read_text(encoding="utf-8").splitlines():
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 5 or not parts[1].startswith("D") or not parts[-2]:
                continue
            dataset, relative, digest = parts[1], parts[2], parts[-2].lower()
            if not relative.startswith("数据集/原始数据/"):
                continue
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                continue
            name = f"{dataset}:{relative}"
            files[name] = Path(relative)
            hashes[name] = digest
        if not files:
            raise Gate1Failure("AUTHORITY_MANIFEST", "frozen contract contains no raw authority table")
        return cls(root=root, files=files, expected_hashes=hashes)


@dataclass(frozen=True)
class FieldDecision:
    field: str
    dataset: str
    authority: str
    available_at: str
    history_rule: str
    forecast_rule: str
    missing_rule: str
    status: str
    failure_code: str | None = None


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: str
    role: str
    transform: str
    consumers: tuple[str, ...]

    def descriptor(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "role": self.role,
            "transform": self.transform,
            "consumers": list(self.consumers),
        }


class AvailabilityResolver:
    """Resolve field-level availability and refuse generic future filling."""

    def resolve(
        self,
        dataset: str,
        field: str,
        *,
        available_at: object,
        fill_method: str | None = None,
        origin: object | None = None,
    ) -> FieldDecision:
        dataset = str(dataset).upper()
        field = str(field)
        method = None if fill_method is None else str(fill_method).lower().replace("-", "_")
        known_by_dataset = {
            "D1": set(_DATE_FIELDS),
            "D2": set(_DATE_FIELDS) | {"PROMO", "Promo"},
            "D3": set(_DATE_FIELDS) | set(D3_FORBIDDEN_MODEL) | {"SchoolHoliday"},
            "D4": set(_DATE_FIELDS) | set(D4_APPROVED_FUTURE) | set(D4_AUDIT_ONLY),
            "D5": set(_DATE_FIELDS) | {"perishable", "onpromotion", "transactions", "oil_price", "is_holiday", "week"},
            "D6": set(_DATE_FIELDS) | set(D6_CALENDAR_OUTPUTS) | set(D6_CALENDAR_INPUTS) | {"sell_price"},
        }
        if field not in known_by_dataset.get(dataset, set()):
            raise Gate1Failure("UNKNOWN_FIELD", f"{dataset}/{field} is not declared by the contract")
        if method in _GENERIC_FILL_TOKENS:
            if (dataset, field) in {("D2", "Promo"), ("D3", "Promo"), ("D5", "oil_price")}:
                raise Gate1Failure("MISSING_RULE", f"{dataset}/{field} cannot use {fill_method}")
            raise Gate1Failure("GENERIC_FILL", f"generic fill is forbidden for {dataset}/{field}")
        if origin is not None and str(available_at)[:10] > str(origin)[:10]:
            raise Gate1Failure("AVAILABILITY_LATE", f"{dataset}/{field} is not available at origin")
        rules = {
            ("D3", "Open"): ("rossmann historical sales", "never forecast", "sales-dependent zero/one"),
            ("D3", "SchoolHoliday"): ("rossmann raw", "raw benchmark field", "missing=0"),
            ("D5", "transactions"): ("favorita transactions", "exclude forecast", "history missing=0"),
            ("D5", "oil_price"): ("favorita oil.csv", "prior-only lag-one", "prior-only ffill"),
            ("D5", "onpromotion"): ("favorita train/test", "benchmark future-known", "missing=0"),
            ("D5", "perishable"): ("favorita items.csv", "static-known", "missing=fail"),
            ("D6", "sell_price"): ("m5 sell_prices.csv", "exact-key benchmark future-known", "missing=fail"),
        }
        authority, forecast_rule, missing_rule = rules.get(
            (dataset, field),
            ("contract-declared authority", "contract-declared rule", "contract-declared rule"),
        )
        return FieldDecision(
            field=field,
            dataset=dataset,
            authority=authority,
            available_at=str(available_at)[:10],
            history_rule=forecast_rule if field != "Open" else "sales-dependent historical reconstruction",
            forecast_rule=forecast_rule,
            missing_rule=missing_rule,
            status="accepted",
        )


class SchemaRegistry:
    """Closed-world view schema registry for Gate 1 safe target views."""
    _dtypes = MappingProxyType(
        {
            "year": "int64",
            "month": "int64",
            "day": "int64",
            "SchoolHoliday": "int64",
            "onpromotion": "int64",
            "is_holiday": "int64",
            "snap": "int64",
        }
    )

    def allowed(self, dataset: str, view: str) -> tuple[str, ...]:
        dataset = str(dataset).upper()
        view = str(view).lower()
        if dataset not in {f"D{number}" for number in range(1, 7)}:
            raise Gate1Failure("SCHEMA_DATASET", f"unknown dataset: {dataset}")
        if view == "audit":
            return ()
        if view == "knn":
            return ("date",) + get_knn_schema(dataset).ordered_names
        if view in {"worker", "forecast", "model"}:
            names = tuple(
                name for name in get_predictor_schema(dataset).ordered_names if name != "sales"
            )
            return ("date",) + names
        raise Gate1Failure("SCHEMA_VIEW", f"unknown view: {view}")

    def validate(self, dataset: str, view: str, frame: pd.DataFrame) -> None:
        frame = _as_frame(frame, label=f"{dataset}/{view}")
        dataset = str(dataset).upper()
        view = str(view).lower()
        allowed = set(self.allowed(dataset, view))
        extras = sorted(set(frame.columns) - allowed)
        if extras:
            raise Gate1Failure("SCHEMA_EXTRA", f"{dataset}/{view} contains unregistered fields: {extras}")
        if dataset == "D5" and "week" in frame.columns:
            raise Gate1Failure("SCHEMA_EXTRA", "week is deleted from all D5 formal views")
        if view in {"worker", "forecast", "model"}:
            forbidden = set(D3_FORBIDDEN_MODEL if dataset == "D3" else ())
            forbidden.update(D5_FORBIDDEN_FORECAST if dataset == "D5" else ())
            forbidden.update(D4_AUDIT_ONLY if dataset == "D4" else ())
            hit = sorted(forbidden.intersection(frame.columns))
            if hit:
                raise Gate1Failure("SCHEMA_FORBIDDEN", f"forbidden fields in {dataset}/{view}: {hit}")
        expected_dtypes = {spec.name: spec.dtype for spec in self.fields(dataset, view)}
        for name, expected in expected_dtypes.items():
            if name == "date" or name not in frame.columns:
                continue
            actual = str(frame[name].dtype)
            if actual != expected:
                raise Gate1Failure("SCHEMA_DTYPE", f"{dataset}/{view}/{name}: expected {expected}, got {actual}")

    def fields(self, dataset: str, view: str) -> tuple[FieldSpec, ...]:
        dataset = str(dataset).upper()
        view = str(view).lower()
        specs: list[FieldSpec] = []
        predictor = get_predictor_schema(dataset)
        knn = get_knn_schema(dataset)
        for name in self.allowed(dataset, view):
            if name == "date":
                dtype = "datetime64[ns]"
            elif view == "knn":
                dtype = knn.field(name).dtype
            else:
                dtype = predictor.field(name).dtype
            role = "key" if name == "date" else "future-known"
            if name == "sales":
                role = "target"
            if name in {"event_name_1", "event_type_1", "event_name_2", "event_type_2"}:
                dtype = "object"
            consumers = ("knn",) if view == "knn" else ("cnn", "rfe", "transfer")
            specs.append(FieldSpec(name, dtype, role, "identity", consumers))
        return tuple(specs)

    def digest(self, dataset: str, view: str) -> str:
        fields = [spec.descriptor() for spec in self.fields(dataset, view)]
        return _digest({"dataset": str(dataset).upper(), "view": str(view).lower(), "fields": fields})


@dataclass(frozen=True)
class HistoryResult:
    frame: pd.DataFrame
    repair_counts: Mapping[str, int]
    affected_rows: tuple[str, ...]
    repair_mask_digest: str
    source_digest: str


class HistoryReconstructionProducer:
    """Reconstruct only observed historical facts, never forecast rows."""

    def build(self, dataset: str, frame: pd.DataFrame, *, origin: object | None = None) -> HistoryResult:
        dataset = str(dataset).upper()
        out = _add_date_fields(_as_frame(frame, label=f"{dataset} history"))
        if origin is not None:
            cutoff = pd.Timestamp(origin)
            if (out["date"] > cutoff).any():
                raise Gate1Failure("HISTORY_FUTURE_ROW", f"{dataset} history contains a post-origin row")
        repairs: dict[str, int] = {}
        affected: list[str] = []
        if dataset == "D3":
            if "Open" in out.columns:
                if "sales" not in out.columns:
                    raise Gate1Failure("OPEN_NO_SALES", "D3 Open reconstruction requires historical sales")
                missing = out["Open"].isna()
                out.loc[missing, "Open"] = (out.loc[missing, "sales"] > 0).astype("int64")
                out["Open"] = out["Open"].astype("int64")
                repairs["Open"] = int(missing.sum())
                affected.extend(out.index[missing].astype(str).tolist())
            if "SchoolHoliday" in out.columns:
                missing = out["SchoolHoliday"].isna()
                out["SchoolHoliday"] = pd.to_numeric(out["SchoolHoliday"], errors="coerce").fillna(0).astype("int64")
                repairs["SchoolHoliday"] = int(missing.sum())
                affected.extend(out.index[missing].astype(str).tolist())
        if dataset == "D5":
            if "transactions" in out.columns:
                missing = out["transactions"].isna()
                out["transactions"] = pd.to_numeric(out["transactions"], errors="coerce").fillna(0)
                repairs["transactions"] = int(missing.sum())
                affected.extend(out.index[missing].astype(str).tolist())
            if "onpromotion" in out.columns:
                missing = out["onpromotion"].isna()
                out["onpromotion"] = pd.to_numeric(out["onpromotion"], errors="coerce").fillna(0).astype("int64")
                repairs["onpromotion"] = int(missing.sum())
                affected.extend(out.index[missing].astype(str).tolist())
            if "oil_price" in out.columns:
                raw = pd.to_numeric(out["oil_price"], errors="coerce")
                if raw.notna().sum() == 0:
                    raise Gate1Failure("OIL_NO_PRIOR", "D5 oil_price has no historical prior")
                filled = raw.ffill()
                lagged = filled.shift(1)
                if lagged.notna().sum() == 0:
                    raise Gate1Failure("OIL_NO_PRIOR", "D5 oil_price has no prior at any row")
                missing = raw.isna()
                out["oil_price"] = lagged
                repairs["oil_price"] = int(missing.sum())
                affected.extend(out.index[missing].astype(str).tolist())
        if dataset == "D4" and "provenance" in out.columns:
            if out["provenance"].astype(str).str.lower().isin(_GENERIC_FILL_TOKENS).any():
                raise Gate1Failure("GENERIC_FILL", "D4 generic bfill/ffill provenance is forbidden")
        repair_mask = {"counts": repairs, "affected_rows": affected}
        return HistoryResult(
            frame=out,
            repair_counts=MappingProxyType(dict(repairs)),
            affected_rows=tuple(affected),
            repair_mask_digest=_digest(repair_mask),
            source_digest=_frame_digest(out),
        )


@dataclass(frozen=True)
class BlindResult:
    worker: pd.DataFrame
    forecast_covariates: pd.DataFrame
    audit: pd.DataFrame
    exclusions: Mapping[str, str]
    manifest_classes: Mapping[str, str]
    digest: str


class ForecastBlindProducer:
    """Generate an origin-blind view by selecting only contract-approved fields."""

    def build(
        self,
        dataset: str,
        history: HistoryResult,
        forecast: pd.DataFrame,
        *,
        origin: object,
    ) -> BlindResult:
        dataset = str(dataset).upper()
        source = _add_date_fields(_as_frame(forecast, label=f"{dataset} forecast"))
        cutoff = pd.Timestamp(origin)
        if (source["date"] <= cutoff).any():
            raise Gate1Failure("FORECAST_ORIGIN", f"{dataset} forecast includes an origin or historical row")
        if "provenance" in source.columns and source["provenance"].astype(str).str.lower().isin(_GENERIC_FILL_TOKENS).any():
            raise Gate1Failure("GENERIC_FILL", "generic fill provenance is forbidden")
        registry = SchemaRegistry()
        allowed = set(registry.allowed(dataset, "forecast"))
        worker_columns = [
            column
            for column in registry.allowed(dataset, "forecast")
            if column in source.columns and column != "sales"
        ]
        worker = source.loc[:, worker_columns].copy()
        audit_columns = [column for column in source.columns if column not in set(worker_columns)]
        audit = source.loc[:, audit_columns].copy()
        exclusions = {column: "not approved for origin-blind forecast view" for column in audit_columns}
        classes: dict[str, str] = {}
        if dataset == "D4":
            classes = {field: "benchmark-provided future covariate" for field in D4_APPROVED_FUTURE if field in worker}
        if dataset == "D5" and "onpromotion" in worker:
            worker["onpromotion"] = pd.to_numeric(worker["onpromotion"], errors="coerce").fillna(0).astype("int64")
        if dataset == "D5" and "oil_price" in worker:
            if "oil_price" not in history.frame.columns:
                raise Gate1Failure("OIL_NO_PRIOR", "forecast oil_price has no historical authority")
            prior = history.frame["oil_price"].dropna()
            if prior.empty:
                raise Gate1Failure("OIL_NO_PRIOR", "forecast oil_price has no historical prior")
            worker["oil_price"] = float(prior.iloc[-1])
        if dataset == "D6":
            for field in D6_CALENDAR_OUTPUTS:
                if field in worker and field == "snap":
                    worker[field] = worker[field].astype("int64")
        for field in get_predictor_schema(dataset).fields:
            if field.name not in worker:
                continue
            try:
                worker[field.name] = worker[field.name].astype(field.dtype)
            except (TypeError, ValueError) as exc:
                raise Gate1Failure(
                    "SCHEMA_DTYPE",
                    f"{dataset}/forecast/{field.name} cannot normalize to {field.dtype}",
                ) from exc
        registry.validate(dataset, "forecast", worker)
        return BlindResult(
            worker=worker,
            forecast_covariates=worker.copy(deep=True),
            audit=audit,
            exclusions=MappingProxyType(exclusions),
            manifest_classes=MappingProxyType(classes),
            digest=_digest({"worker": _frame_digest(worker), "exclusions": exclusions}),
        )


@dataclass(frozen=True)
class SafeTargetViews:
    worker: pd.DataFrame
    knn: pd.DataFrame
    forecast: pd.DataFrame
    label_truth: pd.DataFrame
    evaluator_truth: pd.DataFrame
    audit: pd.DataFrame
    sample_range: tuple[str, str]
    digests: Mapping[str, str]


class SafeTargetViewOperator:
    """Separate worker, KNN, forecast covariates, labels, truth, and audit views."""

    def build(self, dataset: str, history: pd.DataFrame, forecast: pd.DataFrame) -> SafeTargetViews:
        dataset = str(dataset).upper()
        history_frame = _add_date_fields(_as_frame(history, label=f"{dataset} history view"))
        forecast_frame = _add_date_fields(_as_frame(forecast, label=f"{dataset} forecast view"))
        registry = SchemaRegistry()
        worker = forecast_frame.loc[:, [
            column
            for column in registry.allowed(dataset, "worker")
            if column in forecast_frame.columns and column != "sales"
        ]].copy()
        knn = history_frame.loc[:, [
            column
            for column in registry.allowed(dataset, "knn")
            if column in history_frame.columns
        ]].copy()
        predictor = get_predictor_schema(dataset)
        knn_schema = get_knn_schema(dataset)
        for column in worker.columns:
            if column != "date":
                worker[column] = worker[column].astype(predictor.field(column).dtype)
        for column in knn.columns:
            if column != "date":
                knn[column] = knn[column].astype(knn_schema.field(column).dtype)
        forecast_view = worker.copy(deep=True)
        label_truth = forecast_frame.loc[:, [c for c in forecast_frame.columns if c in {"date", "sales"}]].copy()
        evaluator_truth = forecast_frame.copy(deep=True)
        used = set(worker.columns) | set(knn.columns) | {"date", "sales"}
        audit_columns = sorted((set(history_frame.columns) | set(forecast_frame.columns)) - used)
        audit_parts = [frame.reindex(columns=audit_columns) for frame in (history_frame, forecast_frame)]
        audit = pd.concat(audit_parts, axis=0, ignore_index=True) if audit_columns else pd.DataFrame()
        registry.validate(dataset, "worker", worker)
        registry.validate(dataset, "knn", knn)
        if dataset == "D4":
            for name in D4_AUDIT_ONLY:
                if name in history_frame.columns:
                    if name not in audit.columns:
                        raise Gate1Failure("AUDIT_MISSING", f"D4 audit field was dropped: {name}")
        dates = pd.concat([history_frame["date"], forecast_frame["date"]], ignore_index=True)
        sample_range = (dates.min().date().isoformat(), dates.max().date().isoformat())
        digests = MappingProxyType({
            "worker": _frame_digest(worker),
            "knn": _frame_digest(knn),
            "forecast": _frame_digest(forecast_view),
            "label": _frame_digest(label_truth),
            "evaluator": _frame_digest(evaluator_truth),
            "audit": _frame_digest(audit),
        })
        return SafeTargetViews(worker, knn, forecast_view, label_truth, evaluator_truth, audit, sample_range, digests)


class SourcePoolOperator:
    target_store = 10

    @staticmethod
    def region_for_store(store: int) -> int:
        try:
            store = int(store)
        except (TypeError, ValueError) as exc:
            raise Gate1Failure("DOMAIN_MAPPING", f"invalid Store: {store!r}") from exc
        if not 1 <= store <= 30:
            raise Gate1Failure("DOMAIN_MAPPING", f"Store outside frozen 1-30 domain: {store}")
        return (store - 1) // 10 + 1

    def candidates(self, scenario: str) -> tuple[int, ...]:
        value = str(scenario).lower()
        if value == "without":
            return tuple(range(1, 10))
        if value == "with":
            return tuple(store for store in range(1, 31) if store != self.target_store)
        raise Gate1Failure("DOMAIN_SCENARIO", f"unknown source-sharing scenario: {scenario}")

    @staticmethod
    def validate_domain(domain: str) -> None:
        normalized = str(domain).strip().lower()
        if normalized in {"region = 1", "region=1", "todo_region_unavailable"}:
            raise Gate1Failure("DOMAIN", f"legacy domain token is forbidden: {domain}")
        if "todo_region_unavailable" in normalized or normalized.startswith("region ="):
            raise Gate1Failure("DOMAIN", f"legacy domain token is forbidden: {domain}")

    def select(self, scenario: str, available: Sequence[int]) -> Mapping[str, object]:
        eligible = tuple(sorted(set(int(value) for value in available)))
        expected = self.candidates(scenario)
        if self.target_store in eligible:
            raise Gate1Failure("DOMAIN_TARGET_INCLUDED", "target Store 10 entered source pool")
        selected = tuple(store for store in expected if store in eligible)
        if selected != expected:
            raise Gate1Failure("DOMAIN_SOURCE_MISSING", "candidate pool is not the frozen source set")
        return {
            "eligible_source_count": len(selected),
            "excluded_target_count": int(self.target_store in eligible),
            "ordered_selected_sources": selected,
            "candidate_pool_digest": _digest(expected),
            "selection_result_digest": _digest(selected),
        }


class ModelOperator:
    """Accept only safe views and registered fields; no fit/predict is performed."""

    def consume(self, dataset: str, views: SafeTargetViews, *, method: str) -> Mapping[str, object]:
        if not method or not str(method).strip():
            raise Gate1Failure("MODEL_METHOD", "method is required")
        SchemaRegistry().validate(dataset, "worker", views.worker)
        SchemaRegistry().validate(dataset, "knn", views.knn)
        return {
            "dataset": str(dataset).upper(),
            "method": str(method),
            "worker_digest": views.digests["worker"],
            "knn_digest": views.digests["knn"],
            "provenance_code": _digest({"module": __name__, "method": str(method)}),
        }


class ProofWriter:
    """Create a complete identity-bound proof object for preflight consumption."""

    def build(
        self,
        *,
        contract_digest: str,
        authority: Mapping[str, object],
        schemas: Mapping[str, object],
        resolver: Mapping[str, object],
        views: Mapping[str, object],
        artifacts: Mapping[str, object],
    ) -> dict[str, object]:
        if contract_digest != CONTRACT_DIGEST:
            raise Gate1Failure("PROOF_CONTRACT", "proof is not bound to the frozen contract")
        required_views = {"worker", "knn", "forecast", "label", "audit"}
        if set(views) != required_views:
            raise Gate1Failure("PROOF_VIEWS", "proof must enumerate all five safe views")
        proof: dict[str, object] = {
            "contract_digest": contract_digest,
            "contract_version": CONTRACT_VERSION,
            "authority": authority,
            "schemas": schemas,
            "resolver": resolver,
            "views": views,
            "artifacts": artifacts,
            "row_cardinality": {"before": 0, "after": 0},
        }
        proof["proof_digest"] = _digest(proof)
        return proof


class FormalPreflight:
    """Block every forbidden state before a formal train/predict call."""

    def check(self, state: Mapping[str, object]) -> Mapping[str, object]:
        state = dict(state)
        if "contract_digest" in state and state["contract_digest"] != CONTRACT_DIGEST:
            raise Gate1Failure("CONTRACT_DIGEST", "contract digest mismatch")
        proof = state.get("proof")
        if proof is not None:
            if not isinstance(proof, Mapping) or proof.get("contract_digest") != CONTRACT_DIGEST:
                raise Gate1Failure("PROOF_CONTRACT", "proof identity mismatch")
        if state.get("target_day_actual"):
            raise Gate1Failure("TARGET_DAY_ACTUAL", "target-day actual entered preflight")
        forbidden = state.get("forbidden_fields")
        if forbidden:
            raise Gate1Failure("FORBIDDEN_FIELD", f"forbidden fields: {forbidden}")
        generic_fill = state.get("generic_fill")
        if generic_fill:
            raise Gate1Failure("GENERIC_FILL", f"generic fill provenance: {generic_fill}")
        candidates = state.get("candidate_sources")
        if candidates is not None and 10 in set(candidates):
            raise Gate1Failure("DOMAIN_TARGET_INCLUDED", "candidate pool contains Store 10")
        if state.get("domain"):
            SourcePoolOperator.validate_domain(str(state["domain"]))
        before = state.get("row_count_before")
        after = state.get("row_count_after")
        if before is not None and after is not None and int(before) != int(after):
            raise Gate1Failure("ROW_EXPANSION", "row cardinality changed")
        if state.get("proof_complete") is False:
            raise Gate1Failure("PROOF_INCOMPLETE", "proof completeness check failed")
        for flag, code in (
            ("raw_hash_drift", "RAW_HASH_DRIFT"),
            ("schema_drift", "SCHEMA_DRIFT"),
            ("availability_failure", "AVAILABILITY_FAILURE"),
            ("view_overlap", "VIEW_OVERLAP"),
            ("domain_failure", "DOMAIN_MAPPING"),
            ("artifact_hash_failure", "ARTIFACT_HASH"),
        ):
            if state.get(flag):
                raise Gate1Failure(code, f"preflight flag is set: {flag}")
        if not isinstance(proof, Mapping):
            raise Gate1Failure("PROOF_MISSING", "formal preflight requires a proof object")
        required = {"contract_digest", "authority", "schemas", "resolver", "views", "artifacts", "proof_digest"}
        missing = sorted(required - set(proof))
        if missing:
            raise Gate1Failure("PROOF_INCOMPLETE", f"proof is missing: {missing}")
        validate_proof_digest(proof)
        return {"status": "passed", "contract_digest": CONTRACT_DIGEST, "checks": 13}


class UnifiedRunner:
    """Single dry-run entry point for the frozen path; legacy runners are inert."""

    def __init__(self, *, legacy_runners: Mapping[str, Callable[[], object]] | None = None) -> None:
        self.legacy_runners = dict(legacy_runners or {})
        self.preflight = FormalPreflight()

    def dry_run(self, *, dataset: str, state: Mapping[str, object]) -> Mapping[str, object]:
        if str(dataset).upper() not in {f"D{i}" for i in range(1, 7)}:
            raise Gate1Failure("RUNNER_DATASET", f"unknown dataset: {dataset}")
        preflight = self.preflight.check(state)
        return {
            "dataset": str(dataset).upper(),
            "path": "frozen schema -> availability gate -> safe target view -> model",
            "preflight": preflight,
            "legacy_runners_called": False,
        }


def build_d5_holiday(
    holidays: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    store_state: str,
    store_city: str,
) -> pd.Series:
    """Map Favorita holidays to one value per date without merge expansion."""
    required = {"date", "type", "locale", "locale_name", "transferred"}
    missing = required - set(holidays.columns)
    if missing:
        raise Gate1Failure("HOLIDAY_SCHEMA", f"missing holiday fields: {sorted(missing)}")
    expected = pd.DatetimeIndex(dates)
    if expected.has_duplicates or not expected.is_monotonic_increasing:
        raise Gate1Failure("HOLIDAY_DATES", "expected dates must be unique and sorted")
    table = holidays.copy()
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    table = table.loc[table["date"].notna()].copy()
    values: dict[pd.Timestamp, int] = {}
    for row in table.itertuples(index=False):
        date = pd.Timestamp(getattr(row, "date"))
        holiday_type = str(getattr(row, "type"))
        locale = str(getattr(row, "locale"))
        locale_name = str(getattr(row, "locale_name"))
        transferred = bool(getattr(row, "transferred"))
        applicable = locale == "National" or (locale == "Regional" and locale_name == store_state) or (locale == "Local" and locale_name == store_city)
        if not applicable:
            continue
        if transferred:
            values[date] = 0
            continue
        values[date] = max(values.get(date, 0), int(holiday_type in {"Holiday", "Additional", "Bridge", "Transfer"}))
    return pd.Series([values.get(date, 0) for date in expected], index=expected, dtype="int64", name="is_holiday")


def join_d6_sell_price(target: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join M5 prices on the exact frozen three-column key, fail closed."""
    keys = ["store_id", "item_id", "wm_yr_wk"]
    for name, frame in (("target", target), ("prices", prices)):
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            raise Gate1Failure("PRICE_SCHEMA", f"{name} is missing price keys: {missing}")
    if prices.duplicated(keys).any():
        raise Gate1Failure("PRICE_DUPLICATE_KEY", "sell_price contains a duplicate exact key")
    merged = target.merge(prices[keys + ["sell_price"]], on=keys, how="left", validate="many_to_one", sort=False)
    if len(merged) != len(target):
        raise Gate1Failure("PRICE_ROW_CARDINALITY", "sell_price join expanded rows")
    if merged["sell_price"].isna().any():
        missing = int(merged["sell_price"].isna().sum())
        if int(merged[keys].merge(prices[keys], on=keys, how="left", indicator=True)["_merge"].eq("left_only").sum()) > 0:
            raise Gate1Failure("PRICE_MISSING_KEY", f"{missing} target price key(s) are absent")
        raise Gate1Failure("PRICE_MISSING_VALUE", "sell_price is missing for an exact key")
    return merged


def build_d6_calendar_view(calendar: pd.DataFrame, *, store_state: str) -> pd.DataFrame:
    """Select the sole calendar authority and one state-specific SNAP column."""
    extra = sorted(set(calendar.columns) - set(D6_CALENDAR_INPUTS))
    if extra:
        raise Gate1Failure("CALENDAR_EXTRA", f"calendar contains unapproved fields: {extra}")
    state = str(store_state).upper()
    if state not in _D6_STATES:
        raise Gate1Failure("SNAP_STATE", f"unknown or unmapped store state: {store_state}")
    missing = sorted(set(D6_CALENDAR_INPUTS) - set(calendar.columns))
    if missing:
        raise Gate1Failure("CALENDAR_MISSING", f"calendar is missing fields: {missing}")
    out = calendar.loc[:, [field for field in D6_CALENDAR_INPUTS if field not in {"snap_CA", "snap_TX", "snap_WI"}]].copy()
    out["snap"] = calendar[_D6_STATES[state]].astype("int64")
    return out.loc[:, list(D6_CALENDAR_OUTPUTS)]


__all__ = [
    "AuthorityManifest",
    "AuthorityProducer",
    "AvailabilityResolver",
    "BlindResult",
    "CONTRACT_DIGEST",
    "CONTRACT_VERSION",
    "D4_APPROVED_FUTURE",
    "D6_CALENDAR_INPUTS",
    "FieldSpec",
    "FormalInputLoader",
    "FormalPreflight",
    "ForecastBlindProducer",
    "Gate1Failure",
    "HistoryReconstructionProducer",
    "HistoryResult",
    "ModelOperator",
    "ProofWriter",
    "SafeTargetViewOperator",
    "SafeTargetViews",
    "SchemaRegistry",
    "SourcePoolOperator",
    "UnifiedRunner",
    "build_d5_holiday",
    "build_d6_calendar_view",
    "join_d6_sell_price",
]
