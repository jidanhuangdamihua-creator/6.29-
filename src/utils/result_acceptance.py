"""Fail-closed acceptance for formal cell, mode-matrix, and aggregate results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.constants import RESULT_SCHEMA_COLUMNS
from src.protocols.experiment_protocol import (
    FORMAL_HORIZONS,
    FORMAL_METHODS,
    FORMAL_PROTOCOL_TRACK,
    ProtocolViolation,
    formal_target_entity_keys,
    normalize_scenario,
)
from src.utils.result_schema import REGISTERED_RESULT_EXTRA_COLUMNS_BY_SCHEMA_FAMILY
from src.utils.result_validation import classify_protocol_result, validate_seed_bundle_coverage
from src.utils.artifact_rehydration import resolve_bound_artifact
from src.utils.run_recovery import RunRecovery


FORMAL_KEY_COLUMNS = (
    "dataset_id",
    "protocol_track",
    "scenario",
    "target_entity_key",
    "method",
    "horizon",
    "seed",
)


class AcceptanceScope(str, Enum):
    CELL = "cell"
    MODE_MATRIX = "mode_matrix"
    GLOBAL_AGGREGATE = "global_aggregate"


class AggregateProfile(str, Enum):
    RUN_SELECTION_AGGREGATE = "run_selection_aggregate"
    FULL_D1_D6_BASELINE = "full_d1_d6_baseline"


class ResultAcceptanceError(RuntimeError):
    """Raised when a caller requires a passing acceptance outcome."""


def resolve_accepted_artifact(
    recovery: RunRecovery,
    logical_artifact_id: str,
    *,
    expected_binding_set_digest: str | None = None,
) -> Path:
    """Resolve accepted inputs exclusively through the attempt's frozen binding set."""
    return resolve_bound_artifact(
        recovery,
        logical_artifact_id,
        expected_binding_set_digest=expected_binding_set_digest,
    )


@dataclass(frozen=True)
class ExpectedResultContract:
    scope: AcceptanceScope
    formal: bool
    dataset_ids: tuple[int, ...]
    modes: tuple[str, ...]
    protocol_tracks: tuple[str, ...]
    targets_by_dataset_mode: Mapping[tuple[int, str], tuple[str, ...]]
    methods: tuple[str, ...]
    horizons: tuple[int, ...]
    seeds: tuple[int, ...]
    confirmation_eligible: bool
    aggregate_profile: AggregateProfile | None = None


@dataclass(frozen=True)
class AcceptanceReport:
    passed: bool
    scope: AcceptanceScope
    reasons: tuple[str, ...]
    counts: Mapping[str, int]
    aggregate_profile: AggregateProfile | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "scope": self.scope.value,
            "reasons": list(self.reasons),
            "counts": dict(self.counts),
            "aggregate_profile": (
                self.aggregate_profile.value if self.aggregate_profile is not None else None
            ),
        }


@dataclass(frozen=True)
class AcceptanceOutcome:
    report: AcceptanceReport
    accepted_rows: pd.DataFrame


def build_formal_cell_contract(
    *,
    dataset_id: int,
    mode: str,
    targets: Sequence[str],
    horizon: int,
    seed: int,
) -> ExpectedResultContract:
    normalized_mode = normalize_scenario(mode)
    canonical_targets = formal_target_entity_keys(dataset_id)
    requested_targets = tuple(str(target) for target in targets)
    if requested_targets != canonical_targets:
        raise ProtocolViolation(
            f"D{int(dataset_id)} formal targets must be {canonical_targets!r}, "
            f"got {requested_targets!r}"
        )
    return ExpectedResultContract(
        scope=AcceptanceScope.CELL,
        formal=True,
        dataset_ids=(int(dataset_id),),
        modes=(normalized_mode,),
        protocol_tracks=(FORMAL_PROTOCOL_TRACK,),
        targets_by_dataset_mode={
            (int(dataset_id), normalized_mode): canonical_targets
        },
        methods=FORMAL_METHODS,
        horizons=(int(horizon),),
        seeds=(int(seed),),
        confirmation_eligible=True,
    )


def build_formal_seed_bundle_contract(
    *,
    dataset_id: int,
    mode: str,
    targets: Sequence[str],
    seed: int,
) -> ExpectedResultContract:
    """Return the exact acceptance contract for one h1-h5 seed bundle."""

    contract = build_formal_cell_contract(
        dataset_id=dataset_id,
        mode=mode,
        targets=targets,
        horizon=FORMAL_HORIZONS[0],
        seed=seed,
    )
    return replace(contract, horizons=FORMAL_HORIZONS)


def require_accepted(outcome: AcceptanceOutcome) -> pd.DataFrame:
    if not outcome.report.passed:
        raise ResultAcceptanceError(
            f"{outcome.report.scope.value} acceptance failed: "
            + ",".join(outcome.report.reasons)
        )
    return outcome.accepted_rows


def accept_formal_cell_output(
    path: Path,
    *,
    dataset_id: int,
    mode: str,
    targets: Sequence[str],
    horizon: int,
    seed: int,
) -> pd.DataFrame:
    expected = build_formal_cell_contract(
        dataset_id=dataset_id,
        mode=mode,
        targets=targets,
        horizon=horizon,
        seed=seed,
    )
    return require_accepted(accept_cell_csv(Path(path), expected=expected))


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _report(
    scope: AcceptanceScope,
    reasons: Sequence[str],
    counts: Mapping[str, int],
    *,
    profile: AggregateProfile | None = None,
) -> AcceptanceReport:
    normalized = _dedupe(reasons)
    return AcceptanceReport(not normalized, scope, normalized, dict(counts), profile)


def _dataset_id(value: object) -> int:
    text = str(value).strip().upper()
    if text.startswith("DATASET"):
        text = text[7:]
    elif text.startswith("D"):
        text = text[1:]
    return int(text)


def _read_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    candidate = Path(path)
    if not candidate.is_file():
        return pd.DataFrame(), ["csv_missing"]
    try:
        frame = pd.read_csv(candidate, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame(), ["csv_unreadable"]
    if frame.empty:
        return frame, ["csv_empty"]
    return frame, []


def _string_values(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("").astype(str).str.strip()


def _terminal_error_reasons(frame: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    if "error" in frame.columns and _string_values(frame, "error").ne("").any():
        reasons.append("terminal_error")
    if "result_status" in frame.columns:
        statuses = _string_values(frame, "result_status")
        if statuses.isin({"failed", "protocol_invalid", "legacy_unverified"}).any():
            reasons.append("terminal_status")
    return reasons


def _normalized_modes(frame: pd.DataFrame, column: str) -> tuple[pd.Series | None, list[str]]:
    try:
        values = frame[column].map(normalize_scenario)
    except (KeyError, ValueError):
        return None, ["invalid_mode"]
    return values, []


def _normalized_dataset_ids(frame: pd.DataFrame) -> tuple[pd.Series | None, list[str]]:
    try:
        values = frame["dataset_id"].map(_dataset_id)
    except (KeyError, TypeError, ValueError):
        return None, ["invalid_dataset_id"]
    return values, []


def _formal_keys(frame: pd.DataFrame) -> list[tuple[object, ...]]:
    dataset_ids = frame["dataset_id"].map(_dataset_id)
    modes = frame["scenario"].map(normalize_scenario)
    return list(
        zip(
            dataset_ids,
            frame["protocol_track"].astype(str),
            modes,
            frame["target_entity_key"].astype(str),
            frame["method"].astype(str),
            frame["horizon"].astype(int),
            frame["seed"].astype(int),
        )
    )


def _expected_keys(expected: ExpectedResultContract) -> set[tuple[object, ...]]:
    return {
        (dataset_id, track, mode, target, method, horizon, seed)
        for dataset_id in expected.dataset_ids
        for mode in expected.modes
        for track in expected.protocol_tracks
        for target in expected.targets_by_dataset_mode.get((dataset_id, mode), ())
        for method in expected.methods
        for horizon in expected.horizons
        for seed in expected.seeds
    }


def _same_candidate_content(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if set(left.columns) != set(right.columns):
        return False
    order = sorted(left.columns)
    sort_columns = [column for column in FORMAL_KEY_COLUMNS if column in order]
    normalized = []
    for frame in (left, right):
        value = frame.loc[:, order].copy()
        if sort_columns:
            value = value.sort_values(sort_columns, kind="stable")
        normalized.append(value.reset_index(drop=True).astype(str))
    return normalized[0].equals(normalized[1])


def _schema_reasons(frame: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    if "schema_family" not in frame.columns:
        return ["missing_required_columns"]
    families = set(_string_values(frame, "schema_family"))
    if "" in families or not families.issubset(
        REGISTERED_RESULT_EXTRA_COLUMNS_BY_SCHEMA_FAMILY
    ):
        return ["unknown_schema_family"]
    allowed = set(RESULT_SCHEMA_COLUMNS)
    for family in families:
        allowed.update(REGISTERED_RESULT_EXTRA_COLUMNS_BY_SCHEMA_FAMILY[family])
    if set(frame.columns).difference(allowed):
        reasons.append("unregistered_columns")
    return reasons


def _finite_metric_reasons(frame: pd.DataFrame) -> list[str]:
    reasons: list[str] = []
    for column in ("rmse", "smape"):
        if column not in frame.columns:
            reasons.append("missing_required_columns")
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            reasons.append("nonfinite_primary_metric")
    return reasons


def _identity_reasons(
    frame: pd.DataFrame,
    expected: ExpectedResultContract,
    *,
    require_exact_keys: bool,
) -> list[str]:
    reasons: list[str] = []
    dataset_ids, dataset_reasons = _normalized_dataset_ids(frame)
    modes, mode_reasons = _normalized_modes(frame, "scenario")
    reasons.extend(dataset_reasons)
    reasons.extend(mode_reasons)
    if dataset_ids is None or modes is None:
        return reasons
    if set(dataset_ids) != set(expected.dataset_ids):
        reasons.append("unexpected_dataset")
    if set(modes) != set(expected.modes):
        reasons.append("unexpected_mode")
    if "information_sharing" not in frame.columns:
        reasons.append("missing_required_columns")
    else:
        aliases, alias_reasons = _normalized_modes(frame, "information_sharing")
        reasons.extend(alias_reasons)
        if aliases is not None and not aliases.equals(modes):
            reasons.append("mode_alias_mismatch")
    if set(_string_values(frame, "protocol_track")) != set(expected.protocol_tracks):
        reasons.append("unexpected_protocol_track")
    try:
        horizons = set(frame["horizon"].astype(int))
        seeds = set(frame["seed"].astype(int))
    except (KeyError, TypeError, ValueError):
        reasons.append("invalid_horizon_seed")
        return reasons
    if horizons != set(expected.horizons):
        reasons.append("unexpected_horizon")
    if seeds != set(expected.seeds):
        reasons.append("unexpected_seed")
    try:
        keys = _formal_keys(frame)
    except (KeyError, TypeError, ValueError):
        reasons.append("invalid_formal_key")
        return reasons
    if len(keys) != len(set(keys)):
        reasons.append("duplicate_formal_key")
    if require_exact_keys:
        expected_keys = _expected_keys(expected)
        actual_keys = set(keys)
        if actual_keys != expected_keys:
            actual_methods = set(frame["method"].astype(str)) if "method" in frame else set()
            if actual_methods != set(expected.methods):
                reasons.append("method_coverage_mismatch")
            actual_targets = {
                (dataset_id, mode, target)
                for dataset_id, mode, target in zip(
                    dataset_ids,
                    modes,
                    frame["target_entity_key"].astype(str),
                )
            }
            expected_targets = {
                (dataset_id, mode, target)
                for (dataset_id, mode), targets in expected.targets_by_dataset_mode.items()
                for target in targets
            }
            if actual_targets != expected_targets:
                reasons.append("target_coverage_mismatch")
            if not reasons or actual_keys != expected_keys:
                reasons.append("formal_key_coverage_mismatch")
    return reasons


def accept_cell_csv(
    path: Path,
    *,
    expected: ExpectedResultContract,
) -> AcceptanceOutcome:
    frame, reasons = _read_csv(Path(path))
    counts = {"rows": len(frame)}
    if reasons:
        report = _report(AcceptanceScope.CELL, reasons, counts)
        return AcceptanceOutcome(report, frame.iloc[0:0].copy())
    if expected.scope is not AcceptanceScope.CELL:
        reasons.append("scope_mismatch")
    if not expected.confirmation_eligible:
        reasons.append("confirmation_ineligible")
    required = {
        *FORMAL_KEY_COLUMNS,
        "information_sharing",
        "schema_family",
        "result_status",
        "error",
        "rmse",
        "smape",
    }
    if required.difference(frame.columns):
        reasons.append("missing_required_columns")
    reasons.extend(_terminal_error_reasons(frame))
    if "result_status" in frame.columns and not _string_values(
        frame, "result_status"
    ).eq("trial").all():
        reasons.append("cell_status_must_be_trial")
    reasons.extend(_finite_metric_reasons(frame))
    reasons.extend(_schema_reasons(frame))
    if not required.difference(frame.columns):
        if tuple(expected.horizons) == FORMAL_HORIZONS and len(expected.seeds) == 1:
            try:
                validate_seed_bundle_coverage(frame, seed=expected.seeds[0])
            except ValueError:
                reasons.append("seed_bundle_coverage_mismatch")
        reasons.extend(_identity_reasons(frame, expected, require_exact_keys=True))
        classifications = [
            classify_protocol_result(record) for record in frame.to_dict(orient="records")
        ]
        if any(status != "trial" for status in classifications):
            reasons.append("protocol_invalid")
    report = _report(AcceptanceScope.CELL, reasons, counts)
    return AcceptanceOutcome(
        report,
        frame.copy() if report.passed else frame.iloc[0:0].copy(),
    )


def accept_mode_matrix(
    cell_paths: Sequence[Path],
    *,
    expected: ExpectedResultContract,
    candidate_mode_csv: Path | None = None,
) -> AcceptanceOutcome:
    reasons: list[str] = []
    counts = {"cells": len(cell_paths), "rows": 0}
    if expected.scope is not AcceptanceScope.MODE_MATRIX:
        reasons.append("scope_mismatch")
    expected_horizons = set(expected.horizons)
    expected_seeds = set(expected.seeds)
    discovered: dict[int, Path] = {}
    for path in cell_paths:
        frame, read_reasons = _read_csv(Path(path))
        if read_reasons:
            reasons.extend(read_reasons)
            continue
        try:
            pairs = set(zip(frame["horizon"].astype(int), frame["seed"].astype(int)))
        except (KeyError, TypeError, ValueError):
            reasons.append("invalid_horizon_seed")
            continue
        seeds = {seed for _, seed in pairs}
        horizons = {horizon for horizon, _ in pairs}
        if len(seeds) != 1 or horizons != expected_horizons:
            reasons.append("cell_identity_mismatch")
            continue
        seed = next(iter(seeds))
        if pairs != {(horizon, seed) for horizon in expected_horizons}:
            reasons.append("cell_identity_mismatch")
            continue
        if seed in discovered:
            reasons.append("duplicate_cell")
        discovered[seed] = Path(path)
    if set(discovered) != expected_seeds or len(cell_paths) != len(expected_seeds):
        reasons.append("cell_matrix_mismatch")

    accepted_frames: list[pd.DataFrame] = []
    if not reasons:
        for seed in sorted(discovered):
            cell_expected = replace(
                expected,
                scope=AcceptanceScope.CELL,
                horizons=tuple(expected.horizons),
                seeds=(seed,),
            )
            outcome = accept_cell_csv(discovered[seed], expected=cell_expected)
            if not outcome.report.passed:
                reasons.append("cell_acceptance_failed")
                reasons.extend(outcome.report.reasons)
            else:
                accepted_frames.append(outcome.accepted_rows)
    combined = (
        pd.concat(accepted_frames, ignore_index=True, sort=False)
        if accepted_frames
        else pd.DataFrame()
    )
    counts["rows"] = len(combined)
    if not reasons:
        reasons.extend(_identity_reasons(combined, expected, require_exact_keys=True))
    if not reasons:
        combined = combined.copy()
        combined["result_status"] = "confirmed_baseline"
    if not reasons and candidate_mode_csv is not None:
        candidate, candidate_reasons = _read_csv(Path(candidate_mode_csv))
        reasons.extend(candidate_reasons)
        if not candidate_reasons and not _same_candidate_content(candidate, combined):
            reasons.append("candidate_mode_mismatch")
    report = _report(AcceptanceScope.MODE_MATRIX, reasons, counts)
    return AcceptanceOutcome(
        report,
        combined if report.passed else combined.iloc[0:0].copy(),
    )


def accept_global_aggregate(
    mode_paths: Sequence[Path],
    *,
    expected: ExpectedResultContract,
    candidate_aggregate_csv: Path | None = None,
) -> AcceptanceOutcome:
    reasons: list[str] = []
    frames: list[pd.DataFrame] = []
    if expected.scope is not AcceptanceScope.GLOBAL_AGGREGATE:
        reasons.append("scope_mismatch")
    if expected.aggregate_profile is None:
        reasons.append("aggregate_profile_required")
    for path in mode_paths:
        frame, read_reasons = _read_csv(Path(path))
        reasons.extend(read_reasons)
        if not read_reasons:
            frames.append(frame)
    combined = (
        pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    )
    counts = {"mode_files": len(mode_paths), "rows": len(combined), "dataset_mode_groups": 0}
    if combined.empty and "csv_empty" not in reasons:
        reasons.append("csv_empty")
    required = {
        *FORMAL_KEY_COLUMNS,
        "information_sharing",
        "schema_family",
        "result_status",
        "error",
        "rmse",
        "smape",
    }
    if not combined.empty and required.difference(combined.columns):
        reasons.append("missing_required_columns")
    if not combined.empty and not required.difference(combined.columns):
        reasons.extend(_terminal_error_reasons(combined))
        if not _string_values(combined, "result_status").eq("confirmed_baseline").all():
            reasons.append("aggregate_requires_confirmed_rows")
        reasons.extend(_finite_metric_reasons(combined))
        reasons.extend(_schema_reasons(combined))
        reasons.extend(_identity_reasons(combined, expected, require_exact_keys=True))
        dataset_ids = combined["dataset_id"].map(_dataset_id)
        modes = combined["scenario"].map(normalize_scenario)
        actual_groups = set(zip(dataset_ids, modes))
        expected_groups = {
            (dataset_id, mode)
            for dataset_id in expected.dataset_ids
            for mode in expected.modes
        }
        counts["dataset_mode_groups"] = len(actual_groups)
        if actual_groups != expected_groups:
            reasons.append("dataset_mode_coverage_mismatch")
        if len(mode_paths) != len(expected_groups):
            reasons.append("mode_file_coverage_mismatch")
        expected_rows = len(_expected_keys(expected))
        if len(combined) != expected_rows:
            reasons.append("aggregate_row_count_mismatch")
        if expected.aggregate_profile is AggregateProfile.FULL_D1_D6_BASELINE:
            if expected_groups != {
                (dataset_id, mode)
                for dataset_id in range(1, 7)
                for mode in ("without", "with")
            }:
                reasons.append("full_profile_contract_mismatch")
            if expected_rows != 5400:
                reasons.append("full_profile_row_count_mismatch")
    if not reasons and candidate_aggregate_csv is not None:
        candidate, candidate_reasons = _read_csv(Path(candidate_aggregate_csv))
        reasons.extend(candidate_reasons)
        if not candidate_reasons and not _same_candidate_content(candidate, combined):
            reasons.append("candidate_aggregate_mismatch")
    report = _report(
        AcceptanceScope.GLOBAL_AGGREGATE,
        reasons,
        counts,
        profile=expected.aggregate_profile,
    )
    return AcceptanceOutcome(
        report,
        combined if report.passed else combined.iloc[0:0].copy(),
    )
