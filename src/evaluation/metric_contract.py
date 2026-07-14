"""Canonical contract for formally comparable original-sales-space sMAPE."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Mapping, Sequence

from src.protocols.experiment_protocol import serialize_canonical_target_key


METRIC_CONTRACT_VERSION = "smape_original_v1"
SMAPE_DEFINITION_ID = "smape_2abs_eps1e-8_pct_v1"
SMAPE_UNIT = "percent"
SMAPE_EPSILON = 1e-8
SMAPE_RANGE = (0.0, 200.0)
SALES_VALUE_POLICY = "clip_negative_to_zero_v1"
ORIGINAL_SALES_SPACE = "original_sales_space"
VALID_INVERSE_STATUSES = {"applied", "not_required"}
METRIC_IDENTITY_FIELDS = (
    "metric_target_key",
    "metric_horizon",
    "metric_sample_count",
    "metric_date_start",
    "metric_date_end",
    "metric_index_digest",
)

SMAPE_CONTRACT_FIELDS = {
    "metric_contract_version": METRIC_CONTRACT_VERSION,
    "smape_definition_id": SMAPE_DEFINITION_ID,
    "smape_unit": SMAPE_UNIT,
    "smape_epsilon": SMAPE_EPSILON,
    "smape_range_min": SMAPE_RANGE[0],
    "smape_range_max": SMAPE_RANGE[1],
    "sales_value_policy": SALES_VALUE_POLICY,
}


class MetricProtocolError(ValueError):
    """Typed, serializable failure raised when a strict metric contract fails."""

    def __init__(
        self,
        status: str,
        *,
        missing_fields: Sequence[str] = (),
        detail: str = "",
    ) -> None:
        self.status = str(status)
        self.missing_fields = tuple(str(value) for value in missing_fields)
        self.detail = str(detail)
        message = f"metric protocol error: {self.status}"
        if self.missing_fields:
            message += f"; missing_fields={','.join(self.missing_fields)}"
        if self.detail:
            message += f"; {self.detail}"
        super().__init__(message)


def compute_metric_index_digest(values: Sequence[Any]) -> str:
    """Return an order-sensitive digest for an independently built sample index."""
    encoded = json.dumps(
        [str(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_metric_identity_from_manifest(
    manifest: Any,
    *,
    horizon: int,
) -> dict[str, Any]:
    """Build the orchestration-owned metric identity for one manifest horizon."""
    records = tuple(manifest.for_horizon(int(horizon)))
    if not records:
        raise MetricProtocolError(
            "metric_identity_mismatch",
            detail=f"manifest has no samples for horizon={horizon}",
        )
    target_keys = {tuple(record.target_key) for record in records}
    if len(target_keys) != 1:
        raise MetricProtocolError(
            "metric_identity_mismatch",
            detail=f"manifest has multiple target keys: {sorted(target_keys)}",
        )
    label_dates = [str(record.label_date) for record in records]
    sample_keys = [str(record.sample_key) for record in records]
    dataset_ids = {str(record.dataset_id) for record in records}
    if len(dataset_ids) != 1:
        raise MetricProtocolError(
            "metric_identity_mismatch",
            detail=f"manifest has multiple dataset ids: {sorted(dataset_ids)}",
        )
    return {
        "metric_target_key": serialize_canonical_target_key(
            next(iter(dataset_ids)),
            next(iter(target_keys)),
        ),
        "metric_horizon": int(horizon),
        "metric_sample_count": len(records),
        "metric_date_start": label_dates[0],
        "metric_date_end": label_dates[-1],
        "metric_index_digest": compute_metric_index_digest(sample_keys),
    }


def validate_metric_identity(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Validate method output against an orchestration-owned sample identity."""
    missing = [field for field in METRIC_IDENTITY_FIELDS if field not in payload]
    if missing:
        raise MetricProtocolError("missing_metric_identity", missing_fields=missing)
    mismatches = [
        field
        for field in METRIC_IDENTITY_FIELDS
        if str(payload[field]) != str(expected.get(field))
    ]
    if mismatches:
        detail = "; ".join(
            f"{field}: expected={expected.get(field)!r} actual={payload.get(field)!r}"
            for field in mismatches
        )
        raise MetricProtocolError(
            "metric_identity_mismatch",
            missing_fields=mismatches,
            detail=detail,
        )


def _missing(row: Mapping[str, Any], field: str) -> bool:
    if field not in row:
        return True
    value = row[field]
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _contract_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def is_formally_comparable_smape_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed formal sMAPE eligibility with auditable failure reasons."""
    required = (
        "strict_paper_metrics",
        "paper_metric_space_requested",
        "paper_metric_space_actual",
        "primary_metric_space_actual",
        "smape_metric_space",
        "inverse_transform_status",
        "paper_metric_computed_valid",
        "paper_metric_status",
        "paper_metric_error",
        "smape",
        *METRIC_IDENTITY_FIELDS,
        "metric_contract_version",
        "smape_definition_id",
        "smape_unit",
        "smape_epsilon",
        "smape_range_min",
        "smape_range_max",
        "sales_value_policy",
        "target_negative_count",
    )
    reasons = []
    for field in required:
        if field == "paper_metric_error":
            if field not in row or row[field] is None:
                reasons.append(f"missing:{field}")
        elif _missing(row, field):
            reasons.append(f"missing:{field}")
    if reasons:
        return {"eligible": False, "failure_reasons": reasons}

    exact_values = {
        "paper_metric_space_requested": ORIGINAL_SALES_SPACE,
        "paper_metric_space_actual": ORIGINAL_SALES_SPACE,
        "primary_metric_space_actual": ORIGINAL_SALES_SPACE,
        "smape_metric_space": ORIGINAL_SALES_SPACE,
        "paper_metric_status": "valid",
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "smape_definition_id": SMAPE_DEFINITION_ID,
        "smape_unit": SMAPE_UNIT,
        "sales_value_policy": SALES_VALUE_POLICY,
    }
    for field, expected in exact_values.items():
        if row[field] != expected:
            reasons.append(f"invalid:{field}")
    if _contract_bool(row["strict_paper_metrics"]) is not True:
        reasons.append("invalid:strict_paper_metrics")
    if _contract_bool(row["paper_metric_computed_valid"]) is not True:
        reasons.append("invalid:paper_metric_computed_valid")
    if row["inverse_transform_status"] not in VALID_INVERSE_STATUSES:
        reasons.append("invalid:inverse_transform_status")
    if str(row["paper_metric_error"]).strip():
        reasons.append("invalid:paper_metric_error")
    if not _finite_number(row["smape"]) or not SMAPE_RANGE[0] <= float(row["smape"]) <= SMAPE_RANGE[1]:
        reasons.append("invalid:smape")
    if not _finite_number(row["metric_sample_count"]) or int(float(row["metric_sample_count"])) <= 0:
        reasons.append("invalid:metric_sample_count")
    if (
        not _finite_number(row["metric_horizon"])
        or int(float(row["metric_horizon"])) <= 0
        or float(row["metric_horizon"]) != int(float(row["metric_horizon"]))
    ):
        reasons.append("invalid:metric_horizon")
    if not _finite_number(row["smape_epsilon"]) or float(row["smape_epsilon"]) != SMAPE_EPSILON:
        reasons.append("invalid:smape_epsilon")
    if not _finite_number(row["smape_range_min"]) or float(row["smape_range_min"]) != SMAPE_RANGE[0]:
        reasons.append("invalid:smape_range_min")
    if not _finite_number(row["smape_range_max"]) or float(row["smape_range_max"]) != SMAPE_RANGE[1]:
        reasons.append("invalid:smape_range_max")
    if not _finite_number(row["target_negative_count"]) or int(float(row["target_negative_count"])) != 0:
        reasons.append("invalid:target_negative_count")
    return {"eligible": not reasons, "failure_reasons": reasons}


def filter_formally_comparable_smape_rows(frame: Any) -> tuple[Any, dict[str, int]]:
    """Filter a DataFrame through the canonical row contract and count exclusions."""
    eligible_indices = []
    exclusion_reasons: Counter[str] = Counter()
    for index, row in frame.iterrows():
        decision = is_formally_comparable_smape_row(row.to_dict())
        if decision["eligible"]:
            eligible_indices.append(index)
        else:
            exclusion_reasons.update(decision["failure_reasons"])
    return frame.loc[eligible_indices].copy(), dict(sorted(exclusion_reasons.items()))


def _canonical_sharing_scenario(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"with", "with_information_sharing", "info_sharing"}:
        return "with"
    if text in {
        "without",
        "without_information_sharing",
        "no_information",
        "no_info",
        "none",
    }:
        return "without"
    return text


def build_formal_smape_aggregates(
    frame: Any,
    *,
    expected_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build the fixed-scenario, fixed-horizon formal sMAPE aggregation hierarchy."""
    import pandas as pd

    eligible, reasons = filter_formally_comparable_smape_rows(frame)
    columns = [
        "dataset",
        "target",
        "method",
        "horizon",
        "sharing_scenario",
        "smape",
    ]
    if eligible.empty:
        empty = pd.DataFrame(columns=columns)
        return {
            "eligible_rows": eligible,
            "seed_mean": empty.copy(),
            "dataset_macro": empty.drop(columns=["target"]),
            "cross_dataset_macro": empty.drop(columns=["dataset", "target"]),
            "exclusion_reason_counts": reasons,
        }

    work = eligible.copy()
    if "dataset" not in work.columns:
        work["dataset"] = work.get("dataset_id")
    elif "dataset_id" in work.columns:
        work["dataset"] = work["dataset"].where(
            work["dataset"].notna() & work["dataset"].astype(str).str.strip().ne(""),
            work["dataset_id"],
        )
    if "target_entity_key" in work.columns:
        work["target"] = work["target_entity_key"]
    elif "target_entity_id" in work.columns:
        work["target"] = work["target_entity_id"]
    else:
        raise MetricProtocolError(
            "missing_formal_group_field",
            missing_fields=("target_entity_key",),
        )
    if "information_sharing" in work.columns:
        work["sharing_scenario"] = work["information_sharing"]
    else:
        work["sharing_scenario"] = work.get("scenario")
    for field in ("dataset", "target", "method", "horizon", "sharing_scenario", "seed"):
        if field not in work.columns:
            raise MetricProtocolError(
                "missing_formal_group_field",
                missing_fields=(field,),
            )
        missing = work[field].isna() | work[field].astype(str).str.strip().eq("")
        if missing.any():
            raise MetricProtocolError(
                "missing_formal_group_field",
                missing_fields=(field,),
                detail=f"missing_rows={int(missing.sum())}",
            )
    work["sharing_scenario"] = work["sharing_scenario"].map(
        _canonical_sharing_scenario
    )
    work["smape"] = pd.to_numeric(work["smape"], errors="coerce")
    work["horizon"] = pd.to_numeric(work["horizon"], errors="coerce")
    work["seed"] = pd.to_numeric(work["seed"], errors="coerce")
    invalid_numeric = work[["smape", "horizon", "seed"]].isna().any(axis=1)
    if invalid_numeric.any():
        raise MetricProtocolError(
            "invalid_formal_numeric_field",
            detail=f"invalid_rows={int(invalid_numeric.sum())}",
        )
    non_integral = (
        work["horizon"].ne(work["horizon"].astype(int))
        | work["seed"].ne(work["seed"].astype(int))
    )
    if non_integral.any():
        raise MetricProtocolError(
            "invalid_formal_numeric_field",
            detail=f"non_integral_rows={int(non_integral.sum())}",
        )
    work["horizon"] = work["horizon"].astype(int)
    work["seed"] = work["seed"].astype(int)
    metric_horizon = pd.to_numeric(work["metric_horizon"], errors="coerce")
    horizon_mismatch = metric_horizon.isna() | metric_horizon.ne(work["horizon"])
    target_mismatch = (
        work["metric_target_key"]
        .astype(str)
        .str.replace("/", "_", regex=False)
        .ne(work["target"].astype(str).str.replace("/", "_", regex=False))
    )
    identity_mismatch = horizon_mismatch | target_mismatch
    if identity_mismatch.any():
        mismatch_fields = []
        if horizon_mismatch.any():
            mismatch_fields.append("metric_horizon")
        if target_mismatch.any():
            mismatch_fields.append("metric_target_key")
        raise MetricProtocolError(
            "formal_metric_identity_mismatch",
            missing_fields=tuple(mismatch_fields),
            detail=f"mismatch_rows={int(identity_mismatch.sum())}",
        )

    group = ["dataset", "target", "method", "horizon", "sharing_scenario"]
    full_key = [*group, "seed"]
    duplicated = work.duplicated(full_key, keep=False)
    if duplicated.any():
        duplicate_keys = work.loc[duplicated, full_key].to_dict(orient="records")
        raise MetricProtocolError(
            "duplicate_formal_seed_row",
            detail=f"duplicate_keys={duplicate_keys}",
        )
    if expected_seeds is not None:
        normalized_expected = tuple(int(seed) for seed in expected_seeds)
        if not normalized_expected or len(set(normalized_expected)) != len(
            normalized_expected
        ):
            raise MetricProtocolError(
                "invalid_expected_formal_seeds",
                detail=f"expected_seeds={normalized_expected}",
            )
        expected_set = set(normalized_expected)
        for key, seed_group in work.groupby(group, dropna=False, sort=True):
            actual_set = set(int(seed) for seed in seed_group["seed"])
            if actual_set != expected_set:
                raise MetricProtocolError(
                    "formal_seed_set_mismatch",
                    detail=(
                        f"group={key}; "
                        f"missing={sorted(expected_set - actual_set)}; "
                        f"unexpected={sorted(actual_set - expected_set)}"
                    ),
                )
    seed_mean = work.groupby(group, as_index=False, dropna=False)["smape"].mean()
    dataset_group = ["dataset", "method", "horizon", "sharing_scenario"]
    dataset_macro = seed_mean.groupby(dataset_group, as_index=False, dropna=False)["smape"].mean()
    cross_group = ["method", "horizon", "sharing_scenario"]
    cross_dataset = dataset_macro.groupby(cross_group, as_index=False, dropna=False)["smape"].mean()
    cross_dataset["rank"] = cross_dataset.groupby(
        ["horizon", "sharing_scenario"]
    )["smape"].rank(method="average", ascending=True)
    return {
        "eligible_rows": work,
        "seed_mean": seed_mean,
        "dataset_macro": dataset_macro,
        "cross_dataset_macro": cross_dataset,
        "exclusion_reason_counts": dict(sorted(reasons.items())),
    }
