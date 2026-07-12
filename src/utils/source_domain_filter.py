from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import pandas as pd


DOMAIN_FILTER_SCOPES = frozenset(
    {"target_only", "source_pool", "target_and_source", "metadata_only"}
)


@dataclass(frozen=True)
class SourceDomainPolicyResult:
    frame: pd.DataFrame
    diagnostics: Dict[str, Any]


def normalize_domain_filter(domain_filter: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Normalize KNN JSON domain_filter shapes into a column -> allowed value map."""
    if not domain_filter:
        return {}
    raw = dict(domain_filter)
    if "column" in raw:
        if "value" not in raw:
            raise ValueError(f"domain_filter with column requires value: {raw}")
        return {str(raw["column"]): raw["value"]}
    return {str(key): value for key, value in raw.items()}


def _is_without_information_sharing(information_sharing: str) -> bool:
    return str(information_sharing).strip().lower() in {
        "without",
        "without_information_sharing",
        "no_information",
    }


def domain_filter_mask(source_df: pd.DataFrame, normalized_filter: Mapping[str, Any]) -> pd.Series:
    missing = [column for column in normalized_filter if column not in source_df.columns]
    if missing:
        raise ValueError(f"source_domain_filter missing columns: {missing}")

    mask = pd.Series(True, index=source_df.index)
    for column, allowed in normalized_filter.items():
        if isinstance(allowed, (list, tuple, set, frozenset)):
            mask &= source_df[column].isin(list(allowed))
        else:
            mask &= source_df[column] == allowed
    return mask


def _apply_filter(source_df: pd.DataFrame, normalized_filter: Dict[str, Any]) -> pd.DataFrame:
    return source_df.loc[domain_filter_mask(source_df, normalized_filter)].copy()


def _normalize_scope(domain_filter_scope: object) -> str:
    scope = str(domain_filter_scope).strip().lower()
    if scope not in DOMAIN_FILTER_SCOPES:
        raise ValueError(
            f"unsupported domain_filter_scope {domain_filter_scope!r}; "
            f"expected one of {sorted(DOMAIN_FILTER_SCOPES)!r}"
        )
    return scope


def _domain_filter_details(normalized_filter: Mapping[str, Any]) -> tuple[object, object]:
    if len(normalized_filter) != 1:
        return None, None
    return next(iter(normalized_filter.items()))


def _source_entity_count(
    source_df: pd.DataFrame,
    entity_group_cols: Sequence[str] | None,
) -> int | None:
    """Count source entities only when an explicit or well-known key is available."""
    if "entity_id" in source_df.columns:
        return int(source_df[["entity_id"]].drop_duplicates().shape[0])

    candidates = []
    if entity_group_cols:
        candidates.append(tuple(str(column) for column in entity_group_cols))
    candidates.extend(
        (
            ("source_entity_key",),
            ("store_id", "product_id"),
            ("store_nbr", "item_nbr"),
            ("store_id", "item_id"),
        )
    )
    for columns in candidates:
        if columns and all(column in source_df.columns for column in columns):
            return int(source_df.loc[:, list(columns)].drop_duplicates().shape[0])
    return None


def apply_source_domain_policy(
    source_df: pd.DataFrame,
    knn_json_domain_filter: Mapping[str, Any] | None,
    information_sharing: str,
    entity_group_cols: Sequence[str] | None = None,
    domain_filter_scope: str = "source_pool",
    source_pool_policy: str | None = None,
) -> SourceDomainPolicyResult:
    """Apply source filtering only when the explicit scope includes the source pool."""
    before = int(len(source_df))
    entities_before = _source_entity_count(source_df, entity_group_cols)
    raw_filter = dict(knn_json_domain_filter or {})
    normalized = normalize_domain_filter(raw_filter)
    scope = _normalize_scope(domain_filter_scope)
    filter_column, filter_value = _domain_filter_details(normalized)
    diagnostics: Dict[str, Any] = {
        "knn_json_domain_filter": raw_filter,
        "source_domain_filter": None,
        "source_domain_filter_applied": False,
        "domain_filter_applied_to_source": False,
        "domain_filter_scope": scope,
        "domain_filter_column": filter_column,
        "domain_filter_value": filter_value,
        "target_domain_validation_passed": None,
        "source_pool_policy": source_pool_policy or "",
        "source_domain_filter_reason": "with_information_sharing_all_source_pool",
        "source_pool_size_before_filter": before,
        "source_pool_size_after_filter": before,
        "source_pool_rows_before_filter": before,
        "source_pool_rows_after_filter": before,
        "excluded_source_row_count": 0,
        "source_pool_entities_before_filter": entities_before,
        "source_pool_entities_after_filter": entities_before,
        "excluded_source_entity_count": 0 if entities_before is not None else None,
        "source_domain_filter_error": "",
    }

    if scope not in {"source_pool", "target_and_source"}:
        diagnostics["source_domain_filter_reason"] = f"domain_filter_{scope}"
        frame = source_df.copy()
        frame.attrs.update(source_df.attrs)
        return SourceDomainPolicyResult(frame=frame, diagnostics=diagnostics)

    if not _is_without_information_sharing(information_sharing):
        frame = source_df.copy()
        frame.attrs.update(source_df.attrs)
        return SourceDomainPolicyResult(frame=frame, diagnostics=diagnostics)

    diagnostics["source_domain_filter"] = normalized
    diagnostics["source_domain_filter_reason"] = "without_information_sharing_same_domain_protocol"
    if not normalized:
        diagnostics["source_domain_filter_error"] = "without information sharing requires source_domain_filter"
        raise ValueError(diagnostics["source_domain_filter_error"])

    try:
        frame = _apply_filter(source_df, normalized)
    except Exception as exc:
        diagnostics["source_domain_filter_error"] = str(exc)
        raise

    after = int(len(frame))
    entities_after = _source_entity_count(frame, entity_group_cols)
    diagnostics["source_domain_filter_applied"] = True
    diagnostics["domain_filter_applied_to_source"] = True
    diagnostics["source_pool_size_after_filter"] = after
    diagnostics["source_pool_rows_after_filter"] = after
    diagnostics["excluded_source_row_count"] = before - after
    diagnostics["source_pool_entities_after_filter"] = entities_after
    diagnostics["excluded_source_entity_count"] = (
        entities_before - entities_after
        if entities_before is not None and entities_after is not None
        else None
    )
    if after == 0:
        diagnostics["source_domain_filter_error"] = (
            f"source_domain_filter removed all source rows: {normalized}"
        )
        raise ValueError(diagnostics["source_domain_filter_error"])
    frame.attrs.update(source_df.attrs)
    return SourceDomainPolicyResult(frame=frame, diagnostics=diagnostics)
