from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import pandas as pd


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


def _apply_filter(source_df: pd.DataFrame, normalized_filter: Dict[str, Any]) -> pd.DataFrame:
    missing = [column for column in normalized_filter if column not in source_df.columns]
    if missing:
        raise ValueError(f"source_domain_filter missing columns: {missing}")

    mask = pd.Series(True, index=source_df.index)
    for column, allowed in normalized_filter.items():
        if isinstance(allowed, (list, tuple, set, frozenset)):
            mask &= source_df[column].isin(list(allowed))
        else:
            mask &= source_df[column] == allowed
    return source_df.loc[mask].copy()


def apply_source_domain_policy(
    source_df: pd.DataFrame,
    knn_json_domain_filter: Mapping[str, Any] | None,
    information_sharing: str,
) -> SourceDomainPolicyResult:
    """Apply without-mode source filtering while keeping with-mode full source pool."""
    before = int(len(source_df))
    raw_filter = dict(knn_json_domain_filter or {})
    diagnostics: Dict[str, Any] = {
        "knn_json_domain_filter": raw_filter,
        "source_domain_filter": None,
        "source_domain_filter_applied": False,
        "source_domain_filter_reason": "with_information_sharing_all_source_pool",
        "source_pool_size_before_filter": before,
        "source_pool_size_after_filter": before,
        "source_domain_filter_error": "",
    }

    if not _is_without_information_sharing(information_sharing):
        frame = source_df.copy()
        frame.attrs.update(source_df.attrs)
        return SourceDomainPolicyResult(frame=frame, diagnostics=diagnostics)

    normalized = normalize_domain_filter(raw_filter)
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
    diagnostics["source_domain_filter_applied"] = True
    diagnostics["source_pool_size_after_filter"] = after
    if after == 0:
        diagnostics["source_domain_filter_error"] = (
            f"source_domain_filter removed all source rows: {normalized}"
        )
        raise ValueError(diagnostics["source_domain_filter_error"])
    frame.attrs.update(source_df.attrs)
    return SourceDomainPolicyResult(frame=frame, diagnostics=diagnostics)
