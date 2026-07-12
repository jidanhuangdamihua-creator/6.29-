from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.protocols.experiment_protocol import ProtocolViolation, get_experiment_protocol
from src.utils.source_domain_filter import (
    apply_source_domain_policy,
    domain_filter_mask,
    normalize_domain_filter,
)


def load_default_metric_protocol(project_root: Path) -> Dict[str, Any]:
    config_path = project_root / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    metric_protocol = dict(cfg.get("paper_reproduction", {}).get("metric_protocol", {}))
    metric_protocol["strict_paper_metrics"] = bool(metric_protocol.get("strict_paper_metrics", False))
    return metric_protocol


def apply_runtime_source_domain_policy(
    source_df: pd.DataFrame,
    knn_data: Dict[str, Any],
    config: Dict[str, Any],
) -> pd.DataFrame:
    dataset_id = config.get("dataset_id", knn_data.get("dataset_id"))
    protocol = get_experiment_protocol(dataset_id) if dataset_id is not None else None
    scenario = str(config.get("info_sharing", "without"))
    if protocol is not None and protocol.dataset_id == "D4":
        source_pool_policy = (
            "without_information_sharing_same_store"
            if scenario.strip().lower() == "without"
            else "with_information_sharing_cross_store"
        )
    else:
        source_pool_policy = None
    result = apply_source_domain_policy(
        source_df,
        knn_data.get("domain_filter"),
        information_sharing=scenario,
        entity_group_cols=knn_data.get("group_cols"),
        domain_filter_scope=(
            protocol.source_pool_rule.domain_filter_scope
            if protocol is not None
            else "source_pool"
        ),
        source_pool_policy=source_pool_policy,
    )
    config.update(result.diagnostics)
    return result.frame


def validate_runtime_target_domain(
    target_df: pd.DataFrame,
    target_entity_keys: list[str],
    knn_data: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Validate JSON-authoritative targets when the protocol scopes the filter to targets."""
    protocol = get_experiment_protocol(
        config.get("dataset_id", knn_data.get("dataset_id"))
    )
    scope = protocol.source_pool_rule.domain_filter_scope
    if scope not in {"target_only", "target_and_source"}:
        return

    normalized_filter = normalize_domain_filter(knn_data.get("domain_filter"))
    if not normalized_filter:
        raise ProtocolViolation(
            f"{protocol.dataset_id} target-only domain filter is missing from KNN metadata"
        )
    entity_col = str(config.get("entity_col", "entity_id"))
    if entity_col not in target_df.columns:
        raise ProtocolViolation(
            f"{protocol.dataset_id} target domain validation requires {entity_col!r}"
        )
    selected_keys = {str(key) for key in target_entity_keys}
    selected = target_df[target_df[entity_col].astype(str).isin(selected_keys)].copy()
    found_keys = set(selected[entity_col].astype(str))
    missing_keys = sorted(selected_keys - found_keys)
    if missing_keys:
        raise ProtocolViolation(
            f"{protocol.dataset_id} target domain validation is missing JSON targets: {missing_keys!r}"
        )
    try:
        valid_mask = domain_filter_mask(selected, normalized_filter)
    except ValueError as exc:
        raise ProtocolViolation(
            f"{protocol.dataset_id} target domain validation configuration error: {exc}"
        ) from exc
    invalid_keys = sorted(set(selected.loc[~valid_mask, entity_col].astype(str)))
    if invalid_keys:
        raise ProtocolViolation(
            f"{protocol.dataset_id} target domain validation failed for JSON targets: {invalid_keys!r}"
        )
    config.update(
        {
            "domain_filter_scope": scope,
            "target_domain_validation_passed": True,
            "target_domain_validation_target_count": len(selected_keys),
        }
    )
