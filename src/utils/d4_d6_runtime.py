from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.utils.source_domain_filter import apply_source_domain_policy


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
    result = apply_source_domain_policy(
        source_df,
        knn_data.get("domain_filter"),
        information_sharing=str(config.get("info_sharing", "without")),
        entity_group_cols=knn_data.get("group_cols"),
    )
    config.update(result.diagnostics)
    return result.frame
