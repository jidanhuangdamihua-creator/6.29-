"""Central dataset numbering and alias registry.

This module keeps dataset numbering, display names, legacy aliases, and
default paths in one place so preprocessing and runners stay consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "数据集"
SOLIDIFIED_DATA_DIR = DATA_DIR / "固化数据"


DATASET_REGISTRY: Dict[str, Dict[str, object]] = {
    "Dataset1": {
        "display_name": "需求预测挑战赛",
        "profile": "challenge",
        "preferred_alias": "demand-forecasting",
        "default_path": SOLIDIFIED_DATA_DIR / "dataset1-source.parquet",
        "aliases": {
            "dataset1",
            "dataset 1",
            "demand-forecasting",
            "demand_forecasting",
            "challenge",
            "demand forecasting challenge",
            "需求预测挑战赛",
        },
    },
    "Dataset2": {
        "display_name": "意大利面需求",
        "profile": "pasta",
        "preferred_alias": "italian-pasta-demand",
        "default_path": SOLIDIFIED_DATA_DIR / "dataset2-source.parquet",
        "aliases": {
            "dataset2",
            "dataset 2",
            "italian-pasta-demand",
            "pasta",
            "pasta-demand",
            "italian pasta demand",
            "hierarchical",
            "hierarchical-sales",
            "hierarchical_sales",
            "意大利面需求",
        },
    },
    "Dataset3": {
        "display_name": "Rossmann 门店",
        "profile": "rossmann",
        "preferred_alias": "rossmann-store-sales",
        "default_path": SOLIDIFIED_DATA_DIR / "dataset3-source.parquet",
        "aliases": {
            "dataset3",
            "dataset 3",
            "rossmann",
            "rossmann-store-sales",
            "rossmann store sales",
            "rossmann-store",
            "rossmann 门店",
        },
    },
}


def list_dataset_names() -> List[str]:
    return ["Dataset1", "Dataset2", "Dataset3"]


def normalize_dataset_name(name: str) -> str:
    text = str(name).strip()
    if text in DATASET_REGISTRY:
        return text

    lowered = text.lower()
    for canonical_name, meta in DATASET_REGISTRY.items():
        aliases = meta.get("aliases", set())
        if lowered in aliases:
            return canonical_name

    raise ValueError(
        "Unsupported dataset name. Use Dataset1/2/3 or a configured alias. "
        f"Received: {name}"
    )


def get_dataset_profile(name: str) -> str:
    canonical_name = normalize_dataset_name(name)
    return str(DATASET_REGISTRY[canonical_name]["profile"])


def get_dataset_display_name(name: str) -> str:
    canonical_name = normalize_dataset_name(name)
    return str(DATASET_REGISTRY[canonical_name]["display_name"])


def get_default_dataset_path(name: str) -> str:
    canonical_name = normalize_dataset_name(name)
    return str(DATASET_REGISTRY[canonical_name]["default_path"])


def get_dataset_path_map() -> Dict[str, str]:
    return {
        canonical_name: str(meta["default_path"])
        for canonical_name, meta in DATASET_REGISTRY.items()
    }
