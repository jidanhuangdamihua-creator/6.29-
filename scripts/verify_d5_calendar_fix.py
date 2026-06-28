#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.parquet_data_loader import (  # noqa: E402
    attach_window_attrs,
    load_knn_results,
    read_dataset_windows,
)


EXPECTED_SPLIT_CONFIG = {
    "mode": "days",
    "train_days": 15,
    "val_days": 15,
    "test_days": 181,
}


def main() -> None:
    knn_json_dir = ROOT / "outputs" / "knn_selection" / "Dataset5"
    windows = read_dataset_windows(5, knn_json_dir)
    knn_data = load_knn_results(knn_json_dir, "without")
    target_entity_keys = list(knn_data["results"].keys())

    assert len(target_entity_keys) == 5, (
        f"expected exactly 5 D5 target entity keys, got {len(target_entity_keys)}: "
        f"{target_entity_keys}"
    )

    target_path = ROOT / "数据集" / "固化数据" / "dataset5-target.parquet"
    target_df = pd.read_parquet(target_path)
    target_df = attach_window_attrs(target_df, windows, role="target")

    entity_checks = []
    for entity_key in target_entity_keys:
        entity_df = target_df[target_df["entity_id"].astype(str) == str(entity_key)].copy()
        unique_date_count = int(entity_df["date"].nunique())
        split_config = entity_df.attrs.get("split_config")
        print(
            f"[D5 calendar verify] entity={entity_key} "
            f"unique_dates={unique_date_count} split_config={split_config}"
        )
        entity_checks.append((entity_key, entity_df.empty, unique_date_count, split_config))

    for entity_key, is_empty, unique_date_count, split_config in entity_checks:
        assert not is_empty, f"D5 target entity missing after calendar attach: {entity_key}"
        assert unique_date_count == 211, (
            f"D5 target entity {entity_key} has {unique_date_count} unique dates, expected 211"
        )
        assert split_config == EXPECTED_SPLIT_CONFIG, (
            f"D5 target entity {entity_key} split_config={split_config}, "
            f"expected {EXPECTED_SPLIT_CONFIG}"
        )

    print("[D5 calendar verify] PASS: all 5 target entities have complete 211-day calendars.")


if __name__ == "__main__":
    main()
