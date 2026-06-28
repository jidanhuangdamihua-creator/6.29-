"""Add the solidified Dataset3 region column required by KNN filtering."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from pandas.api.types import is_integer_dtype


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "数据集" / "固化数据" / "dataset3-source.parquet"
TARGET_PATH = ROOT / "数据集" / "固化数据" / "dataset3-target.parquet"
KNN_PATH = ROOT / "outputs" / "knn_selection" / "Dataset3" / "knn_without_info_sharing.json"
BACKUP_DIR = ROOT / "outputs" / "protection" / "backups"
SOURCE_BACKUP_PATH = BACKUP_DIR / "dataset3-source.before_region_fix.parquet"
TARGET_BACKUP_PATH = BACKUP_DIR / "dataset3-target.before_region_fix.parquet"
SOURCE_TMP_PATH = SOURCE_PATH.with_name(SOURCE_PATH.name + ".tmp")
TARGET_TMP_PATH = TARGET_PATH.with_name(TARGET_PATH.name + ".tmp")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sorted_int_unique(series: pd.Series) -> list[int]:
    return sorted(int(value) for value in series.unique().tolist())


def load_knn_domain_filter(path: Path = KNN_PATH) -> Dict[str, Any]:
    _require(path.exists(), f"KNN JSON missing: {path.relative_to(ROOT)}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    domain_filter = payload.get("domain_filter")
    _require(isinstance(domain_filter, dict), "KNN JSON domain_filter missing")
    return domain_filter


def derive_region(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["region"] = ((out["entity_id"].astype("int64") - 1) // 10) + 1
    out["region"] = out["region"].astype("int64")
    return out


def assert_region_frame(df: pd.DataFrame, label: str) -> None:
    _require("region" in df.columns, f"{label} region column missing")
    _require(df["region"].notna().all(), f"{label} region column has nulls")
    _require(
        is_integer_dtype(df["region"]),
        f"{label} region dtype unexpected: {df['region'].dtype}",
    )
    region_values = _sorted_int_unique(df["region"])
    _require(
        set(region_values).issubset({1, 2, 3}),
        f"{label} unexpected region values: {region_values}",
    )


def assert_d3_region_semantics(source_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
    source_region1_entities = _sorted_int_unique(
        source_df.loc[source_df["region"] == 1, "entity_id"]
    )
    target_entities = _sorted_int_unique(target_df["entity_id"])
    source_entities = _sorted_int_unique(source_df["entity_id"])

    _require(
        target_entities == [10],
        f"unexpected D3 target entity_id values: {target_entities}",
    )
    _require(10 not in source_entities, "target entity_id=10 leaked into source parquet")
    _require(
        source_region1_entities == [1, 2, 3, 4, 5, 6, 7, 8, 9],
        f"unexpected D3 Region 1 source entities: {source_region1_entities}",
    )


def assert_phase1_inputs(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    domain_filter: Dict[str, Any],
) -> None:
    _require(SOURCE_PATH.exists(), f"D3 source parquet missing: {SOURCE_PATH.relative_to(ROOT)}")
    _require(TARGET_PATH.exists(), f"D3 target parquet missing: {TARGET_PATH.relative_to(ROOT)}")
    _require("entity_id" in source_df.columns, "source entity_id column missing")
    _require("entity_id" in target_df.columns, "target entity_id column missing")
    _require("region" not in source_df.columns, "source already has region column")
    _require("region" not in target_df.columns, "target already has region column")
    _require(
        domain_filter.get("column") == "region",
        f"unexpected KNN domain_filter.column: {domain_filter.get('column')}",
    )
    _require(
        domain_filter.get("value") == 1,
        f"unexpected KNN domain_filter.value: {domain_filter.get('value')}",
    )

    derived_source = derive_region(source_df)
    derived_target = derive_region(target_df)
    assert_d3_region_semantics(derived_source, derived_target)


def _assert_tmp_paths_absent() -> None:
    _require(not SOURCE_TMP_PATH.exists(), f"tmp path already exists: {SOURCE_TMP_PATH}")
    _require(not TARGET_TMP_PATH.exists(), f"tmp path already exists: {TARGET_TMP_PATH}")


def _assert_backup_paths_absent() -> None:
    _require(
        not SOURCE_BACKUP_PATH.exists(),
        f"backup path already exists: {SOURCE_BACKUP_PATH.relative_to(ROOT)}",
    )
    _require(
        not TARGET_BACKUP_PATH.exists(),
        f"backup path already exists: {TARGET_BACKUP_PATH.relative_to(ROOT)}",
    )


def _assert_all_region_checks(source_df: pd.DataFrame, target_df: pd.DataFrame) -> None:
    assert_region_frame(source_df, "source")
    assert_region_frame(target_df, "target")
    assert_d3_region_semantics(source_df, target_df)


def main() -> None:
    source_df = pd.read_parquet(SOURCE_PATH)
    target_df = pd.read_parquet(TARGET_PATH)
    domain_filter = load_knn_domain_filter()

    assert_phase1_inputs(source_df, target_df, domain_filter)
    _assert_tmp_paths_absent()
    _assert_backup_paths_absent()

    source_with_region = derive_region(source_df)
    target_with_region = derive_region(target_df)
    _assert_all_region_checks(source_with_region, target_with_region)

    source_with_region.to_parquet(SOURCE_TMP_PATH, index=False)
    target_with_region.to_parquet(TARGET_TMP_PATH, index=False)

    tmp_source_df = pd.read_parquet(SOURCE_TMP_PATH)
    tmp_target_df = pd.read_parquet(TARGET_TMP_PATH)
    _assert_all_region_checks(tmp_source_df, tmp_target_df)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_PATH, SOURCE_BACKUP_PATH)
    shutil.copy2(TARGET_PATH, TARGET_BACKUP_PATH)

    os.replace(SOURCE_TMP_PATH, SOURCE_PATH)
    os.replace(TARGET_TMP_PATH, TARGET_PATH)

    print(f"Backed up source parquet to {SOURCE_BACKUP_PATH.relative_to(ROOT)}")
    print(f"Backed up target parquet to {TARGET_BACKUP_PATH.relative_to(ROOT)}")
    print("D3 source/target parquet region fix completed.")


if __name__ == "__main__":
    main()
