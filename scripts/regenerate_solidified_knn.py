from __future__ import annotations

import argparse
import copy
from datetime import date, datetime
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import SOLIDIFIED_KNN_ROOT, SOURCE_HISTORY_DAYS
from src.data_processing.data_preprocessing import infer_source_selection_feature_columns
from src.source_selection.source_selector import SourceSelector
from src.utils.parquet_data_loader import (
    attach_window_attrs,
    load_parquet_source_target,
    read_dataset_windows,
)


def _file_digest(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_knn_config_files(root: str | Path) -> Dict[str, Dict[str, Any]]:
    """Capture md5 and mtime for solidified KNN JSON files."""
    base = Path(root)
    snapshot: Dict[str, Dict[str, Any]] = {}
    for path in sorted(base.glob("Dataset*/knn_*_info_sharing.json")):
        stat = path.stat()
        snapshot[str(path.relative_to(base))] = {
            "md5": _file_digest(path),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return snapshot


def verify_knn_config_unchanged(root: str | Path, before: Dict[str, Dict[str, Any]]) -> None:
    """Raise if check-only mode changed solidified KNN configs."""
    after = snapshot_knn_config_files(root)
    if after != before:
        raise AssertionError(
            "check-only mode modified configs/solidified/knn files: "
            f"before={before} after={after}"
        )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_json_sort_key(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert pandas/numpy/path values to strict JSON-safe values."""
    if isinstance(obj, dict):
        out: Dict[Any, Any] = {}
        for key, value in obj.items():
            safe_key = _to_jsonable(key)
            if not isinstance(safe_key, (str, int, float, bool, type(None))):
                safe_key = str(key)
            out[safe_key] = _to_jsonable(value)
        return out

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(item) for item in obj]

    if isinstance(obj, (set, frozenset)):
        values = [_to_jsonable(item) for item in obj]
        return sorted(values, key=_stable_json_sort_key)

    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return obj.isoformat()

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj

    try:
        missing = pd.isna(obj)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None

    return str(obj)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _to_jsonable(payload)
    path.write_text(
        json.dumps(safe_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )




def _scenario_file(dataset_id: int, scenario: str, root: Path) -> Path:
    return root / f"Dataset{int(dataset_id)}" / f"knn_{scenario}_info_sharing.json"


def _source_entity_from_key(source_key: Sequence[Any]) -> str:
    return "_".join(str(part) for part in source_key)


def _result_row(source: Dict[str, Any], group_cols: Sequence[str]) -> Dict[str, Any]:
    key = tuple(source["source_key"]) if isinstance(source["source_key"], (list, tuple)) else (source["source_key"],)
    row: Dict[str, Any] = {
        "source_entity": _source_entity_from_key(key),
        "distance": float(source["distance"]),
        "weight": float(source["weight"]),
    }
    for idx, col in enumerate(group_cols):
        if idx < len(key):
            row[f"source_{col}"] = key[idx]
    return row


def _top_sources(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(row.get("source_entity", "")) for row in rows]


def _distances(rows: Sequence[Dict[str, Any]]) -> List[float]:
    return [float(row.get("distance", 0.0)) for row in rows]


def _distance_delta(old: Sequence[float], new: Sequence[float]) -> List[float]:
    return [float(n - o) for o, n in zip(old, new)]


def _filter_source_for_scenario(
    source_df: pd.DataFrame,
    *,
    dataset_id: int,
    scenario: str,
    old_payload: Dict[str, Any],
) -> pd.DataFrame:
    # Existing D4 without-information JSON uses a reduced same-domain source pool;
    # D5/D6 solidified files currently record identical with/without pool sizes.
    if int(dataset_id) == 4 and scenario == "without":
        domain_filter = old_payload.get("domain_filter", {})
        column = domain_filter.get("column")
        value = domain_filter.get("value")
        if column in source_df.columns:
            return source_df[source_df[column] == value].copy()
    return source_df.copy()


def regenerate_dataset_scenario(
    *,
    dataset_id: int,
    scenario: str,
    knn_root: Path,
    output_root: Path,
    write: bool,
) -> Dict[str, Any]:
    path = _scenario_file(dataset_id, scenario, knn_root)
    old_payload = _load_json(path)
    windows = read_dataset_windows(dataset_id, knn_root / f"Dataset{int(dataset_id)}")
    source_df, target_df = load_parquet_source_target(
        dataset_id=dataset_id,
        parquet_dir="数据集/固化数据",
        windows=windows,
        source_history_days=SOURCE_HISTORY_DAYS,
    )
    source_df = attach_window_attrs(source_df, windows, role="source")
    target_df = attach_window_attrs(target_df, windows, role="target")
    source_df = _filter_source_for_scenario(
        source_df,
        dataset_id=dataset_id,
        scenario=scenario,
        old_payload=old_payload,
    )

    feature_info = infer_source_selection_feature_columns(source_df, target_df)
    feature_cols = list(feature_info["selected_features"])
    group_cols = tuple(old_payload.get("group_cols", ["entity_id", "item_id"]))
    k = int(old_payload.get("k", 3))
    selector = SourceSelector()

    new_results: Dict[str, List[Dict[str, Any]]] = {}
    diff_rows: List[Dict[str, Any]] = []
    for target_entity_id, old_rows in old_payload.get("results", {}).items():
        target_entity_df = target_df[target_df["entity_id"].astype(str) == str(target_entity_id)].copy()
        if target_entity_df.empty:
            new_rows: List[Dict[str, Any]] = []
        else:
            selected = selector.select_top_k_sources(
                target_df=target_entity_df,
                source_df=source_df,
                feature_cols=feature_cols,
                k=k,
                group_cols=group_cols,
            )
            new_rows = [_result_row(row, group_cols) for row in selected.get("sources", [])]
        new_results[str(target_entity_id)] = new_rows

        old_features = list((old_payload.get("feature_info", {}) or {}).get("selected_features", old_payload.get("feature_cols", [])))
        new_features = list(feature_cols)
        old_top = _top_sources(old_rows)
        new_top = _top_sources(new_rows)
        old_dist = _distances(old_rows)
        new_dist = _distances(new_rows)
        diff_rows.append(
            {
                "dataset_id": int(dataset_id),
                "information_sharing": scenario,
                "target_entity_id": str(target_entity_id),
                "old_selected_features": old_features,
                "new_selected_features": new_features,
                "feature_added": [f for f in new_features if f not in set(old_features)],
                "feature_removed": [f for f in old_features if f not in set(new_features)],
                "old_top_k_sources": old_top,
                "new_top_k_sources": new_top,
                "source_changed": old_top != new_top,
                "old_distances": old_dist,
                "new_distances": new_dist,
                "distance_delta": _distance_delta(old_dist, new_dist),
            }
        )

    new_payload = copy.deepcopy(old_payload)
    new_payload["feature_cols"] = feature_cols
    new_payload["feature_info"] = feature_info
    new_payload["source_pool_size"] = int(len(source_df))
    new_payload["results"] = new_results

    generated_path = output_root / "generated_json" / f"Dataset{int(dataset_id)}" / path.name
    _write_json(generated_path, new_payload)
    if write:
        print(f"[WRITE MODE] overwriting {path}")
        _write_json(path, new_payload)

    return {
        "dataset_id": int(dataset_id),
        "information_sharing": scenario,
        "json_path": str(path),
        "generated_path": str(generated_path),
        "diff_rows": diff_rows,
    }


def _summary_markdown(records: Sequence[Dict[str, Any]]) -> str:
    lines = ["# KNN Diff Summary", ""]
    grouped: Dict[tuple[int, str], List[Dict[str, Any]]] = {}
    for record in records:
        for row in record["diff_rows"]:
            grouped.setdefault((int(row["dataset_id"]), str(row["information_sharing"])), []).append(row)
    for (dataset_id, scenario), rows in sorted(grouped.items()):
        changed = sum(1 for row in rows if row["source_changed"])
        lines.append(f"- D{dataset_id} {scenario}: changed_entities = {changed} / {len(rows)}")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate/check solidified D4-D6 KNN JSON without overwriting by default.")
    parser.add_argument("--datasets", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--diff-out", type=Path, default=Path("outputs/feature_consistency"))
    parser.add_argument("--write", action="store_true", help="Overwrite configs/solidified/knn JSON files.")
    parser.add_argument("--knn-root", type=Path, default=SOLIDIFIED_KNN_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = args.diff_out if args.diff_out.is_absolute() else PROJECT_ROOT / args.diff_out
    output_root.mkdir(parents=True, exist_ok=True)
    before = snapshot_knn_config_files(args.knn_root)

    records: List[Dict[str, Any]] = []
    for dataset_id in args.datasets:
        for scenario in ("without", "with"):
            records.append(
                regenerate_dataset_scenario(
                    dataset_id=int(dataset_id),
                    scenario=scenario,
                    knn_root=args.knn_root,
                    output_root=output_root,
                    write=bool(args.write),
                )
            )

    all_diff_rows = [row for record in records for row in record["diff_rows"]]
    summary = {
        "write": bool(args.write),
        "records": records,
        "diff_rows": all_diff_rows,
    }
    _write_json(output_root / "knn_diff_summary.json", summary)
    (output_root / "knn_diff_summary.md").write_text(_summary_markdown(records), encoding="utf-8")

    if not args.write:
        verify_knn_config_unchanged(args.knn_root, before)


if __name__ == "__main__":
    main()
