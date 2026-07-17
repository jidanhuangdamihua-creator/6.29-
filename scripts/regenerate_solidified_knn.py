from __future__ import annotations

import argparse
import copy
from datetime import date, datetime
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (
    D4_D6_RUNTIME_KNN_PROTOCOL_VERSION,
    SOLIDIFIED_KNN_ROOT,
    SOURCE_HISTORY_DAYS,
)
from src.protocols.experiment_protocol import PROTOCOL_VERSION, get_experiment_protocol
from src.protocols.runner_adapter import configure_protocol_frames
from src.protocols.formal_input_paths import resolve_formal_dataset_paths
from src.source_selection.source_selector import SourceSelector
from src.utils.d4_d6_runtime import (
    apply_runtime_source_domain_policy,
    validate_runtime_target_domain,
)
from src.utils.parquet_data_loader import (
    load_parquet_source_target,
    read_dataset_windows,
)
from src.utils.source_domain_filter import SourceDomainPolicyResult, apply_source_domain_policy


def _file_digest(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        _to_jsonable(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
) -> SourceDomainPolicyResult:
    if int(dataset_id) not in {4, 5, 6}:
        raise ValueError(f"regeneration only supports D4-D6: dataset_id={dataset_id}")
    if int(dataset_id) == 4:
        runtime_config: Dict[str, Any] = {
            "dataset_id": 4,
            "info_sharing": str(scenario),
            "entity_col": "entity_id",
        }
        frame = apply_runtime_source_domain_policy(source_df, old_payload, runtime_config)
        diagnostics = {
            key: value
            for key, value in runtime_config.items()
            if key not in {"dataset_id", "info_sharing", "entity_col"}
        }
        diagnostics["source_pool_entity_count"] = diagnostics.get(
            "source_pool_entities_after_filter"
        )
        return SourceDomainPolicyResult(frame=frame, diagnostics=diagnostics)
    return apply_source_domain_policy(
        source_df,
        old_payload.get("domain_filter"),
        information_sharing=scenario,
        entity_group_cols=old_payload.get("group_cols"),
    )


def _select_d4_shared_protocol(
    *,
    source_df: pd.DataFrame,
    target_entity_df: pd.DataFrame,
    scenario: str,
    feature_cols: Sequence[str],
    k: int,
    group_cols: Sequence[str],
) -> Dict[str, Any]:
    """Select a Dataset4 target through the formal shared-protocol path."""
    observed_start = target_entity_df.attrs.get(
        "knn_observed_start",
        target_entity_df.attrs.get(
            "target_observed_start",
            pd.to_datetime(target_entity_df["date"], errors="raise").min(),
        ),
    )
    configured_source, configured_target = configure_protocol_frames(
        source_df,
        target_entity_df,
        dataset_id=4,
        scenario=scenario,
        group_cols=group_cols,
        grouping_col=None,
        observed_start=observed_start,
    )
    selected = SourceSelector().select_top_k_sources(
        target_df=configured_target,
        source_df=configured_source,
        feature_cols=feature_cols,
        k=k,
        group_cols=tuple(group_cols),
    )
    metadata = selected.get("meta", {})
    if not isinstance(metadata, dict) or metadata.get("selection_path") != "shared_protocol":
        raise ValueError("Dataset4 regeneration did not use the shared protocol selector path")
    return selected


def _prepare_d4_runtime_source_pool(
    *,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_entity_keys: Sequence[str],
    scenario: str,
    old_payload: Mapping[str, Any],
) -> SourceDomainPolicyResult:
    """Apply the formal D4 source policy and validate JSON-selected targets."""
    runtime_config: Dict[str, Any] = {
        "dataset_id": 4,
        "info_sharing": str(scenario),
        "entity_col": "entity_id",
    }
    configured_source = apply_runtime_source_domain_policy(
        source_df,
        dict(old_payload),
        runtime_config,
    )
    validate_runtime_target_domain(
        target_df,
        [str(key) for key in target_entity_keys],
        dict(old_payload),
        runtime_config,
    )
    diagnostics = {
        key: value
        for key, value in runtime_config.items()
        if key not in {"dataset_id", "info_sharing", "entity_col"}
    }
    diagnostics["source_pool_entity_count"] = diagnostics.get(
        "source_pool_entities_after_filter"
    )
    return SourceDomainPolicyResult(frame=configured_source, diagnostics=diagnostics)


def _build_regenerated_payload(
    *,
    old_payload: Dict[str, Any],
    feature_cols: Sequence[str],
    feature_info: Dict[str, Any],
    source_pool_size: int,
    results: Dict[str, List[Dict[str, Any]]],
    selection_metadata: Dict[str, Dict[str, Any]],
    source_domain_policy_diagnostics: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a regenerated payload without inheriting stale selection metadata."""
    for field in ("k", "group_cols"):
        if field not in old_payload:
            raise ValueError(f"KNN payload missing required protocol field: {field}")
    if not feature_cols:
        raise ValueError("KNN payload requires non-empty feature_cols")
    for target_key, metadata in selection_metadata.items():
        authority = metadata.get("selection_authority")
        version = metadata.get("protocol_version")
        valid_runtime_metadata = (
            authority == "runtime"
            and version == D4_D6_RUNTIME_KNN_PROTOCOL_VERSION
        )
        valid_shared_metadata = (
            authority == "shared_protocol"
            and version == PROTOCOL_VERSION
        )
        if not (valid_runtime_metadata or valid_shared_metadata):
            raise ValueError(
                f"selection_metadata[{target_key!r}] has unsupported selection authority/version"
            )

    stable_keys = (
        "dataset_id",
        "dataset",
        "info_sharing",
        "k",
        "window_size",
        "horizon",
        "target_train_window",
        "domain_filter",
        "group_cols",
    )
    new_payload = {
        key: copy.deepcopy(old_payload[key])
        for key in stable_keys
        if key in old_payload
    }
    new_payload.update(
        {
            "selection_authority": "runtime",
            "protocol_version": D4_D6_RUNTIME_KNN_PROTOCOL_VERSION,
            "results_semantics": "json_top_k_diagnostic_not_training_authority",
            "training_selection_authority": "runtime_source_selector",
            "json_results_used_for": [
                "target_list",
                "smoke_or_source_limit_candidate_pool",
                "diagnostics",
            ],
            "feature_cols": list(feature_cols),
            "feature_info": copy.deepcopy(feature_info),
            "source_pool_size": int(source_pool_size),
            "source_domain_policy_diagnostics": copy.deepcopy(
                dict(source_domain_policy_diagnostics or {})
            ),
            "results": copy.deepcopy(results),
            "selection_metadata": copy.deepcopy(selection_metadata),
        }
    )
    return new_payload


def _d4_exact_key_validation_proof(
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    digest_input = dict(metadata["candidate_pool_digest_input"])
    target_key = tuple(str(part) for part in digest_input["target_key"])
    candidate_keys = {
        tuple(str(part) for part in key)
        for key in digest_input["candidate_keys"]
    }
    target_store, target_product = target_key
    cross_store_same_product_count = sum(
        key[0] != target_store and key[1] == target_product
        for key in candidate_keys
    )
    proof: Dict[str, Any] = {
        "entity_key_fields": ["store_id", "product_id"],
        "exact_target_tuple": list(target_key),
        "exact_target_tuple_excluded": target_key not in candidate_keys,
        "cross_store_same_product_available_count": cross_store_same_product_count,
        "cross_store_same_product_retained_count": cross_store_same_product_count,
        "same_store_other_product_retained_count": sum(
            key[0] == target_store and key[1] != target_product
            for key in candidate_keys
        ),
        "cross_store_other_product_retained_count": sum(
            key[0] != target_store and key[1] != target_product
            for key in candidate_keys
        ),
        "candidate_digest_verified": True,
        "consumer_fingerprint_verified": True,
    }
    if not proof["exact_target_tuple_excluded"]:
        raise ValueError(f"D4 exact target entered candidate pool: {target_key!r}")
    return proof


def _d4_manifest_identity(
    *,
    scenario: str,
    selection_metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    formal_paths = resolve_formal_dataset_paths(4, repository_root=PROJECT_ROOT)
    source_path = formal_paths.source_path
    target_path = formal_paths.target_path
    identity: Dict[str, Any] = {
        "identity_version": "d4_exact_composite_key_v1",
        "dataset_id": "D4",
        "scenario": str(scenario),
        "entity_key_fields": ["store_id", "product_id"],
        "source_parquet_sha256": _sha256_file(source_path),
        "target_parquet_sha256": _sha256_file(target_path),
        "target_consumers": {
            str(target): {
                "candidate_pool_digest": metadata["candidate_pool_digest"],
                "selection_result_digest": metadata["selection_result_digest"],
                "source_pool_fingerprint": metadata["source_pool_fingerprint"],
                "consumer_fingerprint": metadata["consumer_fingerprint"],
                "validation_proof_digest": metadata["d4_validation_proof_digest"],
            }
            for target, metadata in sorted(selection_metadata.items())
        },
    }
    identity["manifest_identity_digest"] = _canonical_sha256(identity)
    return identity


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
    if int(old_payload.get("dataset_id", -1)) != int(dataset_id):
        raise ValueError(f"KNN payload dataset_id mismatch: expected={dataset_id}")
    if str(old_payload.get("info_sharing", "")).strip().lower() != str(scenario).strip().lower():
        raise ValueError(f"KNN payload info_sharing mismatch: expected={scenario}")
    windows = read_dataset_windows(
        dataset_id,
        knn_root / f"Dataset{int(dataset_id)}",
        info_sharing=scenario,
        knn_payload=old_payload,
    )
    formal_paths = resolve_formal_dataset_paths(
        dataset_id,
        repository_root=PROJECT_ROOT,
    )
    source_df, target_df = load_parquet_source_target(
        dataset_id=dataset_id,
        source_path=formal_paths.source_path,
        target_path=formal_paths.target_path,
        windows=windows,
        source_history_days=SOURCE_HISTORY_DAYS,
    )
    if int(dataset_id) == 4:
        source_domain_policy = _prepare_d4_runtime_source_pool(
            source_df=source_df,
            target_df=target_df,
            target_entity_keys=[str(key) for key in old_payload.get("results", {})],
            scenario=scenario,
            old_payload=old_payload,
        )
    else:
        source_domain_policy = _filter_source_for_scenario(
            source_df,
            dataset_id=dataset_id,
            scenario=scenario,
            old_payload=old_payload,
        )
    source_df = source_domain_policy.frame

    existing_feature_info = old_payload.get("feature_info", {})
    feature_cols = list(
        existing_feature_info.get("selected_features", [])
        if isinstance(existing_feature_info, dict)
        else []
    ) or list(old_payload.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("KNN payload missing required feature_cols")
    missing_features = [
        col for col in feature_cols if col not in source_df.columns or col not in target_df.columns
    ]
    if missing_features:
        raise ValueError(f"KNN payload feature_cols missing from runtime frames: {missing_features}")
    feature_info = copy.deepcopy(existing_feature_info) if isinstance(existing_feature_info, dict) else {}
    feature_info["selected_features"] = list(feature_cols)
    if "group_cols" not in old_payload:
        raise ValueError("KNN payload missing required protocol field: group_cols")
    group_cols = tuple(old_payload["group_cols"])
    if "k" not in old_payload or int(old_payload["k"]) <= 0:
        raise ValueError("KNN payload requires positive k")
    k = int(old_payload["k"])
    selector = SourceSelector()

    new_results: Dict[str, List[Dict[str, Any]]] = {}
    new_selection_metadata: Dict[str, Dict[str, Any]] = {}
    diff_rows: List[Dict[str, Any]] = []
    for target_entity_id, old_rows in old_payload.get("results", {}).items():
        target_entity_df = target_df[target_df["entity_id"].astype(str) == str(target_entity_id)].copy()
        if target_entity_df.empty:
            raise ValueError(f"KNN target entity missing from runtime target frame: {target_entity_id}")
        if int(dataset_id) == 4:
            selected = _select_d4_shared_protocol(
                source_df=source_df,
                target_entity_df=target_entity_df,
                scenario=scenario,
                feature_cols=feature_cols,
                k=k,
                group_cols=group_cols,
            )
        else:
            selected = selector.select_top_k_sources(
                target_df=target_entity_df,
                source_df=source_df,
                feature_cols=feature_cols,
                k=k,
                group_cols=group_cols,
            )
        new_rows = [_result_row(row, group_cols) for row in selected.get("sources", [])]
        selected_meta = selected.get("meta", {})
        if not isinstance(selected_meta, dict):
            raise ValueError(f"Runtime selector returned invalid metadata for target={target_entity_id}")
        if int(dataset_id) == 4:
            rule = get_experiment_protocol(4).source_pool_rule
            selected_meta = {
                **selected_meta,
                "source_pool_policy": source_domain_policy.diagnostics.get(
                    "source_pool_policy", ""
                ),
                "domain_filter_scope": source_domain_policy.diagnostics.get(
                    "domain_filter_scope", ""
                ),
                "domain_filter_applied_to_source": source_domain_policy.diagnostics.get(
                    "domain_filter_applied_to_source", False
                ),
                "source_pool_entity_count": source_domain_policy.diagnostics.get(
                    "source_pool_entity_count"
                ),
                "require_same_group": rule.require_same_group,
                "excluded_candidate_key_fields": [
                    group_cols[position]
                    for position in rule.candidate_exclusion_positions()
                ],
                "target_domain_filter": copy.deepcopy(old_payload.get("domain_filter")),
            }
            d4_validation_proof = _d4_exact_key_validation_proof(selected_meta)
            selected_meta["d4_validation_proof"] = d4_validation_proof
            selected_meta["d4_validation_proof_digest"] = _canonical_sha256(
                d4_validation_proof
            )
        new_selection_metadata[str(target_entity_id)] = copy.deepcopy(selected_meta)
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

    new_payload = _build_regenerated_payload(
        old_payload=old_payload,
        feature_cols=feature_cols,
        feature_info=feature_info,
        source_pool_size=int(len(source_df)),
        source_domain_policy_diagnostics=source_domain_policy.diagnostics,
        results=new_results,
        selection_metadata=new_selection_metadata,
    )
    if int(dataset_id) == 4:
        new_payload["d4_manifest_identity"] = _d4_manifest_identity(
            scenario=scenario,
            selection_metadata=new_selection_metadata,
        )

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
