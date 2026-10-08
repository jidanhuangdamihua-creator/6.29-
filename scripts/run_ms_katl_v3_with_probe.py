"""Run exactly four frozen-LR V3 cells with information sharing enabled."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tf_compat  # noqa: F401

import numpy as np
import pandas as pd

from scripts.run_ms_katl_fto_three_track_robustness import (
    _failed_cell_payload,
    _run_arm,
    _run_stage_a,
)
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    RunIdentity,
    classify_cell_artifact,
    robustness_arm_registry,
    write_cell_artifact,
)
from src.transfer_methods.ms_katl_v3_frame_assembly import assemble_v3_cell_frames
from src.transfer_methods.ms_katl_v3_with_probe import (
    EXPECTED_PROBE_CELLS,
    FROZEN_WITHOUT_RESULTS_RELPATH,
    PROBE_MODE,
    build_probe_identities,
    build_probe_manifest,
    format_probe_preflight,
    paired_comparison_row,
    probe_cell_path,
    probe_mode_root,
    probe_run_root,
    source_authority_identity,
    summarize_comparisons,
    write_probe_manifest,
)
from src.utils.d4_d6_runtime import load_default_metric_protocol

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Mode-neutral probe run root; artifacts are isolated below mode=with.",
    )
    return parser.parse_args()


def _without_cell_path(identity: RunIdentity) -> Path:
    target_slug = identity.target_identity.replace("/", "-")
    filename = (
        f"v3_{identity.dataset}_{target_slug}_h{identity.horizon}"
        f"_s{identity.seed}_{identity.selected_lr_label}.json"
    )
    return PROJECT_ROOT / FROZEN_WITHOUT_RESULTS_RELPATH / "cells" / filename


def _load_without_cell(identity: RunIdentity) -> Dict[str, Any]:
    path = _without_cell_path(identity)
    if not path.is_file():
        raise RuntimeError(f"matched frozen V3-without cell missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("status")) != "OK" or str(payload.get("mode")) != "without":
        raise RuntimeError(f"matched frozen V3-without cell is not accepted: {path}")
    return payload


def _sample_alignment(without: Mapping[str, Any], with_row: Mapping[str, Any]) -> Dict[str, str]:
    without_path = Path(str(without.get("prediction_artifact_path") or ""))
    with_path = Path(str(with_row.get("prediction_artifact_path") or ""))
    if not without_path.is_file() or not with_path.is_file():
        return {
            "SAMPLE_COUNT_MATCH": "UNAVAILABLE",
            "TEST_TIMESTAMP_MATCH": "UNAVAILABLE",
            "GROUND_TRUTH_MATCH": "UNAVAILABLE",
        }
    old = pd.read_csv(without_path)
    new = pd.read_csv(with_path)
    sample_match = len(old) == len(new)
    truth_columns = ("y_true_normalized", "y_true_original")
    truth_available = all(column in old and column in new for column in truth_columns)
    truth_match = truth_available and sample_match and all(
        np.array_equal(old[column].to_numpy(), new[column].to_numpy())
        for column in truth_columns
    )
    timestamp_columns = [
        column
        for column in ("timestamp", "date", "test_timestamp")
        if column in old and column in new
    ]
    timestamp_match = (
        "UNAVAILABLE"
        if not timestamp_columns
        else (
            "YES"
            if sample_match
            and all(old[column].equals(new[column]) for column in timestamp_columns)
            else "NO"
        )
    )
    return {
        "SAMPLE_COUNT_MATCH": "YES" if sample_match else "NO",
        "TEST_TIMESTAMP_MATCH": timestamp_match,
        "GROUND_TRUTH_MATCH": (
            "UNAVAILABLE" if not truth_available else ("YES" if truth_match else "NO")
        ),
    }


def _write_run_metadata(run_root: Path, manifest: Mapping[str, Any]) -> None:
    mode_root = probe_mode_root(run_root)
    mode_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        "method": "FUSED_TARGET_ONLY_V3",
        "mode": PROBE_MODE,
        "source_mode": PROBE_MODE,
        "fused_init_source_mode": PROBE_MODE,
        "expected_cells": EXPECTED_PROBE_CELLS,
        "horizons": [1],
        "seeds": [42],
        "lr_screening": False,
        "full_matrix": False,
        "v3_architecture_changed": False,
        "v3_training_recipe_changed": False,
    }
    (mode_root / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    write_probe_manifest(mode_root / "v3_with_probe_manifest.json", manifest)


def _write_results(
    run_root: Path,
    with_rows: list[Mapping[str, Any]],
    identities: tuple[RunIdentity, ...],
) -> Dict[str, Any]:
    if len(with_rows) != EXPECTED_PROBE_CELLS:
        raise RuntimeError(f"completed with cells must equal 4, got {len(with_rows)}")
    comparisons = []
    alignments: Dict[str, Any] = {}
    for identity, with_row in zip(identities, with_rows):
        without = _load_without_cell(identity)
        comparisons.append(paired_comparison_row(without, with_row))
        alignments[identity.dataset] = _sample_alignment(without, with_row)
    summary = summarize_comparisons(comparisons)
    results_dir = probe_mode_root(run_root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(with_rows).to_csv(
        results_dir / "v3_with_probe_all_runs.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(comparisons).to_csv(
        results_dir / "v3_with_probe_paired_comparison.csv", index=False, encoding="utf-8"
    )
    payload = {"mode": PROBE_MODE, "summary": summary, "sample_alignment": alignments}
    (results_dir / "v3_with_probe_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return {"comparisons": comparisons, "summary": summary, "alignments": alignments}


def main() -> None:
    args = _parse_args()
    identities = build_probe_identities(PROJECT_ROOT)
    run_root = probe_run_root(PROJECT_ROOT, args.output_dir)
    print(format_probe_preflight(project_root=PROJECT_ROOT, run_root=run_root, identities=identities), end="")
    if args.preflight:
        return

    frozen_without = (PROJECT_ROOT / FROZEN_WITHOUT_RESULTS_RELPATH).resolve()
    if run_root.resolve() == frozen_without or frozen_without in run_root.resolve().parents:
        raise RuntimeError("refusing to write inside frozen V3-without results")
    if probe_mode_root(run_root).name != "mode=with":
        raise RuntimeError("with output root must be explicitly mode=with")

    manifest = build_probe_manifest(
        project_root=PROJECT_ROOT,
        run_root=run_root,
        identities=identities,
        status="RUNNING",
        completed_cells=0,
    )
    _write_run_metadata(run_root, manifest)

    registry = robustness_arm_registry()
    v3_spec = registry["v3"]
    metric_protocol = load_default_metric_protocol(PROJECT_ROOT)
    completed: list[Mapping[str, Any]] = []
    for identity in identities:
        artifact_path = probe_cell_path(run_root, identity)
        status = classify_cell_artifact(artifact_path, identity, v3_spec)
        if status == "OK":
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            completed.append(payload)
            print(f"SKIP_OK {identity.cell_id}")
            continue
        print(
            f"RUN_CELL {identity.dataset}|{identity.target_identity}|{identity.mode}"
            f"|H{identity.horizon}|seed{identity.seed}|LR={identity.selected_lr_label}"
        )
        try:
            cell = assemble_v3_cell_frames(
                identity.dataset,
                identity.target_identity,
                horizon=identity.horizon,
                seed=identity.seed,
                project_root=PROJECT_ROOT,
                mode=identity.mode,
            )
            if cell.get("mode") != PROBE_MODE or cell.get("scenario") != "with_information_sharing":
                raise RuntimeError(f"with frame routing guard failed: {cell.get('mode')}, {cell.get('scenario')}")
            print(f"SOURCE_MODE={cell['mode']}")
            print(f"SOURCE_AUTHORITY={cell['source_authority_identity']}")
            stage_a, bundle = _run_stage_a(
                cell=cell,
                horizon=identity.horizon,
                seed=identity.seed,
                epochs=50,
                batch_size=16,
            )
            print("FUSED_INIT_SOURCE_MODE=with")
            payload = _run_arm(
                identity=identity,
                spec=v3_spec,
                stage_a=stage_a,
                bundle=bundle,
                epochs=50,
                batch_size=16,
                metric_protocol=metric_protocol,
                pred_root=run_root / "predictions",
                parsed=cell["parsed"],
            )
            payload.update(
                {
                    "mode": PROBE_MODE,
                    "source_mode": PROBE_MODE,
                    "fused_init_source_mode": PROBE_MODE,
                    "source_models_rebuilt": True,
                    "without_source_params_reused": False,
                    "source_authority_identity": source_authority_identity(PROJECT_ROOT, identity),
                    "source_count": len(stage_a.source_identities),
                    "output_artifact_path": str(artifact_path.resolve()),
                    "sample_count": int(len(pd.read_csv(payload["prediction_artifact_path"]))),
                    "inverse_transform_identity": {
                        "target_scaler_digest": payload.get("target_scaler_digest"),
                        "target_feature_columns": list(bundle.target_feature_columns),
                    },
                }
            )
            write_cell_artifact(artifact_path, payload)
            completed.append(payload)
            print(
                f"CELL_OK {identity.cell_id} RMSE={payload['rmse_original']:.12g}"
                f" sMAPE={payload['smape_original']:.12g}"
            )
        except Exception as exc:
            failed = _failed_cell_payload(identity, v3_spec, f"{type(exc).__name__}: {exc}")
            failed.update(
                {
                    "mode": PROBE_MODE,
                    "source_mode": PROBE_MODE,
                    "fused_init_source_mode": PROBE_MODE,
                    "source_authority_identity": source_authority_identity(PROJECT_ROOT, identity),
                    "output_artifact_path": str(artifact_path.resolve()),
                }
            )
            write_cell_artifact(artifact_path, failed)
            manifest = build_probe_manifest(
                project_root=PROJECT_ROOT,
                run_root=run_root,
                identities=identities,
                status="FAILED",
                completed_cells=len(completed),
            )
            manifest["failure"] = failed["failure_reason"]
            _write_run_metadata(run_root, manifest)
            raise
        manifest = build_probe_manifest(
            project_root=PROJECT_ROOT,
            run_root=run_root,
            identities=identities,
            status="RUNNING",
            completed_cells=len(completed),
        )
        _write_run_metadata(run_root, manifest)

    results = _write_results(run_root, completed, identities)
    manifest = build_probe_manifest(
        project_root=PROJECT_ROOT,
        run_root=run_root,
        identities=identities,
        status="COMPLETE",
        completed_cells=len(completed),
    )
    manifest.update(results)
    _write_run_metadata(run_root, manifest)
    print(json.dumps(results["summary"], indent=2))
    print(f"PROBE_COMPLETED_CELLS={len(completed)}")
    print(f"RUN_ROOT={run_root.resolve()}")


if __name__ == "__main__":
    main()
