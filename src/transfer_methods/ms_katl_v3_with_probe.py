"""Contracts and publishers for the bounded V3-with four-cell probe."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from src.protocols.formal_input_paths import resolve_formal_dataset_paths
from src.transfer_methods.ms_katl_fto_three_track_robustness import (
    RunIdentity,
    RobustnessContractError,
    cell_path,
    load_v3_frozen_robustness_manifest,
)
from src.transfer_methods.ms_katl_primary_target_lr_screening import format_lr, parse_lr
from src.transfer_methods.ms_katl_v3_frame_assembly import (
    v3_knn_authority_path,
    v3_scenario_for_mode,
)

PROBE_MODE = "with"
PROBE_HORIZON = 1
PROBE_SEED = 42
PROBE_TARGETS: Tuple[Tuple[str, str], ...] = (
    ("D1", "1/10"),
    ("D4", "166/311"),
    ("D5", "48/1159414"),
    ("D6", "CA_1/FOODS_3_080"),
)
EXPECTED_PROBE_CELLS = 4
DEFAULT_PROBE_RUN_RELPATH = (
    Path("outputs") / "runs" / "20260823_MS_KATL_V3_WITH_4_CELL_PROBE"
)
FROZEN_WITHOUT_RESULTS_RELPATH = (
    Path("outputs")
    / "runs"
    / "20260818_190337_640037_MS_KATL_FTO_THREE_TRACK_ROBUSTNESS"
)


def build_probe_identities(project_root: Path) -> Tuple[RunIdentity, ...]:
    frozen = load_v3_frozen_robustness_manifest(Path(project_root))
    identities = []
    for dataset, target in PROBE_TARGETS:
        row = dict((frozen.get("datasets") or {}).get(dataset) or {})
        if str(row.get("primary_target") or "") != target:
            raise RobustnessContractError(
                f"frozen V3 target mismatch for {dataset}: {row.get('primary_target')!r} != {target!r}"
            )
        lr = parse_lr(row.get("learning_rate"))
        identities.append(
            RunIdentity(
                arm="v3",
                dataset=dataset,
                target_identity=target,
                horizon=PROBE_HORIZON,
                seed=PROBE_SEED,
                selected_lr=lr,
                selected_lr_label=format_lr(lr),
                mode=PROBE_MODE,
            )
        )
    assert_probe_contract(identities, project_root=project_root)
    return tuple(identities)


def probe_run_root(project_root: Path, requested: Path | None = None) -> Path:
    root = DEFAULT_PROBE_RUN_RELPATH if requested is None else Path(requested)
    if not root.is_absolute():
        root = Path(project_root) / root
    return root


def probe_mode_root(run_root: Path) -> Path:
    return Path(run_root) / f"mode={PROBE_MODE}"


def probe_cell_path(run_root: Path, identity: RunIdentity) -> Path:
    return cell_path(Path(run_root) / "cells", identity)


def source_authority_identity(project_root: Path, identity: RunIdentity) -> str:
    if identity.mode != PROBE_MODE:
        raise RobustnessContractError("probe source authority is defined only for mode=with")
    if identity.dataset in {"D1", "D2", "D3"}:
        paths = resolve_formal_dataset_paths(
            int(identity.dataset[1:]), repository_root=Path(project_root)
        )
        return (
            f"formal_lifecycle:{v3_scenario_for_mode(identity.mode)}"
            f"|sealed_source:{Path(paths.source_path).resolve()}"
        )
    authority = v3_knn_authority_path(identity.dataset, identity.mode).resolve()
    if not authority.is_file():
        raise RobustnessContractError(f"missing with KNN authority: {authority}")
    return str(authority)


def assert_probe_contract(
    identities: Sequence[RunIdentity],
    *,
    project_root: Path,
    run_root: Path | None = None,
) -> None:
    if len(identities) != EXPECTED_PROBE_CELLS:
        raise RobustnessContractError(
            f"probe must contain exactly {EXPECTED_PROBE_CELLS} cells, got {len(identities)}"
        )
    if len({item.identity_tuple for item in identities}) != EXPECTED_PROBE_CELLS:
        raise RobustnessContractError("probe identities are not unique")
    observed = tuple((item.dataset, item.target_identity) for item in identities)
    if observed != PROBE_TARGETS:
        raise RobustnessContractError(f"probe target drift: {observed!r}")
    for identity in identities:
        if (
            identity.arm != "v3"
            or identity.mode != PROBE_MODE
            or identity.horizon != PROBE_HORIZON
            or identity.seed != PROBE_SEED
        ):
            raise RobustnessContractError(f"out-of-contract probe identity: {identity!r}")
        without = RunIdentity(
            arm=identity.arm,
            dataset=identity.dataset,
            target_identity=identity.target_identity,
            horizon=identity.horizon,
            seed=identity.seed,
            selected_lr=identity.selected_lr,
            selected_lr_label=identity.selected_lr_label,
            mode="without",
        )
        if identity.identity_tuple == without.identity_tuple:
            raise RobustnessContractError("with/without identity collision")
        cells_root = probe_run_root(project_root, run_root) / "cells"
        if cell_path(cells_root, identity) == cell_path(cells_root, without):
            raise RobustnessContractError("with/without output collision")


def build_probe_manifest(
    *,
    project_root: Path,
    run_root: Path,
    identities: Sequence[RunIdentity],
    status: str = "PLANNED",
    completed_cells: int = 0,
) -> Dict[str, Any]:
    assert_probe_contract(identities, project_root=project_root, run_root=run_root)
    return {
        "method": "FUSED_TARGET_ONLY_V3",
        "mode": PROBE_MODE,
        "source_mode": PROBE_MODE,
        "fused_init_source_mode": PROBE_MODE,
        "status": str(status),
        "expected_probe_cells": EXPECTED_PROBE_CELLS,
        "completed_probe_cells": int(completed_cells),
        "horizons": [PROBE_HORIZON],
        "seeds": [PROBE_SEED],
        "lr_screening": False,
        "full_matrix": False,
        "v3_architecture_changed": False,
        "v3_training_recipe_changed": False,
        "run_root": str(Path(run_root).resolve()),
        "mode_root": str(probe_mode_root(run_root).resolve()),
        "frozen_v3_manifest": str(
            (Path(project_root) / "configs" / "ms_katl_fto_v3_robustness_manifest.json").resolve()
        ),
        "frozen_without_results_root": str(
            (Path(project_root) / FROZEN_WITHOUT_RESULTS_RELPATH).resolve()
        ),
        "cells": [
            {
                "identity": list(identity.identity_tuple),
                "dataset": identity.dataset,
                "target": identity.target_identity,
                "mode": identity.mode,
                "horizon": identity.horizon,
                "seed": identity.seed,
                "selected_lr": identity.selected_lr,
                "selected_lr_label": identity.selected_lr_label,
                "source_authority": source_authority_identity(project_root, identity),
                "output_path": str(probe_cell_path(run_root, identity).resolve()),
            }
            for identity in identities
        ],
    }


def write_probe_manifest(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def format_probe_preflight(
    *, project_root: Path, run_root: Path, identities: Sequence[RunIdentity]
) -> str:
    assert_probe_contract(identities, project_root=project_root, run_root=run_root)
    frozen_root = (Path(project_root) / FROZEN_WITHOUT_RESULTS_RELPATH).resolve()
    with_root = probe_mode_root(run_root).resolve()
    if with_root == frozen_root or frozen_root in with_root.parents:
        raise RobustnessContractError("with output root collides with frozen without root")
    lines = [
        "==================================================",
        "V3-WITH 4-CELL PROBE PREFLIGHT",
        "==================================================",
        "",
        "MODE=with",
        "",
    ]
    for index, identity in enumerate(identities, start=1):
        lines.append(
            f"CELL_{index}={identity.dataset}|{identity.target_identity}|with|H1|seed42"
        )
    lines.extend(
        [
            "",
            "EXPECTED_CELLS=4",
            f"ACTUAL_CELLS={len(identities)}",
            f"UNIQUE_CELLS={len({item.identity_tuple for item in identities})}",
            "",
            "HORIZONS=1",
            "SEEDS=42",
            "MODES=with",
            "",
            "LR_SCREENING=NO",
            "FULL_MATRIX=NO",
            f"WITH_OUTPUT_ROOT={with_root}",
            f"WITHOUT_OUTPUT_ROOT={frozen_root}",
            "OUTPUT_ROOT_COLLISION=NO",
            "",
        ]
    )
    for identity in identities:
        lines.extend(
            [
                f"[{identity.dataset}]",
                f"dataset={identity.dataset}",
                f"target={identity.target_identity}",
                f"mode={identity.mode}",
                f"H={identity.horizon}",
                f"seed={identity.seed}",
                f"selected_lr={identity.selected_lr_label}",
                f"source_authority={source_authority_identity(project_root, identity)}",
                f"output_path={probe_cell_path(run_root, identity).resolve()}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def paired_comparison_row(without: Mapping[str, Any], with_row: Mapping[str, Any]) -> Dict[str, Any]:
    identity_fields = ("dataset", "target_identity", "horizon", "seed")
    for key in identity_fields:
        if str(without.get(key)) != str(with_row.get(key)):
            raise RobustnessContractError(f"paired identity mismatch on {key}")
    if str(without.get("mode")) != "without" or str(with_row.get("mode")) != "with":
        raise RobustnessContractError("paired comparison requires without vs with")
    if format_lr(parse_lr(without.get("learning_rate_label") or without.get("learning_rate"))) != format_lr(
        parse_lr(with_row.get("learning_rate_label") or with_row.get("learning_rate"))
    ):
        raise RobustnessContractError("paired LR mismatch")
    if str(without.get("architecture_digest")) != str(with_row.get("architecture_digest")):
        raise RobustnessContractError("paired V3 architecture mismatch")
    rmse_without = float(without["rmse_original"])
    rmse_with = float(with_row["rmse_original"])
    smape_without = float(without["smape_original"])
    smape_with = float(with_row["smape_original"])
    return {
        "dataset": str(with_row["dataset"]),
        "target": str(with_row["target_identity"]),
        "learning_rate": format_lr(with_row.get("learning_rate_label") or with_row["learning_rate"]),
        "rmse_without": rmse_without,
        "rmse_with": rmse_with,
        "rmse_gain_percent": (rmse_without - rmse_with) / rmse_without * 100.0,
        "smape_without": smape_without,
        "smape_with": smape_with,
        "smape_gain_percent": (smape_without - smape_with) / smape_without * 100.0,
    }


def summarize_comparisons(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(rows) != EXPECTED_PROBE_CELLS:
        raise RobustnessContractError("paired summary requires exactly four rows")
    rmse = [float(row["rmse_gain_percent"]) for row in rows]
    smape = [float(row["smape_gain_percent"]) for row in rows]
    rmse_wins = sum(value > 0.0 for value in rmse)
    smape_wins = sum(value > 0.0 for value in smape)
    if rmse_wins >= 3 and any(value > 0.1 for value in rmse):
        recommendation = "12_CELL_HORIZON_PROBE"
    elif rmse_wins <= 1:
        recommendation = "KEEP_WITHOUT_AS_PRIMARY"
    else:
        recommendation = "PAUSE_AND_REVIEW"
    return {
        "WITH_RMSE_WINS": rmse_wins,
        "WITH_SMAPE_WINS": smape_wins,
        "MEAN_RMSE_GAIN_PERCENT": statistics.fmean(rmse),
        "MEDIAN_RMSE_GAIN_PERCENT": statistics.median(rmse),
        "MEAN_SMAPE_GAIN_PERCENT": statistics.fmean(smape),
        "MEDIAN_SMAPE_GAIN_PERCENT": statistics.median(smape),
        "RECOMMEND_NEXT": recommendation,
    }
