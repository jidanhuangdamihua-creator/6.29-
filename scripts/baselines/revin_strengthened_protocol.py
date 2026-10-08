"""Shared paper-scope protocol for RevIN-strengthened DLinear / TCN robustness."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.transfer_methods.revin_torch import REVIN_CONTRACT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_BASELINE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "robustness" / "revin_strengthened_baselines"
FORBIDDEN_TARGETS = {"166/311", "166/313"}
PAPER_TARGETS = {
    "D1": "1/10",
    "D4": "166/258",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}

PAPER_DATASETS = tuple(PAPER_TARGETS)
PAPER_HORIZONS = (1, 3, 5)
PAPER_SEEDS = (42, 43, 44, 45, 46)
PAPER_MODE = "without"
ROBUSTNESS_METHOD_IDS = ("BL6_DLinear_RevIN", "BL7_TCN_RevIN")
PAPER_METHODS = ("DLinear_RevIN", "TCN_RevIN")
PARENT_METHODS = {
    "BL6_DLinear_RevIN": "DLinear",
    "BL7_TCN_RevIN": "TCN",
    "DLinear_RevIN": "DLinear",
    "TCN_RevIN": "TCN",
}
METHOD_ID_TO_PAPER = {
    "BL6_DLinear_RevIN": "DLinear_RevIN",
    "BL7_TCN_RevIN": "TCN_RevIN",
}
PAPER_TO_METHOD_ID = {
    "DLinear_RevIN": "BL6_DLinear_RevIN",
    "TCN_RevIN": "BL7_TCN_RevIN",
}
CELL_COLUMNS = ["dataset", "target", "mode", "method", "horizon", "seed"]
EXPECTED_CELLS = (
    len(PAPER_DATASETS) * len(PAPER_HORIZONS) * len(PAPER_SEEDS) * len(PAPER_METHODS)
)
FORMAL_METHODS_MUST_NOT_MATCH = {"DLinear", "TCN", "BL6_DLinear", "BL7_TCN"}
DEFAULT_FREEZE = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/"
    "MS_KATL_Paper_Data_Freeze_v1_4_STATS_FROZEN_BL6_BL7_INTEGRATED.xlsx"
)


def expected_cell_keys() -> set[tuple[str, str, str, str, int, int]]:
    return {
        (dataset, PAPER_TARGETS[dataset], PAPER_MODE, method, horizon, seed)
        for dataset in PAPER_DATASETS
        for method in PAPER_METHODS
        for horizon in PAPER_HORIZONS
        for seed in PAPER_SEEDS
    }


def paper_method_name(method: str) -> str:
    text = str(method).strip()
    if text in PAPER_TO_METHOD_ID:
        return text
    if text in METHOD_ID_TO_PAPER:
        return METHOD_ID_TO_PAPER[text]
    raise ValueError(f"unsupported RevIN robustness method: {method!r}")


def method_id_name(method: str) -> str:
    text = str(method).strip()
    if text in METHOD_ID_TO_PAPER:
        return text
    if text in PAPER_TO_METHOD_ID:
        return PAPER_TO_METHOD_ID[text]
    raise ValueError(f"unsupported RevIN robustness method: {method!r}")


def assert_isolated_output_dir(output_dir: Path) -> Path:
    destination = Path(output_dir).resolve()
    formal = FORMAL_BASELINE_OUTPUT_DIR.resolve()
    if destination == formal or formal in destination.parents:
        raise ValueError(
            "RevIN robustness results must not be written under the formal "
            f"baseline directory {formal}"
        )
    freeze_markers = (
        "MS_KATL_Paper_Data_Freeze",
        "Section4_Comparison_Statistics",
        "BL6_BL7_FORMAL",
    )
    if any(marker.lower() in str(destination).lower() for marker in freeze_markers):
        raise ValueError(f"RevIN robustness results must not overwrite freeze paths: {destination}")
    return destination


def assert_identity_isolation(methods: Iterable[str]) -> None:
    observed = {str(method) for method in methods}
    overlap = observed.intersection(FORMAL_METHODS_MUST_NOT_MATCH)
    if overlap:
        raise ValueError(f"RevIN robustness methods collided with formal identities: {sorted(overlap)}")
    for method in observed:
        paper = paper_method_name(method)
        if paper in {"DLinear", "TCN"}:
            raise ValueError(f"RevIN robustness method collapsed to a formal name: {method!r}")


def filter_paper_entities(entity_slices: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for data in entity_slices:
        dataset = str(data["dataset_id"]).upper()
        target = str(data["entity_key"])
        if target in FORBIDDEN_TARGETS:
            continue
        if dataset in PAPER_TARGETS and target == PAPER_TARGETS[dataset]:
            selected.append(dict(data))
    if len(selected) != 1:
        identities = [(str(item["dataset_id"]), str(item["entity_key"])) for item in entity_slices]
        raise ValueError(
            "paper-scope filter must retain exactly one formal target per dataset, "
            f"got {selected!r} from {identities!r}"
        )
    return selected


def annotate_robustness_rows(
    frame: pd.DataFrame,
    *,
    prediction_file: str = "",
    git_head: str = "",
    timestamp: str = "",
) -> pd.DataFrame:
    output = frame.copy()
    output["method_id"] = output["method"].map(method_id_name)
    output["method"] = output["method"].map(paper_method_name)
    output["parent_method"] = output["method"].map(PARENT_METHODS)
    output["robustness_variant"] = "RevIN"
    output["dataset"] = output["dataset_id"].astype(str).str.upper()
    output["target"] = output["target_entity_key"].astype(str)
    output["mode"] = output["scenario"].astype(str)
    output["revin_enabled"] = True
    output["revin_affine"] = REVIN_CONTRACT["affine"]
    output["revin_eps"] = REVIN_CONTRACT["eps"]
    output["revin_target_channel"] = REVIN_CONTRACT["target_channel"]
    output["revin_configuration"] = json.dumps(REVIN_CONTRACT, sort_keys=True)
    output["formal_baseline_config_identity"] = output["parent_method"].map(
        lambda parent: "scripts/baselines/bl6_dlinear.py" if parent == "DLinear" else "scripts/baselines/bl7_tcn.py"
    )
    output["code_commit"] = git_head or "unavailable"
    output["timestamp"] = timestamp or datetime.now(timezone.utc).isoformat()
    if prediction_file:
        output["prediction_file_path"] = str(prediction_file)
    output["n_test_observations"] = output["sample_count"].astype(int)
    output["result_status"] = "robustness_trial"
    if (output["metric_space"].astype(str) != "original_sales_space").any():
        raise ValueError("RevIN robustness metrics must stay in original_sales_space")
    if set(output["mode"].astype(str)) != {PAPER_MODE}:
        raise ValueError("RevIN robustness must inherit formal without-sharing mode")
    if set(output["target"].astype(str)).intersection(FORBIDDEN_TARGETS):
        raise ValueError("RevIN robustness included a retired D4 target")
    assert_identity_isolation(output["method"])
    return output


def cell_completeness(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("robustness result frame is empty")
    keys = list(
        zip(
            frame["dataset"].astype(str).str.upper(),
            frame["target"].astype(str),
            frame["mode"].astype(str),
            frame["method"].astype(str),
            frame["horizon"].astype(int),
            frame["seed"].astype(int),
        )
    )
    counts = pd.Series(keys).value_counts()
    observed = set(keys)
    expected = expected_cell_keys()
    duplicate = int((counts > 1).sum())
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    return {
        "EXPECTED_CELLS": int(len(expected)),
        "ACTUAL_CELLS": int(len(observed)),
        "DUPLICATE_CELLS": duplicate,
        "MISSING_CELLS": int(len(missing)),
        "EXTRA_CELLS": int(len(extra)),
        "missing_keys": missing,
        "extra_keys": extra,
        "complete": len(observed) == len(expected) and duplicate == 0 and not extra,
    }


def aggregate_five_seed(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["method", "dataset", "target", "horizon", "mode"], dropna=False, sort=True)
    rows = []
    for identity, group in grouped:
        method, dataset, target, horizon, mode = identity
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "target": target,
                "horizon": int(horizon),
                "mode": mode,
                "rmse_mean": float(group["rmse"].mean()),
                "rmse_std": float(group["rmse"].std(ddof=1)) if len(group) > 1 else 0.0,
                "smape_mean": float(group["smape"].mean()),
                "smape_std": float(group["smape"].std(ddof=1)) if len(group) > 1 else 0.0,
                "n_seeds": int(len(group)),
                "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"].astype(int).unique())),
            }
        )
    aggregated = pd.DataFrame(rows)
    if aggregated.empty:
        raise ValueError("five-seed aggregation produced no rows")
    if (aggregated["n_seeds"] != len(PAPER_SEEDS)).any():
        raise ValueError("five-seed aggregation requires all formal seeds; best-seed reporting is forbidden")
    return aggregated


def git_head(root: Path = PROJECT_ROOT) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_error_change(before: float, after: float) -> float:
    baseline = float(before)
    if baseline == 0.0:
        raise ValueError("relative change is undefined for a zero baseline error")
    return (float(after) - baseline) / baseline * 100.0


def relative_improvement(baseline_error: float, sita_error: float) -> float:
    reference = float(baseline_error)
    if reference <= 0.0:
        raise ValueError("relative_improvement requires a positive baseline error")
    return (reference - float(sita_error)) / reference * 100.0
