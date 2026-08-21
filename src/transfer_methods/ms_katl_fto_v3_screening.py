"""FUSED_TARGET_ONLY_V3 screening contract (V2 recipe + RevIN).

Development / screening track. Same FUSED_TARGET_ONLY core as V1/V2
(fused init, trainable backbone, transfer/gate off) and the same V2 Stage-B
recipe (Huber + CosineDecay + EMA). The only architecture delta is RevIN
wrapped around the target CNN input / forecast output:

- each MinMax-scaled window is instance-normalized (mean/std over time)
- the 1-step forecast is denormalized with the sales-channel stats

Does not modify FORMAL_METHODS, MSML, No-TL, baselines, the v1/v2 screening
code paths, or the frozen v1/v2 selection manifests.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from src.transfer_methods.ms_katl_fto_v2_screening import (
    DEFAULT_V1_RUN_DIRNAME,
    DEFAULT_V2_RUN_DIRNAME,
    V2_TRAINING_RECIPE,
    v2_architecture,
    v2_architecture_digest,
    v2_training_recipe,
)
from src.transfer_methods.ms_katl_mechanism_ablation import validate_training_recipe
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    METHOD_DISPLAY as V1_METHOD_DISPLAY,
    ScreeningContractError,
    architecture_digest as v1_architecture_digest,
    assert_lr_arms_share_stage_a,
    fused_target_only_architecture,
    parse_lr,
)
from src.transfer_methods.revin import (
    DEFAULT_REVIN_AFFINE,
    DEFAULT_REVIN_EPS,
    DEFAULT_REVIN_TARGET_CHANNEL,
)

V3_METHOD_DISPLAY = "FUSED_TARGET_ONLY_V3"
V3_METHOD_ARM = "fused_target_only"
V3_TRACK = "development_screening_v3"
V3_SCREENING_LABEL = "MS_KATL_FTO_V3_SCREENING"

V3_TRAINING_RECIPE: Dict[str, Any] = dict(V2_TRAINING_RECIPE)

V3_REVIN_CONTRACT: Dict[str, Any] = {
    "revin_enabled": True,
    "revin_affine": bool(DEFAULT_REVIN_AFFINE),
    "revin_eps": float(DEFAULT_REVIN_EPS),
    "revin_target_channel": int(DEFAULT_REVIN_TARGET_CHANNEL),
}


def v3_training_recipe() -> Dict[str, Any]:
    recipe = validate_training_recipe(V3_TRAINING_RECIPE)
    if recipe != V3_TRAINING_RECIPE:
        raise ScreeningContractError(f"v3 training recipe drifted: {recipe}")
    if recipe != v2_training_recipe():
        raise ScreeningContractError("v3 recipe must stay identical to the frozen v2 recipe")
    return dict(recipe)


def v3_architecture() -> Dict[str, Any]:
    """FUSED_TARGET_ONLY core plus the frozen RevIN wrap."""
    core = fused_target_only_architecture()
    if core != v2_architecture():
        raise ScreeningContractError("v3 FUSED_TARGET_ONLY core drifted from v2")
    arch = dict(core)
    arch.update(V3_REVIN_CONTRACT)
    if not arch["revin_enabled"]:
        raise ScreeningContractError("v3 architecture must enable RevIN")
    return arch


def v3_architecture_digest() -> str:
    payload = json.dumps(
        {
            "method": V3_METHOD_DISPLAY,
            "arm": V3_METHOD_ARM,
            "architecture": v3_architecture(),
            "training_recipe": v3_training_recipe(),
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if digest in {v1_architecture_digest(), v2_architecture_digest()}:
        raise ScreeningContractError("v3 digest must differ from frozen v1 and v2 digests")
    return digest


def assert_v3_lr_arms_share_stage_a(rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    return assert_lr_arms_share_stage_a(rows, expected_method=V3_METHOD_DISPLAY)


def default_v1_all_runs_csv(project_root: Path) -> Path:
    return (
        Path(project_root)
        / "outputs"
        / "runs"
        / DEFAULT_V1_RUN_DIRNAME
        / "results"
        / "screening_all_runs.csv"
    )


def default_v2_all_runs_csv(project_root: Path) -> Path:
    return (
        Path(project_root)
        / "outputs"
        / "runs"
        / DEFAULT_V2_RUN_DIRNAME
        / "results"
        / "v2_screening_all_runs.csv"
    )


def _lookup_lr(value: object) -> float:
    try:
        return parse_lr(value)
    except ScreeningContractError:
        return float(value)


def _attach_named_deltas(
    row: Mapping[str, Any],
    index: Mapping[tuple, Mapping[str, float]],
    *,
    prefix: str,
) -> Dict[str, Any]:
    out = dict(row)
    key = (
        str(row.get("dataset", "")).strip(),
        str(row.get("target_identity", "")).strip(),
        _lookup_lr(row["learning_rate"]),
    )
    ref = index.get(key)
    rmse_key = f"{prefix}_rmse_original"
    smape_key = f"{prefix}_smape_original"
    d_rmse_key = f"delta_rmse_vs_{prefix}"
    d_smape_key = f"delta_smape_vs_{prefix}"
    beats_key = f"v3_beats_{prefix}_rmse"
    status_key = f"{prefix}_status"
    if ref is None:
        out[rmse_key] = None
        out[smape_key] = None
        out[d_rmse_key] = None
        out[d_smape_key] = None
        out[beats_key] = None
        out[status_key] = "MISSING"
        return out
    v3_rmse = row.get("rmse_original")
    v3_smape = row.get("smape_original")
    out[rmse_key] = float(ref["rmse_original"])
    out[smape_key] = float(ref["smape_original"])
    out[d_rmse_key] = (
        float(v3_rmse) - float(ref["rmse_original"]) if v3_rmse is not None else None
    )
    out[d_smape_key] = (
        float(v3_smape) - float(ref["smape_original"]) if v3_smape is not None else None
    )
    out[beats_key] = (
        bool(float(v3_rmse) < float(ref["rmse_original"])) if v3_rmse is not None else None
    )
    out[status_key] = "FOUND"
    return out


def attach_v1_deltas(
    row: Mapping[str, Any],
    v1_index: Mapping[tuple, Mapping[str, float]],
) -> Dict[str, Any]:
    return _attach_named_deltas(row, v1_index, prefix="v1")


def attach_v2_deltas(
    row: Mapping[str, Any],
    v2_index: Mapping[tuple, Mapping[str, float]],
) -> Dict[str, Any]:
    return _attach_named_deltas(row, v2_index, prefix="v2")


def _fmt_metric(value: object) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "MISSING"
    return f"{float(value):.4f}"


def _count_negative(rows: Sequence[Mapping[str, Any]], key: str) -> tuple:
    compared = 0
    wins = 0
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        compared += 1
        wins += int(float(value) < 0.0)
    return wins, compared


def format_v3_vs_frozen_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    """Per-run V3 vs frozen V2 / V1 table plus win counts."""
    lines = [
        "================================================================================",
        "FUSED_TARGET_ONLY_V3 vs frozen V2 / V1 (original RMSE / sMAPE)",
        "================================================================================",
        f"{'DATASET':<9}{'TARGET':<22}{'LR':<8}"
        f"{'V3_RMSE':<11}{'V2_RMSE':<11}{'dRMSE_v2':<11}"
        f"{'V3_sMAPE':<11}{'V2_sMAPE':<11}{'dSMAPE_v2':<11}",
        "--------------------------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{str(row.get('dataset', '')):<9}"
            f"{str(row.get('target_identity', '')):<22}"
            f"{str(row.get('learning_rate_label', '')):<8}"
            f"{_fmt_metric(row.get('rmse_original')):<11}"
            f"{_fmt_metric(row.get('v2_rmse_original')):<11}"
            f"{_fmt_metric(row.get('delta_rmse_vs_v2')):<11}"
            f"{_fmt_metric(row.get('smape_original')):<11}"
            f"{_fmt_metric(row.get('v2_smape_original')):<11}"
            f"{_fmt_metric(row.get('delta_smape_vs_v2')):<11}"
        )
    rmse_v2, n_v2 = _count_negative(rows, "delta_rmse_vs_v2")
    smape_v2, n_smape_v2 = _count_negative(rows, "delta_smape_vs_v2")
    rmse_v1, n_v1 = _count_negative(rows, "delta_rmse_vs_v1")
    smape_v1, n_smape_v1 = _count_negative(rows, "delta_smape_vs_v1")
    lines.extend(
        [
            "--------------------------------------------------------------------------------",
            f"V3_RMSE_WINS_VS_V2  = {rmse_v2}/{n_v2}",
            f"V3_SMAPE_WINS_VS_V2 = {smape_v2}/{n_smape_v2}",
            f"V3_RMSE_WINS_VS_V1  = {rmse_v1}/{n_v1}",
            f"V3_SMAPE_WINS_VS_V1 = {smape_v1}/{n_smape_v1}",
            "================================================================================",
        ]
    )
    return "\n".join(lines) + "\n"
