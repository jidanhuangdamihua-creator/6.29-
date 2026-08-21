"""FUSED_TARGET_ONLY_V2 screening contract (Huber + CosineDecay + EMA).

Development / screening track. The architecture is IDENTICAL to the frozen
FUSED_TARGET_ONLY arm (fused init, trainable backbone, transfer branch off,
gate zero). Only the Stage-B training recipe differs:

- loss:        MSE      -> Huber(delta=1.0)   (normalized space)
- lr schedule: constant -> CosineDecay(lr -> 0 over total steps)
- weights:     last     -> EMA(momentum=0.99) applied at end of training

Does not modify FORMAL_METHODS, MSML, No-TL, baselines, the v1 screening
code path, or the frozen v1 selection manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.transfer_methods.ms_katl_mechanism_ablation import validate_training_recipe
from src.transfer_methods.ms_katl_primary_target_lr_screening import (
    LR_GRID,
    METHOD_DISPLAY as V1_METHOD_DISPLAY,
    RMSE_TIE_RELATIVE,
    ScreeningContractError,
    architecture_digest as v1_architecture_digest,
    assert_lr_arms_share_stage_a,
    format_lr,
    fused_target_only_architecture,
    parse_lr,
)

V2_METHOD_DISPLAY = "FUSED_TARGET_ONLY_V2"
V2_METHOD_ARM = "fused_target_only"
V2_TRACK = "development_screening_v2"
V2_SCREENING_LABEL = "MS_KATL_FTO_V2_SCREENING"
V2_EXTRA_LR_LABEL = "MS_KATL_FTO_V2_EXTRA_LR"
V2_CANONICAL_BOUNDARY_LR = LR_GRID[-1]

V2_TRAINING_RECIPE: Dict[str, Any] = {
    "loss_name": "huber",
    "huber_delta": 1.0,
    "lr_schedule": "cosine",
    "use_ema": True,
    "ema_momentum": 0.99,
}

DEFAULT_V1_RUN_DIRNAME = "20260818_155005_113008_MS_KATL_PRIMARY_TARGET_LR_SCREENING"
DEFAULT_V2_RUN_DIRNAME = "20260818_163754_196799_MS_KATL_FTO_V2_SCREENING"


def v2_training_recipe() -> Dict[str, Any]:
    """Validated copy of the frozen v2 recipe."""
    recipe = validate_training_recipe(V2_TRAINING_RECIPE)
    if recipe != V2_TRAINING_RECIPE:
        raise ScreeningContractError(f"v2 training recipe drifted: {recipe}")
    return dict(recipe)


def v2_architecture() -> Dict[str, Any]:
    """V2 must keep the exact FUSED_TARGET_ONLY architecture contract."""
    return fused_target_only_architecture()


def v2_architecture_digest() -> str:
    payload = json.dumps(
        {
            "method": V2_METHOD_DISPLAY,
            "arm": V2_METHOD_ARM,
            "architecture": v2_architecture(),
            "training_recipe": v2_training_recipe(),
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if digest == v1_architecture_digest():
        raise ScreeningContractError("v2 digest must differ from the frozen v1 digest")
    return digest


def assert_v2_lr_arms_share_stage_a(rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    return assert_lr_arms_share_stage_a(rows, expected_method=V2_METHOD_DISPLAY)


def format_exploratory_lr(value: object) -> str:
    """Compact scientific label for LRs outside the canonical grid (e.g. 0.001 -> 1e-3)."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ScreeningContractError(f"exploratory lr must be positive and finite, got {value!r}")
    exponent = int(math.floor(math.log10(number)))
    mantissa = number / (10 ** exponent)
    mantissa_i = int(round(mantissa))
    if math.isclose(mantissa, float(mantissa_i), rel_tol=0.0, abs_tol=1e-8):
        return f"{mantissa_i}e{exponent}"
    text = f"{number:.0e}".replace("e+0", "e+").replace("e-0", "e-")
    return text.replace("e+", "e")


def parse_exploratory_lr(value: object) -> float:
    """Parse an extra LR that is required to sit outside canonical LR_GRID."""
    if isinstance(value, str):
        text = value.strip().lower().replace(" ", "")
        if not text:
            raise ScreeningContractError("extra lr is empty")
        number = float(text)
    else:
        number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ScreeningContractError(f"extra lr must be a positive finite value, got {value!r}")
    for canonical in LR_GRID:
        if math.isclose(number, canonical, rel_tol=0.0, abs_tol=1e-18):
            raise ScreeningContractError(
                f"extra lr {value!r} collides with canonical LR_GRID={list(LR_GRID)}"
            )
    return number


def resolve_extra_lrs(spec: object) -> Tuple[float, ...]:
    """Parse --extra-lr. Empty/None means canonical LR_GRID is used unchanged."""
    if spec is None:
        return ()
    if isinstance(spec, (list, tuple)):
        parts = [str(item).strip() for item in spec if str(item).strip()]
    else:
        parts = [
            part.strip()
            for part in str(spec).replace(";", ",").split(",")
            if part.strip()
        ]
    if not parts:
        raise ScreeningContractError("--extra-lr requires at least one learning rate")
    resolved: list[float] = []
    for part in parts:
        lr = parse_exploratory_lr(part)
        if any(math.isclose(lr, prev, rel_tol=0.0, abs_tol=1e-18) for prev in resolved):
            raise ScreeningContractError(f"duplicate extra lr {part!r}")
        resolved.append(lr)
    return tuple(resolved)


def assert_v2_exploratory_arms_share_stage_a(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    """Stage-A sharing for extra-lr arms. Does not require canonical LR_GRID membership."""
    if not rows:
        raise ScreeningContractError("extra-lr arms are empty")
    identities = {str(row.get("target_identity")) for row in rows}
    if len(identities) != 1:
        raise ScreeningContractError(f"extra-lr arms mix targets: {sorted(identities)}")
    methods = {str(row.get("method")) for row in rows}
    if methods != {V2_METHOD_DISPLAY}:
        raise ScreeningContractError(
            f"extra-lr arms must all be {V2_METHOD_DISPLAY}, got {methods}"
        )

    def _flag(key: str) -> str:
        values = [row.get(key) for row in rows]
        return "YES" if all(value == values[0] for value in values) else "NO"

    flags = {
        "SOURCE_IDENTITIES_MATCH": _flag("source_identities"),
        "SOURCE_DISTANCES_MATCH": _flag("source_distances"),
        "SOURCE_MODEL_DIGESTS_MATCH": _flag("source_model_digests"),
        "SOURCE_SCALER_DIGESTS_MATCH": _flag("source_scaler_digests"),
        "TARGET_SPLIT_MATCH": _flag("target_split_digest"),
        "TARGET_SCALER_MATCH": _flag("target_scaler_digest"),
        "LR_ARMS_FUSED_INIT_MATCH": _flag("fused_init_digest"),
        "ARCHITECTURE_MATCH": _flag("architecture_digest"),
    }
    if any(value != "YES" for value in flags.values()):
        raise ScreeningContractError(
            f"shared Stage-A / fused-init contract failed for extra-lr: {flags}"
        )
    return flags


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


def _index_ok_runs_by_target_lr(
    csv_path: Optional[Path],
    expected_method: str,
) -> Dict[tuple, Dict[str, float]]:
    if csv_path is None or not Path(csv_path).exists():
        return {}
    import csv as _csv

    index: Dict[tuple, Dict[str, float]] = {}
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in _csv.DictReader(handle):
            if str(row.get("method", "")).strip() != expected_method:
                continue
            if str(row.get("run_status", "")).strip().upper() != "OK":
                continue
            try:
                lr = parse_lr(row["learning_rate"])
            except (KeyError, ValueError, ScreeningContractError):
                continue
            key = (
                str(row.get("dataset", "")).strip(),
                str(row.get("target_identity", "")).strip(),
                lr,
            )
            try:
                index[key] = {
                    "rmse_original": float(row["rmse_original"]),
                    "smape_original": float(row["smape_original"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return index


def load_v1_reference_runs(csv_path: Optional[Path]) -> Dict[tuple, Dict[str, float]]:
    """Index frozen v1 rows by (dataset, target_identity, lr) for delta columns."""
    return _index_ok_runs_by_target_lr(csv_path, V1_METHOD_DISPLAY)


def load_v2_reference_runs(csv_path: Optional[Path]) -> Dict[tuple, Dict[str, float]]:
    """Index frozen v2 canonical-grid rows by (dataset, target_identity, lr)."""
    return _index_ok_runs_by_target_lr(csv_path, V2_METHOD_DISPLAY)


def _lookup_lr(value: object) -> float:
    try:
        return parse_lr(value)
    except ScreeningContractError:
        return float(value)


def attach_v1_deltas(
    row: Mapping[str, Any],
    v1_index: Mapping[tuple, Mapping[str, float]],
) -> Dict[str, Any]:
    """Add v1 metrics and v2-minus-v1 deltas (negative delta = v2 better)."""
    out = dict(row)
    key = (
        str(row.get("dataset", "")).strip(),
        str(row.get("target_identity", "")).strip(),
        _lookup_lr(row["learning_rate"]),
    )
    ref = v1_index.get(key)
    if ref is None:
        out["v1_rmse_original"] = None
        out["v1_smape_original"] = None
        out["delta_rmse_vs_v1"] = None
        out["delta_smape_vs_v1"] = None
        out["v2_beats_v1_rmse"] = None
        out["v1_status"] = "MISSING"
        return out
    v2_rmse = row.get("rmse_original")
    v2_smape = row.get("smape_original")
    out["v1_rmse_original"] = float(ref["rmse_original"])
    out["v1_smape_original"] = float(ref["smape_original"])
    out["delta_rmse_vs_v1"] = (
        float(v2_rmse) - float(ref["rmse_original"]) if v2_rmse is not None else None
    )
    out["delta_smape_vs_v1"] = (
        float(v2_smape) - float(ref["smape_original"]) if v2_smape is not None else None
    )
    out["v2_beats_v1_rmse"] = (
        bool(float(v2_rmse) < float(ref["rmse_original"])) if v2_rmse is not None else None
    )
    out["v1_status"] = "FOUND"
    return out


def _boundary_verdict(extra_rmse: object, boundary_rmse: object) -> str:
    if extra_rmse is None or boundary_rmse is None:
        return "MISSING"
    extra = float(extra_rmse)
    boundary = float(boundary_rmse)
    if not math.isfinite(extra) or not math.isfinite(boundary):
        return "MISSING"
    if boundary == 0.0:
        relative = 0.0 if extra == 0.0 else math.inf
    else:
        relative = abs(extra - boundary) / abs(boundary)
    if relative < RMSE_TIE_RELATIVE:
        return "TIE"
    if extra < boundary:
        return "EXTRA_BETTER"
    return "BOUNDARY_BETTER"


def attach_boundary_deltas(
    row: Mapping[str, Any],
    v2_index: Mapping[tuple, Mapping[str, float]],
    *,
    boundary_lr: float = V2_CANONICAL_BOUNDARY_LR,
) -> Dict[str, Any]:
    """Compare an extra-lr row against the frozen V2 canonical-grid max (3e-4)."""
    out = dict(row)
    key = (
        str(row.get("dataset", "")).strip(),
        str(row.get("target_identity", "")).strip(),
        parse_lr(boundary_lr),
    )
    out["boundary_lr"] = format_lr(boundary_lr)
    ref = v2_index.get(key)
    if ref is None:
        out["boundary_rmse_original"] = None
        out["boundary_smape_original"] = None
        out["delta_rmse_vs_boundary"] = None
        out["delta_smape_vs_boundary"] = None
        out["extra_beats_boundary_rmse"] = None
        out["boundary_status"] = "MISSING"
        out["boundary_verdict"] = "MISSING"
        return out
    extra_rmse = row.get("rmse_original")
    extra_smape = row.get("smape_original")
    out["boundary_rmse_original"] = float(ref["rmse_original"])
    out["boundary_smape_original"] = float(ref["smape_original"])
    out["delta_rmse_vs_boundary"] = (
        float(extra_rmse) - float(ref["rmse_original"]) if extra_rmse is not None else None
    )
    out["delta_smape_vs_boundary"] = (
        float(extra_smape) - float(ref["smape_original"]) if extra_smape is not None else None
    )
    out["extra_beats_boundary_rmse"] = (
        bool(float(extra_rmse) < float(ref["rmse_original"])) if extra_rmse is not None else None
    )
    out["boundary_status"] = "FOUND"
    out["boundary_verdict"] = _boundary_verdict(extra_rmse, ref["rmse_original"])
    return out


def extra_lr_recommendation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    compared = [row for row in rows if row.get("boundary_status") == "FOUND"]
    extra_better = sum(1 for row in compared if row.get("boundary_verdict") == "EXTRA_BETTER")
    boundary_better = sum(
        1 for row in compared if row.get("boundary_verdict") == "BOUNDARY_BETTER"
    )
    ties = sum(1 for row in compared if row.get("boundary_verdict") == "TIE")
    n = len(compared)
    if n == 0:
        rec = "MISSING_BOUNDARY_REFERENCE"
    elif extra_better == 0:
        rec = "BOUNDARY_IS_PEAK"
    elif extra_better == n:
        rec = "GRID_HEADROOM_YES"
    else:
        rec = "MIXED_RESELECT"
    return {
        "compared": n,
        "extra_better": extra_better,
        "boundary_better": boundary_better,
        "ties": ties,
        "recommendation": rec,
        "canonical_lr_grid_unchanged": True,
        "selection_frozen": False,
    }


def _fmt_metric(value: object) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "MISSING"
    return f"{float(value):.4f}"


def format_extra_lr_dataset_terminal(
    dataset: str,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "================================================================================",
        f"{dataset} EXTRA-LR vs V2@{format_lr(V2_CANONICAL_BOUNDARY_LR)}",
        "================================================================================",
        f"{'TARGET':<22}{'EXTRA':<8}{'EXTRA_RMSE':<12}{'V2_3e-4':<12}{'dRMSE':<12}{'VERDICT'}",
        "--------------------------------------------------------------------------------",
    ]
    extra_wins = 0
    compared = 0
    for row in rows:
        lines.append(
            f"{str(row.get('target_identity', '')):<22}"
            f"{str(row.get('learning_rate_label', '')):<8}"
            f"{_fmt_metric(row.get('rmse_original')):<12}"
            f"{_fmt_metric(row.get('boundary_rmse_original')):<12}"
            f"{_fmt_metric(row.get('delta_rmse_vs_boundary')):<12}"
            f"{row.get('boundary_verdict') or 'MISSING'}"
        )
        if row.get("boundary_status") == "FOUND":
            compared += 1
            extra_wins += int(row.get("boundary_verdict") == "EXTRA_BETTER")
    lines.extend(
        [
            "--------------------------------------------------------------------------------",
            f"EXTRA_RMSE_WINS = {extra_wins}/{compared}",
            "SELECTION_ROLE=extra_lr_exploration",
            "CANONICAL_LR_GRID_CHANGE=NO",
            "================================================================================",
        ]
    )
    return "\n".join(lines) + "\n"


def format_extra_lr_boundary_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    """Per-target extra-lr vs frozen V2@3e-4 table plus freeze/headroom recommendation."""
    rec = extra_lr_recommendation(rows)
    lines = [
        "================================================================================",
        f"FUSED_TARGET_ONLY_V2 extra-lr vs canonical boundary {format_lr(V2_CANONICAL_BOUNDARY_LR)}",
        "================================================================================",
        f"{'DATASET':<9}{'TARGET':<22}{'EXTRA':<8}{'EXTRA_RMSE':<12}{'V2_3e-4':<12}{'dRMSE':<12}"
        f"{'EXTRA_sMAPE':<13}{'V2_sMAPE':<12}{'dSMAPE':<12}{'VERDICT'}",
        "--------------------------------------------------------------------------------",
    ]
    for row in rows:
        lines.append(
            f"{str(row.get('dataset', '')):<9}"
            f"{str(row.get('target_identity', '')):<22}"
            f"{str(row.get('learning_rate_label', '')):<8}"
            f"{_fmt_metric(row.get('rmse_original')):<12}"
            f"{_fmt_metric(row.get('boundary_rmse_original')):<12}"
            f"{_fmt_metric(row.get('delta_rmse_vs_boundary')):<12}"
            f"{_fmt_metric(row.get('smape_original')):<13}"
            f"{_fmt_metric(row.get('boundary_smape_original')):<12}"
            f"{_fmt_metric(row.get('delta_smape_vs_boundary')):<12}"
            f"{row.get('boundary_verdict') or 'MISSING'}"
        )
    smape_wins = sum(
        1
        for row in rows
        if row.get("delta_smape_vs_boundary") is not None
        and float(row["delta_smape_vs_boundary"]) < 0.0
    )
    smape_compared = sum(
        1 for row in rows if row.get("delta_smape_vs_boundary") is not None
    )
    lines.extend(
        [
            "--------------------------------------------------------------------------------",
            f"EXTRA_RMSE_WINS  = {rec['extra_better']}/{rec['compared']}",
            f"BOUNDARY_WINS    = {rec['boundary_better']}/{rec['compared']}",
            f"TIES             = {rec['ties']}/{rec['compared']}",
            f"EXTRA_SMAPE_WINS = {smape_wins}/{smape_compared}",
            f"RECOMMENDATION   = {rec['recommendation']}",
            "CANONICAL_LR_GRID_UNCHANGED = YES",
            "SELECTION_FROZEN = NO",
            "================================================================================",
        ]
    )
    return "\n".join(lines) + "\n"


def format_v1_v2_summary(rows: Sequence[Mapping[str, Any]]) -> str:
    """Per-run v1-vs-v2 comparison table plus win counts."""
    lines = [
        "================================================================================",
        "FUSED_TARGET_ONLY_V2 vs frozen V1 (original RMSE / sMAPE)",
        "================================================================================",
        f"{'DATASET':<9}{'TARGET':<22}{'LR':<8}{'V2_RMSE':<11}{'V1_RMSE':<11}{'dRMSE':<11}{'V2_sMAPE':<11}{'V1_sMAPE':<11}{'dSMAPE':<11}",
        "--------------------------------------------------------------------------------",
    ]

    rmse_wins = 0
    smape_wins = 0
    compared = 0
    for row in rows:
        lines.append(
            f"{str(row.get('dataset', '')):<9}"
            f"{str(row.get('target_identity', '')):<22}"
            f"{str(row.get('learning_rate_label', '')):<8}"
            f"{_fmt_metric(row.get('rmse_original')):<11}"
            f"{_fmt_metric(row.get('v1_rmse_original')):<11}"
            f"{_fmt_metric(row.get('delta_rmse_vs_v1')):<11}"
            f"{_fmt_metric(row.get('smape_original')):<11}"
            f"{_fmt_metric(row.get('v1_smape_original')):<11}"
            f"{_fmt_metric(row.get('delta_smape_vs_v1')):<11}"
        )
        if row.get("delta_rmse_vs_v1") is not None:
            compared += 1
            rmse_wins += int(float(row["delta_rmse_vs_v1"]) < 0.0)
        if row.get("delta_smape_vs_v1") is not None:
            smape_wins += int(float(row["delta_smape_vs_v1"]) < 0.0)
    lines.extend(
        [
            "--------------------------------------------------------------------------------",
            f"V2_RMSE_WINS  = {rmse_wins}/{compared}",
            f"V2_SMAPE_WINS = {smape_wins}/{compared}",
            "================================================================================",
        ]
    )
    return "\n".join(lines) + "\n"
