"""Identities, isolation, source selection, and statistics for SITA-TCN."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts.baselines.revin_strengthened_protocol import (
    FORBIDDEN_TARGETS as REVIN_FORBIDDEN_TARGETS,
    PAPER_TARGETS as BL7_PAPER_TARGETS,
)
from src.constants import SOLIDIFIED_KNN_ROOT
from src.evaluation.metrics import compute_original_scale_metrics
from src.protocols.experiment_protocol import FORMAL_SEEDS
from src.protocols.runner_adapter import source_key_mask
from src.source_selection.source_selector import SourceSelector
from src.transfer_methods.ms_katl_v3_frame_assembly import assemble_v3_cell_frames
from src.transfer_methods.sita_tcn import (
    BL7_METHOD_ID,
    BL7_METHOD_NAME,
    METHOD_NAME,
    SITATCNContractError,
    WEIGHT_SUM_TOLERANCE,
    official_inverse_distance_weights,
)
from tools.promote_bl6_bl7_formal_results import FORMAL_TARGETS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "sita_tcn"
PILOT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "pilot"
FORMAL_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "formal"
SOURCE_CHECKPOINT_ROOT = DEFAULT_OUTPUT_ROOT / "source_checkpoints"
AUDIT_ROOT = DEFAULT_OUTPUT_ROOT / "audits"
MODE = "without"
DATASETS = ("D1", "D4", "D5", "D6")
TARGETS = dict(FORMAL_TARGETS)
HORIZONS = (1, 3, 5)
SEEDS = tuple(FORMAL_SEEDS)
EXPECTED_TARGET_RUNS = 60
PILOT_DATASET = "D4"
PILOT_TARGET = "166/258"
PILOT_HORIZON = 3
PILOT_SEEDS = (42,)
TIE_RULE = "exact_numeric_no_tolerance"
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_RANDOM_SEED = 20260903
SOURCE_SEED_POLICY = "(dataset, horizon, seed)"
WEIGHT_MODE = "inverse_distance"
SOURCE_SELECTION_AUTHORITY = "src.source_selection.source_selector.SourceSelector"
FRAME_AUTHORITY = "src.transfer_methods.ms_katl_v3_frame_assembly.assemble_v3_cell_frames"
FORBIDDEN_D4_TARGETS = {"166/311", "166/313"}
PROTECTED_PATH_MARKERS = (
    "MS_KATL_Paper_Data_Freeze",
    "Section4_Comparison_Statistics",
    "BL6_BL7_FORMAL",
    "MS_KATL_FTO_THREE_TRACK_ROBUSTNESS",
    "FTO_V3_SCREENING",
    "outputs/baselines",
)
FORMAL_METHOD_COLLISIONS = {
    "SITA",
    "FUSED_TARGET_ONLY_V3",
    "TCN",
    "BL7_TCN",
    "DLinear",
    "MSML-TL",
}
CODE_IDENTITY_PATHS = (
    "src/transfer_methods/sita_tcn.py",
    "scripts/sita_tcn_protocol.py",
    "scripts/run_tcn_backbone_generalization.py",
    "scripts/baselines/bl7_tcn.py",
    "scripts/baselines/bl7_tcn_revin.py",
    "src/transfer_methods/revin_torch.py",
    "src/source_selection/source_selector.py",
    "src/evaluation/metrics.py",
    "scripts/baselines/baseline_data_loader.py",
)
SOLIDIFIED_D4_258_DIAGNOSTIC_SOURCES = ("166/184", "166/530", "166/229")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(root: Path = ROOT) -> dict[str, str]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {
        "git_branch": run(["branch", "--show-current"]),
        "git_head": run(["rev-parse", "HEAD"]),
        "git_status_short": run(["status", "--short"]),
    }


def record_code_identity(root: Path = ROOT) -> dict[str, str]:
    git = git_identity(root)
    payload = {
        "git_branch": git["git_branch"],
        "git_head": git["git_head"],
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        payload["framework_version"] = str(torch.__version__)
    except Exception:
        payload["framework_version"] = "unavailable"
    for relative in CODE_IDENTITY_PATHS:
        path = root / relative
        payload[f"sha256:{relative}"] = sha256_file(path) if path.is_file() else "missing"
    return payload


def json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def assert_isolated_output_dir(output_dir: Path) -> Path:
    destination = Path(output_dir).resolve()
    allowed_root = DEFAULT_OUTPUT_ROOT.resolve()
    if destination != allowed_root and allowed_root not in destination.parents:
        raise SITATCNContractError(
            f"SITA-TCN outputs must stay under {allowed_root}, got {destination}"
        )
    freeze = (ROOT / "outputs" / "baselines").resolve()
    if destination == freeze or freeze in destination.parents:
        raise SITATCNContractError("SITA-TCN must not write under outputs/baselines")
    for marker in PROTECTED_PATH_MARKERS:
        text = str(destination).lower()
        if marker.lower() in text and "sita_tcn" not in text:
            raise SITATCNContractError(f"refusing to write into protected path {destination}")
    return destination


def assert_identity_isolation(methods: Iterable[str]) -> None:
    observed = {str(method) for method in methods}
    if METHOD_NAME in FORMAL_METHOD_COLLISIONS:
        raise SITATCNContractError("SITA-TCN identity collided with a frozen method name")
    overlap = observed.intersection(FORMAL_METHOD_COLLISIONS)
    if overlap:
        raise SITATCNContractError(f"result identities collided with freeze names: {sorted(overlap)}")
    if BL7_METHOD_NAME in observed and METHOD_NAME in observed:
        return
    for method in observed:
        if method not in {METHOD_NAME, BL7_METHOD_NAME, BL7_METHOD_ID}:
            raise SITATCNContractError(f"unsupported SITA-TCN comparison identity: {method}")


def assert_paper_target(dataset: str, target: str) -> None:
    dataset = str(dataset).upper()
    target = str(target)
    if target in FORBIDDEN_D4_TARGETS or target in REVIN_FORBIDDEN_TARGETS:
        raise SITATCNContractError(f"retired D4 target is forbidden: {target}")
    expected = TARGETS[dataset]
    if target != expected:
        raise SITATCNContractError(f"{dataset} target must be {expected}, got {target}")
    if BL7_PAPER_TARGETS[dataset] != expected:
        raise SITATCNContractError("paper target drifted from BL7 RevIN protocol")


def format_source_id(source_key: object) -> str:
    if isinstance(source_key, (list, tuple)):
        return "/".join(str(part) for part in source_key)
    text = str(source_key).strip()
    if "/" in text:
        return text
    parts = text.split("_", 1)
    if len(parts) == 2 and parts[0] and parts[1]:
        return f"{parts[0]}/{parts[1]}"
    return text


def cell_dirname(dataset: str, target: str, horizon: int, seed: int, init_mode: str) -> str:
    slug = str(target).replace("/", "_")
    return f"{dataset}__{slug}__H{int(horizon)}__S{int(seed)}__without__{init_mode}"


def cell_paths(output_root: Path, dataset: str, target: str, horizon: int, seed: int, init_mode: str) -> dict[str, Path]:
    cell_dir = Path(output_root) / "cells" / cell_dirname(dataset, target, horizon, seed, init_mode)
    return {
        "dir": cell_dir,
        "predictions": cell_dir / "predictions.csv",
        "metrics": cell_dir / "metrics.json",
        "manifest": cell_dir / "cell_manifest.json",
        "complete": cell_dir / "cell_complete.json",
        "history": cell_dir / "EARLY_CONVERGENCE_TRACE.csv",
        "error": cell_dir / "cell_error.txt",
    }


def solidified_diagnostic_sources(dataset: str, target: str) -> tuple[str, ...]:
    dataset_id = int(str(dataset)[1:])
    path = Path(SOLIDIFIED_KNN_ROOT) / f"Dataset{dataset_id}" / "knn_without_info_sharing.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    entity = str(target).replace("/", "_")
    rows = payload.get("results", {}).get(entity) or payload.get("knn_results", {}).get(entity)
    if not rows:
        raise SITATCNContractError(f"solidified KNN diagnostic missing {dataset}/{target}")
    identities = []
    for row in rows[:3]:
        store = row.get("source_store_id")
        product = row.get("source_product_id")
        if store and product:
            identities.append(f"{store}/{product}")
        else:
            identities.append(format_source_id(row.get("source_entity") or row.get("source_key")))
    return tuple(identities)


def select_formal_sita_sources(
    dataset: str,
    target: str,
    *,
    horizon: int,
    seed: int,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    assert_paper_target(dataset, target)
    cell = assemble_v3_cell_frames(
        dataset,
        target,
        horizon=int(horizon),
        seed=int(seed),
        project_root=project_root,
        mode=MODE,
    )
    context = cell["k3_selection_context"]
    evidence = context.selection_for_method(
        target_df=cell["target_model_df"],
        source_df=cell["source_model_df"],
        feature_cols=cell["model_feature_cols"],
        group_cols=cell["group_cols"],
        k=3,
        weight_mode=WEIGHT_MODE,
    )
    payload = evidence.method_wrapper(
        lifecycle_identity=context.lifecycle_identity,
        target_df=cell["target_model_df"],
        source_df=cell["source_model_df"],
        feature_cols=cell["model_feature_cols"],
        group_cols=cell["group_cols"],
        k=3,
        weight_mode=WEIGHT_MODE,
    )
    raw_sources = list(payload.get("sources") or [])
    if len(raw_sources) != 3:
        raise SITATCNContractError(f"formal SITA selection did not return Top-3, got {len(raw_sources)}")
    distances = [float(row["distance"]) for row in raw_sources]
    recomputed = official_inverse_distance_weights(distances)
    selected = []
    source_df = cell["source_model_df"]
    group_cols = tuple(cell["group_cols"])
    for index, row in enumerate(raw_sources):
        source_key = row["source_key"]
        mask = source_key_mask(source_df, group_cols, source_key)
        series = source_df.loc[mask].copy()
        if series.empty:
            raise SITATCNContractError(f"selected source {source_key!r} missing from source frame")
        series["date"] = pd.to_datetime(series["date"], errors="raise").dt.normalize()
        series = series.sort_values("date")
        sales = pd.to_numeric(series["sales"], errors="coerce").to_numpy(dtype=np.float64)
        if sales.size == 0 or not np.isfinite(sales).all():
            raise SITATCNContractError(f"source {source_key!r} sales are empty or non-finite")
        official_weight = float(row["weight"])
        if abs(official_weight - float(recomputed[index])) > 1e-12:
            raise SITATCNContractError(
                f"selector weight {official_weight} != compute_source_weights {float(recomputed[index])}"
            )
        selected.append(
            {
                "source_rank": int(row.get("source_rank", index + 1)),
                "source_id": format_source_id(source_key),
                "source_key": source_key,
                "distance": float(distances[index]),
                "weight": float(recomputed[index]),
                "sales": sales,
                "n_obs": int(sales.size),
            }
        )
    weight_sum = float(sum(item["weight"] for item in selected))
    if abs(weight_sum - 1.0) >= WEIGHT_SUM_TOLERANCE:
        raise SITATCNContractError(f"weight_sum={weight_sum} is not 1")
    if dataset == "D4" and target == "166/258":
        observed = tuple(item["source_id"] for item in selected)
        diagnostic = solidified_diagnostic_sources(dataset, target)
        if observed != diagnostic and observed != SOLIDIFIED_D4_258_DIAGNOSTIC_SOURCES:
            # Identity must match formal SITA. Diagnostic JSON is checked but runtime is authority.
            # A mismatch against both the helper and the frozen diagnostic list is a blocker.
            if observed != diagnostic:
                raise SITATCNContractError(
                    f"D4/166/258 runtime Top-3 {observed} != solidified diagnostic {diagnostic}"
                )
    meta = dict(payload.get("meta") or {})
    return {
        "sources": selected,
        "meta": meta,
        "group_cols": list(group_cols),
        "selection_authority": str(meta.get("selection_authority", "shared_protocol")),
        "distance_metric": str(meta.get("distance_metric", "euclidean")),
        "weight_mode": WEIGHT_MODE,
        "weight_sum": weight_sum,
        "candidate_pool_digest": str(meta.get("candidate_pool_digest", "")),
        "selection_result_digest": str(meta.get("selection_result_digest", "")),
        "frame_authority": FRAME_AUTHORITY,
    }


def relative_improvement_pct(baseline_value: float, candidate_value: float) -> float:
    baseline_value = float(baseline_value)
    if not math.isfinite(baseline_value) or baseline_value == 0:
        raise SITATCNContractError(f"relative improvement requires a positive baseline: {baseline_value}")
    return (baseline_value - float(candidate_value)) / baseline_value * 100.0


def win_tie_loss(baseline_value: float, candidate_value: float) -> str:
    left = float(baseline_value)
    right = float(candidate_value)
    if right < left:
        return "W"
    if right == left:
        return "T"
    return "L"


def bootstrap_mean_ci(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 12:
        raise SITATCNContractError("bootstrap input must contain exactly 12 condition values")
    array = np.asarray([float(value) for value in values], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def wilcoxon_paired(differences: Sequence[float]) -> dict[str, Any]:
    from scipy import stats

    values = np.asarray([float(value) for value in differences], dtype=float)
    zero_count = int(np.count_nonzero(values == 0.0))
    nonzero = values[values != 0.0]
    payload = {
        "n": int(len(values)),
        "zero_difference_count": zero_count,
        "method": "exact",
        "alternative": "two-sided",
        "status": "PASS",
        "tie_rule": TIE_RULE,
    }
    if nonzero.size == 0:
        payload.update({"statistic": float("nan"), "p_value": float("nan"), "status": "ALL_ZERO"})
        return payload
    try:
        if zero_count:
            raise ValueError("zero differences block exact Wilcoxon freeze protocol")
        abs_values = np.abs(nonzero)
        if len(set(np.round(abs_values, 12).tolist())) != len(abs_values):
            raise ValueError("tied absolute differences block exact Wilcoxon freeze protocol")
        result = stats.wilcoxon(
            nonzero,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="exact",
            nan_policy="raise",
        )
        payload["statistic"] = float(result.statistic)
        payload["p_value"] = float(result.pvalue)
    except ValueError as exc:
        result = stats.wilcoxon(
            nonzero,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            nan_policy="raise",
        )
        payload.update(
            {
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "status": "FALLBACK_NONEXACT",
                "fallback_reason": str(exc),
            }
        )
    if nonzero.size:
        ranks = stats.rankdata(np.abs(nonzero))
        positive = float(np.sum(ranks[nonzero > 0]))
        negative = float(np.sum(ranks[nonzero < 0]))
        denom = float(nonzero.size * (nonzero.size + 1) / 2.0)
        payload["rank_biserial"] = (positive - negative) / denom if denom else float("nan")
        payload["cohens_d_paired"] = float(nonzero.mean() / nonzero.std(ddof=1)) if nonzero.size > 1 and nonzero.std(ddof=1) > 0 else float("nan")
    return payload


def recompute_original_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, Any]:
    metrics = compute_original_scale_metrics(np.asarray(y_true), np.asarray(y_pred))
    return {
        "rmse": float(metrics["rmse"]),
        "smape": float(metrics["smape"]),
        "n": int(np.asarray(y_true).reshape(-1).size),
        "metric_space": "original_sales_space",
        "primary_metric_space": str(metrics["primary_metric_space"]),
    }


def environment_payload() -> dict[str, Any]:
    identity = record_code_identity()
    return {
        **identity,
        "device": "cpu",
        "deterministic_flags": {
            "python_hashseed": "set_via_set_protocol_seed",
            "numpy": True,
            "torch.manual_seed": True,
            "torch.use_deterministic_algorithms": False,
            "note": "matches formal BL7 seeding; extra deterministic algorithms are not part of BL7",
        },
        "created_at": now_utc(),
    }
