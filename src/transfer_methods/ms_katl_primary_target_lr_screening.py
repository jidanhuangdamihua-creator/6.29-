"""D1-D6 Primary Target × learning-rate screening for FUSED_TARGET_ONLY.

Development / screening track. Does not modify FORMAL_METHODS, MSML, No-TL,
baselines, or the FUSED_TARGET_ONLY architecture.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metric_contract import ORIGINAL_SALES_SPACE
from src.evaluation.metrics import inverse_transform_sales, smape
from src.protocols.experiment_protocol import (
    FORMAL_METHODS,
    formal_target_entity_keys,
    get_experiment_protocol,
    serialize_canonical_target_key,
    validate_canonical_target_key,
)
from src.transfer_methods.ms_katl_cross_target import (
    BL_METHODS,
    D4_CROSS_TARGET_IDENTITIES,
    FORMAL_MSML_METHOD,
    FORMAL_NO_TL_METHOD,
)
from src.transfer_methods.ms_katl_lite import METHOD_NAME
from src.transfer_methods.ms_katl_mechanism_ablation import (
    ABLATION_ARM_SPECS,
    assert_formal_methods_untouched,
)
from src.transfer_methods.ms_katl_metric_space_repair import (
    MetricSpaceMismatchError,
    require_same_metric_space,
)
from src.utils.experiment_hyperparams import FIXED_EPOCHS, FIXED_LEARNING_RATE

SCREENING_LABEL = "MS_KATL_PRIMARY_TARGET_LR_SCREENING"
METHOD_DISPLAY = "FUSED_TARGET_ONLY"
METHOD_ARM = "fused_target_only"
TRACK = "development_screening"
SCREENING_DATASETS: Tuple[str, ...] = ("D1", "D2", "D3", "D4", "D5", "D6")
LR_GRID: Tuple[float, ...] = (3e-5, 1e-4, 3e-4)
LR_GRID_LABELS: Tuple[str, ...] = ("3e-5", "1e-4", "3e-4")
STAGE_A_LEARNING_RATE = float(FIXED_LEARNING_RATE)
SCREENING_HORIZON = 1
SCREENING_SEED = 42
SCREENING_EPOCHS = int(FIXED_EPOCHS)
SCREENING_MODE = "without"
RMSE_TIE_RELATIVE = 0.01
REFERENCE_MISSING = "MISSING"
NA = "NA"
NORMALIZED_SPACE = "normalized_minmax_space"
ACCEPTED_FORMAL_RUN = "20260816_183000_d1d6_630ae4ef_h1s42_d5first_p8"
ROBUSTNESS_LAUNCH_FORBIDDEN = (
    "H={1,3,5}",
    "seed={42,43,44}",
    "run_robustness",
    "multi_seed_selection",
)
FUSED_TARGET_ONLY_CONTRACT = {
    "use_fused_target_init": True,
    "target_backbone_trainable": True,
    "transfer_branch_enabled": False,
    "gate_mode": "zero",
    "lambda_adv": 0.0,
    "attention_used_in_prediction": False,
    "adapter_used_in_prediction": False,
}


class ScreeningContractError(RuntimeError):
    """Fail-closed screening contract violation."""


def _dataset_id_int(dataset_id: object) -> int:
    return int(get_experiment_protocol(dataset_id).dataset_id[1:])


def dataset_name(dataset_id: object) -> str:
    return get_experiment_protocol(dataset_id).dataset_id


def format_lr(value: object) -> str:
    number = float(value)
    for expected, label in zip(LR_GRID, LR_GRID_LABELS):
        if math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-18):
            return label
    raise ScreeningContractError(f"learning_rate {value!r} is outside LR_GRID={list(LR_GRID)}")


def parse_lr(value: object) -> float:
    if isinstance(value, str):
        text = value.strip().lower().replace(" ", "")
        mapping = {label: number for label, number in zip(LR_GRID_LABELS, LR_GRID)}
        mapping.update({"0.00003": 3e-5, "0.0001": 1e-4, "0.0003": 3e-4})
        if text in mapping:
            return float(mapping[text])
    number = float(value)
    format_lr(number)
    return number


def assert_lr_grid(values: Sequence[object]) -> Tuple[float, ...]:
    resolved = tuple(parse_lr(item) for item in values)
    if resolved != LR_GRID:
        raise ScreeningContractError(
            f"LR_GRID must be exactly {list(LR_GRID)}, got {list(resolved)}"
        )
    return resolved


def canonical_candidate_targets(dataset_id: object) -> Tuple[str, ...]:
    """Return protocol identities. D4 uses the established five-target order, not CSV order."""
    protocol_keys = formal_target_entity_keys(dataset_id)
    dataset = dataset_name(dataset_id)
    if dataset == "D4":
        if set(protocol_keys) != set(D4_CROSS_TARGET_IDENTITIES):
            raise ScreeningContractError(
                "D4 screening inventory must equal the canonical D4 five-target set "
                f"{list(D4_CROSS_TARGET_IDENTITIES)}, got {list(protocol_keys)}"
            )
        return D4_CROSS_TARGET_IDENTITIES
    return protocol_keys


def parse_target_identity(dataset_id: object, value: object) -> Dict[str, str]:
    """Parse a canonical identity without using CSV row order."""
    dataset = dataset_name(dataset_id)
    inventory = canonical_candidate_targets(dataset)
    text = str(value).strip()
    if not text or text.upper() == REFERENCE_MISSING:
        raise ScreeningContractError(f"empty {dataset} target identity: {value!r}")
    protocol = get_experiment_protocol(dataset)
    arity = len(protocol.source_pool_rule.key_fields)
    matched: Optional[Tuple[str, ...]] = None
    for identity in inventory:
        key = tuple(identity.split("/")) if "/" in identity else (identity,)
        if len(key) != arity:
            continue
        entity_key = "_".join(key)
        if text in {identity, entity_key}:
            matched = key
            break
    if matched is None:
        if "/" in text:
            parts = tuple(part.strip() for part in text.split("/"))
        elif arity == 1:
            parts = (text,)
        else:
            raise ScreeningContractError(
                f"{dataset} target identity must use canonical slash form, got {value!r}"
            )
        matched = validate_canonical_target_key(dataset, parts)
    else:
        matched = validate_canonical_target_key(dataset, matched)
    identity = serialize_canonical_target_key(dataset, matched)
    if identity not in inventory:
        raise ScreeningContractError(
            f"{dataset} target {identity} is not in the canonical inventory {list(inventory)}"
        )
    entity_key = "_".join(matched)
    return {
        "dataset": dataset,
        "target_identity": identity,
        "entity_key": entity_key,
        "prediction_dirname": entity_key.replace("/", "_"),
        "key": "/".join(matched),
    }


def plan_screening(
    *,
    datasets: Sequence[object] = SCREENING_DATASETS,
    lr_grid: Sequence[object] = LR_GRID,
) -> Dict[str, Any]:
    resolved_datasets = tuple(dataset_name(item) for item in datasets)
    if tuple(resolved_datasets) != SCREENING_DATASETS and set(resolved_datasets) != set(SCREENING_DATASETS):
        missing = [item for item in SCREENING_DATASETS if item not in resolved_datasets]
        extra = [item for item in resolved_datasets if item not in SCREENING_DATASETS]
        if extra or (missing and set(resolved_datasets) != set(SCREENING_DATASETS)):
            if extra:
                raise ScreeningContractError(f"unknown screening datasets: {extra}")
    assert_lr_grid(lr_grid)
    inventory: Dict[str, Tuple[str, ...]] = {}
    planned: List[Dict[str, Any]] = []
    for dataset in resolved_datasets:
        targets = canonical_candidate_targets(dataset)
        inventory[dataset] = targets
        spec = ABLATION_ARM_SPECS[METHOD_ARM]
        for identity in targets:
            parsed = parse_target_identity(dataset, identity)
            for lr in LR_GRID:
                planned.append(
                    {
                        "dataset": dataset,
                        "target_identity": parsed["target_identity"],
                        "entity_key": parsed["entity_key"],
                        "prediction_dirname": parsed["prediction_dirname"],
                        "learning_rate": float(lr),
                        "learning_rate_label": format_lr(lr),
                        "method": METHOD_DISPLAY,
                        "arm": METHOD_ARM,
                        "horizon": SCREENING_HORIZON,
                        "seed": SCREENING_SEED,
                        "epochs": SCREENING_EPOCHS,
                        "mode": SCREENING_MODE,
                        **{f"arch_{key}": value for key, value in FUSED_TARGET_ONLY_CONTRACT.items()},
                        "letter": spec.letter,
                    }
                )
    total_targets = sum(len(values) for values in inventory.values())
    return {
        "datasets": resolved_datasets,
        "inventory": inventory,
        "lr_grid": LR_GRID,
        "lr_grid_labels": LR_GRID_LABELS,
        "planned": planned,
        "total_targets": int(total_targets),
        "total_screening_runs": int(total_targets) * len(LR_GRID),
        "horizon": SCREENING_HORIZON,
        "seed": SCREENING_SEED,
        "epochs": SCREENING_EPOCHS,
        "method": METHOD_DISPLAY,
        "track": TRACK,
        "metric_space": ORIGINAL_SALES_SPACE,
        "stage_a_learning_rate": STAGE_A_LEARNING_RATE,
    }


def format_preflight(plan: Optional[Mapping[str, Any]] = None) -> str:
    resolved = dict(plan or plan_screening())
    inventory: Mapping[str, Sequence[str]] = resolved["inventory"]
    lines = [
        "================================================================================",
        "D1-D6 PRIMARY TARGET × LR SCREENING PREFLIGHT",
        "================================================================================",
        "",
        f"METHOD={METHOD_DISPLAY}",
        f"TRACK={TRACK}",
        "",
        f"HORIZON={SCREENING_HORIZON}",
        f"SEED={SCREENING_SEED}",
        f"EPOCHS={SCREENING_EPOCHS}",
        "",
        "LR_GRID:",
    ]
    lines.extend(LR_GRID_LABELS)
    lines.append("")
    for dataset in SCREENING_DATASETS:
        targets = list(inventory.get(dataset, ()))
        lines.append(f"{dataset} TARGETS={len(targets)}")
        lines.extend(targets)
        lines.append("")
    lines.extend(
        [
            f"TOTAL_TARGETS={resolved['total_targets']}",
            f"TOTAL_SCREENING_RUNS={resolved['total_screening_runs']}",
            "",
            f"METRIC_SPACE={ORIGINAL_SALES_SPACE}",
            "FORMAL_METHODS_CHANGE=NO",
            "ARCHITECTURE_CHANGE=NO",
            "================================================================================",
        ]
    )
    return "\n".join(lines) + "\n"


def print_screening_preflight(plan: Optional[Mapping[str, Any]] = None) -> str:
    text = format_preflight(plan)
    print(text, end="")
    return text


def fused_target_only_architecture() -> Dict[str, Any]:
    spec = ABLATION_ARM_SPECS[METHOD_ARM]
    contract = {
        "use_fused_target_init": spec.use_fused_target_init,
        "target_backbone_trainable": spec.target_backbone_trainable,
        "transfer_branch_enabled": spec.transfer_branch_enabled,
        "gate_mode": spec.gate_mode,
        "lambda_adv": 0.0,
        "attention_used_in_prediction": False,
        "adapter_used_in_prediction": False,
    }
    if contract != FUSED_TARGET_ONLY_CONTRACT:
        raise ScreeningContractError(
            f"FUSED_TARGET_ONLY architecture drifted: {contract} != {FUSED_TARGET_ONLY_CONTRACT}"
        )
    if spec.transfer_branch_enabled or spec.gate_mode != "zero":
        raise ScreeningContractError("FUSED_TARGET_ONLY must disable transfer/gate")
    return dict(contract)


def assert_screening_does_not_change_formal_methods() -> Tuple[str, ...]:
    expected = assert_formal_methods_untouched()
    if METHOD_DISPLAY in FORMAL_METHODS or METHOD_NAME in FORMAL_METHODS:
        raise ScreeningContractError("screening must not add FUSED_TARGET_ONLY to FORMAL_METHODS")
    return expected


def default_formal_csv(dataset_id: object, project_root: Path) -> Path:
    dataset = dataset_name(dataset_id).lower()
    number = _dataset_id_int(dataset_id)
    return (
        Path(project_root)
        / "outputs"
        / "runs"
        / ACCEPTED_FORMAL_RUN
        / f"{dataset}_without"
        / "results"
        / f"dataset{number}_without_results.csv"
    )


def default_baseline_csv(dataset_id: object, project_root: Path) -> Path:
    dataset = dataset_name(dataset_id).lower()
    return Path(project_root) / "outputs" / "baselines" / f"{dataset}_baselines_multiseed.csv"


def _normalize_mode(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if "without" in text:
        return "without"
    if text in {"with", "with_info_sharing", "with_information_sharing"}:
        return "with"
    return text


def _optional_float(value: object) -> Optional[float]:
    if value is None or value == "" or str(value).strip().upper() in {REFERENCE_MISSING, NA}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _row_identity(dataset_id: object, row: Mapping[str, Any]) -> Optional[str]:
    dataset = dataset_name(dataset_id)
    for key in ("target_entity_key", "target_identity", "entity_key", "canonical_target_key"):
        if key in row and str(row.get(key, "")).strip():
            try:
                return parse_target_identity(dataset, row[key])["target_identity"]
            except (ScreeningContractError, ValueError):
                continue
    return None


def _normalize_dataset_token(value: object, dataset_id: object) -> str:
    dataset = dataset_name(dataset_id)
    text = str(value).strip().upper().replace("DATASET", "D")
    if text in {"", dataset, dataset[1:], f"D{dataset[1:]}"}:
        return dataset
    if text == f"DATASET{dataset[1:]}":
        return dataset
    return text


def _reference_metric_space(row: Mapping[str, Any]) -> str:
    return str(
        row.get("rmse_metric_space")
        or row.get("metric_space")
        or row.get("current_metric_space_actual")
        or row.get("primary_metric_space_actual")
        or row.get("paper_metric_space_actual")
        or ""
    ).strip()


def _original_reference_rmse(row: Mapping[str, Any]) -> Tuple[Optional[float], str, str]:
    """Return original-sales RMSE only. Never use normalized RMSE for selection."""
    original = _optional_float(
        row.get("original_scale_rmse")
        if row.get("original_scale_rmse") not in (None, "")
        else row.get("rmse_paper")
    )
    space = _reference_metric_space(row)
    if original is not None:
        return original, ORIGINAL_SALES_SPACE, "FOUND"
    normalized_tokens = {NORMALIZED_SPACE, "normalized", "normalized_minmax", "normalized_sales"}
    if space.lower() in normalized_tokens or space in normalized_tokens:
        return None, space or NORMALIZED_SPACE, "WRONG_SPACE"
    if space in {ORIGINAL_SALES_SPACE, "original_sales", "original"}:
        rmse = _optional_float(row.get("rmse"))
        if rmse is None:
            return None, ORIGINAL_SALES_SPACE, REFERENCE_MISSING
        return rmse, ORIGINAL_SALES_SPACE, "FOUND"
    if not space:
        # Accepted formal/baseline protocol is original-sales; still refuse if the
        # only available field is clearly labeled normalized.
        if str(row.get("current_metric_space", "")).strip() in normalized_tokens:
            return None, str(row.get("current_metric_space")), "WRONG_SPACE"
        rmse = _optional_float(row.get("rmse"))
        if rmse is None:
            return None, ORIGINAL_SALES_SPACE, REFERENCE_MISSING
        return rmse, ORIGINAL_SALES_SPACE, "FOUND"
    return None, space, "WRONG_SPACE"


def lookup_reference_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset_id: object,
    target_identity: str,
    method: str,
    horizon: int = SCREENING_HORIZON,
    seed: int = SCREENING_SEED,
    mode: str = SCREENING_MODE,
) -> Dict[str, Any]:
    dataset = dataset_name(dataset_id)
    wanted = parse_target_identity(dataset, target_identity)["target_identity"]
    matches: List[Mapping[str, Any]] = []
    for row in rows:
        identity = _row_identity(dataset, row)
        if identity != wanted:
            continue
        if str(row.get("method", "")).strip() != method:
            continue
        if int(row.get("horizon", -1)) != int(horizon):
            continue
        if int(row.get("seed", row.get("random_state", -1))) != int(seed):
            continue
        row_mode = _normalize_mode(row.get("information_sharing", row.get("scenario", row.get("mode", ""))))
        if row_mode != _normalize_mode(mode):
            continue
        dataset_value = _normalize_dataset_token(
            row.get("dataset_id", row.get("dataset", dataset)),
            dataset,
        )
        if dataset_value not in {"", dataset}:
            continue
        matches.append(row)
    missing = {
        "status": REFERENCE_MISSING,
        "method": method,
        "target_identity": wanted,
        "rmse": None,
        "smape": None,
        "metric_space": None,
    }
    if not matches:
        return missing
    rmses = set()
    usable: List[Mapping[str, Any]] = []
    for row in matches:
        rmse, space, status = _original_reference_rmse(row)
        if status == "WRONG_SPACE":
            raise MetricSpaceMismatchError(
                target_identity=wanted,
                candidate_method=METHOD_DISPLAY,
                candidate_metric_space=ORIGINAL_SALES_SPACE,
                reference_method=method,
                reference_metric_space=space,
            )
        if rmse is None:
            continue
        rmses.add(rmse)
        usable.append(row)
    if not usable:
        return missing
    if len(rmses) != 1:
        raise ScreeningContractError(
            f"conflicting {method} original RMSE values for {dataset} {wanted} "
            f"H={horizon} seed={seed}: {rmses}"
        )
    chosen = usable[0]
    rmse, space, _status = _original_reference_rmse(chosen)
    require_same_metric_space(
        target_identity=wanted,
        candidate_method=METHOD_DISPLAY,
        candidate_metric_space=ORIGINAL_SALES_SPACE,
        reference_method=method,
        reference_metric_space=space,
    )
    return {
        "status": "FOUND",
        "method": method,
        "target_identity": wanted,
        "rmse": rmse,
        "smape": _optional_float(chosen.get("original_scale_smape") or chosen.get("smape_paper") or chosen.get("smape")),
        "n_matches": len(usable),
        "metric_space": ORIGINAL_SALES_SPACE,
        "rmse_metric_space": ORIGINAL_SALES_SPACE,
        "smape_metric_space": ORIGINAL_SALES_SPACE,
    }


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_formal_references(
    path: Optional[Path],
    *,
    dataset_id: object,
    target_identity: str,
    horizon: int = SCREENING_HORIZON,
    seed: int = SCREENING_SEED,
    mode: str = SCREENING_MODE,
) -> Dict[str, Dict[str, Any]]:
    if path is None or not Path(path).exists():
        missing = {
            "status": REFERENCE_MISSING,
            "target_identity": target_identity,
            "rmse": None,
            "smape": None,
            "metric_space": None,
        }
        return {
            "no_tl": {**missing, "method": FORMAL_NO_TL_METHOD},
            "msml": {**missing, "method": FORMAL_MSML_METHOD},
        }
    rows = _read_csv_rows(Path(path))
    return {
        "no_tl": lookup_reference_row(
            rows,
            dataset_id=dataset_id,
            target_identity=target_identity,
            method=FORMAL_NO_TL_METHOD,
            horizon=horizon,
            seed=seed,
            mode=mode,
        ),
        "msml": lookup_reference_row(
            rows,
            dataset_id=dataset_id,
            target_identity=target_identity,
            method=FORMAL_MSML_METHOD,
            horizon=horizon,
            seed=seed,
            mode=mode,
        ),
    }


def load_baseline_references(
    path: Optional[Path],
    *,
    dataset_id: object,
    target_identity: str,
    horizon: int = SCREENING_HORIZON,
    seed: int = SCREENING_SEED,
    mode: str = SCREENING_MODE,
) -> Dict[str, Any]:
    refs: Dict[str, Any] = {}
    if path is None or not Path(path).exists():
        for key, method, _display in BL_METHODS:
            refs[key] = {
                "status": REFERENCE_MISSING,
                "method": method,
                "target_identity": target_identity,
                "rmse": None,
                "smape": None,
            }
        refs["best"] = {"status": REFERENCE_MISSING, "method": None, "rmse": None, "smape": None}
        return refs
    rows = _read_csv_rows(Path(path))
    found: List[Tuple[str, Dict[str, Any]]] = []
    for key, method, _display in BL_METHODS:
        row = lookup_reference_row(
            rows,
            dataset_id=dataset_id,
            target_identity=target_identity,
            method=method,
            horizon=horizon,
            seed=seed,
            mode=mode,
        )
        refs[key] = row
        if row["status"] == "FOUND" and row["rmse"] is not None:
            found.append((key, row))
    if found:
        _best_key, best_row = min(found, key=lambda item: float(item[1]["rmse"]))
        refs["best"] = {
            "status": "FOUND",
            "method": best_row["method"],
            "rmse": best_row["rmse"],
            "smape": best_row["smape"],
            "metric_space": ORIGINAL_SALES_SPACE,
        }
    else:
        refs["best"] = {"status": REFERENCE_MISSING, "method": None, "rmse": None, "smape": None}
    return refs


def _delta(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _relative_improvement(reference: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if reference is None or candidate is None:
        return None
    if float(reference) == 0.0:
        return None
    return (float(reference) - float(candidate)) / float(reference)


def _beats(candidate: Optional[float], reference: Optional[float]) -> Optional[bool]:
    if candidate is None or reference is None:
        return None
    return float(candidate) < float(reference)


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return REFERENCE_MISSING
    return f"{float(value):.{digits}f}"


def _csv_value(value: Any) -> Any:
    if value is None:
        return NA
    if isinstance(value, float) and not math.isfinite(value):
        return NA
    if isinstance(value, bool):
        return "YES" if value else "NO"
    return value


def _yes_no(value: Optional[bool]) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    return REFERENCE_MISSING


def sales_minmax(scaler: object, feature_columns: Sequence[str]) -> Tuple[Optional[float], Optional[float]]:
    features = [str(col) for col in feature_columns]
    if "sales" not in features:
        return None, None
    idx = features.index("sales")
    data_min = np.asarray(getattr(scaler, "data_min_", []), dtype=np.float64)
    data_max = np.asarray(getattr(scaler, "data_max_", []), dtype=np.float64)
    if idx >= data_min.size or idx >= data_max.size:
        return None, None
    return float(data_min[idx]), float(data_max[idx])


def original_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.size == 0 or true.shape != pred.shape:
        raise ScreeningContractError(f"RMSE arrays must match: {true.shape}, {pred.shape}")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ScreeningContractError("original RMSE requires finite arrays")
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def compute_screening_metrics(
    y_true_normalized: np.ndarray,
    y_pred_normalized: np.ndarray,
    *,
    target_scaler: object,
    feature_columns: Sequence[str],
) -> Dict[str, Any]:
    y_true_n = np.asarray(y_true_normalized, dtype=np.float64).reshape(-1)
    y_pred_n = np.asarray(y_pred_normalized, dtype=np.float64).reshape(-1)
    rmse_normalized = original_rmse(y_true_n, y_pred_n)
    smape_normalized = float(smape(y_true_n, y_pred_n))
    y_true_o = inverse_transform_sales(
        y_true_n, sales_scaler=target_scaler, feature_columns=feature_columns
    )
    y_pred_o = inverse_transform_sales(
        y_pred_n, sales_scaler=target_scaler, feature_columns=feature_columns
    )
    rmse_o = original_rmse(y_true_o, y_pred_o)
    smape_o = float(smape(y_true_o, y_pred_o))
    sales_min, sales_max = sales_minmax(target_scaler, feature_columns)
    return {
        "rmse_original": rmse_o,
        "smape_original": smape_o,
        "rmse_normalized": rmse_normalized,
        "smape_normalized": smape_normalized,
        "y_true_original": y_true_o,
        "y_pred_original": y_pred_o,
        "metric_space": ORIGINAL_SALES_SPACE,
        "sales_min": sales_min,
        "sales_max": sales_max,
    }


def assert_metric_space_for_selection(row: Mapping[str, Any]) -> None:
    space = str(row.get("metric_space") or "").strip()
    if space != ORIGINAL_SALES_SPACE:
        raise MetricSpaceMismatchError(
            target_identity=str(row.get("target_identity")),
            candidate_method=METHOD_DISPLAY,
            candidate_metric_space=space,
            reference_method="selection",
            reference_metric_space=ORIGINAL_SALES_SPACE,
        )
    if row.get("rmse_original") is None:
        raise ScreeningContractError("selection requires rmse_original")
    if row.get("rmse_normalized") is not None:
        if math.isclose(
            float(row["rmse_original"]),
            float(row["rmse_normalized"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            sales_min = row.get("sales_min")
            sales_max = row.get("sales_max")
            if (
                sales_min is not None
                and sales_max is not None
                and not math.isclose(float(sales_min), 0.0, abs_tol=1e-12)
                and not math.isclose(float(sales_max), 1.0, abs_tol=1e-12)
            ):
                raise MetricSpaceMismatchError(
                    target_identity=str(row.get("target_identity")),
                    candidate_method=METHOD_DISPLAY,
                    candidate_metric_space="ambiguous_normalized_equals_original",
                    reference_method="selection",
                    reference_metric_space=ORIGINAL_SALES_SPACE,
                )


def assert_lr_arms_share_stage_a(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_method: str = METHOD_DISPLAY,
) -> Dict[str, str]:
    if len(rows) != len(LR_GRID):
        raise ScreeningContractError(
            f"expected {len(LR_GRID)} LR arms, got {len(rows)}"
        )
    labels = [format_lr(row["learning_rate"]) for row in rows]
    if sorted(labels) != sorted(LR_GRID_LABELS):
        raise ScreeningContractError(f"LR arms must be {list(LR_GRID_LABELS)}, got {labels}")
    identities = [str(row.get("target_identity")) for row in rows]
    if len(set(identities)) != 1:
        raise ScreeningContractError(f"LR arms mix targets: {identities}")
    methods = {str(row.get("method")) for row in rows}
    if methods != {str(expected_method)}:
        raise ScreeningContractError(f"LR arms must all be {expected_method}, got {methods}")

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
        raise ScreeningContractError(f"shared Stage-A / fused-init contract failed: {flags}")
    lrs = [format_lr(row["learning_rate"]) for row in rows]
    if len(set(lrs)) != 3:
        raise ScreeningContractError("learning rate is the only allowed arm difference")
    return flags


def select_target_best_lr(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ok_rows = [dict(row) for row in rows if str(row.get("run_status", "OK")).upper() == "OK"]
    if not ok_rows:
        first = dict(rows[0]) if rows else {}
        return {
            **first,
            "best_lr": None,
            "best_rmse_original": None,
            "best_smape_original": None,
            "lr_rank_3e-5": NA,
            "lr_rank_1e-4": NA,
            "lr_rank_3e-4": NA,
            "run_status": "FAILED",
        }
    for row in ok_rows:
        assert_metric_space_for_selection(row)
    ranked = sorted(
        ok_rows,
        key=lambda row: (
            float(row["rmse_original"]),
            float(row.get("smape_original") or math.inf),
            parse_lr(row["learning_rate"]),
        ),
    )
    best_rmse = float(ranked[0]["rmse_original"])
    tied = []
    for row in ranked:
        rmse = float(row["rmse_original"])
        if best_rmse == 0.0:
            relative = 0.0 if rmse == 0.0 else math.inf
        else:
            relative = abs(rmse - best_rmse) / best_rmse
        if relative < RMSE_TIE_RELATIVE:
            tied.append(row)
    chosen = min(tied, key=lambda row: parse_lr(row["learning_rate"]))
    rank_source = sorted(
        ok_rows,
        key=lambda row: (
            float(row["rmse_original"]),
            float(row.get("smape_original") or math.inf),
            parse_lr(row["learning_rate"]),
        ),
    )
    ranks = {format_lr(row["learning_rate"]): index + 1 for index, row in enumerate(rank_source)}
    return {
        **chosen,
        "best_lr": format_lr(chosen["learning_rate"]),
        "best_rmse_original": float(chosen["rmse_original"]),
        "best_smape_original": float(chosen["smape_original"]),
        "lr_rank_3e-5": ranks.get("3e-5", NA),
        "lr_rank_1e-4": ranks.get("1e-4", NA),
        "lr_rank_3e-4": ranks.get("3e-4", NA),
    }


def _rmse_rank(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["best_rmse_original"])
            if row.get("best_rmse_original") is not None
            else math.inf,
            str(row["target_identity"]),
        ),
    )
    return {str(row["target_identity"]): index + 1 for index, row in enumerate(ordered)}


def _transfer_sort_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    def _flag(value: Optional[bool]) -> int:
        return 0 if value is True else 1

    gap = row.get("relative_improvement_vs_best_bl")
    gap_key = -float(gap) if gap is not None else math.inf
    rmse = float(row["best_rmse_original"]) if row.get("best_rmse_original") is not None else math.inf
    return (
        _flag(row.get("beats_no_tl")),
        _flag(row.get("beats_msml")),
        _flag(row.get("beats_bl4")),
        gap_key,
        rmse,
        str(row["target_identity"]),
    )


def select_dataset_primary_target(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    completed = [
        dict(row)
        for row in rows
        if str(row.get("run_status", "OK")).upper() == "OK"
        and row.get("best_rmse_original") is not None
    ]
    if not completed:
        return {
            "selected": None,
            "verdict": "NO_CLEAR_PRIMARY_TARGET",
            "evidence": "FAILED",
            "rmse_ranks": {},
            "transfer_ranks": {},
        }
    rmse_ranks = _rmse_rank(completed)
    transfer_ordered = sorted(completed, key=_transfer_sort_key)
    transfer_ranks = {
        str(row["target_identity"]): index + 1 for index, row in enumerate(transfer_ordered)
    }
    pool = list(completed)
    g1 = [row for row in pool if row.get("beats_no_tl") is True]
    used_no_tl_gate = bool(g1)
    if g1:
        pool = g1
        g2 = [row for row in pool if row.get("beats_msml") is True]
        if g2:
            pool = g2
            g3 = [row for row in pool if row.get("beats_bl4") is True]
            if g3:
                pool = g3
    selected = sorted(pool, key=_transfer_sort_key)[0]
    beats_no_tl = selected.get("beats_no_tl") is True
    beats_msml = selected.get("beats_msml") is True
    beats_bl4 = selected.get("beats_bl4") is True
    if not used_no_tl_gate:
        verdict = "NO_CLEAR_PRIMARY_TARGET"
        evidence = "NO_TARGET_BEATS_NO_TL"
    elif beats_no_tl and beats_msml:
        verdict = "PRIMARY_TARGET_STRONG_CANDIDATE"
        evidence = "STRONG"
    else:
        verdict = "PRIMARY_TARGET_MIXED_CANDIDATE"
        evidence = "MIXED"
    return {
        "selected": selected,
        "verdict": verdict,
        "evidence": evidence,
        "rmse_ranks": rmse_ranks,
        "transfer_ranks": transfer_ranks,
        "beats_no_tl": beats_no_tl,
        "beats_msml": beats_msml,
        "beats_bl4": beats_bl4,
        "beats_best_bl": selected.get("beats_best_bl") is True,
    }


def attach_reference_deltas(
    row: Mapping[str, Any],
    *,
    formal_refs: Mapping[str, Mapping[str, Any]],
    baseline_refs: Mapping[str, Any],
) -> Dict[str, Any]:
    out = dict(row)
    b_rmse = _optional_float(out.get("rmse_original"))
    no_tl = formal_refs.get("no_tl") or {}
    msml = formal_refs.get("msml") or {}
    bl4 = baseline_refs.get("bl4") or {}
    best = baseline_refs.get("best") or {}
    out["no_tl_rmse"] = no_tl.get("rmse")
    out["msml_rmse"] = msml.get("rmse")
    out["bl4_rmse"] = bl4.get("rmse")
    out["no_tl_status"] = no_tl.get("status", REFERENCE_MISSING)
    out["msml_status"] = msml.get("status", REFERENCE_MISSING)
    out["bl4_status"] = bl4.get("status", REFERENCE_MISSING)
    for key, method, _display in BL_METHODS:
        out[f"{key}_rmse"] = (baseline_refs.get(key) or {}).get("rmse")
    out["best_bl_method"] = best.get("method")
    out["best_bl_rmse"] = best.get("rmse")
    out["delta_vs_no_tl"] = _delta(b_rmse, out["no_tl_rmse"])
    out["delta_vs_msml"] = _delta(b_rmse, out["msml_rmse"])
    out["delta_vs_bl4"] = _delta(b_rmse, out["bl4_rmse"])
    out["delta_vs_best_bl"] = _delta(b_rmse, out["best_bl_rmse"])
    out["relative_improvement_vs_no_tl"] = _relative_improvement(out["no_tl_rmse"], b_rmse)
    out["relative_improvement_vs_msml"] = _relative_improvement(out["msml_rmse"], b_rmse)
    out["relative_improvement_vs_bl4"] = _relative_improvement(out["bl4_rmse"], b_rmse)
    out["relative_improvement_vs_best_bl"] = _relative_improvement(out["best_bl_rmse"], b_rmse)
    out["beats_no_tl"] = _beats(b_rmse, out["no_tl_rmse"])
    out["beats_msml"] = _beats(b_rmse, out["msml_rmse"])
    out["beats_bl4"] = _beats(b_rmse, out["bl4_rmse"])
    out["beats_best_bl"] = _beats(b_rmse, out["best_bl_rmse"])
    return out


def architecture_digest() -> str:
    payload = json.dumps(
        {
            "method": METHOD_DISPLAY,
            "arm": METHOD_ARM,
            **FUSED_TARGET_ONLY_CONTRACT,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_frozen_manifest(
    primary_rows: Sequence[Mapping[str, Any]],
    *,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    datasets: Dict[str, Any] = {}
    for dataset in SCREENING_DATASETS:
        match = next((row for row in primary_rows if row.get("dataset") == dataset), None)
        if match is None:
            datasets[dataset] = {
                "primary_target": None,
                "learning_rate": None,
                "selection_frozen": False,
                "run_status": "FAILED",
            }
            continue
        datasets[dataset] = {
            "primary_target": match.get("selected_primary_target"),
            "learning_rate": match.get("selected_learning_rate"),
            "rmse_original": match.get("rmse_original"),
            "smape_original": match.get("smape_original"),
            "beats_no_tl": match.get("beats_no_tl"),
            "beats_msml": match.get("beats_msml"),
            "beats_bl4": match.get("beats_bl4"),
            "beats_best_bl": match.get("beats_best_bl"),
            "verdict": match.get("verdict"),
            "selection_role": "development_selected",
            "selection_frozen": True,
            "target_selection_stage": TRACK,
        }
    if set(datasets) != set(SCREENING_DATASETS) or len(datasets) != 6:
        raise ScreeningContractError("frozen manifest must contain exactly D1-D6")
    return {
        "method": METHOD_DISPLAY,
        "selection_stage": TRACK,
        "metric_space": ORIGINAL_SALES_SPACE,
        "horizon_used_for_screening": SCREENING_HORIZON,
        "seed_used_for_screening": SCREENING_SEED,
        "epochs": SCREENING_EPOCHS,
        "lr_grid": list(LR_GRID),
        "lr_grid_labels": list(LR_GRID_LABELS),
        "stage_a_learning_rate": STAGE_A_LEARNING_RATE,
        "selection_frozen": True,
        "targets_frozen_after_screening": True,
        "auto_start_robustness": False,
        "architecture": fused_target_only_architecture(),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "datasets": datasets,
    }


def format_dataset_terminal(
    dataset: str,
    run_rows: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Any],
) -> str:
    lines = [
        "================================================================================",
        f"{dataset} SCREENING",
        "================================================================================",
        "",
        f"{'TARGET':<22}{'LR':<8}{'RMSE_o':<10}{'NoTL':<10}{'MSML':<10}{'BL4':<10}{'BEST_BL':<10}",
        "-------------------------------------------------------------------------",
    ]
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in run_rows:
        grouped.setdefault(str(row["target_identity"]), []).append(row)
    for identity, rows in grouped.items():
        for row in sorted(rows, key=lambda item: parse_lr(item["learning_rate"])):
            lines.append(
                f"{identity:<22}{format_lr(row['learning_rate']):<8}"
                f"{_fmt(row.get('rmse_original')):<10}"
                f"{_fmt(row.get('no_tl_rmse')):<10}"
                f"{_fmt(row.get('msml_rmse')):<10}"
                f"{_fmt(row.get('bl4_rmse')):<10}"
                f"{_fmt(row.get('best_bl_rmse')):<10}"
            )
        lines.append("")
    selected = primary.get("selected") or {}
    lines.extend(
        [
            "-------------------------------------------------------------------------",
            f"PRIMARY TARGET = {selected.get('target_identity', REFERENCE_MISSING)}",
            f"PRIMARY LR     = {selected.get('best_lr', REFERENCE_MISSING)}",
            "",
            f"BEATS No-TL    = {_yes_no(primary.get('beats_no_tl'))}",
            f"BEATS MSML     = {_yes_no(primary.get('beats_msml'))}",
            f"BEATS BL4      = {_yes_no(primary.get('beats_bl4'))}",
            f"BEATS BEST BL  = {_yes_no(primary.get('beats_best_bl'))}",
            "",
            "SELECTION_ROLE=development_selected",
            f"VERDICT={primary.get('verdict', 'NO_CLEAR_PRIMARY_TARGET')}",
            "================================================================================",
        ]
    )
    return "\n".join(lines) + "\n"


def format_global_summary(primary_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "===========================================================================================================",
        "D1-D6 PRIMARY TARGET × LR SCREENING",
        "===========================================================================================================",
        "",
        f"{'DATASET':<10}{'PRIMARY TARGET':<22}{'LR':<8}{'RMSE_o':<10}{'NoTL':<10}{'MSML':<10}{'BL4':<10}{'BEST_BL':<10}{'BEST_BL_METHOD'}",
        "-----------------------------------------------------------------------------------------------------------",
    ]
    beats = {"no_tl": 0, "msml": 0, "bl4": 0, "best_bl": 0}
    for dataset in SCREENING_DATASETS:
        row = next((item for item in primary_rows if item.get("dataset") == dataset), {})
        lines.append(
            f"{dataset:<10}{str(row.get('selected_primary_target') or REFERENCE_MISSING):<22}"
            f"{str(row.get('selected_learning_rate') or REFERENCE_MISSING):<8}"
            f"{_fmt(row.get('rmse_original')):<10}"
            f"{_fmt(row.get('no_tl_rmse')):<10}"
            f"{_fmt(row.get('msml_rmse')):<10}"
            f"{_fmt(row.get('bl4_rmse')):<10}"
            f"{_fmt(row.get('best_bl_rmse')):<10}"
            f"{row.get('best_bl_method') or REFERENCE_MISSING}"
        )
        beats["no_tl"] += int(row.get("beats_no_tl") is True)
        beats["msml"] += int(row.get("beats_msml") is True)
        beats["bl4"] += int(row.get("beats_bl4") is True)
        beats["best_bl"] += int(row.get("beats_best_bl") is True)
    lines.extend(
        [
            "-----------------------------------------------------------------------------------------------------------",
            "",
            f"BEATS_NO_TL     = {beats['no_tl']}/6",
            f"BEATS_MSML      = {beats['msml']}/6",
            f"BEATS_BL4       = {beats['bl4']}/6",
            f"BEATS_BEST_BL   = {beats['best_bl']}/6",
            "",
            f"TARGET_SELECTION_STAGE={TRACK}",
            "TARGETS_FROZEN_AFTER_SCREENING=YES",
            f"METRIC_SPACE={ORIGINAL_SALES_SPACE}",
            "===========================================================================================================",
        ]
    )
    return "\n".join(lines) + "\n"


def write_screening_outputs(
    run_dir: Path,
    *,
    all_runs: Sequence[Mapping[str, Any]],
    target_best: Sequence[Mapping[str, Any]],
    primary_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_csv = results_dir / "screening_all_runs.csv"
    best_csv = results_dir / "screening_target_best_lr.csv"
    primary_csv = results_dir / "screening_dataset_primary_targets.csv"
    comparison_csv = results_dir / "screening_reference_comparison.csv"
    provenance_json = results_dir / "screening_provenance.json"
    manifest_json = results_dir / "primary_target_lr_selection_manifest.json"

    def _dump(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        frame = pd.DataFrame([{key: _csv_value(value) for key, value in row.items()} for row in rows])
        frame.to_csv(path, index=False, encoding="utf-8")

    _dump(all_csv, all_runs)
    _dump(best_csv, target_best)
    _dump(primary_csv, primary_rows)
    comparison_rows = [
        {
            "dataset": row.get("dataset"),
            "target_identity": row.get("target_identity"),
            "learning_rate": row.get("learning_rate_label") or row.get("best_lr"),
            "rmse_original": row.get("rmse_original") or row.get("best_rmse_original"),
            "no_tl_rmse": row.get("no_tl_rmse"),
            "msml_rmse": row.get("msml_rmse"),
            "bl4_rmse": row.get("bl4_rmse"),
            "best_bl_rmse": row.get("best_bl_rmse"),
            "best_bl_method": row.get("best_bl_method"),
            "delta_vs_no_tl": row.get("delta_vs_no_tl"),
            "delta_vs_msml": row.get("delta_vs_msml"),
            "delta_vs_bl4": row.get("delta_vs_bl4"),
            "delta_vs_best_bl": row.get("delta_vs_best_bl"),
            "no_tl_status": row.get("no_tl_status"),
            "msml_status": row.get("msml_status"),
            "metric_space": ORIGINAL_SALES_SPACE,
        }
        for row in all_runs
    ]
    _dump(comparison_csv, comparison_rows)
    provenance_json.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    manifest_json.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {
        "all_runs_csv": all_csv,
        "target_best_csv": best_csv,
        "primary_csv": primary_csv,
        "comparison_csv": comparison_csv,
        "provenance_json": provenance_json,
        "manifest_json": manifest_json,
    }


def write_frozen_config_manifest(project_root: Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(project_root) / "configs" / "primary_target_lr_selection_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def assert_runner_does_not_launch_robustness(source: str) -> None:
    lowered = source.lower()
    if "auto_start_robustness" in lowered and "true" in lowered.split("auto_start_robustness", 1)[-1][:40]:
        raise ScreeningContractError("screening runner must not auto-start robustness")
    for token in ("horizon in (1, 3, 5)", "seeds = (42, 43, 44)", "run_robustness_experiment"):
        if token in source:
            raise ScreeningContractError(f"screening runner contains forbidden robustness launch: {token}")
