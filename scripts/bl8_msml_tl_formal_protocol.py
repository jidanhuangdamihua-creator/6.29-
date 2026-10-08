#!/usr/bin/env python3
"""BL8 MSML-TL formal 60-cell identities, audit helpers, and comparison math.

This module does not train models and does not write paper freeze workbooks.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_original_scale_metrics
from src.transfer_methods.msml_tl import (
    BL8_FUSED_PARAMETER_LAYERS,
    BL8_FROZEN_TARGET_LAYERS,
    BL8_TARGET_TRAINABLE_PARAMETER_LAYERS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "formal_bl8_msml_tl"
SITA_FREEZE_XLSX = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/"
    "MS_KATL_Paper_Data_Freeze_v1_4_STATS_FROZEN_BL6_BL7_INTEGRATED.xlsx"
)
SITA_SHEET = "03_V3_Raw_5Seed"

METHOD = "MSML-TL"
METHOD_ID = "BL8_MSML_TL"
MODE = "without"
DATASETS = ("D1", "D4", "D5", "D6")
EXECUTION_ORDER = ("D4", "D5", "D6", "D1")
TARGETS = {
    "D1": "1/10",
    "D4": "166/258",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}
ENTITY_IDS = {
    "D1": "1_10",
    "D4": "166_258",
    "D5": "48_1159414",
    "D6": "CA_1_FOODS_3_080",
}
HORIZONS = (1, 3, 5)
SEEDS = (42, 43, 44, 45, 46)
EXPECTED_CELLS = 60
EXPECTED_AGGREGATES = 12
PARITY_TOLERANCE = 1e-10
WEIGHT_SUM_TOLERANCE = 1e-6
DRY_RUN_D4_H1_S42 = {
    "rmse": 0.564157,
    "smape": 42.287281,
    "n": 170,
}
FORBIDDEN_D4_TARGETS = {"166/311", "166/313", "166/432", "166/433", "166_311", "166_313", "166_432", "166_433"}
CODE_IDENTITY_PATHS = (
    "src/transfer_methods/msml_tl.py",
    "src/models/cnn_model.py",
    "src/source_selection/source_selector.py",
    "src/experiment/experiment_runner.py",
    "src/utils/entity_experiment.py",
    "src/evaluation/metrics.py",
    "scripts/bl8_msml_tl_formal_protocol.py",
    "scripts/run_bl8_msml_tl_formal_60cell.py",
)
REQUIRED_OUTPUT_FILES = (
    "BL8_MSML_TL_FORMAL_60CELL_RESULTS.csv",
    "BL8_MSML_TL_FORMAL_5SEED_AGGREGATED.csv",
    "BL8_SITA_VS_MSML_TL_12CONDITION_COMPARISON.csv",
    "BL8_SITA_VS_MSML_TL_60CELL_PAIRED_COMPARISON.csv",
    "BL8_MSML_TL_FORMAL_60CELL_COMPLETENESS_AUDIT.md",
    "BL8_MSML_TL_FORMAL_EXECUTION_MANIFEST.json",
)


class BL8FormalError(RuntimeError):
    """Fail-closed BL8 formal contract error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(root: Path = ROOT) -> dict[str, str]:
    def run(args: list[str]) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

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
    }
    for relative in CODE_IDENTITY_PATHS:
        path = root / relative
        payload[f"sha256:{relative}"] = sha256_file(path)
    return payload


def expected_cell_keys() -> list[tuple[str, str, int, int, str, str]]:
    return [
        (dataset, TARGETS[dataset], horizon, seed, MODE, METHOD)
        for dataset in EXECUTION_ORDER
        for horizon in HORIZONS
        for seed in SEEDS
    ]


def canonical_key(dataset: str, target: str, horizon: int, seed: int, mode: str = MODE, method: str = METHOD) -> str:
    return f"{dataset}|{target}|{int(horizon)}|{int(seed)}|{mode}|{method}"


def cell_dirname(dataset: str, target: str, horizon: int, seed: int) -> str:
    safe_target = str(target).replace("/", "_")
    return f"{dataset}__{safe_target}__H{int(horizon)}__S{int(seed)}__{MODE}__{METHOD}"


def assert_allowed_matrix_cell(dataset: str, target: str, horizon: int, seed: int, mode: str, method: str) -> None:
    if dataset not in DATASETS:
        raise BL8FormalError(f"unexpected dataset: {dataset}")
    if str(target) != TARGETS[dataset]:
        raise BL8FormalError(f"unexpected target for {dataset}: {target}")
    if str(target) in FORBIDDEN_D4_TARGETS or (dataset == "D4" and str(target) != "166/258"):
        raise BL8FormalError(f"forbidden D4 target: {target}")
    if int(horizon) not in HORIZONS:
        raise BL8FormalError(f"unexpected horizon: {horizon}")
    if int(seed) not in SEEDS:
        raise BL8FormalError(f"unexpected seed: {seed}")
    if str(mode) != MODE:
        raise BL8FormalError(f"unexpected mode: {mode}")
    if str(method) != METHOD:
        raise BL8FormalError(f"unexpected method: {method}")


def _as_layer_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text.replace("'", '"'))
                return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise BL8FormalError(f"unsupported layer identity: {value!r}")


def _format_source_key(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "/".join(str(part) for part in value)
    return str(value)


def relative_improvement_pct(msml_value: float, sita_value: float) -> float:
    msml_value = float(msml_value)
    sita_value = float(sita_value)
    if not math.isfinite(msml_value) or msml_value == 0:
        raise BL8FormalError(f"MSML reference must be finite and non-zero: {msml_value}")
    return (msml_value - sita_value) / msml_value * 100.0


def winner_label(msml_value: float, sita_value: float) -> str:
    if float(sita_value) < float(msml_value):
        return "SITA"
    if float(msml_value) < float(sita_value):
        return "MSML-TL"
    return "TIE"


def recompute_original_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    metrics = compute_original_scale_metrics(np.asarray(y_true), np.asarray(y_pred))
    return {"rmse": float(metrics["rmse"]), "smape": float(metrics["smape"]), "n": int(len(np.asarray(y_true).reshape(-1)))}


def mechanism_failures(manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if not bool(manifest.get("formal_bl8", False)):
        failures.append("formal_bl8!=true")
    if int(manifest.get("source_count_requested", -1)) != 3:
        failures.append("source_count_requested!=3")
    if int(manifest.get("source_count_selected", -1)) != 3:
        failures.append("source_count_selected!=3")
    if int(manifest.get("source_count_successful", -1)) != 3:
        failures.append("source_count_successful!=3")
    if int(manifest.get("source_failure_count", -1)) != 0:
        failures.append("source_failure_count!=0")
    if str(manifest.get("weight_mode")) != "inverse_distance":
        failures.append("weight_mode!=inverse_distance")
    if str(manifest.get("selection_authority")) != "shared_protocol":
        failures.append("sealed_knn_unavailable")
    fused = _as_layer_list(manifest.get("fused_layers"))
    frozen = _as_layer_list(manifest.get("frozen_layers"))
    trainable = _as_layer_list(manifest.get("target_trainable_parameter_layers"))
    if fused != list(BL8_FUSED_PARAMETER_LAYERS):
        failures.append(f"fused_layers={fused}")
    if frozen != list(BL8_FROZEN_TARGET_LAYERS):
        failures.append(f"frozen_layers={frozen}")
    if trainable != list(BL8_TARGET_TRAINABLE_PARAMETER_LAYERS):
        failures.append(f"trainable_layers={trainable}")
    if "conv3" in fused:
        failures.append("conv3_fused")
    if "conv3" in frozen:
        failures.append("conv3_frozen")
    if bool(manifest.get("rfe", True)):
        failures.append("rfe_not_off")
    weight_sum = float(manifest.get("weight_sum", float("nan")))
    if not math.isfinite(weight_sum) or abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        failures.append(f"weight_sum={weight_sum}")
    if bool(manifest.get("target_test_used_for_fit", True)):
        failures.append("target_test_used_for_fit")
    if bool(manifest.get("future_leakage_detected", True)):
        failures.append("future_leakage_detected")
    return failures


def completeness_audit(rows: pd.DataFrame) -> dict[str, Any]:
    expected = {canonical_key(*key) for key in expected_cell_keys()}
    if rows.empty:
        actual_keys: list[str] = []
    else:
        actual_keys = [
            canonical_key(
                str(row["dataset"]),
                str(row["target"]),
                int(row["horizon"]),
                int(row["seed"]),
                str(row["mode"]),
                str(row["method"]),
            )
            for row in rows.to_dict(orient="records")
        ]
    actual_set = set(actual_keys)
    counts = pd.Series(actual_keys).value_counts() if actual_keys else pd.Series(dtype=int)
    duplicates = sorted(str(key) for key, count in counts.items() if int(count) > 1)
    unexpected = sorted(actual_set - expected)
    failed = []
    passed = []
    if not rows.empty:
        for rec in rows.to_dict(orient="records"):
            key = canonical_key(
                str(rec["dataset"]),
                str(rec["target"]),
                int(rec["horizon"]),
                int(rec["seed"]),
                str(rec.get("mode", MODE)),
                str(rec.get("method", METHOD)),
            )
            if str(rec.get("cell_status", "PASS")).upper() == "PASS":
                passed.append(key)
            else:
                failed.append(key)
    missing = sorted(expected - actual_set)
    passed_set = set(passed)
    per_dataset = {}
    for dataset in DATASETS:
        dataset_keys = {key for key in expected if key.startswith(f"{dataset}|")}
        per_dataset[dataset] = int(sum(1 for key in passed_set if key in dataset_keys))
    return {
        "expected_cells": EXPECTED_CELLS,
        "actual_cells": int(len(passed_set)),
        "unique_cells": int(len(passed_set)),
        "duplicate_cells": int(len(duplicates)),
        "duplicate_keys": duplicates,
        "missing_cells": int(len(missing)),
        "missing_keys": missing,
        "unexpected_cells": int(len(unexpected)),
        "unexpected_keys": unexpected,
        "failed_cells": int(len(failed)),
        "failed_keys": sorted(failed),
        "per_dataset": per_dataset,
    }


def aggregate_five_seed(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "target", "horizon", "seed", "rmse", "smape"}
    missing = required - set(rows.columns)
    if missing:
        raise BL8FormalError(f"aggregation missing columns: {sorted(missing)}")
    records = []
    for dataset, target, horizon in (
        (dataset, TARGETS[dataset], horizon) for dataset in DATASETS for horizon in HORIZONS
    ):
        group = rows[
            (rows["dataset"].astype(str) == dataset)
            & (rows["target"].astype(str) == target)
            & (rows["horizon"].astype(int) == int(horizon))
        ].copy()
        seeds = sorted(int(value) for value in group["seed"].tolist())
        if seeds != list(SEEDS):
            raise BL8FormalError(f"{dataset}/{target}/H{horizon} seeds={seeds} expected={list(SEEDS)}")
        rmse_values = [float(value) for value in group["rmse"].tolist()]
        smape_values = [float(value) for value in group["smape"].tolist()]
        records.append(
            {
                "dataset": dataset,
                "target": target,
                "horizon": int(horizon),
                "mode": MODE,
                "method": METHOD,
                "n_seeds": 5,
                "rmse_mean": float(statistics.mean(rmse_values)),
                "rmse_std": float(statistics.stdev(rmse_values)),
                "smape_mean": float(statistics.mean(smape_values)),
                "smape_std": float(statistics.stdev(smape_values)),
                "n_mean": float(statistics.mean(float(value) for value in group["n"].tolist())) if "n" in group.columns else float("nan"),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != EXPECTED_AGGREGATES:
        raise BL8FormalError(f"expected {EXPECTED_AGGREGATES} aggregates, got {len(frame)}")
    if not (frame["n_seeds"] == 5).all():
        raise BL8FormalError("not all aggregates have 5 seeds")
    return frame


def load_sita_authority(path: Path = SITA_FREEZE_XLSX) -> pd.DataFrame:
    if not path.exists():
        raise BL8FormalError(f"SITA freeze workbook missing: {path}")
    frame = pd.read_excel(path, sheet_name=SITA_SHEET)
    frame.columns = [str(column).strip() for column in frame.columns]
    method_col = "method" if "method" in frame.columns else None
    if method_col is None:
        raise BL8FormalError(f"{SITA_SHEET} missing method column")
    mapped = frame.copy()
    mapped["method"] = mapped[method_col].astype(str).replace({"MS-KATL-V3": "SITA", "SITA": "SITA"})
    mapped = mapped[mapped["method"].astype(str) == "SITA"].copy()
    rmse_col = "rmse_original" if "rmse_original" in mapped.columns else "rmse"
    smape_col = "smape_original" if "smape_original" in mapped.columns else "smape"
    target_col = "target" if "target" in mapped.columns else "target_entity_key"
    mapped["dataset"] = mapped["dataset"].astype(str)
    mapped["target"] = mapped[target_col].astype(str)
    mapped["horizon"] = mapped["horizon"].astype(int)
    mapped["seed"] = mapped["seed"].astype(int)
    mapped["rmse"] = mapped[rmse_col].astype(float)
    mapped["smape"] = mapped[smape_col].astype(float)
    keep = mapped[["dataset", "target", "horizon", "seed", "rmse", "smape"]].copy()
    expected = {(dataset, TARGETS[dataset], horizon, seed) for dataset in DATASETS for horizon in HORIZONS for seed in SEEDS}
    actual = set(zip(keep["dataset"], keep["target"], keep["horizon"], keep["seed"]))
    if actual != expected and "mode" in mapped.columns:
        filtered = mapped[mapped["mode"].astype(str).isin(["without", "without_information_sharing", MODE])]
        keep = filtered[["dataset", "target", "horizon", "seed", "rmse", "smape"]].copy()
        actual = set(zip(keep["dataset"], keep["target"], keep["horizon"], keep["seed"]))
    if actual != expected:
        raise BL8FormalError(
            "SITA authority is not the exact 60-cell without matrix: "
            f"missing={sorted(expected - actual)[:8]} unexpected={sorted(actual - expected)[:8]}"
        )
    return keep


def compare_12_conditions(msml_agg: pd.DataFrame, sita_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sita_agg = aggregate_five_seed(sita_rows.rename(columns={"rmse": "rmse", "smape": "smape"}).assign(mode=MODE, method="SITA"))
    records = []
    for _, msml in msml_agg.iterrows():
        sita = sita_agg[
            (sita_agg["dataset"] == msml["dataset"])
            & (sita_agg["target"] == msml["target"])
            & (sita_agg["horizon"] == msml["horizon"])
        ].iloc[0]
        ri_rmse = relative_improvement_pct(float(msml["rmse_mean"]), float(sita["rmse_mean"]))
        ri_smape = relative_improvement_pct(float(msml["smape_mean"]), float(sita["smape_mean"]))
        records.append(
            {
                "dataset": msml["dataset"],
                "target": msml["target"],
                "horizon": int(msml["horizon"]),
                "msml_rmse_mean": float(msml["rmse_mean"]),
                "msml_rmse_std": float(msml["rmse_std"]),
                "sita_rmse_mean": float(sita["rmse_mean"]),
                "sita_rmse_std": float(sita["rmse_std"]),
                "rmse_relative_improvement_pct": ri_rmse,
                "rmse_winner": winner_label(float(msml["rmse_mean"]), float(sita["rmse_mean"])),
                "msml_smape_mean": float(msml["smape_mean"]),
                "msml_smape_std": float(msml["smape_std"]),
                "sita_smape_mean": float(sita["smape_mean"]),
                "sita_smape_std": float(sita["smape_std"]),
                "smape_relative_improvement_pct": ri_smape,
                "smape_winner": winner_label(float(msml["smape_mean"]), float(sita["smape_mean"])),
            }
        )
    frame = pd.DataFrame.from_records(records)
    summary = {
        "sita_rmse_wins_12": int((frame["rmse_winner"] == "SITA").sum()),
        "sita_smape_wins_12": int((frame["smape_winner"] == "SITA").sum()),
        "mean_ri_rmse_pct": float(frame["rmse_relative_improvement_pct"].mean()),
        "median_ri_rmse_pct": float(frame["rmse_relative_improvement_pct"].median()),
        "mean_ri_smape_pct": float(frame["smape_relative_improvement_pct"].mean()),
        "median_ri_smape_pct": float(frame["smape_relative_improvement_pct"].median()),
    }
    return frame, summary


def compare_60_paired(msml_rows: pd.DataFrame, sita_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = msml_rows.merge(
        sita_rows,
        on=["dataset", "target", "horizon", "seed"],
        suffixes=("_msml", "_sita"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_CELLS:
        raise BL8FormalError(f"paired comparison expected 60 rows, got {len(merged)}")
    merged["delta_rmse"] = merged["rmse_msml"] - merged["rmse_sita"]
    merged["delta_smape"] = merged["smape_msml"] - merged["smape_sita"]
    merged["relative_improvement_rmse_pct"] = [
        relative_improvement_pct(float(msml), float(sita))
        for msml, sita in zip(merged["rmse_msml"], merged["rmse_sita"])
    ]
    merged["relative_improvement_smape_pct"] = [
        relative_improvement_pct(float(msml), float(sita))
        for msml, sita in zip(merged["smape_msml"], merged["smape_sita"])
    ]
    merged["rmse_winner"] = [
        winner_label(float(msml), float(sita))
        for msml, sita in zip(merged["rmse_msml"], merged["rmse_sita"])
    ]
    merged["smape_winner"] = [
        winner_label(float(msml), float(sita))
        for msml, sita in zip(merged["smape_msml"], merged["smape_sita"])
    ]
    summary = {
        "sita_rmse_wins_60": int((merged["rmse_winner"] == "SITA").sum()),
        "sita_smape_wins_60": int((merged["smape_winner"] == "SITA").sum()),
    }
    return merged, summary


def timestamp_seed_invariance(prediction_frames: Mapping[tuple[str, str, int], Mapping[int, list[str]]]) -> list[str]:
    failures = []
    for condition, by_seed in prediction_frames.items():
        if set(by_seed) != set(SEEDS):
            failures.append(f"{condition} seeds={sorted(by_seed)}")
            continue
        reference = by_seed[SEEDS[0]]
        for seed in SEEDS[1:]:
            if by_seed[seed] != reference:
                failures.append(f"{condition} timestamps differ for seed={seed}")
    return failures


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
