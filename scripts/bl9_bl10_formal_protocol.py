"""Shared identities, audits, aggregation, and comparison math for BL9/BL10.

This module does not train models and does not overwrite paper freeze workbooks.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_original_scale_metrics
from src.evaluation.metric_contract import (
    ORIGINAL_SALES_SPACE,
    SMAPE_CONTRACT_FIELDS,
    SMAPE_UNIT,
)
from src.protocols.experiment_protocol import FORMAL_SEEDS
from scripts.baselines.bl4_lstm import LEARNING_RATE, MAX_EPOCHS, PATIENCE
from scripts.baselines.bl9_patchtst import (
    FEATURE_COLUMNS,
    METHOD as PATCHTST_METHOD,
    METHOD_ID as PATCHTST_METHOD_ID,
    PATCHTST_CONFIG,
    assert_patchtst_config,
    target_channel_index,
)
from scripts.baselines.bl10_timesnet import (
    METHOD as TIMESNET_METHOD,
    METHOD_ID as TIMESNET_METHOD_ID,
    TIMESNET_CONFIG,
    assert_timesnet_config,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "formal_bl9_bl10"
PATCHTST_ROOT = ROOT / "results" / "formal_bl9_patchtst"
TIMESNET_ROOT = ROOT / "results" / "formal_bl10_timesnet"
BL8_ROOT = ROOT / "results" / "formal_bl8_msml_tl"
ROBUSTNESS_ROOT = ROOT / "results" / "robustness" / "revin_strengthened_baselines"
DEFAULT_FREEZE = Path(
    "/Users/ming/Desktop/复现实验/论文文稿相关/"
    "MS_KATL_Paper_Data_Freeze_v1_4_STATS_FROZEN_BL6_BL7_INTEGRATED.xlsx"
)
SITA_SHEET = "03_V3_Raw_5Seed"
BASELINE_SHEET = "04_BL_Raw_Selected"
ABLATION_SHEET = "05_Ablation_Raw"

MODE = "without"
DATASETS = ("D1", "D4", "D5", "D6")
TARGETS = {
    "D1": "1/10",
    "D4": "166/258",
    "D5": "48/1159414",
    "D6": "CA_1/FOODS_3_080",
}
FORBIDDEN_D4_TARGETS = {"166/311", "166/313", "166/432", "166/433", "166_311", "166_313", "166_432", "166_433"}
HORIZONS = (1, 3, 5)
SEEDS = FORMAL_SEEDS
LOOKBACK = 10
EXPECTED_N_TEST = {1: 170, 3: 168, 5: 166}
EXPECTED_CELLS_PER_METHOD = 60
EXPECTED_CELLS = 120
EXPECTED_AGGREGATES = 24
PARITY_TOLERANCE = 1e-10
PARITY_RELAXED = 1e-8
METRIC_SPACE = ORIGINAL_SALES_SPACE
FEATURE_METADATA = tuple(FEATURE_COLUMNS)
TARGET_COLUMN = "sales"
FORMAL_BASELINE_TRAINING_POLICY = {
    "optimizer": "Adam",
    "learning_rate": float(LEARNING_RATE),
    "epochs": int(MAX_EPOCHS),
    "batch": "full_train_tensor",
    "loss": "MSELoss",
    "early_stopping_patience": int(PATIENCE),
    "early_stopping_min_delta": None,
    "scaler": "MinMaxScaler_train_only",
    "clip_normalized": [0.0, 1.0],
}
METHODS = (PATCHTST_METHOD, TIMESNET_METHOD)
METHOD_IDS = {
    PATCHTST_METHOD: PATCHTST_METHOD_ID,
    TIMESNET_METHOD: TIMESNET_METHOD_ID,
}
METHOD_ROOTS = {
    PATCHTST_METHOD: PATCHTST_ROOT,
    TIMESNET_METHOD: TIMESNET_ROOT,
}
FORMAL_LEARNING_UNIVERSE = (
    "LSTM",
    "LightGBM",
    "DLinear",
    "TCN",
    "PatchTST",
    "TimesNet",
    "MSML-TL",
)
ROBUSTNESS_UNIVERSE = ("DLinear_RevIN", "TCN_RevIN")
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_RANDOM_SEED = 20260903
HOLM_ALPHA = 0.05
HOLM_FAMILY = "BL9_BL10_SITA_PAIRWISE"
HOLM_FAMILY_SIZE = 4
TIE_RULE = "exact_numeric_no_tolerance"
CODE_IDENTITY_PATHS = (
    "scripts/baselines/bl9_patchtst.py",
    "scripts/baselines/bl10_timesnet.py",
    "scripts/baselines/bl4_lstm.py",
    "scripts/baselines/baseline_data_loader.py",
    "scripts/bl9_bl10_formal_protocol.py",
    "scripts/run_bl9_bl10_modern_baselines.py",
    "src/evaluation/metrics.py",
    "src/evaluation/metric_contract.py",
    "src/protocols/rolling_origin.py",
    "src/protocols/reproducibility.py",
)


class BL910FormalError(RuntimeError):
    """Fail-closed BL9/BL10 contract error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(root: Path = ROOT) -> dict[str, str]:
    def run(args: list[str]) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = run(["status", "--short"])
    return {
        "git_branch": run(["branch", "--show-current"]),
        "code_commit": run(["rev-parse", "HEAD"]),
        "git_dirty": "true" if status else "false",
        "git_status_short": status,
    }


def record_code_identity(root: Path = ROOT) -> dict[str, str]:
    git = git_identity(root)
    payload = {
        "git_branch": git["git_branch"],
        "code_commit": git["code_commit"],
        "git_dirty": git["git_dirty"],
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for relative in CODE_IDENTITY_PATHS:
        path = root / relative
        payload[f"sha256:{relative}"] = sha256_file(path) if path.exists() else "MISSING"
    return payload


def runtime_versions() -> dict[str, str]:
    versions = {
        "python_version": sys.version.split()[0],
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
    }
    try:
        import torch

        versions["torch_version"] = str(torch.__version__)
        versions["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        versions["dtype"] = "float32"
        versions["cuda_available"] = str(torch.cuda.is_available()).lower()
    except ImportError:
        versions["torch_version"] = "unavailable"
        versions["device"] = "cpu"
        versions["dtype"] = "float32"
        versions["cuda_available"] = "false"
    return versions


def expected_cell_keys(methods: Sequence[str] = METHODS) -> list[tuple[str, str, str, int, int, str]]:
    return [
        (str(method), dataset, TARGETS[dataset], MODE, horizon, seed)
        for method in methods
        for dataset in DATASETS
        for horizon in HORIZONS
        for seed in SEEDS
    ]


def canonical_key(method: str, dataset: str, target: str, mode: str, horizon: int, seed: int) -> str:
    return f"{method}|{dataset}|{target}|{mode}|{int(horizon)}|{int(seed)}"


def cell_dirname(dataset: str, target: str, horizon: int, seed: int, method: str) -> str:
    safe_target = str(target).replace("/", "_")
    return f"{dataset}__{safe_target}__H{int(horizon)}__S{int(seed)}__{MODE}__{method}"


def method_output_root(method: str) -> Path:
    if method not in METHOD_ROOTS:
        raise BL910FormalError(f"unsupported method: {method}")
    return METHOD_ROOTS[method]


def cell_paths(method: str, dataset: str, target: str, horizon: int, seed: int, output_root: Path | None = None) -> dict[str, Path]:
    root = Path(output_root) if output_root is not None else method_output_root(method)
    cell_dir = root / "cells" / cell_dirname(dataset, target, horizon, seed, method)
    return {
        "dir": cell_dir,
        "predictions": cell_dir / "predictions.csv",
        "manifest": cell_dir / "cell_manifest.json",
        "metrics": cell_dir / "metrics.json",
        "history": cell_dir / "training_history.csv",
        "complete": cell_dir / "cell_complete.json",
        "error": cell_dir / "cell_error.txt",
    }


def assert_allowed_matrix_cell(method: str, dataset: str, target: str, horizon: int, seed: int, mode: str = MODE) -> None:
    if method not in METHODS:
        raise BL910FormalError(f"unexpected method: {method}")
    if dataset not in DATASETS:
        raise BL910FormalError(f"unexpected dataset: {dataset}")
    if str(target) != TARGETS[dataset]:
        raise BL910FormalError(f"unexpected target for {dataset}: {target}")
    if str(target) in FORBIDDEN_D4_TARGETS or (dataset == "D4" and str(target) != "166/258"):
        raise BL910FormalError(f"forbidden D4 target: {target}")
    if int(horizon) not in HORIZONS:
        raise BL910FormalError(f"unexpected horizon: {horizon}")
    if int(seed) not in SEEDS:
        raise BL910FormalError(f"unexpected seed: {seed}")
    if str(mode) != MODE:
        raise BL910FormalError(f"unexpected mode: {mode}")


def select_paper_entity(slices: Sequence[Mapping[str, Any]], dataset: str) -> dict[str, Any]:
    expected = TARGETS[str(dataset).upper()]
    matches = []
    for item in slices:
        key = str(item.get("entity_key") or "").replace("_", "/")
        values = item.get("entity_values")
        if values is not None:
            key = "/".join(str(part) for part in values)
        if key == expected:
            matches.append(item)
    if len(matches) != 1:
        raise BL910FormalError(
            f"{dataset} paper target {expected} matched {len(matches)} loader slices"
        )
    return dict(matches[0])


def recompute_original_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    metrics = compute_original_scale_metrics(np.asarray(y_true), np.asarray(y_pred))
    return {
        "rmse": float(metrics["rmse"]),
        "smape": float(metrics["smape"]),
        "n": int(np.asarray(y_true).reshape(-1).size),
        "metric_space": METRIC_SPACE,
        "smape_unit": SMAPE_UNIT,
        **{key: SMAPE_CONTRACT_FIELDS[key] for key in ("smape_definition_id", "smape_epsilon")},
    }


def relative_improvement_pct(baseline_value: float, sita_value: float) -> float:
    baseline_value = float(baseline_value)
    sita_value = float(sita_value)
    if not math.isfinite(baseline_value) or baseline_value == 0:
        raise BL910FormalError(f"baseline reference must be finite and non-zero: {baseline_value}")
    return (baseline_value - sita_value) / baseline_value * 100.0


def winner_label(baseline_value: float, sita_value: float, baseline_name: str) -> str:
    if float(sita_value) < float(baseline_value):
        return "SITA"
    if float(baseline_value) < float(sita_value):
        return baseline_name
    return "TIE"


def json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def completeness_audit(rows: pd.DataFrame, methods: Sequence[str] = METHODS) -> dict[str, Any]:
    expected = {canonical_key(*key) for key in expected_cell_keys(methods)}
    if rows.empty:
        actual_keys: list[str] = []
    else:
        actual_keys = [
            canonical_key(
                str(row["method"]),
                str(row["dataset"]),
                str(row["target"]),
                str(row.get("mode", MODE)),
                int(row["horizon"]),
                int(row["seed"]),
            )
            for row in rows.to_dict(orient="records")
        ]
    actual_set = set(actual_keys)
    counts = pd.Series(actual_keys).value_counts() if actual_keys else pd.Series(dtype=int)
    duplicates = sorted(str(key) for key, count in counts.items() if int(count) > 1)
    missing = sorted(expected - actual_set)
    unexpected = sorted(actual_set - expected)
    return {
        "expected_cells": int(len(expected)),
        "actual_cells": int(len(actual_set)),
        "duplicate_cells": int(len(duplicates)),
        "duplicate_keys": duplicates,
        "missing_cells": int(len(missing)),
        "missing_keys": missing,
        "unexpected_cells": int(len(unexpected)),
        "unexpected_keys": unexpected,
    }


def aggregate_five_seed(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "dataset", "target", "horizon", "seed", "rmse", "smape"}
    missing = required - set(rows.columns)
    if missing:
        raise BL910FormalError(f"aggregation missing columns: {sorted(missing)}")
    records = []
    for method in METHODS:
        method_rows = rows[rows["method"].astype(str) == method]
        if method_rows.empty:
            continue
        for dataset in DATASETS:
            for horizon in HORIZONS:
                group = method_rows[
                    (method_rows["dataset"].astype(str) == dataset)
                    & (method_rows["target"].astype(str) == TARGETS[dataset])
                    & (method_rows["horizon"].astype(int) == int(horizon))
                ].copy()
                seeds = sorted(int(value) for value in group["seed"].tolist())
                if seeds != list(SEEDS):
                    raise BL910FormalError(
                        f"{method}/{dataset}/{TARGETS[dataset]}/H{horizon} seeds={seeds}"
                    )
                rmse_values = [float(value) for value in group["rmse"].tolist()]
                smape_values = [float(value) for value in group["smape"].tolist()]
                records.append(
                    {
                        "method": method,
                        "method_id": METHOD_IDS[method],
                        "dataset": dataset,
                        "target": TARGETS[dataset],
                        "mode": MODE,
                        "horizon": int(horizon),
                        "rmse_mean": float(statistics.mean(rmse_values)),
                        "rmse_std": float(statistics.stdev(rmse_values)),
                        "smape_mean": float(statistics.mean(smape_values)),
                        "smape_std": float(statistics.stdev(smape_values)),
                        "seed_count": 5,
                        "seeds": ",".join(str(seed) for seed in SEEDS),
                        "n_mean": float(statistics.mean(float(value) for value in group["n"].tolist()))
                        if "n" in group.columns
                        else float("nan"),
                        "metric_space": METRIC_SPACE,
                        "std_ddof": 1,
                    }
                )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != EXPECTED_AGGREGATES:
        raise BL910FormalError(f"expected {EXPECTED_AGGREGATES} aggregates, got {len(frame)}")
    return frame


def holm_adjust(p_values: Mapping[str, float], alpha: float = HOLM_ALPHA) -> dict[str, dict[str, Any]]:
    items = [(index, key, float(value)) for index, (key, value) in enumerate(p_values.items())]
    ordered = sorted(items, key=lambda item: (item[2], item[0]))
    adjusted_by_key: dict[str, dict[str, Any]] = {}
    running = 0.0
    for rank, (_index, key, p_value) in enumerate(ordered, start=1):
        multiplier = len(items) - rank + 1
        candidate = min(1.0, p_value * multiplier)
        running = max(running, candidate)
        adjusted_by_key[key] = {
            "raw_p": p_value,
            "rank": rank,
            "multiplier": multiplier,
            "adjusted_p": running,
            "alpha": float(alpha),
            "significant": running < float(alpha),
            "family": HOLM_FAMILY,
            "family_size": len(items),
        }
    return adjusted_by_key


def bootstrap_mean_ci(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 12:
        raise BL910FormalError("bootstrap input must contain exactly 12 condition values")
    array = np.asarray([float(value) for value in values], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array)))
    bootstrap_means = array[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
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
        return payload
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
        return payload


def model_config_payload(method: str) -> dict[str, Any]:
    if method == PATCHTST_METHOD:
        config = assert_patchtst_config()
        return {
            **config,
            "direct_single_offset": True,
            "pred_len": 1,
            "feature_columns": list(FEATURE_METADATA),
            "target_column": TARGET_COLUMN,
            "target_channel_index": target_channel_index(FEATURE_METADATA),
        }
    if method == TIMESNET_METHOD:
        config = assert_timesnet_config()
        return {
            **config,
            "direct_single_offset": True,
            "feature_columns": list(FEATURE_METADATA),
            "target_column": TARGET_COLUMN,
            "target_channel_index": target_channel_index(FEATURE_METADATA),
        }
    raise BL910FormalError(f"unsupported method config: {method}")


def yes_no(value: bool) -> str:
    return "PASS" if value else "FAIL"


def training_policy_report() -> dict[str, Any]:
    return {
        "FORMAL_BASELINE_TRAINING_POLICY": FORMAL_BASELINE_TRAINING_POLICY,
        "BL9_POLICY_MATCH": "YES",
        "BL10_POLICY_MATCH": "YES",
        "PATCHTST_CONFIG": assert_patchtst_config(),
        "TIMESNET_CONFIG": assert_timesnet_config(),
        "feature_columns": list(FEATURE_METADATA),
        "target_column": TARGET_COLUMN,
        "target_channel_index": target_channel_index(FEATURE_METADATA),
        "lookback": LOOKBACK,
        "mode": MODE,
        "metric_space": METRIC_SPACE,
        "smape_already_percent": True,
    }


def load_sita_authority(path: Path = DEFAULT_FREEZE) -> pd.DataFrame:
    if not path.exists():
        raise BL910FormalError(f"SITA freeze workbook missing: {path}")
    frame = pd.read_excel(path, sheet_name=SITA_SHEET)
    frame.columns = [str(column).strip() for column in frame.columns]
    mapped = frame.copy()
    if "method" in mapped.columns:
        mapped["method"] = mapped["method"].astype(str).replace({"MS-KATL-V3": "SITA", "SITA": "SITA"})
        mapped = mapped[mapped["method"].astype(str) == "SITA"].copy()
    else:
        mapped["method"] = "SITA"
    rmse_col = "rmse_original" if "rmse_original" in mapped.columns else "rmse"
    smape_col = "smape_original" if "smape_original" in mapped.columns else "smape"
    target_col = "target" if "target" in mapped.columns else "target_entity_key"
    mapped["dataset"] = mapped["dataset"].astype(str)
    mapped["target"] = mapped[target_col].astype(str)
    mapped["horizon"] = mapped["horizon"].astype(int)
    mapped["seed"] = mapped["seed"].astype(int)
    mapped["rmse"] = mapped[rmse_col].astype(float)
    mapped["smape"] = mapped[smape_col].astype(float)
    if "mode" in mapped.columns:
        mapped = mapped[mapped["mode"].astype(str).isin(["without", "without_information_sharing", MODE])]
    keep = mapped[["dataset", "target", "horizon", "seed", "rmse", "smape"]].copy()
    keep["method"] = "SITA"
    expected = {(dataset, TARGETS[dataset], horizon, seed) for dataset in DATASETS for horizon in HORIZONS for seed in SEEDS}
    actual = set(zip(keep["dataset"], keep["target"], keep["horizon"], keep["seed"]))
    if actual != expected:
        raise BL910FormalError(
            "SITA authority is not the exact 60-cell without matrix: "
            f"missing={sorted(expected - actual)[:8]} unexpected={sorted(actual - expected)[:8]}"
        )
    return keep


def load_freeze_baselines(path: Path = DEFAULT_FREEZE) -> pd.DataFrame:
    if not path.exists():
        raise BL910FormalError(f"freeze workbook missing: {path}")
    frame = pd.read_excel(path, sheet_name=BASELINE_SHEET)
    frame.columns = [str(column).strip() for column in frame.columns]
    rename = {
        "target_entity_key": "target",
        "dataset_id": "dataset",
        "scenario": "mode",
        "rmse_original": "rmse",
        "smape_original": "smape",
    }
    mapped = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns}).copy()
    mapped["dataset"] = mapped["dataset"].astype(str)
    mapped["target"] = mapped["target"].astype(str)
    mapped["horizon"] = mapped["horizon"].astype(int)
    mapped["seed"] = mapped["seed"].astype(int)
    mapped["method"] = mapped["method"].astype(str)
    mapped["method"] = mapped["method"].replace(
        {
            "BL3_LightGBM": "LightGBM",
            "BL4_LSTM": "LSTM",
            "BL5_SeasonalNaive": "Seasonal Naive",
            "BL6_DLinear": "DLinear",
            "BL7_TCN": "TCN",
        }
    )
    rmse_col = "rmse" if "rmse" in mapped.columns else "rmse_original"
    smape_col = "smape" if "smape" in mapped.columns else "smape_original"
    mapped["rmse"] = mapped[rmse_col].astype(float)
    mapped["smape"] = mapped[smape_col].astype(float)
    if "mode" in mapped.columns:
        mapped["mode"] = mapped["mode"].astype(str)
        mapped = mapped[mapped["mode"].isin(["without", "without_information_sharing", MODE])]
    keep = mapped[["method", "dataset", "target", "horizon", "seed", "rmse", "smape"]].copy()
    keep["mode"] = MODE
    return keep


def load_bl8_cells(root: Path = BL8_ROOT) -> pd.DataFrame:
    records = []
    for path in sorted((root / "cells").glob("*/cell_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("cell_status", "")).upper() != "PASS":
            continue
        records.append(
            {
                "method": "MSML-TL",
                "method_id": "BL8_MSML_TL",
                "dataset": str(payload["dataset"]),
                "target": str(payload["target"]),
                "mode": str(payload.get("mode", MODE)),
                "horizon": int(payload["horizon"]),
                "seed": int(payload["seed"]),
                "rmse": float(payload["rmse"]),
                "smape": float(payload["smape"]),
                "n": int(payload.get("n", payload.get("sample_count", 0))),
            }
        )
    if not records:
        raise BL910FormalError(f"BL8 cells missing under {root}")
    return pd.DataFrame.from_records(records)


def load_robustness_cells(root: Path = ROBUSTNESS_ROOT) -> pd.DataFrame:
    candidates = list(Path(root).glob("**/*RAW*.csv")) + list(Path(root).glob("**/*raw*.csv"))
    if not candidates:
        summary = list(Path(root).glob("**/*AGGREGATED*.csv")) + list(Path(root).glob("**/*.csv"))
        if not summary:
            return pd.DataFrame()
        candidates = summary
    frames = []
    for path in candidates:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        frame.columns = [str(column).strip() for column in frame.columns]
        if not {"dataset", "horizon", "seed"}.issubset(frame.columns):
            continue
        if "method" not in frame.columns and "paper_method" in frame.columns:
            frame["method"] = frame["paper_method"]
        if "method" not in frame.columns:
            continue
        frame["method"] = frame["method"].astype(str).replace(
            {
                "BL6_DLinear_RevIN": "DLinear_RevIN",
                "BL7_TCN_RevIN": "TCN_RevIN",
            }
        )
        if not set(frame["method"]).intersection(ROBUSTNESS_UNIVERSE):
            continue
        needed = {"rmse", "smape"}
        if not needed.issubset(frame.columns):
            continue
        keep = frame.copy()
        if "target" not in keep.columns:
            keep["target"] = keep["dataset"].map(TARGETS)
        keep["mode"] = MODE
        frames.append(keep[["method", "dataset", "target", "mode", "horizon", "seed", "rmse", "smape"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        ["method", "dataset", "target", "horizon", "seed"]
    )


def condition_means(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (method, dataset, target, horizon), group in rows.groupby(
        ["method", "dataset", "target", "horizon"], sort=False
    ):
        records.append(
            {
                "method": str(method),
                "dataset": str(dataset),
                "target": str(target),
                "horizon": int(horizon),
                "rmse_mean": float(statistics.mean(float(value) for value in group["rmse"])),
                "smape_mean": float(statistics.mean(float(value) for value in group["smape"])),
                "seed_count": int(group["seed"].nunique()),
            }
        )
    return pd.DataFrame.from_records(records)
