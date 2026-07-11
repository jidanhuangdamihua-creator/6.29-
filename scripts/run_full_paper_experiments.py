"""Full paper experiment runner with scenario and sensitivity coverage.

This script supports:
1. Three datasets x six methods core evaluation (including No-TL).
2. Information-sharing scenarios:
    - without_information_sharing (same-store/entity source pool only)
    - with_information_sharing (cross-store/entity source pool)
3. Source-count sensitivity analysis:
    - paper track: k in [1, 3, 5]
    - extended track: k in [6, 9]

All failures are captured per experiment row and do not interrupt the full run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tf_compat  # must be imported before tensorflow/keras

import numpy as np
import pandas as pd

from dataset_registry import list_dataset_names, normalize_dataset_name

from src.utils.environment import setup_logging
from src.constants import (
    D3_WITHOUT_INFO_SHARING_DOMAIN_FILTER,
    NOT_APPLICABLE,
    RESULT_CONTRACT_VERSION,
    RESULT_SCHEMA_COLUMNS,
    SCHEMA_FAMILY_D1_D3,
    UNKNOWN,
)
from src.utils.result_schema import (
    align_d1_d3_result_records,
    align_result_records,
    normalize_information_sharing_contract as _normalize_information_sharing_contract,
)
from src.protocols.runner_adapter import configure_protocol_frames
from src.protocols.rolling_origin import build_sample_manifest
from src.utils.result_validation import annotate_silent_metric_failure
from paper_reproduction_protocol import (
    MULTI_SOURCE_TL_METHODS,
    build_alignment_fields,
    ensure_paper_track_allowed,
    get_extended_source_counts,
    get_paper_source_counts,
    get_results_output_paths,
    load_paper_protocol,
    resolve_experiment_track,
    resolve_strict_paper_mode,
    validate_paper_protocol_config,
)
from src.utils.console_reporter import (
    print_completion,
    print_dataset_header,
    print_final_summary,
    print_global_progress,
    print_method_result,
    print_method_start,
    print_pipeline_header,
)
from src.utils.progress_tracker import ExperimentProgressTracker
from src.utils.runtime_control import set_verbose_mode


DATASETS = list_dataset_names()
ONLY_DATASET_CHOICES = ["dataset1", "dataset2", "dataset3"]
METHODS = ["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"]
INFO_SHARING_SCENARIOS = [
    "without_information_sharing",
    "with_information_sharing",
]
FORMAL_DATASET_PATHS = {
    "Dataset1": "数据集/原始数据/Dataset 1/train.csv",
    "Dataset2": "数据集/原始数据/Dataset 2.csv",
    "Dataset3": "数据集/原始数据/Dataset 3.csv",
}
SOLIDIFIED_DATASET_PATHS = {
    "Dataset1": {
        "source": "数据集/固化数据/dataset1-source.parquet",
        "target": "数据集/固化数据/dataset1-target.parquet",
    },
    "Dataset2": {
        "source": "数据集/固化数据/dataset2-source.parquet",
        "target": "数据集/固化数据/dataset2-target.parquet",
    },
    "Dataset3": {
        "source": "数据集/固化数据/dataset3-source.parquet",
        "target": "数据集/固化数据/dataset3-target.parquet",
    },
}
FORMAL_LR = 1e-4
FORMAL_EPOCHS = 50
FORMAL_CLIPNORM = None
FORMAL_DROPOUT = 0.1
DISABLE_COMPAT_RESULTS_COPY_ENV = "RFE_DISABLE_COMPAT_RESULTS_COPY"
DIAGNOSTIC_COLUMNS = [
    "y_pred_nan_count",
    "y_pred_inf_count",
    "y_true_nan_count",
    "y_true_inf_count",
    "X_test_nan_count",
    "X_test_inf_count",
    "model_weight_nan_count",
    "model_weight_inf_count",
]


def _should_sync_latest_results_copy() -> bool:
    return os.environ.get(DISABLE_COMPAT_RESULTS_COPY_ENV) != "1"


def _print_run_paths(output_paths: Dict[str, Path]) -> None:
    print(f"Run ID: {output_paths['run_id']}")
    print(f"Run directory: {output_paths['run_dir'].relative_to(ROOT)}")
    print(f"Results directory: {output_paths['results_dir'].relative_to(ROOT)}")


def _resolve_output_paths(
    *,
    protocol: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Resolve run output paths, optionally reusing a caller-provided run dir."""
    if output_dir is None:
        return get_results_output_paths(ROOT, protocol)

    run_dir = Path(output_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    outputs = protocol.get("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}

    results_dir = run_dir / "results"
    reports_dir = run_dir / "results_reports"
    alignment_dir = run_dir / "paper_alignment"
    audits_dir = run_dir / "audits"
    for path in (results_dir, reports_dir, alignment_dir, audits_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "run_id": run_dir.name,
        "run_dir": run_dir,
        "results_dir": results_dir,
        "reports_dir": reports_dir,
        "alignment_dir": alignment_dir,
        "audits_dir": audits_dir,
        "paper_csv": results_dir / str(outputs.get("paper_results_csv", "paper_results.csv")),
        "full_paper_csv": results_dir / str(outputs.get("full_paper_results_csv", "full_paper_results.csv")),
        "extended_csv": results_dir / str(outputs.get("extended_results_csv", "extended_results.csv")),
        "full_results_csv": results_dir / "full_results.csv",
        "ranking_csv": results_dir / "ranking.csv",
        "summary_csv": results_dir / "summary.csv",
    }


def _resolve_selected_datasets(only_dataset: Optional[str]) -> Tuple[str, ...]:
    if only_dataset is None:
        return tuple(DATASETS)
    return (normalize_dataset_name(only_dataset),)


def _build_ranking_df(results_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = results_df.copy()
    if "error" in ok_df.columns:
        ok_df = ok_df[ok_df["error"].fillna("").astype(str).str.strip().eq("")].copy()
    if ok_df.empty or "smape" not in ok_df.columns:
        return pd.DataFrame(columns=["dataset", "rank", "method", "smape", "rmse", "accuracy"])

    ranked_parts: List[pd.DataFrame] = []
    for _, group in ok_df.groupby("dataset", sort=True):
        ranked_parts.append(add_rank_column(group, metric_col="smape", ascending=True))
    return pd.concat(ranked_parts, ignore_index=True)


def _build_summary_df(results_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = results_df.copy()
    if "error" in ok_df.columns:
        ok_df = ok_df[ok_df["error"].fillna("").astype(str).str.strip().eq("")].copy()
    if ok_df.empty or "smape" not in ok_df.columns:
        return pd.DataFrame(
            columns=[
                "dataset",
                "best_method",
                "best_smape",
                "smape",
                "best_rmse",
                "best_accuracy",
                "total_methods",
            ]
        )

    rows: List[Dict[str, Any]] = []
    for dataset_name, group in ok_df.groupby("dataset", sort=True):
        best_idx = group["smape"].astype(float).idxmin()
        best_row = group.loc[best_idx]
        rows.append(
            {
                "dataset": dataset_name,
                "best_method": str(best_row["method"]),
                "best_smape": float(best_row["smape"]),
                "smape": float(best_row["smape"]),
                "best_rmse": float(best_row["rmse"]),
                "best_accuracy": float(best_row["accuracy"]),
                "total_methods": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _save_dataset_result_csvs(
    results_df: pd.DataFrame,
    results_dir: Path,
    datasets: Optional[Sequence[str]] = None,
    info_sharing_suffix: Optional[str] = None,
    dataset_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Path]:
    saved: Dict[str, Path] = {}
    selected_datasets = tuple(DATASETS) if datasets is None else tuple(datasets)
    for dataset_name in selected_datasets:
        dataset_slug = normalize_dataset_name(dataset_name).lower()
        dataset_df = results_df[results_df["dataset"] == dataset_name].copy()
        if dataset_frames is not None:
            dataset_frames[dataset_name] = dataset_df
        if info_sharing_suffix is None:
            filename = f"{dataset_slug}_results.csv"
        else:
            filename = f"{dataset_slug}_{info_sharing_suffix}_results.csv"
        out_path = results_dir / filename
        dataset_df.to_csv(out_path, index=False, encoding="utf-8")
        saved[dataset_name] = out_path
    return saved


def _sync_latest_results_copy(
    paper_results_df: pd.DataFrame,
    extended_results_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    dataset_csv_paths: Dict[str, Path],
    output_paths: Dict[str, Path],
    dataset_result_frames: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Write optional latest copies under outputs/experiment_results/ for backward compatibility."""
    compat_dir = ROOT / "outputs" / "experiment_results"
    compat_dir.mkdir(parents=True, exist_ok=True)

    copies = {
        output_paths["paper_csv"].name: paper_results_df,
        output_paths["full_paper_csv"].name: paper_results_df,
        output_paths["extended_csv"].name: extended_results_df,
        output_paths["full_results_csv"].name: paper_results_df,
        output_paths["ranking_csv"].name: ranking_df,
        output_paths["summary_csv"].name: summary_df,
    }
    for dataset_name, src_path in dataset_csv_paths.items():
        frame = None if dataset_result_frames is None else dataset_result_frames.get(dataset_name)
        copies[src_path.name] = frame if frame is not None else pd.read_csv(src_path)

    for filename, frame in copies.items():
        frame.to_csv(compat_dir / filename, index=False, encoding="utf-8")


def _save_run_results(
    paper_results_df: pd.DataFrame,
    extended_results_df: pd.DataFrame,
    output_paths: Dict[str, Path],
    datasets: Optional[Sequence[str]] = None,
    info_sharing_suffix: Optional[str] = None,
) -> Dict[str, Path]:
    results_dir = output_paths["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    paper_results_df.to_csv(output_paths["paper_csv"], index=False, encoding="utf-8")
    if output_paths["paper_csv"] != output_paths["full_paper_csv"]:
        paper_results_df.to_csv(output_paths["full_paper_csv"], index=False, encoding="utf-8")
    extended_results_df.to_csv(output_paths["extended_csv"], index=False, encoding="utf-8")
    paper_results_df.to_csv(output_paths["full_results_csv"], index=False, encoding="utf-8")

    dataset_result_frames: Dict[str, pd.DataFrame] = {}
    dataset_csv_paths = _save_dataset_result_csvs(
        paper_results_df,
        results_dir,
        datasets=datasets,
        info_sharing_suffix=info_sharing_suffix,
        dataset_frames=dataset_result_frames,
    )
    ranking_df = _build_ranking_df(paper_results_df)
    summary_df = _build_summary_df(paper_results_df)
    ranking_df.to_csv(output_paths["ranking_csv"], index=False, encoding="utf-8")
    summary_df.to_csv(output_paths["summary_csv"], index=False, encoding="utf-8")

    if _should_sync_latest_results_copy():
        _sync_latest_results_copy(
            paper_results_df=paper_results_df,
            extended_results_df=extended_results_df,
            ranking_df=ranking_df,
            summary_df=summary_df,
            dataset_csv_paths=dataset_csv_paths,
            output_paths=output_paths,
            dataset_result_frames=dataset_result_frames,
        )

    saved_paths = {
        "paper_csv": output_paths["paper_csv"],
        "full_paper_csv": output_paths["full_paper_csv"],
        "extended_csv": output_paths["extended_csv"],
        "full_results_csv": output_paths["full_results_csv"],
        "ranking_csv": output_paths["ranking_csv"],
        "summary_csv": output_paths["summary_csv"],
        **dataset_csv_paths,
    }
    return saved_paths


def _print_saved_results(saved_paths: Dict[str, Path]) -> None:
    for key in ("Dataset1", "Dataset2", "Dataset3"):
        if key in saved_paths:
            print(f"Saved {key} results: {saved_paths[key].relative_to(ROOT)}")
    for label, path_key in (
        ("full results", "full_results_csv"),
        ("ranking", "ranking_csv"),
        ("summary", "summary_csv"),
    ):
        if path_key in saved_paths:
            print(f"Saved {label}: {saved_paths[path_key].relative_to(ROOT)}")


def _load_experiment_runners() -> None:
    global add_rank_column
    global prepare_base_data_for_experiments
    global run_result_visualization
    global run_msml_experiment
    global run_msml_rfe_experiment
    global run_mssb_experiment
    global run_mswa_experiment
    global run_no_tl_experiment
    global run_ss_tl_experiment

    from src.experiment.experiment_runner import (
        prepare_base_data_for_experiments,
        run_msml_experiment,
        run_msml_rfe_experiment,
        run_mssb_experiment,
        run_mswa_experiment,
        run_no_tl_experiment,
        run_ss_tl_experiment,
    )
    from src.visualization.result_visualizer import add_rank_column, run_result_visualization


def _strict_multi_source_topk(protocol: Dict[str, Any]) -> int:
    strict_cfg = protocol.get("strict_source_selection", {})
    if not isinstance(strict_cfg, dict):
        return 3
    return int(strict_cfg.get("multi_source_top_k", 3))


def _enforce_strict_multi_source_topk(protocol: Dict[str, Any]) -> bool:
    strict_cfg = protocol.get("strict_source_selection", {})
    if not isinstance(strict_cfg, dict):
        return True
    return bool(strict_cfg.get("enforce_multi_source_topk3", True))


def _scenario_to_bool(scenario: str) -> bool:
    if scenario == "with_information_sharing":
        return True
    if scenario == "without_information_sharing":
        return False
    raise ValueError(f"Unsupported information sharing scenario: {scenario}")


def normalize_information_sharing_contract(value: Any) -> str:
    """Normalize only the information_sharing contract field to with/without."""
    return _normalize_information_sharing_contract(value)


def _info_sharing_cli_to_scenario(info_sharing: Optional[str]) -> Optional[str]:
    if info_sharing is None:
        return None
    if info_sharing == "without":
        return "without_information_sharing"
    if info_sharing == "with":
        return "with_information_sharing"
    raise ValueError(f"Unsupported --info-sharing value: {info_sharing}")


def _use_id_static_features_in_signature(cfg: Dict[str, Any]) -> bool:
    paper_cfg = cfg.get("paper_reproduction", {}) if isinstance(cfg, dict) else {}
    return bool(paper_cfg.get("use_id_static_features_in_signature", False))


def _load_config() -> Dict[str, Any]:
    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    _apply_formal_config_overrides(cfg)
    return cfg


def _apply_formal_config_overrides(cfg: Dict[str, Any]) -> None:
    cfg.setdefault("dataset_paths", {}).update(FORMAL_DATASET_PATHS)

    exp_cfg = cfg.setdefault("single_experiment", {})
    exp_cfg["learning_rate"] = FORMAL_LR
    exp_cfg["source_epochs"] = FORMAL_EPOCHS
    exp_cfg["target_epochs"] = FORMAL_EPOCHS
    exp_cfg["epochs"] = FORMAL_EPOCHS
    exp_cfg["clipnorm"] = FORMAL_CLIPNORM
    exp_cfg["dropout"] = FORMAL_DROPOUT


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _solidified_paths_for_dataset(dataset_name: str) -> Dict[str, Path]:
    if dataset_name not in SOLIDIFIED_DATASET_PATHS:
        raise ValueError(f"No solidified parquet paths configured for {dataset_name}")
    return {
        split: ROOT / rel_path
        for split, rel_path in SOLIDIFIED_DATASET_PATHS[dataset_name].items()
    }


def _strict_target_split_days(dataset_name: str, cfg: Dict[str, Any]) -> Dict[str, int]:
    strict_protocol = _cfg_get(cfg, "paper_reproduction.strict_dataset_protocol", {})
    if not isinstance(strict_protocol, dict):
        raise ValueError("Missing config field paper_reproduction.strict_dataset_protocol")
    dataset_protocol = strict_protocol.get(dataset_name)
    if not isinstance(dataset_protocol, dict):
        raise ValueError(f"Missing config field paper_reproduction.strict_dataset_protocol.{dataset_name}")
    split_days = dataset_protocol.get("target_split_days")
    if not isinstance(split_days, dict):
        raise ValueError(
            f"Missing config field paper_reproduction.strict_dataset_protocol.{dataset_name}.target_split_days"
        )

    resolved: Dict[str, int] = {}
    for key in ("train_days", "val_days", "test_days"):
        if key not in split_days:
            raise ValueError(
                f"Missing config field paper_reproduction.strict_dataset_protocol."
                f"{dataset_name}.target_split_days.{key}"
            )
        value = int(split_days[key])
        if value <= 0:
            raise ValueError(
                f"Invalid non-positive paper split day for {dataset_name}: {key}={value}"
            )
        resolved[key] = value
    return resolved


def _normalize_solidified_columns(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply runner-local column compatibility for solidified parquet inputs."""
    if dataset_name != "Dataset3":
        return df

    rename_map = {
        "Customers": "customers",
        "Open": "open",
        "Promo": "promo",
        "SchoolHoliday": "school_holiday",
    }
    normalized = df.copy()
    for old_col, new_col in rename_map.items():
        if old_col not in normalized.columns:
            continue
        if new_col in normalized.columns:
            normalized = normalized.drop(columns=[old_col])
        else:
            normalized = normalized.rename(columns={old_col: new_col})
    return normalized


def _assert_dataset3_target_is_store10(target_df: pd.DataFrame) -> None:
    """Fail fast unless Dataset3 target provenance is exactly Rossmann store 10."""
    required_columns = ("entity_id", "store_id")
    missing = [column for column in required_columns if column not in target_df.columns]
    if missing:
        raise ValueError(
            "D3 target missing required columns: "
            + ", ".join(missing)
        )

    for column in required_columns:
        values = sorted(target_df[column].dropna().astype(str).unique().tolist())
        if values != ["10"]:
            label = "entity" if column == "entity_id" else "store"
            raise ValueError(
                f"D3 target {label} mismatch: expected ['10'], got {values}"
            )


def _dataset3_target_metadata(target_df: pd.DataFrame) -> Dict[str, Any]:
    """Return Dataset3 target identity metadata after validating target provenance."""
    _assert_dataset3_target_is_store10(target_df)
    metadata: Dict[str, Any] = {
        "target_entity_id": "10",
        "target_store_id": "10",
        "target_item_id": "",
    }
    if "item_id" in target_df.columns:
        item_values = sorted(target_df["item_id"].dropna().astype(str).unique().tolist())
        if len(item_values) == 1:
            metadata["target_item_id"] = item_values[0]
    return metadata


def _attach_target_metadata(
    dataset_name: str,
    row: Dict[str, Any],
    target_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge dataset-specific target identity metadata into a result row."""
    out = dict(row)
    if dataset_name == "Dataset3":
        out.update(target_metadata or {})
    return out


def _apply_solidified_split_attrs(
    dataset_name: str,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cfg: Dict[str, Any],
    strict_paper_mode: bool,
    strict_paper_split: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Attach split metadata expected by paper alignment reporting."""
    source_df = source_df.copy()
    target_df = target_df.copy()

    source_df.attrs["dataset_name"] = str(dataset_name)
    source_df.attrs["split_role"] = "source"
    source_df.attrs["split_mode"] = _cfg_get(cfg, "preprocessing.source_split_mode", "ratio")
    source_df.attrs["split_config"] = {
        "train_ratio": _cfg_get(cfg, "preprocessing.source_train_ratio", 0.8),
        "val_ratio": _cfg_get(cfg, "preprocessing.source_val_ratio", 0.1),
        "test_ratio": _cfg_get(cfg, "preprocessing.source_test_ratio", 0.1),
        "date_boundaries": _cfg_get(cfg, "preprocessing.source_date_boundaries", {}),
    }
    source_df.attrs["strict_paper_mode"] = bool(strict_paper_mode)
    source_df.attrs["strict_paper_split"] = bool(strict_paper_split)
    source_df.attrs["strict_dataset_name"] = str(dataset_name)

    if not strict_paper_mode and "date" in target_df.columns and not target_df.empty:
        target_dates = (
            pd.to_datetime(target_df["date"], errors="coerce")
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        if len(target_dates) < 210:
            raise ValueError(
                f"{dataset_name} non-strict target requires at least 210 unique dates, "
                f"got {len(target_dates)}"
            )
        keep_dates = set(target_dates.tail(210))
        target_df = target_df[
            pd.to_datetime(target_df["date"], errors="coerce").isin(keep_dates)
        ].copy()

    target_unique_days = int(target_df["date"].nunique()) if "date" in target_df.columns else 0
    if "date" in target_df.columns and not target_df.empty:
        target_range_days = int((target_df["date"].max() - target_df["date"].min()).days + 1)
    else:
        target_range_days = target_unique_days

    target_df.attrs["dataset_name"] = str(dataset_name)
    target_df.attrs["split_role"] = "target"
    target_df.attrs["split_mode"] = "days" if strict_paper_mode else "paper_split_protocol"
    if target_df.attrs["split_mode"] == "days":
        target_df.attrs["split_config"] = _strict_target_split_days(dataset_name, cfg)
    else:
        target_df.attrs["split_config"] = {
            "train_days": 15,
            "val_days": 15,
            "test_days": 180,
        }
        target_df.attrs["paper_split_protocol"] = "solidified_non_strict_train15_val15_test180"
        target_df.attrs["observed_days"] = 30
        target_df.attrs["test_days"] = 180
        target_df.attrs["train_days"] = 15
        target_df.attrs["val_days"] = 15
    target_df.attrs["strict_paper_mode"] = bool(strict_paper_mode)
    target_df.attrs["strict_paper_split"] = bool(strict_paper_split)
    target_df.attrs["strict_dataset_name"] = str(dataset_name)
    target_df.attrs["target_window_expected_days"] = 210 if not strict_paper_mode else target_unique_days
    target_df.attrs["target_window_range_days"] = target_range_days
    target_df.attrs["target_window_unique_days"] = target_unique_days
    return source_df, target_df


def _load_solidified_base_data(
    dataset_name: str,
    cfg: Dict[str, Any],
    strict_paper_mode: bool,
    strict_paper_split: bool,
) -> Dict[str, pd.DataFrame]:
    """Load D1-D3 from solidified source/target parquet without CSV preprocessing."""
    paths = _solidified_paths_for_dataset(dataset_name)
    source_df = pd.read_parquet(paths["source"])
    target_df = pd.read_parquet(paths["target"])

    source_df = _normalize_solidified_columns(dataset_name, source_df)
    target_df = _normalize_solidified_columns(dataset_name, target_df)
    if dataset_name == "Dataset3":
        _assert_dataset3_target_is_store10(target_df)
    source_df, target_df = _apply_solidified_split_attrs(
        dataset_name=dataset_name,
        source_df=source_df,
        target_df=target_df,
        cfg=cfg,
        strict_paper_mode=strict_paper_mode,
        strict_paper_split=strict_paper_split,
    )
    processed_df = pd.concat([source_df, target_df], ignore_index=True)
    processed_df.attrs["dataset_name"] = str(dataset_name)
    return {
        "raw_df": processed_df.copy(),
        "processed_df": processed_df,
        "source_df": source_df,
        "target_df": target_df,
    }


def _prepare_runner_base_data(
    dataset_name: str,
    data_path: str,
    cfg: Dict[str, Any],
    verbose_mode: str,
    strict_paper_mode: bool,
    strict_paper_split: bool,
) -> Dict[str, pd.DataFrame]:
    if dataset_name in SOLIDIFIED_DATASET_PATHS:
        return _load_solidified_base_data(
            dataset_name=dataset_name,
            cfg=cfg,
            strict_paper_mode=strict_paper_mode,
            strict_paper_split=strict_paper_split,
        )
    return prepare_base_data_for_experiments(
        dataset_name=dataset_name,
        data_path=data_path,
        config=cfg,
        verbose_mode=verbose_mode,
    )


def _coalesce_metric(*values: Any) -> Any:
    """Return the first numeric/reporting value that is not None/NaN."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            continue
        return value
    return None


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _source_identifier_from_meta(method_name: str, method_meta: Dict[str, Any]) -> str:
    if method_name == "No-TL":
        return NOT_APPLICABLE
    if "source_key" in method_meta and method_meta.get("source_key") not in (None, ""):
        return str(method_meta.get("source_key"))
    if method_name == "SS-TL":
        selected = method_meta.get("selected_sources")
        if isinstance(selected, list) and len(selected) == 1 and isinstance(selected[0], dict):
            source_key = selected[0].get("source_key")
            return UNKNOWN if source_key in (None, "") else str(source_key)
    return NOT_APPLICABLE if method_name in MULTI_SOURCE_TL_METHODS else UNKNOWN


def _selected_sources_from_meta(method_name: str, method_meta: Dict[str, Any]) -> str:
    if method_name == "No-TL":
        return NOT_APPLICABLE
    if "selected_sources" not in method_meta:
        return UNKNOWN
    selected = method_meta.get("selected_sources")
    if isinstance(selected, list):
        return _stable_json_dumps(selected)
    return UNKNOWN


def _source_domain_contract_fields(source_df: pd.DataFrame) -> Dict[str, Any]:
    domain_filter = source_df.attrs.get("domain_filter_used", None)
    scope_mode = str(source_df.attrs.get("source_pool_scope_mode", "") or "")
    if domain_filter is None:
        source_domain_filter = UNKNOWN
    elif isinstance(domain_filter, dict):
        source_domain_filter = _stable_json_dumps(domain_filter)
    else:
        source_domain_filter = str(domain_filter)
    return {
        "knn_json_domain_filter": NOT_APPLICABLE,
        "source_domain_filter": source_domain_filter,
        "source_domain_filter_name": scope_mode or UNKNOWN,
        "source_domain_filter_applied": bool(domain_filter not in (None, {}) and domain_filter != {"mode": "full_source_pool"}),
        "source_domain_filter_reason": scope_mode or UNKNOWN,
        "source_pool_size_before_filter": int(source_df.attrs.get("source_pool_size_before_filter", len(source_df))),
        "source_pool_size_after_filter": int(len(source_df)),
        "source_domain_filter_error": "",
    }


def _apply_result_contract_defaults(row: Dict[str, Any], schema_family: str) -> Dict[str, Any]:
    out = dict(row)
    out.setdefault("result_contract_version", RESULT_CONTRACT_VERSION)
    out.setdefault("schema_family", schema_family)
    return annotate_silent_metric_failure(out)


def _align_frame_to_preferred_columns(df: pd.DataFrame, schema_family: str) -> pd.DataFrame:
    if df.empty and len(df.columns) > 0:
        template_record = {
            column: "with" if column == "information_sharing" else ""
            for column in df.columns
        }
        template = align_result_records(
            [template_record],
            schema_family=schema_family,
            preferred_columns=RESULT_SCHEMA_COLUMNS,
            normalize_information_sharing=True,
            fill_missing_contract_defaults=True,
        )
        return template.iloc[0:0].copy()
    return align_result_records(
        df.to_dict(orient="records"),
        schema_family=schema_family,
        preferred_columns=RESULT_SCHEMA_COLUMNS,
        normalize_information_sharing=True,
        fill_missing_contract_defaults=True,
    )


def _finalize_result_metrics(result: Dict[str, Any]) -> None:
    """Ensure sMAPE aliases and metric_space_used are consistent for CSV output."""
    result["original_scale_smape"] = _coalesce_metric(
        result.get("original_scale_smape"),
        result.get("smape_paper"),
    )
    result["normalized_smape"] = _coalesce_metric(
        result.get("normalized_smape"),
        result.get("smape_current"),
    )
    smape_value = _coalesce_metric(
        result.get("original_scale_smape"),
        result.get("smape"),
        result.get("normalized_smape"),
    )
    result["smape"] = float(smape_value) if smape_value is not None else float("nan")

    if result.get("metric_space_used"):
        result["metric_space_used"] = str(result["metric_space_used"])
    elif _coalesce_metric(result.get("original_scale_smape"), result.get("smape_paper")) is not None:
        result["metric_space_used"] = str(result.get("metric_space_paper", "original_sales_space"))
    elif _coalesce_metric(result.get("normalized_smape"), result.get("smape_current")) is not None:
        result["metric_space_used"] = str(result.get("metric_space_current", "normalized_minmax_space"))
    else:
        result["metric_space_used"] = "normalized"


def _is_identifier_like_feature_col(column: str) -> bool:
    # D1-D3 legacy rule intentionally differs from D4-D6
    # data_preprocessing._is_identifier_like_column:
    # it additionally excludes date/qty_key/promo_key/region_id and does not
    # currently apply the generic *_nbr suffix rule. Do not silently unify in
    # this task; D4-D6 use paper_available_features_no_ids_v2.
    name = str(column).strip().lower()
    if not name:
        return False
    if name in {"date", "entity_id", "item_id", "qty_key", "promo_key", "region_id"}:
        return True
    if name == "id" or name.endswith("_id"):
        return True
    return False


def _is_shared_numeric_model_feature(
    column: str,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> bool:
    if column not in source_df.columns or column not in target_df.columns:
        return False
    if _is_identifier_like_feature_col(column):
        return False
    return bool(
        pd.api.types.is_numeric_dtype(source_df[column])
        and pd.api.types.is_numeric_dtype(target_df[column])
    )


def _sanitize_feature_cols(
    candidate_cols: Sequence[str],
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> List[str]:
    cols: List[str] = []
    for raw_col in candidate_cols:
        col = str(raw_col)
        if col in cols:
            continue
        if _is_shared_numeric_model_feature(col, source_df, target_df):
            cols.append(col)
    if "sales" in source_df.columns and "sales" in target_df.columns and "sales" not in cols:
        if _is_shared_numeric_model_feature("sales", source_df, target_df):
            cols.insert(0, "sales")
    if "sales" in cols:
        cols = ["sales"] + [c for c in cols if c != "sales"]
    return cols


def _project_modeling_frames(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop non-modeling baggage before normalize_features scans whole frames."""
    identity_cols = [
        "date",
        "entity_id",
        "item_id",
        "store_id",
        "brand_id",
        "region_id",
    ]
    keep_cols = list(
        dict.fromkeys(
            [
                c
                for c in identity_cols + list(feature_cols)
                if c in source_df.columns and c in target_df.columns
            ]
        )
    )
    projected_source = source_df[keep_cols].copy()
    projected_target = target_df[keep_cols].copy()
    projected_source.attrs = source_df.attrs.copy()
    projected_target.attrs = target_df.attrs.copy()
    return projected_source, projected_target


def _resolve_dataset_feature_cols(
    dataset_name: str,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> List[str]:
    """Resolve dataset-specific numeric feature columns shared by source/target."""
    default_cols = [str(c) for c in cfg.get("features", {}).get("default_feature_cols", ["sales", "year", "month", "week", "day"])]
    per_dataset_defaults: Dict[str, List[str]] = {
        "Dataset1": ["sales", "year", "month", "week", "day"],
        "Dataset2": ["sales", "year", "month", "week", "day", "promo", "item_id", "brand_code", "entity_id_code"],
        "Dataset3": [
            "sales",
            "year",
            "month",
            "week",
            "day",
            "day_of_week",
            "customers",
            "open",
            "promo",
            "state_holiday",
            "school_holiday",
            "store_id",
            "region_code",
            "holiday_promo_profile",
        ],
    }

    candidate = per_dataset_defaults.get(dataset_name, default_cols)
    resolved = _sanitize_feature_cols(candidate, source_df, target_df)

    if len(resolved) < 2:
        fallback = _sanitize_feature_cols(default_cols, source_df, target_df)
        resolved = fallback

    if not resolved:
        source_numeric = {c for c in source_df.columns if pd.api.types.is_numeric_dtype(source_df[c])}
        target_numeric = {c for c in target_df.columns if pd.api.types.is_numeric_dtype(target_df[c])}
        raise ValueError(
            f"No shared numeric features available for {dataset_name}. "
            f"source_numeric={sorted(source_numeric)} target_numeric={sorted(target_numeric)}"
        )
    return resolved


def _signature_static_features_for_dataset(
    dataset_name: str,
    use_information_sharing: bool,
    cfg: Dict[str, Any],
) -> List[str]:
    """Return extra static/profile features used in source signature construction."""
    if not use_information_sharing:
        return []

    include_id_static = _use_id_static_features_in_signature(cfg)

    if dataset_name == "Dataset2":
        profile_features = ["promo"]
        id_features = ["brand_code", "entity_id_code", "item_id"]
        return profile_features + (id_features if include_id_static else [])
    if dataset_name == "Dataset3":
        profile_features = [
            "holiday_promo_profile",
            "promo",
            "state_holiday",
            "school_holiday",
            "open",
        ]
        id_features = ["store_id", "region_code"]
        return profile_features + (id_features if include_id_static else [])
    return []


def _without_sharing_domain_filter(dataset_name: str) -> Dict[str, Any]:
    if dataset_name == "Dataset1":
        return {"column": "store_id", "value": 1}
    if dataset_name == "Dataset2":
        return {"column": "brand_id", "value": 1}
    if dataset_name == "Dataset3":
        return dict(D3_WITHOUT_INFO_SHARING_DOMAIN_FILTER)
    raise ValueError(f"No without_information_sharing domain filter configured for {dataset_name}")


def _apply_source_domain_filter(
    dataset_name: str,
    source_df: pd.DataFrame,
    domain_filter: Dict[str, Any],
) -> pd.DataFrame:
    column = str(domain_filter["column"])
    value = domain_filter["value"]
    if column not in source_df.columns:
        raise ValueError(
            f"Missing source domain filter column for {dataset_name}: "
            f"domain_filter.column={column}"
        )
    filtered = source_df[source_df[column] == value].copy()
    filtered.attrs["domain_filter_used"] = {"column": column, "value": value}
    if filtered.empty:
        raise ValueError(
            f"No source rows left under without_information_sharing scenario for {dataset_name}; "
            f"domain_filter.column={column} domain_filter.value={value}"
        )
    return filtered


def _source_scope_mode_for_domain_filter(dataset_name: str) -> str:
    if dataset_name == "Dataset1":
        return "without_information_sharing_same_store"
    if dataset_name == "Dataset2":
        return "without_information_sharing_same_brand"
    return "without_information_sharing_domain_filter"


def _apply_information_sharing_filter(
    dataset_name: str,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    use_information_sharing: bool,
    strict_paper_mode: bool,
    protocol: Dict[str, Any],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Apply source-pool filter for information-sharing scenario.

    without_information_sharing:
    - Dataset1: same store
    - Dataset2: same brand
    - Dataset3: same region (if region metadata unavailable, mark PARTIAL/TODO fallback)
    with_information_sharing: keep full source pool.
    """
    scenario = "with_information_sharing" if use_information_sharing else "without_information_sharing"
    source_pool_size_before_filter = int(len(source_df))
    source_df = source_df.copy()
    source_df.attrs["information_sharing_scenario"] = scenario
    source_df.attrs["source_pool_size_before_filter"] = source_pool_size_before_filter
    source_df.attrs["signature_static_feature_cols"] = _signature_static_features_for_dataset(
        dataset_name=dataset_name,
        use_information_sharing=use_information_sharing,
        cfg=cfg,
    )

    if use_information_sharing:
        source_df.attrs["source_pool_scope_mode"] = "with_information_sharing_full_pool"
        source_df.attrs["domain_filter_used"] = {"mode": "full_source_pool"}
        source_df.attrs["source_pool_size_after_filter"] = int(len(source_df))
        return source_df

    domain_filter = _without_sharing_domain_filter(dataset_name)
    filtered = _apply_source_domain_filter(dataset_name, source_df, domain_filter)
    filtered.attrs["source_pool_size_before_filter"] = source_pool_size_before_filter
    filtered.attrs["source_pool_size_after_filter"] = int(len(filtered))
    filtered.attrs["source_pool_scope_mode"] = _source_scope_mode_for_domain_filter(dataset_name)
    return filtered


def run_experiment(
    dataset_name: str,
    method_name: str,
    source_count: int,
    information_sharing_scenario: str,
    cfg: Dict[str, Any],
    protocol: Dict[str, Any],
    strict_paper_mode: bool,
    verbose_mode: str = "summary",
    base_data: Dict[str, pd.DataFrame] | None = None,
) -> Dict[str, Any]:
    """Run one experiment with explicit method/scenario/sensitivity settings."""
    ds_paths = cfg["dataset_paths"]
    exp_cfg = cfg["single_experiment"]

    base = base_data
    if base is None:
        base = _prepare_runner_base_data(
            dataset_name=dataset_name,
            data_path=ds_paths[dataset_name],
            cfg=cfg,
            verbose_mode=verbose_mode,
            strict_paper_mode=strict_paper_mode,
            strict_paper_split=bool(
                strict_paper_mode or cfg.get("paper_reproduction", {}).get("strict_paper_split", False)
            ),
        )
    source_df = base["source_df"]
    target_df = base["target_df"].copy()
    target_metadata = _dataset3_target_metadata(target_df) if dataset_name == "Dataset3" else {}

    feature_cols = _resolve_dataset_feature_cols(
        dataset_name=dataset_name,
        source_df=source_df,
        target_df=target_df,
        cfg=cfg,
    )

    use_information_sharing = _scenario_to_bool(information_sharing_scenario)
    requested_source_count = int(source_count)
    if strict_paper_mode and method_name in MULTI_SOURCE_TL_METHODS and _enforce_strict_multi_source_topk(protocol):
        source_count = _strict_multi_source_topk(protocol)

    source_df = _apply_information_sharing_filter(
        dataset_name=dataset_name,
        source_df=source_df,
        target_df=target_df,
        use_information_sharing=use_information_sharing,
        strict_paper_mode=strict_paper_mode,
        protocol=protocol,
        cfg=cfg,
    )
    target_df.attrs["information_sharing_scenario"] = source_df.attrs.get("information_sharing_scenario", "")
    target_df.attrs["signature_static_feature_cols"] = list(source_df.attrs.get("signature_static_feature_cols", []))
    source_df.attrs["method"] = method_name
    target_df.attrs["method"] = method_name
    source_df, target_df = _project_modeling_frames(source_df, target_df, feature_cols)
    protocol_group_cols = {
        "Dataset1": ("store_id", "item_id"),
        "Dataset2": ("brand_id", "item_id"),
        "Dataset3": ("store_id",),
    }[dataset_name]
    source_df, target_df = configure_protocol_frames(
        source_df,
        target_df,
        dataset_id=dataset_name,
        scenario=information_sharing_scenario,
        group_cols=protocol_group_cols,
        observed_start=pd.to_datetime(target_df["date"], errors="raise").min(),
    )
    protocol_manifest = build_sample_manifest(
        target_df,
        dataset_id=target_df.attrs["protocol_dataset_id"],
        track=target_df.attrs["protocol_track"],
        scenario=target_df.attrs["protocol_scenario"],
        target_key=target_df.attrs["protocol_target_key"],
        observed_end=target_df.attrs["knn_observed_end"],
        first_forecast_origin=pd.Timestamp(target_df.attrs["knn_observed_end"])
        + pd.Timedelta(days=int(exp_cfg["window_size"])),
        input_window=int(exp_cfg["window_size"]),
    )
    strict_metric_protocol = dict(protocol.get("metric_protocol", {}) or {})
    strict_metric_protocol.update(
        {
            "strict_paper_metrics": True,
            "paper_metric_space": "original_sales_space",
        }
    )

    common_kwargs: Dict[str, Any] = {
        "source_df": source_df,
        "target_df": target_df,
        "feature_cols": feature_cols,
        "horizon": int(exp_cfg["horizon"]),
        "window_size": int(exp_cfg["window_size"]),
        "learning_rate": float(exp_cfg.get("learning_rate", 0.001)),
        "source_epochs": int(exp_cfg["source_epochs"]),
        "target_epochs": int(exp_cfg["target_epochs"]),
        "batch_size": int(exp_cfg["batch_size"]),
        "metric_protocol": strict_metric_protocol,
        "group_cols": protocol_group_cols,
    }

    number_of_sources = int(source_count) if method_name in MULTI_SOURCE_TL_METHODS else (1 if method_name == "SS-TL" else 0)
    ensure_paper_track_allowed(
        method_name=method_name,
        requested_source_count=number_of_sources,
        protocol=protocol,
        strict_paper_mode=strict_paper_mode,
    )

    _t0 = time.perf_counter()
    if method_name == "No-TL":
        raw = run_no_tl_experiment(
            target_df=target_df,
            horizon=int(exp_cfg["horizon"]),
            window_size=int(exp_cfg["window_size"]),
            learning_rate=float(exp_cfg.get("learning_rate", 0.001)),
            target_epochs=int(exp_cfg["target_epochs"]),
            batch_size=int(exp_cfg["batch_size"]),
            metric_protocol=strict_metric_protocol,
            feature_cols=feature_cols,
        )
    elif method_name == "SS-TL":
        raw = run_ss_tl_experiment(**common_kwargs)
    elif method_name == "MSWA-TL":
        raw = run_mswa_experiment(
            **common_kwargs,
            number_of_sources=number_of_sources,
            weight_mode=str(exp_cfg["weight_mode"]),
        )
    elif method_name == "MSSB-TL":
        raw = run_mssb_experiment(
            **common_kwargs,
            number_of_sources=number_of_sources,
            weight_mode=str(exp_cfg["weight_mode"]),
        )
    elif method_name == "MSML-TL":
        raw = run_msml_experiment(
            **common_kwargs,
            number_of_sources=number_of_sources,
            weight_mode=str(exp_cfg["weight_mode"]),
        )
    elif method_name == "MSML-TL-RFE":
        raw = run_msml_rfe_experiment(
            **common_kwargs,
            number_of_sources=number_of_sources,
            weight_mode=str(exp_cfg["weight_mode"]),
            estimator_name=str(exp_cfg.get("estimator_name", "random_forest")),
            keep_ratio=float(exp_cfg["keep_ratio"]),
        )
    else:
        raise ValueError(f"Unsupported method: {method_name}")
    training_time = _coalesce_metric(raw.get("training_time"), raw.get("training_time_seconds"))
    if training_time is None:
        training_time = time.perf_counter() - _t0

    alignment = build_alignment_fields(
        method_name=str(raw["method"]),
        requested_source_count=number_of_sources,
        method_meta=raw.get("meta", {}),
        base_data=base,
        protocol=protocol,
    )

    source_identification: List[Dict[str, Any]] = []
    method_meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
    selection_requested_k = int(method_meta.get("requested_k", number_of_sources))
    selection_effective_k = int(method_meta.get("effective_k", len(method_meta.get("selected_sources", [])) if isinstance(method_meta.get("selected_sources"), list) else number_of_sources))
    valid_source_count = int(method_meta.get("valid_source_count", selection_effective_k))
    skipped_source_count = int(method_meta.get("skipped_source_count", 0))
    failed_source_count = int(method_meta.get("failed_source_count", skipped_source_count))
    failed_source_keys = method_meta.get("failed_source_keys", [])
    skipped_nonfinite_source_count = int(method_meta.get("skipped_nonfinite_source_count", 0))
    failed_sources = method_meta.get("failed_sources", [])
    date_alignment_mode = str(method_meta.get("date_alignment_mode", ""))
    information_sharing_contract = normalize_information_sharing_contract(
        information_sharing_scenario
    )
    if method_name == "SS-TL":
        key = method_meta.get("source_key")
        if isinstance(key, list):
            key = tuple(key)
        source_identification.append(
            {
                "dataset": dataset_name,
                "method": str(raw["method"]),
                "information_sharing": information_sharing_contract,
                "requested_source_count": int(requested_source_count),
                "effective_source_count": int(selection_effective_k),
                "requested_k": int(selection_requested_k),
                "effective_k": int(selection_effective_k),
                "valid_source_count": int(valid_source_count),
                "skipped_source_count": int(skipped_source_count),
                "failed_source_count": int(failed_source_count),
                "failed_source_keys": failed_source_keys,
                "skipped_nonfinite_source_count": int(skipped_nonfinite_source_count),
                "failed_sources": failed_sources,
                "date_alignment_mode": date_alignment_mode,
                "source_rank": 1,
                "source_key": str(key),
                "distance": float(method_meta.get("source_distance", 0.0)),
                "weight": float(method_meta.get("source_weight", 1.0)),
                "source_pool_scope_mode": str(source_df.attrs.get("source_pool_scope_mode", "")),
                "source_pool_scope_note": str(source_df.attrs.get("source_pool_scope_note", "")),
                "signature_base_features": "|".join(feature_cols),
                "signature_static_features": "|".join(source_df.attrs.get("signature_static_feature_cols", [])),
            }
        )
    else:
        selected = method_meta.get("selected_sources", []) if isinstance(method_meta, dict) else []
        if isinstance(selected, list):
            for idx, source_meta in enumerate(selected, start=1):
                source_identification.append(
                    {
                        "dataset": dataset_name,
                        "method": str(raw["method"]),
                        "information_sharing": information_sharing_contract,
                        "requested_source_count": int(requested_source_count),
                        "effective_source_count": int(selection_effective_k),
                        "requested_k": int(selection_requested_k),
                        "effective_k": int(selection_effective_k),
                        "valid_source_count": int(valid_source_count),
                        "skipped_source_count": int(skipped_source_count),
                        "failed_source_count": int(failed_source_count),
                        "failed_source_keys": failed_source_keys,
                        "skipped_nonfinite_source_count": int(skipped_nonfinite_source_count),
                        "failed_sources": failed_sources,
                        "date_alignment_mode": date_alignment_mode,
                        "source_rank": int(idx),
                        "source_key": str(source_meta.get("source_key")),
                        "distance": float(source_meta.get("distance", 0.0)),
                        "weight": float(source_meta.get("weight", 0.0)),
                        "source_pool_scope_mode": str(source_df.attrs.get("source_pool_scope_mode", "")),
                        "source_pool_scope_note": str(source_df.attrs.get("source_pool_scope_note", "")),
                        "signature_base_features": "|".join(feature_cols),
                        "signature_static_features": "|".join(source_df.attrs.get("signature_static_feature_cols", [])),
                    }
                )

    result = {
        "result_contract_version": RESULT_CONTRACT_VERSION,
        "schema_family": SCHEMA_FAMILY_D1_D3,
        "protocol_track": target_df.attrs["protocol_track"],
        "protocol_version": target_df.attrs["protocol_version"],
        "knn_observed_start": target_df.attrs["knn_observed_start"],
        "knn_observed_end": target_df.attrs["knn_observed_end"],
        "knn_representation": target_df.attrs["knn_representation"],
        "target_test_excluded": True,
        "source_future_excluded": True,
        "candidate_pool_digest": method_meta.get(
            "candidate_pool_digest", NOT_APPLICABLE
        ),
        "selection_result_digest": method_meta.get(
            "selection_result_digest", NOT_APPLICABLE
        ),
        "horizon": int(exp_cfg["horizon"]),
        "seed": int(exp_cfg.get("seed", 42)),
        "primary_metric_space": "original_sales",
        "sample_manifest_digest": protocol_manifest.digest,
        "sample_count": len(protocol_manifest.for_horizon(int(exp_cfg["horizon"]))),
        "dataset": dataset_name,
        "method": str(raw["method"]),
        "information_sharing": information_sharing_contract,
        "source_count": int(number_of_sources),
        "experiment_scope": alignment["experiment_scope"],
        "experiment_track": alignment["experiment_track"],
        "source_protocol_aligned": bool(alignment.get("source_protocol_aligned", False)),
        "strict_paper_mode": bool(strict_paper_mode),
        "alignment_status": alignment["alignment_status"],
        "metric_alignment_status": alignment["metric_alignment_status"],
        "split_alignment_status": alignment["split_alignment_status"],
        "source_pretrained_alignment_status": alignment["source_pretrained_alignment_status"],
        "paper_metric_space": alignment["paper_metric_space"],
        "metric_space_current": str(raw.get("metric_space_current", alignment["current_metric_space"])),
        "metric_space_paper": str(raw.get("metric_space_paper", alignment["paper_metric_space"])),
        "metric_space_used": str(raw.get("metric_space_used", "normalized")),
        "metric_space": str(raw.get("metric_space", raw.get("metric_space_used", "normalized"))),
        "rmse_metric_space": str(raw.get("rmse_metric_space", raw.get("metric_space_used", "normalized"))),
        "smape_metric_space": str(raw.get("smape_metric_space", raw.get("metric_space_used", "normalized"))),
        "paper_metric_aligned": bool(raw.get("paper_metric_aligned", False)),
        "inverse_transform_applied": bool(raw.get("inverse_transform_applied", False)),
        "inverse_transform_available": bool(raw.get("inverse_transform_available", False)),
        "metric_protocol": _stable_json_dumps(protocol.get("metric_protocol", {})),
        "metric_protocol_note": str(raw.get("metric_protocol_note", "")),
        "metric_protocol_error": str(raw.get("metric_protocol_error", "")),
        "metric_notes": str(raw.get("metric_notes", "")),
        "paper_split_reference": alignment["paper_split_reference"],
        "target_start_date": alignment["target_start_date"],
        "target_end_date": alignment["target_end_date"],
        "target_window_days": alignment["target_window_days"],
        "target_window_expected_days": alignment["target_window_expected_days"],
        "target_window_range_days": alignment["target_window_range_days"],
        "target_window_unique_days": alignment["target_window_unique_days"],
        "target_strict_paper_mode": alignment["target_strict_paper_mode"],
        "target_split_mode": alignment["target_split_mode"],
        "source_split_mode": alignment["source_split_mode"],
        "train_days": target_df.attrs.get("train_days", ""),
        "val_days": target_df.attrs.get("val_days", ""),
        "test_days": target_df.attrs.get("test_days", ""),
        "paper_pretrained_model_cap": alignment["paper_pretrained_model_cap"],
        "pretrained_model_count": alignment["actual_pretrained_model_count"],
        "requested_source_count": alignment["requested_source_count"],
        "actual_pretrained_model_count": alignment["actual_pretrained_model_count"],
        "requested_k": int(selection_requested_k),
        "effective_k": int(selection_effective_k),
        "valid_source_count": int(valid_source_count),
        "skipped_source_count": int(skipped_source_count),
        "failed_source_count": int(failed_source_count),
        "failed_source_keys": failed_source_keys,
        "skipped_nonfinite_source_count": int(skipped_nonfinite_source_count),
        "failed_sources": failed_sources,
        "date_alignment_mode": date_alignment_mode,
        "learning_rate": float(exp_cfg.get("learning_rate", 0.001)),
        "source_epochs": int(exp_cfg["source_epochs"]),
        "target_epochs": int(exp_cfg["target_epochs"]),
        "epochs": int(exp_cfg.get("epochs", exp_cfg["target_epochs"])),
        "clipnorm": exp_cfg.get("clipnorm"),
        "dropout": float(exp_cfg.get("dropout", 0.1)),
        "rmse": float(raw["rmse"]),
        "accuracy": float(raw["accuracy"]),
        "training_time": float(training_time),
        "mae": float(raw.get("mae", np.nan)),
        "mape": float(raw.get("mape", np.nan)),
        "smape": float(raw.get("smape", np.nan)),
        "rmse_current": float(raw.get("rmse_current", np.nan)),
        "accuracy_current": float(raw.get("accuracy_current", np.nan)),
        "mae_current": float(raw.get("mae_current", np.nan)),
        "mape_current": float(raw.get("mape_current", np.nan)),
        "smape_current": float(raw.get("smape_current", np.nan)),
        "rmse_paper": float(raw.get("rmse_paper", np.nan)),
        "accuracy_paper": float(raw.get("accuracy_paper", np.nan)),
        "mae_paper": float(raw.get("mae_paper", np.nan)),
        "mape_paper": float(raw.get("mape_paper", np.nan)),
        "smape_paper": float(raw.get("smape_paper", np.nan)),
        "normalized_rmse": raw.get("normalized_rmse", np.nan),
        "normalized_accuracy": raw.get("normalized_accuracy", np.nan),
        "normalized_mae": raw.get("normalized_mae", np.nan),
        "normalized_mape": raw.get("normalized_mape", np.nan),
        "normalized_smape": raw.get("normalized_smape", np.nan),
        "original_scale_rmse": raw.get("original_scale_rmse", np.nan),
        "original_scale_accuracy": raw.get("original_scale_accuracy", np.nan),
        "original_scale_mae": raw.get("original_scale_mae", np.nan),
        "original_scale_mape": raw.get("original_scale_mape", np.nan),
        "original_scale_smape": raw.get("original_scale_smape", np.nan),
        "prediction_shape": str(raw["prediction_shape"]),
        **{column: raw.get(column, np.nan) for column in DIAGNOSTIC_COLUMNS},
        **_source_domain_contract_fields(source_df),
        "source_identifier": _source_identifier_from_meta(method_name, method_meta),
        "selected_sources": _selected_sources_from_meta(method_name, method_meta),
        "selected_source_count": int(method_meta.get("selected_source_count", len(method_meta.get("selected_sources", [])) if isinstance(method_meta.get("selected_sources"), list) else 0)),
        "source_selection_feature_cols": _stable_json_dumps(list(feature_cols)),
        "model_feature_cols": _stable_json_dumps(list(feature_cols)),
        "feature_source": "runtime_feature_cols",
        "knn_feature_mode": NOT_APPLICABLE,
        "feature_consistency_status": "runtime_no_knn_json",
        "json_only_features": NOT_APPLICABLE,
        "runtime_only_features": _stable_json_dumps(list(feature_cols)),
        "source_numeric_na_repaired": NOT_APPLICABLE,
        "repaired_columns": NOT_APPLICABLE,
        "source_failure_messages": _stable_json_dumps(method_meta.get("source_failure_messages", [])),
        "alignment_notes": alignment["alignment_notes"],
        "error": "",
        "source_identification": source_identification,
        "feature_cols_final": list(feature_cols),
        "rfe_candidate_features": list(feature_cols),
        "rfe_selected_features": list(
            ((raw.get("meta", {}) or {}).get("selected_feature_cols", []))
            if isinstance(raw, dict)
            else []
        ),
        "signature_components": {
            "scenario": str(source_df.attrs.get("information_sharing_scenario", "")),
            "base_features": list(feature_cols),
            "static_features": list(source_df.attrs.get("signature_static_feature_cols", [])),
        },
    }
    result = _attach_target_metadata(
        dataset_name=dataset_name,
        row=result,
        target_metadata=target_metadata,
    )
    _finalize_result_metrics(result)
    return _apply_result_contract_defaults(result, SCHEMA_FAMILY_D1_D3)


def _build_run_plan(
    methods: Sequence[str],
    protocol: Dict[str, Any],
    strict_paper_mode: bool,
) -> List[Tuple[str, int, str, str]]:
    plan: List[Tuple[str, int, str, str]] = []
    paper_counts = get_paper_source_counts(protocol)
    extended_counts = [] if strict_paper_mode else get_extended_source_counts(protocol)
    for method_name in methods:
        if method_name in {"No-TL", "SS-TL"}:
            plan.append((method_name, 1, "without_information_sharing", "paper"))
            continue
        for scenario in INFO_SHARING_SCENARIOS:
            effective_paper_counts = list(paper_counts)
            if strict_paper_mode and _enforce_strict_multi_source_topk(protocol):
                effective_paper_counts = [_strict_multi_source_topk(protocol)]
            for source_count in effective_paper_counts:
                plan.append((method_name, int(source_count), scenario, "paper"))
            for source_count in extended_counts:
                plan.append((method_name, int(source_count), scenario, "extended"))
    return plan


def _print_runtime_steps() -> None:
    print("[1/5] 检查系统架构...")
    print(f"Python: {sys.executable}")
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"系统架构: {platform.machine()}")
    print("[2/5] 准备共享虚拟环境（固定路径，避免项目搬家后重复下载）...")
    print("环境由启动器统一管理，当前直接执行实验脚本。")
    print("[3/5] 校验 Python 架构...")
    print(f"解释器架构: {platform.machine()}")
    print("[4/5] 安装依赖（仅首次）...")
    print("依赖检查由当前解释器环境负责。")
    print("[5/5] 启动实验脚本...")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full paper experiments with summary/full console mode.")
    parser.add_argument(
        "--verbose-mode",
        choices=["summary", "full"],
        default="summary",
        help="Console output mode. summary suppresses low-level logs and shows high-level progress with ETA.",
    )
    parser.add_argument(
        "--strict-paper-mode",
        action="store_true",
        help="Run only paper-track experiments and block all extended settings.",
    )
    parser.add_argument(
        "--strict-paper-split",
        action="store_true",
        help="Force strict paper split protocol (1-month observed + 6-month forecast) without fallback.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config overrides, dataset paths, hyperparameters, and CSV output columns without running experiments.",
    )
    parser.add_argument(
        "--only-dataset",
        choices=ONLY_DATASET_CHOICES,
        default=None,
        help="Limit non-smoke full or dry-run execution to one dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional run directory. Defaults to a fresh timestamped outputs/runs directory.",
    )
    parser.add_argument(
        "--info-sharing",
        choices=["without", "with"],
        default=None,
        help="Limit D1-D3 execution to one information-sharing scenario.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the minimal D1-D3 scenario smoke matrix without saving full-run outputs.",
    )
    parser.add_argument(
        "--smoke-method",
        choices=METHODS,
        default="MSWA-TL",
        help="Method used by --smoke for each D1-D3 information-sharing scenario.",
    )
    return parser.parse_args()


def _dry_run_record(
    dataset_name: str,
    method_name: str,
    scenario: str,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cfg: Dict[str, Any],
    protocol: Dict[str, Any],
) -> Dict[str, Any]:
    filtered_source = _apply_information_sharing_filter(
        dataset_name=dataset_name,
        source_df=source_df,
        target_df=target_df,
        use_information_sharing=_scenario_to_bool(scenario),
        strict_paper_mode=bool(protocol.get("strict_paper_mode", False)),
        protocol=protocol,
        cfg=cfg,
    )
    split_config = target_df.attrs.get("split_config", {}) or {}
    return {
        "dataset_id": dataset_name,
        "method": method_name,
        "scenario": scenario,
        "target_rows": int(len(target_df)),
        "source_rows_before_filter": int(len(source_df)),
        "source_rows_after_filter": int(len(filtered_source)),
        "domain_filter_used": filtered_source.attrs.get("domain_filter_used"),
        "train_days": split_config.get("train_days"),
        "val_days": split_config.get("val_days"),
        "test_days": split_config.get("test_days"),
        "observed_days": int(target_df["date"].nunique()) if "date" in target_df.columns else 0,
    }


def _run_dry_run_checks(
    cfg: Dict[str, Any],
    protocol: Dict[str, Any],
    strict_paper_mode: bool,
    datasets: Optional[Sequence[str]] = None,
    info_sharing_scenario: Optional[str] = None,
) -> None:
    print("[dry-run] config loaded: configs/default_config.json")
    print("[dry-run] D1-D3 source pool and split records")
    run_plan = _build_run_plan(METHODS, protocol=protocol, strict_paper_mode=strict_paper_mode)
    if info_sharing_scenario is not None:
        run_plan = [item for item in run_plan if item[2] == info_sharing_scenario]
    selected_datasets = tuple(DATASETS) if datasets is None else tuple(datasets)
    for dataset_name in selected_datasets:
        base = _load_solidified_base_data(
            dataset_name=dataset_name,
            cfg=cfg,
            strict_paper_mode=strict_paper_mode,
            strict_paper_split=bool(
                strict_paper_mode or cfg.get("paper_reproduction", {}).get("strict_paper_split", False)
            ),
        )
        source_df = base["source_df"]
        target_df = base["target_df"]
        feature_cols = _resolve_dataset_feature_cols(dataset_name, source_df, target_df, cfg)
        projected_source, projected_target = _project_modeling_frames(source_df, target_df, feature_cols)
        todo_found = any(
            frame.astype(str)
            .apply(lambda col: col.str.contains("TODO_REGION_UNAVAILABLE", regex=False).any())
            .any()
            for frame in (source_df, target_df)
        )
        bad_projected = {
            c: str(projected_source[c].dtype)
            for c in projected_source.columns
            if not _is_identifier_like_feature_col(c)
            and (
                pd.api.types.is_object_dtype(projected_source[c])
                or pd.api.types.is_string_dtype(projected_source[c])
            )
        }
        paths = _solidified_paths_for_dataset(dataset_name)
        print(
            f"{dataset_name}: "
            f"source={paths['source'].relative_to(ROOT)} rows={len(source_df)} columns={len(source_df.columns)} "
            f"target={paths['target'].relative_to(ROOT)} rows={len(target_df)} columns={len(target_df.columns)} "
            f"feature_cols={feature_cols} "
            f"projected_string_feature_cols={bad_projected} "
            f"todo_region_unavailable={todo_found}"
        )
        for method_name, _, scenario, _ in run_plan:
            record = _dry_run_record(
                dataset_name=dataset_name,
                method_name=method_name,
                scenario=scenario,
                source_df=source_df,
                target_df=target_df,
                cfg=cfg,
                protocol=protocol,
            )
            print(json.dumps(record, ensure_ascii=True, sort_keys=True))

    exp_cfg = cfg["single_experiment"]
    print(
        "[dry-run] hyperparams "
        "lr=1e-4 "
        f"epochs={int(exp_cfg['epochs'])} "
        f"clipnorm={exp_cfg['clipnorm']} "
        f"dropout={float(exp_cfg['dropout'])}"
    )
    print("[dry-run] result CSV columns include metric_space_used=True training_time=True")


def _run_smoke(
    cfg: Dict[str, Any],
    protocol: Dict[str, Any],
    strict_paper_mode: bool,
    strict_paper_split: bool,
    verbose_mode: str,
    smoke_method: str,
) -> None:
    _load_experiment_runners()
    source_count = _strict_multi_source_topk(protocol) if smoke_method in MULTI_SOURCE_TL_METHODS else 1
    base_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    failures: List[str] = []
    print(
        json.dumps(
            {
                "smoke_method": smoke_method,
                "source_count": int(source_count),
                "strict_paper_mode": bool(strict_paper_mode),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    for dataset_name in ("Dataset1", "Dataset2", "Dataset3"):
        base_cache[dataset_name] = _prepare_runner_base_data(
            dataset_name=dataset_name,
            data_path=cfg["dataset_paths"][dataset_name],
            cfg=cfg,
            verbose_mode=verbose_mode,
            strict_paper_mode=strict_paper_mode,
            strict_paper_split=strict_paper_split,
        )
        for scenario in INFO_SHARING_SCENARIOS:
            try:
                record = run_experiment(
                    dataset_name=dataset_name,
                    method_name=smoke_method,
                    source_count=source_count,
                    information_sharing_scenario=scenario,
                    cfg=cfg,
                    protocol=protocol,
                    strict_paper_mode=strict_paper_mode,
                    verbose_mode=verbose_mode,
                    base_data=base_cache[dataset_name],
                )
                print(
                    json.dumps(
                        {
                            "dataset_id": dataset_name,
                            "method": smoke_method,
                            "scenario": scenario,
                            "rmse": float(record.get("rmse", np.nan)),
                            "smape": float(record.get("smape", np.nan)),
                            "accuracy": float(record.get("accuracy", np.nan)),
                            "requested_k": int(record.get("requested_k", source_count)),
                            "effective_k": int(record.get("effective_k", source_count)),
                            "valid_source_count": int(record.get("valid_source_count", source_count)),
                            "skipped_source_count": int(record.get("skipped_source_count", 0)),
                            "failed_source_count": int(record.get("failed_source_count", record.get("skipped_source_count", 0))),
                            "skipped_nonfinite_source_count": int(record.get("skipped_nonfinite_source_count", 0)),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )
            except Exception as exc:
                message = f"{dataset_name}/{scenario}/{smoke_method}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(json.dumps({"error": message}, ensure_ascii=True, sort_keys=True))
    if failures:
        raise RuntimeError("Smoke failed: " + " | ".join(failures))


def _build_error_row(
    dataset_name: str,
    method_name: str,
    source_count: int,
    information_sharing_scenario: str,
    protocol: Dict[str, Any],
    strict_paper_mode: bool,
    exc: Exception,
    target_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    requested_source_count = int(source_count) if method_name in MULTI_SOURCE_TL_METHODS else (1 if method_name == "SS-TL" else 0)
    alignment = build_alignment_fields(
        method_name=method_name,
        requested_source_count=requested_source_count,
        method_meta={},
        base_data=None,
        protocol=protocol,
    )
    result = {
        "result_contract_version": RESULT_CONTRACT_VERSION,
        "schema_family": SCHEMA_FAMILY_D1_D3,
        "dataset": dataset_name,
        "method": method_name,
        "information_sharing": normalize_information_sharing_contract(
            information_sharing_scenario
        ),
        "source_count": int(source_count),
        "experiment_scope": alignment["experiment_scope"],
        "experiment_track": alignment["experiment_track"],
        "source_protocol_aligned": bool(alignment.get("source_protocol_aligned", False)),
        "strict_paper_mode": bool(strict_paper_mode),
        "alignment_status": alignment["alignment_status"],
        "metric_alignment_status": alignment["metric_alignment_status"],
        "split_alignment_status": alignment["split_alignment_status"],
        "source_pretrained_alignment_status": alignment["source_pretrained_alignment_status"],
        "paper_metric_space": alignment["paper_metric_space"],
        "metric_space_current": alignment["current_metric_space"],
        "metric_space_paper": alignment["paper_metric_space"],
        "metric_space_used": "normalized",
        "metric_space": "normalized",
        "rmse_metric_space": "normalized",
        "smape_metric_space": "normalized",
        "paper_metric_aligned": False,
        "inverse_transform_applied": False,
        "inverse_transform_available": False,
        "metric_protocol": _stable_json_dumps(protocol.get("metric_protocol", {})),
        "metric_protocol_note": "",
        "metric_protocol_error": "",
        "metric_notes": "",
        "paper_split_reference": alignment["paper_split_reference"],
        "target_start_date": alignment["target_start_date"],
        "target_end_date": alignment["target_end_date"],
        "target_window_days": alignment["target_window_days"],
        "target_window_expected_days": alignment["target_window_expected_days"],
        "target_window_range_days": alignment["target_window_range_days"],
        "target_window_unique_days": alignment["target_window_unique_days"],
        "target_strict_paper_mode": alignment["target_strict_paper_mode"],
        "target_split_mode": alignment["target_split_mode"],
        "source_split_mode": alignment["source_split_mode"],
        "train_days": "",
        "val_days": "",
        "test_days": "",
        "paper_pretrained_model_cap": alignment["paper_pretrained_model_cap"],
        "pretrained_model_count": alignment["actual_pretrained_model_count"],
        "requested_source_count": alignment["requested_source_count"],
        "actual_pretrained_model_count": alignment["actual_pretrained_model_count"],
        "requested_k": requested_source_count,
        "effective_k": 0,
        "valid_source_count": 0,
        "skipped_source_count": 0,
        "failed_source_count": 0,
        "failed_source_keys": [],
        "skipped_nonfinite_source_count": 0,
        "failed_sources": [],
        "date_alignment_mode": "",
        "learning_rate": FORMAL_LR,
        "source_epochs": FORMAL_EPOCHS,
        "target_epochs": FORMAL_EPOCHS,
        "epochs": FORMAL_EPOCHS,
        "clipnorm": FORMAL_CLIPNORM,
        "dropout": FORMAL_DROPOUT,
        "rmse": np.nan,
        "accuracy": np.nan,
        "training_time": 0.0,
        "mae": np.nan,
        "mape": np.nan,
        "smape": np.nan,
        "rmse_current": np.nan,
        "accuracy_current": np.nan,
        "mae_current": np.nan,
        "mape_current": np.nan,
        "smape_current": np.nan,
        "rmse_paper": np.nan,
        "accuracy_paper": np.nan,
        "mae_paper": np.nan,
        "mape_paper": np.nan,
        "smape_paper": np.nan,
        "normalized_rmse": np.nan,
        "normalized_accuracy": np.nan,
        "normalized_mae": np.nan,
        "normalized_mape": np.nan,
        "normalized_smape": np.nan,
        "original_scale_rmse": np.nan,
        "original_scale_accuracy": np.nan,
        "original_scale_mae": np.nan,
        "original_scale_mape": np.nan,
        "original_scale_smape": np.nan,
        "prediction_shape": "N/A",
        **{column: np.nan for column in DIAGNOSTIC_COLUMNS},
        "knn_json_domain_filter": NOT_APPLICABLE,
        "source_domain_filter": UNKNOWN,
        "source_domain_filter_name": UNKNOWN,
        "source_domain_filter_applied": False,
        "source_domain_filter_reason": UNKNOWN,
        "source_pool_size_before_filter": "",
        "source_pool_size_after_filter": "",
        "source_domain_filter_error": "",
        "source_identifier": NOT_APPLICABLE if method_name == "No-TL" else UNKNOWN,
        "selected_sources": NOT_APPLICABLE if method_name == "No-TL" else UNKNOWN,
        "selected_source_count": 0,
        "source_selection_feature_cols": UNKNOWN,
        "model_feature_cols": UNKNOWN,
        "feature_source": UNKNOWN,
        "knn_feature_mode": NOT_APPLICABLE,
        "feature_consistency_status": UNKNOWN,
        "json_only_features": NOT_APPLICABLE,
        "runtime_only_features": UNKNOWN,
        "source_numeric_na_repaired": NOT_APPLICABLE,
        "repaired_columns": NOT_APPLICABLE,
        "source_failure_messages": "[]",
        "alignment_notes": alignment["alignment_notes"],
        "error": f"{type(exc).__name__}: {exc}",
    }
    result = _attach_target_metadata(
        dataset_name=dataset_name,
        row=result,
        target_metadata=target_metadata,
    )
    _finalize_result_metrics(result)
    return _apply_result_contract_defaults(result, SCHEMA_FAMILY_D1_D3)


def _result_columns() -> List[str]:
    return list(RESULT_SCHEMA_COLUMNS)


def _materialize_result_dataframes(
    paper_records: Sequence[Dict[str, Any]],
    extended_records: Sequence[Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return (
        align_d1_d3_result_records(paper_records, schema_family=SCHEMA_FAMILY_D1_D3),
        align_d1_d3_result_records(extended_records, schema_family=SCHEMA_FAMILY_D1_D3),
    )


def main() -> None:
    args = _parse_args()
    verbose_mode = str(args.verbose_mode).lower()
    selected_datasets = _resolve_selected_datasets(args.only_dataset)
    info_sharing_scenario = _info_sharing_cli_to_scenario(args.info_sharing)

    set_verbose_mode(verbose_mode)
    setup_logging(log_level="WARNING" if verbose_mode == "summary" else "INFO", log_file=None)

    cfg = _load_config()
    protocol = load_paper_protocol(cfg)
    strict_paper_mode = resolve_strict_paper_mode(cfg, explicit=bool(args.strict_paper_mode))
    strict_paper_split = bool(
        args.strict_paper_split
        or strict_paper_mode
        or cfg.get("paper_reproduction", {}).get("strict_paper_split", False)
    )
    protocol["strict_paper_mode"] = strict_paper_mode
    protocol["paper_strict_mode"] = strict_paper_mode
    protocol.setdefault("metric_protocol", {})["strict_paper_metrics"] = bool(strict_paper_mode)
    cfg.setdefault("paper_reproduction", {})["strict_paper_mode"] = strict_paper_mode
    cfg["paper_reproduction"]["paper_strict_mode"] = strict_paper_mode
    cfg["paper_reproduction"]["strict_paper_split"] = strict_paper_split
    cfg["paper_reproduction"]["paper_strict_split"] = strict_paper_split
    cfg["paper_reproduction"].setdefault("metric_protocol", {})["strict_paper_metrics"] = bool(strict_paper_mode)
    validation = validate_paper_protocol_config(protocol=protocol, strict_paper_mode=strict_paper_mode)
    print(
        "[paper_protocol_validation] "
        f"status={validation['status']} strict_paper_mode={strict_paper_mode} "
        f"strict_paper_split={strict_paper_split} "
        f"failures={len(validation['failures'])} warnings={len(validation['warnings'])}"
    )
    for warning in validation["warnings"]:
        print(f"[paper_protocol_todo] {warning}")
    if args.dry_run:
        _run_dry_run_checks(
            cfg,
            protocol=protocol,
            strict_paper_mode=strict_paper_mode,
            datasets=selected_datasets,
            info_sharing_scenario=info_sharing_scenario,
        )
        return
    if args.smoke:
        _run_smoke(
            cfg=cfg,
            protocol=protocol,
            strict_paper_mode=strict_paper_mode,
            strict_paper_split=strict_paper_split,
            verbose_mode=verbose_mode,
            smoke_method=str(args.smoke_method),
        )
        return

    _load_experiment_runners()
    output_paths = _resolve_output_paths(protocol=protocol, output_dir=args.output_dir)
    _print_run_paths(output_paths)
    if verbose_mode == "summary":
        _print_runtime_steps()
        print_pipeline_header()

    paper_records: List[Dict[str, Any]] = []
    extended_records: List[Dict[str, Any]] = []
    source_identification_records: List[Dict[str, Any]] = []

    dataset_run_plan = _build_run_plan(METHODS, protocol=protocol, strict_paper_mode=strict_paper_mode)
    if info_sharing_scenario is not None:
        dataset_run_plan = [item for item in dataset_run_plan if item[2] == info_sharing_scenario]
    total_runs = len(selected_datasets) * len(dataset_run_plan)
    tracker = ExperimentProgressTracker(total_runs=max(1, total_runs))

    base_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    method_index = {name: idx for idx, name in enumerate(METHODS, start=1)}
    method_total = len(METHODS)
    run_index = 0

    for dataset_name in selected_datasets:
        if dataset_name not in base_cache:
            base_cache[dataset_name] = _prepare_runner_base_data(
                dataset_name=dataset_name,
                data_path=cfg["dataset_paths"][dataset_name],
                cfg=cfg,
                verbose_mode=verbose_mode,
                strict_paper_mode=strict_paper_mode,
                strict_paper_split=strict_paper_split,
            )

        dataset_target_metadata = (
            _dataset3_target_metadata(base_cache[dataset_name]["target_df"])
            if dataset_name == "Dataset3"
            else {}
        )

        if verbose_mode == "summary":
            target_shape = str(tuple(base_cache[dataset_name]["target_df"].shape))
            source_unique = int(
                len(
                    base_cache[dataset_name]["source_df"][["entity_id", "item_id"]]
                    .drop_duplicates()
                )
            )
            print_dataset_header(dataset_name, target_shape, source_unique)

        for method_name, source_count, info_scenario, configured_track in dataset_run_plan:
            run_index += 1
            if verbose_mode == "summary":
                snapshot = tracker.update(current_dataset=dataset_name, current_method=method_name)
                print_global_progress(
                    current=run_index,
                    total=total_runs,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    eta_seconds=snapshot.eta_seconds,
                )
                print_method_start(
                    method_index.get(method_name, 0),
                    method_total,
                    f"{method_name} (k={source_count}, scenario={info_scenario})",
                )
            else:
                print(
                    f"Running experiment: {dataset_name} - {method_name} "
                    f"(k={source_count}, scenario={info_scenario}, track={configured_track})"
                )

            experiment_start = time.perf_counter()
            try:
                record = run_experiment(
                    dataset_name=dataset_name,
                    method_name=method_name,
                    source_count=source_count,
                    information_sharing_scenario=info_scenario,
                    cfg=cfg,
                    protocol=protocol,
                    strict_paper_mode=strict_paper_mode,
                    verbose_mode=verbose_mode,
                    base_data=base_cache[dataset_name],
                )
                if record["experiment_track"] == "paper":
                    paper_records.append(record)
                else:
                    extended_records.append(record)
                source_identification_records.extend(record.get("source_identification", []))
                if verbose_mode == "summary":
                    print_method_result(
                        method_name=record["method"],
                        rmse=float(record["rmse"]),
                        accuracy=float(record["accuracy"]),
                        smape=float(record.get("smape", np.nan)),
                        original_scale_smape=record.get("original_scale_smape"),
                    )
                else:
                    original_smape = pd.to_numeric(pd.Series([record.get("original_scale_smape")]), errors="coerce").iloc[0]
                    print(
                        f"Finished: sMAPE={record['smape']:.6f}, "
                        f"Original-scale sMAPE={float(original_smape):.6f}, "
                        f"RMSE={record['rmse']:.6f}, Accuracy={record['accuracy']:.6f}"
                    )
            except Exception as exc:
                error_row = _build_error_row(
                    dataset_name=dataset_name,
                    method_name=method_name,
                    source_count=source_count,
                    information_sharing_scenario=info_scenario,
                    protocol=protocol,
                    strict_paper_mode=strict_paper_mode,
                    exc=exc,
                    target_metadata=dataset_target_metadata,
                )
                error_row["training_time"] = float(time.perf_counter() - experiment_start)
                if error_row["experiment_track"] == "paper":
                    paper_records.append(error_row)
                else:
                    extended_records.append(error_row)
                print("Finished: sMAPE=nan, Original-scale sMAPE=nan, RMSE=nan, Accuracy=nan")
                print(f"Error: {error_row['error']}")
            finally:
                tracker.mark_completed()

    paper_results_df, extended_results_df = _materialize_result_dataframes(
        paper_records=paper_records,
        extended_records=extended_records,
    )

    output_paths["results_dir"].mkdir(parents=True, exist_ok=True)
    saved_paths = _save_run_results(
        paper_results_df=paper_results_df,
        extended_results_df=extended_results_df,
        output_paths=output_paths,
        datasets=selected_datasets,
        info_sharing_suffix=args.info_sharing,
    )

    # Source identification audit report for Table 5/6 style verification.
    paper_alignment_dir = output_paths["alignment_dir"]
    paper_alignment_dir.mkdir(parents=True, exist_ok=True)
    source_report_csv = paper_alignment_dir / "source_identification_report.csv"
    source_report_json = paper_alignment_dir / "source_identification_report.json"
    source_report_df = pd.DataFrame(source_identification_records)
    if not source_report_df.empty:
        source_report_df.to_csv(source_report_csv, index=False, encoding="utf-8")
        source_report_json.write_text(
            json.dumps(source_identification_records, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    viz_report = run_result_visualization(
        csv_path=str(output_paths["paper_csv"]),
        output_dir=str(output_paths["reports_dir"]),
        method_order=["No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"],
    )
    extended_viz_report = None
    if not extended_results_df.empty:
        extended_viz_report = run_result_visualization(
            csv_path=str(output_paths["extended_csv"]),
            output_dir=str(output_paths["reports_dir"]),
            method_order=["MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"],
        )

    if verbose_mode == "summary":
        print_final_summary(paper_results_df)
        print("\n结果文件")
        _print_saved_results(saved_paths)
        print(f"论文结果 CSV: {output_paths['paper_csv'].relative_to(ROOT)}")
        print(f"扩展结果 CSV: {output_paths['extended_csv'].relative_to(ROOT)}")
        print(f"表格: {viz_report['formatted_table_path']}")
        print(f"sMAPE 图: {viz_report.get('smape_plot_path', viz_report['rmse_plot_path'])}")
        print(f"Accuracy 图: {viz_report['accuracy_plot_path']}")
        if not source_report_df.empty:
            print(f"Source identification 报告: {source_report_csv.relative_to(ROOT)}")
        if extended_viz_report is not None:
            print(f"扩展表格: {extended_viz_report['formatted_table_path']}")
        print_completion()
    else:
        print("Full paper matrix experiments completed.")
        _print_saved_results(saved_paths)
        print("Paper Results CSV Path:")
        print(str(output_paths["paper_csv"].relative_to(ROOT)))
        print("Extended Results CSV Path:")
        print(str(output_paths["extended_csv"].relative_to(ROOT)))
        print("Formatted Ranking Table Path:")
        print(viz_report["formatted_table_path"])
        print("sMAPE Plot Path:")
        print(viz_report.get("smape_plot_path", viz_report["rmse_plot_path"]))
        print("Accuracy Plot Path:")
        print(viz_report["accuracy_plot_path"])


if __name__ == "__main__":
    main()
