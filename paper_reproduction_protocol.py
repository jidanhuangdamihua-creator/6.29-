"""Paper reproduction protocol helpers for strict and auditable runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from src.utils.run_utils import create_run_dir


STATUS_ALIGNED = "ALIGNED"
STATUS_PARTIAL = "PARTIAL"
STATUS_TODO = "TODO"
STATUS_EXTENDED = "EXTENDED"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"

TL_METHODS = {"SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"}
MULTI_SOURCE_TL_METHODS = {"MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"}


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _is_todo_placeholder(value: Any) -> bool:
    text = str(value or "").strip().upper()
    if not text:
        return True
    return text.startswith("TODO") or "UNCONFIRMED" in text


def _resolve_paper_source_protocol(paper_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve source/pretraining protocol with backward compatibility.

    Preferred key: paper_source_protocol
    Legacy key: source_pretrained_protocol
    """
    source_cfg_new = _as_dict(paper_cfg.get("paper_source_protocol", {}))
    source_cfg_legacy = _as_dict(paper_cfg.get("source_pretrained_protocol", {}))

    merged: Dict[str, Any] = {}

    merged["max_pretrained_models_from_paper"] = int(
        source_cfg_new.get(
            "max_pretrained_models_from_paper",
            source_cfg_legacy.get("paper_max_pretrained_models", 5),
        )
    )
    merged["default_paper_multi_source_count"] = int(
        source_cfg_new.get("default_paper_multi_source_count", 3)
    )
    merged["allow_extended_source_counts"] = bool(
        source_cfg_new.get("allow_extended_source_counts", True)
    )

    paper_counts = source_cfg_new.get(
        "paper_source_counts",
        source_cfg_legacy.get("paper_source_counts", [1, 3, 5]),
    )
    extended_counts = source_cfg_new.get(
        "extended_source_counts",
        source_cfg_legacy.get("extended_source_counts", [6, 9]),
    )

    merged["paper_source_counts"] = [int(v) for v in paper_counts]
    merged["extended_source_counts"] = [int(v) for v in extended_counts]
    merged["alignment_status"] = str(
        source_cfg_new.get(
            "alignment_status",
            source_cfg_legacy.get("alignment_status", STATUS_ALIGNED),
        )
    )
    merged["notes"] = str(
        source_cfg_new.get(
            "notes",
            source_cfg_legacy.get(
                "notes",
                "Paper-track multi-source TL experiments must use at most five pretrained source models.",
            ),
        )
    )
    return merged


def load_paper_protocol(config: Optional[Any]) -> Dict[str, Any]:
    if isinstance(config, Mapping):
        paper_cfg = _as_dict(config.get("paper_reproduction", {}))
        outputs_cfg = _as_dict(config.get("outputs", {}))
    else:
        paper_cfg = {}
        outputs_cfg = {}

    metric_cfg = _as_dict(paper_cfg.get("metric_protocol", {}))
    split_cfg = _as_dict(paper_cfg.get("split_protocol", {}))
    paper_split_cfg = _as_dict(paper_cfg.get("paper_split_protocol", {}))
    source_cfg = _resolve_paper_source_protocol(paper_cfg)

    if not paper_split_cfg:
        target_window = _as_dict(split_cfg.get("target_window", {}))
        target_eval = _as_dict(split_cfg.get("target_eval_split", {}))
        paper_split_cfg = {
            "target_observed_window_days": int(target_window.get("train_val_days", 30)),
            "target_forecast_window_days": int(target_window.get("test_days", 180)),
            "validation_strategy": str(target_eval.get("mode", "time_holdout")),
            "rolling_or_fixed_split": str(target_window.get("kind", "rolling_recent_days")),
            "source_selection_window": "full_history",
            "source_pool_scope": "all_source_items",
        }

    strict_mode = bool(
        paper_cfg.get(
            "strict_paper_mode",
            paper_cfg.get("paper_strict_mode", False),
        )
    )

    protocol = {
        "strict_paper_mode": strict_mode,
        "paper_strict_mode": strict_mode,
        "strict_paper_split": bool(
            paper_cfg.get("strict_paper_split", paper_cfg.get("paper_strict_split", False))
        ),
        "metric_protocol": metric_cfg,
        "split_protocol": split_cfg,
        "paper_split_protocol": paper_split_cfg,
        "paper_source_protocol": source_cfg,
        # Backward compatibility for existing callers.
        "source_pretrained_protocol": source_cfg,
        "strict_dataset_protocol": _as_dict(paper_cfg.get("strict_dataset_protocol", {})),
        "strict_source_selection": _as_dict(
            paper_cfg.get(
                "strict_source_selection",
                {
                    "enforce_ss_tl_knn_top1": True,
                    "enforce_multi_source_topk3": True,
                    "multi_source_top_k": 3,
                    "alignment_status": STATUS_PARTIAL,
                    "notes": "KNN source-selection is enforced in strict mode; some dataset-level region metadata may still be TODO.",
                },
            )
        ),
        "outputs": {
            "experiment_results_dir": outputs_cfg.get("experiment_results_dir", "outputs/experiment_results"),
            "results_reports_dir": outputs_cfg.get("results_reports_dir", "outputs/results_reports"),
            "paper_alignment_dir": outputs_cfg.get("paper_alignment_dir", "outputs/paper_alignment_reports"),
            "paper_results_csv": outputs_cfg.get("paper_results_csv", "paper_results.csv"),
            "full_paper_results_csv": outputs_cfg.get("full_paper_results_csv", "full_paper_results.csv"),
            "extended_results_csv": outputs_cfg.get("extended_results_csv", "extended_results.csv"),
        },
    }
    return protocol


def resolve_strict_paper_mode(config: Optional[Any], explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    protocol = load_paper_protocol(config)
    return bool(protocol.get("strict_paper_mode", protocol.get("paper_strict_mode", False)))


def validate_paper_protocol_config(
    protocol: Dict[str, Any],
    strict_paper_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """Validate protocol config for auditable paper-alignment runs.

    This validates structural constraints and keeps unresolved paper evidence
    explicitly marked as TODO without pretending full paper equivalence.
    """
    metric_cfg = _as_dict(protocol.get("metric_protocol", {}))
    split_cfg = _as_dict(protocol.get("split_protocol", {}))
    paper_split_cfg = _as_dict(protocol.get("paper_split_protocol", {}))
    source_cfg = _as_dict(protocol.get("paper_source_protocol", protocol.get("source_pretrained_protocol", {})))

    strict_mode = bool(
        protocol.get("strict_paper_mode", False) if strict_paper_mode is None else strict_paper_mode
    )

    failures: List[str] = []
    warnings: List[str] = []

    current_metric_space = str(metric_cfg.get("current_metric_space", "")).strip()
    if current_metric_space != "normalized_minmax_space":
        failures.append(
            "metric_protocol.current_metric_space must be 'normalized_minmax_space' "
            f"for current implementation, got '{current_metric_space or 'EMPTY'}'."
        )

    if _is_todo_placeholder(metric_cfg.get("paper_metric_space")):
        warnings.append(
            "TODO: metric_protocol.paper_metric_space is not confirmed from the original paper yet."
        )
    if _is_todo_placeholder(metric_cfg.get("paper_accuracy_definition")):
        warnings.append(
            "TODO: metric_protocol.paper_accuracy_definition is not confirmed from the original paper yet."
        )

    target_window = _as_dict(split_cfg.get("target_window", {}))
    train_val_days = int(target_window.get("train_val_days", 30))
    test_days = int(target_window.get("test_days", 180))
    if train_val_days <= 0 or test_days <= 0:
        failures.append(
            "split_protocol.target_window.train_val_days and test_days must be positive integers."
        )

    if _is_todo_placeholder(split_cfg.get("paper_reference")):
        warnings.append("TODO: split_protocol.paper_reference is not confirmed from the original paper yet.")

    required_split_keys = [
        "target_observed_window_days",
        "target_forecast_window_days",
        "validation_strategy",
        "rolling_or_fixed_split",
        "source_selection_window",
        "source_pool_scope",
    ]
    missing_split_keys = [key for key in required_split_keys if key not in paper_split_cfg]
    if missing_split_keys:
        failures.append(
            "paper_split_protocol missing required keys: " + ", ".join(missing_split_keys)
        )

    cap = int(source_cfg.get("max_pretrained_models_from_paper", 5))
    paper_counts = [int(v) for v in source_cfg.get("paper_source_counts", [1, 3, 5])]
    extended_counts = [int(v) for v in source_cfg.get("extended_source_counts", [6, 9])]

    if cap != 5:
        failures.append(
            "source_pretrained_protocol.paper_max_pretrained_models must remain 5 for paper track "
            f"(got {cap})."
        )
    if not paper_counts:
        failures.append("source_pretrained_protocol.paper_source_counts must not be empty.")
    if any(count > cap for count in paper_counts):
        failures.append(
            "source_pretrained_protocol.paper_source_counts contains values above paper cap 5: "
            f"{paper_counts}."
        )
    if 5 not in paper_counts:
        warnings.append(
            "TODO: paper_source_counts does not include k=5; paper protocol may be incompletely retained."
        )
    overlap = sorted(set(paper_counts).intersection(set(extended_counts)))
    if overlap:
        failures.append(
            "paper_source_counts and extended_source_counts must be disjoint, overlap="
            f"{overlap}."
        )

    status = "PASS"
    if failures:
        status = "FAIL"
    elif warnings:
        status = "WARN"

    report = {
        "status": status,
        "strict_paper_mode": strict_mode,
        "failures": failures,
        "warnings": warnings,
        "checked": {
            "current_metric_space": current_metric_space,
            "target_window_train_val_days": train_val_days,
            "target_window_test_days": test_days,
            "paper_split_protocol": paper_split_cfg,
            "max_pretrained_models_from_paper": cap,
            "paper_source_counts": paper_counts,
            "extended_source_counts": extended_counts,
        },
    }

    if strict_mode and failures:
        raise ValueError(
            "Paper protocol configuration validation failed in strict mode: "
            + " | ".join(failures)
        )

    return report


def get_paper_source_counts(protocol: Dict[str, Any]) -> List[int]:
    source_cfg = _as_dict(protocol.get("paper_source_protocol", protocol.get("source_pretrained_protocol", {})))
    values = source_cfg.get("paper_source_counts", [1, 3, 5])
    return [int(v) for v in values]


def get_extended_source_counts(protocol: Dict[str, Any]) -> List[int]:
    source_cfg = _as_dict(protocol.get("paper_source_protocol", protocol.get("source_pretrained_protocol", {})))
    if not bool(source_cfg.get("allow_extended_source_counts", True)):
        return []
    values = source_cfg.get("extended_source_counts", [6, 9])
    return [int(v) for v in values]


def get_max_pretrained_models(protocol: Dict[str, Any]) -> int:
    source_cfg = _as_dict(protocol.get("paper_source_protocol", protocol.get("source_pretrained_protocol", {})))
    return int(source_cfg.get("max_pretrained_models_from_paper", 5))


def resolve_experiment_track(
    method_name: str,
    requested_source_count: int,
    protocol: Dict[str, Any],
) -> str:
    if method_name not in MULTI_SOURCE_TL_METHODS:
        return "paper"

    paper_counts = set(get_paper_source_counts(protocol))
    max_pretrained_models = get_max_pretrained_models(protocol)
    if int(requested_source_count) <= max_pretrained_models and int(requested_source_count) in paper_counts:
        return "paper"
    return "extended"


def count_pretrained_models(method_name: str, method_meta: Optional[Dict[str, Any]]) -> int:
    meta = _as_dict(method_meta)
    if method_name == "No-TL":
        return 0
    if method_name == "SS-TL":
        return 1

    if isinstance(meta.get("source_models_info"), list):
        return len(meta["source_models_info"])
    if isinstance(meta.get("selected_sources"), list):
        return len(meta["selected_sources"])
    if isinstance(meta.get("individual_results"), list):
        return len(meta["individual_results"])

    return 0


def summarize_split_runtime(base_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    bundle = base_data if isinstance(base_data, dict) else {}
    target_df = bundle.get("target_df") if isinstance(bundle.get("target_df"), pd.DataFrame) else None
    source_df = bundle.get("source_df") if isinstance(bundle.get("source_df"), pd.DataFrame) else None

    target_start = None
    target_end = None
    target_days = None
    target_mode = None
    target_config = {}
    target_expected_days = None
    target_range_days = None
    target_unique_days = None
    target_strict_mode = None
    if target_df is not None and not target_df.empty:
        target_start = pd.Timestamp(target_df["date"].min()).strftime("%Y-%m-%d")
        target_end = pd.Timestamp(target_df["date"].max()).strftime("%Y-%m-%d")
        target_days = int((target_df["date"].max() - target_df["date"].min()).days + 1)
        target_mode = str(target_df.attrs.get("split_mode", "unknown"))
        target_config = _as_dict(target_df.attrs.get("split_config", {}))
        target_expected_days = int(target_df.attrs.get("target_window_expected_days", target_days))
        target_range_days = int(target_df.attrs.get("target_window_range_days", target_days))
        target_unique_days = int(target_df.attrs.get("target_window_unique_days", target_df["date"].nunique()))
        target_strict_mode = bool(target_df.attrs.get("strict_paper_mode", False))

    source_mode = None
    source_config = {}
    if source_df is not None:
        source_mode = str(source_df.attrs.get("split_mode", "unknown"))
        source_config = _as_dict(source_df.attrs.get("split_config", {}))

    return {
        "target_start_date": target_start or "N/A",
        "target_end_date": target_end or "N/A",
        "target_window_days": target_days if target_days is not None else -1,
        "target_split_mode": target_mode or "unknown",
        "target_split_config": target_config,
        "target_window_expected_days": target_expected_days if target_expected_days is not None else -1,
        "target_window_range_days": target_range_days if target_range_days is not None else -1,
        "target_window_unique_days": target_unique_days if target_unique_days is not None else -1,
        "target_strict_paper_mode": bool(target_strict_mode),
        "source_split_mode": source_mode or "unknown",
        "source_split_config": source_config,
    }


def assess_metric_alignment(protocol: Dict[str, Any]) -> Dict[str, str]:
    metric_cfg = _as_dict(protocol.get("metric_protocol", {}))
    notes = str(metric_cfg.get("notes", ""))
    return {
        "metric_alignment_status": str(metric_cfg.get("alignment_status", STATUS_PARTIAL)),
        "paper_metric_space": str(metric_cfg.get("paper_metric_space", "TODO")),
        "current_metric_space": str(metric_cfg.get("current_metric_space", "normalized_minmax_space")),
        "strict_paper_metrics": str(bool(metric_cfg.get("strict_paper_metrics", False))),
        "metric_alignment_notes": notes or "TODO: paper metric space reference not filled.",
    }


def assess_split_alignment(protocol: Dict[str, Any], base_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    split_cfg = _as_dict(protocol.get("split_protocol", {}))
    paper_split_cfg = _as_dict(protocol.get("paper_split_protocol", {}))
    target_window_cfg = _as_dict(split_cfg.get("target_window", {}))
    target_eval_cfg = _as_dict(split_cfg.get("target_eval_split", {}))
    runtime = summarize_split_runtime(base_data)

    expected_window_days = int(paper_split_cfg.get("target_observed_window_days", target_window_cfg.get("train_val_days", 30))) + int(
        paper_split_cfg.get("target_forecast_window_days", target_window_cfg.get("test_days", 180))
    )
    runtime_days = int(runtime.get("target_window_days", -1))
    runtime_expected_days = int(runtime.get("target_window_expected_days", -1))
    runtime_range_days = int(runtime.get("target_window_range_days", -1))
    matches_window = (
        runtime_days in {expected_window_days, -1}
        and runtime_expected_days in {expected_window_days, -1}
        and runtime_range_days in {expected_window_days, -1}
    )
    expected_validation_mode = str(
        paper_split_cfg.get("validation_strategy", target_eval_cfg.get("mode", "ratio"))
    ).lower()
    runtime_mode = str(runtime["target_split_mode"]).lower()
    matches_mode = runtime_mode in {expected_validation_mode, "ratio", "time_holdout"}

    status = str(split_cfg.get("alignment_status", STATUS_PARTIAL))
    if status == STATUS_ALIGNED and not (matches_window and matches_mode):
        status = STATUS_PARTIAL

    return {
        "split_alignment_status": status,
        "paper_split_reference": str(split_cfg.get("paper_reference", "TODO_UNCONFIRMED_FROM_ORIGINAL_PAPER")),
        "split_alignment_notes": str(split_cfg.get("notes", "")) or "TODO: split protocol notes missing.",
        "target_start_date": runtime["target_start_date"],
        "target_end_date": runtime["target_end_date"],
        "target_window_days": runtime["target_window_days"],
        "target_window_expected_days": runtime["target_window_expected_days"],
        "target_window_range_days": runtime["target_window_range_days"],
        "target_window_unique_days": runtime["target_window_unique_days"],
        "target_split_mode": runtime["target_split_mode"],
        "source_split_mode": runtime["source_split_mode"],
        "target_strict_paper_mode": str(runtime.get("target_strict_paper_mode", False)),
        "split_runtime_matches_config": str(bool(matches_window and matches_mode)),
    }


def assess_source_pretrained_alignment(
    method_name: str,
    requested_source_count: int,
    actual_pretrained_model_count: int,
    protocol: Dict[str, Any],
    experiment_track: str,
) -> Dict[str, Any]:
    source_cfg = _as_dict(protocol.get("paper_source_protocol", protocol.get("source_pretrained_protocol", {})))
    cap = get_max_pretrained_models(protocol)
    status = str(source_cfg.get("alignment_status", STATUS_ALIGNED))

    if method_name == "No-TL":
        return {
            "source_pretrained_alignment_status": STATUS_NOT_APPLICABLE,
            "paper_pretrained_model_cap": cap,
            "requested_source_count": 0,
            "actual_pretrained_model_count": 0,
            "source_pretrained_alignment_notes": "No-TL does not use pretrained source models.",
        }

    if experiment_track != "paper":
        status = STATUS_EXTENDED
    elif actual_pretrained_model_count > cap or requested_source_count > cap:
        status = STATUS_PARTIAL

    return {
        "source_pretrained_alignment_status": status,
        "paper_pretrained_model_cap": cap,
        "requested_source_count": int(requested_source_count),
        "actual_pretrained_model_count": int(actual_pretrained_model_count),
        "source_pretrained_alignment_notes": str(source_cfg.get("notes", ""))
        or "TODO: source/pretrained protocol notes missing.",
    }


def summarize_protocol_alignment(
    experiment_track: str,
    metric_status: str,
    split_status: str,
    source_status: str,
) -> str:
    if experiment_track != "paper":
        return STATUS_EXTENDED
    statuses = [metric_status, split_status, source_status]
    if STATUS_TODO in statuses:
        return STATUS_TODO
    if STATUS_PARTIAL in statuses:
        return STATUS_PARTIAL
    if all(status in {STATUS_ALIGNED, STATUS_NOT_APPLICABLE} for status in statuses):
        return STATUS_ALIGNED
    return STATUS_PARTIAL


def build_alignment_fields(
    method_name: str,
    requested_source_count: int,
    method_meta: Optional[Dict[str, Any]],
    base_data: Optional[Dict[str, Any]],
    protocol: Dict[str, Any],
) -> Dict[str, Any]:
    track = resolve_experiment_track(method_name, requested_source_count, protocol)
    actual_pretrained_model_count = count_pretrained_models(method_name, method_meta)
    metric = assess_metric_alignment(protocol)
    split = assess_split_alignment(protocol, base_data)
    source = assess_source_pretrained_alignment(
        method_name=method_name,
        requested_source_count=requested_source_count,
        actual_pretrained_model_count=actual_pretrained_model_count,
        protocol=protocol,
        experiment_track=track,
    )
    alignment_status = summarize_protocol_alignment(
        experiment_track=track,
        metric_status=str(metric["metric_alignment_status"]),
        split_status=str(split["split_alignment_status"]),
        source_status=str(source["source_pretrained_alignment_status"]),
    )
    notes = " | ".join(
        [
            str(metric["metric_alignment_notes"]),
            str(split["split_alignment_notes"]),
            str(source["source_pretrained_alignment_notes"]),
        ]
    )

    return {
        "strict_paper_mode": bool(protocol.get("strict_paper_mode", False)),
        "experiment_track": track,
        "experiment_scope": track,
        "alignment_status": alignment_status,
        "source_protocol_aligned": bool(
            str(source["source_pretrained_alignment_status"]) in {STATUS_ALIGNED, STATUS_NOT_APPLICABLE}
        ),
        "source_count": int(requested_source_count),
        "pretrained_model_count": int(actual_pretrained_model_count),
        "number_of_sources": int(requested_source_count),
        "number_of_pretrained_models": int(actual_pretrained_model_count),
        "number_of_methods": 1,
        **metric,
        **split,
        **source,
        "alignment_notes": notes,
    }


def ensure_paper_track_allowed(
    method_name: str,
    requested_source_count: int,
    protocol: Dict[str, Any],
    strict_paper_mode: bool,
) -> None:
    if not strict_paper_mode:
        return
    if resolve_experiment_track(method_name, requested_source_count, protocol) != "paper":
        raise ValueError(
            "strict paper mode forbids extended experiment settings: "
            f"method={method_name} requested_source_count={requested_source_count}"
        )


def _create_run_output_dir(root: Path) -> Path:
    """Create a unique timestamped run directory and update outputs/latest_run.txt."""
    base_output_dir = root / "outputs"
    run_output_dir = create_run_dir(root, "paper")

    latest_file = base_output_dir / "latest_run.txt"
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(str(run_output_dir), encoding="utf-8")
    return run_output_dir


def get_results_output_paths(root: Path, protocol: Dict[str, Any]) -> Dict[str, Path]:
    outputs = _as_dict(protocol.get("outputs", {}))
    run_dir = _create_run_output_dir(root)
    results_dir = run_dir / "results"
    reports_dir = run_dir / "results_reports"
    alignment_dir = run_dir / "paper_alignment"
    audits_dir = run_dir / "audits"
    for output_dir in (results_dir, reports_dir, alignment_dir, audits_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
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
