"""Check whether runtime split behavior follows paper split assumptions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment.experiment_runner import prepare_base_data_for_experiments
from src.data_processing.data_preprocessing import temporal_split_by_ratio_or_dates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check paper split protocol alignment.")
    parser.add_argument(
        "--strict-paper-split",
        action="store_true",
        help="Force strict paper split mode during validation.",
    )
    parser.add_argument(
        "--day-tolerance",
        type=int,
        default=3,
        help="Allowed tolerance (in days) for observed/forecast window checks.",
    )
    return parser.parse_args()


def _nunique_days(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df["date"].nunique())


def _date_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _ordered_time_ok(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    if train_df.empty or val_df.empty or test_df.empty:
        return False
    train_end = train_df["date"].max()
    val_start = val_df["date"].min()
    val_end = val_df["date"].max()
    test_start = test_df["date"].min()
    return bool(train_end < val_start <= val_end < test_start)


def _resolve_expected_split_days(
    cfg: Dict[str, Any],
    dataset_name: str,
    strict_split_enabled: bool,
    day_tolerance: int,
) -> Dict[str, Any]:
    split_cfg = cfg.get("paper_reproduction", {}).get("paper_split_protocol", {})
    default_observed_days = int(split_cfg.get("target_observed_window_days", 30))
    default_forecast_days = int(split_cfg.get("target_forecast_window_days", 180))

    strict_ds_cfg = cfg.get("paper_reproduction", {}).get("strict_dataset_protocol", {})
    ds_cfg = strict_ds_cfg.get(dataset_name, {})
    split_days = ds_cfg.get("target_split_days", {}) if strict_split_enabled else {}

    expected_train_days = int(split_days.get("train_days", max(default_observed_days - 15, 1)))
    expected_val_days = int(split_days.get("val_days", min(15, default_observed_days)))
    expected_test_days = int(split_days.get("test_days", default_forecast_days))
    expected_observed_days = expected_train_days + expected_val_days

    return {
        "strict_split_enabled": strict_split_enabled,
        "tolerance_days": int(day_tolerance),
        "expected_train_days": expected_train_days,
        "expected_val_days": expected_val_days,
        "expected_observed_days": expected_observed_days,
        "expected_test_days": expected_test_days,
        "expected_forecast_days": expected_test_days,
    }


def _source_target_leakage(source_df: pd.DataFrame, target_df: pd.DataFrame) -> Dict[str, Any]:
    source_item_ids = set(source_df["item_id"].dropna().astype(int).tolist())
    target_item_ids = set(target_df["item_id"].dropna().astype(int).tolist())
    item_overlap = sorted(source_item_ids.intersection(target_item_ids))

    source_keys = set(
        tuple(v)
        for v in source_df[["entity_id", "item_id", "date"]].drop_duplicates().to_numpy().tolist()
    )
    target_keys = set(
        tuple(v)
        for v in target_df[["entity_id", "item_id", "date"]].drop_duplicates().to_numpy().tolist()
    )
    key_overlap_count = int(len(source_keys.intersection(target_keys)))

    return {
        "item_overlap_count": int(len(item_overlap)),
        "item_overlap_examples": item_overlap[:5],
        "source_target_key_overlap_count": key_overlap_count,
        "no_leakage": bool(len(item_overlap) == 0 and key_overlap_count == 0),
    }


def main() -> None:
    args = _parse_args()

    config_path = ROOT / "configs" / "default_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("paper_reproduction", {})
    if args.strict_paper_split:
        cfg["paper_reproduction"]["strict_paper_split"] = True
        cfg["paper_reproduction"]["paper_strict_split"] = True

    strict_split_enabled = bool(
        cfg["paper_reproduction"].get("strict_paper_split", False)
        or cfg["paper_reproduction"].get("paper_strict_split", False)
    )

    rows: List[Dict[str, Any]] = []
    detailed: Dict[str, Any] = {
        "paper_split_protocol": cfg["paper_reproduction"].get("paper_split_protocol", {}),
        "strict_paper_split": strict_split_enabled,
        "datasets": {},
    }

    for dataset_name, dataset_path in cfg["dataset_paths"].items():
        base = prepare_base_data_for_experiments(
            dataset_name=dataset_name,
            data_path=dataset_path,
            config=cfg,
            verbose_mode="summary",
        )
        source_df = base["source_df"]
        target_df = base["target_df"]

        target_train, target_val, target_test = temporal_split_by_ratio_or_dates(target_df)

        expected = _resolve_expected_split_days(
            cfg=cfg,
            dataset_name=dataset_name,
            strict_split_enabled=strict_split_enabled,
            day_tolerance=int(args.day_tolerance),
        )

        train_days = _nunique_days(target_train)
        val_days = _nunique_days(target_val)
        observed_days = _nunique_days(pd.concat([target_train, target_val], ignore_index=True))
        forecast_days = _nunique_days(target_test)

        train_ok = abs(train_days - expected["expected_train_days"]) <= int(args.day_tolerance)
        val_ok = abs(val_days - expected["expected_val_days"]) <= int(args.day_tolerance)
        observed_ok = abs(observed_days - expected["expected_observed_days"]) <= int(args.day_tolerance)
        forecast_ok = abs(forecast_days - expected["expected_forecast_days"]) <= int(args.day_tolerance)

        validation_present = not target_val.empty
        validation_order_ok = _ordered_time_ok(target_train, target_val, target_test)
        validation_ok = bool(validation_present and validation_order_ok)

        leakage = _source_target_leakage(source_df, target_df)
        leakage_ok = bool(leakage["no_leakage"])

        status = (
            "PASS"
            if train_ok and val_ok and observed_ok and forecast_ok and validation_ok and leakage_ok
            else "FAIL"
        )

        row = {
            "dataset": dataset_name,
            "status": status,
            "strict_split_enabled": expected["strict_split_enabled"],
            "expected_train_days": expected["expected_train_days"],
            "actual_train_days": train_days,
            "train_window_ok": train_ok,
            "expected_val_days": expected["expected_val_days"],
            "actual_val_days": val_days,
            "val_window_ok": val_ok,
            "expected_observed_days": expected["expected_observed_days"],
            "actual_observed_days": observed_days,
            "expected_forecast_days": expected["expected_forecast_days"],
            "actual_forecast_days": forecast_days,
            "observed_window_ok": observed_ok,
            "forecast_window_ok": forecast_ok,
            "validation_present": validation_present,
            "validation_order_ok": validation_order_ok,
            "validation_ok": validation_ok,
            "source_target_leakage_ok": leakage_ok,
            "source_target_key_overlap_count": leakage["source_target_key_overlap_count"],
            "target_train_start": _date_text(target_train["date"].min() if not target_train.empty else None),
            "target_train_end": _date_text(target_train["date"].max() if not target_train.empty else None),
            "target_val_start": _date_text(target_val["date"].min() if not target_val.empty else None),
            "target_val_end": _date_text(target_val["date"].max() if not target_val.empty else None),
            "target_test_start": _date_text(target_test["date"].min() if not target_test.empty else None),
            "target_test_end": _date_text(target_test["date"].max() if not target_test.empty else None),
        }
        rows.append(row)

        detailed["datasets"][dataset_name] = {
            "status": status,
            "checks": {
                "target_train_window_ok": train_ok,
                "target_validation_window_ok": val_ok,
                "target_observed_approx_one_month": observed_ok,
                "target_forecast_approx_six_months": forecast_ok,
                "source_target_no_leakage": leakage_ok,
                "validation_exists_and_reasonable": validation_ok,
            },
            "expected_split_days": {
                "train": expected["expected_train_days"],
                "validation": expected["expected_val_days"],
                "test": expected["expected_test_days"],
                "observed": expected["expected_observed_days"],
                "tolerance": expected["tolerance_days"],
            },
            "split_ranges": {
                "target_train": {
                    "start": row["target_train_start"],
                    "end": row["target_train_end"],
                    "days": _nunique_days(target_train),
                },
                "target_validation": {
                    "start": row["target_val_start"],
                    "end": row["target_val_end"],
                    "days": _nunique_days(target_val),
                },
                "target_test_forecast": {
                    "start": row["target_test_start"],
                    "end": row["target_test_end"],
                    "days": _nunique_days(target_test),
                },
            },
            "leakage_details": leakage,
            "notes": "按论文相对窗口复刻",
        }

    out_dir = ROOT / "outputs" / "paper_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_csv = out_dir / "check_paper_split_alignment_report.csv"
    report_json = out_dir / "check_paper_split_alignment_report.json"

    report_df = pd.DataFrame(rows)
    report_df.to_csv(report_csv, index=False, encoding="utf-8")
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=True, indent=2)

    print("Paper split alignment check completed")
    print(report_csv)
    print(report_json)
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
