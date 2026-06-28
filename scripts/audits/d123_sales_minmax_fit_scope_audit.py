"""Read-only D1/D2/D3 sales MinMaxScaler fit-scope audit."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing import (  # noqa: E402
    STRICT_DATASET_PROTOCOL,
    build_source_target_split,
    extract_datetime_features,
    load_dataset,
    temporal_split_by_ratio_or_dates,
)
from dataset_registry import (  # noqa: E402
    get_dataset_display_name,
    get_default_dataset_path,
)


OUTPUT_DIR = ROOT / "outputs" / "audits"
DETAIL_CSV = OUTPUT_DIR / "d123_sales_minmax_fit_scope_audit.csv"
SUMMARY_CSV = OUTPUT_DIR / "d123_sales_minmax_fit_scope_summary.csv"
REPORT_MD = OUTPUT_DIR / "d123_sales_minmax_fit_scope_audit.md"

DATASETS = ["Dataset1", "Dataset2", "Dataset3"]
DATASET_ID = {"Dataset1": 1, "Dataset2": 2, "Dataset3": 3}

DETAIL_COLUMNS = [
    "dataset_id",
    "dataset_name",
    "dataset_display_name",
    "fit_scope",
    "source_path",
    "target_store",
    "target_entity",
    "target_domain",
    "per_store_sales_min",
    "per_store_sales_max",
    "sample_count",
    "included_store_count",
    "included_store_list",
    "global_sales_min",
    "global_sales_max",
    "global_sample_count",
    "data_scope",
]

SUMMARY_COLUMNS = [
    "dataset_id",
    "dataset_name",
    "target_store",
    "per_store_sales_min",
    "per_store_sales_max",
    "global_sales_min",
    "global_sales_max",
    "min_difference",
    "max_difference",
    "range_ratio",
    "is_same_scope",
]


def _strict_config(dataset_name: str) -> Dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "paper_reproduction": {
            "strict_paper_mode": True,
            "strict_paper_split": True,
        },
    }


def _fmt_number(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.10g}"


def _target_labels(dataset_name: str) -> Tuple[str, str, str]:
    spec = STRICT_DATASET_PROTOCOL[dataset_name]
    if dataset_name == "Dataset1":
        store = f"store={spec['target_entity_id']}, item={spec['target_item_id']}"
        return store, str(spec["target_entity_id"]), ""
    if dataset_name == "Dataset2":
        store = f"entity={spec['target_entity_id']}, item={spec['target_item_id']}"
        return store, str(spec["target_entity_id"]), ""
    store = f"store={spec['target_store_id']}"
    return store, "Region 1", "Region 1"


def _entity_unit_columns(df: pd.DataFrame, dataset_name: str) -> List[str]:
    if dataset_name == "Dataset3":
        return ["store_id"] if "store_id" in df.columns else ["item_id"]
    return ["entity_id", "item_id"]


def _unit_list(df: pd.DataFrame, dataset_name: str) -> List[str]:
    cols = _entity_unit_columns(df, dataset_name)
    units = df[cols].drop_duplicates().sort_values(cols)
    if dataset_name == "Dataset3":
        return [f"store={int(row[cols[0]])}" for _, row in units.iterrows()]
    return [f"entity={row['entity_id']},item={int(row['item_id'])}" for _, row in units.iterrows()]


def _summarize_units(units: Sequence[str], limit: int = 30) -> str:
    if len(units) <= limit:
        return "|".join(units)
    head = "|".join(units[:limit])
    return f"{head}|... ({len(units)} total)"


def _sales_minmax(df: pd.DataFrame) -> Tuple[float, float, int]:
    sales = pd.to_numeric(df["sales"], errors="coerce").dropna()
    if sales.empty:
        raise ValueError("No numeric sales values available for min/max audit.")
    return float(sales.min()), float(sales.max()), int(sales.shape[0])


def _date_range_text(df: pd.DataFrame) -> str:
    if df.empty or "date" not in df.columns:
        return "date range unavailable"
    return f"{df['date'].min().date()} to {df['date'].max().date()}, unique_dates={df['date'].nunique()}"


def _build_dataset_rows(dataset_name: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    dataset_id = DATASET_ID[dataset_name]
    source_path = ROOT / get_default_dataset_path(dataset_name)
    display_name = get_dataset_display_name(dataset_name)
    target_store, target_entity, target_domain = _target_labels(dataset_name)

    loaded = load_dataset(dataset_name, str(source_path))
    featured = extract_datetime_features(loaded)
    source_df, target_df = build_source_target_split(featured, _strict_config(dataset_name))
    target_train, target_val, _target_test = temporal_split_by_ratio_or_dates(target_df)
    target_fit_df = pd.concat([target_train, target_val], ignore_index=True)

    per_min, per_max, per_count = _sales_minmax(target_fit_df)
    global_min, global_max, global_count = _sales_minmax(featured)

    units = _unit_list(featured, dataset_name)
    included_store_list = _summarize_units(units)

    per_scope = (
        "current target MinMaxScaler fit proxy: strict-paper target train+val observed window "
        f"after load_dataset/extract_datetime_features/build_source_target_split; {_date_range_text(target_fit_df)}; "
        "test window excluded because normalize_features fits on train+val."
    )
    global_scope = (
        "global comparison: cleaned and standardized full dataset after load_dataset/extract_datetime_features; "
        f"all available stores/entities/items/domains merged; {_date_range_text(featured)}."
    )

    common = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "dataset_display_name": display_name,
        "source_path": str(source_path),
        "target_store": target_store,
        "target_entity": target_entity,
        "target_domain": target_domain,
    }

    per_row = {
        **common,
        "fit_scope": "per_store",
        "per_store_sales_min": per_min,
        "per_store_sales_max": per_max,
        "sample_count": per_count,
        "included_store_count": "",
        "included_store_list": "",
        "global_sales_min": "",
        "global_sales_max": "",
        "global_sample_count": "",
        "data_scope": per_scope,
    }
    global_row = {
        **common,
        "fit_scope": "global",
        "per_store_sales_min": "",
        "per_store_sales_max": "",
        "sample_count": "",
        "included_store_count": len(units),
        "included_store_list": included_store_list,
        "global_sales_min": global_min,
        "global_sales_max": global_max,
        "global_sample_count": global_count,
        "data_scope": global_scope,
    }

    global_range = global_max - global_min
    per_range = per_max - per_min
    range_ratio = per_range / global_range if global_range != 0 else math.nan
    is_same_scope = bool(per_min == global_min and per_max == global_max)
    summary_row = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "target_store": target_store,
        "per_store_sales_min": per_min,
        "per_store_sales_max": per_max,
        "global_sales_min": global_min,
        "global_sales_max": global_max,
        "min_difference": per_min - global_min,
        "max_difference": per_max - global_max,
        "range_ratio": range_ratio,
        "is_same_scope": is_same_scope,
    }
    return per_row, global_row, summary_row


def build_audit() -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for dataset_name in DATASETS:
        per_row, global_row, summary_row = _build_dataset_rows(dataset_name)
        detail_rows.extend([per_row, global_row])
        summary_rows.append(summary_row)
    details = pd.DataFrame(detail_rows, columns=DETAIL_COLUMNS)
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    return details, summary


def build_markdown_report(details: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines: List[str] = [
        "# D1/D2/D3 Sales MinMaxScaler Fit Scope Audit",
        "",
        "## 1. 审计目的",
        "检查 D1/D2/D3 的 sales MinMaxScaler fit scope：当前 target per-store train+val observed window 口径，和 full cleaned dataset global 口径是否一致。",
        "",
        "## 2. 数据来源路径",
    ]
    for _, row in details.drop_duplicates("dataset_name").iterrows():
        lines.append(f"- {row['dataset_name']} ({row['dataset_display_name']}): `{row['source_path']}`")

    lines.extend(["", "## 3. D1/D2/D3 的 target store"])
    for _, row in details.drop_duplicates("dataset_name").iterrows():
        domain = f", target_domain={row['target_domain']}" if str(row["target_domain"]) else ""
        lines.append(
            f"- {row['dataset_name']}: target_store={row['target_store']}, "
            f"target_entity={row['target_entity']}{domain}"
        )

    lines.extend(["", "## 4. Per-store fit 的 sales_min / sales_max"])
    for _, row in details[details["fit_scope"] == "per_store"].iterrows():
        lines.append(
            f"- {row['dataset_name']}: min={_fmt_number(row['per_store_sales_min'])}, "
            f"max={_fmt_number(row['per_store_sales_max'])}, sample_count={row['sample_count']}; "
            f"scope: {row['data_scope']}"
        )

    lines.extend(["", "## 5. Global fit 的 sales_min / sales_max"])
    for _, row in details[details["fit_scope"] == "global"].iterrows():
        lines.append(
            f"- {row['dataset_name']}: min={_fmt_number(row['global_sales_min'])}, "
            f"max={_fmt_number(row['global_sales_max'])}, global_sample_count={row['global_sample_count']}, "
            f"included_store_count={row['included_store_count']}; scope: {row['data_scope']}"
        )

    lines.extend(["", "## 6. Per-store 与 Global 差异"])
    for _, row in summary.iterrows():
        same = "TRUE" if bool(row["is_same_scope"]) else "FALSE"
        lines.append(
            f"- {row['dataset_name']}: min_difference={_fmt_number(row['min_difference'])}, "
            f"max_difference={_fmt_number(row['max_difference'])}, "
            f"range_ratio={_fmt_number(row['range_ratio'])}, is_same_scope={same}"
        )

    any_diff = any(not bool(v) for v in summary["is_same_scope"].tolist())
    lines.extend(
        [
            "",
            "## 7. 是否可能解释当前 normalized RMSE 与论文 RMSE 的差异",
        ]
    )
    if any_diff:
        lines.append(
            "可能。normalized RMSE 对 sales 的 MinMax range 敏感；如果论文使用 global fit，而当前流程使用 target per-store train+val observed window fit，"
            "相同原始误差会被不同分母缩放，进而改变 normalized RMSE。此审计只能证明 scaler sales range 口径存在或不存在差异，不能单独证明论文一定使用 global fit。"
        )
    else:
        lines.append(
            "从 sales_min/sales_max 看，本次审计没有发现 per-store 与 global range 差异；该因素不太可能单独解释 normalized RMSE 与论文 RMSE 的差异。"
        )

    lines.extend(
        [
            "",
            "## 8. 只读与主实验代码说明",
            "本审计只读读取数据和现有预处理函数，只新增 `scripts/audits/`、`outputs/audits/`、`tests/` 下的文件。",
            "未修改主实验代码，未修改 KNN/RFE/CNN/TL/No-TL/数据清洗逻辑，未覆盖主实验结果。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(details: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = output_dir / DETAIL_CSV.name
    summary_csv = output_dir / SUMMARY_CSV.name
    report_md = output_dir / REPORT_MD.name
    details.to_csv(detail_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    report_md.write_text(build_markdown_report(details, summary), encoding="utf-8")
    return detail_csv, summary_csv, report_md


def print_terminal_summary(detail_csv: Path, summary_csv: Path, report_md: Path, summary: pd.DataFrame) -> None:
    print("Generated files:")
    print(f"- {detail_csv}")
    print(f"- {summary_csv}")
    print(f"- {report_md}")
    print("")
    print("Sales MinMax by Dataset:")
    for _, row in summary.iterrows():
        same = "TRUE" if bool(row["is_same_scope"]) else "FALSE"
        print(
            f"- {row['dataset_name']} target {row['target_store']}: "
            f"per-store min={_fmt_number(row['per_store_sales_min'])}, "
            f"max={_fmt_number(row['per_store_sales_max'])}; "
            f"global min={_fmt_number(row['global_sales_min'])}, "
            f"max={_fmt_number(row['global_sales_max'])}; "
            f"same_scope={same}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for audit-only outputs. Defaults to outputs/audits.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    details, summary = build_audit()
    detail_csv, summary_csv, report_md = write_outputs(details, summary, args.output_dir)
    print_terminal_summary(detail_csv, summary_csv, report_md, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
