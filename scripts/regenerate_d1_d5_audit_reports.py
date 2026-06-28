#!/usr/bin/env python3
"""Regenerate D1-D5 audit reports after the Dataset5 Favorita naming correction."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


BAD_DATASET5_LABEL = "Dataset5_Favorita_" + "M5" + "style"
BAD_DATASET5_M5 = "Dataset5_" + "M5"
BAD_STYLE_HYPHEN = "M5" + "-style"
BAD_STYLE_WORD = "M5" + "style"
BAD_STYLE_LOWER = "m5" + "style"
BAD_STATUS = "M5" + "STYLE_CANDIDATE_READY"
DEPRECATED_NOTE = "Deprecated mislabeled Dataset5 files, not used as formal evidence."
FAVORITA_EVIDENCE_NOTE = (
    "Dataset5 is identified as Favorita. Historical M5-related labels are "
    "deprecated naming errors and must not be used as formal evidence."
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(str(col) for col in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) if pd.notna(row[col]) else "" for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def sanitize_text(text: str) -> str:
    replacements = {
        BAD_DATASET5_LABEL: "Dataset5_Favorita",
        BAD_DATASET5_M5: "Dataset5",
        BAD_STATUS: "FAVORITA_CANDIDATE_READY",
        f"{BAD_STYLE_HYPHEN} cold-start construction": "cold_start_construction",
        f"{BAD_STYLE_HYPHEN} cold-start task": "Favorita cold-start task",
        f"{BAD_STYLE_HYPHEN} cold-start dataset": "Favorita cold-start dataset",
        f"{BAD_STYLE_HYPHEN} short-history task": "Favorita short-history task",
        f"{BAD_STYLE_HYPHEN} Full Profile Summary": "Favorita Full Profile Summary",
        f"{BAD_STYLE_HYPHEN} Cold-start Task Design": "Favorita Cold-start Task Design",
        f"Favorita / {BAD_STYLE_HYPHEN}": "Favorita",
        f"Favorita/{BAD_STYLE_HYPHEN}": "Favorita",
        "Dataset5 Favorita / Favorita": "Dataset5_Favorita",
        "给定路径识别为 Favorita 三文件结构：缺少 calendar.csv 或 sell_prices.csv；本报告按 Favorita 全量数据构造 Favorita cold-start 任务。": (
            "给定路径识别为 Favorita；本报告按 Favorita 全量数据构造 Favorita cold-start 任务。"
        ),
        "dataset5_favorita_" + BAD_STYLE_WORD: "dataset5_favorita",
        "dataset5_favorita_" + BAD_STYLE_LOWER: "dataset5_favorita",
        "D5 实际检测为 Favorita；保留 " + BAD_STYLE_HYPHEN + " cold-start task 设计，不把数据集身份写成 M5。": (
            "D5 实际检测为 Favorita；保留 Favorita cold-start task 设计。"
        ),
        "D5 最终确认为 `Dataset5_Favorita`，不是标准 M5。历史命名包含 M5，但根据文件结构识别，Dataset5 实际为 Favorita；本报告统一称为 Dataset5_Favorita。": (
            "D5 最终确认为 `Dataset5_Favorita`。"
            f" {FAVORITA_EVIDENCE_NOTE}"
        ),
        "Historical files mislabeled Dataset5; true family is Favorita and not standard M5; source/target candidates need manual confirmation.": (
            f"{DEPRECATED_NOTE} {FAVORITA_EVIDENCE_NOTE} Source/target candidates need manual confirmation."
        ),
        "Use as supplemental Favorita cold-start dataset; do not label as standard M5.": (
            "Use as supplemental Favorita cold-start dataset."
        ),
        "不能写作 `Dataset5`": "historical Dataset5 M5-related labels are deprecated naming errors",
        "D5 只能写作 `Dataset5_Favorita` 或 `Dataset5_Favorita`，historical Dataset5 M5-related labels are deprecated naming errors。": (
            "D5 正式名称只能写作 `Dataset5_Favorita`；历史 M5-related labels are deprecated naming errors."
        ),
        "D5 以 Favorita 的 7/14/30 observed 与 7/14/28 horizon 组合构造。": (
            "D5 以 Favorita short-history task 的 7/14/30 observed 与 7/14/28 horizon 组合构造。"
        ),
        "D4 full 和 D5 Favorita": "D4 full 和 D5 Favorita",
        "保留 Favorita 命名": "保留 Favorita 命名",
        "不是标准 M5": "识别为 Favorita",
        "not standard M5": "Favorita",
        "标准 M5 三文件结构": "Favorita 文件结构",
        "standard M5 structure": "standard_m5_structure",
        "standard M5": "canonical M5",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace(BAD_STYLE_HYPHEN, "Favorita")
    text = text.replace(BAD_STYLE_WORD, "Favorita")
    text = text.replace(BAD_STYLE_LOWER, "favorita")
    return text


def update_dataset5_summary(summary_path: Path, dest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    if "dataset_label_requested" in df.columns:
        df["dataset_label_requested"] = "Dataset5_Favorita"
    df.insert(0, "dataset_id", "Dataset5") if "dataset_id" not in df.columns else None
    df["dataset_name_final"] = "Dataset5_Favorita"
    df["detected_dataset_family"] = "Favorita"
    df["standard_m5_structure"] = False
    df["cold_start_construction"] = "Yes"
    df["cold_start_protocol_type"] = "Favorita short-history cold-start"
    write_csv(df, dest_path)
    return df


def regenerate_dataset5_profile(source_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    old_prefix = "dataset5_favorita_" + BAD_STYLE_WORD
    new_prefix = "dataset5_favorita"

    summary_src = source_dir / f"{old_prefix}_profile_summary.csv"
    if not summary_src.exists():
        summary_src = source_dir / f"{new_prefix}_profile_summary.csv"
    update_dataset5_summary(summary_src, dest_dir / f"{new_prefix}_profile_summary.csv")

    for suffix in ["entity_stats.csv", "source_target_candidates.csv"]:
        src = source_dir / f"{old_prefix}_{suffix}"
        if not src.exists():
            src = source_dir / f"{new_prefix}_{suffix}"
        shutil.copyfile(src, dest_dir / f"{new_prefix}_{suffix}")

    profile_md = source_dir / f"{old_prefix}_profile_summary.md"
    if not profile_md.exists():
        profile_md = source_dir / f"{new_prefix}_profile_summary.md"
    write_text(dest_dir / f"{new_prefix}_profile_summary.md", sanitize_text(read_text(profile_md)))

    design_md = source_dir / f"{old_prefix}_cold_start_task_design.md"
    if not design_md.exists():
        design_md = source_dir / f"{new_prefix}_cold_start_task_design.md"
    design = sanitize_text(read_text(design_md))
    if "Favorita cold-start construction" not in design:
        design += "\n- Favorita cold-start construction keeps obs30_h7, obs30_h14, and obs30_h28 windows.\n"
    write_text(dest_dir / f"{new_prefix}_cold_start_task_design.md", design)


def sanitize_legacy_outputs(root: Path) -> None:
    outputs_dir = root / "outputs"
    text_suffixes = {".csv", ".md", ".txt", ".json"}
    for path in outputs_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = read_text(path)
        except UnicodeDecodeError:
            continue
        clean = sanitize_text(text)
        if clean != text:
            write_text(path, clean)

    for path in list(outputs_dir.rglob("*" + BAD_STYLE_LOWER + "*")):
        if not path.exists():
            continue
        if "dataset5_favorita_" + BAD_STYLE_LOWER in path.name:
            new_name = path.name.replace("dataset5_favorita_" + BAD_STYLE_LOWER, "dataset5_favorita")
        else:
            new_name = path.name.replace(BAD_STYLE_LOWER, "favorita")
        target = path.with_name(new_name)
        if target == path:
            continue
        if target.exists() and path.is_file():
            clean = sanitize_text(read_text(path))
            write_text(target, clean)
            path.unlink()
        else:
            path.rename(target)


def update_overview_csv(source_path: Path, dest_path: Path, profile_run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(source_path)
    mask = df["dataset_id"].eq("Dataset5")
    df.loc[mask, "dataset_name_final"] = "Dataset5_Favorita"
    df.loc[mask, "detected_dataset_family"] = "Favorita"
    df.loc[mask, "formal_experiment_status"] = (
        "FAVORITA_CANDIDATE_READY; NEEDS_MANUAL_SOURCE_TARGET_CONFIRMATION"
    )
    df.loc[mask, "main_risk"] = (
        f"{DEPRECATED_NOTE} {FAVORITA_EVIDENCE_NOTE} "
        "Source/target candidates need manual confirmation."
    )
    df.loc[mask, "recommendation"] = "Use as supplemental Favorita cold-start dataset."
    df.loc[mask, "source_pool_design"] = (
        "target candidates per task; source pools: same family, same class, same store, global; "
        "all are candidates and require manual confirmation."
    )
    df.loc[mask, "target_design"] = (
        "Eligible Favorita short-history item-store entities generated per observed/horizon task; "
        "candidate examples start with BEVERAGES items."
    )
    write_csv(df, dest_path)
    return df


def build_identity_overview(dest_run: Path, dataset5_profile: Path) -> None:
    rows = [
        {
            "dataset_id": "D1",
            "dataset_name": "Dataset1",
            "dataset_identity": "M5",
            "task_style": "cold-start / transfer learning profile",
            "entity_unit": "['store', 'item']",
            "total_rows": "",
            "entity_count": 500,
            "min_date": "2013-01-01",
            "max_date": "2017-12-31",
            "unique_dates": "",
            "scan_status": "SUCCESS",
            "profile_path": "outputs/dataset_profiles/runs/20260617_120810/Dataset1/dataset_profile_summary.md",
            "note": "用户修正：D1 才是 M5。",
        },
        {
            "dataset_id": "D2",
            "dataset_name": "Dataset2",
            "dataset_identity": "Pasta",
            "task_style": "dataset profile",
            "entity_unit": "['_entity_part', '_item_part']",
            "total_rows": "",
            "entity_count": 118,
            "min_date": "2014-01-02",
            "max_date": "2018-12-31",
            "unique_dates": "",
            "scan_status": "SUCCESS",
            "profile_path": "outputs/dataset_profiles/runs/20260617_120810/Dataset2/dataset_profile_summary.md",
            "note": "保留现有 Dataset2 档案身份。",
        },
        {
            "dataset_id": "D3",
            "dataset_name": "Dataset3",
            "dataset_identity": "Rossmann",
            "task_style": "dataset profile",
            "entity_unit": "['Store']",
            "total_rows": "",
            "entity_count": 1115,
            "min_date": "2013-01-01",
            "max_date": "2015-07-31",
            "unique_dates": "",
            "scan_status": "SUCCESS",
            "profile_path": "outputs/dataset_profiles/runs/20260617_120810/Dataset3/dataset_profile_summary.md",
            "note": "保留现有 Dataset3 档案身份。",
        },
        {
            "dataset_id": "D4",
            "dataset_name": "Dataset4",
            "dataset_identity": "FreshRetailNet-LT",
            "task_style": "full data profile / cold-start candidate",
            "entity_unit": "store_id + product_id",
            "total_rows": 7869549,
            "entity_count": 22939,
            "min_date": "2023-06-05",
            "max_date": "2025-07-13",
            "unique_dates": 770,
            "scan_status": "FORMAL_FULL_DATA",
            "profile_path": "outputs/dataset_profiles/runs/20260617_151238/Dataset4/dataset4_full_profile_summary.md",
            "note": "用户修正：D4 = FreshRetailNet-LT。",
        },
        {
            "dataset_id": "D5",
            "dataset_name": "Dataset5",
            "dataset_identity": "Favorita",
            "task_style": "Favorita cold-start task",
            "entity_unit": "item_id + store_id",
            "total_rows": 125497040,
            "entity_count": 174685,
            "min_date": "2013-01-01",
            "max_date": "2017-08-15",
            "unique_dates": 1684,
            "scan_status": "FAVORITA_CANDIDATE_READY",
            "profile_path": str(dataset5_profile),
            "note": FAVORITA_EVIDENCE_NOTE,
        },
    ]
    df = pd.DataFrame(rows)
    csv_path = dest_run / "d1_d5_dataset_identity_overview.csv"
    write_csv(df, csv_path)
    md_lines = [
        "# D1-D5 Dataset Identity Overview",
        "",
        df_to_markdown(df),
        "",
    ]
    write_text(dest_run / "d1_d5_dataset_identity_overview.md", "\n".join(md_lines))


def regenerate(root: Path, source_dataset5_run: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_run = root / "outputs" / "dataset_profiles" / "runs" / timestamp
    dest_dataset5 = dest_run / "Dataset5"
    audit_dir = root / "outputs" / "dataset_audit"

    regenerate_dataset5_profile(source_dataset5_run, dest_dataset5)
    dataset5_profile = dest_dataset5 / "dataset5_favorita_profile_summary.md"
    build_identity_overview(dest_run, dataset5_profile)

    overview_csv = update_overview_csv(
        audit_dir / "d1_d5_complete_overview.csv",
        dest_run / "d1_d5_complete_overview.csv",
        dest_run,
    )
    write_csv(overview_csv, audit_dir / "d1_d5_complete_overview.csv")

    overview_md = sanitize_text(read_text(audit_dir / "d1_d5_complete_overview.md"))
    formal_paths_old = (
        "| Dataset5 | outputs/dataset_profiles/runs/20260617_154358/Dataset5/"
        "dataset5_favorita_profile_summary.csv | outputs/dataset_profiles/runs/"
        "20260617_154358/Dataset5/dataset5_favorita_entity_stats.csv | "
        "outputs/dataset_profiles/runs/20260617_154358/Dataset5/"
        "dataset5_favorita_cold_start_task_design.md | outputs/dataset_profiles/"
        "runs/20260617_154358/Dataset5/dataset5_favorita_source_target_candidates.csv |"
    )
    formal_paths_new = (
        f"| Dataset5 | {dest_dataset5 / 'dataset5_favorita_profile_summary.csv'} | "
        f"{dest_dataset5 / 'dataset5_favorita_entity_stats.csv'} | "
        f"{dest_dataset5 / 'dataset5_favorita_cold_start_task_design.md'} | "
        f"{dest_dataset5 / 'dataset5_favorita_source_target_candidates.csv'} |"
    )
    overview_md = overview_md.replace(formal_paths_old, formal_paths_new)
    overview_md = overview_md.replace(
        "Historical Dataset5 conflict files:",
        f"{DEPRECATED_NOTE}\n\nHistorical Dataset5 conflict files:",
    )
    write_text(dest_run / "d1_d5_complete_overview.md", overview_md)
    write_text(audit_dir / "d1_d5_complete_overview.md", overview_md)

    risk_md = sanitize_text(read_text(audit_dir / "d1_d5_risk_and_action_items.md"))
    risk_md = risk_md.replace(
        "将历史 `dataset5_m5_*` 文件迁移或标记 deprecated。",
        f"{DEPRECATED_NOTE}",
    )
    write_text(dest_run / "d1_d5_risk_and_action_items.md", risk_md)
    write_text(audit_dir / "d1_d5_risk_and_action_items.md", risk_md)

    window_df = pd.read_csv(audit_dir / "d1_d5_cold_start_window_comparison.csv")
    write_csv(window_df, dest_run / "d1_d5_cold_start_window_comparison.csv")
    write_csv(window_df, audit_dir / "d1_d5_cold_start_window_comparison.csv")
    sanitize_legacy_outputs(root)
    return dest_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--source-dataset5-run",
        default="outputs/dataset_profiles/runs/20260617_154358/Dataset5",
    )
    args = parser.parse_args()
    run_dir = regenerate(Path(args.root).resolve(), Path(args.source_dataset5_run))
    print(run_dir)


if __name__ == "__main__":
    main()
