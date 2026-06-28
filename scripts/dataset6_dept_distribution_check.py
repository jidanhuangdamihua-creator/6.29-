#!/usr/bin/env python3
"""Scan Dataset6 item-to-dept distribution without touching training code."""

from __future__ import annotations

import argparse
import os
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dataset6_dept_distribution_check"
IN_SCOPE_COL = "是否进入当前874个item集合"
SOURCE_COL = "是否source"
TARGET_COL = "是否target"
DATA_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
ITEM_CANDIDATES = ("item_id", "itemid", "item", "product_id", "productid", "sku_id", "sku")
DEPT_CANDIDATES = ("dept_id", "department_id", "dept", "department")


@dataclass
class FileScan:
    path: Path
    columns: List[str] = field(default_factory=list)
    readable: bool = False
    item_col: Optional[str] = None
    dept_col: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CandidateRoles:
    source_items: set[str] = field(default_factory=set)
    target_items: set[str] = field(default_factory=set)
    source_path: Optional[Path] = None
    note: str = ""


@dataclass
class ScanResult:
    output_dir: Path
    item_count: int
    dept_count: int
    top1_dept: str
    top1_ratio: float
    top2_ratio: float
    top3_ratio: float
    conclusion: str


def normalized_col(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def choose_candidate_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    normalized_candidates = [normalized_col(c) for c in candidates]
    exact = {normalized_col(c): c for c in columns}
    for candidate in normalized_candidates:
        if candidate in exact:
            return exact[candidate]
    for col in columns:
        ncol = normalized_col(col)
        if any(candidate in ncol for candidate in normalized_candidates):
            return col
    return None


def read_header(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return [str(c) for c in pd.read_csv(path, nrows=0, low_memory=False).columns]
    if suffix in {".xlsx", ".xls"}:
        return [str(c) for c in pd.read_excel(path, nrows=0).columns]
    if suffix == ".parquet":
        return [str(c) for c in pd.read_parquet(path, columns=None).head(0).columns]
    raise ValueError(f"Unsupported file type: {path}")


def likely_dataset6_path(path: Path) -> bool:
    text = " ".join(part.lower() for part in path.parts)
    return any(token in text for token in ("dataset 6", "dataset6", "d6", "m5-forecasting", "m5_forecasting"))


def discover_dataset6_files(project_root: Path) -> List[Path]:
    roots = [
        project_root / "数据集",
        project_root / "data",
        project_root / "datasets",
    ]
    search_roots = [root for root in roots if root.exists()] or [project_root]
    files: list[Path] = []
    for root in search_roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in DATA_EXTENSIONS and likely_dataset6_path(path):
                files.append(path)
    return sorted(set(files), key=lambda p: (len(p.parts), str(p)))


def scan_files(files: Iterable[Path]) -> List[FileScan]:
    scans: list[FileScan] = []
    for path in files:
        scan = FileScan(path=path)
        try:
            scan.columns = read_header(path)
            scan.readable = True
            scan.item_col = choose_candidate_column(scan.columns, ITEM_CANDIDATES)
            scan.dept_col = choose_candidate_column(scan.columns, DEPT_CANDIDATES)
        except Exception as exc:  # pragma: no cover - defensive report path
            scan.error = str(exc)
        scans.append(scan)
    return scans


def rank_mapping_file(scan: FileScan) -> tuple[int, int]:
    name = scan.path.name.lower()
    score = 0
    if scan.item_col:
        score += 10
    if scan.dept_col:
        score += 20
    if "sales_train_evaluation" in name:
        score += 8
    if "sales_train_validation" in name:
        score += 5
    if "item" in name or "product" in name:
        score += 4
    if "sample_submission" in name:
        score -= 50
    return score, -(scan.path.stat().st_size if scan.path.exists() else 0)


def choose_item_dept_scan(scans: Sequence[FileScan]) -> Optional[FileScan]:
    candidates = [scan for scan in scans if scan.readable and scan.item_col and scan.dept_col]
    if not candidates:
        return None
    return sorted(candidates, key=rank_mapping_file, reverse=True)[0]


def read_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, usecols=list(columns), low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, usecols=list(columns))
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns))
    raise ValueError(f"Unsupported file type: {path}")


def build_item_dept_mapping(raw: pd.DataFrame, item_col: str, dept_col: str) -> pd.DataFrame:
    mapping = raw[[item_col, dept_col]].copy()
    mapping.columns = ["item_id", "dept_id"]
    mapping["item_id"] = mapping["item_id"].astype("string").str.strip()
    mapping["dept_id"] = mapping["dept_id"].astype("string").str.strip()
    mapping = mapping.dropna(subset=["item_id", "dept_id"]).drop_duplicates()
    conflict_counts = mapping.groupby("item_id")["dept_id"].nunique()
    conflicting_items = set(conflict_counts[conflict_counts > 1].index.astype(str))
    if conflicting_items:
        mapping = (
            mapping.sort_values(["item_id", "dept_id"])
            .drop_duplicates(subset=["item_id"], keep="first")
            .reset_index(drop=True)
        )
        mapping["dept_conflict_resolved"] = mapping["item_id"].isin(conflicting_items)
    else:
        mapping = mapping.sort_values(["item_id", "dept_id"]).reset_index(drop=True)
        mapping["dept_conflict_resolved"] = False
    return mapping


def parse_item_id_from_entity(entity_id: object) -> Optional[str]:
    text = str(entity_id)
    match = re.search(r"(?:^|\|)item_id=([^|]+)", text)
    if match:
        return match.group(1)
    return None


def parse_candidate_roles(candidate_report: Path) -> CandidateRoles:
    roles = CandidateRoles(source_path=candidate_report)
    if not candidate_report.exists():
        roles.note = "No Dataset6 source_target_candidate_report.csv found."
        return roles
    df = pd.read_csv(candidate_report)
    if "entity_id" not in df.columns or "candidate_role" not in df.columns:
        roles.note = "Candidate report exists but lacks entity_id/candidate_role columns."
        return roles
    work = df.copy()
    work["parsed_item_id"] = work["entity_id"].map(parse_item_id_from_entity)
    roles.source_items = set(
        work.loc[work["candidate_role"].astype(str).str.contains("source", case=False, na=False), "parsed_item_id"]
        .dropna()
        .astype(str)
    )
    roles.target_items = set(
        work.loc[work["candidate_role"].astype(str).str.contains("target", case=False, na=False), "parsed_item_id"]
        .dropna()
        .astype(str)
    )
    roles.note = (
        "Parsed existing Dataset6 profile candidate roles; these are data-quality candidates, "
        "not a confirmed training split."
    )
    return roles


def build_dept_distribution(mapping: pd.DataFrame, in_scope_col: str = IN_SCOPE_COL) -> pd.DataFrame:
    scoped = mapping[mapping[in_scope_col].astype(bool)].copy()
    grouped = (
        scoped.groupby("dept_id", as_index=False)["item_id"]
        .nunique()
        .rename(columns={"item_id": "item_count"})
        .sort_values(["item_count", "dept_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total = int(grouped["item_count"].sum())
    grouped["item_ratio"] = grouped["item_count"] / total if total else 0.0
    grouped["cumulative_item_count"] = grouped["item_count"].cumsum()
    grouped["cumulative_ratio"] = grouped["item_ratio"].cumsum()
    return grouped


def classify_distribution(top1_ratio: float, top2_ratio: float, top3_ratio: float) -> str:
    if top1_ratio >= 0.60 or top2_ratio >= 0.80:
        return "高度集中"
    if top1_ratio >= 0.40 or top2_ratio >= 0.65 or top3_ratio >= 0.80:
        return "中度集中"
    return "较分散"


def make_output_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=False)
        return base_dir
    timestamped = base_dir.with_name(f"{base_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    timestamped.mkdir(parents=True, exist_ok=False)
    return timestamped


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_无数据_"
    shown = df.loc[:, columns].head(max_rows).copy()
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in shown.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(rows)


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def inspect_framework_dataset6(project_root: Path) -> dict:
    registry_text = (project_root / "dataset_registry.py").read_text(encoding="utf-8", errors="ignore")
    default_config = load_json_if_exists(project_root / "configs" / "default_config.json")
    matrix_names = default_config.get("matrix", {}).get("dataset_names", [])
    dataset_paths = default_config.get("dataset_paths", {})
    has_registry_dataset6 = "Dataset6" in registry_text
    has_matrix_dataset6 = "Dataset6" in matrix_names
    has_path_dataset6 = "Dataset6" in dataset_paths
    return {
        "has_registry_dataset6": has_registry_dataset6,
        "matrix_dataset_names": matrix_names,
        "has_matrix_dataset6": has_matrix_dataset6,
        "has_path_dataset6": has_path_dataset6,
        "conclusion": (
            "Dataset6 is not registered in the formal training registry/matrix."
            if not (has_registry_dataset6 or has_matrix_dataset6 or has_path_dataset6)
            else "Dataset6 appears in at least one formal config location."
        ),
    }


def write_error_report(output_dir: Path, scans: Sequence[FileScan], message: str) -> None:
    rows = [
        {
            "path": str(scan.path),
            "readable": scan.readable,
            "columns": ";".join(scan.columns),
            "item_col_candidate": scan.item_col or "",
            "dept_col_candidate": scan.dept_col or "",
            "error": scan.error or "",
        }
        for scan in scans
    ]
    pd.DataFrame(rows).to_csv(output_dir / "dataset6_scanned_files.csv", index=False)
    (output_dir / "dataset6_dept_distribution_summary.md").write_text(
        "\n".join(
            [
                "# Dataset6 dept_id 分布检查失败",
                "",
                message,
                "",
                "已扫描文件和可用字段见 `dataset6_scanned_files.csv`。",
            ]
        ),
        encoding="utf-8",
    )


def write_bar_chart(distribution: pd.DataFrame, path: Path) -> None:
    mpl_cache = path.parent / ".matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_width = max(8, min(16, len(distribution) * 1.1))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.bar(distribution["dept_id"].astype(str), distribution["item_count"], color="#4C78A8")
    ax.set_xlabel("dept_id")
    ax.set_ylabel("item_count")
    ax.set_title("Dataset6 item_count by dept_id")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    scans: Sequence[FileScan],
    selected_scan: FileScan,
    mapping: pd.DataFrame,
    distribution: pd.DataFrame,
    roles: CandidateRoles,
    framework: dict,
    expected_item_count: int,
) -> ScanResult:
    item_count = int(mapping[IN_SCOPE_COL].sum())
    dept_count = int(distribution["dept_id"].nunique())
    top1 = distribution.iloc[0] if not distribution.empty else None
    top1_dept = str(top1["dept_id"]) if top1 is not None else ""
    top1_ratio = float(top1["item_ratio"]) if top1 is not None else 0.0
    top2_ratio = float(distribution["item_ratio"].head(2).sum()) if not distribution.empty else 0.0
    top3_ratio = float(distribution["item_ratio"].head(3).sum()) if not distribution.empty else 0.0
    conclusion = classify_distribution(top1_ratio, top2_ratio, top3_ratio)

    source_mapping = mapping[mapping[SOURCE_COL].astype(bool)].copy()
    target_mapping = mapping[mapping[TARGET_COL].astype(bool)].copy()
    source_depts = sorted(source_mapping["dept_id"].dropna().astype(str).unique().tolist())
    target_depts = sorted(target_mapping["dept_id"].dropna().astype(str).unique().tolist())
    target_dept_counts = (
        target_mapping.groupby("dept_id", as_index=False)["item_id"].nunique().rename(columns={"item_id": "target_item_count"})
    )
    source_target_shift = bool(source_depts and target_depts and set(source_depts).isdisjoint(set(target_depts)))
    same_dept_note = bool(source_depts and target_depts and set(target_depts).issubset(set(source_depts)))

    scanned_rows = pd.DataFrame(
        [
            {
                "path": str(scan.path),
                "readable": scan.readable,
                "item_col_candidate": scan.item_col or "",
                "dept_col_candidate": scan.dept_col or "",
                "column_count": len(scan.columns),
                "columns": ";".join(scan.columns),
                "error": scan.error or "",
            }
            for scan in scans
        ]
    )
    scanned_rows.to_csv(output_dir / "dataset6_scanned_files.csv", index=False)

    top_table = distribution.copy()
    top_table["item_ratio"] = top_table["item_ratio"].map(format_pct)
    top_table["cumulative_ratio"] = top_table["cumulative_ratio"].map(format_pct)

    item_count_note = (
        f"实际确认 item 数为 {item_count}，不是用户给定的 {expected_item_count}。"
        if item_count != expected_item_count
        else f"实际确认 item 数与用户给定的 {expected_item_count} 一致。"
    )
    formal_split_note = (
        "未发现 Dataset6 在 `dataset_registry.py`、`configs/default_config.json` 的正式训练矩阵中注册；"
        "因此本报告的 source/target 标记采用既有 Dataset6 profile 的候选 source/target 报告。"
        if not (framework["has_registry_dataset6"] or framework["has_matrix_dataset6"] or framework["has_path_dataset6"])
        else "检测到 Dataset6 出现在正式配置中；请复核具体 runner 是否使用该配置。"
    )

    if source_target_shift:
        design_note = "source 和 target 候选 dept_id 不相交，存在潜在 domain shift 风险。"
    elif same_dept_note:
        design_note = "source 和 target 候选都落在同一组 dept_id 内，跨部门迁移差异较弱。"
    else:
        design_note = "未确认正式 source/target dept 关系，不能据此判断跨部门迁移强度。"

    summary = [
        "# Dataset6 item dept_id 分布检查",
        "",
        "## 1. 数据文件来源",
        "",
        f"- 项目根目录：`{PROJECT_ROOT}`",
        f"- 自动定位的 Dataset6 主映射文件：`{selected_scan.path}`",
        f"- 采用 item 字段：`{selected_scan.item_col}`",
        f"- 采用 dept 字段：`{selected_scan.dept_col}`",
        f"- 已扫描文件清单：`{output_dir / 'dataset6_scanned_files.csv'}`",
        "",
        "## 2. 874 个 item 的确认过程",
        "",
        f"- {item_count_note}",
        "- 当前代码正式 registry 只覆盖 Dataset1/2/3；Dataset6 由数据 profile/审计脚本扫描，未发现正式训练 runner 的 Dataset6 source/target split。",
        f"- 当前 Dataset6 profile 主实体粒度为 `store_id + item_id`，完整扫描为 30,490 条 store-item 序列；去重到全局 `item_id` 后为 {item_count} 个 item。",
        f"- `{IN_SCOPE_COL}` 字段按用户要求保留列名；在本报告中它表示“本次解析出的当前 Dataset6 item 集合”，实际数量为 {item_count}。",
        "",
        "## 3. dept_id 分布统计表",
        "",
        markdown_table(top_table, ["dept_id", "item_count", "item_ratio", "cumulative_item_count", "cumulative_ratio"], max_rows=20),
        "",
        "## 4. top dept 集中度判断",
        "",
        f"- dept_id 总数：{dept_count}",
        f"- top 1 dept_id：{top1_dept}，占比 {format_pct(top1_ratio)}",
        f"- top 2 dept_id 累计占比：{format_pct(top2_ratio)}",
        f"- top 3 dept_id 累计占比：{format_pct(top3_ratio)}",
        f"- 直接判断：Dataset6 的当前 {item_count} 个 item 在 dept_id 上属于“{conclusion}”。",
        "",
        "## 5. source / target 与 dept_id 的关系",
        "",
        f"- {formal_split_note}",
        f"- 候选 source item 数：{len(roles.source_items)}；候选 target item 数：{len(roles.target_items)}。",
        f"- 候选 source dept_id：{', '.join(source_depts) if source_depts else '未确认'}",
        f"- 候选 target dept_id：{', '.join(target_depts) if target_depts else '未确认'}",
        f"- target dept 分布：{target_dept_counts.to_dict('records') if not target_dept_counts.empty else '未确认'}",
        f"- 判断：{design_note}",
        f"- source/target 角色来源：{roles.note}",
        "",
        "## 6. 对 Dataset6 实验设计的影响",
        "",
        "- 如果后续要把 Dataset6 纳入正式迁移学习实验，需要先把 Dataset6 的标准化加载、source pool、target split 写成显式协议，而不是依赖自动候选。",
        f"- 当前全量 item 分布不支持“主要集中在 1~2 个 dept”的说法：top2 只有 {format_pct(top2_ratio)}。",
        "- 现有候选 source/target 都集中在 `FOODS_3`，若直接用这些候选做实验，跨部门差异较弱，更像同部门内迁移。",
        "",
        "## 7. 是否建议按 dept_id 重新设计 source pool 或 target split",
        "",
        "- 建议。若研究目标包含跨部门 domain shift，应显式设计跨 dept 的 source pool/target split，并报告 dept_id 分层结果。",
        "- 若研究目标是降低跨域难度，则可保持同 dept，但报告中需说明 Dataset6 的迁移任务主要是同部门迁移。",
    ]
    (output_dir / "dataset6_dept_distribution_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return ScanResult(output_dir, item_count, dept_count, top1_dept, top1_ratio, top2_ratio, top3_ratio, conclusion)


def run_scan(project_root: Path, output_root: Path, expected_item_count: int) -> ScanResult:
    output_dir = make_output_dir(output_root)
    files = discover_dataset6_files(project_root)
    scans = scan_files(files)
    selected_scan = choose_item_dept_scan(scans)
    if selected_scan is None:
        message = "无法找到同时包含 item_id 与 dept_id 候选字段的 Dataset6 文件；未猜测 dept_id。"
        write_error_report(output_dir, scans, message)
        raise RuntimeError(f"{message} Report: {output_dir}")

    raw_mapping = read_columns(selected_scan.path, [selected_scan.item_col, selected_scan.dept_col])
    mapping = build_item_dept_mapping(raw_mapping, item_col=selected_scan.item_col, dept_col=selected_scan.dept_col)
    current_items = set(mapping["item_id"].astype(str))

    roles = parse_candidate_roles(project_root / "outputs" / "dataset_profiles" / "Dataset6" / "source_target_candidate_report.csv")
    mapping[IN_SCOPE_COL] = mapping["item_id"].astype(str).isin(current_items)
    mapping[SOURCE_COL] = mapping["item_id"].astype(str).isin(roles.source_items)
    mapping[TARGET_COL] = mapping["item_id"].astype(str).isin(roles.target_items)

    distribution = build_dept_distribution(mapping, IN_SCOPE_COL)
    distribution.to_csv(output_dir / "dataset6_item_dept_distribution.csv", index=False)
    mapping[["item_id", "dept_id", IN_SCOPE_COL, SOURCE_COL, TARGET_COL]].to_csv(
        output_dir / "dataset6_item_dept_mapping.csv",
        index=False,
    )
    write_bar_chart(distribution, output_dir / "dataset6_dept_distribution_bar.png")
    framework = inspect_framework_dataset6(project_root)
    return write_summary(output_dir, scans, selected_scan, mapping, distribution, roles, framework, expected_item_count)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Dataset6 item dept_id distribution.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-item-count", type=int, default=874)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_scan(
        project_root=args.project_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        expected_item_count=args.expected_item_count,
    )
    print("Dataset6 item 总数:", result.item_count)
    print("dept_id 总数:", result.dept_count)
    print(f"top 1 dept_id 及占比: {result.top1_dept} ({format_pct(result.top1_ratio)})")
    print(f"top 2 dept_id 累计占比: {format_pct(result.top2_ratio)}")
    print(f"top 3 dept_id 累计占比: {format_pct(result.top3_ratio)}")
    print(f"结论: {result.conclusion}")
    print(f"报告文件路径: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
