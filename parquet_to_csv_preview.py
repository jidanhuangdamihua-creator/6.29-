"""
parquet_to_csv_preview.py
=========================
将固化数据目录下的 parquet 文件转为 CSV 预览文件：
  - target parquet  → 全量保留
  - source parquet  → 按 entity_id 分层抽样，合计约 N 条（默认 500）

用法示例
-------
# 转换单个文件（自动判断 target / source）
python parquet_to_csv_preview.py --file 数据集/固化数据/D1_target.parquet

# 转换整个目录（递归扫描所有 .parquet）
python parquet_to_csv_preview.py --dir 数据集/固化数据/

# 指定输出目录 & 抽样数量
python parquet_to_csv_preview.py --dir 数据集/固化数据/ --out outputs/csv_preview/ --n 500

文件命名规则
-----------
脚本通过文件名判断 target / source：
  - 文件名含 "target"（不区分大小写）→ 全量保留
  - 文件名含 "source"（不区分大小写）→ 分层抽样
  - 两者都不含 → 打印警告，默认按 source 处理（可用 --force-target / --force-source 覆盖）
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd


# ──────────────────────────────────────────────
# 核心：分层抽样
# ──────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, n_total: int, strata_col: str) -> pd.DataFrame:
    """
    按 strata_col 分层抽样，合计约 n_total 条。

    - 若原始行数 <= n_total，直接返回全量
    - 若 entity 数 > n_total：随机抽 n_total 个 entity，每个取 1 条（严格 n_total 行）
    - 否则：等比分层 + 每组至少 1 条（实际行数可能略多于 n_total）
    """
    if len(df) <= n_total:
        print(f"    [INFO] 原始行数 {len(df)} <= 目标 {n_total}，全量保留。")
        return df

    n_entities = df[strata_col].nunique()
    if n_entities > n_total:
        print(
            f"    [INFO] entity 数 {n_entities} > 目标 {n_total}，"
            f"随机抽 {n_total} 个 entity，各取 1 条。"
        )
        chosen_entities = df[strata_col].drop_duplicates().sample(n=n_total, random_state=42)
        sampled_parts = []
        for entity_id in chosen_entities:
            group = df[df[strata_col] == entity_id]
            sampled_parts.append(group.sample(n=1, random_state=42))
        return pd.concat(sampled_parts, ignore_index=True)

    groups = df.groupby(strata_col, sort=False)
    total_rows = len(df)
    sampled_parts = []

    for entity_id, group in groups:
        proportion = len(group) / total_rows
        k = max(1, math.floor(proportion * n_total))
        sampled = group.sample(n=min(k, len(group)), random_state=42)
        sampled_parts.append(sampled)

    return pd.concat(sampled_parts, ignore_index=True)


# ──────────────────────────────────────────────
# 判断文件类型
# ──────────────────────────────────────────────

def detect_file_role(path: Path, force_target: bool, force_source: bool) -> str:
    """返回 'target' 或 'source'"""
    if force_target:
        return "target"
    if force_source:
        return "source"

    name_lower = path.name.lower()
    if "target" in name_lower:
        return "target"
    if "source" in name_lower:
        return "source"

    print(f"  [WARN] 文件名 '{path.name}' 未含 target/source 关键词，默认按 source 处理。")
    print(f"         如需全量保留，请加 --force-target 参数。")
    return "source"


# ──────────────────────────────────────────────
# 单文件处理
# ──────────────────────────────────────────────

def process_file(
    parquet_path: Path,
    out_dir: Path,
    n_total: int,
    force_target: bool,
    force_source: bool,
) -> None:
    print(f"\n{'─'*55}")
    print(f"处理：{parquet_path}")

    # 读取
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"  [ERROR] 读取失败：{e}")
        return

    print(f"  原始行数：{len(df):,}   列数：{df.shape[1]}")

    # 判断角色
    role = detect_file_role(parquet_path, force_target, force_source)
    print(f"  文件角色：{role}")

    # 处理
    if role == "target":
        result = df
        print(f"  全量保留：{len(result):,} 行")
    else:
        if "entity_id" not in df.columns:
            available = list(df.columns[:10])
            print(f"  [WARN] 未找到 entity_id 列，可用列（前10）：{available}")
            print(f"         回退为随机抽样 {n_total} 条。")
            result = df.sample(n=min(n_total, len(df)), random_state=42)
        else:
            result = stratified_sample(df, n_total, strata_col="entity_id")
            n_entities = df["entity_id"].nunique()
            print(f"  分层抽样：entity_id 共 {n_entities} 个实体 → 抽后 {len(result):,} 行")

    # 输出路径
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (parquet_path.stem + "_preview.csv")
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  已保存：{out_path}")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parquet → CSV 预览工具（target 全量 / source 分层抽样）"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="单个 parquet 文件路径")
    group.add_argument("--dir", type=str, help="目录路径（递归扫描所有 .parquet）")

    parser.add_argument(
        "--out", type=str, default="outputs/csv_preview/",
        help="输出目录（默认：outputs/csv_preview/）"
    )
    parser.add_argument(
        "--n", type=int, default=500,
        help="source 分层抽样目标行数（默认：500）"
    )
    parser.add_argument(
        "--force-target", action="store_true",
        help="强制将所有文件视为 target（全量保留）"
    )
    parser.add_argument(
        "--force-source", action="store_true",
        help="强制将所有文件视为 source（分层抽样）"
    )

    args = parser.parse_args()
    out_dir = Path(args.out)

    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(Path(args.dir).rglob("*.parquet"))
        print(f"扫描到 {len(files)} 个 parquet 文件：{args.dir}")
        if not files:
            print("[ERROR] 未找到任何 .parquet 文件，请检查路径。")
            sys.exit(1)

    for f in files:
        process_file(f, out_dir, args.n, args.force_target, args.force_source)

    print(f"\n{'─'*55}")
    print(f"全部完成，CSV 已输出至：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
