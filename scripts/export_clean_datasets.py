"""
数据清洗导出脚本
================
读取 Dataset1 / Dataset2 / Dataset3 原始 CSV，
调用现有预处理逻辑完成清洗，并将结果保存到 outputs/clean_datasets/。

用法：
    python scripts/export_clean_datasets.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保项目根目录在 sys.path，使得 data_preprocessing 等模块可被导入
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from data_preprocessing import extract_datetime_features, load_dataset
from dataset_registry import get_default_dataset_path

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("export_clean_datasets")

# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------
OUTPUT_DIR = _REPO_ROOT / "outputs" / "clean_datasets"

# 文件名映射
_OUTPUT_FILENAME: dict[str, str] = {
    "Dataset1": "dataset1_clean.csv",
    "Dataset2": "dataset2_clean.csv",
    "Dataset3": "dataset3_clean.csv",
}


def export_clean_dataset(dataset_name: str) -> None:
    """
    读取、清洗并导出单个数据集。

    处理步骤（与现有实验保持完全一致）：
    1. load_dataset   —— 读取原始 CSV，统一字段命名，转换 date 为 datetime，dropna
    2. extract_datetime_features —— 生成 year / month / week / day 特征列

    Args:
        dataset_name: 规范数据集名称，如 "Dataset1"、"Dataset2"、"Dataset3"
    """
    data_path = get_default_dataset_path(dataset_name)
    logger.info("=" * 60)
    logger.info("[%s] 开始处理，原始文件：%s", dataset_name, data_path)

    # 步骤 1：加载并标准化
    raw_df = pd.read_csv(data_path, low_memory=False)
    raw_rows = len(raw_df)
    logger.info("[%s] 原始行数：%d", dataset_name, raw_rows)

    df = load_dataset(dataset_name, data_path)

    # 步骤 2：提取时间特征
    df = extract_datetime_features(df)

    # 验证必要列
    required_cols = {"sales", "year", "month", "week", "day"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(
            f"[{dataset_name}] 清洗后缺少必要列：{sorted(missing)}。"
            "请检查 data_preprocessing.py 是否有变更。"
        )

    # 确保 date 为 datetime 类型
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    clean_rows = len(df)
    logger.info("[%s] 清洗后行数：%d（丢弃 %d 行）", dataset_name, clean_rows, raw_rows - clean_rows)

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / _OUTPUT_FILENAME[dataset_name]
    df.to_csv(out_path, index=False)

    logger.info("[%s] 已保存 → %s", dataset_name, out_path)
    logger.info(
        "[%s] 列列表：%s",
        dataset_name,
        list(df.columns),
    )


def main() -> None:
    for ds in ("Dataset1", "Dataset2", "Dataset3"):
        export_clean_dataset(ds)

    logger.info("=" * 60)
    logger.info("全部数据集清洗完成，输出目录：%s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
