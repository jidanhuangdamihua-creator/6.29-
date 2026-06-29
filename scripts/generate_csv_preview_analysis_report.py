#!/usr/bin/env python3
"""Scan 数据集/csv_preview/ CSVs and write outputs/csv_preview/analysis_report.md."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "数据集" / "csv_preview"
OUT_PATH = ROOT / "outputs" / "csv_preview" / "analysis_report.md"
PENDING = "【待补充】"


@dataclass(frozen=True)
class ColumnInfo:
    meaning: str
    knn_role: str
    tl_role: str

    @property
    def combined_role(self) -> str:
        return f"KNN: {self.knn_role}；TL: {self.tl_role}"


COLUMN_DICT: dict[str, ColumnInfo] = {
    # 核心列（原始字典）
    "date": ColumnInfo("日期", "用于对齐 source-target 时间窗口", "时序切分依据"),
    "ds": ColumnInfo("日期", "用于对齐 source-target 时间窗口", "时序切分依据"),
    "entity_id": ColumnInfo(
        "实体唯一标识（store×SKU 组合）", "计算相似度的分组键", "区分域的标签来源"
    ),
    "store_id": ColumnInfo("门店编号", "候选 source 筛选条件", "域标签辅助信息"),
    "target_entity_id": ColumnInfo("目标实体 ID", "标记 KNN 查询对象", "DANN 目标域标识"),
    "target_store_id": ColumnInfo("目标门店 ID", "标记 KNN 查询对象", "DANN 目标域标识"),
    "sales": ColumnInfo("销售量（预测标签）", "计算历史销量相似度", "监督信号"),
    "y": ColumnInfo("销售量（预测标签）", "计算历史销量相似度", "监督信号"),
    "oil_price": ColumnInfo("油价（外部经济特征）", "纳入 KNN 距离特征", "RFE 候选特征"),
    "transactions": ColumnInfo("门店当日交易笔数", "纳入 KNN 距离特征", "RFE 候选特征"),
    "onpromotion": ColumnInfo("是否促销（0/1）", "促销模式相似度", "RFE 候选特征，影响域偏移"),
    "sell_price": ColumnInfo("商品售价（M5 数据集）", "价格相似度", "RFE 候选特征"),
    "item_nbr": ColumnInfo("商品编号", "商品维度筛选", "辅助标识，不入模型"),
    "sku_id": ColumnInfo("商品编号", "商品维度筛选", "辅助标识，不入模型"),
    # 通用时间拆分列（D1–D6）
    "year": ColumnInfo("年份（从 date 拆分）", "时间对齐辅助", "季节性周期特征，RFE 候选"),
    "month": ColumnInfo("月份（1–12）", "季节性相似度", "周期性特征，RFE 候选"),
    "week": ColumnInfo("ISO 周数（1–53）", "周度促销节奏相似度", "周期性特征，RFE 候选"),
    "day": ColumnInfo("日（1–31）", "月内节律相似度", "细粒度时序特征，RFE 候选"),
    "item_id": ColumnInfo(
        "商品统一编号（各数据集预处理后标准化）",
        "商品维度筛选辅助键",
        "辅助标识，不直接入模型",
    ),
    # D2（OTC 零售）
    "brand_id": ColumnInfo("品牌编号", "同品牌商品相似度筛选", "品牌域偏移辅助标识"),
    "promo": ColumnInfo("是否促销（0/1）", "促销模式相似度", "RFE 候选特征，影响域偏移"),
    # D3（Rossmann）
    "Customers": ColumnInfo(
        "当日到店顾客数", "客流相似度（重要特征）", "RFE 候选，与 sales 高度相关"
    ),
    "Open": ColumnInfo(
        "门店当日是否营业（1=营业，0=关闭）",
        "过滤关店日",
        "数据清洗标记，关店日 sales=0 需屏蔽",
    ),
    "Promo": ColumnInfo(
        "当日是否有促销活动（0/1）", "促销节奏相似度", "RFE 候选特征，影响域偏移"
    ),
    "SchoolHoliday": ColumnInfo(
        "当日是否学校假期（0/1）", "假期模式相似度", "外部日历特征，RFE 候选"
    ),
    "region": ColumnInfo("门店所在地区", "地区同质性筛选", "地理域偏移辅助标识"),
    # D4（中国零售）
    "city_id": ColumnInfo("城市编号", "同城门店优先筛选", "地理域标签"),
    "management_group_id": ColumnInfo(
        "门店管理分组编号", "同管理体系门店相似度", "运营域偏移辅助标识"
    ),
    "first_category_id": ColumnInfo(
        "商品一级品类编号", "大类相似度筛选", "类目域偏移辅助标识"
    ),
    "second_category_id": ColumnInfo(
        "商品二级品类编号", "中类相似度筛选", "RFE 候选辅助特征"
    ),
    "third_category_id": ColumnInfo(
        "商品三级品类编号（最细粒度）", "细类相似度筛选", "RFE 候选辅助特征"
    ),
    "product_id": ColumnInfo(
        "商品 SKU 编号（D4 原始字段）", "商品维度筛选键", "辅助标识，不入模型"
    ),
    "stock_hour6_22_cnt": ColumnInfo(
        "06:00–22:00 营业时段有货小时数",
        "供货稳定性相似度",
        "RFE 候选，反映库存充足程度",
    ),
    "activity_flag": ColumnInfo(
        "营销活动标记（0/1）", "活动节奏相似度", "RFE 候选，影响域偏移"
    ),
    "discount": ColumnInfo("折扣率（0–1）", "价格策略相似度", "RFE 候选特征"),
    "holiday_flag": ColumnInfo(
        "节假日标记（0/1）", "节假日模式相似度", "外部日历特征，RFE 候选"
    ),
    "precpt": ColumnInfo("日降水量（mm）", "天气相似度", "外部气象特征，RFE 候选"),
    "avg_temperature": ColumnInfo(
        "日平均气温（℃）", "气候相似度（影响客流）", "外部气象特征，RFE 候选"
    ),
    "avg_humidity": ColumnInfo(
        "日平均相对湿度（%）", "气候相似度辅助", "外部气象特征，RFE 候选"
    ),
    "avg_wind_level": ColumnInfo(
        "日平均风力等级", "气候相似度辅助", "外部气象特征，RFE 候选"
    ),
    "hours_sale_sum_leakage_risk": ColumnInfo(
        "当日分时销售额累计（⚠️ 含未来信息，有泄漏风险）",
        "禁止用于相似度计算",
        "禁止入模型，仅供离线分析参考",
    ),
    "hours_sale_max_leakage_risk": ColumnInfo(
        "当日分时销售额峰值（⚠️ 含未来信息，有泄漏风险）",
        "禁止用于相似度计算",
        "禁止入模型，仅供离线分析参考",
    ),
    "hours_sale_nonzero_hours_leakage_risk": ColumnInfo(
        "当日非零销售小时数（⚠️ 含未来信息，有泄漏风险）",
        "禁止用于相似度计算",
        "禁止入模型，仅供离线分析参考",
    ),
    "hours_stock_sum_leakage_risk": ColumnInfo(
        "当日分时库存累计（⚠️ 含未来信息，有泄漏风险）",
        "禁止用于相似度计算",
        "禁止入模型，仅供离线分析参考",
    ),
    "hours_stock_max_leakage_risk": ColumnInfo(
        "当日分时库存峰值（⚠️ 含未来信息，有泄漏风险）",
        "禁止用于相似度计算",
        "禁止入模型，仅供离线分析参考",
    ),
    "hours_stock_nonzero_hours_leakage_risk": ColumnInfo(
        "当日非零库存小时数（⚠️ 含未来信息，有泄漏风险）",
        "禁止用于相似度计算",
        "禁止入模型，仅供离线分析参考",
    ),
    # D5（Corporación Favorita）
    "store_nbr": ColumnInfo(
        "门店编号（D5 原始字段，对应 store_id）",
        "候选 source 筛选条件",
        "域标签辅助信息",
    ),
    "family": ColumnInfo(
        "商品大类（如 BEVERAGES、DAIRY、PRODUCE 等）",
        "品类相似度筛选",
        "类目域偏移辅助标识",
    ),
    "class": ColumnInfo(
        "商品子类（比 family 更细一级）", "细粒度品类相似度", "RFE 候选辅助特征"
    ),
    "perishable": ColumnInfo(
        "是否易腐商品（1=易腐，0=非易腐）",
        "商品属性相似度",
        "RFE 候选，影响销售波动模式",
    ),
    "city": ColumnInfo("门店所在城市名称", "地理同质性筛选", "地理域偏移辅助标识"),
    "state": ColumnInfo("门店所在州/省名称", "地理同质性筛选", "地理域偏移辅助标识"),
    "type": ColumnInfo(
        "门店类型（A/B/C/D/E，按规模与业态分类）",
        "同类型门店优先筛选",
        "业态域偏移辅助标识",
    ),
    "cluster": ColumnInfo(
        "门店聚类分组编号（Kaggle 官方按相似性聚类）",
        "同簇门店高相似度，直接用于筛选",
        "域偏移度参考",
    ),
    "is_holiday": ColumnInfo(
        "是否节假日（0/1，含地方节假日）", "假期模式相似度", "外部日历特征，RFE 候选"
    ),
    # D6（M5 Walmart）
    "id": ColumnInfo(
        "M5 商品-门店唯一标识（格式：item_id_store_id_evaluation）",
        "仅作索引，不参与相似度计算",
        "辅助标识",
    ),
    "dept_id": ColumnInfo(
        "部门编号（如 HOBBIES_1，比 cat_id 更细）",
        "部门相似度筛选",
        "RFE 候选辅助特征",
    ),
    "cat_id": ColumnInfo(
        "商品大类（HOBBIES / HOUSEHOLD / FOODS）",
        "大类相似度筛选",
        "类目域偏移辅助标识",
    ),
    "state_id": ColumnInfo("州编号（CA / TX / WI）", "地理同质性筛选", "地理域偏移辅助标识"),
    "wm_yr_wk": ColumnInfo(
        "Walmart 内部年-周编号（格式 YYYYWW）",
        "周度对齐辅助",
        "可替代 week 使用，RFE 候选",
    ),
    "event_name_1": ColumnInfo(
        "主要节日/活动名称（如 Christmas、SuperBowl，无活动为 NaN）",
        "活动同步性相似度",
        "与 is_event_1 配合使用，分类特征需编码",
    ),
    "event_type_1": ColumnInfo(
        "主要活动类型（National / Sporting / Cultural / Religious）",
        "活动类型相似度",
        "RFE 候选，需编码，影响促销域偏移",
    ),
    "event_name_2": ColumnInfo(
        "次要节日/活动名称（同日第二活动，极少见）",
        "意义较小",
        "RFE 候选，大概率被 RFE 剔除",
    ),
    "event_type_2": ColumnInfo("次要活动类型", "意义较小", "RFE 候选，大概率被 RFE 剔除"),
    "weekday": ColumnInfo(
        "星期几名称（Monday–Sunday）",
        "周内节律相似度",
        "周期性特征，与 day_of_week 等价，RFE 候选",
    ),
    "is_event_1": ColumnInfo(
        "是否有主要活动（0/1，由 event_name_1 非空衍生）",
        "活动日模式相似度",
        "RFE 重点候选，比 event_name_1 更直接入模型",
    ),
    "is_event_2": ColumnInfo(
        "是否有次要活动（0/1，由 event_name_2 非空衍生）",
        "意义较小",
        "RFE 候选，大概率被剔除",
    ),
}

PATTERN_RULES: list[tuple[re.Pattern[str], ColumnInfo]] = [
    (
        re.compile(r"^snap_.+$"),
        ColumnInfo("SNAP 补贴项目标记（M5）", "外部干预特征", "RFE 候选特征"),
    ),
    (
        re.compile(r"^snap$"),
        ColumnInfo("SNAP 补贴项目标记（M5）", "外部干预特征", "RFE 候选特征"),
    ),
    (
        re.compile(r"^lag_\d+$"),
        ColumnInfo(
            "历史滞后特征",
            "序列模式相似度核心特征",
            "DANN 输入特征，RFE 重点筛选对象",
        ),
    ),
    (
        re.compile(r"^rolling_mean_.+$"),
        ColumnInfo("滚动均值特征", "趋势相似度", "RFE 候选特征"),
    ),
    (
        re.compile(r"^rolling_std_.+$"),
        ColumnInfo("滚动标准差特征", "波动相似度", "RFE 候选特征"),
    ),
]


def lookup_column(col: str) -> ColumnInfo | None:
    if col in COLUMN_DICT:
        return COLUMN_DICT[col]
    for pattern, info in PATTERN_RULES:
        if pattern.match(col):
            return info
    return None


def read_columns(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        return next(csv.reader(fh))


def count_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        return sum(1 for _ in fh) - 1


def file_role(name: str) -> str:
    return "target（全量）" if "target" in name.lower() else "source（抽样）"


def missing_rates(csv_path: Path, columns: list[str]) -> dict[str, float]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=columns, low_memory=False)
    return {col: float(df[col].isna().mean()) for col in columns}


def column_table_md(columns: list[str]) -> tuple[str, set[str]]:
    unknown: set[str] = set()
    lines = ["| 列名 | 含义 | KNN+TL 作用 |", "|------|------|-------------|"]
    for col in columns:
        info = lookup_column(col)
        if info is None:
            unknown.add(col)
            lines.append(f"| `{col}` | {PENDING} | {PENDING} |")
        else:
            lines.append(f"| `{col}` | {info.meaning} | {info.combined_role} |")
    return "\n".join(lines), unknown


def render_file_section(csv_path: Path) -> tuple[str, set[str]]:
    columns = read_columns(csv_path)
    rows = count_rows(csv_path)
    missing = missing_rates(csv_path, columns)
    high_missing = [c for c, rate in missing.items() if rate > 0]
    table_md, unknown = column_table_md(columns)

    lines = [
        f"## {csv_path.name}",
        "",
        f"- **路径**: `{csv_path.relative_to(ROOT)}`",
        f"- **角色**: {file_role(csv_path.name)}",
        f"- **行数**: {rows:,}",
        f"- **列数**: {len(columns)}",
    ]
    if high_missing:
        parts = [f"`{c}`={missing[c]:.1%}" for c in high_missing[:8]]
        suffix = " …" if len(high_missing) > 8 else ""
        lines.append(f"- **含缺失值的列**: {', '.join(parts)}{suffix}")
    else:
        lines.append("- **含缺失值的列**: 无")

    lines.extend(["", "### 列说明", "", table_md, ""])
    return "\n".join(lines), unknown


def main() -> None:
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"未找到 CSV: {CSV_DIR}")

    all_unknown: set[str] = set()
    sections: list[str] = [
        "# CSV Preview 分析报告",
        "",
        f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"> 扫描目录：`{CSV_DIR.relative_to(ROOT)}`",
        f"> 文件数：{len(csv_files)}",
        "",
    ]

    for csv_path in csv_files:
        section, unknown = render_file_section(csv_path)
        sections.append(section)
        all_unknown.update(unknown)

    if all_unknown:
        sections.extend(
            [
                "## 未识别列汇总",
                "",
                f"共 {len(all_unknown)} 个唯一列名（字典中无定义，均标注 `{PENDING}`）：",
                "",
            ]
        )
        for col in sorted(all_unknown):
            sections.append(f"- `{col}`")
    else:
        sections.extend(
            [
                "## 未识别列汇总",
                "",
                "所有列均已完成说明，无待补充项。",
            ]
        )
    sections.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(
        f"已写入: {OUT_PATH} "
        f"({len(csv_files)} 个文件, {len(all_unknown)} 个未识别列)"
    )


if __name__ == "__main__":
    main()
