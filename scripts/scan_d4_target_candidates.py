#!/usr/bin/env python3
"""
D4 数据集 Target 候选组合扫描脚本

功能：
1. 扫描所有 (store_id, first_category_id) 组合
2. 对每个组合计算：
   - 候选商品数（排除该组已有的target）
   - 30日完整候选数
   - 跨越的second_category数
3. 筛选满足条件的组合：
   - 候选数≥10
   - 完整候选≥6
   - second_category跨度≤2

设计原则：
- 预先定义筛选标准，批量扫描，可复现
- 避免"先看数字再选target"的数据挑选风险
- 所有结果保存到审计文件，供论文审稿追溯
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


# ============================================================================
# 配置参数
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
D4_DATA_PATH = PROJECT_ROOT / "数据集" / "原始数据" / "Dataset 4叮咚数据集" / "data" / "train.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_audit"

# 筛选标准（预先定义，不可修改）
FILTER_CRITERIA = {
    "min_candidate_count": 10,          # 最少候选商品数
    "min_complete_30day_count": 6,      # 最少30日完整候选数
    "max_second_category_span": 2,      # 最大second_category跨度
}

# 30天完整性定义
COMPLETENESS_WINDOW_DAYS = 30
MIN_REQUIRED_DAYS = 30  # 30天内必须有30天的数据


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ProductInfo:
    """商品信息"""
    product_id: int
    store_id: int
    first_category_id: int
    second_category_id: int
    third_category_id: int | None
    date_range_start: str
    date_range_end: str
    total_days: int
    has_30day_completeness: bool


@dataclass
class TargetGroupCandidate:
    """Target组合候选信息"""
    store_id: int
    first_category_id: int
    
    # 基础统计
    total_products: int
    candidate_products: int
    complete_30day_candidates: int
    
    # 类目跨度
    second_category_ids: List[int]
    second_category_span: int
    third_category_ids: List[int]
    
    # 数据范围
    date_range_start: str
    date_range_end: str
    total_date_span_days: int
    
    # 筛选结果
    meets_criteria: bool
    failure_reasons: List[str]
    
    # 详细候选列表（仅在满足条件时保存）
    candidate_product_ids: List[int] = None
    complete_candidate_product_ids: List[int] = None


# ============================================================================
# 核心函数
# ============================================================================

def load_d4_data_chunked() -> pd.DataFrame:
    """
    分chunk加载D4数据，只读取必要的列
    
    Returns:
        包含商品、门店、类目、日期信息的DataFrame
    """
    print("\n[1/6] 加载D4数据集...")
    
    if not D4_DATA_PATH.exists():
        raise FileNotFoundError(f"D4数据集不存在: {D4_DATA_PATH}")
    
    if pq is None:
        raise ImportError("需要安装 pyarrow 来读取 parquet 文件")
    
    # 只读取必要的列
    required_cols = [
        'store_id',
        'product_id',
        'first_category_id',
        'second_category_id',
        'third_category_id',
        'dt',
    ]
    
    pf = pq.ParquetFile(D4_DATA_PATH)
    total_rows = pf.metadata.num_rows
    n_groups = pf.metadata.num_row_groups
    
    print(f"  - 文件路径: {D4_DATA_PATH}")
    print(f"  - 总行数: {total_rows:,}")
    print(f"  - Row groups: {n_groups}")
    print(f"  - 读取列: {required_cols}")
    
    # 分chunk读取
    chunks = []
    for i in range(n_groups):
        chunk = pf.read_row_group(i, columns=required_cols).to_pandas()
        chunks.append(chunk)
        if (i + 1) % 10 == 0 or i == n_groups - 1:
            print(f"  - 进度: {i+1}/{n_groups} row groups, {sum(len(c) for c in chunks):,} 行")
    
    df = pd.concat(chunks, ignore_index=True)
    
    # 转换日期
    df['dt'] = pd.to_datetime(df['dt'])
    
    print(f"  ✓ 加载完成: {len(df):,} 行")
    print(f"  - 日期范围: {df['dt'].min().date()} ~ {df['dt'].max().date()}")
    print(f"  - 唯一store数: {df['store_id'].nunique()}")
    print(f"  - 唯一product数: {df['product_id'].nunique()}")
    print(f"  - 唯一first_category数: {df['first_category_id'].nunique()}")
    
    return df


def analyze_product_completeness(df: pd.DataFrame) -> Dict[int, ProductInfo]:
    """
    分析每个商品的时间跨度和30天完整性
    
    Args:
        df: D4数据集DataFrame
        
    Returns:
        product_id -> ProductInfo 的字典
    """
    print("\n[2/6] 分析商品时间跨度和完整性...")
    
    product_info = {}
    
    # 按商品分组统计
    grouped = df.groupby([
        'product_id',
        'store_id',
        'first_category_id',
        'second_category_id',
        'third_category_id'
    ])
    
    for (product_id, store_id, first_cat, second_cat, third_cat), group in grouped:
        dates = group['dt'].sort_values()
        date_start = dates.min()
        date_end = dates.max()
        total_days = (date_end - date_start).days + 1
        
        # 检查30天完整性（检查最近30天）
        has_30day = False
        if total_days >= COMPLETENESS_WINDOW_DAYS:
            # 取最后30天的数据
            last_30_days = dates[dates >= (date_end - timedelta(days=COMPLETENESS_WINDOW_DAYS-1))]
            unique_days = last_30_days.nunique()
            has_30day = (unique_days >= MIN_REQUIRED_DAYS)
        
        product_info[product_id] = ProductInfo(
            product_id=int(product_id),
            store_id=int(store_id),
            first_category_id=int(first_cat),
            second_category_id=int(second_cat),
            third_category_id=int(third_cat) if pd.notna(third_cat) else None,
            date_range_start=date_start.strftime('%Y-%m-%d'),
            date_range_end=date_end.strftime('%Y-%m-%d'),
            total_days=int(total_days),
            has_30day_completeness=bool(has_30day),
        )
    
    complete_count = sum(1 for p in product_info.values() if p.has_30day_completeness)
    
    print(f"  ✓ 完成分析: {len(product_info)} 个商品")
    print(f"  - 满足30天完整性: {complete_count} ({complete_count/len(product_info)*100:.1f}%)")
    
    return product_info


def find_all_store_category_groups(
    df: pd.DataFrame,
    product_info: Dict[int, ProductInfo]
) -> Dict[Tuple[int, int], TargetGroupCandidate]:
    """
    找出所有 (store_id, first_category_id) 组合并计算统计信息
    
    Args:
        df: D4数据集DataFrame
        product_info: 商品信息字典
        
    Returns:
        (store_id, first_category_id) -> TargetGroupCandidate 的字典
    """
    print("\n[3/6] 扫描所有 (store_id, first_category_id) 组合...")
    
    # 按 (store_id, first_category_id) 分组
    groups = df.groupby(['store_id', 'first_category_id'])
    
    candidates = {}
    
    for (store_id, first_cat_id), group in groups:
        # 获取该组的所有商品
        products_in_group = group['product_id'].unique()
        
        # 统计second_category跨度
        second_cats = group['second_category_id'].dropna().unique()
        third_cats = group['third_category_id'].dropna().unique()
        
        # 获取日期范围
        dates = group['dt']
        date_start = dates.min()
        date_end = dates.max()
        date_span = (date_end - date_start).days + 1
        
        # 计算候选数（这里假设所有商品都可能成为候选，实际使用时可能需要排除某些target）
        candidate_products = []
        complete_candidates = []
        
        for product_id in products_in_group:
            if product_id in product_info:
                pinfo = product_info[product_id]
                # 候选商品：属于该store和first_category的所有商品
                candidate_products.append(int(product_id))
                # 完整候选：满足30天完整性的商品
                if pinfo.has_30day_completeness:
                    complete_candidates.append(int(product_id))
        
        # 判断是否满足筛选条件
        candidate_count = len(candidate_products)
        complete_count = len(complete_candidates)
        second_cat_span = len(second_cats)
        
        meets_criteria = True
        failure_reasons = []
        
        if candidate_count < FILTER_CRITERIA["min_candidate_count"]:
            meets_criteria = False
            failure_reasons.append(
                f"候选数不足 (实际:{candidate_count}, 要求:{FILTER_CRITERIA['min_candidate_count']})"
            )
        
        if complete_count < FILTER_CRITERIA["min_complete_30day_count"]:
            meets_criteria = False
            failure_reasons.append(
                f"30天完整候选不足 (实际:{complete_count}, 要求:{FILTER_CRITERIA['min_complete_30day_count']})"
            )
        
        if second_cat_span > FILTER_CRITERIA["max_second_category_span"]:
            meets_criteria = False
            failure_reasons.append(
                f"second_category跨度过大 (实际:{second_cat_span}, 要求:≤{FILTER_CRITERIA['max_second_category_span']})"
            )
        
        # 创建候选对象
        candidate = TargetGroupCandidate(
            store_id=int(store_id),
            first_category_id=int(first_cat_id),
            total_products=len(products_in_group),
            candidate_products=candidate_count,
            complete_30day_candidates=complete_count,
            second_category_ids=sorted([int(x) for x in second_cats]),
            second_category_span=int(second_cat_span),
            third_category_ids=sorted([int(x) for x in third_cats]),
            date_range_start=date_start.strftime('%Y-%m-%d'),
            date_range_end=date_end.strftime('%Y-%m-%d'),
            total_date_span_days=int(date_span),
            meets_criteria=meets_criteria,
            failure_reasons=failure_reasons,
            candidate_product_ids=candidate_products if meets_criteria else None,
            complete_candidate_product_ids=complete_candidates if meets_criteria else None,
        )
        
        candidates[(store_id, first_cat_id)] = candidate
    
    total_groups = len(candidates)
    qualified_groups = sum(1 for c in candidates.values() if c.meets_criteria)
    
    print(f"  ✓ 完成扫描: {total_groups} 个组合")
    print(f"  - 满足筛选条件: {qualified_groups} ({qualified_groups/total_groups*100:.1f}%)")
    
    return candidates


def generate_summary_report(
    candidates: Dict[Tuple[int, int], TargetGroupCandidate],
    output_dir: Path
) -> None:
    """
    生成汇总报告
    
    Args:
        candidates: 所有候选组合字典
        output_dir: 输出目录
    """
    print("\n[4/6] 生成汇总报告...")
    
    # 分离满足条件和不满足条件的组合
    qualified = [c for c in candidates.values() if c.meets_criteria]
    disqualified = [c for c in candidates.values() if not c.meets_criteria]
    
    # 生成Markdown报告
    report_lines = []
    report_lines.append("# D4 数据集 Target 候选组合扫描报告")
    report_lines.append("")
    report_lines.append(f"**扫描时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**数据源**: {D4_DATA_PATH}")
    report_lines.append("")
    
    report_lines.append("## 1. 筛选标准（预先定义）")
    report_lines.append("")
    report_lines.append("| 标准 | 阈值 | 说明 |")
    report_lines.append("|------|------|------|")
    report_lines.append(f"| 最少候选商品数 | ≥{FILTER_CRITERIA['min_candidate_count']} | 确保有足够的候选池，避免K=3刚好卡在边界 |")
    report_lines.append(f"| 最少30天完整候选数 | ≥{FILTER_CRITERIA['min_complete_30day_count']} | 确保候选商品有足够的历史数据 |")
    report_lines.append(f"| 最大second_category跨度 | ≤{FILTER_CRITERIA['max_second_category_span']} | 避免语义稀释，保持类目相关性 |")
    report_lines.append("")
    
    report_lines.append("## 2. 扫描结果汇总")
    report_lines.append("")
    report_lines.append(f"- **总组合数**: {len(candidates)}")
    report_lines.append(f"- **满足条件**: {len(qualified)} ({len(qualified)/len(candidates)*100:.1f}%)")
    report_lines.append(f"- **不满足条件**: {len(disqualified)} ({len(disqualified)/len(candidates)*100:.1f}%)")
    report_lines.append("")
    
    if qualified:
        report_lines.append("## 3. 满足条件的组合（推荐候选）")
        report_lines.append("")
        report_lines.append("| store_id | first_category_id | 候选数 | 30天完整候选数 | second_category跨度 | second_categories | 日期范围 |")
        report_lines.append("|----------|-------------------|--------|----------------|---------------------|-------------------|----------|")
        
        for c in sorted(qualified, key=lambda x: (x.complete_30day_candidates, x.candidate_products), reverse=True):
            report_lines.append(
                f"| {c.store_id} | {c.first_category_id} | {c.candidate_products} | "
                f"{c.complete_30day_candidates} | {c.second_category_span} | "
                f"{c.second_category_ids} | {c.date_range_start} ~ {c.date_range_end} |"
            )
        
        report_lines.append("")
    
    if disqualified:
        report_lines.append("## 4. 不满足条件的组合（前50个）")
        report_lines.append("")
        report_lines.append("| store_id | first_category_id | 候选数 | 30天完整候选数 | second_category跨度 | 失败原因 |")
        report_lines.append("|----------|-------------------|--------|----------------|---------------------|----------|")
        
        for c in sorted(disqualified, key=lambda x: (x.candidate_products), reverse=True)[:50]:
            reasons = "; ".join(c.failure_reasons)
            report_lines.append(
                f"| {c.store_id} | {c.first_category_id} | {c.candidate_products} | "
                f"{c.complete_30day_candidates} | {c.second_category_span} | {reasons} |"
            )
        
        report_lines.append("")
        if len(disqualified) > 50:
            report_lines.append(f"*（共{len(disqualified)}个不满足条件的组合，仅展示前50个）*")
            report_lines.append("")
    
    report_lines.append("## 5. 统计分布")
    report_lines.append("")
    
    # 候选数分布
    candidate_counts = [c.candidate_products for c in candidates.values()]
    complete_counts = [c.complete_30day_candidates for c in candidates.values()]
    span_counts = [c.second_category_span for c in candidates.values()]
    
    report_lines.append("### 5.1 候选商品数分布")
    report_lines.append("")
    report_lines.append(f"- **最小值**: {min(candidate_counts)}")
    report_lines.append(f"- **最大值**: {max(candidate_counts)}")
    report_lines.append(f"- **平均值**: {np.mean(candidate_counts):.1f}")
    report_lines.append(f"- **中位数**: {np.median(candidate_counts):.1f}")
    report_lines.append("")
    
    report_lines.append("### 5.2 30天完整候选数分布")
    report_lines.append("")
    report_lines.append(f"- **最小值**: {min(complete_counts)}")
    report_lines.append(f"- **最大值**: {max(complete_counts)}")
    report_lines.append(f"- **平均值**: {np.mean(complete_counts):.1f}")
    report_lines.append(f"- **中位数**: {np.median(complete_counts):.1f}")
    report_lines.append("")
    
    report_lines.append("### 5.3 second_category 跨度分布")
    report_lines.append("")
    report_lines.append(f"- **最小值**: {min(span_counts)}")
    report_lines.append(f"- **最大值**: {max(span_counts)}")
    report_lines.append(f"- **平均值**: {np.mean(span_counts):.1f}")
    report_lines.append(f"- **中位数**: {np.median(span_counts):.1f}")
    report_lines.append("")
    
    report_lines.append("## 6. 选择建议")
    report_lines.append("")
    report_lines.append("### 6.1 如何选择Target组合")
    report_lines.append("")
    report_lines.append("1. **保留原有的store166组合**（如果满足条件）：")
    report_lines.append("   - 已经验证过的数据，无需重复工作")
    report_lines.append("   - 作为多组target之一，扩展而非替换")
    report_lines.append("")
    report_lines.append("2. **从满足条件的组合中选择**：")
    report_lines.append("   - 方式1：随机抽取N个（N=3-5）")
    report_lines.append("   - 方式2：选择候选数接近中位数的组合（代表性强）")
    report_lines.append("   - 方式3：选择不同second_category跨度的组合（覆盖多种场景）")
    report_lines.append("")
    report_lines.append("3. **避免的做法**：")
    report_lines.append("   - ❌ 不要手动挑选\"看起来好\"的组合")
    report_lines.append("   - ❌ 不要只挑候选数最多的组合")
    report_lines.append("   - ❌ 不要在看到具体数字后再定标准")
    report_lines.append("")
    
    report_lines.append("### 6.2 审计追溯")
    report_lines.append("")
    report_lines.append("本次扫描的所有结果已保存到以下文件，供论文审稿追溯：")
    report_lines.append("")
    report_lines.append("- `d4_target_candidates_summary.md`: 本报告")
    report_lines.append("- `d4_target_candidates_qualified.json`: 满足条件的组合详细信息")
    report_lines.append("- `d4_target_candidates_all.json`: 所有组合的完整信息")
    report_lines.append("- `d4_target_candidates_qualified.csv`: 满足条件的组合（CSV格式，便于Excel查看）")
    report_lines.append("")
    
    # 写入Markdown报告
    report_path = output_dir / "d4_target_candidates_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  ✓ Markdown报告: {report_path}")
    
    return report_path


def save_results(
    candidates: Dict[Tuple[int, int], TargetGroupCandidate],
    output_dir: Path
) -> None:
    """
    保存扫描结果到文件
    
    Args:
        candidates: 所有候选组合字典
        output_dir: 输出目录
    """
    print("\n[5/6] 保存扫描结果...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 分离满足条件和不满足条件的组合
    qualified = [c for c in candidates.values() if c.meets_criteria]
    all_candidates = list(candidates.values())
    
    # 保存满足条件的组合（JSON格式，包含详细候选商品列表）
    qualified_data = {
        "scan_timestamp": datetime.now().isoformat(),
        "filter_criteria": FILTER_CRITERIA,
        "total_groups": len(candidates),
        "qualified_count": len(qualified),
        "qualified_groups": [asdict(c) for c in qualified]
    }
    
    qualified_path = output_dir / "d4_target_candidates_qualified.json"
    with qualified_path.open("w", encoding="utf-8") as f:
        json.dump(qualified_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 满足条件的组合JSON: {qualified_path}")
    
    # 保存所有组合（JSON格式，但不包含详细商品列表以节省空间）
    all_data = {
        "scan_timestamp": datetime.now().isoformat(),
        "filter_criteria": FILTER_CRITERIA,
        "total_groups": len(candidates),
        "qualified_count": len(qualified),
        "all_groups": [
            {
                "store_id": c.store_id,
                "first_category_id": c.first_category_id,
                "total_products": c.total_products,
                "candidate_products": c.candidate_products,
                "complete_30day_candidates": c.complete_30day_candidates,
                "second_category_span": c.second_category_span,
                "second_category_ids": c.second_category_ids,
                "date_range_start": c.date_range_start,
                "date_range_end": c.date_range_end,
                "meets_criteria": c.meets_criteria,
                "failure_reasons": c.failure_reasons,
            }
            for c in all_candidates
        ]
    }
    
    all_path = output_dir / "d4_target_candidates_all.json"
    with all_path.open("w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 所有组合JSON: {all_path}")
    
    # 保存满足条件的组合（CSV格式，便于Excel查看）
    if qualified:
        qualified_df = pd.DataFrame([
            {
                "store_id": c.store_id,
                "first_category_id": c.first_category_id,
                "total_products": c.total_products,
                "candidate_products": c.candidate_products,
                "complete_30day_candidates": c.complete_30day_candidates,
                "second_category_span": c.second_category_span,
                "second_category_ids": str(c.second_category_ids),
                "third_category_count": len(c.third_category_ids),
                "date_range_start": c.date_range_start,
                "date_range_end": c.date_range_end,
                "total_date_span_days": c.total_date_span_days,
            }
            for c in qualified
        ])
        
        csv_path = output_dir / "d4_target_candidates_qualified.csv"
        qualified_df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # UTF-8 with BOM for Excel
        print(f"  ✓ 满足条件的组合CSV: {csv_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("D4 数据集 Target 候选组合扫描")
    print("=" * 80)
    print("")
    print("筛选标准:")
    for key, value in FILTER_CRITERIA.items():
        print(f"  - {key}: {value}")
    print("")
    
    # 1. 加载数据
    df = load_d4_data_chunked()
    
    # 2. 分析商品完整性
    product_info = analyze_product_completeness(df)
    
    # 3. 扫描所有组合
    candidates = find_all_store_category_groups(df, product_info)
    
    # 4. 生成报告
    report_path = generate_summary_report(candidates, OUTPUT_DIR)
    
    # 5. 保存结果
    save_results(candidates, OUTPUT_DIR)
    
    # 6. 汇总输出
    print("\n[6/6] 扫描完成！")
    print("")
    print("=" * 80)
    print("汇总统计")
    print("=" * 80)
    
    qualified = [c for c in candidates.values() if c.meets_criteria]
    
    print(f"总组合数: {len(candidates)}")
    print(f"满足条件: {len(qualified)} ({len(qualified)/len(candidates)*100:.1f}%)")
    print("")
    
    if qualified:
        print("满足条件的组合（前10个）:")
        print("-" * 80)
        print(f"{'store_id':<12} {'first_cat':<12} {'候选数':<10} {'完整候选':<12} {'跨度':<8}")
        print("-" * 80)
        
        for c in sorted(qualified, key=lambda x: (x.complete_30day_candidates, x.candidate_products), reverse=True)[:10]:
            print(f"{c.store_id:<12} {c.first_category_id:<12} {c.candidate_products:<10} "
                  f"{c.complete_30day_candidates:<12} {c.second_category_span:<8}")
        
        if len(qualified) > 10:
            print(f"... 还有 {len(qualified) - 10} 个满足条件的组合")
    else:
        print("⚠️  没有满足条件的组合！需要调整筛选标准。")
    
    print("")
    print("=" * 80)
    print("输出文件:")
    print("=" * 80)
    print(f"  - 报告: {OUTPUT_DIR / 'd4_target_candidates_summary.md'}")
    print(f"  - 满足条件的组合: {OUTPUT_DIR / 'd4_target_candidates_qualified.json'}")
    print(f"  - 所有组合: {OUTPUT_DIR / 'd4_target_candidates_all.json'}")
    print(f"  - CSV格式: {OUTPUT_DIR / 'd4_target_candidates_qualified.csv'}")
    print("")


if __name__ == "__main__":
    main()
