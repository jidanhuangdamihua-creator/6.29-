#!/usr/bin/env python3
"""
分析D4扫描结果中first_category_id的分布情况
"""

import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALIFIED_JSON = PROJECT_ROOT / "outputs" / "dataset_audit" / "d4_target_candidates_qualified.json"

def main():
    print("=" * 80)
    print("D4 满足条件组合的 first_category_id 分布分析")
    print("=" * 80)
    print("")
    
    # 读取扫描结果
    with open(QUALIFIED_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qualified_groups = data['qualified_groups']
    total_qualified = len(qualified_groups)
    
    print(f"满足条件的组合总数: {total_qualified}")
    print("")
    
    # 按 first_category_id 分组
    category_groups = defaultdict(list)
    for group in qualified_groups:
        first_cat = group['first_category_id']
        category_groups[first_cat].append(group)
    
    # 统计每个 first_category_id 的数量
    print(f"不同的 first_category_id 数量: {len(category_groups)}")
    print("")
    
    print("=" * 80)
    print("按 first_category_id 分组统计")
    print("=" * 80)
    print("")
    
    # 按组数量排序
    sorted_categories = sorted(category_groups.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"{'first_category_id':<20} {'组合数':<10} {'占比':<10} {'候选数范围':<20} {'中位数组合信息'}")
    print("-" * 120)
    
    category_stats = []
    
    for first_cat, groups in sorted_categories:
        count = len(groups)
        percentage = count / total_qualified * 100
        
        # 按候选数排序
        sorted_groups = sorted(groups, key=lambda x: x['candidate_products'])
        candidate_counts = [g['candidate_products'] for g in sorted_groups]
        
        min_candidates = min(candidate_counts)
        max_candidates = max(candidate_counts)
        
        # 找中位数组合
        median_idx = len(sorted_groups) // 2
        median_group = sorted_groups[median_idx]
        
        median_info = f"store={median_group['store_id']}, 候选={median_group['candidate_products']}, 完整={median_group['complete_30day_candidates']}"
        
        print(f"{first_cat:<20} {count:<10} {percentage:>6.1f}%   {min_candidates:>3}-{max_candidates:<3}{'':>10} {median_info}")
        
        category_stats.append({
            'first_category_id': first_cat,
            'count': count,
            'percentage': percentage,
            'min_candidates': min_candidates,
            'max_candidates': max_candidates,
            'median_group': median_group,
            'all_groups': sorted_groups
        })
    
    print("")
    print("=" * 80)
    print("详细信息：每个 first_category_id 下的所有组合")
    print("=" * 80)
    print("")
    
    for stat in category_stats:
        first_cat = stat['first_category_id']
        groups = stat['all_groups']
        
        print(f"\n## first_category_id = {first_cat} ({stat['count']} 个组合)")
        print("-" * 80)
        print(f"{'序号':<6} {'store_id':<12} {'候选数':<10} {'完整候选':<12} {'跨度':<8} {'中位数标记'}")
        print("-" * 80)
        
        median_idx = len(groups) // 2
        
        for idx, group in enumerate(groups):
            is_median = "← 中位数" if idx == median_idx else ""
            is_store166 = "← store166" if group['store_id'] == 166 else ""
            marker = f"{is_median} {is_store166}".strip()
            
            print(f"{idx+1:<6} {group['store_id']:<12} {group['candidate_products']:<10} "
                  f"{group['complete_30day_candidates']:<12} {group['second_category_span']:<8} {marker}")
    
    # 检查 store166 是否在满足条件的组合中
    print("")
    print("=" * 80)
    print("store166 状态检查")
    print("=" * 80)
    print("")
    
    store166_groups = [g for g in qualified_groups if g['store_id'] == 166]
    
    if store166_groups:
        print(f"✓ store166 存在于满足条件的组合中（共 {len(store166_groups)} 个组合）")
        print("")
        for g in store166_groups:
            print(f"  - first_category_id={g['first_category_id']}, "
                  f"候选数={g['candidate_products']}, "
                  f"完整候选={g['complete_30day_candidates']}, "
                  f"跨度={g['second_category_span']}")
    else:
        print("✗ store166 不在满足条件的组合中")
    
    # 生成选择建议
    print("")
    print("=" * 80)
    print("选择建议（按预定规则）")
    print("=" * 80)
    print("")
    
    print("**预定选择规则**:")
    print("1. 从满足条件的组合中，按 first_category_id 分组")
    print("2. 每个 first_category_id 分组内，取中位数附近的1组")
    print("3. 保留 store166（如果满足条件）作为其中一组")
    print("4. 总共选择4-5组，覆盖至少3个不同的 first_category_id")
    print("")
    
    # 选择策略1：取前N个最多组合的first_category
    TOP_N = 5
    top_categories = sorted_categories[:TOP_N]
    
    print(f"**策略1：从组合数最多的{TOP_N}个first_category中各选1组（中位数）**")
    print("")
    
    selected_groups_strategy1 = []
    for first_cat, groups in top_categories:
        sorted_groups = sorted(groups, key=lambda x: x['candidate_products'])
        median_idx = len(sorted_groups) // 2
        median_group = sorted_groups[median_idx]
        selected_groups_strategy1.append(median_group)
        
        print(f"  - first_category={first_cat}, store={median_group['store_id']}, "
              f"候选数={median_group['candidate_products']}, "
              f"完整候选={median_group['complete_30day_candidates']}")
    
    # 检查是否包含store166
    if not any(g['store_id'] == 166 for g in selected_groups_strategy1) and store166_groups:
        print("")
        print("  注意: 上述选择未包含store166，需要手动替换或添加")
        print(f"  store166可用组合: first_category={store166_groups[0]['first_category_id']}")
    
    print("")
    print(f"**策略2：确保包含store166的选择**")
    print("")
    
    if store166_groups:
        selected_groups_strategy2 = []
        
        # 先添加store166
        store166_group = store166_groups[0]
        store166_first_cat = store166_group['first_category_id']
        selected_groups_strategy2.append(store166_group)
        
        print(f"  1. [保留] first_category={store166_first_cat}, store=166, "
              f"候选数={store166_group['candidate_products']}, "
              f"完整候选={store166_group['complete_30day_candidates']}")
        
        # 从其他first_category中选择
        other_categories = [cat for cat, _ in top_categories if cat != store166_first_cat][:4]
        
        for idx, first_cat in enumerate(other_categories, start=2):
            groups = category_groups[first_cat]
            sorted_groups = sorted(groups, key=lambda x: x['candidate_products'])
            median_idx = len(sorted_groups) // 2
            median_group = sorted_groups[median_idx]
            selected_groups_strategy2.append(median_group)
            
            print(f"  {idx}. [新增] first_category={first_cat}, store={median_group['store_id']}, "
                  f"候选数={median_group['candidate_products']}, "
                  f"完整候选={median_group['complete_30day_candidates']}")
        
        # 保存策略2的选择到JSON
        output_path = PROJECT_ROOT / "outputs" / "dataset_audit" / "d4_selected_target_groups.json"
        selected_data = {
            "selection_strategy": "strategy2_with_store166",
            "selection_rules": [
                "1. 从满足条件的组合中，按 first_category_id 分组",
                "2. 保留 store166（first_category=15）作为第一组",
                "3. 从组合数最多的其他 first_category 中，各选取中位数候选数的1组",
                "4. 总共选择5组，覆盖5个不同的 first_category_id",
                "5. 避免选择候选数最高或最低的组合，取中位数确保代表性"
            ],
            "total_qualified_groups": total_qualified,
            "unique_first_categories": len(category_groups),
            "selected_groups": [
                {
                    "store_id": g['store_id'],
                    "first_category_id": g['first_category_id'],
                    "candidate_products": g['candidate_products'],
                    "complete_30day_candidates": g['complete_30day_candidates'],
                    "second_category_span": g['second_category_span'],
                    "second_category_ids": g['second_category_ids'],
                    "date_range_start": g['date_range_start'],
                    "date_range_end": g['date_range_end'],
                }
                for g in selected_groups_strategy2
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(selected_data, f, indent=2, ensure_ascii=False)
        
        print("")
        print(f"✓ 选择结果已保存到: {output_path}")
    
    print("")
    print("=" * 80)
    print("汇总")
    print("=" * 80)
    print("")
    print(f"- 满足条件的组合: {total_qualified} 个")
    print(f"- 不同的 first_category_id: {len(category_groups)} 种")
    print(f"- 推荐选择: 5 组（覆盖5个不同的first_category）")
    print(f"- 包含 store166: {'是' if store166_groups else '否'}")
    print("")


if __name__ == "__main__":
    main()
