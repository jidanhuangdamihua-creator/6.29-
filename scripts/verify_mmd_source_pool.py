#!/usr/bin/env python3
"""
验证MMD分析使用的source pool是否与fixed脚本一致
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.protocols.formal_input_paths import resolve_formal_dataset_paths

print("=" * 80)
print("验证 MMD 分析的 Source Pool 数据来源")
print("=" * 80)
print()

# ============================================================================
# 1. 读取 warehouse_mmd_scan.csv 中的 store166 数据
# ============================================================================

print("[1] 读取 warehouse_mmd_scan.csv 中的 store166 信息...")
mmd_csv_path = PROJECT_ROOT / "outputs/domain_adaptation/Dataset4/target_selection/warehouse_mmd_scan.csv"

if not mmd_csv_path.exists():
    print(f"  ✗ 文件不存在: {mmd_csv_path}")
    exit(1)

mmd_df = pd.read_csv(mmd_csv_path)
store166_row = mmd_df[mmd_df['store_id'] == 166]

if store166_row.empty:
    print("  ✗ 未找到 store166 的记录")
    exit(1)

store166_data = store166_row.iloc[0]

print(f"  ✓ 找到 store166 记录")
print(f"    - main_category: {store166_data['main_category']}")
print(f"    - source_sku_count: {store166_data['source_sku_count']}")
print(f"    - source_store_count: {store166_data['source_store_count']}")
print(f"    - target_feature_count: {store166_data['target_feature_count']}")
print(f"    - mmd: {store166_data['mmd']:.6f}")
print(f"    - p_value: {store166_data['p_value']}")

mmd_source_skus = int(store166_data['source_sku_count'])
mmd_source_stores = int(store166_data['source_store_count'])
mmd_main_category = int(store166_data['main_category'])

print()

# ============================================================================
# 2. 读取 fixed 脚本使用的固化数据
# ============================================================================

print("[2] 读取 fixed 脚本使用的固化数据...")
formal_paths = resolve_formal_dataset_paths(4, repository_root=PROJECT_ROOT)
source_parquet = formal_paths.source_path
target_parquet = formal_paths.target_path

if not source_parquet.exists():
    print(f"  ✗ 文件不存在: {source_parquet}")
    exit(1)

if not target_parquet.exists():
    print(f"  ✗ 文件不存在: {target_parquet}")
    exit(1)

source_df = pd.read_parquet(source_parquet)
target_df = pd.read_parquet(target_parquet)

print(f"  ✓ source_df shape: {source_df.shape}")
print(f"  ✓ target_df shape: {target_df.shape}")
print()

# ============================================================================
# 3. 按 fixed 脚本逻辑计算 WITH 场景的候选池
# ============================================================================

print("[3] 按 fixed 脚本逻辑计算 WITH (Cross-Store) 场景的候选池...")

# Get target info
target_entities = target_df[['store_id', 'product_id', 'first_category_id', 'second_category_id']].drop_duplicates()
target_store = target_entities.iloc[0]['store_id']
target_first_cat = target_entities.iloc[0]['first_category_id']
target_second_cat = target_entities.iloc[0]['second_category_id']

print(f"  Target info:")
print(f"    - store_id: {target_store}")
print(f"    - first_category_id: {target_first_cat}")
print(f"    - second_category_id: {target_second_cat}")
print(f"    - target entities: {len(target_entities)}")

# Get target entity set for exclusion
target_entity_set = set(
    target_entities[['store_id', 'product_id']].itertuples(index=False, name=None)
)

# Define observation window (from fixed script)
obs_start = pd.Timestamp("2024-11-17")
obs_end = pd.Timestamp("2024-12-16")
required_dates = pd.date_range(obs_start, obs_end, freq='D')

print(f"  Observation window: {obs_start.date()} to {obs_end.date()} ({len(required_dates)} days)")

# Ensure date column is datetime
if 'date' in source_df.columns:
    source_df['date'] = pd.to_datetime(source_df['date'])
elif 'dt' in source_df.columns:
    source_df['date'] = pd.to_datetime(source_df['dt'])

# WITH scenario: Cross-Store
with_filter = (source_df['store_id'] != target_store)

# Filter by second_category
candidates_second = source_df[
    with_filter &
    (source_df['second_category_id'] == target_second_cat)
]

# Get unique entities
candidate_entities_second = candidates_second[['store_id', 'product_id']].drop_duplicates()
candidate_entity_set_second = set(candidate_entities_second.itertuples(index=False, name=None))

# Exclude ALL target entities
candidate_entity_set_second = candidate_entity_set_second - target_entity_set

total_candidates_second = len(candidate_entity_set_second)

# Check 30-day completeness
def check_30day_completeness(entity_df, required_dates):
    entity_dates = pd.to_datetime(entity_df['date'].unique()).normalize()
    required_set = set(required_dates.normalize())
    entity_set = set(entity_dates)
    missing = required_set - entity_set
    return len(missing) == 0

valid_entities_second = []
for store_id, product_id in candidate_entity_set_second:
    entity_df = candidates_second[
        (candidates_second['store_id'] == store_id) &
        (candidates_second['product_id'] == product_id)
    ]
    if check_30day_completeness(entity_df, required_dates):
        valid_entities_second.append((store_id, product_id))

valid_count_second = len(valid_entities_second)

# Get unique stores and products
unique_stores = set(s for s, p in candidate_entity_set_second)
unique_products = set(p for s, p in candidate_entity_set_second)

print()
print(f"  WITH + second_category={target_second_cat}:")
print(f"    - Total candidate entities: {total_candidates_second}")
print(f"    - Valid candidate entities (30-day complete): {valid_count_second}")
print(f"    - Unique product_id values: {len(unique_products)}")
print(f"    - Unique stores contributing: {len(unique_stores)}")

fixed_total_entities = total_candidates_second
fixed_valid_entities = valid_count_second
fixed_unique_products = len(unique_products)
fixed_unique_stores = len(unique_stores)

# ============================================================================
# 4. 对比 MMD 数据与 fixed 脚本数据
# ============================================================================

print()
print("=" * 80)
print("[4] 数据对比结果")
print("=" * 80)
print()

print("| 指标 | MMD 分析 | Fixed 脚本 | 差异 | 一致性 |")
print("|------|----------|------------|------|--------|")

# Source SKU count
sku_diff = abs(mmd_source_skus - fixed_unique_products)
sku_match = "✓ 一致" if sku_diff <= 5 else "✗ 不一致"
print(f"| Source SKU 数 | {mmd_source_skus} | {fixed_unique_products} | {sku_diff} | {sku_match} |")

# Source store count
store_diff = abs(mmd_source_stores - fixed_unique_stores)
store_match = "✓ 一致" if store_diff <= 5 else "✗ 不一致"
print(f"| Source Store 数 | {mmd_source_stores} | {fixed_unique_stores} | {store_diff} | {store_match} |")

# Category
cat_match = "✓ 一致" if mmd_main_category == target_second_cat else "✗ 不一致"
print(f"| Category ID | {mmd_main_category} | {target_second_cat} | - | {cat_match} |")

print()

# ============================================================================
# 5. 结论和建议
# ============================================================================

print("=" * 80)
print("[5] 结论和建议")
print("=" * 80)
print()

all_match = (sku_diff <= 5 and store_diff <= 5 and mmd_main_category == target_second_cat)

if all_match:
    print("✓ MMD 分析使用的 source pool 与 fixed 脚本基本一致")
    print()
    print("说明:")
    print("  - 数据来源相同或非常接近")
    print("  - MMD 结果可以采信")
    print("  - 可以将 MMD 结果写入论文")
else:
    print("✗ MMD 分析使用的 source pool 与 fixed 脚本存在显著差异")
    print()
    print("可能原因:")
    
    if sku_diff > 100:
        print(f"  1. SKU 数量相差 {sku_diff} 个（{sku_diff/max(mmd_source_skus, fixed_unique_products)*100:.1f}%）")
        print("     - 可能使用了不同的 product_id 去重逻辑")
        print("     - 可能没有做 30 天完整性过滤")
        print("     - 可能使用了不同的 category 过滤条件")
    
    if store_diff > 50:
        print(f"  2. Store 数量相差 {store_diff} 个（{store_diff/max(mmd_source_stores, fixed_unique_stores)*100:.1f}%）")
        print("     - 可能使用了不同的 store 筛选逻辑")
        print("     - 可能包含了 target store 本身")
    
    if mmd_main_category != target_second_cat:
        print(f"  3. Category 不匹配: MMD 使用 {mmd_main_category}, fixed 使用 {target_second_cat}")
        print("     - 可能使用了 first_category 而非 second_category")
        print("     - 可能使用了不同的 category 定义")
    
    print()
    print("建议:")
    print("  1. ✗ 不要将当前的 MMD 结果写入论文")
    print("  2. 找到生成 warehouse_mmd_scan.csv 的原始脚本")
    print("  3. 对比脚本中的 source pool 提取逻辑")
    print("  4. 使用 fixed 脚本的逻辑重新计算 MMD")
    print("  5. 确保 source pool 定义与实验中使用的一致")

print()
print("=" * 80)
