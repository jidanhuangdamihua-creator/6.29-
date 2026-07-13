#!/usr/bin/env python3
"""
D4 Category Hierarchy Impact Analysis (Read-Only)

Analyze the impact of changing grouping field from second_category_id to first_category_id
on source candidate pool size and with/without scenario differentiation.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("=" * 100)
print("D4 CATEGORY HIERARCHY IMPACT ANALYSIS")
print("=" * 100)

# Load parquets
source_path = PROJECT_ROOT / "数据集/固化数据/dataset4-source.parquet"
target_path = PROJECT_ROOT / "数据集/固化数据/dataset4-target.parquet"

print(f"\nLoading source parquet...")
source_df = pd.read_parquet(source_path)
print(f"Loading target parquet...")
target_df = pd.read_parquet(target_path)

print(f"\nSource shape: {source_df.shape}")
print(f"Target shape: {target_df.shape}")

# Get target list
target_entities = target_df[['store_id', 'product_id', 'first_category_id', 'second_category_id']].drop_duplicates()
print(f"\nTarget entities: {len(target_entities)}")
print(target_entities.to_string(index=False))

print("\n" + "=" * 100)
print("SCENARIO 1: CURRENT (second_category_id)")
print("=" * 100)

results_second = []

for _, target in target_entities.iterrows():
    store_id = target['store_id']
    product_id = target['product_id']
    second_cat = target['second_category_id']
    
    # Same-store, same-second-category candidates (excluding current target)
    same_store_candidates = source_df[
        (source_df['store_id'] == store_id) &
        (source_df['second_category_id'] == second_cat) &
        (source_df['product_id'] != product_id)
    ]['product_id'].unique()
    
    # Cross-store, same-second-category candidates
    cross_store_candidates = source_df[
        (source_df['store_id'] != store_id) &
        (source_df['second_category_id'] == second_cat)
    ]['product_id'].unique()
    
    results_second.append({
        'store_id': store_id,
        'product_id': product_id,
        'second_category_id': second_cat,
        'same_store_count': len(same_store_candidates),
        'cross_store_count': len(cross_store_candidates),
        'same_store_products': sorted(same_store_candidates.tolist()),
        'has_k3_same_store': len(same_store_candidates) >= 3,
        'has_k3_cross_store': len(cross_store_candidates) >= 3,
    })

df_second = pd.DataFrame(results_second)
print("\nPer-target candidate counts:")
print(df_second[['product_id', 'second_category_id', 'same_store_count', 'cross_store_count', 'has_k3_same_store', 'has_k3_cross_store']].to_string(index=False))

print("\n" + "=" * 100)
print("SCENARIO 2: ALTERNATIVE (first_category_id)")
print("=" * 100)

results_first = []

for _, target in target_entities.iterrows():
    store_id = target['store_id']
    product_id = target['product_id']
    first_cat = target['first_category_id']
    
    # Same-store, same-first-category candidates (excluding current target)
    same_store_candidates = source_df[
        (source_df['store_id'] == store_id) &
        (source_df['first_category_id'] == first_cat) &
        (source_df['product_id'] != product_id)
    ]['product_id'].unique()
    
    # Cross-store, same-first-category candidates
    cross_store_candidates = source_df[
        (source_df['store_id'] != store_id) &
        (source_df['first_category_id'] == first_cat)
    ]['product_id'].unique()
    
    results_first.append({
        'store_id': store_id,
        'product_id': product_id,
        'first_category_id': first_cat,
        'same_store_count': len(same_store_candidates),
        'cross_store_count': len(cross_store_candidates),
        'same_store_products': sorted(same_store_candidates.tolist()),
        'has_k3_same_store': len(same_store_candidates) >= 3,
        'has_k3_cross_store': len(cross_store_candidates) >= 3,
    })

df_first = pd.DataFrame(results_first)
print("\nPer-target candidate counts:")
print(df_first[['product_id', 'first_category_id', 'same_store_count', 'cross_store_count', 'has_k3_same_store', 'has_k3_cross_store']].to_string(index=False))

print("\n" + "=" * 100)
print("COMPARISON & IMPACT ANALYSIS")
print("=" * 100)

# Create comparison dataframe
comparison = pd.DataFrame({
    'product_id': df_second['product_id'],
    'second_cat': df_second['second_category_id'],
    'first_cat': df_first['first_category_id'],
    'same_store_2nd': df_second['same_store_count'],
    'same_store_1st': df_first['same_store_count'],
    'delta_same_store': df_first['same_store_count'] - df_second['same_store_count'],
    'cross_store_2nd': df_second['cross_store_count'],
    'cross_store_1st': df_first['cross_store_count'],
    'k3_same_2nd': df_second['has_k3_same_store'],
    'k3_same_1st': df_first['has_k3_same_store'],
    'status_change': (df_second['has_k3_same_store'] != df_first['has_k3_same_store'])
})

print("\n1. CANDIDATE POOL SIZE CHANGES:")
print(comparison[['product_id', 'second_cat', 'first_cat', 'same_store_2nd', 'same_store_1st', 'delta_same_store']].to_string(index=False))

print("\n2. K=3 FEASIBILITY CHANGES:")
targets_fixed = comparison[comparison['status_change'] & comparison['k3_same_1st']].copy()
print(f"\nTargets that become feasible for without-mode (same-store K≥3):")
if len(targets_fixed) > 0:
    print(targets_fixed[['product_id', 'same_store_2nd', 'same_store_1st']].to_string(index=False))
    print(f"\n✅ {len(targets_fixed)} target(s) would be fixed!")
else:
    print("❌ No targets would be fixed.")

print("\n3. SCENARIO DIFFERENTIATION IMPACT:")
print(f"\nWith second_category_id:")
print(f"  - WITHOUT feasible: {df_second['has_k3_same_store'].sum()}/{len(df_second)} targets")
print(f"  - WITH feasible: {df_second['has_k3_cross_store'].sum()}/{len(df_second)} targets")
print(f"  - Scenario differentiation: {df_second['has_k3_cross_store'].sum() - df_second['has_k3_same_store'].sum()} targets difference")

print(f"\nWith first_category_id:")
print(f"  - WITHOUT feasible: {df_first['has_k3_same_store'].sum()}/{len(df_first)} targets")
print(f"  - WITH feasible: {df_first['has_k3_cross_store'].sum()}/{len(df_first)} targets")
print(f"  - Scenario differentiation: {df_first['has_k3_cross_store'].sum() - df_first['has_k3_same_store'].sum()} targets difference")

# Calculate category granularity
print("\n" + "=" * 100)
print("CATEGORY HIERARCHY STATISTICS")
print("=" * 100)

# Unique categories
unique_first = source_df['first_category_id'].nunique()
unique_second = source_df['second_category_id'].nunique()

print(f"\nGlobal category counts:")
print(f"  - Unique first_category_id: {unique_first}")
print(f"  - Unique second_category_id: {unique_second}")
print(f"  - Hierarchy ratio: {unique_second / unique_first:.2f}x more granular")

# Products per category
products_per_first = source_df.groupby('first_category_id')['product_id'].nunique()
products_per_second = source_df.groupby('second_category_id')['product_id'].nunique()

print(f"\nAverage products per category:")
print(f"  - first_category_id: {products_per_first.mean():.1f} ± {products_per_first.std():.1f}")
print(f"  - second_category_id: {products_per_second.mean():.1f} ± {products_per_second.std():.1f}")

# For target store 166
store_166_df = source_df[source_df['store_id'] == 166]
print(f"\nFor store_id=166:")
print(f"  - Unique first_category_id: {store_166_df['first_category_id'].nunique()}")
print(f"  - Unique second_category_id: {store_166_df['second_category_id'].nunique()}")

# Category mapping for targets
print("\nTarget category mapping:")
for _, row in target_entities.iterrows():
    first = row['first_category_id']
    second = row['second_category_id']
    
    # Count products in same first category
    same_first = source_df[
        (source_df['store_id'] == row['store_id']) &
        (source_df['first_category_id'] == first)
    ]['product_id'].nunique()
    
    same_second = source_df[
        (source_df['store_id'] == row['store_id']) &
        (source_df['second_category_id'] == second)
    ]['product_id'].nunique()
    
    print(f"  product_id={row['product_id']}: first={first} ({same_first} products in store), second={second} ({same_second} products in store)")

print("\n" + "=" * 100)
print("SEMANTIC & BUSINESS RELEVANCE ANALYSIS")
print("=" * 100)

print("\n⚠️  POTENTIAL ISSUES WITH first_category_id:")
print("\n1. WEAKENED BUSINESS RELEVANCE:")
print("   - First-level categories are more coarse-grained")
print("   - Products may share first_category but have different consumer behavior patterns")
print("   - Example: 'Fresh Produce' (first) vs 'Leafy Vegetables' (second)")
print("   - KNN similarity assumption may be violated")

print("\n2. CHANGED PAPER SEMANTICS:")
print("   - Paper likely assumes 'same category' = fine-grained business category")
print("   - Changing to first_category relaxes the 'same domain' constraint")
print("   - May invalidate the cold-start transfer learning motivation")

print("\n3. SCENARIO DIFFERENTIATION:")
if df_first['has_k3_same_store'].sum() > df_second['has_k3_same_store'].sum():
    print("   - WITHOUT mode becomes MORE feasible (easier to find K≥3)")
    print("   - ⚠️  This REDUCES the distinction between WITH and WITHOUT")
    print("   - May weaken the experimental contrast the paper aims to show")
else:
    print("   - No significant change in scenario differentiation")

print("\n4. KNN EFFECTIVENESS:")
print("   - Broader category → more diverse source candidates")
print("   - May include products with weaker feature correlation")
print("   - Could degrade source selection quality and transfer performance")

print("\n" + "=" * 100)
print("RECOMMENDATIONS")
print("=" * 100)

print("\n✅ KEEP second_category_id IF:")
print("   1. You want to preserve paper semantics ('same fine-grained category')")
print("   2. You want strong WITH/WITHOUT differentiation")
print("   3. You prioritize source relevance over candidate availability")
print("   4. You can accept that some targets fail K<3 constraint")

print("\n⚠️  CONSIDER first_category_id IF:")
print("   1. Running experiments is more important than semantic purity")
print("   2. You're willing to document this as a protocol relaxation")
print("   3. You validate that first-level categories still have business coherence")
print("   4. You re-evaluate KNN performance on coarser grouping")

print("\n🔍 ALTERNATIVE APPROACHES:")
print("   1. Select different target store/category combinations with sufficient K≥3")
print("   2. Add more targets from categories with dense source pools")
print("   3. Reduce K to 2 (but document as deviation from paper)")
print("   4. Run D4-with only (cross-store) and skip D4-without")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
