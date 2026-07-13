#!/usr/bin/env python3
"""
Check business semantics of category hierarchy
Understand what first_category=15 and second_category=20 actually represent
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("=" * 100)
print("D4 CATEGORY SEMANTICS ANALYSIS")
print("=" * 100)

# Load data
source_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-source.parquet')
target_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-target.parquet')

print("\nTarget categories:")
targets = target_df[['product_id', 'first_category_id', 'second_category_id', 'third_category_id']].drop_duplicates()
print(targets.to_string(index=False))

first_cat = targets['first_category_id'].iloc[0]
second_cat = targets['second_category_id'].iloc[0]

print(f"\n" + "=" * 100)
print(f"CATEGORY HIERARCHY FOR TARGETS")
print(f"=" * 100)
print(f"\nfirst_category_id:  {first_cat}")
print(f"second_category_id: {second_cat}")

# Find all products in store 166 with same first_category
print(f"\n" + "=" * 100)
print(f"PRODUCTS IN STORE 166 WITH first_category={first_cat}")
print(f"=" * 100)

store_166_first_cat = source_df[
    (source_df['store_id'] == 166) &
    (source_df['first_category_id'] == first_cat)
][['product_id', 'second_category_id', 'third_category_id']].drop_duplicates().sort_values('product_id')

print(f"\nTotal products: {len(store_166_first_cat)}")
print("\nProduct list with category hierarchy:")
print(store_166_first_cat.to_string(index=False))

# Group by second_category
print(f"\n" + "=" * 100)
print(f"SECOND_CATEGORY BREAKDOWN WITHIN first_category={first_cat}")
print(f"=" * 100)

second_cat_counts = store_166_first_cat.groupby('second_category_id').size().sort_index()
print("\nProducts per second_category:")
for sec_cat, count in second_cat_counts.items():
    marker = " ← TARGET CATEGORY" if sec_cat == second_cat else ""
    print(f"  second_category={sec_cat}: {count} products{marker}")

# Check category diversity
print(f"\n" + "=" * 100)
print(f"CATEGORY DIVERSITY ANALYSIS")
print(f"=" * 100)

print(f"\nIn store 166, first_category={first_cat}:")
print(f"  - Contains {len(second_cat_counts)} different second_category values")
print(f"  - Contains {len(store_166_first_cat)} unique products")

# Check if categories are evenly distributed
if len(second_cat_counts) > 1:
    print(f"\n⚠️  DIVERSITY WARNING:")
    print(f"  first_category={first_cat} spans {len(second_cat_counts)} second-level categories")
    print(f"  This suggests first_category is a broader business grouping")
    print(f"  Products from different second_category may have different:")
    print(f"    - Consumer purchase patterns")
    print(f"    - Seasonal/promotional behavior")
    print(f"    - Price ranges")
    print(f"    - Stock turnover rates")
else:
    print(f"\n✅ All products in first_category={first_cat} share second_category={second_cat}")
    print(f"  → first_category and second_category are equivalent for this domain")

# Global category statistics
print(f"\n" + "=" * 100)
print(f"GLOBAL CATEGORY STATISTICS")
print(f"=" * 100)

print("\nAcross all stores and products:")
print(f"  - Unique first_category_id: {source_df['first_category_id'].nunique()}")
print(f"  - Unique second_category_id: {source_df['second_category_id'].nunique()}")
print(f"  - Unique third_category_id: {source_df['third_category_id'].nunique()}")

# Average products per category level
print("\nAverage products per category:")
products_per_first = source_df.groupby('first_category_id')['product_id'].nunique().mean()
products_per_second = source_df.groupby('second_category_id')['product_id'].nunique().mean()
print(f"  - Per first_category: {products_per_first:.1f} products")
print(f"  - Per second_category: {products_per_second:.1f} products")

# Check consistency: do second_categories map to unique first_categories?
print(f"\n" + "=" * 100)
print(f"CATEGORY HIERARCHY CONSISTENCY")
print(f"=" * 100)

cat_mapping = source_df[['first_category_id', 'second_category_id']].drop_duplicates()
second_to_first = cat_mapping.groupby('second_category_id')['first_category_id'].nunique()
inconsistent = second_to_first[second_to_first > 1]

if len(inconsistent) > 0:
    print(f"\n⚠️  WARNING: {len(inconsistent)} second_category values map to multiple first_category values!")
    print("  This indicates inconsistent category hierarchy")
else:
    print("\n✅ Each second_category maps to exactly one first_category (consistent hierarchy)")

# Recommendation
print(f"\n" + "=" * 100)
print(f"SEMANTIC ASSESSMENT")
print(f"=" * 100)

if len(second_cat_counts) <= 2:
    print("\n✅ ACCEPTABLE SEMANTIC BROADENING:")
    print(f"  first_category={first_cat} contains ≤2 second_categories")
    print("  Business relevance likely preserved")
    print("  KNN source selection should remain meaningful")
elif len(second_cat_counts) <= 4:
    print("\n⚠️  MODERATE SEMANTIC BROADENING:")
    print(f"  first_category={first_cat} contains {len(second_cat_counts)} second_categories")
    print("  May reduce source relevance but still defensible")
    print("  Document as 'broader category grouping' in paper")
else:
    print("\n❌ SIGNIFICANT SEMANTIC BROADENING:")
    print(f"  first_category={first_cat} contains {len(second_cat_counts)} second_categories")
    print("  May severely weaken 'same category' assumption")
    print("  Consider alternative solutions instead")

print(f"\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print(f"=" * 100)

print("\nNOTE: Without category name labels, numerical IDs can't reveal true business meaning.")
print("Check original data documentation or business glossary if available.")
