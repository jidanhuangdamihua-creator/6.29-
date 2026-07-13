#!/usr/bin/env python3
"""
Investigate category anonymization and document limitations
Since category IDs are encoded, semantic validation is impossible
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("=" * 100)
print("D4 CATEGORY ANONYMIZATION INVESTIGATION")
print("=" * 100)

# Load data
source_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-source.parquet')
target_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-target.parquet')

targets = target_df[['store_id', 'product_id', 'first_category_id', 'second_category_id']].drop_duplicates()
first_cat = targets['first_category_id'].iloc[0]
second_cat = targets['second_category_id'].iloc[0]

print("\n📋 DATA SOURCE:")
print("   Dataset: FreshRetailNet-LT (Dingdong Inc.)")
print("   Domain: Fresh Retail / Perishable Goods")
print("   README: 'The encoded first/second/third category id'")
print("   → All category IDs are anonymized")

print("\n" + "=" * 100)
print("CATEGORY HIERARCHY STRUCTURE (Numerical Only)")
print("=" * 100)

# Build category hierarchy for target
store_166_first = source_df[
    (source_df['store_id'] == 166) &
    (source_df['first_category_id'] == first_cat)
][['product_id', 'second_category_id', 'third_category_id']].drop_duplicates()

print(f"\nTarget categories:")
print(f"  first_category_id:  {first_cat}")
print(f"  second_category_id: {second_cat}")

# Map structure
print(f"\nHierarchy mapping for first_category={first_cat} in store 166:")
second_cats_in_first = store_166_first['second_category_id'].unique()
print(f"  Contains second_category_id: {sorted(second_cats_in_first)}")
print(f"  Total distinct second_categories: {len(second_cats_in_first)}")

# Product distribution
for sec_cat in sorted(second_cats_in_first):
    count = len(store_166_first[store_166_first['second_category_id'] == sec_cat])
    marker = " ← TARGET CATEGORY" if sec_cat == second_cat else ""
    print(f"    second_category={sec_cat}: {count} products{marker}")

# Global statistics
print(f"\n" + "=" * 100)
print("GLOBAL CATEGORY STATISTICS")
print("=" * 100)

print("\nAcross all stores/products:")
print(f"  - Unique first_category_id: {source_df['first_category_id'].nunique()}")
print(f"  - Unique second_category_id: {source_df['second_category_id'].nunique()}")
print(f"  - Unique third_category_id: {source_df['third_category_id'].nunique()}")

# Average products per category
avg_products_first = source_df.groupby('first_category_id')['product_id'].nunique().mean()
avg_products_second = source_df.groupby('second_category_id')['product_id'].nunique().mean()
print(f"\nAverage products per category:")
print(f"  - first_category:  {avg_products_first:.1f}")
print(f"  - second_category: {avg_products_second:.1f}")
print(f"  - Granularity ratio: {avg_products_first / avg_products_second:.2f}x broader")

print("\n" + "=" * 100)
print("SEMANTIC VALIDATION: IMPOSSIBLE")
print("=" * 100)

print("\n❌ LIMITATION IDENTIFIED:")
print("   The dataset documentation explicitly states that category IDs are 'encoded'")
print("   No category name mapping table is provided")
print("   Cannot verify business semantics of:")
print(f"     - first_category={first_cat}")
print(f"     - second_category={second_cat}")
print(f"     - Their mutual relationship")

print("\n⚠️  IMPLICATIONS FOR PAPER:")
print("   1. Cannot claim 'semantically coherent category grouping'")
print("   2. Cannot validate if first_category preserves business relevance")
print("   3. Must document this as a data limitation")

print("\n" + "=" * 100)
print("RECOMMENDED PAPER WORDING")
print("=" * 100)

print("""
📝 SUGGESTED TEXT FOR METHODOLOGY SECTION:

"Due to the sparsity of same-store, same-category candidates in the D4 dataset,
we relaxed the category granularity from second-level to first-level categories
for the WITHOUT information-sharing scenario. Specifically:

- **WITH scenario**: Allows cross-store candidates within first_category=15
- **WITHOUT scenario**: Restricts to same-store candidates within first_category=15

This modification ensures K≥3 candidate availability while maintaining the
fundamental distinction between information-sharing scenarios (same-store vs
cross-store sources).

⚠️ **Limitation**: The D4 dataset (FreshRetailNet-LT) uses anonymized category IDs
without semantic labels. We cannot verify whether first_category=15 represents a
coherent business grouping. The numerical analysis shows first_category=15 contains
{} distinct second-level categories in the target store, suggesting it is a
broader product grouping. This may introduce additional source heterogeneity
compared to fine-grained category matching, potentially weakening the transfer
learning effectiveness. Future work should validate this approach on datasets
with explicit category semantics."
""".format(len(second_cats_in_first)))

print("\n" + "=" * 100)
print("ALTERNATIVE APPROACHES (If Semantic Validation Required)")
print("=" * 100)

print("""
If reviewers require semantic validation, consider:

1. **Contact dataset creators**
   - Request category name mapping or business taxonomy
   - Cite their response in paper

2. **Select different target domain**
   - Find store/category combinations with K≥3 at second_category level
   - Avoid category granularity changes entirely

3. **Reduce K to 2**
   - Document as protocol deviation
   - Acknowledge potential impact on multi-source methods

4. **Skip D4-without entirely**
   - Run only D4-with (cross-store)
   - Focus experiments on other datasets (D1-D3, D5-D6)
""")

print("\n" + "=" * 100)
print("INVESTIGATION COMPLETE")
print("=" * 100)

print("\n💡 RECOMMENDATION:")
print("   Given the anonymization constraint, the most defensible approach is:")
print("   1. Document the limitation honestly in the paper")
print("   2. Use first_category for BOTH with/without (consistency)")
print("   3. Add disclaimer about potential semantic heterogeneity")
print("   4. Emphasize this is a data-driven pragmatic choice")
print("\n   Reviewers will respect honest documentation more than unverifiable claims.")
