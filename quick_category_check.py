#!/usr/bin/env python3
"""Quick category hierarchy comparison"""
import pandas as pd

source_df = pd.read_parquet('数据集/固化数据/dataset4-source.parquet')
target_df = pd.read_parquet('数据集/固化数据/dataset4-target.parquet')

targets = target_df[['store_id', 'product_id', 'first_category_id', 'second_category_id']].drop_duplicates()

print("TARGET | SECOND_CAT | FIRST_CAT | SAME_STORE_2ND | SAME_STORE_1ST | CHANGE")
print("-" * 80)

for _, t in targets.iterrows():
    # Count with second_category
    same_2nd = source_df[
        (source_df['store_id'] == t['store_id']) &
        (source_df['second_category_id'] == t['second_category_id']) &
        (source_df['product_id'] != t['product_id'])
    ]['product_id'].nunique()
    
    # Count with first_category
    same_1st = source_df[
        (source_df['store_id'] == t['store_id']) &
        (source_df['first_category_id'] == t['first_category_id']) &
        (source_df['product_id'] != t['product_id'])
    ]['product_id'].nunique()
    
    status = "✅ FIXED!" if same_2nd < 3 and same_1st >= 3 else ("OK" if same_2nd >= 3 else "FAIL")
    
    print(f"{t['product_id']:6d} | {t['second_category_id']:10d} | {t['first_category_id']:9d} | {same_2nd:14d} | {same_1st:14d} | {status}")

print("\nSUMMARY:")
print(f"- Store: {targets.iloc[0]['store_id']}")
print(f"- Targets: {len(targets)}")

# Count feasibility changes
feasible_2nd = sum(1 for _, t in targets.iterrows() if source_df[
    (source_df['store_id'] == t['store_id']) &
    (source_df['second_category_id'] == t['second_category_id']) &
    (source_df['product_id'] != t['product_id'])
]['product_id'].nunique() >= 3)

feasible_1st = sum(1 for _, t in targets.iterrows() if source_df[
    (source_df['store_id'] == t['store_id']) &
    (source_df['first_category_id'] == t['first_category_id']) &
    (source_df['product_id'] != t['product_id'])
]['product_id'].nunique() >= 3)

print(f"\nWITHOUT-mode feasible:")
print(f"  second_category: {feasible_2nd}/{len(targets)} targets")
print(f"  first_category:  {feasible_1st}/{len(targets)} targets")
print(f"  Improvement: +{feasible_1st - feasible_2nd} targets")
