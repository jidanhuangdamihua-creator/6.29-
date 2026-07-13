#!/usr/bin/env python3
"""D4 Source Pool Final Audit - Check store=166, category=20 products"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("="*80)
print("D4 SOURCE POOL - STORE 166, CATEGORY 20 AUDIT")
print("="*80)

# Load parquets
source_path = PROJECT_ROOT / "数据集/固化数据/dataset4-source.parquet"
target_path = PROJECT_ROOT / "数据集/固化数据/dataset4-target.parquet"

print(f"\nLoading source parquet: {source_path.name}")
source_df = pd.read_parquet(source_path)

print(f"Loading target parquet: {target_path.name}")
target_df = pd.read_parquet(target_path)

print(f"\nSource parquet shape: {source_df.shape}")
print(f"Target parquet shape: {target_df.shape}")

# Check store=166, category=20 in SOURCE
print("\n" + "="*80)
print("SOURCE PARQUET - Store 166, Category 20")
print("="*80)

domain_entities = source_df[
    (source_df['store_id'] == 166) & 
    (source_df['second_category_id'] == 20)
]['product_id'].unique()

print(f"\nUnique products found: {len(domain_entities)}")
print(f"Product IDs: {sorted(domain_entities)}")

# Check each product's date coverage
print("\nDetailed product information:")
for pid in sorted(domain_entities):
    product_df = source_df[
        (source_df['store_id'] == 166) & 
        (source_df['product_id'] == pid)
    ]
    min_date = product_df['date'].min()
    max_date = product_df['date'].max()
    unique_dates = product_df['date'].nunique()
    
    print(f"\nproduct_id={pid}:")
    print(f"  Total rows: {len(product_df)}")
    print(f"  Unique dates: {unique_dates}")
    print(f"  Date range: {min_date} to {max_date}")
    
    # Check if covers the 30-day observation window
    # Observation window: 2024-11-17 to 2024-12-16 (30 days before train start)
    obs_start = pd.Timestamp("2024-11-17")
    obs_end = pd.Timestamp("2024-12-16")
    
    in_window = product_df[
        (product_df['date'] >= obs_start) & 
        (product_df['date'] <= obs_end)
    ]
    
    if len(in_window) > 0:
        window_dates = in_window['date'].nunique()
        print(f"  Observation window (2024-11-17 to 2024-12-16): {window_dates} days")
        if window_dates == 30:
            print(f"  ✅ Complete 30-day observation")
        else:
            print(f"  ❌ Incomplete observation (missing {30 - window_dates} days)")
    else:
        print(f"  ❌ No data in observation window")

# Check TARGET
print("\n" + "="*80)
print("TARGET PARQUET - Store 166, Category 20")
print("="*80)

target_entities = target_df[
    (target_df['store_id'] == 166) & 
    (target_df['second_category_id'] == 20)
]['product_id'].unique()

print(f"\nTarget products: {sorted(target_entities)}")

# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

all_products = set(domain_entities) | set(target_entities)
target_set = set(target_entities)
source_set = set(domain_entities)

print(f"\nAll products in domain (source ∪ target): {sorted(all_products)}")
print(f"Count: {len(all_products)}")

print(f"\nTarget products: {sorted(target_set)}")
print(f"Source products: {sorted(source_set)}")
print(f"Overlap (products in both): {sorted(target_set & source_set)}")

print(f"\nSource-only products (candidates): {sorted(source_set - target_set)}")
print(f"Candidate count: {len(source_set - target_set)}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

candidates = sorted(source_set - target_set)
print(f"\nValid source candidates for D4-without: {candidates}")
print(f"Required K: 3")
print(f"Available candidates: {len(candidates)}")

if len(candidates) >= 3:
    print("✅ Sufficient candidates")
else:
    print(f"❌ INSUFFICIENT: Only {len(candidates)} candidates, need 3")
    print(f"   This explains the 'valid candidates={len(candidates)} is below required K=3' error")

print("\n" + "="*80)
print("AUDIT COMPLETE")
print("="*80)
