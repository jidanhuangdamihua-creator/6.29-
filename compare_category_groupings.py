#!/usr/bin/env python3
"""
Check first_category vs second_category candidate pools for Store 166
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "数据集/原始数据/Dataset 4叮咚数据集/data/train.parquet"

STORE_166_TARGETS = {
    'store_id': 166,
    'target_products': [258, 311, 313, 432, 433],
    'first_category': 15,
    'second_category': 20,
}

# Corrected observation window
TARGET_TRAIN_START = pd.Timestamp("2024-12-16")
KNN_OBSERVED_END = TARGET_TRAIN_START - pd.Timedelta(days=1)  # 2024-12-15
TARGET_OBSERVED_START = KNN_OBSERVED_END - pd.Timedelta(days=29)  # 2024-11-16
SOURCE_OBSERVATION_CUTOFF = KNN_OBSERVED_END

def main():
    print("="*100)
    print("STORE 166 - first_category vs second_category COMPARISON")
    print("="*100)
    
    df = pd.read_parquet(DATA_PATH)
    df['dt'] = pd.to_datetime(df['dt'])
    
    store_id = STORE_166_TARGETS['store_id']
    target_products = STORE_166_TARGETS['target_products']
    first_category = STORE_166_TARGETS['first_category']
    second_category = STORE_166_TARGETS['second_category']
    
    obs_dates = pd.date_range(TARGET_OBSERVED_START, KNN_OBSERVED_END, freq='D')
    required_obs_dates = set(obs_dates.normalize())
    
    print(f"\nTarget: Store {store_id}")
    print(f"  first_category: {first_category}")
    print(f"  second_category: {second_category}")
    print(f"  Target products: {target_products}")
    print(f"\nObservation window: {TARGET_OBSERVED_START.date()} to {KNN_OBSERVED_END.date()} (30 days)")
    print(f"Source cutoff: <= {SOURCE_OBSERVATION_CUTOFF.date()}")
    
    # ========================================================================
    # Option 1: second_category grouping
    # ========================================================================
    print("\n" + "="*100)
    print(f"OPTION 1: second_category={second_category} (WITHOUT scenario)")
    print("="*100)
    
    source_second = df[
        (df['store_id'] == store_id) &
        (df['second_category_id'] == second_category) &
        (~df['product_id'].isin(target_products)) &
        (df['dt'] <= SOURCE_OBSERVATION_CUTOFF)
    ]
    
    candidates_second = source_second['product_id'].unique()
    print(f"\nTotal candidates: {len(candidates_second)}")
    
    valid_second = []
    for product_id in sorted(candidates_second):
        product_data = source_second[source_second['product_id'] == product_id]
        product_dates = pd.to_datetime(product_data['dt'].unique())
        obs_dates_product = product_dates[(product_dates >= TARGET_OBSERVED_START) & (product_dates <= KNN_OBSERVED_END)]
        covered = set(obs_dates_product.normalize())
        if len(covered) == len(required_obs_dates):
            valid_second.append(product_id)
            print(f"  ✅ Product {product_id}: 30/30 days")
        else:
            print(f"  ❌ Product {product_id}: {len(covered)}/30 days")
    
    print(f"\nValid candidates: {len(valid_second)}")
    print(f"K≥3 feasible: {'✅ YES' if len(valid_second) >= 3 else '❌ NO'}")
    
    # ========================================================================
    # Option 2: first_category grouping
    # ========================================================================
    print("\n" + "="*100)
    print(f"OPTION 2: first_category={first_category} (WITHOUT scenario)")
    print("="*100)
    
    source_first = df[
        (df['store_id'] == store_id) &
        (df['first_category_id'] == first_category) &
        (~df['product_id'].isin(target_products)) &
        (df['dt'] <= SOURCE_OBSERVATION_CUTOFF)
    ]
    
    candidates_first = source_first['product_id'].unique()
    print(f"\nTotal candidates: {len(candidates_first)}")
    
    # Show second_category distribution
    first_products_info = source_first[['product_id', 'second_category_id']].drop_duplicates()
    second_cat_dist = first_products_info['second_category_id'].value_counts()
    print(f"\nSecond category distribution:")
    for second_cat, count in second_cat_dist.items():
        print(f"  second_category={int(second_cat)}: {count} products")
    
    valid_first = []
    for product_id in sorted(candidates_first):
        product_data = source_first[source_first['product_id'] == product_id]
        product_dates = pd.to_datetime(product_data['dt'].unique())
        obs_dates_product = product_dates[(product_dates >= TARGET_OBSERVED_START) & (product_dates <= KNN_OBSERVED_END)]
        covered = set(obs_dates_product.normalize())
        
        # Get second_category for this product
        product_second_cat = int(product_data['second_category_id'].iloc[0])
        
        if len(covered) == len(required_obs_dates):
            valid_first.append(product_id)
            print(f"  ✅ Product {product_id} (second_cat={product_second_cat}): 30/30 days")
        else:
            print(f"  ❌ Product {product_id} (second_cat={product_second_cat}): {len(covered)}/30 days")
    
    print(f"\nValid candidates: {len(valid_first)}")
    print(f"K≥3 feasible: {'✅ YES' if len(valid_first) >= 3 else '❌ NO'}")
    if len(valid_first) > 0:
        print(f"Valid candidate IDs: {sorted([int(x) for x in valid_first])}")
    
    # ========================================================================
    # Option 3: WITH scenario (cross-store, second_category)
    # ========================================================================
    print("\n" + "="*100)
    print(f"OPTION 3: second_category={second_category} (WITH scenario - cross-store)")
    print("="*100)
    
    source_with = df[
        (df['store_id'] != store_id) &
        (df['second_category_id'] == second_category) &
        (~df['product_id'].isin(target_products)) &
        (df['dt'] <= SOURCE_OBSERVATION_CUTOFF)
    ]
    
    candidates_with = source_with['product_id'].unique()
    print(f"\nTotal candidates (from other stores): {len(candidates_with)}")
    
    if len(candidates_with) > 50:
        print(f"(Too many to check individually, checking first 20)")
        candidates_to_check = sorted(candidates_with)[:20]
    else:
        candidates_to_check = sorted(candidates_with)
    
    valid_with = []
    for product_id in candidates_to_check:
        product_data = source_with[source_with['product_id'] == product_id]
        product_dates = pd.to_datetime(product_data['dt'].unique())
        obs_dates_product = product_dates[(product_dates >= TARGET_OBSERVED_START) & (product_dates <= KNN_OBSERVED_END)]
        covered = set(obs_dates_product.normalize())
        
        if len(covered) == len(required_obs_dates):
            valid_with.append(product_id)
    
    # Count all valid
    if len(candidates_with) > 50:
        print(f"Checking remaining {len(candidates_with) - 20} candidates...")
        for product_id in sorted(candidates_with)[20:]:
            product_data = source_with[source_with['product_id'] == product_id]
            product_dates = pd.to_datetime(product_data['dt'].unique())
            obs_dates_product = product_dates[(product_dates >= TARGET_OBSERVED_START) & (product_dates <= KNN_OBSERVED_END)]
            covered = set(obs_dates_product.normalize())
            if len(covered) == len(required_obs_dates):
                valid_with.append(product_id)
    
    print(f"\nValid candidates: {len(valid_with)}")
    print(f"K≥3 feasible: {'✅ YES' if len(valid_with) >= 3 else '❌ NO'}")
    if len(valid_with) <= 10:
        print(f"Valid candidate IDs: {sorted([int(x) for x in valid_with])}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    print(f"\nStore {store_id} - Candidate pool comparison:")
    print(f"\n  Option 1 (WITHOUT + second_category={second_category}):")
    print(f"    Total: {len(candidates_second)}, Valid: {len(valid_second)}, K≥3: {'✅' if len(valid_second) >= 3 else '❌'}")
    
    print(f"\n  Option 2 (WITHOUT + first_category={first_category}):")
    print(f"    Total: {len(candidates_first)}, Valid: {len(valid_first)}, K≥3: {'✅' if len(valid_first) >= 3 else '❌'}")
    
    print(f"\n  Option 3 (WITH + second_category={second_category}):")
    print(f"    Total: {len(candidates_with)}, Valid: {len(valid_with)}, K≥3: {'✅' if len(valid_with) >= 3 else '❌'}")
    
    print(f"\n{'Recommendation':}")
    if len(valid_first) >= 3:
        print(f"  ✅ Use first_category={first_category} grouping (WITHOUT scenario)")
        print(f"     This provides {len(valid_first)} valid candidates")
    elif len(valid_with) >= 3:
        print(f"  ✅ Use WITH scenario (cross-store, second_category={second_category})")
        print(f"     This provides {len(valid_with)} valid candidates")
    else:
        print(f"  ⚠️  No configuration satisfies K≥3 for this store")
        print(f"     May need to adjust window length or use different category granularity")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
