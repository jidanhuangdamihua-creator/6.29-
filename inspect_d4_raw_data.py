#!/usr/bin/env python3
"""
Direct D4 Raw Data Inspection
==============================

Purpose: Directly inspect D4 train.parquet to answer:
1. What stores exist in the data?
2. What products exist for target stores (166, 155, 240, 293)?
3. What categories do these products belong to?
4. What is the actual date range for each product?

This is a READ-ONLY inspection, no complex validation logic.
"""

import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease install dependencies:")
    print("  pip install pandas pyarrow numpy")
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "数据集/原始数据/Dataset 4叮咚数据集/data/train.parquet"

TARGET_STORES = [166, 155, 240, 293]

# D4 target train window (from solidified config)
TARGET_TRAIN_START = pd.Timestamp("2024-12-16")
TARGET_TEST_END = pd.Timestamp("2025-07-13")

# Observation window (30 days before train start)
OBS_START = pd.Timestamp("2024-11-17")
OBS_END = pd.Timestamp("2024-12-16")

# ============================================================================
# Helper Functions
# ============================================================================

def load_data():
    """Load D4 train.parquet."""
    print(f"Loading: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        print(f"❌ File not found: {DATA_PATH}")
        sys.exit(1)
    
    file_size_mb = DATA_PATH.stat().st_size / 1024 / 1024
    print(f"File size: {file_size_mb:.1f} MB")
    
    try:
        df = pd.read_parquet(DATA_PATH)
        df['dt'] = pd.to_datetime(df['dt'])
        
        print(f"✓ Loaded {len(df):,} rows")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df['dt'].min().date()} to {df['dt'].max().date()}")
        print(f"  Stores: {df['store_id'].nunique()}")
        print(f"  Products: {df['product_id'].nunique()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def inspect_store_products(df, store_id):
    """Inspect products that appear in the target window for a given store."""
    print(f"\n{'='*100}")
    print(f"STORE {store_id} - Products in Target Window")
    print(f"{'='*100}")
    
    # Filter: store + target window
    store_target_data = df[
        (df['store_id'] == store_id) &
        (df['dt'] >= TARGET_TRAIN_START) &
        (df['dt'] <= TARGET_TEST_END)
    ]
    
    if store_target_data.empty:
        print(f"⚠️  No data for store {store_id} in target window")
        return None
    
    # Get unique products with their categories
    products = store_target_data[[
        'product_id',
        'first_category_id',
        'second_category_id',
        'third_category_id'
    ]].drop_duplicates()
    
    print(f"\nTotal products in target window: {len(products)}")
    print(f"\nProducts by category:")
    print("-" * 100)
    print(f"{'Product ID':<15} {'First Cat':<12} {'Second Cat':<12} {'Third Cat':<12}")
    print("-" * 100)
    
    for _, row in products.head(20).iterrows():
        print(f"{int(row['product_id']):<15} "
              f"{int(row['first_category_id']):<12} "
              f"{int(row['second_category_id']):<12} "
              f"{int(row['third_category_id']) if pd.notna(row['third_category_id']) else 'N/A':<12}")
    
    if len(products) > 20:
        print(f"... and {len(products) - 20} more products")
    
    # Category distribution
    print(f"\n{'Category Distribution':}")
    print(f"  Unique first_category: {products['first_category_id'].nunique()}")
    print(f"  Unique second_category: {products['second_category_id'].nunique()}")
    print(f"  Most common first_category: {products['first_category_id'].mode()[0]}")
    print(f"  Most common second_category: {products['second_category_id'].mode()[0]}")
    
    # For potential target products, check their full date range
    print(f"\n{'Date Coverage for Target Products':}")
    print("-" * 100)
    
    for product_id in products['product_id'].head(10):
        product_data = df[
            (df['store_id'] == store_id) &
            (df['product_id'] == product_id)
        ]
        
        dates = product_data['dt'].sort_values()
        date_start = dates.min()
        date_end = dates.max()
        total_days = (date_end - date_start).days + 1
        unique_days = dates.nunique()
        
        # Check observation window coverage
        obs_dates = dates[(dates >= OBS_START) & (dates <= OBS_END)]
        obs_coverage = len(obs_dates)
        
        print(f"  Product {int(product_id)}: "
              f"{date_start.date()} to {date_end.date()} "
              f"({unique_days}/{total_days} days, "
              f"obs window: {obs_coverage}/30)")
    
    return products


def inspect_store_source_candidates(df, store_id, sample_target_product_id, target_second_category):
    """Inspect source candidates for a target product (WITHOUT scenario)."""
    print(f"\n{'='*100}")
    print(f"STORE {store_id} - Source Candidates (WITHOUT scenario)")
    print(f"Target product: {sample_target_product_id}, second_category: {target_second_category}")
    print(f"{'='*100}")
    
    # Filter: same store, same second_category, NOT the target product, before target window
    source_data = df[
        (df['store_id'] == store_id) &
        (df['second_category_id'] == target_second_category) &
        (df['product_id'] != sample_target_product_id) &
        (df['dt'] < TARGET_TRAIN_START)
    ]
    
    candidate_products = source_data['product_id'].unique()
    print(f"\nTotal candidate products: {len(candidate_products)}")
    
    if len(candidate_products) == 0:
        print("⚠️  No candidates found!")
        return
    
    # Check 30-day completeness for each candidate
    required_obs_dates = pd.date_range(OBS_START, OBS_END, freq='D')
    
    print(f"\nChecking 30-day completeness (need all {len(required_obs_dates)} days):")
    print("-" * 100)
    print(f"{'Product ID':<15} {'Date Range':<40} {'Unique Days':<15} {'30-day Complete':<20}")
    print("-" * 100)
    
    valid_count = 0
    
    for product_id in candidate_products[:20]:
        product_data = source_data[source_data['product_id'] == product_id]
        product_dates = pd.to_datetime(product_data['dt'].unique())
        
        date_start = product_dates.min()
        date_end = product_dates.max()
        unique_days = len(product_dates)
        
        # Check 30-day completeness
        obs_dates_covered = set(product_dates.normalize()).intersection(set(required_obs_dates.normalize()))
        is_complete = len(obs_dates_covered) == len(required_obs_dates)
        
        if is_complete:
            valid_count += 1
        
        status = "✅ YES" if is_complete else f"❌ NO ({len(obs_dates_covered)}/30)"
        
        print(f"{int(product_id):<15} "
              f"{date_start.date()} to {date_end.date():<23} "
              f"{unique_days:<15} "
              f"{status:<20}")
    
    if len(candidate_products) > 20:
        print(f"... and {len(candidate_products) - 20} more candidates (checking remaining)")
        
        # Check remaining
        for product_id in candidate_products[20:]:
            product_data = source_data[source_data['product_id'] == product_id]
            product_dates = pd.to_datetime(product_data['dt'].unique())
            obs_dates_covered = set(product_dates.normalize()).intersection(set(required_obs_dates.normalize()))
            if len(obs_dates_covered) == len(required_obs_dates):
                valid_count += 1
    
    print("-" * 100)
    print(f"\nSummary:")
    print(f"  Total candidates: {len(candidate_products)}")
    print(f"  With 30-day completeness: {valid_count}")
    print(f"  K≥3 feasible: {'✅ YES' if valid_count >= 3 else '❌ NO'}")


def main():
    """Main inspection routine."""
    print("=" * 100)
    print("D4 RAW DATA INSPECTION")
    print("=" * 100)
    print(f"\nTarget stores: {TARGET_STORES}")
    print(f"Target window: {TARGET_TRAIN_START.date()} to {TARGET_TEST_END.date()}")
    print(f"Observation window: {OBS_START.date()} to {OBS_END.date()}")
    print()
    
    df = load_data()
    
    # Check which target stores actually exist
    existing_stores = df['store_id'].unique()
    print(f"\n{'Store Availability':}")
    for store_id in TARGET_STORES:
        exists = store_id in existing_stores
        status = "✅" if exists else "❌"
        print(f"  Store {store_id}: {status}")
    
    # Inspect each target store
    for store_id in TARGET_STORES:
        if store_id not in existing_stores:
            print(f"\n⚠️  Skipping store {store_id} (not in dataset)")
            continue
        
        products = inspect_store_products(df, store_id)
        
        if products is not None and len(products) > 0:
            # Use the most common second_category and its first product as example
            most_common_second_cat = products['second_category_id'].mode()[0]
            sample_product = products[
                products['second_category_id'] == most_common_second_cat
            ].iloc[0]['product_id']
            
            inspect_store_source_candidates(
                df,
                store_id,
                int(sample_product),
                int(most_common_second_cat)
            )
    
    print("\n" + "=" * 100)
    print("INSPECTION COMPLETE")
    print("=" * 100)
    print("\nThis raw data inspection should help verify:")
    print("1. What target products actually exist for each store")
    print("2. What their categories are")
    print("3. How many source candidates exist")
    print("4. How many satisfy 30-day completeness")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
