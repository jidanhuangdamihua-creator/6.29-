#!/usr/bin/env python3
"""
D4 Raw Data Inspection - Fixed Version
========================================

Fixes:
1. Use correct target products and categories for store 166
2. Check observation window boundary calculation
3. Fix date display formatting
4. Report detailed date coverage
"""

import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "数据集/原始数据/Dataset 4叮咚数据集/data/train.parquet"

# Store 166 targets (from previous validation)
STORE_166_TARGETS = {
    'store_id': 166,
    'target_products': [258, 311, 313, 432, 433],
    'first_category': 15,
    'second_category': 20,
}

# Observation window
OBS_START = pd.Timestamp("2024-11-17")
OBS_END = pd.Timestamp("2024-12-16")

# Target window
TARGET_TRAIN_START = pd.Timestamp("2024-12-16")

def load_data():
    """Load D4 data."""
    print(f"Loading: {DATA_PATH}\n")
    df = pd.read_parquet(DATA_PATH)
    df['dt'] = pd.to_datetime(df['dt'])
    print(f"✓ Loaded {len(df):,} rows")
    print(f"  Date range: {df['dt'].min().date()} to {df['dt'].max().date()}\n")
    return df

def analyze_observation_window():
    """Analyze the observation window definition."""
    print("="*100)
    print("OBSERVATION WINDOW ANALYSIS")
    print("="*100)
    
    # Create date range
    date_range = pd.date_range(OBS_START, OBS_END, freq='D')
    
    print(f"\nObservation window definition:")
    print(f"  Start: {OBS_START.date()}")
    print(f"  End: {OBS_END.date()}")
    print(f"  Expected days: 30")
    print(f"  Actual days in range: {len(date_range)}")
    
    print(f"\nAll dates in observation window:")
    for i, date in enumerate(date_range, 1):
        print(f"  {i:2d}. {date.date()}")
    
    print(f"\nBoundary check:")
    print(f"  Is 2024-11-17 included? {pd.Timestamp('2024-11-17') in date_range}")
    print(f"  Is 2024-12-16 included? {pd.Timestamp('2024-12-16') in date_range}")
    print(f"  Total days: {len(date_range)}")
    
    return date_range

def check_store_166_targets(df, required_dates):
    """Check store 166 target products."""
    print("\n" + "="*100)
    print("STORE 166 - TARGET PRODUCTS VERIFICATION")
    print("="*100)
    
    store_id = STORE_166_TARGETS['store_id']
    target_products = STORE_166_TARGETS['target_products']
    
    print(f"\nExpected targets: {target_products}")
    print(f"Expected category: first_category={STORE_166_TARGETS['first_category']}, "
          f"second_category={STORE_166_TARGETS['second_category']}")
    
    print(f"\nChecking each target product:")
    print("-"*100)
    
    for product_id in target_products:
        product_data = df[
            (df['store_id'] == store_id) &
            (df['product_id'] == product_id)
        ]
        
        if product_data.empty:
            print(f"\n❌ Product {product_id}: NOT FOUND in dataset")
            continue
        
        # Get category info
        first_cat = product_data['first_category_id'].iloc[0]
        second_cat = product_data['second_category_id'].iloc[0]
        
        # Get date range
        dates = pd.to_datetime(product_data['dt'].unique())
        date_start = dates.min()
        date_end = dates.max()
        total_span = (date_end - date_start).days + 1
        unique_days = len(dates)
        
        # Check observation window
        obs_dates = dates[(dates >= OBS_START) & (dates <= OBS_END)]
        obs_coverage = len(obs_dates)
        
        # Check which dates are covered
        covered_obs_dates = set(obs_dates.normalize())
        required_obs_dates = set(required_dates.normalize())
        missing_dates = required_obs_dates - covered_obs_dates
        
        is_complete = len(missing_dates) == 0
        
        print(f"\n✅ Product {product_id}:")
        print(f"   Category: first={first_cat}, second={second_cat}")
        print(f"   Full date range: {date_start.date()} to {date_end.date()}")
        print(f"   Unique days: {unique_days}/{total_span}")
        print(f"   Observation window coverage: {obs_coverage}/{len(required_dates)} days")
        
        if is_complete:
            print(f"   30-day completeness: ✅ YES (all {len(required_dates)} days present)")
        else:
            print(f"   30-day completeness: ❌ NO (missing {len(missing_dates)} days)")
            if len(missing_dates) <= 5:
                print(f"   Missing dates: {sorted([d.date() for d in missing_dates])}")

def check_store_166_candidates(df, required_dates):
    """Check source candidates for store 166."""
    print("\n" + "="*100)
    print("STORE 166 - SOURCE CANDIDATES (second_category=20)")
    print("="*100)
    
    store_id = STORE_166_TARGETS['store_id']
    target_products = STORE_166_TARGETS['target_products']
    second_category = STORE_166_TARGETS['second_category']
    
    # Get source candidates
    source_data = df[
        (df['store_id'] == store_id) &
        (df['second_category_id'] == second_category) &
        (~df['product_id'].isin(target_products)) &
        (df['dt'] < TARGET_TRAIN_START)
    ]
    
    candidate_products = source_data['product_id'].unique()
    
    print(f"\nFilters:")
    print(f"  Store: {store_id}")
    print(f"  second_category: {second_category}")
    print(f"  Exclude target products: {target_products}")
    print(f"  Date before: {TARGET_TRAIN_START.date()}")
    
    print(f"\nTotal candidate products: {len(candidate_products)}")
    
    if len(candidate_products) == 0:
        print("❌ No candidates found!")
        return
    
    print(f"\nDetailed candidate analysis:")
    print("-"*100)
    
    valid_candidates = []
    
    for product_id in candidate_products:
        product_data = source_data[source_data['product_id'] == product_id]
        product_dates = pd.to_datetime(product_data['dt'].unique())
        
        date_start = product_dates.min()
        date_end = product_dates.max()
        unique_days = len(product_dates)
        
        # Check observation window coverage
        obs_dates = product_dates[(product_dates >= OBS_START) & (product_dates <= OBS_END)]
        obs_coverage = len(obs_dates)
        
        # Check completeness
        covered_obs_dates = set(obs_dates.normalize())
        required_obs_dates = set(required_dates.normalize())
        missing_dates = required_obs_dates - covered_obs_dates
        
        is_complete = len(missing_dates) == 0
        
        if is_complete:
            valid_candidates.append(product_id)
        
        status = "✅ YES" if is_complete else f"❌ NO ({obs_coverage}/{len(required_dates)})"
        
        print(f"\nProduct {int(product_id)}:")
        print(f"  Date range: {date_start.date()} to {date_end.date()} ({unique_days} days)")
        print(f"  Observation coverage: {obs_coverage}/{len(required_dates)} days")
        print(f"  30-day complete: {status}")
        
        if not is_complete and len(missing_dates) <= 5:
            print(f"  Missing dates: {sorted([d.date() for d in missing_dates])}")
    
    print("\n" + "="*100)
    print(f"SUMMARY:")
    print(f"  Total candidates: {len(candidate_products)}")
    print(f"  With 30-day completeness: {len(valid_candidates)}")
    print(f"  K≥3 feasible: {'✅ YES' if len(valid_candidates) >= 3 else '❌ NO'}")
    
    if len(valid_candidates) > 0:
        print(f"  Valid candidate IDs: {sorted([int(x) for x in valid_candidates])}")

def check_global_date_coverage(df):
    """Check if there's any global data gap in the observation window."""
    print("\n" + "="*100)
    print("GLOBAL DATA COVERAGE CHECK")
    print("="*100)
    
    obs_data = df[(df['dt'] >= OBS_START) & (df['dt'] <= OBS_END)]
    
    print(f"\nObservation window: {OBS_START.date()} to {OBS_END.date()}")
    print(f"Total rows in this window: {len(obs_data):,}")
    
    # Check dates present
    unique_dates = obs_data['dt'].unique()
    unique_dates_sorted = sorted(pd.to_datetime(unique_dates))
    
    print(f"\nUnique dates with data: {len(unique_dates_sorted)}/30")
    
    # Check for missing dates
    expected_dates = pd.date_range(OBS_START, OBS_END, freq='D')
    dates_with_data = set(pd.to_datetime(unique_dates_sorted).normalize())
    expected_dates_set = set(expected_dates.normalize())
    
    missing_dates = expected_dates_set - dates_with_data
    
    if len(missing_dates) == 0:
        print("✅ All 30 days have data globally")
    else:
        print(f"❌ {len(missing_dates)} days are completely missing from dataset:")
        for date in sorted(missing_dates):
            print(f"  - {date.date()}")
    
    # Show date coverage
    print(f"\nDates with data:")
    for i, date in enumerate(unique_dates_sorted, 1):
        row_count = len(obs_data[obs_data['dt'] == date])
        print(f"  {i:2d}. {date.date()}: {row_count:,} rows")

def main():
    """Main analysis."""
    print("="*100)
    print("D4 RAW DATA INSPECTION - FIXED VERSION")
    print("="*100)
    print()
    
    df = load_data()
    
    # Analyze observation window
    required_dates = analyze_observation_window()
    
    # Check global coverage
    check_global_date_coverage(df)
    
    # Check store 166 targets
    check_store_166_targets(df, required_dates)
    
    # Check store 166 candidates
    check_store_166_candidates(df, required_dates)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
