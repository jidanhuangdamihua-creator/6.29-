#!/usr/bin/env python3
"""
D4 Raw Data Inspection - CORRECTED Observation Window
======================================================

FIX: Use correct observation window according to protocol:
- knn_observed_end = target_observed_start + 29 days
- Observation window: 2024-11-16 to 2024-12-15 (30 days)
- Source cutoff: <= 2024-12-15 (inclusive)
"""

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "数据集/原始数据/Dataset 4叮咚数据集/data/train.parquet"

# Store 166 targets
STORE_166_TARGETS = {
    'store_id': 166,
    'target_products': [258, 311, 313, 432, 433],
    'first_category': 15,
    'second_category': 20,
}

# CORRECTED observation window (according to protocol)
TARGET_TRAIN_START = pd.Timestamp("2024-12-16")
KNN_OBSERVED_END = TARGET_TRAIN_START - pd.Timedelta(days=1)  # 2024-12-15
TARGET_OBSERVED_START = KNN_OBSERVED_END - pd.Timedelta(days=29)  # 2024-11-16

OBS_START = TARGET_OBSERVED_START
OBS_END = KNN_OBSERVED_END
SOURCE_OBSERVATION_CUTOFF = KNN_OBSERVED_END

def load_data():
    """Load D4 data."""
    print(f"Loading: {DATA_PATH}\n")
    df = pd.read_parquet(DATA_PATH)
    df['dt'] = pd.to_datetime(df['dt'])
    print(f"✓ Loaded {len(df):,} rows\n")
    return df

def main():
    """Main analysis."""
    print("="*100)
    print("D4 OBSERVATION WINDOW - CORRECTED")
    print("="*100)
    
    print(f"\nProtocol definition:")
    print(f"  target_train_start: {TARGET_TRAIN_START.date()}")
    print(f"  knn_observed_end = target_train_start - 1 day = {KNN_OBSERVED_END.date()}")
    print(f"  target_observed_start = knn_observed_end - 29 days = {TARGET_OBSERVED_START.date()}")
    print(f"  source_observation_cutoff = knn_observed_end = {SOURCE_OBSERVATION_CUTOFF.date()}")
    
    # Create observation window
    obs_dates = pd.date_range(OBS_START, OBS_END, freq='D')
    
    print(f"\nObservation window:")
    print(f"  Start: {OBS_START.date()}")
    print(f"  End: {OBS_END.date()}")
    print(f"  Total days: {len(obs_dates)}")
    
    print(f"\nKey difference from previous incorrect version:")
    print(f"  ❌ Previous: 2024-11-17 to 2024-12-16, source < 2024-12-16")
    print(f"  ✅ Correct:  2024-11-16 to 2024-12-15, source <= 2024-12-15")
    print(f"  Impact: Source CAN have data on observation end date now!")
    
    # Load data
    df = load_data()
    
    # Check store 166 candidates
    print("\n" + "="*100)
    print("STORE 166 - SOURCE CANDIDATES (CORRECTED)")
    print("="*100)
    
    store_id = STORE_166_TARGETS['store_id']
    target_products = STORE_166_TARGETS['target_products']
    second_category = STORE_166_TARGETS['second_category']
    
    # CORRECTED filter: source <= SOURCE_OBSERVATION_CUTOFF (not < TARGET_TRAIN_START)
    source_data = df[
        (df['store_id'] == store_id) &
        (df['second_category_id'] == second_category) &
        (~df['product_id'].isin(target_products)) &
        (df['dt'] <= SOURCE_OBSERVATION_CUTOFF)  # CORRECTED: <= not <
    ]
    
    candidate_products = source_data['product_id'].unique()
    
    print(f"\nFilters:")
    print(f"  Store: {store_id}")
    print(f"  second_category: {second_category}")
    print(f"  Exclude targets: {target_products}")
    print(f"  Date <= {SOURCE_OBSERVATION_CUTOFF.date()} (CORRECTED)")
    
    print(f"\nTotal candidate products: {len(candidate_products)}")
    
    if len(candidate_products) == 0:
        print("❌ No candidates found!")
        return
    
    # Check 30-day completeness
    required_obs_dates = set(obs_dates.normalize())
    
    print(f"\nChecking 30-day completeness (need all {len(obs_dates)} days):")
    print("-"*100)
    print(f"{'Product ID':<15} {'Date Range':<40} {'Obs Coverage':<20} {'30-day Complete':<20}")
    print("-"*100)
    
    valid_candidates = []
    
    for product_id in sorted(candidate_products):
        product_data = source_data[source_data['product_id'] == product_id]
        product_dates = pd.to_datetime(product_data['dt'].unique())
        
        date_start = product_dates.min()
        date_end = product_dates.max()
        
        # Check observation window coverage
        obs_dates_product = product_dates[(product_dates >= OBS_START) & (product_dates <= OBS_END)]
        covered_obs_dates = set(obs_dates_product.normalize())
        missing_dates = required_obs_dates - covered_obs_dates
        
        is_complete = len(missing_dates) == 0
        
        if is_complete:
            valid_candidates.append(product_id)
        
        coverage_str = f"{len(covered_obs_dates)}/{len(obs_dates)}"
        status = "✅ YES" if is_complete else f"❌ NO"
        
        print(f"{int(product_id):<15} "
              f"{date_start.date()} to {date_end.date():<23} "
              f"{coverage_str:<20} "
              f"{status:<20}")
        
        if not is_complete and len(missing_dates) <= 3:
            print(f"{'':>15} Missing: {sorted([d.date() for d in missing_dates])}")
    
    print("-"*100)
    print(f"\n{'SUMMARY':}")
    print(f"  Total candidates: {len(candidate_products)}")
    print(f"  With 30-day completeness: {len(valid_candidates)}")
    print(f"  K≥3 feasible: {'✅ YES' if len(valid_candidates) >= 3 else '❌ NO'}")
    
    if len(valid_candidates) > 0:
        print(f"  Valid candidate IDs: {sorted([int(x) for x in valid_candidates])}")
    
    print("\n" + "="*100)
    print("COMPARISON WITH PREVIOUS INCORRECT VERSION")
    print("="*100)
    print(f"\n❌ Previous (incorrect) observation window: 2024-11-17 to 2024-12-16")
    print(f"   Source filter: < 2024-12-16")
    print(f"   Result: ALL candidates missing 2024-12-16, 0 valid")
    
    print(f"\n✅ Corrected observation window: 2024-11-16 to 2024-12-15")
    print(f"   Source filter: <= 2024-12-15")
    print(f"   Result: {len(valid_candidates)} valid candidates")
    
    if len(valid_candidates) >= 3:
        print(f"\n✅✅✅ K≥3 IS FEASIBLE with correct observation window!")
    else:
        print(f"\n⚠️  K≥3 still not feasible even with corrected window")
        print(f"   Need to investigate other category groupings or window lengths")
    
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
