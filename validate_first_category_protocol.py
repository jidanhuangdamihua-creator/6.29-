#!/usr/bin/env python3
"""
Complete Protocol Validation: first_category vs second_category
Validates both candidate count AND 30-day observation window completeness
"""

import pandas as pd
from pathlib import Path
from datetime import timedelta

PROJECT_ROOT = Path(__file__).parent

print("=" * 100)
print("D4 CATEGORY PROTOCOL VALIDATION (WITH 30-DAY OBSERVATION WINDOW CHECK)")
print("=" * 100)

# Load data
source_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-source.parquet')
target_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-target.parquet')

# Get targets
targets = target_df[['store_id', 'product_id', 'first_category_id', 'second_category_id']].drop_duplicates()

# D4 observation window: 30 days before train_start (2024-12-16)
# Observation window: 2024-11-17 to 2024-12-16
obs_start = pd.Timestamp("2024-11-17")
obs_end = pd.Timestamp("2024-12-16")
required_dates = pd.date_range(obs_start, obs_end, freq='D')

print(f"\nObservation Window: {obs_start.date()} to {obs_end.date()} ({len(required_dates)} days)")
print(f"Targets: {len(targets)}")
print(f"Store: {targets.iloc[0]['store_id']}")

# Convert date column
source_df['date'] = pd.to_datetime(source_df['date'])

def check_30day_completeness(product_df, required_dates):
    """Check if a product has complete 30-day observation"""
    product_dates = product_df['date'].unique()
    product_dates_set = set(pd.to_datetime(product_dates).normalize())
    required_set = set(required_dates.normalize())
    
    missing = required_set - product_dates_set
    has_complete = len(missing) == 0
    
    return {
        'has_complete_30day': has_complete,
        'observed_days': len(product_dates_set & required_set),
        'missing_days': len(missing),
        'missing_dates': sorted([d.strftime('%Y-%m-%d') for d in missing]) if missing else []
    }

print("\n" + "=" * 100)
print("VALIDATION RESULTS")
print("=" * 100)

results = []

for _, target in targets.iterrows():
    store_id = target['store_id']
    product_id = target['product_id']
    first_cat = target['first_category_id']
    second_cat = target['second_category_id']
    
    print(f"\n{'='*100}")
    print(f"TARGET: product_id={product_id}")
    print(f"  first_category={first_cat}, second_category={second_cat}")
    print(f"{'='*100}")
    
    # --- Scenario 1: second_category ---
    candidates_2nd = source_df[
        (source_df['store_id'] == store_id) &
        (source_df['second_category_id'] == second_cat) &
        (source_df['product_id'] != product_id)
    ]
    
    candidate_products_2nd = candidates_2nd['product_id'].unique()
    print(f"\n[SECOND_CATEGORY={second_cat}]")
    print(f"  Total candidate products: {len(candidate_products_2nd)}")
    
    valid_candidates_2nd = []
    for cand_pid in candidate_products_2nd:
        cand_df = candidates_2nd[candidates_2nd['product_id'] == cand_pid]
        check = check_30day_completeness(cand_df, required_dates)
        
        if check['has_complete_30day']:
            valid_candidates_2nd.append(cand_pid)
            print(f"    ✅ product_id={cand_pid}: Complete 30 days")
        else:
            print(f"    ❌ product_id={cand_pid}: Missing {check['missing_days']} days")
    
    print(f"  → Valid candidates (K): {len(valid_candidates_2nd)}")
    
    # --- Scenario 2: first_category ---
    candidates_1st = source_df[
        (source_df['store_id'] == store_id) &
        (source_df['first_category_id'] == first_cat) &
        (source_df['product_id'] != product_id)
    ]
    
    candidate_products_1st = candidates_1st['product_id'].unique()
    print(f"\n[FIRST_CATEGORY={first_cat}]")
    print(f"  Total candidate products: {len(candidate_products_1st)}")
    
    valid_candidates_1st = []
    for cand_pid in candidate_products_1st:
        cand_df = candidates_1st[candidates_1st['product_id'] == cand_pid]
        check = check_30day_completeness(cand_df, required_dates)
        
        if check['has_complete_30day']:
            valid_candidates_1st.append(cand_pid)
            print(f"    ✅ product_id={cand_pid}: Complete 30 days")
        else:
            print(f"    ❌ product_id={cand_pid}: Missing {check['missing_days']} days")
    
    print(f"  → Valid candidates (K): {len(valid_candidates_1st)}")
    
    # Summary
    results.append({
        'product_id': product_id,
        'first_cat': first_cat,
        'second_cat': second_cat,
        'total_2nd': len(candidate_products_2nd),
        'valid_2nd': len(valid_candidates_2nd),
        'total_1st': len(candidate_products_1st),
        'valid_1st': len(valid_candidates_1st),
        'feasible_2nd': len(valid_candidates_2nd) >= 3,
        'feasible_1st': len(valid_candidates_1st) >= 3,
        'fixed_by_1st': len(valid_candidates_2nd) < 3 and len(valid_candidates_1st) >= 3,
        'valid_products_2nd': sorted(valid_candidates_2nd),
        'valid_products_1st': sorted(valid_candidates_1st),
    })

print("\n" + "=" * 100)
print("SUMMARY TABLE")
print("=" * 100)

df_results = pd.DataFrame(results)
print("\nCandidate Counts (Total / Valid):")
print(df_results[['product_id', 'second_cat', 'first_cat', 
                   'total_2nd', 'valid_2nd', 'total_1st', 'valid_1st']].to_string(index=False))

print("\nFeasibility (K≥3):")
print(df_results[['product_id', 'feasible_2nd', 'feasible_1st', 'fixed_by_1st']].to_string(index=False))

print("\n" + "=" * 100)
print("CRITICAL FINDINGS")
print("=" * 100)

fixed_count = df_results['fixed_by_1st'].sum()
print(f"\n1. TARGETS FIXED BY SWITCHING TO first_category:")
print(f"   {fixed_count} out of {len(targets)} targets")

if fixed_count > 0:
    fixed_targets = df_results[df_results['fixed_by_1st']]
    for _, row in fixed_targets.iterrows():
        print(f"\n   Target {row['product_id']}:")
        print(f"     second_category={row['second_cat']}: {row['valid_2nd']} valid candidates {row['valid_products_2nd']}")
        print(f"     first_category={row['first_cat']}: {row['valid_1st']} valid candidates {row['valid_products_1st']}")
        new_candidates = set(row['valid_products_1st']) - set(row['valid_products_2nd'])
        print(f"     ✅ NEW valid candidates: {sorted(new_candidates)}")
else:
    print("   ❌ NO TARGETS FIXED - first_category doesn't help!")

print("\n2. SCENARIO FEASIBILITY:")
print(f"   second_category: {df_results['feasible_2nd'].sum()}/{len(targets)} targets feasible for WITHOUT mode")
print(f"   first_category:  {df_results['feasible_1st'].sum()}/{len(targets)} targets feasible for WITHOUT mode")

# Check if all targets share same first_category
unique_first = df_results['first_cat'].nunique()
unique_second = df_results['second_cat'].nunique()
print(f"\n3. CATEGORY CONSISTENCY:")
print(f"   Unique first_category in targets: {unique_first}")
print(f"   Unique second_category in targets: {unique_second}")

if unique_first == 1 and unique_second == 1:
    print("   ✅ All targets share same category hierarchy")
    print("   → Changing protocol affects ALL targets uniformly (no inconsistency)")
elif unique_first == 1:
    print("   ⚠️  All targets share same first_category but different second_category")
    print("   → Switching may merge previously distinct domains")
else:
    print("   ⚠️  Targets span multiple first_category values")
    print("   → Impact varies by target")

print("\n" + "=" * 100)
print("NEXT STEPS CHECKLIST")
print("=" * 100)

if fixed_count > 0:
    print("\n✅ STEP 1: Valid candidates confirmed")
    print(f"   → {fixed_count} target(s) will become feasible with first_category")
    print("\n⏭️  STEP 2: Check business semantics")
    print("   → Run the category_semantics_check.py script")
    print("\n⏭️  STEP 3: Decide WITH-sharing policy")
    print("   → Should with-sharing also use first_category for consistency?")
    print("\n⏭️  STEP 4: Update protocol & regenerate")
    print("   → Modify domain_filter in KNN config")
    print("   → Regenerate solidified parquet")
else:
    print("\n❌ STEP 1 FAILED: first_category doesn't fix the K<3 issue")
    print("   → Valid candidates still insufficient after 30-day check")
    print("   → Need alternative solutions:")
    print("     1. Select different target store/category")
    print("     2. Reduce K to 2 (document as protocol deviation)")
    print("     3. Skip D4-without entirely")

print("\n" + "=" * 100)
print("VALIDATION COMPLETE")
print("=" * 100)
