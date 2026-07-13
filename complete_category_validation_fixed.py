#!/usr/bin/env python3
"""
FIXED: Complete D4 Category Validation
Correctly uses (store_id, product_id) as entity key
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("=" * 120)
print("D4 COMPLETE CATEGORY VALIDATION (FIXED - Using store_id + product_id as entity)")
print("=" * 120)

# Load data
print("\nLoading parquet files...")
source_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-source.parquet')
target_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-target.parquet')

# Get targets as (store, product) pairs
target_entities = target_df[['store_id', 'product_id', 'first_category_id', 'second_category_id']].drop_duplicates()
target_store = target_entities.iloc[0]['store_id']
target_first_cat = target_entities.iloc[0]['first_category_id']
target_second_cat = target_entities.iloc[0]['second_category_id']

print(f"Targets: {len(target_entities)}")
print(f"Target store: {target_store}")
print(f"Target first_category: {target_first_cat}")
print(f"Target second_category: {target_second_cat}")

# D4 observation window
obs_start = pd.Timestamp("2024-11-17")
obs_end = pd.Timestamp("2024-12-16")
required_dates = pd.date_range(obs_start, obs_end, freq='D')

print(f"\nObservation Window: {obs_start.date()} to {obs_end.date()} ({len(required_dates)} days)")

source_df['date'] = pd.to_datetime(source_df['date'])

# Get target entity set for exclusion
target_entity_set = set(
    target_entities[['store_id', 'product_id']].itertuples(index=False, name=None)
)

def check_30day_completeness(entity_df, required_dates):
    """Check if an entity has complete 30-day observation"""
    entity_dates = pd.to_datetime(entity_df['date'].unique()).normalize()
    required_set = set(required_dates.normalize())
    entity_set = set(entity_dates)
    missing = required_set - entity_set
    return len(missing) == 0

def analyze_scenario(scenario_name, store_filter_desc, store_filter, category_level, category_value):
    """Analyze candidates for a specific scenario"""
    print(f"\n{'='*120}")
    print(f"{scenario_name} ({store_filter_desc}) | {category_level}={category_value}")
    print(f"{'='*120}")
    
    results = []
    
    for _, target in target_entities.iterrows():
        target_store_id = target['store_id']
        target_product_id = target['product_id']
        
        # Get candidate pool
        if category_level == 'second_category_id':
            candidates = source_df[
                store_filter &
                (source_df['second_category_id'] == category_value)
            ]
        else:  # first_category_id
            candidates = source_df[
                store_filter &
                (source_df['first_category_id'] == category_value)
            ]
        
        # Get unique entities
        candidate_entities = candidates[['store_id', 'product_id']].drop_duplicates()
        candidate_entity_set = set(candidate_entities.itertuples(index=False, name=None))
        
        # Exclude ALL target entities (not just current one)
        candidate_entity_set = candidate_entity_set - target_entity_set
        
        total_candidates = len(candidate_entity_set)
        
        # Check 30-day completeness
        valid_entities = []
        for store_id, product_id in candidate_entity_set:
            entity_df = candidates[
                (candidates['store_id'] == store_id) &
                (candidates['product_id'] == product_id)
            ]
            if check_30day_completeness(entity_df, required_dates):
                valid_entities.append((store_id, product_id))
        
        valid_count = len(valid_entities)
        
        print(f"\n  Target: store={target_store_id}, product={target_product_id}")
        print(f"    Total candidates: {total_candidates}")
        print(f"    Valid candidates (30-day): {valid_count}")
        print(f"    Feasible (K≥3): {'✅ YES' if valid_count >= 3 else '❌ NO'}")
        
        results.append({
            'store_id': target_store_id,
            'product_id': target_product_id,
            'total': total_candidates,
            'valid': valid_count,
            'feasible': valid_count >= 3,
        })
    
    return results

# ============================================================================
# WITHOUT (Same-Store) Scenarios
# ============================================================================

without_filter = (source_df['store_id'] == target_store)

results_without_second = analyze_scenario(
    "WITHOUT",
    "Same-Store",
    without_filter,
    'second_category_id',
    target_second_cat
)

results_without_first = analyze_scenario(
    "WITHOUT",
    "Same-Store",
    without_filter,
    'first_category_id',
    target_first_cat
)

# ============================================================================
# WITH (Cross-Store) Scenarios  
# ============================================================================

with_filter = (source_df['store_id'] != target_store)

results_with_second = analyze_scenario(
    "WITH",
    "Cross-Store",
    with_filter,
    'second_category_id',
    target_second_cat
)

results_with_first = analyze_scenario(
    "WITH",
    "Cross-Store",
    with_filter,
    'first_category_id',
    target_first_cat
)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 120)
print("SUMMARY COMPARISON")
print("=" * 120)

summary_data = []
for i in range(len(target_entities)):
    summary_data.append({
        'product_id': results_without_second[i]['product_id'],
        'without_2nd_total': results_without_second[i]['total'],
        'without_2nd_valid': results_without_second[i]['valid'],
        'without_1st_total': results_without_first[i]['total'],
        'without_1st_valid': results_without_first[i]['valid'],
        'with_2nd_total': results_with_second[i]['total'],
        'with_2nd_valid': results_with_second[i]['valid'],
        'with_1st_total': results_with_first[i]['total'],
        'with_1st_valid': results_with_first[i]['valid'],
    })

df_summary = pd.DataFrame(summary_data)

print("\n1. WITHOUT SCENARIO (Same-Store):")
print(df_summary[['product_id', 'without_2nd_total', 'without_2nd_valid',
                   'without_1st_total', 'without_1st_valid']].to_string(index=False))

print("\n2. WITH SCENARIO (Cross-Store):")
print(df_summary[['product_id', 'with_2nd_total', 'with_2nd_valid',
                   'with_1st_total', 'with_1st_valid']].to_string(index=False))

# Feasibility
print("\n" + "=" * 120)
print("FEASIBILITY ANALYSIS (K≥3)")
print("=" * 120)

feasibility = pd.DataFrame({
    'product_id': df_summary['product_id'],
    'without_2nd': df_summary['without_2nd_valid'] >= 3,
    'without_1st': df_summary['without_1st_valid'] >= 3,
    'with_2nd': df_summary['with_2nd_valid'] >= 3,
    'with_1st': df_summary['with_1st_valid'] >= 3,
})

print("\nPer-target feasibility:")
print(feasibility.to_string(index=False))

print("\nAggregate (targets with K≥3):")
print(f"  WITHOUT + second_category: {feasibility['without_2nd'].sum()}/{len(target_entities)}")
print(f"  WITHOUT + first_category:  {feasibility['without_1st'].sum()}/{len(target_entities)}")
print(f"  WITH + second_category:    {feasibility['with_2nd'].sum()}/{len(target_entities)}")
print(f"  WITH + first_category:     {feasibility['with_1st'].sum()}/{len(target_entities)}")

print("\n" + "=" * 120)
print("VALIDATION COMPLETE (VERIFIED)")
print("=" * 120)
