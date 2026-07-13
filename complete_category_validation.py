#!/usr/bin/env python3
"""
Complete D4 Category Validation: WITH + WITHOUT scenarios
Validates candidate pool size and 30-day observation completeness
for both second_category and first_category groupings
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

print("=" * 120)
print("D4 COMPLETE CATEGORY VALIDATION: WITH + WITHOUT SCENARIOS")
print("=" * 120)

# Load data
print("\nLoading parquet files...")
source_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-source.parquet')
target_df = pd.read_parquet(PROJECT_ROOT / '数据集/固化数据/dataset4-target.parquet')

# Get targets
targets = target_df[['store_id', 'product_id', 'first_category_id', 'second_category_id']].drop_duplicates()
target_store = targets.iloc[0]['store_id']
target_first_cat = targets.iloc[0]['first_category_id']
target_second_cat = targets.iloc[0]['second_category_id']

print(f"Targets: {len(targets)}")
print(f"Target store: {target_store}")
print(f"Target first_category: {target_first_cat}")
print(f"Target second_category: {target_second_cat}")

# D4 observation window: 30 days before train_start (2024-12-16)
obs_start = pd.Timestamp("2024-11-17")
obs_end = pd.Timestamp("2024-12-16")
required_dates = pd.date_range(obs_start, obs_end, freq='D')

print(f"\nObservation Window: {obs_start.date()} to {obs_end.date()} ({len(required_dates)} days)")

# Convert date
source_df['date'] = pd.to_datetime(source_df['date'])

def check_30day_completeness(product_df, required_dates):
    """Check if a product has complete 30-day observation"""
    product_dates = pd.to_datetime(product_df['date'].unique()).normalize()
    required_set = set(required_dates.normalize())
    product_set = set(product_dates)
    
    missing = required_set - product_set
    return len(missing) == 0

def analyze_scenario(scenario_name, candidate_filter, category_level, category_value):
    """Analyze candidates for a specific scenario and category level"""
    print(f"\n{'='*120}")
    print(f"{scenario_name} | {category_level}={category_value}")
    print(f"{'='*120}")
    
    results = []
    
    for _, target in targets.iterrows():
        product_id = target['product_id']
        
        # Get candidates
        if category_level == 'second_category_id':
            candidates = source_df[
                candidate_filter &
                (source_df['second_category_id'] == category_value) &
                (source_df['product_id'] != product_id)
            ]
        else:  # first_category_id
            candidates = source_df[
                candidate_filter &
                (source_df['first_category_id'] == category_value) &
                (source_df['product_id'] != product_id)
            ]
        
        candidate_products = candidates['product_id'].unique()
        total_candidates = len(candidate_products)
        
        # Check 30-day completeness
        valid_candidates = []
        for cand_pid in candidate_products:
            cand_df = candidates[candidates['product_id'] == cand_pid]
            if check_30day_completeness(cand_df, required_dates):
                valid_candidates.append(cand_pid)
        
        valid_count = len(valid_candidates)
        
        print(f"\n  Target product_id={product_id}:")
        print(f"    Total candidates: {total_candidates}")
        print(f"    Valid candidates (30-day complete): {valid_count}")
        print(f"    Feasible (K≥3): {'✅ YES' if valid_count >= 3 else '❌ NO'}")
        
        if valid_count > 0 and valid_count <= 10:
            print(f"    Valid product IDs: {sorted(valid_candidates)}")
        
        results.append({
            'product_id': product_id,
            'total': total_candidates,
            'valid': valid_count,
            'feasible': valid_count >= 3,
            'valid_products': sorted(valid_candidates)
        })
    
    return results

# ============================================================================
# SCENARIO 1: WITHOUT (Same-Store) with second_category
# ============================================================================

without_second_filter = (source_df['store_id'] == target_store)
results_without_second = analyze_scenario(
    "WITHOUT (Same-Store)",
    without_second_filter,
    'second_category_id',
    target_second_cat
)

# ============================================================================
# SCENARIO 2: WITHOUT (Same-Store) with first_category
# ============================================================================

without_first_filter = (source_df['store_id'] == target_store)
results_without_first = analyze_scenario(
    "WITHOUT (Same-Store)",
    without_first_filter,
    'first_category_id',
    target_first_cat
)

# ============================================================================
# SCENARIO 3: WITH (Cross-Store) with second_category
# ============================================================================

with_second_filter = (source_df['store_id'] != target_store)
results_with_second = analyze_scenario(
    "WITH (Cross-Store)",
    with_second_filter,
    'second_category_id',
    target_second_cat
)

# ============================================================================
# SCENARIO 4: WITH (Cross-Store) with first_category
# ============================================================================

with_first_filter = (source_df['store_id'] != target_store)
results_with_first = analyze_scenario(
    "WITH (Cross-Store)",
    with_first_filter,
    'first_category_id',
    target_first_cat
)

# ============================================================================
# SUMMARY COMPARISON
# ============================================================================

print("\n" + "=" * 120)
print("SUMMARY COMPARISON")
print("=" * 120)

summary_data = []
for i, target in enumerate(targets.iterrows()):
    _, t = target
    summary_data.append({
        'product_id': t['product_id'],
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

# ============================================================================
# FEASIBILITY ANALYSIS
# ============================================================================

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

print("\nAggregate feasibility (targets with K≥3):")
print(f"  WITHOUT + second_category: {feasibility['without_2nd'].sum()}/{len(targets)} targets")
print(f"  WITHOUT + first_category:  {feasibility['without_1st'].sum()}/{len(targets)} targets")
print(f"  WITH + second_category:    {feasibility['with_2nd'].sum()}/{len(targets)} targets")
print(f"  WITH + first_category:     {feasibility['with_1st'].sum()}/{len(targets)} targets")

# ============================================================================
# IMPACT ANALYSIS
# ============================================================================

print("\n" + "=" * 120)
print("IMPACT ANALYSIS: Switching from second_category to first_category")
print("=" * 120)

without_fixed = (feasibility['without_2nd'] == False) & (feasibility['without_1st'] == True)
with_affected = feasibility['with_2nd'] != feasibility['with_1st']

print(f"\n1. WITHOUT SCENARIO:")
print(f"   Targets fixed by switching: {without_fixed.sum()}/{len(targets)}")
if without_fixed.sum() > 0:
    fixed_ids = feasibility[without_fixed]['product_id'].tolist()
    print(f"   Fixed target IDs: {fixed_ids}")
    for pid in fixed_ids:
        row = df_summary[df_summary['product_id'] == pid].iloc[0]
        print(f"     product_id={pid}:")
        print(f"       second_category: {row['without_2nd_valid']} valid (FAIL)")
        print(f"       first_category:  {row['without_1st_valid']} valid (PASS)")

print(f"\n2. WITH SCENARIO:")
print(f"   Already feasible with second_category: {feasibility['with_2nd'].sum()}/{len(targets)}")
print(f"   Targets affected by switching: {with_affected.sum()}/{len(targets)}")

# Average pool size
print(f"\n3. CANDIDATE POOL SIZE CHANGES:")
print(f"   WITHOUT scenario:")
print(f"     second_category avg: {df_summary['without_2nd_valid'].mean():.1f} valid candidates")
print(f"     first_category avg:  {df_summary['without_1st_valid'].mean():.1f} valid candidates")
print(f"     Change: +{df_summary['without_1st_valid'].mean() - df_summary['without_2nd_valid'].mean():.1f}")

print(f"\n   WITH scenario:")
print(f"     second_category avg: {df_summary['with_2nd_valid'].mean():.1f} valid candidates")
print(f"     first_category avg:  {df_summary['with_1st_valid'].mean():.1f} valid candidates")
print(f"     Change: {df_summary['with_1st_valid'].mean() - df_summary['with_2nd_valid'].mean():+.1f}")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 120)
print("RECOMMENDATIONS")
print("=" * 120)

all_without_feasible_second = feasibility['without_2nd'].all()
all_without_feasible_first = feasibility['without_1st'].all()
all_with_feasible_second = feasibility['with_2nd'].all()

if all_without_feasible_first and not all_without_feasible_second:
    print("\n✅ RECOMMENDATION: Switch to first_category")
    print("\n   RATIONALE:")
    print("   1. WITHOUT becomes feasible (all targets have K≥3)")
    print("   2. WITH remains feasible")
    print("   3. Consistent category definition across scenarios")
    
    print("\n   REQUIRED ACTIONS:")
    print("   1. Update domain_filter in configs/solidified/knn/Dataset4/knn_*.json:")
    print('      "domain_filter": {"column": "first_category_id", "value": 15}')
    print("   2. Document in paper methodology:")
    print("      - Reason: Candidate sparsity at second_category level")
    print("      - Limitation: Category IDs are anonymized, semantic coherence unverified")
    print("      - Impact: Broader grouping may introduce source heterogeneity")
    
elif all_without_feasible_second:
    print("\n✅ RECOMMENDATION: Keep second_category (no change needed)")
    print("\n   RATIONALE:")
    print("   1. Both scenarios already feasible with second_category")
    print("   2. Finer granularity preserves semantic coherence")
    print("   3. No need to relax protocol")
    
else:
    print("\n⚠️  MIXED RESULTS: Some targets remain infeasible")
    print("\n   ALTERNATIVES:")
    print("   1. Select different target store/category combination")
    print("   2. Reduce K to 2 (document as protocol deviation)")
    print("   3. Skip infeasible targets from D4-without")
    print("   4. Run only D4-with (cross-store)")

# Check consistency
if target_first_cat == target_second_cat:
    print("\n⚠️  WARNING: first_category_id == second_category_id")
    print("   This may indicate category hierarchy collapse or data issue")

print("\n" + "=" * 120)
print("VALIDATION COMPLETE")
print("=" * 120)
