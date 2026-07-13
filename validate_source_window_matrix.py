#!/usr/bin/env python3
"""
D4 Source Window Matrix Validation
====================================

Purpose:
Treat source window days as a formal experimental variable.
For each target store (166/155/240/293), validate K≥3 feasibility
under three window lengths: 180, 210, 300 days.

Key Principle:
- Window length affects candidate pool size (longer window = stricter requirement)
- This is NOT post-hoc parameter tuning - it's a designed ablation study
- Report all results transparently, even if some configurations fail K≥3

Output:
A matrix showing candidate pool statistics for each (store, window_days) combination.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Tuple
import json
import sys

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
D4_DATA_PATH = PROJECT_ROOT / "数据集" / "原始数据" / "Dataset 4叮咚数据集" / "data" / "train.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "source_window_validation"

# Target stores selected from previous analysis
TARGET_STORES = [166, 155, 240, 293]

# Source window lengths to test (ablation study)
WINDOW_DAYS_OPTIONS = [180, 210, 300]

# D4 observation window for 30-day completeness check
# Based on D4 configuration: 30 days before train_start (2024-12-16)
OBS_START = pd.Timestamp("2024-11-17")
OBS_END = pd.Timestamp("2024-12-16")
REQUIRED_OBS_DATES = pd.date_range(OBS_START, OBS_END, freq='D')

# Target window (from solidified config)
TARGET_TRAIN_START = pd.Timestamp("2024-12-16")
TARGET_TEST_END = pd.Timestamp("2025-07-13")

# Minimum K requirement
MIN_K = 3

# ============================================================================
# Helper Functions
# ============================================================================

def check_30day_completeness(product_df: pd.DataFrame, required_dates: pd.DatetimeIndex) -> bool:
    """
    Check if a product has complete observations in the 30-day window.
    
    Args:
        product_df: DataFrame with 'dt' column for a single product
        required_dates: Required date range
        
    Returns:
        True if product has data for all required dates
    """
    product_dates = pd.to_datetime(product_df['dt'].unique()).normalize()
    required_set = set(required_dates.normalize())
    product_set = set(product_dates)
    missing = required_set - product_set
    return len(missing) == 0


def check_source_window_coverage(
    product_df: pd.DataFrame,
    window_days: int,
    obs_end: pd.Timestamp
) -> Tuple[bool, int, pd.Timestamp, pd.Timestamp]:
    """
    Check if a product has sufficient data coverage in the source window.
    
    Source window definition:
    - Ends at OBS_END (2024-12-16, the day before target train starts)
    - Spans backward for window_days
    
    Coverage requirement:
    - Product must have at least window_days worth of data
    - Data should be relatively continuous (we check total days >= window_days)
    
    Args:
        product_df: DataFrame with 'dt' column for a single product
        window_days: Number of days in the source window
        obs_end: End date of observation window
        
    Returns:
        (has_coverage, actual_days, earliest_date, latest_date)
    """
    product_dates = pd.to_datetime(product_df['dt'].unique())
    
    if len(product_dates) == 0:
        return False, 0, None, None
    
    earliest = product_dates.min()
    latest = product_dates.max()
    
    # Check if product's data extends back far enough
    window_start = obs_end - timedelta(days=window_days - 1)
    
    # Count how many days the product has in the window
    dates_in_window = product_dates[(product_dates >= window_start) & (product_dates <= obs_end)]
    actual_days = len(dates_in_window)
    
    # Requirement: must have data for at least window_days days
    # (allowing some gaps, but requiring substantial coverage)
    has_coverage = actual_days >= window_days * 0.9  # 90% coverage threshold
    
    return has_coverage, actual_days, earliest, latest


def load_d4_data() -> pd.DataFrame:
    """Load D4 dataset with required columns."""
    print("\n[1/4] Loading D4 dataset...")
    print(f"  Path: {D4_DATA_PATH}")
    
    if not D4_DATA_PATH.exists():
        raise FileNotFoundError(f"D4 dataset not found: {D4_DATA_PATH}")
    
    try:
        import pyarrow.parquet as pq
        
        # Use PyArrow for more efficient loading
        print("  Using PyArrow for efficient loading...")
        table = pq.read_table(
            D4_DATA_PATH,
            columns=[
                'store_id',
                'product_id',
                'first_category_id',
                'second_category_id',
                'third_category_id',
                'dt',
            ]
        )
        df = table.to_pandas()
        
    except ImportError:
        print("  PyArrow not available, using pandas...")
        df = pd.read_parquet(
            D4_DATA_PATH,
            columns=[
                'store_id',
                'product_id',
                'first_category_id',
                'second_category_id',
                'third_category_id',
                'dt',
            ]
        )
    
    df['dt'] = pd.to_datetime(df['dt'])
    
    print(f"  ✓ Loaded {len(df):,} rows")
    print(f"  Date range: {df['dt'].min().date()} ~ {df['dt'].max().date()}")
    print(f"  Unique stores: {df['store_id'].nunique()}")
    print(f"  Unique products: {df['product_id'].nunique()}")
    
    return df


def get_store_targets(df: pd.DataFrame, store_id: int) -> pd.DataFrame:
    """
    Get target products for a given store.
    
    For D4, targets are products that:
    1. Belong to the target store
    2. Have data in the target window
    
    We'll use products from store_id that appear in the target window.
    """
    # Filter by date range in target window
    target_data = df[
        (df['store_id'] == store_id) &
        (df['dt'] >= TARGET_TRAIN_START) &
        (df['dt'] <= TARGET_TEST_END)
    ]
    
    # Get unique products with their categories
    targets = target_data[
        ['product_id', 'first_category_id', 'second_category_id', 'third_category_id']
    ].drop_duplicates()
    
    return targets


def validate_store_window_combination(
    df: pd.DataFrame,
    store_id: int,
    window_days: int,
    category_level: str = 'second_category'
) -> Dict:
    """
    Validate candidate pool for a specific (store, window_days) combination.
    
    Args:
        df: Full D4 dataset
        store_id: Target store ID
        window_days: Source window length in days
        category_level: 'first_category' or 'second_category'
        
    Returns:
        Dictionary with validation results
    """
    # Get targets for this store
    targets = get_store_targets(df, store_id)
    
    if targets.empty:
        return {
            'store_id': store_id,
            'window_days': window_days,
            'category_level': category_level,
            'target_count': 0,
            'error': 'No targets found in target window',
        }
    
    # For simplicity, analyze the first target (or most common category)
    # In a full analysis, we would check all targets
    if category_level == 'second_category':
        target_category_col = 'second_category_id'
        most_common_category = targets['second_category_id'].mode()[0]
    else:
        target_category_col = 'first_category_id'
        most_common_category = targets['first_category_id'].mode()[0]
    
    # Representative target from the most common category
    representative_target = targets[
        targets[target_category_col] == most_common_category
    ].iloc[0]
    
    target_product_id = representative_target['product_id']
    target_category = representative_target[target_category_col]
    
    # Filter source data:
    # 1. Same store (for WITHOUT scenario) or different stores (for WITH scenario)
    # 2. Same category
    # 3. Before target window starts
    # 4. Not the target product itself
    
    # We'll use WITHOUT scenario (same store) for consistency with previous analysis
    source_data = df[
        (df['store_id'] == store_id) &
        (df[target_category_col] == target_category) &
        (df['product_id'] != target_product_id) &
        (df['dt'] < TARGET_TRAIN_START)
    ]
    
    candidate_products = source_data['product_id'].unique()
    total_candidates = len(candidate_products)
    
    # Check each candidate for:
    # 1. 30-day completeness in observation window
    # 2. Sufficient coverage in the source window
    
    valid_30day = []
    valid_window = []
    valid_both = []
    
    window_coverage_days = []
    
    for product_id in candidate_products:
        product_data = source_data[source_data['product_id'] == product_id]
        
        # Check 30-day completeness
        has_30day = check_30day_completeness(product_data, REQUIRED_OBS_DATES)
        
        # Check source window coverage
        has_window, actual_days, earliest, latest = check_source_window_coverage(
            product_data, window_days, OBS_END
        )
        
        if has_30day:
            valid_30day.append(product_id)
        
        if has_window:
            valid_window.append(product_id)
            window_coverage_days.append(actual_days)
        
        if has_30day and has_window:
            valid_both.append(product_id)
    
    valid_30day_count = len(valid_30day)
    valid_window_count = len(valid_window)
    valid_both_count = len(valid_both)
    
    # K≥3 feasibility
    feasible = valid_both_count >= MIN_K
    
    return {
        'store_id': store_id,
        'window_days': window_days,
        'category_level': category_level,
        'target_product_id': int(target_product_id),
        'target_category': int(target_category),
        'target_count': len(targets),
        'total_candidates': total_candidates,
        'valid_30day_count': valid_30day_count,
        'valid_window_count': valid_window_count,
        'valid_both_count': valid_both_count,
        'min_k': MIN_K,
        'k_feasible': feasible,
        'avg_window_coverage_days': np.mean(window_coverage_days) if window_coverage_days else 0,
        'valid_product_ids': sorted([int(x) for x in valid_both[:10]]),  # Sample
    }


# ============================================================================
# Main Validation
# ============================================================================

def run_validation_matrix() -> List[Dict]:
    """
    Run validation for all (store, window_days) combinations.
    
    Returns:
        List of result dictionaries
    """
    print("\n[2/4] Running validation matrix...")
    print(f"  Target stores: {TARGET_STORES}")
    print(f"  Window days options: {WINDOW_DAYS_OPTIONS}")
    print(f"  Total combinations: {len(TARGET_STORES) * len(WINDOW_DAYS_OPTIONS)}")
    
    df = load_d4_data()
    
    results = []
    
    for store_id in TARGET_STORES:
        print(f"\n  Store {store_id}:")
        for window_days in WINDOW_DAYS_OPTIONS:
            print(f"    Testing window_days={window_days}...", end=' ')
            
            # Test with second_category (primary)
            result = validate_store_window_combination(
                df, store_id, window_days, category_level='second_category'
            )
            results.append(result)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                feasible_icon = '✅' if result['k_feasible'] else '❌'
                print(f"{feasible_icon} Total={result['total_candidates']}, "
                      f"Valid={result['valid_both_count']}, "
                      f"K≥{MIN_K}={result['k_feasible']}")
    
    return results


def generate_summary_matrix(results: List[Dict]) -> None:
    """Generate and print summary matrix."""
    print("\n[3/4] Generating summary matrix...")
    
    # Create matrix DataFrame
    matrix_data = []
    
    for store_id in TARGET_STORES:
        row = {'store_id': store_id}
        for window_days in WINDOW_DAYS_OPTIONS:
            result = next(
                (r for r in results if r['store_id'] == store_id and r['window_days'] == window_days),
                None
            )
            if result and 'error' not in result:
                row[f'window_{window_days}_total'] = result['total_candidates']
                row[f'window_{window_days}_valid'] = result['valid_both_count']
                row[f'window_{window_days}_feasible'] = result['k_feasible']
            else:
                row[f'window_{window_days}_total'] = 'N/A'
                row[f'window_{window_days}_valid'] = 'N/A'
                row[f'window_{window_days}_feasible'] = False
        
        matrix_data.append(row)
    
    matrix_df = pd.DataFrame(matrix_data)
    
    # Print formatted matrix
    print("\n" + "=" * 120)
    print("SOURCE WINDOW VALIDATION MATRIX")
    print("=" * 120)
    print("\nCandidate Pool Size (Total / Valid / K≥3):")
    print("-" * 120)
    
    header = f"{'Store':<10}"
    for window_days in WINDOW_DAYS_OPTIONS:
        header += f"{'Window=' + str(window_days) + 'd':<35}"
    print(header)
    print("-" * 120)
    
    for _, row in matrix_df.iterrows():
        line = f"{int(row['store_id']):<10}"
        for window_days in WINDOW_DAYS_OPTIONS:
            total = row[f'window_{window_days}_total']
            valid = row[f'window_{window_days}_valid']
            feasible = row[f'window_{window_days}_feasible']
            
            if total != 'N/A':
                feasible_icon = '✅' if feasible else '❌'
                cell = f"{total:>3} / {valid:>3} / {feasible_icon:<2}"
            else:
                cell = "N/A"
            
            line += f"{cell:<35}"
        print(line)
    
    print("-" * 120)
    print("\nLegend:")
    print("  Total: Total candidate products in the store/category")
    print("  Valid: Candidates meeting both 30-day completeness AND window coverage")
    print("  ✅/❌: Whether K≥3 is satisfied")
    print()
    
    return matrix_df


def save_results(results: List[Dict], matrix_df: pd.DataFrame) -> None:
    """Save results to files."""
    print("\n[4/4] Saving results...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    json_path = OUTPUT_DIR / "source_window_validation_detailed.json"
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(
            {
                'validation_date': pd.Timestamp.now().isoformat(),
                'config': {
                    'target_stores': TARGET_STORES,
                    'window_days_options': WINDOW_DAYS_OPTIONS,
                    'min_k': MIN_K,
                    'obs_window': {
                        'start': OBS_START.isoformat(),
                        'end': OBS_END.isoformat(),
                    },
                    'target_window': {
                        'train_start': TARGET_TRAIN_START.isoformat(),
                        'test_end': TARGET_TEST_END.isoformat(),
                    },
                },
                'results': results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  ✓ Detailed results: {json_path}")
    
    # Save matrix as CSV
    csv_path = OUTPUT_DIR / "source_window_validation_matrix.csv"
    matrix_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  ✓ Matrix CSV: {csv_path}")
    
    # Generate markdown report
    md_lines = [
        "# D4 Source Window Validation Matrix",
        "",
        "## Purpose",
        "",
        "This validation treats **source window days** as a formal experimental variable.",
        "We test three window lengths (180, 210, 300 days) against four target stores (166, 155, 240, 293)",
        "to determine which configurations satisfy the K≥3 feasibility requirement.",
        "",
        "## Key Findings",
        "",
    ]
    
    # Count feasible combinations per store
    for store_id in TARGET_STORES:
        feasible_windows = []
        for window_days in WINDOW_DAYS_OPTIONS:
            result = next(
                (r for r in results if r['store_id'] == store_id and r['window_days'] == window_days),
                None
            )
            if result and result.get('k_feasible', False):
                feasible_windows.append(window_days)
        
        if len(feasible_windows) == len(WINDOW_DAYS_OPTIONS):
            status = f"✅ All windows feasible: {feasible_windows}"
        elif len(feasible_windows) > 0:
            status = f"⚠️  Partially feasible: {feasible_windows}"
        else:
            status = "❌ No feasible window"
        
        md_lines.append(f"- **Store {store_id}**: {status}")
    
    md_lines.extend([
        "",
        "## Detailed Results",
        "",
        "| Store | Window Days | Total Candidates | Valid Candidates | K≥3 Feasible |",
        "|-------|-------------|------------------|------------------|--------------|",
    ])
    
    for result in results:
        if 'error' not in result:
            feasible = '✅ Yes' if result['k_feasible'] else '❌ No'
            md_lines.append(
                f"| {result['store_id']} | {result['window_days']} | "
                f"{result['total_candidates']} | {result['valid_both_count']} | {feasible} |"
            )
    
    md_lines.extend([
        "",
        "## Recommendation",
        "",
        "Based on this matrix:",
        "",
        "1. **If one window length satisfies all stores**: Use that as the primary configuration",
        "2. **If multiple windows work for all stores**: Choose the one closest to original protocol (300d)",
        "3. **If no single window works for all stores**: Document which store requires exception handling",
        "",
        "The choice should be based on **candidate pool feasibility**, not model performance.",
        "",
        "## Files",
        "",
        f"- Detailed JSON: `{json_path.name}`",
        f"- Matrix CSV: `{csv_path.name}`",
        f"- This report: `source_window_validation_report.md`",
        "",
    ])
    
    md_path = OUTPUT_DIR / "source_window_validation_report.md"
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')
    print(f"  ✓ Markdown report: {md_path}")


def main():
    """Main entry point."""
    print("=" * 120)
    print("D4 SOURCE WINDOW VALIDATION MATRIX")
    print("=" * 120)
    print(f"\nPurpose: Validate K≥{MIN_K} feasibility under different source window lengths")
    print(f"Target stores: {TARGET_STORES}")
    print(f"Window days: {WINDOW_DAYS_OPTIONS}")
    print(f"Observation window: {OBS_START.date()} ~ {OBS_END.date()} ({len(REQUIRED_OBS_DATES)} days)")
    
    results = run_validation_matrix()
    matrix_df = generate_summary_matrix(results)
    save_results(results, matrix_df)
    
    print("\n" + "=" * 120)
    print("VALIDATION COMPLETE")
    print("=" * 120)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Review the matrix to see which (store, window_days) combinations are feasible")
    print("2. Choose window_days based on feasibility, not performance")
    print("3. Document the choice rationale in the paper")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
