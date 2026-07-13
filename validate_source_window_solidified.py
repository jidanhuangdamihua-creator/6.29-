#!/usr/bin/env python3
"""
D4 Source Window Matrix Validation (Using Solidified Data)
===========================================================

This version uses pre-processed solidified parquet files which should be
lighter and less prone to loading issues.

For stores 166, 155, 240, 293 and window days 180, 210, 300,
validate K≥3 feasibility.
"""

import sys
from pathlib import Path
import json

def safe_import_pandas():
    """Safely import pandas with error handling."""
    try:
        import pandas as pd
        import numpy as np
        return pd, np, None
    except Exception as e:
        return None, None, str(e)

pd, np, import_error = safe_import_pandas()

if import_error:
    print(f"""
❌ ERROR: Cannot import pandas
{import_error}

This is likely a binary compatibility issue in your Python environment.

SOLUTION:
1. Open a Terminal (not Cursor's integrated terminal)
2. Navigate to the project directory
3. Run: python validate_source_window_matrix.py

Or fix the environment first:
   rm -rf .venv
   python3 -m venv .venv
   source .venv/bin/activate
   pip install pandas pyarrow numpy
""", file=sys.stderr)
    sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent

# Use solidified data (lighter, pre-processed)
SOURCE_PATH = PROJECT_ROOT / "数据集/固化数据/dataset4-source.parquet"
TARGET_PATH = PROJECT_ROOT / "数据集/固化数据/dataset4-target.parquet"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "source_window_validation"

TARGET_STORES = [166, 155, 240, 293]
WINDOW_DAYS_OPTIONS = [180, 210, 300]

# D4 observation window (30 days before train_start)
OBS_START = pd.Timestamp("2024-11-17")
OBS_END = pd.Timestamp("2024-12-16")

MIN_K = 3

# ============================================================================
# Analysis Functions
# ============================================================================

def load_solidified_data():
    """Load solidified D4 data."""
    print("\n[1/4] Loading solidified D4 data...")
    
    if not SOURCE_PATH.exists():
        print(f"❌ Source file not found: {SOURCE_PATH}")
        print("\nPlease ensure the solidified data exists.")
        print("You may need to run the data preparation script first.")
        sys.exit(1)
    
    if not TARGET_PATH.exists():
        print(f"❌ Target file not found: {TARGET_PATH}")
        sys.exit(1)
    
    try:
        source_df = pd.read_parquet(SOURCE_PATH)
        target_df = pd.read_parquet(TARGET_PATH)
        
        # Convert date columns
        if 'date' in source_df.columns:
            source_df['date'] = pd.to_datetime(source_df['date'])
        if 'date' in target_df.columns:
            target_df['date'] = pd.to_datetime(target_df['date'])
        
        print(f"  ✓ Source: {len(source_df):,} rows")
        print(f"  ✓ Target: {len(target_df):,} rows")
        
        return source_df, target_df
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def analyze_store_window(source_df, target_df, store_id, window_days):
    """
    Analyze candidate pool for a specific (store, window_days) combination.
    
    Returns:
        Dict with analysis results
    """
    # Get target info for this store
    targets = target_df[
        target_df['store_id'] == store_id
    ][['product_id', 'first_category_id', 'second_category_id']].drop_duplicates()
    
    if targets.empty:
        return {
            'store_id': store_id,
            'window_days': window_days,
            'error': 'No targets found for this store',
        }
    
    # Use the most common category
    target_second_cat = targets['second_category_id'].mode()[0]
    target_product_id = targets[
        targets['second_category_id'] == target_second_cat
    ].iloc[0]['product_id']
    
    # Filter source data:
    # - Same store (WITHOUT scenario)
    # - Same second_category
    # - Exclude target product
    candidates = source_df[
        (source_df['store_id'] == store_id) &
        (source_df['second_category_id'] == target_second_cat) &
        (source_df['product_id'] != target_product_id)
    ]
    
    candidate_products = candidates['product_id'].unique()
    total_candidates = len(candidate_products)
    
    # Check 30-day completeness and window coverage
    required_obs_dates = pd.date_range(OBS_START, OBS_END, freq='D')
    window_start = OBS_END - pd.Timedelta(days=window_days - 1)
    
    valid_both = []
    
    for product_id in candidate_products:
        product_data = candidates[candidates['product_id'] == product_id]
        product_dates = pd.to_datetime(product_data['date'].unique())
        
        # Check 30-day completeness
        obs_dates_covered = set(product_dates).intersection(set(required_obs_dates))
        has_30day = len(obs_dates_covered) == len(required_obs_dates)
        
        # Check window coverage
        dates_in_window = product_dates[
            (product_dates >= window_start) & (product_dates <= OBS_END)
        ]
        has_window = len(dates_in_window) >= window_days * 0.9  # 90% coverage
        
        if has_30day and has_window:
            valid_both.append(int(product_id))
    
    valid_count = len(valid_both)
    feasible = valid_count >= MIN_K
    
    return {
        'store_id': store_id,
        'window_days': window_days,
        'target_product_id': int(target_product_id),
        'target_second_category': int(target_second_cat),
        'total_candidates': total_candidates,
        'valid_candidates': valid_count,
        'k_feasible': feasible,
        'sample_valid_products': valid_both[:5],  # Sample
    }


def run_analysis(source_df, target_df):
    """Run analysis for all store×window combinations."""
    print("\n[2/4] Running validation matrix...")
    print(f"  Target stores: {TARGET_STORES}")
    print(f"  Window days: {WINDOW_DAYS_OPTIONS}")
    
    results = []
    
    for store_id in TARGET_STORES:
        # Check if store exists in source data
        if store_id not in source_df['store_id'].values:
            print(f"\n  ⚠️  Store {store_id} not found in source data, skipping...")
            continue
        
        print(f"\n  Store {store_id}:")
        
        for window_days in WINDOW_DAYS_OPTIONS:
            result = analyze_store_window(source_df, target_df, store_id, window_days)
            results.append(result)
            
            if 'error' in result:
                print(f"    Window {window_days}d: ❌ {result['error']}")
            else:
                icon = '✅' if result['k_feasible'] else '❌'
                print(f"    Window {window_days}d: {icon} "
                      f"Total={result['total_candidates']}, "
                      f"Valid={result['valid_candidates']}, "
                      f"K≥{MIN_K}={result['k_feasible']}")
    
    return results


def generate_summary(results):
    """Generate and display summary matrix."""
    print("\n[3/4] Generating summary matrix...")
    
    print("\n" + "=" * 120)
    print("SOURCE WINDOW VALIDATION MATRIX")
    print("=" * 120)
    print("\nCandidate Pool Statistics (Total / Valid / K≥3):")
    print("-" * 120)
    
    header = f"{'Store':<10}"
    for wd in WINDOW_DAYS_OPTIONS:
        header += f"{'Window=' + str(wd) + 'd':<35}"
    print(header)
    print("-" * 120)
    
    for store_id in TARGET_STORES:
        line = f"{store_id:<10}"
        for window_days in WINDOW_DAYS_OPTIONS:
            result = next(
                (r for r in results 
                 if r['store_id'] == store_id and r['window_days'] == window_days),
                None
            )
            
            if result and 'error' not in result:
                icon = '✅' if result['k_feasible'] else '❌'
                cell = f"{result['total_candidates']:>3} / {result['valid_candidates']:>3} / {icon:<2}"
            else:
                cell = "N/A"
            
            line += f"{cell:<35}"
        print(line)
    
    print("-" * 120)
    print("\nLegend: Total Candidates / Valid Candidates / K≥3 Feasible")
    print()


def save_results(results):
    """Save results to files."""
    print("\n[4/4] Saving results...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON
    json_path = OUTPUT_DIR / "source_window_validation_detailed.json"
    with json_path.open('w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'target_stores': TARGET_STORES,
                'window_days_options': WINDOW_DAYS_OPTIONS,
                'min_k': MIN_K,
                'data_source': 'solidified',
            },
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ JSON: {json_path}")
    
    # Generate markdown report
    md_lines = [
        "# D4 Source Window Validation Matrix",
        "",
        "## Results Summary",
        "",
    ]
    
    for store_id in TARGET_STORES:
        feasible_windows = [
            r['window_days'] for r in results
            if r['store_id'] == store_id and r.get('k_feasible', False)
        ]
        
        if len(feasible_windows) == len(WINDOW_DAYS_OPTIONS):
            status = f"✅ All windows feasible: {feasible_windows}"
        elif feasible_windows:
            status = f"⚠️  Partially feasible: {feasible_windows}"
        else:
            status = "❌ No feasible window"
        
        md_lines.append(f"- **Store {store_id}**: {status}")
    
    md_lines.extend([
        "",
        "## Detailed Results",
        "",
        "| Store | Window (days) | Total Candidates | Valid Candidates | K≥3 Feasible |",
        "|-------|---------------|------------------|------------------|--------------|",
    ])
    
    for r in results:
        if 'error' not in r:
            feasible = '✅ Yes' if r['k_feasible'] else '❌ No'
            md_lines.append(
                f"| {r['store_id']} | {r['window_days']} | "
                f"{r['total_candidates']} | {r['valid_candidates']} | {feasible} |"
            )
    
    md_lines.extend([
        "",
        "## Recommendation",
        "",
        "Based on this matrix, select the source window days that:",
        "1. Satisfies K≥3 for all (or most) target stores",
        "2. Is closest to the original protocol (300 days) when possible",
        "3. Is documented with clear rationale for the choice",
        "",
        "The choice should be based on **feasibility**, not model performance.",
        "",
    ])
    
    md_path = OUTPUT_DIR / "source_window_validation_report.md"
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')
    
    print(f"  ✓ Report: {md_path}")
    
    # Save simple CSV
    try:
        import csv
        csv_path = OUTPUT_DIR / "source_window_validation_matrix.csv"
        
        with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'store_id', 'window_days', 'total_candidates',
                'valid_candidates', 'k_feasible'
            ])
            writer.writeheader()
            for r in results:
                if 'error' not in r:
                    writer.writerow({
                        'store_id': r['store_id'],
                        'window_days': r['window_days'],
                        'total_candidates': r['total_candidates'],
                        'valid_candidates': r['valid_candidates'],
                        'k_feasible': r['k_feasible'],
                    })
        
        print(f"  ✓ CSV: {csv_path}")
        
    except Exception as e:
        print(f"  ⚠️  Could not save CSV: {e}")


def main():
    """Main entry point."""
    print("=" * 120)
    print("D4 SOURCE WINDOW VALIDATION MATRIX")
    print("=" * 120)
    print(f"\nValidating K≥{MIN_K} feasibility for:")
    print(f"  Stores: {TARGET_STORES}")
    print(f"  Window days: {WINDOW_DAYS_OPTIONS}")
    print(f"  Data source: Solidified parquet files")
    
    try:
        source_df, target_df = load_solidified_data()
        results = run_analysis(source_df, target_df)
        generate_summary(results)
        save_results(results)
        
        print("\n" + "=" * 120)
        print("VALIDATION COMPLETE")
        print("=" * 120)
        print(f"\nOutputs saved to: {OUTPUT_DIR}")
        print("\nNext steps:")
        print("1. Review the matrix to identify which window lengths are feasible")
        print("2. Choose window_days based on feasibility, not performance")
        print("3. Document the rationale in your paper")
        print()
        
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
