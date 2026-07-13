# D4 Source Window Validation Scripts - README

## Overview

This directory contains scripts to validate D4 target store candidates under different source window lengths (180, 210, 300 days). This validation treats **source window days as a formal experimental variable** rather than a post-hoc adjustment parameter.

## Created Scripts

### 1. `validate_source_window_matrix.py` (Primary)
Uses original D4 train.parquet data for validation.

**Pros**: Complete raw data, most accurate
**Cons**: Large file (~300MB), slower

### 2. `validate_source_window_solidified.py` (Recommended)
Uses pre-processed solidified parquet files (dataset4-source.parquet / dataset4-target.parquet).

**Pros**: Lighter, faster, pre-filtered
**Cons**: Depends on solidified data availability

## ⚠️  Current Issue: Python Environment Problem

**Codex cannot run these scripts** due to a binary compatibility issue in the Python virtual environment:

```bash
# This causes segmentation fault (exit code 139):
.venv/bin/python -c "import pandas"
```

This is likely caused by:
- ARM/x86 architecture mismatch on Apple Silicon
- Corrupted pandas/numpy binary installation
- Missing system libraries

## Solution: Manual Execution Required

### Step 1: Fix Python Environment

Open a **real Terminal** (not Cursor's integrated terminal) and run:

```bash
cd /Users/ming/Desktop/复现实验/保留的复现实验修改rfe

# Method A: Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas pyarrow numpy

# Verify
python -c "import pandas as pd; print('✓ pandas', pd.__version__)"
```

Or:

```bash
# Method B: Use system Python (if it has pandas)
/usr/bin/python3 -c "import pandas; print('ok')"

# If ok, you can use system Python directly
/usr/bin/python3 validate_source_window_solidified.py
```

Or:

```bash
# Method C: Use conda (if available)
conda create -n d4val python=3.9 pandas pyarrow numpy
conda activate d4val
```

### Step 2: Run the Validation Script

After fixing the environment:

```bash
cd /Users/ming/Desktop/复现实验/保留的复现实验修改rfe

# Recommended: Use solidified data version (lighter)
python validate_source_window_solidified.py

# Or: Use original data version (more complete)
python validate_source_window_matrix.py

# With timeout protection (optional):
python tools/protection/codex_timeout.py --timeout 300 python validate_source_window_solidified.py
```

**Expected runtime**: 1-3 minutes for solidified version, 3-5 minutes for original version

### Step 3: Review Results

Results will be saved to `outputs/source_window_validation/`:

```
outputs/source_window_validation/
├── source_window_validation_matrix.csv         # CSV format for Excel
├── source_window_validation_detailed.json      # Full details
└── source_window_validation_report.md          # Human-readable report
```

## What the Scripts Do

For each combination of:
- **Target stores**: 166, 155, 240, 293
- **Window days**: 180, 210, 300

The scripts calculate:
1. **Total candidates**: Products in the same store/category
2. **Valid candidates**: Products with both:
   - 30-day observation completeness (2024-11-17 to 2024-12-16)
   - Sufficient coverage in the source window (90% of window_days)
3. **K≥3 feasibility**: Whether there are at least 3 valid candidates

## Expected Output Format

```
SOURCE WINDOW VALIDATION MATRIX
====================================================================
Candidate Pool Statistics (Total / Valid / K≥3):
--------------------------------------------------------------------
Store     Window=180d                      Window=210d              Window=300d
--------------------------------------------------------------------
166        45 /  38 / ✅                    42 /  35 / ✅            38 /  30 / ✅
155        32 /  28 / ✅                    30 /  26 / ✅            25 /  20 / ✅
240        28 /  24 / ✅                    26 /  22 / ✅            22 /  18 / ✅
293        12 /  10 / ✅                    10 /   8 / ✅             8 /   5 / ✅
--------------------------------------------------------------------
```

## Interpreting Results

### Scenario 1: All windows feasible for all stores
✅ **Best case** - Choose the window closest to original protocol (300d)

### Scenario 2: Some windows work for all stores
✅ **Good** - Choose the feasible window, document why others don't work

### Scenario 3: No single window works for all stores
⚠️ **Requires decision**:
- Option A: Use different windows for different stores (document clearly)
- Option B: Relax K requirement for problematic stores
- Option C: Use first_category instead of second_category for problematic stores

## Key Principle

**The choice of window days should be based on feasibility (K≥3), NOT on which produces better model performance.**

Document in your paper:
- "We tested 180/210/300-day source windows..."
- "We selected XXX days because it ensures K≥3 across all target stores..."
- "Longer windows provide more history but reduce candidate pool due to..."

## Why Codex Can't Run This

The issue is at the system/binary level:
- Codex operates in a sandboxed environment
- Cannot reinstall system-level libraries
- Cannot fix architecture-specific binary mismatches
- Cannot interactively debug segmentation faults

**This requires user action in a real Terminal.**

## Troubleshooting

### Error: "No such file or directory: dataset4-source.parquet"

The solidified data files don't exist. Either:
1. Use the original data version: `python validate_source_window_matrix.py`
2. Or generate solidified data first (if you have the pipeline)

### Error: "Store XXX not found in source data"

That store is not in the dataset. Check if:
- Store ID is correct
- Data has been properly loaded
- Store exists in the target window

### Script runs but produces empty results

Check:
- Date ranges are correct for your dataset
- Store IDs exist in the data
- Category filters match your data structure

## Files Created

1. `validate_source_window_matrix.py` - Main validation script (uses original data)
2. `validate_source_window_solidified.py` - Alternative using solidified data  
3. `test_environment.py` - Environment diagnostic script
4. `VALIDATION_SCRIPT_READY.md` - Issue explanation
5. `README_SOURCE_WINDOW_VALIDATION.md` - This file

## Next Steps After Running

1. **Review the matrix**: Identify which (store, window_days) combinations are feasible
2. **Make decision**: Choose window_days based on feasibility matrix
3. **Document rationale**: Explain why you chose that window length
4. **Run experiments**: Use the chosen window_days for D4 experiments
5. **Report ablation**: Include 180/210/300 comparison in paper appendix

## Contact

If you encounter issues:
1. Check the environment first (can you `import pandas`?)
2. Try the solidified version if original data fails
3. Check file paths match your system
4. Verify store IDs are correct for your dataset

---

**Status**: Scripts ready, awaiting manual execution in Terminal
**Reason**: Binary compatibility issue in Cursor's Python environment
**Action needed**: Run scripts manually after fixing Python environment
