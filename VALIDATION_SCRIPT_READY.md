# D4 Source Window Matrix Validation - Manual Execution Required

## Issue Detected

The Python virtual environment (`.venv`) has a **binary compatibility issue** with pandas/numpy packages.
Even simple `import pandas` causes a segmentation fault (exit code 139).

This is likely due to:
- Incompatible binary builds (possibly ARM vs x86 mismatch on Apple Silicon)
- Corrupted package installation
- Missing system libraries

## Verification

```bash
# This command fails with segfault:
.venv/bin/python -c "import pandas; print('ok')"
```

## Solutions

### Option 1: Fix the Virtual Environment (Recommended)

```bash
cd /Users/ming/Desktop/复现实验/保留的复现实验修改rfe

# Remove corrupted venv
rm -rf .venv

# Create fresh venv
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Reinstall dependencies
pip install --upgrade pip
pip install pandas pyarrow numpy

# Test
python -c "import pandas as pd; print('pandas', pd.__version__)"
```

### Option 2: Use System Python

If your system Python has pandas installed:

```bash
# Use system python3 directly
/usr/bin/python3 validate_source_window_matrix.py
```

### Option 3: Use a Different Python Installation

If you have conda or another Python manager:

```bash
conda create -n d4_validation python=3.9 pandas pyarrow numpy
conda activate d4_validation
python validate_source_window_matrix.py
```

## The Script is Ready

The validation script `validate_source_window_matrix.py` has been created and is ready to run.
It will:

1. Load D4 dataset (from original train.parquet)
2. For each target store (166, 155, 240, 293)
3. Test three source window lengths (180, 210, 300 days)
4. Calculate candidate pool statistics
5. Check K≥3 feasibility
6. Generate a matrix report

## Manual Execution

Once you've fixed the Python environment, run:

```bash
cd /Users/ming/Desktop/复现实验/保留的复现实验修改rfe

# With timeout protection:
python tools/protection/codex_timeout.py --timeout 300 python validate_source_window_matrix.py

# Or directly:
python validate_source_window_matrix.py
```

Expected runtime: 2-5 minutes

## Output Files

Results will be saved to `outputs/source_window_validation/`:

- `source_window_validation_matrix.csv` - Matrix in CSV format
- `source_window_validation_detailed.json` - Detailed results
- `source_window_validation_report.md` - Human-readable report

## Codex Limitation

Codex cannot fix the Python environment issue automatically, as it requires:
- Installing system libraries
- Potential architecture-specific recompilation
- Interactive debugging

**Action Required**: Please run the script manually in your Terminal after fixing the environment.
