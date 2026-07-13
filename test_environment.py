#!/usr/bin/env python3
"""
Simple test to verify the environment can load D4 data
"""
import sys
from pathlib import Path

print("Testing Python environment...")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import pandas as pd
    print(f"✓ pandas imported: {pd.__version__}")
except Exception as e:
    print(f"❌ pandas import failed: {e}")
    sys.exit(1)

try:
    import pyarrow
    print(f"✓ pyarrow imported: {pyarrow.__version__}")
except Exception as e:
    print(f"⚠️  pyarrow not available: {e}")

PROJECT_ROOT = Path(__file__).parent
source_path = PROJECT_ROOT / '数据集/固化数据/dataset4-source.parquet'

print(f"\nTrying to load: {source_path}")
print(f"File exists: {source_path.exists()}")

if source_path.exists():
    try:
        df = pd.read_parquet(source_path)
        print(f"✓ Loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        print(f"  Columns: {list(df.columns[:10])}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"❌ Failed to load: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n✓ Environment test passed!")
