"""Runtime environment check before training.

Prints interpreter/runtime information and dependency availability,
then reports whether the environment is ready.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _import_with_version(module_name: str):
    try:
        module = __import__(module_name)
        return True, getattr(module, "__version__", "unknown")
    except Exception:
        return False, "N/A"


def main() -> None:
    print("=== Runtime Environment Check ===")
    print(f"sys.executable: {sys.executable}")
    print(f"sys.version: {sys.version}")
    print(f"cwd: {os.getcwd()}")

    tf_ok = False
    tf_version = "N/A"
    try:
        import tf_compat  # must be imported before tensorflow/keras
        import tensorflow as tf

        tf_ok = True
        tf_version = getattr(tf, "__version__", "unknown")
        print(f"tensorflow: {tf_version}")
    except Exception:
        print("TensorFlow not available in current interpreter")

    np_ok, np_ver = _import_with_version("numpy")
    pd_ok, pd_ver = _import_with_version("pandas")
    sk_ok, sk_ver = _import_with_version("sklearn")

    print(f"numpy: {np_ver}")
    print(f"pandas: {pd_ver}")
    print(f"scikit-learn: {sk_ver}")

    ready = tf_ok and np_ok and pd_ok and sk_ok
    print(f"Environment Ready: {'YES' if ready else 'NO'}")


if __name__ == "__main__":
    main()
