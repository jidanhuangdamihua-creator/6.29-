"""Training environment readiness check.

Checks Python version and required runtime dependencies for training scripts,
then writes a CSV report to outputs/pipeline_checks/training_environment_check.csv.
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "outputs" / "pipeline_checks"
OUTPUT_CSV = OUTPUT_DIR / "training_environment_check.csv"


def _check_python_version() -> Dict[str, Any]:
    version = platform.python_version()
    major = sys.version_info.major
    minor = sys.version_info.minor

    # TensorFlow stable wheels are typically available for 3.8-3.12.
    if major == 3 and 8 <= minor <= 12:
        status = "pass"
        detail = f"Python {version} is generally compatible with TensorFlow stable wheels."
    else:
        status = "warn"
        detail = (
            f"Python {version} may be incompatible with TensorFlow stable wheels; "
            "consider Python 3.10-3.12 for training."
        )

    return {
        "check_name": "python_version_for_tensorflow",
        "status": status,
        "detail": detail,
    }


def _import_check(module_name: str) -> Dict[str, Any]:
    try:
        if module_name == "tensorflow":
            import tf_compat  # must be imported before tensorflow/keras
        module = importlib.import_module(module_name)
        module_ver = getattr(module, "__version__", "unknown")
        return {
            "check_name": f"import:{module_name}",
            "status": "pass",
            "detail": f"import ok, version={module_ver}",
        }
    except Exception as exc:
        return {
            "check_name": f"import:{module_name}",
            "status": "fail",
            "detail": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    checks: List[Dict[str, Any]] = []
    checks.append(_check_python_version())

    required_modules = [
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
        "tensorflow",
        "tqdm",
        "yaml",
        "seaborn",
        "scipy",
        "statsmodels",
    ]
    for module_name in required_modules:
        checks.append(_import_check(module_name))

    # Directly check project modules that are required by minimal training.
    training_modules = ["cnn_model", "single_source_tl", "msml_tl", "experiment_runner"]
    for module_name in training_modules:
        checks.append(_import_check(module_name))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_df = pd.DataFrame(checks, columns=["check_name", "status", "detail"])
    report_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("=== Training Environment Check ===")
    print(report_df.to_string(index=False))
    print(f"\nSaved report to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
