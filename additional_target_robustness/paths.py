"""Repository paths and import hooks for reused packages."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = REPO_ROOT / "消融实验"
BASELINE_ROOT = REPO_ROOT / "scripts" / "baselines"
SEALED_ROOT = REPO_ROOT / "数据集" / "固化数据" / "d1_d6_sealed_v1"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "additional_target_robustness_20261004"
MANIFEST_JSON = OUTPUT_ROOT / "additional_targets_manifest.json"
SOURCE_AUDIT_JSON = OUTPUT_ROOT / "source_cache_audit.json"
SOURCE_MODEL_CACHE = OUTPUT_ROOT / "source_model_cache"
FULL_RUN_DIR = OUTPUT_ROOT / "full_run"
CANARY_DIR = OUTPUT_ROOT / "canary_D1_2_6"
FULL_RUN_DIR = OUTPUT_ROOT / "full_run"
CANARY_DIR = OUTPUT_ROOT / "canary_D1_2_6"


def ensure_ablation_path() -> None:
    text = str(ABLATION_ROOT)
    if text not in sys.path:
        sys.path.insert(0, text)


def ensure_baseline_path() -> None:
    text = str(BASELINE_ROOT)
    if text not in sys.path:
        sys.path.insert(0, text)
