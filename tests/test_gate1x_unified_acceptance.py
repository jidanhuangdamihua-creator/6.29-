from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_unified_d1_d6 import build_formal_dry_run_plan
from tools.operations.gate1x_unified_acceptance import AcceptanceFailure, accept


ROOT = Path(__file__).resolve().parents[1]
SEALED_ROOT = ROOT / "数据集" / "固化数据" / "d1_d6_sealed_v1"


def test_red_final_preflight_is_ready() -> None:
    manifest_path = SEALED_ROOT / "deployment-manifest.json"
    payload: dict[str, object] = {}
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded
    datasets = payload.get("datasets")
    assert (
        payload.get("publication_state") == "authoritative"
        and isinstance(datasets, dict)
        and len(datasets) == 6
    ), "FINAL_PREFLIGHT_NOT_READY"


def _fake_preflight() -> dict[str, object]:
    datasets = {}
    for dataset_id in range(1, 7):
        datasets[f"D{dataset_id}"] = {
            "source": {
                "path": f"数据集/固化数据/d1_d6_sealed_v1/dataset{dataset_id}/source.parquet",
                "sha256": str(dataset_id) * 64,
            },
            "target": {
                "path": f"数据集/固化数据/d1_d6_sealed_v1/dataset{dataset_id}/target.parquet",
                "sha256": str(dataset_id) * 64,
            },
            "source_schema_digest": "a" * 64,
            "target_schema_digest": "b" * 64,
            "consumer_fingerprint": "c" * 64,
        }
    return {
        "manifest": {
            "formal_identity": {"combined_identity_sha256": "sha256:" + "d" * 64},
            "datasets": datasets,
            "d4_selection_authority": {"exact_key_proof_digest": "e" * 64},
        },
        "manifest_sha256": "f" * 64,
        "root_identity_sha256": "1" * 64,
        "code_inventory_sha256": "2" * 64,
    }


def test_formal_dry_run_plan_has_300_unique_cells_and_no_side_effect_flags(
    tmp_path: Path,
) -> None:
    output = Path("/tmp") / tmp_path.name
    plan = build_formal_dry_run_plan(output, preflight=_fake_preflight())
    assert plan["preflight_status"] == "ready"
    assert plan["datasets_ready"] == 6
    assert plan["cell_count"] == 300
    assert plan["unique_cell_count"] == 300
    assert plan["training_started"] is False
    assert plan["results_created"] is False
    assert plan["publication_performed"] is False


def test_acceptance_rejects_output_outside_tmp() -> None:
    with pytest.raises(AcceptanceFailure) as captured:
        accept(
            ROOT,
            expected_branch="codex/zuihou",
            expected_head="3c4e06b17d660d344ed9a0a75d9489874e19d900",
            sealed_root=SEALED_ROOT,
            output_dir=ROOT / "forbidden-acceptance-output",
            run_full_tests=True,
        )
    assert captured.value.code == "OUTPUT_DIR_NOT_TMP"
