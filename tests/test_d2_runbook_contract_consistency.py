from __future__ import annotations

import re
from pathlib import Path

from src.protocols.experiment_protocol import D2_KNN_FEATURES, get_experiment_protocol
from src.protocols.selection_metadata import build_selection_metadata_contract


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs/protocol/d1_d6_protocol_v1_runbook.md"
AUTHORITATIVE_FORECAST_PROMO_TEST = (
    "def test_d2_configured_forecast_consumer_excludes_real_promo"
)


def _runbook_text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def test_runbook_states_the_complete_frozen_d2_knn_contract() -> None:
    text = _runbook_text()
    required_statements = (
        "D2 KNN 特征固定为 `(\"sales\", \"promo\")`，即 `sales` + historical `promo`",
        "`historical promo` 是正式 KNN 特征，不是可选字段",
        "D2 KNN 的 historical `sales` 和 historical `promo` 均只能来自 `date <= origin`",
        "KNN 输入的 source 和 target rows 都不得晚于 `origin`",
        "`forecast horizon` 的 `promo` 不可用，必须从 D2 `consumer frame` 排除（forecast promo excluded）",
        "forecast promo 的不可用性不影响历史 `promo` 作为 KNN 特征使用",
        "任何 `sales-only` KNN 配置都不属于 Gate 1R 1R.1.0 正式协议",
        "正式执行必须使用当前冻结并验证过的配置，不允许操作者根据旧说明重新生成 sales-only selection",
    )
    for statement in required_statements:
        assert statement in text


def test_runbook_contains_no_sales_only_or_forecast_promo_conflict() -> None:
    text = re.sub(r"\s+", " ", _runbook_text())
    forbidden_patterns = (
        r"D2\s*KNN[^。.!?\n]*(?:只|仅)使用\s*`?sales`?",
        r"D2\s*KNN[^。.!?\n]*(?:uses|using)\s+`?sales`?\s+only",
        r"(?:historical\s+promo|历史\s*`?promo`?)[^。.!?\n]{0,20}(?:不影响|不会影响|未用于|不用于)\s*KNN",
        r"historical\s+promo[^。.!?\n]{0,20}(?:is\s+not\s+used\s+for|does\s+not\s+affect)\s*KNN",
        r"(?:forecast\s+promo|forecast\s+horizon[^。.!?\n]*promo)[^。.!?\n]{0,30}(?:填\s*0|fill\s*0)[^。.!?\n]{0,20}(?:即可|可以|继续|用于预测|进入\s*consumer)",
    )
    for pattern in forbidden_patterns:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None


def test_runbook_d2_features_and_forecast_exclusion_match_protocol_authority() -> None:
    protocol = get_experiment_protocol("D2")
    metadata = build_selection_metadata_contract(protocol)

    assert D2_KNN_FEATURES == ("sales", "promo")
    assert protocol.knn_feature_columns == D2_KNN_FEATURES
    assert metadata["historical_feature_columns"] == list(D2_KNN_FEATURES)
    assert metadata["max_allowed_date_relation"] == "date<=origin"
    assert metadata["forecast_excluded_columns"] == ["promo"]


def test_existing_d2_forecast_promo_isolation_test_remains_authoritative() -> None:
    existing_test = ROOT / "tests/test_d2_knn_feature_contract.py"
    assert AUTHORITATIVE_FORECAST_PROMO_TEST in existing_test.read_text(encoding="utf-8")
