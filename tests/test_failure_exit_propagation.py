from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.protocols.experiment_protocol import (
    FORMAL_PROTOCOL_TRACK,
    formal_target_entity_keys,
    get_experiment_protocol,
    resolve_result_protocol_tracks,
)
from src.utils.result_acceptance import (
    ResultAcceptanceError,
    accept_formal_cell_output,
)


def test_formal_result_track_is_separate_from_frozen_d4_source_pool_track() -> None:
    source_pool_track = get_experiment_protocol("D4").track

    result_track, recorded_source_pool_track = resolve_result_protocol_tracks(
        source_pool_track,
        formal=True,
    )

    assert source_pool_track == "extended"
    assert result_track == FORMAL_PROTOCOL_TRACK
    assert recorded_source_pool_track == "extended"
    assert resolve_result_protocol_tracks(source_pool_track, formal=False) == (
        "extended",
        "extended",
    )


def test_zero_returncode_cannot_mask_invalid_written_cell(tmp_path: Path) -> None:
    subprocess_returncode = 0
    path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "dataset_id": "D5",
                "protocol_track": "strict_paper",
                "scenario": "with",
                "information_sharing": "with",
                "target_entity_key": "48/364606",
                "method": "No-TL",
                "horizon": 1,
                "seed": 42,
                "result_status": "failed",
                "error": "training failed",
                "rmse": "",
                "smape": "",
            }
        ]
    ).to_csv(path, index=False)

    assert subprocess_returncode == 0
    with pytest.raises(ResultAcceptanceError, match="terminal_error"):
        accept_formal_cell_output(
            path,
            dataset_id=5,
            mode="with",
            targets=formal_target_entity_keys(5),
            horizon=1,
            seed=42,
        )
