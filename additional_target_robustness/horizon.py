"""Direct-horizon contract for PatchTST, TimesNet, and the frozen SITA index.

Formal BL9 / BL10 are direct-H models. ``pred_len`` stays 1. The single
output is trained on ``y[o + h]``, where ``o`` is the last input day. They do
not roll a one-step forecast forward, and they do not rename a one-step label
as horizon 3 or 5.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from additional_target_robustness.paths import ensure_ablation_path, ensure_baseline_path
from additional_target_robustness.protocol import (
    BLOCKED_HORIZON_MISMATCH,
    HORIZONS,
    WINDOW_SIZE,
    ProtocolHold,
    build_evaluation_index,
    calendar_for,
)


def verify_direct_horizon() -> dict[str, object]:
    """Raise ``BLOCKED_HORIZON_MISMATCH`` when a forecast is not ``y[o + h]``."""
    ensure_baseline_path()
    ensure_ablation_path()
    from bl4_lstm import _direct_h_arrays
    from bl9_patchtst import FittedBL9, assert_patchtst_config, fit_bl9
    from bl10_timesnet import FittedBL10, assert_timesnet_config, fit_bl10
    from sita_ablation.windows import build_test_windows

    patch = assert_patchtst_config()
    times = assert_timesnet_config()
    if int(patch["target_window"]) != 1 or int(times["pred_len"]) != 1:
        raise ProtocolHold(BLOCKED_HORIZON_MISMATCH)
    if "output[:, channel, -1]" not in inspect.getsource(FittedBL9.predict):
        raise ProtocolHold(BLOCKED_HORIZON_MISMATCH)
    if "output[:, -1, channel]" not in inspect.getsource(FittedBL10.predict):
        raise ProtocolHold(BLOCKED_HORIZON_MISMATCH)
    for source in (inspect.getsource(fit_bl9), inspect.getsource(fit_bl10)):
        if "_direct_h_arrays" not in source or "resolved_horizon" not in source:
            raise ProtocolHold(BLOCKED_HORIZON_MISMATCH)
    _require_direct_arrays(_direct_h_arrays)
    _require_index_offset()
    _require_sita_offset(build_test_windows)
    return {
        "mode": "direct_h",
        "label": "y[forecast_origin + horizon]",
        "patchtst_target_window": 1,
        "timesnet_pred_len": 1,
        "recursive": False,
    }


def _require_direct_arrays(builder) -> None:
    values = np.arange(40, dtype=np.float64)
    for horizon in HORIZONS:
        inputs, labels = builder(values, horizon=horizon, lookback=WINDOW_SIZE, name="series")
        if labels.shape[1] != 1:
            raise ProtocolHold(BLOCKED_HORIZON_MISMATCH)
        for index in range(inputs.shape[0]):
            origin = index + WINDOW_SIZE - 1
            label_at = origin + horizon
            if float(labels[index, 0]) != float(values[label_at]):
                raise ProtocolHold(BLOCKED_HORIZON_MISMATCH)
            if horizon != 1 and float(labels[index, 0]) == float(values[origin + 1]):
                raise ProtocolHold(BLOCKED_HORIZON_MISMATCH)


def _require_index_offset() -> None:
    for dataset in ("D1", "D4", "D5", "D6"):
        calendar = calendar_for(dataset)
        for horizon in HORIZONS:
            index = build_evaluation_index(dataset, horizon)
            if index.rows[0].label_date != pd.Timestamp(calendar.test_start):
                raise ProtocolHold(BLOCKED_HORIZON_MISMATCH)
            for row in index.rows:
                if row.label_date != row.forecast_origin + pd.Timedelta(days=horizon):
                    raise ProtocolHold(BLOCKED_HORIZON_MISMATCH)
                if row.input_end != row.forecast_origin or row.input_dates[-1] != row.forecast_origin:
                    raise ProtocolHold(BLOCKED_HORIZON_MISMATCH)


def _require_sita_offset(build_test_windows) -> None:
    calendar = calendar_for("D1")
    days = pd.date_range(calendar.val_start, calendar.test_end, freq="D")
    frame = pd.DataFrame(
        {
            "date": days,
            "sales": np.linspace(0.2, 0.8, len(days)),
            "sales_original": np.arange(len(days), dtype=np.float64),
        }
    )
    for horizon in HORIZONS:
        batch = build_test_windows(
            frame,
            test_start=calendar.test_start,
            test_end=calendar.test_end,
            feature_columns=("sales",),
            horizon=horizon,
        )
        labels = pd.to_datetime(batch.label_date)
        origins = pd.to_datetime(batch.input_end)
        if len(labels) != 180 or any((label - origin).days != horizon for label, origin in zip(labels, origins)):
            raise ProtocolHold(BLOCKED_HORIZON_MISMATCH)
