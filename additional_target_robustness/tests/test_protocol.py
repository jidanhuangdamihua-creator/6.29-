"""Protocol checks for the additional-target runner. These tests do not train."""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from additional_target_robustness.baselines import partition_sales, predict_seasonal
from additional_target_robustness.manifest import load_manifest, require_manifest_target
from additional_target_robustness.paths import REPO_ROOT, ensure_ablation_path
from additional_target_robustness.protocol import (
    HOLD_SOURCE_CACHE_HASH_MISMATCH,
    HOLD_TARGET_MANIFEST_MISMATCH,
    MANIFEST_SHA256,
    METHODS,
    ProtocolError,
    ProtocolHold,
    assert_file_hash,
    build_evaluation_index,
    calendar_for,
    method_label_dates,
    sha256_file,
)
from additional_target_robustness.runner import main, seasonal_seed_records
from additional_target_robustness.source_selection import select_top3
from additional_target_robustness.target_data import source_cache_path, split_target_frame, target_authority_path

ensure_ablation_path()
from sita_ablation.scaling import fit_target_model_scaler  # noqa: E402
from sita_ablation.windows import build_test_windows  # noqa: E402


class ProtocolTests(unittest.TestCase):
    def test_target_split_is_15_15_180_ordered_and_unique(self) -> None:
        for dataset in ("D1", "D4", "D5", "D6"):
            calendar = calendar_for(dataset)
            frame = _span_frame(calendar.train_start, calendar.test_end, sales=1.0)
            train, validation, test = split_target_frame(frame, calendar)
            self.assertEqual(len(train), 15)
            self.assertEqual(len(validation), 15)
            self.assertEqual(len(test), 180)
            self.assertEqual(str(train["date"].iloc[0].date()), calendar.train_start)
            self.assertEqual(str(validation["date"].iloc[0].date()), calendar.val_start)
            self.assertEqual(str(test["date"].iloc[0].date()), calendar.test_start)
            self.assertEqual(str(test["date"].iloc[-1].date()), calendar.test_end)
            combined = pd.concat([train, validation, test], ignore_index=True)
            self.assertFalse(combined["date"].duplicated().any())
            self.assertTrue(combined["date"].is_monotonic_increasing)
            with self.assertRaises(ProtocolError):
                split_target_frame(frame.iloc[:-1].copy(), calendar)

    def test_forecast_index_covers_180_test_dates_for_every_horizon(self) -> None:
        for dataset in ("D1", "D4", "D5", "D6"):
            calendar = calendar_for(dataset)
            labels = []
            for horizon in (1, 3, 5):
                index = build_evaluation_index(dataset, horizon)
                text = index.label_date_text()
                self.assertEqual(len(text), 180)
                self.assertEqual(text[0], calendar.test_start)
                self.assertEqual(text[-1], calendar.test_end)
                self.assertEqual(len(set(text)), 180)
                labels.append(text)
                self.assertLess(index.rows[0].forecast_origin, index.rows[0].label_date)
                self.assertGreaterEqual(index.rows[0].input_start, pd.Timestamp(calendar.val_start))
                self.assertLess(index.rows[0].input_start, pd.Timestamp(calendar.test_start))
            self.assertEqual(labels[0], labels[1])
            self.assertEqual(labels[1], labels[2])

    def test_forecast_index_matches_frozen_sita_windows(self) -> None:
        for dataset in ("D1", "D4", "D5", "D6"):
            calendar = calendar_for(dataset)
            days = pd.date_range(calendar.val_start, calendar.test_end, freq="D")
            frame = pd.DataFrame(
                {
                    "date": days,
                    "sales": np.linspace(0.1, 0.9, len(days)),
                    "sales_original": np.linspace(1.0, 2.0, len(days)),
                }
            )
            for horizon in (1, 3, 5):
                index = build_evaluation_index(dataset, horizon)
                batch = build_test_windows(
                    frame,
                    test_start=calendar.test_start,
                    test_end=calendar.test_end,
                    feature_columns=("sales",),
                    horizon=horizon,
                )
                got = tuple(pd.Timestamp(day).strftime("%Y-%m-%d") for day in batch.label_date)
                ends = tuple(pd.Timestamp(day).strftime("%Y-%m-%d") for day in batch.input_end)
                self.assertEqual(got, index.label_date_text())
                self.assertEqual(ends, tuple(row.input_end.strftime("%Y-%m-%d") for row in index.rows))
                self.assertEqual(len(batch.label_date), 180)

    def test_four_methods_share_the_evaluation_dates(self) -> None:
        for dataset in ("D1", "D4", "D5", "D6"):
            for horizon in (1, 3, 5):
                index = build_evaluation_index(dataset, horizon)
                dates = [method_label_dates(method, index) for method in METHODS]
                self.assertEqual(len({tuple(item) for item in dates}), 1)
                self.assertEqual(len(dates[0]), 180)

    def test_inputs_scaler_and_selection_do_not_see_test_labels(self) -> None:
        calendar = calendar_for("D1")
        index = build_evaluation_index("D1", 5)
        for row in index.rows:
            self.assertEqual(max(row.input_dates), row.forecast_origin)
            self.assertLess(row.forecast_origin, row.label_date)
            self.assertNotIn(row.label_date, row.input_dates)
            self.assertTrue(all(day <= row.forecast_origin for day in row.input_dates))
        frame = _d1_target_frame()
        train, _validation, test = split_target_frame(frame, calendar)
        scaler = fit_target_model_scaler(train, ("sales", "year", "month", "week", "day"))
        self.assertTrue(set(scaler.fit_dates).isdisjoint({day.strftime("%Y-%m-%d") for day in test["date"]}))
        self.assertEqual(partition_sales(frame, calendar.train_start, calendar.train_end).tolist(), [1.0] * 15)
        self.assertEqual(partition_sales(frame, calendar.val_start, calendar.val_end).tolist(), [2.0] * 15)
        self.assertTrue(np.all(test["sales"].to_numpy() == 3.0))
        outcome = select_top3("D1", "2/6", frame, _d1_source_frame(), source_universe_id="c57f402fbbc701b7")
        self.assertLess(pd.Timestamp(outcome.comparison_end), pd.Timestamp(calendar.test_start))

    def test_d4_d5_d6_selection_excludes_the_target(self) -> None:
        cases = (
            ("D4", "166/100", {"store_id": "166", "product_id": "100"}, ("166/1", "166/2", "166/3", "166/4")),
            (
                "D5",
                "48/100",
                {"store_nbr": "48", "item_nbr": "100", "family": "GROCERY I", "onpromotion": 0.0, "oil_price": 1.0},
                ("48/1", "48/2", "48/3", "48/4"),
            ),
            (
                "D6",
                "CA_1/FOODS_3_900",
                {"store_id": "CA_1", "item_id": "FOODS_3_900", "dept_id": "FOODS_3"},
                ("CA_1/FOODS_3_001", "CA_1/FOODS_3_002", "CA_1/FOODS_3_003", "CA_1/FOODS_3_004"),
            ),
        )
        for dataset, identity, target_columns, others in cases:
            target = _history_frame(dataset, identity, sales=0.0, columns=target_columns)
            frames = [target]
            key_fields = calendar_for(dataset).key_fields
            for offset, other in enumerate(others, start=1):
                columns = dict(target_columns)
                store, sku = other.split("/", 1)
                columns[key_fields[0]] = store
                columns[key_fields[1]] = sku
                frames.append(_history_frame(dataset, other, sales=float(offset + 3), columns=columns))
            outcome = select_top3(dataset, identity, target, pd.concat(frames, ignore_index=True))
            self.assertEqual(outcome.status, "COMPLETED", dataset)
            selected = [item.identity for item in outcome.selected]
            self.assertNotIn(identity, selected)
            self.assertEqual(len(selected), 3)

    def test_d1_selection_stays_inside_store_1_items_1_to_9(self) -> None:
        outcome = select_top3(
            "D1",
            "2/6",
            _d1_target_frame(),
            _d1_source_frame(),
            source_universe_id="c57f402fbbc701b7",
        )
        self.assertEqual(outcome.status, "COMPLETED")
        self.assertEqual(outcome.source_universe_id, "c57f402fbbc701b7")
        self.assertEqual([item.identity for item in outcome.selected], ["1/1", "1/2", "1/3"])
        self.assertAlmostEqual(sum(item.weight for item in outcome.selected), 1.0, places=12)

    def test_insufficient_candidates_are_not_padded(self) -> None:
        target = _history_frame("D4", "166/100", sales=0.0, columns={"store_id": "166", "product_id": "100"})
        other = _history_frame("D4", "166/1", sales=4.0, columns={"store_id": "166", "product_id": "1"})
        outcome = select_top3("D4", "166/100", target, pd.concat([target, other], ignore_index=True))
        self.assertEqual(outcome.status, "SKIPPED_INSUFFICIENT_CANDIDATES")
        self.assertEqual(outcome.selected, ())
        self.assertEqual(outcome.legal_count, 1)

    def test_manifest_is_frozen_and_runner_does_not_invent_targets(self) -> None:
        records = load_manifest()
        self.assertEqual(len(records), 40)
        manifest_path = REPO_ROOT / "outputs/additional_target_robustness_20261004/additional_targets_manifest.json"
        self.assertEqual(sha256_file(manifest_path), MANIFEST_SHA256)
        found = require_manifest_target(records, "D1", "2/6")
        self.assertEqual(found.selected_rank, 1)
        self.assertEqual(found.target_sampling_seed, 20261004)
        with self.assertRaises(ProtocolError):
            require_manifest_target(records, "D1", "9/9")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"records": []}), encoding="utf-8")
            with self.assertRaises(ProtocolHold) as caught:
                load_manifest(path)
            self.assertEqual(caught.exception.token, HOLD_TARGET_MANIFEST_MISMATCH)
        with self.assertRaises(SystemExit):
            main(["all-targets"])
        runner_text = (REPO_ROOT / "additional_target_robustness/runner.py").read_text(encoding="utf-8")
        self.assertNotIn("random.Random", runner_text)
        package = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (REPO_ROOT / "additional_target_robustness").rglob("*.py")
            if "tests" not in path.parts
        )
        self.assertNotIn("reconstruct_d5", package)

    def test_source_cache_hash_and_paths_stay_frozen(self) -> None:
        self.assertEqual(source_cache_path("D5").name, "prepared_180day_source.parquet")
        self.assertEqual(target_authority_path("D5").name, "source.parquet")
        self.assertNotIn("prepared_180day_source", target_authority_path("D5").name)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.parquet"
            path.write_bytes(b"frozen-source-calendar")
            expected = sha256_file(path)
            self.assertEqual(assert_file_hash(path, expected, token=HOLD_SOURCE_CACHE_HASH_MISMATCH), expected)
            path.write_bytes(b"rewritten")
            with self.assertRaises(ProtocolHold) as caught:
                assert_file_hash(path, expected, token=HOLD_SOURCE_CACHE_HASH_MISMATCH)
            self.assertEqual(caught.exception.token, HOLD_SOURCE_CACHE_HASH_MISMATCH)
            self.assertEqual(path.name, "source.parquet")
        banned = "\n".join(
            (REPO_ROOT / "additional_target_robustness" / name).read_text(encoding="utf-8")
            for name in ("target_data.py", "source_selection.py", "protocol.py", "manifest.py")
        )
        self.assertNotIn("to_parquet", banned)

    def test_seasonal_naive_is_lag7_on_all_180_test_dates(self) -> None:
        calendar = calendar_for("D1")
        frame = _d1_target_frame()
        sales = {
            pd.Timestamp(day).strftime("%Y-%m-%d"): float(value)
            for day, value in zip(frame["date"], frame["sales"])
        }
        for horizon in (1, 3, 5):
            index = build_evaluation_index("D1", horizon)
            predicted = predict_seasonal("D1", "2/6", horizon, index, sales)
            self.assertEqual(predicted.shape, (180,))
            for day, value in zip(index.label_date_text(), predicted):
                lag = (pd.Timestamp(day) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
                origin = (pd.Timestamp(day) - pd.Timedelta(days=horizon)).strftime("%Y-%m-%d")
                self.assertLessEqual(lag, origin)
                self.assertEqual(value, sales[lag])
            first_lag = (pd.Timestamp(calendar.test_start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            self.assertLess(first_lag, calendar.test_start)
            self.assertEqual(predicted[0], sales[first_lag])

    def test_seasonal_naive_reuses_one_forecast_on_seeds_42_to_46(self) -> None:
        index = build_evaluation_index("D1", 1)
        frame = _d1_target_frame()
        sales = {
            pd.Timestamp(day).strftime("%Y-%m-%d"): float(value)
            for day, value in zip(frame["date"], frame["sales"])
        }
        predicted = predict_seasonal("D1", "2/6", 1, index, sales)
        records = seasonal_seed_records(predicted)
        self.assertEqual([record["seed"] for record in records], [42, 43, 44, 45, 46])
        self.assertTrue(all(record["deterministic_reuse"] is True for record in records))
        self.assertTrue(all(np.array_equal(predicted, record["y_pred"]) for record in records))
        self.assertEqual(len({record["y_pred"].tobytes() for record in records}), 1)


def _span_frame(start: str, end: str, sales: float) -> pd.DataFrame:
    days = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": days, "sales": np.full(len(days), sales, dtype=np.float64)})


def _d1_target_frame() -> pd.DataFrame:
    calendar = calendar_for("D1")
    days = pd.date_range(calendar.train_start, calendar.test_end, freq="D")
    sales = np.where(
        days <= pd.Timestamp(calendar.train_end),
        1.0,
        np.where(days <= pd.Timestamp(calendar.val_end), 2.0, 3.0),
    )
    return pd.DataFrame(
        {
            "date": days,
            "store_id": "2",
            "item_id": "6",
            "sales": sales,
            "year": 2017.0,
            "month": 6.0,
            "week": 23.0,
            "day": days.day.astype(float),
        }
    )


def _d1_source_frame() -> pd.DataFrame:
    frames = [
        _history_frame("D1", "2/6", sales=0.0, columns={"store_id": "2", "item_id": "6"}),
        _history_frame("D1", "2/1", sales=0.0, columns={"store_id": "2", "item_id": "1"}),
    ]
    for item, sales in (("1", 1.0), ("2", 2.0), ("3", 3.0), ("4", 9.0)):
        frames.append(_history_frame("D1", f"1/{item}", sales=sales, columns={"store_id": "1", "item_id": item}))
    return pd.concat(frames, ignore_index=True)


def _history_frame(dataset: str, identity: str, *, sales: float, columns: dict[str, object]) -> pd.DataFrame:
    calendar = calendar_for(dataset)
    days = pd.date_range(calendar.history_start, calendar.history_end, freq="D")
    data: dict[str, object] = {"date": days, "sales": np.full(len(days), sales, dtype=np.float64)}
    data.update(columns)
    return pd.DataFrame(data)


if __name__ == "__main__":
    unittest.main()
