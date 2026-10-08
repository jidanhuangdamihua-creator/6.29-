"""D5 source-history reconstruction and the four candidate gates.

These tests use synthetic frames and the authority CSVs. They do not read
prepared_180day_source.parquet, do not write a selection file, and do not train.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sita_ablation.data import ConditionFrames, DateSpan, load_dataset_contract
from sita_ablation.d5_source_history import authoritative_oil_by_date, reconstruct_d5_source_history
from sita_ablation.scaling import fit_selection_scaler
from sita_ablation.source_selection import rank_candidates
from tests.test_source_selection import _fill, _frames_from_records, _row, make_frames


_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO / "数据集/固化数据/d1_d6_sealed_v1/dataset5/prepared_180day_source_manifest.json"
_PREPARED = "prepared_180day_source.parquet"


class HistoryGateTests(unittest.TestCase):
    def test_four_defects_are_excluded_before_the_scaler(self):
        frames = make_frames(
            "D1",
            [
                {"role": "target", "key": ("1", "10"), "values": {"sales": _fill(0.0)}},
                {"role": "source", "key": ("1", "1"), "values": {"sales": _fill(4.0)}},
                {
                    "role": "source",
                    "key": ("1", "2"),
                    "values": {"sales": _fill(999.0)},
                    "drop_dates": ("2017-01-02",),
                },
                {"role": "source", "key": ("1", "3"), "values": {"sales": _fill(8.0)}},
                {
                    "role": "source",
                    "key": ("1", "4"),
                    "values": {"sales": _fill(7.0)},
                    "drop_dates": ("2017-06-02",),
                },
                {"role": "source", "key": ("1", "5"), "values": {"sales": _broken_sales()}},
            ],
            extras=[
                {"role": "source", "key": ("1", "3"), "date": "2017-01-03", "values": {"sales": 8.0}},
            ],
        )
        seen = {}

        def spy(target, candidates, variables):
            seen["target"] = np.array(target, copy=True)
            seen["candidates"] = np.array(candidates, copy=True)
            return fit_selection_scaler(target, candidates, variables)

        with patch("sita_ablation.source_selection.fit_selection_scaler", side_effect=spy):
            result = rank_candidates(frames.dataset, frames, comparison_days=30, rule="top", k=1, seed=42)
        reasons = {record.source_key: record for record in result.exclusions}
        self.assertEqual(reasons[("1", "2")].reason, "incomplete_source_history")
        self.assertEqual(reasons[("1", "2")].gate, "source_history")
        self.assertIn("2017-01-02", reasons[("1", "2")].missing_dates)
        self.assertEqual(reasons[("1", "3")].reason, "duplicate_source_history")
        self.assertEqual(reasons[("1", "3")].gate, "source_history")
        self.assertEqual(reasons[("1", "4")].reason, "missing_observed_dates")
        self.assertEqual(reasons[("1", "4")].gate, "comparison_window")
        self.assertEqual(reasons[("1", "4")].missing_dates, ("2017-06-02",))
        self.assertEqual(reasons[("1", "5")].reason, "non_finite_similarity")
        self.assertEqual(reasons[("1", "5")].gate, "comparison_variables")
        self.assertEqual(reasons[("1", "5")].variable, "sales")
        self.assertEqual(result.legal_keys, (("1", "1"),))
        self.assertEqual(result.scaler_population_keys, (("1", "1"),))
        self.assertEqual(result.gate_counts.identity, 5)
        self.assertEqual(result.gate_counts.source_history, 2)
        self.assertEqual(result.gate_counts.comparison_window, 2)
        self.assertEqual(result.gate_counts.comparison_variables, 1)
        self.assertEqual(seen["candidates"].shape[0], 1)
        self.assertNotIn(999.0, seen["candidates"])
        self.assertEqual(result.selection_scaler.maximum[0], 4.0)
        self.assertEqual(result.source_history_start, "2017-01-02")
        self.assertEqual(result.source_history_end, "2017-06-30")
        self.assertFalse(result.reconstruction.applied)

    def test_thirty_day_candidates_do_not_enter_the_scaler_for_d1_d4_or_d6(self):
        specs = {
            "D1": (("1", "10"), ("1", "1"), ("1", "2"), None),
            "D4": (("166", "258"), ("166", "1"), ("166", "2"), None),
            "D6": (("CA_1", "FOODS_3_080"), ("CA_1", "FOODS_3_090"), ("CA_1", "FOODS_3_091"), "FOODS_3"),
        }
        for dataset, (target, legal, illegal, group) in specs.items():
            with self.subTest(dataset=dataset):
                target_spec = {"role": "target", "key": target, "values": {"sales": _fill(0.0)}}
                legal_spec = {"role": "source", "key": legal, "values": {"sales": _fill(4.0)}}
                illegal_spec = {
                    "role": "source",
                    "key": illegal,
                    "values": {"sales": _fill(999.0)},
                    "history": "comparison_only",
                }
                if group is not None:
                    target_spec["group"] = group
                    legal_spec["group"] = group
                    illegal_spec["group"] = group
                frames = make_frames(dataset, [target_spec, legal_spec, illegal_spec])
                seen = {}

                def spy(target_window, candidates, variables):
                    seen["candidates"] = np.array(candidates, copy=True)
                    return fit_selection_scaler(target_window, candidates, variables)

                with patch("sita_ablation.source_selection.fit_selection_scaler", side_effect=spy):
                    result = rank_candidates(dataset, frames, comparison_days=30, rule="top", k=1, seed=42)
                    self.assertEqual(seen["candidates"].shape, (1, 1, 30))
                    self.assertNotIn(999.0, seen["candidates"])
                    short = rank_candidates(dataset, frames, comparison_days=15, rule="top", k=1, seed=42)
                self.assertEqual(result.legal_keys, (legal,))
                self.assertEqual(result.scaler_population_keys, (legal,))
                self.assertNotIn(illegal, result.legal_keys)
                self.assertEqual(result.selection_scaler.maximum[0], 4.0)
                self.assertEqual(short.legal_keys, (legal,))
                self.assertNotIn(illegal, short.legal_keys)
                short_reason = {record.source_key: record for record in short.exclusions}[illegal]
                self.assertEqual(short_reason.reason, "incomplete_source_history")
                self.assertEqual(short_reason.gate, "source_history")
                self.assertEqual(short.reconstruction.applied, False)
                self.assertEqual(len(pd.date_range(short.source_history_start, short.source_history_end, freq="D")), 180)

    def test_ell_15_still_requires_the_full_history(self):
        frames = make_frames(
            "D1",
            [
                {"role": "target", "key": ("1", "10"), "values": {"sales": _fill(0.0)}},
                {"role": "source", "key": ("1", "1"), "values": {"sales": _fill(4.0)}},
                {
                    "role": "source",
                    "key": ("1", "2"),
                    "values": {"sales": _fill(5.0)},
                    "drop_dates": ("2017-06-16",),
                },
            ],
        )
        result = rank_candidates(frames.dataset, frames, comparison_days=15, rule="top", k=1, seed=42)
        reasons = {record.source_key: record for record in result.exclusions}
        self.assertEqual(result.legal_keys, (("1", "1"),))
        self.assertEqual(result.scaler_population_keys, (("1", "1"),))
        self.assertEqual(reasons[("1", "2")].reason, "incomplete_source_history")
        self.assertIn("2017-06-16", reasons[("1", "2")].missing_dates)
        self.assertEqual(result.comparison_dates[0], "2017-06-01")
        self.assertEqual(result.comparison_dates[-1], "2017-06-15")
        self.assertEqual(result.source_history_end, "2017-06-30")


class D5ReconstructionTests(unittest.TestCase):
    def test_cutoff_rows_and_covariates_follow_the_authority_csvs(self):
        with patch("pandas.read_parquet", side_effect=AssertionError(_PREPARED)):
            hashes = _authority_hashes()
            oil = _expected_oil()
            self.assertEqual(float(oil.loc[oil["date"].eq(pd.Timestamp("2013-01-03")), "oil_price"].iloc[0]), 93.14)
            observed = _observed_pair()
            reconstructed, summary = reconstruct_d5_source_history(observed)

        self.assertTrue(summary.applied)
        self.assertEqual(summary.source_history_start, "2016-08-20")
        self.assertEqual(summary.source_history_end, "2017-02-15")
        self.assertEqual(summary.expected_day_count, 180)
        self.assertEqual(summary.input_entity_count, 1)
        self.assertEqual(summary.synthetic_row_count, 179)
        self.assertEqual(summary.original_row_count, 1)
        self.assertTrue(summary.original_rows_unchanged)
        self.assertEqual(summary.identities_only_after_cutoff, (("48", "99197"),))
        self.assertGreaterEqual(summary.rows_outside_window, 2)
        self.assertEqual(dict(summary.authority_sha256), hashes)
        self.assertNotIn(_PREPARED, json.dumps(summary.authority_sha256))
        self.assertFalse((reconstructed["date"] > pd.Timestamp("2017-02-15")).any())
        self.assertNotIn("48_99197", set(reconstructed["entity_id"]))
        self.assertEqual(len(reconstructed), 180)
        self.assertEqual(reconstructed["date"].nunique(), 180)

        observed_row = reconstructed.loc[reconstructed["date"].eq(pd.Timestamp("2017-02-15"))].iloc[0]
        self.assertEqual(observed_row["sales"], 7.0)
        self.assertEqual(observed_row["onpromotion"], 1.0)
        self.assertEqual(observed_row["family"], "SENTINEL")
        self.assertEqual(observed_row["class"], 999)
        self.assertEqual(observed_row["city"], "Nowhere")
        self.assertEqual(observed_row["oil_price"], -1.0)
        self.assertEqual(observed_row["transactions"], -5)
        self.assertEqual(observed_row["is_holiday"], 7)
        self.assertEqual(observed_row["item_id"], "keep-me")

        synthetic = reconstructed.loc[reconstructed["date"].ne(pd.Timestamp("2017-02-15"))].copy()
        self.assertTrue((synthetic["sales"] == 0).all())
        self.assertTrue((synthetic["onpromotion"] == 0).all())
        self.assertTrue((synthetic["family"] == "GROCERY I").all())
        self.assertTrue((synthetic["class"] == 1093).all())
        self.assertTrue((synthetic["perishable"] == 0).all())
        self.assertTrue((synthetic["city"] == "Quito").all())
        self.assertTrue((synthetic["state"] == "Pichincha").all())
        self.assertTrue((synthetic["type"] == "A").all())
        self.assertTrue((synthetic["cluster"] == 14).all())
        self.assertTrue((synthetic["item_id"] == "keep-me").all())
        merged = synthetic.merge(oil, on="date", how="left", suffixes=("", "_expected"))
        self.assertTrue(np.allclose(merged["oil_price"], merged["oil_price_expected"]))
        transactions = _expected_transactions(synthetic["date"])
        self.assertTrue(synthetic["transactions"].reset_index(drop=True).equals(transactions))
        by_date = synthetic.set_index("date")["is_holiday"]
        self.assertEqual(int(by_date.loc[pd.Timestamp("2016-12-25")]), 1)
        self.assertEqual(int(by_date.loc[pd.Timestamp("2017-01-01")]), 0)
        self.assertEqual(int(by_date.loc[pd.Timestamp("2017-01-02")]), 1)
        self.assertEqual(int(by_date.loc[pd.Timestamp("2016-08-24")]), 0)
        self.assertEqual(int(by_date.loc[pd.Timestamp("2016-08-20")]), 0)
        sample = synthetic.loc[synthetic["date"].eq(pd.Timestamp("2016-12-25"))].iloc[0]
        self.assertEqual(int(sample["year"]), 2016)
        self.assertEqual(int(sample["month"]), 12)
        self.assertEqual(int(sample["day"]), 25)
        self.assertEqual(int(sample["week"]), int(pd.Timestamp("2016-12-25").isocalendar().week))

    def test_rank_candidates_drops_cutoff_identities_and_does_not_pad(self):
        frames = _cutoff_frames()
        with patch("pandas.read_parquet", side_effect=AssertionError(_PREPARED)):
            with patch("sita_ablation.source_selection.fit_selection_scaler") as scaler:
                result = rank_candidates("D5", frames, comparison_days=30, rule="top", k=3, seed=42)
        scaler.assert_not_called()
        self.assertEqual(result.status, "SKIPPED_INSUFFICIENT_CANDIDATES")
        self.assertEqual(result.selected, ())
        self.assertIsNone(result.selection_scaler)
        self.assertEqual(result.scaler_population_keys, ())
        self.assertEqual(result.legal_keys, (("48", "96995"),))
        self.assertNotIn(("48", "99197"), result.legal_keys)
        self.assertEqual(result.reconstruction.identities_only_after_cutoff, (("48", "99197"),))
        self.assertEqual(result.reconstruction.synthetic_row_count, 179)
        self.assertTrue(result.reconstruction.original_rows_unchanged)
        self.assertEqual(result.reconstruction.input_entity_count, 1)
        self.assertEqual(result.source_history_start, "2016-08-20")
        self.assertEqual(result.source_history_end, "2017-02-15")
        self.assertEqual(result.gate_counts.identity, 1)
        self.assertEqual(result.gate_counts.comparison_variables, 1)
        self.assertGreaterEqual(result.reconstruction.rows_outside_window, 2)

    def test_unknown_covariate_is_not_given_a_default(self):
        frame = _observed_pair()
        frame = frame.loc[frame["entity_id"].eq("48_96995")].copy()
        frame["mystery"] = 1
        with self.assertRaises(Exception) as caught:
            reconstruct_d5_source_history(frame)
        self.assertIn("without reconstruction authority", str(caught.exception))


def _broken_sales():
    values = _fill(4.0)
    values[0] = np.nan
    return values


def _authority_hashes():
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    found = {}
    for name, entry in manifest["auxiliary_csvs"].items():
        path = _REPO / entry["path"]
        self_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if self_hash != entry["sha256"]:
            raise AssertionError(f"{name} SHA256 does not match the manifest")
        if path.name == _PREPARED:
            raise AssertionError("authority path points at the prepared parquet")
        found[name] = self_hash
    return found


def _expected_oil():
    path = _REPO / json.loads(_MANIFEST.read_text(encoding="utf-8"))["auxiliary_csvs"]["oil"]["path"]
    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw["date"], errors="coerce").dropna().dt.normalize()
    return authoritative_oil_by_date(raw, expected_dates=pd.date_range(dates.min(), dates.max(), freq="D"))


def _expected_transactions(dates: pd.Series) -> pd.Series:
    path = _REPO / json.loads(_MANIFEST.read_text(encoding="utf-8"))["auxiliary_csvs"]["transactions"]["path"]
    raw = pd.read_csv(path)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    raw = raw.loc[raw["store_nbr"].astype(int).eq(48), ["date", "transactions"]]
    lookup = raw.drop_duplicates("date").set_index("date")["transactions"]
    values = [int(lookup.loc[day]) if day in lookup.index else 0 for day in dates.tolist()]
    return pd.Series(values, dtype="int64")


def _observed_pair() -> pd.DataFrame:
    inside = _source_row("48_96995", 48, 96995, "2017-02-15", sales=7.0, onpromotion=1.0)
    leaked = _source_row("48_96995", 48, 96995, "2017-02-16", sales=12345.0, onpromotion=1.0)
    late = _source_row("48_99197", 48, 99197, "2017-02-16", sales=5.0, onpromotion=0.0)
    return pd.DataFrame([inside, leaked, late])


def _source_row(entity, store, item, day, *, sales, onpromotion) -> dict:
    return {
        "date": pd.Timestamp(day),
        "entity_id": entity,
        "item_id": "keep-me",
        "store_nbr": store,
        "item_nbr": item,
        "sales": sales,
        "onpromotion": onpromotion,
        "family": "SENTINEL",
        "class": 999,
        "perishable": 9,
        "city": "Nowhere",
        "state": "Nowhere",
        "type": "Z",
        "cluster": 1,
        "transactions": -5,
        "oil_price": -1.0,
        "is_holiday": 7,
        "year": 1,
        "month": 1,
        "week": 1,
        "day": 1,
    }


def _cutoff_frames() -> ConditionFrames:
    contract = load_dataset_contract("D5")
    dates = pd.date_range(contract.target_train.start, contract.target_val.end, freq="D")
    target_records = [
        _row(
            contract,
            ("48", "1159414"),
            day,
            {"sales": 0.0, "onpromotion": 0.0, "oil_price": 0.0},
            None,
            "GROCERY I",
        )
        for day in dates
    ]
    source_records = [
        _source_row("48_96995", 48, 96995, "2017-02-15", sales=7.0, onpromotion=1.0),
        _source_row("48_96995", 48, 96995, "2017-02-16", sales=12345.0, onpromotion=1.0),
        _source_row("48_99197", 48, 99197, "2017-02-16", sales=5.0, onpromotion=0.0),
    ]
    for record in source_records:
        record["family"] = "GROCERY I"
    return _frames_from_records(contract, target_records, source_records)


if __name__ == "__main__":
    unittest.main()
