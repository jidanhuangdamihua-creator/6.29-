# D5 Target Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a formal Dataset5 Favorita selector that outputs one target store, 3-5 target SKUs, concrete source entities, MMD statistics, structural-shift diagnostics, and CSV/JSON/MD reports.

**Architecture:** Add one dedicated script with pure helpers and a CLI, mirroring `scripts/select_d4_target_domain.py` while preserving Favorita-specific data semantics. Unit tests use small synthetic CSV fixtures and TDD to cover windows, sales cleaning, transaction gaps, family fallback, source-family alignment, MMD helpers, and output contracts.

**Tech Stack:** Python, pandas, numpy, scikit-learn `StandardScaler`/`rbf_kernel`, pytest/unittest-compatible tests.

## Global Constraints

- Raw D5 data path: `/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/数据集/原始数据/Dataset 5Favorita`
- Output path: `/Users/ming/Desktop/复现实验/保留的复现实验修改rfe/outputs/domain_adaptation/Dataset5/target_selection`
- Required output files: `target_selection_result.json`, `store_candidate_profile.csv`, `target_sku_metrics.csv`, `target_selection_report.md`
- Main table columns: `id,date,store_nbr,item_nbr,unit_sales,onpromotion`
- Item metadata columns: `item_nbr,family,class,perishable`
- Store metadata columns: `store_nbr,city,state,type,cluster`
- Transaction columns: `date,store_nbr,transactions`
- Window config: `15 train + 15 val + 180 test = 210`
- Global max date: `2017-08-15`
- `train_start`: `2017-01-17`
- `val_start`: `2017-02-01`
- `test_start`: `2017-02-16`
- `source_start`: `2016-04-22` when `source_history_days=300`
- Selection metrics use each entity's own `min_date` through `2017-01-16`
- Negative `unit_sales` values are clipped to `0` for selection metrics and features
- Natural-day missing sales are filled with `0` for feature calculation only
- Inserted missing days use `onpromotion=False`
- Transaction hard gap threshold is continuous `>30` days with no transaction record in the 210-day target window
- `has_gap_gt_7_days` is an audit field only, not a hard exclusion
- Store-level MMD uses all product families, samples up to 50 SKUs per store, and is deterministic with `random_state=42`
- Store-level MMD `coverage_ratio` uses `observed_days / 1688`, the global Favorita calendar days, not entity span
- SKU screening `coverage_ratio` uses `observed_days / date_coverage_days`
- MMD features are `mean`, `std`, `cv`, `coverage_ratio`, `iqr`, `acf_lag7`, `trend_slope`
- MMD uses one shared `StandardScaler` fit on target + source features
- MMD gamma uses `1 / (2 * median_pairwise_distance^2)`
- Structural shift matches D4 semantics: output absolute differences between target summary and source summary
- Preferred target families: `GROCERY I`, then `BEVERAGES`
- Excluded families: `BOOKS`, `MAGAZINES`, `LADIESWEAR`, `BABY CARE`, `HARDWARE`
- Source family set follows the actual final target SKU family set
- `source_entities` is canonical and contains concrete `{store_nbr, item_nbr}` objects
- Real reporting permutation count defaults to `500`; tests may use smaller counts

---

### Task 1: Unit Tests for Helpers and Contracts

**Files:**
- Create: `tests/test_d5_target_selection.py`
- Create later: `scripts/select_d5_target_domain.py`

**Interfaces:**
- Consumes: none.
- Produces expected imports from `scripts.select_d5_target_domain`: `WindowConfig`, `compute_target_windows`, `compute_source_window`, `clean_sales_frame`, `complete_daily_series`, `transaction_gap_summary`, `summarize_sales_series`, `compute_sku_metrics`, `filter_family_candidates`, `select_target_skus_with_fallback`, `build_source_entities`, `scaled_mmd`, `compute_structural_shift`, and `run_target_selection`.

- [ ] **Step 1: Write failing helper tests**

Add tests that call the expected functions:

```python
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import select_d5_target_domain as selector


def _favorita_rows(store, item, family, start="2016-01-01", periods=540, base=10.0, promo_every=0):
    rows = []
    for i, dt in enumerate(pd.date_range(start, periods=periods, freq="D")):
        rows.append({
            "id": len(rows),
            "date": dt.strftime("%Y-%m-%d"),
            "store_nbr": store,
            "item_nbr": item,
            "unit_sales": float(base + (1.0 if i % 7 == 0 else 0.0)),
            "onpromotion": bool(promo_every and i % promo_every == 0),
        })
    return rows


class D5TargetSelectionHelperTests(unittest.TestCase):
    def test_windows_match_favorita_global_dates(self):
        windows = selector.compute_target_windows("2017-08-15", selector.WindowConfig())
        source = selector.compute_source_window(windows["train_start"], 300)
        self.assertEqual(windows["train_start"], pd.Timestamp("2017-01-17"))
        self.assertEqual(windows["val_start"], pd.Timestamp("2017-02-01"))
        self.assertEqual(windows["test_start"], pd.Timestamp("2017-02-16"))
        self.assertEqual(source["source_start"], pd.Timestamp("2016-04-22"))

    def test_cleaning_clips_negative_sales_and_complete_series_fills_missing_days(self):
        df = pd.DataFrame({
            "date": ["2017-01-01", "2017-01-03"],
            "unit_sales": [-2.0, 5.0],
            "onpromotion": [True, False],
        })
        clean = selector.clean_sales_frame(df)
        full = selector.complete_daily_series(clean, "2017-01-01", "2017-01-03")
        self.assertEqual(full["unit_sales"].tolist(), [0.0, 0.0, 5.0])
        self.assertEqual(full["onpromotion"].tolist(), [True, False, False])

    def test_transaction_gap_uses_30_day_hard_gate_and_7_day_audit_flag(self):
        tx = pd.DataFrame({
            "date": ["2017-01-17", "2017-01-25", "2017-03-05"],
            "store_nbr": [1, 1, 1],
            "transactions": [1, 1, 1],
        })
        gap = selector.transaction_gap_summary(tx, 1, "2017-01-17", "2017-03-05")
        self.assertGreater(gap["max_transaction_gap_days"], 30)
        self.assertTrue(gap["has_gap_gt_7_days"])
        self.assertFalse(gap["passes_transaction_gap_gate"])

    def test_structural_shift_matches_d4_absolute_difference_semantics(self):
        target = [{"trend_slope": 3.0, "acf_lag7": 0.6, "cv": 0.4, "coverage_ratio": 0.9, "mean": 10.0}]
        source = [{"trend_slope": 1.0, "acf_lag7": -0.2, "cv": 0.1, "coverage_ratio": 0.5, "mean": 7.0}]
        shift, _, _ = selector.compute_structural_shift(target, source)
        self.assertEqual(shift["trend"], 2.0)
        self.assertEqual(shift["seasonality"], 0.8)
        self.assertEqual(shift["scale"], 3.0)

    def test_store_level_coverage_ratio_uses_global_calendar_days(self):
        series = pd.DataFrame({
            "date": pd.date_range("2017-01-01", periods=10, freq="D"),
            "unit_sales": [1.0] * 10,
        })
        summary = selector.summarize_sales_series(series, coverage_denominator=1688)
        self.assertAlmostEqual(summary["coverage_ratio"], 10 / 1688)
```

- [ ] **Step 2: Run helper tests and verify red**

Run: `pytest tests/test_d5_target_selection.py -q`

Expected: import failure for missing `scripts.select_d5_target_domain`.

- [ ] **Step 3: Add failing selection/output tests**

Add tests for family fallback, source-family alignment, shared scaler MMD, and output files:

```python
class D5TargetSelectionSelectionTests(unittest.TestCase):
    def test_family_fallback_mixes_families_and_source_follows_actual_target_families(self):
        metrics = pd.DataFrame({
            "store_nbr": [1, 1, 1, 1],
            "item_nbr": [101, 102, 201, 202],
            "family": ["GROCERY I", "GROCERY I", "BEVERAGES", "BEVERAGES"],
            "date_coverage_days": [520, 520, 520, 520],
            "coverage_ratio": [0.8, 0.8, 0.8, 0.8],
            "onpromotion_ratio": [0.0, 0.0, 0.0, 0.0],
            "cv": [0.2, 0.3, 0.1, 0.4],
            "spike_ratio": [2.0, 2.0, 2.0, 2.0],
        })
        selected = selector.select_target_skus_with_fallback(metrics)
        self.assertEqual(selected["target_family"], "MIXED")
        self.assertEqual(set(selected["target_families"]), {"GROCERY I", "BEVERAGES"})
        items = pd.DataFrame({
            "item_nbr": [101, 102, 201, 202, 301],
            "family": ["GROCERY I", "GROCERY I", "BEVERAGES", "BEVERAGES", "DAIRY"],
        })
        train = pd.DataFrame({
            "store_nbr": [2, 2, 3, 3, 4],
            "item_nbr": [101, 201, 102, 202, 301],
            "date": pd.to_datetime(["2016-04-22"] * 5),
            "unit_sales": [1, 1, 1, 1, 1],
        })
        source = selector.build_source_entities(train, items, target_store=1, target_families=selected["target_families"])
        self.assertEqual({row["item_nbr"] for row in source}, {101, 102, 201, 202})

    def test_scaled_mmd_fits_one_shared_scaler(self):
        target = np.asarray([[1.0, 2.0], [2.0, 3.0]])
        source = np.asarray([[10.0, 20.0], [12.0, 24.0]])
        mmd, gamma, scaler = selector.scaled_mmd(target, source)
        np.testing.assert_allclose(scaler.mean_, np.vstack([target, source]).mean(axis=0))
        self.assertGreaterEqual(mmd, 0.0)
        self.assertGreater(gamma, 0.0)


class D5TargetSelectionOutputTests(unittest.TestCase):
    def test_run_target_selection_writes_required_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data = root / "Dataset5"
            out = root / "target_selection"
            data.mkdir()
            rows = []
            for store, city, typ, base in [(1, "Quito", "A", 10.0), (2, "Quito", "A", 12.0), (3, "Quito", "B", 14.0)]:
                for item, family in [(101, "GROCERY I"), (102, "GROCERY I"), (201, "BEVERAGES"), (202, "BEVERAGES"), (301, "DAIRY")]:
                    rows.extend(_favorita_rows(store, item, family, base=base + item / 1000))
            train = pd.DataFrame(rows)
            train.to_csv(data / "train.csv", index=False)
            pd.DataFrame({
                "item_nbr": [101, 102, 201, 202, 301],
                "family": ["GROCERY I", "GROCERY I", "BEVERAGES", "BEVERAGES", "DAIRY"],
                "class": [1, 1, 2, 2, 3],
                "perishable": [0, 0, 0, 0, 0],
            }).to_csv(data / "items.csv", index=False)
            pd.DataFrame({
                "store_nbr": [1, 2, 3],
                "city": ["Quito", "Quito", "Quito"],
                "state": ["Pichincha", "Pichincha", "Pichincha"],
                "type": ["A", "A", "B"],
                "cluster": [1, 1, 2],
            }).to_csv(data / "stores.csv", index=False)
            tx_rows = [{"date": dt.strftime("%Y-%m-%d"), "store_nbr": store, "transactions": 100}
                       for store in [1, 2, 3] for dt in pd.date_range("2017-01-17", "2017-08-15")]
            pd.DataFrame(tx_rows).to_csv(data / "transactions.csv", index=False)
            result = selector.run_target_selection(data, out, permutations=3, store_sku_sample=3, random_state=7)
            for name in ["target_selection_result.json", "store_candidate_profile.csv", "target_sku_metrics.csv", "target_selection_report.md"]:
                self.assertTrue((out / name).exists(), name)
            payload = json.loads((out / "target_selection_result.json").read_text(encoding="utf-8"))
            self.assertIn("source_entities", payload)
            self.assertGreaterEqual(len(payload["target_skus"]), 3)
            self.assertEqual(result["target_store"], payload["target_store"])
            profile = pd.read_csv(out / "store_candidate_profile.csv")
            self.assertIn("max_transaction_gap_days", profile.columns)
            self.assertIn("has_gap_gt_7_days", profile.columns)
```

- [ ] **Step 4: Run all D5 tests and verify red**

Run: `pytest tests/test_d5_target_selection.py -q`

Expected: import failure for missing `scripts.select_d5_target_domain`.

### Task 2: Implement Core Helpers

**Files:**
- Create: `scripts/select_d5_target_domain.py`
- Modify: `tests/test_d5_target_selection.py` only if the red tests contain mistakes.

**Interfaces:**
- Produces helper functions required by Task 1.
- Later tasks consume these helpers for candidate store scan and CLI outputs.

- [ ] **Step 1: Implement window helpers and sales cleaning**

Create `WindowConfig`, constants, `compute_target_windows`, `compute_source_window`, `clean_sales_frame`, `complete_daily_series`, and `transaction_gap_summary`.

- [ ] **Step 2: Implement feature and MMD helpers**

Implement `summarize_sales_series`, `scaled_mmd`, median-heuristic gamma, permutation MMD, and `compute_structural_shift` with absolute differences.

- [ ] **Step 3: Implement SKU metrics and family fallback helpers**

Implement `compute_sku_metrics`, `filter_family_candidates`, `select_target_skus_with_fallback`, and `build_source_entities`.

- [ ] **Step 4: Run helper tests**

Run: `pytest tests/test_d5_target_selection.py -q`

Expected: helper tests pass; full output test may still fail until Task 3.

### Task 3: Implement Candidate Store Scan and Final Selection

**Files:**
- Modify: `scripts/select_d5_target_domain.py`
- Test: `tests/test_d5_target_selection.py`

**Interfaces:**
- Produces `scan_candidate_stores(train, items, stores, transactions, cfg, store_sku_sample, random_state)`, `select_target_store(scan_df)`, and final target/source feature builders.
- Consumes helpers from Task 2.

- [ ] **Step 1: Implement training CSV loading and metadata joins**

Read only needed columns from `train.csv`; parse dates; clip negative sales; join `items.csv` and `stores.csv` where required for selection and reporting.

- [ ] **Step 2: Implement store-level all-family MMD scan**

For each candidate store, sample up to `store_sku_sample` SKUs across all families, build natural-day-complete training-history features, compute MMD against all other sampled candidate/source features, and write Q25/median/Q75 fields.

- [ ] **Step 3: Implement final store ranking**

Sort by Quito, type A, transaction hard gate, MMD median distance within IQR, eligible target SKU count, and lowest eligible SKU CV.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_d5_target_selection.py -q`

Expected: tests progress to output-contract failures only.

### Task 4: Implement Output Writers and CLI

**Files:**
- Modify: `scripts/select_d5_target_domain.py`
- Test: `tests/test_d5_target_selection.py`

**Interfaces:**
- Produces `run_target_selection(dataset_root, output_dir, permutations, store_sku_sample, random_state, chunksize)` and CLI `main()`.
- Writes the four required artifacts.

- [ ] **Step 1: Implement final result assembly**

Build target features from final target SKUs, source features from source entities, final MMD/permutation p-value, structural shift, summaries, and output payload.

- [ ] **Step 2: Implement CSV/JSON/Markdown writers**

Write `store_candidate_profile.csv`, `target_sku_metrics.csv`, `target_selection_result.json`, and `target_selection_report.md`.

- [ ] **Step 3: Implement CLI**

Support `--dataset-root`, `--output-dir`, `--permutations`, `--store-sku-sample`, `--random-state`, and `--chunksize`.

- [ ] **Step 4: Run D5 tests**

Run: `pytest tests/test_d5_target_selection.py -q`

Expected: all tests pass.

### Task 5: Real Data Smoke Verification

**Files:**
- No intended code changes unless verification exposes a defect.

**Interfaces:**
- Consumes the CLI from Task 4.
- Produces real artifacts in `outputs/domain_adaptation/Dataset5/target_selection/`.

- [ ] **Step 1: Run focused unit tests**

Run: `pytest tests/test_d5_target_selection.py -q`

Expected: all D5 target selection tests pass.

- [ ] **Step 2: Run D5 selector on real raw data**

Run: `python scripts/select_d5_target_domain.py --permutations 100`

Expected: command exits 0 and prints target store, target SKU count, target family/families, source entity count, MMD value, and output directory.

- [ ] **Step 3: Validate JSON artifact**

Run: `python -m json.tool outputs/domain_adaptation/Dataset5/target_selection/target_selection_result.json`

Expected: valid JSON containing non-null `target_store`, 3-5 `target_skus`, concrete `source_entities`, MMD fields, `permutation_p_value`, structural-shift keys, and time windows.
