from __future__ import annotations

import unittest
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preprocess_clean_datasets import (
    clean_d1_dataframe,
    clean_d2_dataframe,
    clean_d3_dataframe,
    clean_d4_dataframe,
    clean_d5_store_dataframe,
    clean_d6_chunk,
    entity_coverage_summary_from_counts,
    preprocess_d5_holidays,
    preprocess_d5_oil,
    validate_clean_frame,
    validate_phase1_clean_frame,
)


class Phase1PreprocessingContractTest(unittest.TestCase):
    def test_clean_d2_melts_qty_and_promo_columns(self) -> None:
        raw = pd.DataFrame(
            {
                "DATE": ["2014-01-02", "2014-01-03"],
                "QTY_B1_1": [7, 5],
                "QTY_B2_10": [3, 4],
                "PROMO_B1_1": [0, 1],
                "PROMO_B2_10": [1, 0],
            }
        )

        clean = clean_d2_dataframe(raw)

        self.assertEqual(["date", "entity_id", "item_id", "sales", "promo", "year", "month", "week", "day"], list(clean.columns))
        self.assertEqual(4, len(clean))
        b2 = clean[(clean["entity_id"] == "B2") & (clean["item_id"] == 10) & (clean["date"] == "2014-01-02")].iloc[0]
        self.assertEqual(3, b2["sales"])
        self.assertEqual(1, b2["promo"])
        validate_clean_frame("D2", clean)

    def test_clean_d1_uses_entity_item_sales_contract(self) -> None:
        raw = pd.DataFrame(
            {
                "date": ["2013-01-01"],
                "store": [2],
                "item": [7],
                "sales": [11],
            }
        )

        clean = clean_d1_dataframe(raw)

        self.assertEqual(["date", "entity_id", "item_id", "sales", "year", "month", "week", "day"], list(clean.columns))
        self.assertEqual(2, clean.loc[0, "entity_id"])
        self.assertEqual(7, clean.loc[0, "item_id"])
        validate_phase1_clean_frame("D1", clean)


    def test_clean_d3_is_store_level_with_constant_item_id_and_leakage_customer_column(self) -> None:
        raw = pd.DataFrame(
            {
                "Date": ["2015-07-31"],
                "Store": [10],
                "Sales": [5263],
                "Customers": [555],
                "Open": [1],
                "Promo": [1],
                "StateHoliday": ["0"],
                "SchoolHoliday": [1],
                "StoreType": ["a"],
            }
        )

        clean = clean_d3_dataframe(raw)

        self.assertEqual(10, clean.loc[0, "entity_id"])
        self.assertEqual(1, clean.loc[0, "item_id"])
        self.assertEqual(10, clean.loc[0, "store_id"])
        self.assertEqual(555, clean.loc[0, "customers_leakage_risk"])
        self.assertNotIn("customers", clean.columns)
        validate_phase1_clean_frame("D3", clean)


    def test_clean_d4_expands_list_columns_with_leakage_suffixes(self) -> None:
        raw = pd.DataFrame(
            {
                "city_id": [1],
                "store_id": [20],
                "management_group_id": [2],
                "first_category_id": [25],
                "second_category_id": [62],
                "third_category_id": [31],
                "product_id": [54],
                "dt": ["2025-04-30"],
                "sale_amount": [0.9],
                "hours_sale": [[0.2, 0.0, 0.7]],
                "stock_hour6_22_cnt": [5],
                "hours_stock_status": [[1, 0, 1]],
                "activity_flag": [True],
                "discount": [1.0],
                "holiday_flag": [False],
                "precpt": [1.4],
                "avg_temperature": [20.8],
                "avg_humidity": [56.3],
                "avg_wind_level": [1.3],
            }
        )

        clean = clean_d4_dataframe(raw)

        self.assertEqual("1_20", clean.loc[0, "entity_id"])
        self.assertEqual(54, clean.loc[0, "item_id"])
        self.assertAlmostEqual(0.9, clean.loc[0, "hours_sale_sum_leakage_risk"])
        self.assertAlmostEqual(0.7, clean.loc[0, "hours_sale_max_leakage_risk"])
        self.assertEqual(2, clean.loc[0, "hours_stock_sum_leakage_risk"])
        self.assertNotIn("hours_sale", clean.columns)
        self.assertNotIn("hours_stock_status", clean.columns)
        validate_phase1_clean_frame("D4", clean)

    def test_clean_d6_chunk_melts_calendar_prices_and_price_available(self) -> None:
        sales = pd.DataFrame(
            {
                "item_id": ["FOO_1"],
                "store_id": ["CA_1"],
                "dept_id": ["FOO"],
                "cat_id": ["FOODS"],
                "state_id": ["CA"],
                "d_1": [0],
                "d_2": [3],
            }
        )
        calendar = pd.DataFrame(
            {
                "d": ["d_1", "d_2"],
                "date": ["2011-01-29", "2011-01-30"],
                "wm_yr_wk": [11101, 11102],
                "weekday": ["Saturday", "Sunday"],
                "wday": [1, 2],
                "month": [1, 1],
                "year": [2011, 2011],
                "event_name_1": [pd.NA, "Event"],
                "event_type_1": [pd.NA, "Cultural"],
                "event_name_2": [pd.NA, pd.NA],
                "snap_CA": [0, 1],
                "snap_TX": [0, 0],
                "snap_WI": [0, 0],
            }
        )
        prices = pd.DataFrame(
            {
                "store_id": ["CA_1"],
                "item_id": ["FOO_1"],
                "wm_yr_wk": [11102],
                "sell_price": [2.5],
            }
        )

        clean = clean_d6_chunk(sales, calendar, prices)

        self.assertEqual(2, len(clean))
        self.assertEqual("CA_1", clean.loc[0, "entity_id"])
        self.assertTrue(pd.isna(clean.loc[0, "sell_price"]))
        self.assertEqual(0, clean.loc[0, "price_available"])
        self.assertEqual(2.5, clean.loc[1, "sell_price"])
        self.assertEqual(1, clean.loc[1, "price_available"])
        self.assertEqual(1, clean.loc[1, "is_event_1"])
        self.assertEqual("Cultural", clean.loc[1, "event_type_1"])
        self.assertEqual(1, clean.loc[1, "snap"])
        validate_clean_frame("D6", clean)

    def test_clean_d5_store_dataframe_uses_active_period_and_no_oil_bfill(self) -> None:
        train = pd.DataFrame(
            {
                "date": ["2013-01-02", "2013-01-04"],
                "store_nbr": [1, 1],
                "item_nbr": [100, 100],
                "unit_sales": [2.0, -1.0],
                "onpromotion": [1, pd.NA],
            }
        )
        items = pd.DataFrame({"item_nbr": [100], "family": ["GROCERY"], "class": [1], "perishable": [0]})
        stores = pd.DataFrame({"store_nbr": [1], "city": ["Quito"], "state": ["Pichincha"], "type": ["A"], "cluster": [1]})
        oil = preprocess_d5_oil(
            pd.DataFrame({"date": ["2013-01-01", "2013-01-02", "2013-01-04"], "dcoilwtico": [90.0, 91.0, 92.0]})
        )
        transactions = pd.DataFrame({"date": ["2013-01-02"], "store_nbr": [1], "transactions": [10]})
        holidays = preprocess_d5_holidays(
            pd.DataFrame(
                {
                    "date": ["2013-01-02", "2013-01-03"],
                    "type": ["Holiday", "Transfer"],
                    "locale": ["Local", "Local"],
                    "locale_name": ["Quito", "Quito"],
                    "description": ["Fundacion de Quito", "Traslado Fundacion de Quito"],
                    "transferred": [True, False],
                }
            ),
            stores,
        )

        clean = clean_d5_store_dataframe(
            train,
            items=items,
            store_row=stores.iloc[0],
            oil=oil,
            transactions=transactions,
            holidays_by_store=holidays,
            global_end_date=pd.Timestamp("2013-01-04"),
        )

        self.assertEqual(["2013-01-02", "2013-01-03", "2013-01-04"], clean["date"].tolist())
        self.assertEqual([2.0, 0.0, 0.0], clean["sales"].tolist())
        self.assertEqual([1, 0, 0], clean["onpromotion"].tolist())
        self.assertEqual([0, 1, 0], clean["is_holiday"].tolist())
        self.assertEqual(90.0, clean.loc[0, "oil_price"])
        self.assertEqual(91.0, clean.loc[1, "oil_price"])
        self.assertEqual(10, clean.loc[0, "transactions"])
        self.assertEqual(0, clean.loc[1, "transactions"])
        validate_clean_frame("D5", clean)

    def test_entity_coverage_summary_reports_median_and_510_day_retention(self) -> None:
        summary = entity_coverage_summary_from_counts([3, 510, 700])

        self.assertEqual(3, summary["min_entity_coverage_days"])
        self.assertEqual(510.0, summary["median_entity_coverage_days"])
        self.assertEqual(700, summary["max_entity_coverage_days"])
        self.assertEqual(2, summary["entities_ge_510_days"])
        self.assertAlmostEqual(2 / 3, summary["entity_coverage_ge_510_rate"])


if __name__ == "__main__":
    unittest.main()
