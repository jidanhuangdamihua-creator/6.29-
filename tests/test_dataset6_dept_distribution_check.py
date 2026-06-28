from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.dataset6_dept_distribution_check import (
    build_dept_distribution,
    build_item_dept_mapping,
    parse_candidate_roles,
)


class Dataset6DeptDistributionCheckTests(unittest.TestCase):
    def test_build_item_dept_mapping_deduplicates_store_item_rows(self):
        raw = pd.DataFrame(
            {
                "item_id": ["A", "A", "B", "C"],
                "dept_id": ["D1", "D1", "D2", "D2"],
                "store_id": ["S1", "S2", "S1", "S1"],
            }
        )

        mapping = build_item_dept_mapping(raw, item_col="item_id", dept_col="dept_id")

        self.assertEqual(
            mapping[["item_id", "dept_id"]].to_dict("records"),
            [
                {"item_id": "A", "dept_id": "D1"},
                {"item_id": "B", "dept_id": "D2"},
                {"item_id": "C", "dept_id": "D2"},
            ],
        )

    def test_build_dept_distribution_sorts_and_accumulates_ratios(self):
        mapping = pd.DataFrame(
            {
                "item_id": ["A", "B", "C", "D"],
                "dept_id": ["D2", "D1", "D2", "D3"],
                "是否进入当前874个item集合": [True, True, True, False],
            }
        )

        distribution = build_dept_distribution(mapping, in_scope_col="是否进入当前874个item集合")

        self.assertEqual(distribution["dept_id"].tolist(), ["D2", "D1"])
        self.assertEqual(distribution["item_count"].tolist(), [2, 1])
        self.assertEqual(distribution["cumulative_item_count"].tolist(), [2, 3])
        self.assertEqual(distribution["item_ratio"].round(6).tolist(), [0.666667, 0.333333])

    def test_parse_candidate_roles_reads_source_and_target_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_report = Path(tmp) / "source_target_candidate_report.csv"
            pd.DataFrame(
                {
                    "entity_id": [
                        "store_id=CA_3|item_id=FOODS_3_586",
                        "store_id=WI_3|item_id=FOODS_3_226",
                    ],
                    "candidate_role": ["candidate_target", "candidate_source"],
                }
            ).to_csv(candidate_report, index=False)

            roles = parse_candidate_roles(candidate_report)

        self.assertEqual(roles.source_items, {"FOODS_3_226"})
        self.assertEqual(roles.target_items, {"FOODS_3_586"})


if __name__ == "__main__":
    unittest.main()
