import unittest

from scripts.run_full_paper_experiments import (
    _build_error_selected_source_audit_fields,
    _build_selected_source_audit_fields,
)


class TestPaperResultsKnnAuditColumns(unittest.TestCase):
    def test_no_tl_marks_selected_sources_not_applicable(self):
        fields = _build_selected_source_audit_fields(
            dataset_name="Dataset1",
            method_name="No-TL",
            source_identification=[],
        )

        self.assertEqual("NOT_APPLICABLE", fields["selected_source_ids"])
        self.assertEqual("NOT_APPLICABLE", fields["selected_source_keys"])
        self.assertEqual("NOT_APPLICABLE", fields["selected_source_raw_columns"])

    def test_dataset2_records_human_keys_and_raw_qty_promo_columns(self):
        fields = _build_selected_source_audit_fields(
            dataset_name="Dataset2",
            method_name="MSWA-TL",
            source_identification=[
                {"source_rank": 1, "source_key": "('B1', 4)"},
                {"source_rank": 2, "source_key": "('B2', 3)"},
                {"source_rank": 3, "source_key": "('B3', 9)"},
            ],
        )

        self.assertEqual("B1:4|B2:3|B3:9", fields["selected_source_ids"])
        self.assertEqual("B1 Item4|B2 Item3|B3 Item9", fields["selected_source_keys"])
        self.assertEqual(
            "QTY_B1_4/PROMO_B1_4|QTY_B2_3/PROMO_B2_3|QTY_B3_9/PROMO_B3_9",
            fields["selected_source_raw_columns"],
        )

    def test_ss_tl_records_one_source(self):
        fields = _build_selected_source_audit_fields(
            dataset_name="Dataset1",
            method_name="SS-TL",
            source_identification=[
                {"source_rank": 1, "source_key": "(2, 4)"},
            ],
        )

        self.assertEqual("2:4", fields["selected_source_ids"])
        self.assertEqual("2 Item4", fields["selected_source_keys"])
        self.assertEqual(1, fields["selected_source_count"])

    def test_error_rows_keep_explicit_selected_source_ids(self):
        no_tl = _build_error_selected_source_audit_fields("No-TL")
        ss_tl = _build_error_selected_source_audit_fields("SS-TL")

        self.assertEqual("NOT_APPLICABLE", no_tl["selected_source_ids"])
        self.assertEqual("SELECTION_UNAVAILABLE", ss_tl["selected_source_ids"])


if __name__ == "__main__":
    unittest.main()
