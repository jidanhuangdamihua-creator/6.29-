from __future__ import annotations

import unittest

from src.protocols.candidate_pool import build_candidate_pool_digest


WITHOUT_INPUT = {
    "protocol_version": "d1_d6_protocol_v1",
    "dataset_id": "D1",
    "scenario": "without",
    "target_key": ("Store1", "Item10"),
    "group_cols": ("store", "item"),
    "candidate_keys": (("Store1", "Item1"), ("Store1", "Item2")),
    "observed_start": "2017-06-05",
    "observed_end": "2017-07-04",
    "feature_cols": ("sales",),
}


class CandidatePoolDigestTest(unittest.TestCase):
    def test_with_and_without_have_fixed_distinct_sha256(self) -> None:
        without = build_candidate_pool_digest(**WITHOUT_INPUT)
        with_sharing = build_candidate_pool_digest(
            **{
                **WITHOUT_INPUT,
                "scenario": "with",
                "candidate_keys": WITHOUT_INPUT["candidate_keys"]
                + (("Store2", "Item1"),),
            }
        )

        self.assertEqual(
            without,
            "7d7e0e0d6a08841426df0cea2273e420ae5d4b4dbc12c4c36e5cbf21e1328c72",
        )
        self.assertEqual(
            with_sharing,
            "e3ea5ab06308c0b6a3826ab98ad1a9a33e026d66f3af4925698ffa8fd1941478",
        )
        self.assertNotEqual(without, with_sharing)

    def test_candidate_order_does_not_change_pool_digest(self) -> None:
        reversed_keys = tuple(reversed(WITHOUT_INPUT["candidate_keys"]))
        self.assertEqual(
            build_candidate_pool_digest(**WITHOUT_INPUT),
            build_candidate_pool_digest(
                **{**WITHOUT_INPUT, "candidate_keys": reversed_keys}
            ),
        )

    def test_each_contract_input_mutation_changes_digest(self) -> None:
        baseline = build_candidate_pool_digest(**WITHOUT_INPUT)
        mutations = (
            ("protocol_version", "d1_d6_protocol_v2"),
            ("dataset_id", "D2"),
            ("scenario", "with"),
            ("target_key", ("Store1", "Item11")),
            ("group_cols", ("item", "store")),
            ("candidate_keys", (("Store1", "Item3"),)),
            ("observed_start", "2017-06-06"),
            ("observed_end", "2017-07-05"),
            ("feature_cols", ("units",)),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                self.assertNotEqual(
                    baseline,
                    build_candidate_pool_digest(
                        **{**WITHOUT_INPUT, field: value}
                    ),
                )

    def test_actual_frame_digests_are_bound_when_supplied(self) -> None:
        baseline = build_candidate_pool_digest(
            **WITHOUT_INPUT,
            source_frame_digest="a" * 64,
            target_frame_digest="b" * 64,
        )
        self.assertNotEqual(
            baseline,
            build_candidate_pool_digest(
                **WITHOUT_INPUT,
                source_frame_digest="c" * 64,
                target_frame_digest="b" * 64,
            ),
        )


if __name__ == "__main__":
    unittest.main()
