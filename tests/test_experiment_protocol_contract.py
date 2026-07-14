from __future__ import annotations

import unittest

from src.protocols.experiment_protocol import (
    FORMAL_METHODS,
    FORMAL_PROTOCOL_TRACK,
    PROTOCOL_VERSION,
    ProtocolViolation,
    SourceIdentity,
    build_candidate_keys,
    get_experiment_protocol,
)


class ExperimentProtocolContractTest(unittest.TestCase):
    def test_tracks_windows_horizons_and_seeds_are_frozen(self) -> None:
        d1 = get_experiment_protocol("D1")
        d4 = get_experiment_protocol("dataset4")

        self.assertEqual(PROTOCOL_VERSION, "d1_d6_protocol_v1")
        self.assertEqual(FORMAL_PROTOCOL_TRACK, "strict_paper")
        self.assertEqual(
            FORMAL_METHODS,
            ("No-TL", "SS-TL", "MSWA-TL", "MSSB-TL", "MSML-TL", "MSML-TL-RFE"),
        )
        self.assertEqual(d1.track, "strict_paper")
        self.assertEqual(d4.track, "extended")
        self.assertEqual(d1.horizons, (1, 2, 3, 4, 5))
        self.assertEqual(d1.seeds, (42, 43, 44, 45, 46))

        window = d1.observation_window("2017-06-05")
        self.assertEqual(window.knn_observed_start.isoformat(), "2017-06-05")
        self.assertEqual(window.knn_observed_end.isoformat(), "2017-07-04")
        self.assertEqual(window.source_observation_cutoff, window.knn_observed_end)
        self.assertTrue(window.is_test_date("2017-07-05"))
        self.assertFalse(window.is_test_date("2017-07-04"))

    def test_d1_without_and_with_candidate_keys_are_exact(self) -> None:
        available = {
            (f"Store{store}", f"Item{item}")
            for store in range(1, 4)
            for item in range(1, 11)
        }
        protocol = get_experiment_protocol("1")

        without = build_candidate_keys(
            protocol,
            "without_info_sharing",
            ("Store1", "Item10"),
            available,
        )
        with_sharing = build_candidate_keys(
            protocol,
            "with_info_sharing",
            ("Store1", "Item10"),
            available,
        )

        self.assertEqual(
            without,
            tuple(("Store1", f"Item{item}") for item in range(1, 10)),
        )
        self.assertEqual(len(with_sharing), 27)
        self.assertNotIn(("Store1", "Item10"), with_sharing)
        self.assertEqual(with_sharing[0], ("Store1", "Item1"))
        self.assertEqual(with_sharing[-1], ("Store3", "Item9"))

    def test_d2_and_d3_candidate_keys_are_exact(self) -> None:
        d2_available = {
            (f"Brand{brand}", f"Item{item}")
            for brand in range(1, 4)
            for item in range(1, 11)
        }
        d2 = get_experiment_protocol("D2")
        self.assertEqual(
            len(build_candidate_keys(d2, "with", ("Brand1", "Item10"), d2_available)),
            27,
        )

        d3_available = {(f"Store{store}",) for store in range(1, 31)}
        d3 = get_experiment_protocol("D3")
        without = build_candidate_keys(d3, "without", ("Store10",), d3_available)
        with_sharing = build_candidate_keys(d3, "with", ("Store10",), d3_available)
        self.assertEqual(without, tuple((f"Store{store}",) for store in range(1, 10)))
        self.assertEqual(len(with_sharing), 29)
        self.assertNotIn(("Store10",), with_sharing)

    def test_extended_pool_stays_in_group_and_excludes_target(self) -> None:
        protocol = get_experiment_protocol("D5")
        available = (
            SourceIdentity(("S1", "I1"), group_value="F1"),
            SourceIdentity(("S1", "I2"), group_value="F1"),
            SourceIdentity(("S2", "I2"), group_value="F1"),
            SourceIdentity(("S2", "I3"), group_value="F2"),
        )

        without = build_candidate_keys(protocol, "without", ("S1", "I1"), available)
        with_sharing = build_candidate_keys(protocol, "with", ("S1", "I1"), available)

        self.assertEqual(without, (("S1", "I2"),))
        self.assertEqual(with_sharing, (("S1", "I2"), ("S2", "I2")))

    def test_strict_pool_fails_when_required_keys_are_missing(self) -> None:
        protocol = get_experiment_protocol("D1")
        incomplete = {("Store1", f"Item{item}") for item in range(1, 10)}

        with self.assertRaisesRegex(ProtocolViolation, "missing required candidate keys"):
            build_candidate_keys(
                protocol,
                "with",
                ("Store1", "Item10"),
                incomplete,
            )

    def test_duplicate_extended_source_key_fails(self) -> None:
        protocol = get_experiment_protocol("D6")
        duplicated = (
            SourceIdentity(("S1", "I1"), group_value="D1"),
            SourceIdentity(("S1", "I1"), group_value="D1"),
        )

        with self.assertRaisesRegex(ProtocolViolation, "duplicate source key"):
            build_candidate_keys(protocol, "with", ("S1", "I2"), duplicated)


if __name__ == "__main__":
    unittest.main()
