import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts import organize_dataset_profiles as organizer


class OrganizeDatasetProfilesCleanupTests(unittest.TestCase):
    def test_cleanup_deletes_only_manifested_dataset_source_folders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outputs = root / "outputs"
            profiles = outputs / "dataset_profiles"
            source = outputs / "dataset_audit"
            unrecorded = outputs / "dataset_unrecorded"
            non_dataset = outputs / "manual_runs"
            for path in (profiles, source, unrecorded, non_dataset):
                path.mkdir(parents=True)
                (path / "keep.txt").write_text("x", encoding="utf-8")

            for required in ("raw_scan_summary.md", "verification_report.md", "rejected_claims.md"):
                (profiles / required).write_text("ok", encoding="utf-8")
            for dataset in [f"Dataset{i}" for i in range(1, 7)]:
                ds_dir = profiles / dataset
                ds_dir.mkdir()
                (ds_dir / "dataset_profile.md").write_text("profile", encoding="utf-8")

            pd.DataFrame([{
                "source_folder": str(source.resolve()),
                "archived_into": str(profiles.resolve()),
                "file_count": 1,
                "status": "ARCHIVED",
            }]).to_csv(profiles / "dataset_profile_inventory.csv", index=False)
            (profiles / "provenance_manifest.json").write_text(json.dumps({
                "source_folders": [{
                    "source_folder": str(source.resolve()),
                    "archived_into": str(profiles.resolve()),
                    "files": [str((source / "keep.txt").resolve())],
                    "status": "ARCHIVED",
                }]
            }), encoding="utf-8")

            summary = organizer.cleanup_archived_dataset_source_folders(outputs, profiles)

            self.assertFalse(source.exists())
            self.assertTrue(profiles.exists())
            self.assertTrue(unrecorded.exists())
            self.assertTrue(non_dataset.exists())
            self.assertEqual([row["deleted_path"] for row in summary["deleted"]], [str(source.resolve())])
            self.assertTrue((profiles / "deleted_source_folders_manifest.csv").is_file())
            self.assertTrue((profiles / "deleted_source_folders_manifest.md").is_file())

    def test_cleanup_refuses_candidate_outside_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outputs = root / "outputs"
            profiles = outputs / "dataset_profiles"
            outside = root / "dataset_outside"
            profiles.mkdir(parents=True)
            outside.mkdir()

            with self.assertRaises(RuntimeError):
                organizer._validate_cleanup_candidate(outputs.resolve(), outside.resolve())


if __name__ == "__main__":
    unittest.main()
