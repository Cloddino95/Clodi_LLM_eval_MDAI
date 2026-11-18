from __future__ import annotations

import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

from src.datasets import (
    DatasetError,
    contamination_findings,
    load_inference_dataset,
    load_reference_labels,
    normalize_description,
    sha256_file,
    validate_phase_compatibility,
)
from src.inference import DeviceInput
from src.paths import PROJECT_ROOT


SMOKE_DATA = PROJECT_ROOT / "01_general_information_and_requirements/data/smoke_test/devices.csv"
SMOKE_LABELS = PROJECT_ROOT / "01_general_information_and_requirements/data/smoke_test/reference_labels.csv"


class DatasetTests(unittest.TestCase):
    def test_inference_loader_exposes_only_id_and_description(self):
        devices = load_inference_dataset(SMOKE_DATA)
        self.assertEqual(devices[0], DeviceInput("SMOKE-001", devices[0].description))
        self.assertFalse(hasattr(devices[0], "ground_truth"))

    def test_inference_loader_rejects_embedded_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            combined = Path(directory) / "combined.csv"
            combined.write_text(
                "device_id,description,ground_truth\nD1,Example,CLASS_I\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetError, "separate reference-label"):
                load_inference_dataset(combined)

    def test_reference_labels_are_loaded_separately(self):
        labels = load_reference_labels(SMOKE_LABELS)
        self.assertEqual(labels[0].device_id, "SMOKE-001")
        self.assertEqual(labels[0].ground_truth, "NON_MEDICAL_DEVICE")

    def test_phase_must_match_explicitly(self):
        validate_phase_compatibility("development", "development")
        with self.assertRaisesRegex(DatasetError, "cannot consume"):
            validate_phase_compatibility("development", "final")

    def test_hash_is_content_hash(self):
        digest = sha256_file(SMOKE_DATA)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, sha256_file(SMOKE_DATA))

    def test_normalization_is_deterministic(self):
        self.assertEqual(normalize_description("  ECG—Alert!  "), "ecg alert")

    def test_contamination_reports_all_overlap_classes(self):
        development = [
            DeviceInput("D1", "An ECG alert system."),
            DeviceInput("D2", "A mobile wellness application for users"),
        ]
        final = [
            DeviceInput("D1", "Different description"),
            DeviceInput("F2", "an ecg alert system"),
            DeviceInput("F3", "A mobile wellness application for user"),
        ]
        findings = contamination_findings(development, final, near_duplicate_threshold=0.85)
        kinds = {item.kind for item in findings}
        self.assertIn("device_id", kinds)
        self.assertIn("normalized_description", kinds)
        self.assertIn("near_description", kinds)

    def test_diagnostic_rejects_final_before_importing_vllm(self):
        command = [
            sys.executable,
            str(PROJECT_ROOT / "02_testing/diagnose_prompt.py"),
            "--model", "unused/model",
            "--revision", "unused-revision",
            "--devices", str(SMOKE_DATA),
            "--phase", "development",
            "--dataset-phase", "final",
        ]
        result = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("forbidden on final-phase datasets", result.stderr)


if __name__ == "__main__":
    unittest.main()
