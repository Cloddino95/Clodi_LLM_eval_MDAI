from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from src.model_worker import execute_model
from src.orchestrator import run_study
from src.run_state import load_manifest


class OrchestratorTests(unittest.TestCase):
    def test_orchestrator_coordinates_mock_worker_to_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "devices.csv"
            labels = root / "labels.csv"
            config_path = root / "smoke.yaml"
            specification = root / "analysis.yaml"
            dataset.write_text("device_id,description\nD1,synthetic case\n", encoding="utf-8")
            labels.write_text("device_id,ground_truth\nD1,NON_MEDICAL_DEVICE\n", encoding="utf-8")
            specification.write_text("version: primary_analysis_v1\n", encoding="utf-8")
            config_path.write_text(yaml.safe_dump({
                "phase": "smoke_test",
                "study": {"name": "orchestrator", "run_label": "smoke"},
                "dataset": {"path": str(dataset), "phase": "smoke_test", "name": "synthetic"},
                "reference_labels": {"path": str(labels)},
                "models": [{"repository": "mock://deterministic", "revision": "mock-v1"}],
                "generation": {"repeats": 2, "temperature": 0.0, "seeds": [1, 2], "max_tokens": 32},
                "outputs": {"root": str(root / "runs")},
                "analysis": {"enabled": True, "specification": str(specification)},
                "failure_policy": {"continue_on_model_failure": False},
            }, sort_keys=False), encoding="utf-8")

            def in_process_worker(config, paths, index):
                execute_model(config, paths, index)
                return 0

            paths = run_study(
                config_path,
                run_id="orchestrated",
                worker_runner=in_process_worker,
            )
            manifest = load_manifest(paths)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(paths.responses.read_text(encoding="utf-8").splitlines()), 2)
            self.assertTrue(paths.answers.is_file())
            self.assertTrue(paths.evidence_manifest.is_file())
            self.assertTrue((paths.analysis / "model_summary.csv").is_file())
            self.assertEqual(manifest["raw_evidence"]["generation_count"], 2)
            self.assertEqual(manifest["analysis_run"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
