"""Run part of the models first, then resume with the rest: one run, sealed and analysed only at the end."""

import tempfile
import unittest
from pathlib import Path

import yaml

from src.model_worker import execute_model
from src.orchestrator import run_study
from src.run_state import load_manifest


def write_config(root: Path) -> Path:
    dataset, labels = root / "devices.csv", root / "labels.csv"
    specification, config_path = root / "analysis.yaml", root / "smoke.yaml"
    dataset.write_text("device_id,description\nD1,synthetic case\n", encoding="utf-8")
    labels.write_text("device_id,ground_truth\nD1,NON_MEDICAL_DEVICE\n", encoding="utf-8")
    specification.write_text("version: primary_analysis_v1\n", encoding="utf-8")
    config_path.write_text(yaml.safe_dump({
        "phase": "smoke_test",
        "study": {"name": "selection", "run_label": "smoke"},
        "dataset": {"path": str(dataset), "phase": "smoke_test", "name": "synthetic"},
        "reference_labels": {"path": str(labels)},
        "models": [
            {"repository": "mock://first", "revision": "mock-v1"},
            {"repository": "mock://second", "revision": "mock-v1"},
        ],
        "generation": {"repeats": 2, "temperature": 0.0, "seeds": [1, 2], "max_tokens": 32},
        "outputs": {"root": str(root / "runs")},
        "analysis": {"enabled": True, "specification": str(specification)},
        "failure_policy": {"continue_on_model_failure": False},
    }, sort_keys=False), encoding="utf-8")
    return config_path


def in_process_worker(config, paths, index):
    execute_model(config, paths, index)
    return 0


class ModelSelectionTests(unittest.TestCase):
    def test_partial_selection_then_resume_completes_one_run(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = write_config(Path(directory))

            paths = run_study(config_path, run_id="selected", worker_runner=in_process_worker,
                              models=["mock://first"])
            manifest = load_manifest(paths)
            self.assertEqual(manifest["status"], "partial")
            self.assertFalse(manifest.get("raw_evidence", {}).get("frozen"))
            self.assertNotEqual(manifest.get("analysis_run", {}).get("status"), "completed")
            self.assertEqual(len(paths.responses.read_text(encoding="utf-8").splitlines()), 2)

            paths = run_study(config_path, resume_run_id="selected", worker_runner=in_process_worker,
                              models=["mock://second"])
            manifest = load_manifest(paths)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(paths.responses.read_text(encoding="utf-8").splitlines()), 4)
            self.assertEqual(manifest["raw_evidence"]["generation_count"], 4)
            self.assertEqual(manifest["analysis_run"]["status"], "completed")

    def test_unknown_model_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = write_config(Path(directory))
            with self.assertRaises(ValueError):
                run_study(config_path, run_id="bad", worker_runner=in_process_worker,
                          models=["mock://does-not-exist"])


if __name__ == "__main__":
    unittest.main()
