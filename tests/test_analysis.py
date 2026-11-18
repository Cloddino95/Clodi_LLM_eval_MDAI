from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.analysis import AnalysisError, analyze_run, build_device_model_consistency, build_model_summary
from src.config import validate_config_dict
from src.results import ResultStore, freeze_raw_evidence
from src.run_state import create_run, update_model_run


def response(device: str, classification: str, run_index: int, *, valid: bool = True):
    return {
        "model": "mock://deterministic",
        "model_revision": "mock-v1",
        "device_id": device,
        "run_index": run_index,
        "classification": classification,
        "parse_success": True,
        "schema_valid": valid,
        "parse_ok": valid,
        "citation_present": classification != "NON_MEDICAL_DEVICE",
        "no_external_resources_statement_present": False,
        "response_truncated": False,
        "json_repaired": False,
        "extraneous_text_present": False,
        "raw_response": "{}",
    }


class AnalysisTests(unittest.TestCase):
    def test_accuracy_denominators_and_consistency_are_explicit(self):
        records = [
            {**response("D1", "CLASS_I", 0), "ground_truth": "CLASS_I", "correct": True},
            {**response("D1", "CLASS_IIA", 1, valid=False), "ground_truth": "CLASS_I", "correct": False},
        ]
        summary = build_model_summary(records)[0]
        self.assertEqual(summary["accuracy_all_generations"], 0.5)
        self.assertEqual(summary["accuracy_valid_outputs"], 1.0)
        consistency = build_device_model_consistency(records)[0]
        self.assertEqual(consistency["modal_share_valid_outputs"], 1.0)
        self.assertEqual(consistency["modal_share_all_generations"], 0.5)

    def test_analysis_joins_separate_labels_and_writes_derived_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "devices.csv"
            labels = root / "labels.csv"
            config_path = root / "config.yaml"
            specification = root / "analysis.yaml"
            dataset.write_text("device_id,description\nD1,example\n", encoding="utf-8")
            labels.write_text("device_id,ground_truth\nD1,NON_MEDICAL_DEVICE\n", encoding="utf-8")
            specification.write_text("version: primary_analysis_v1\n", encoding="utf-8")
            config_path.write_text("phase: smoke_test\n", encoding="utf-8")
            config = validate_config_dict({
                "phase": "smoke_test",
                "study": {"name": "analysis", "run_label": "analysis"},
                "dataset": {"path": str(dataset), "phase": "smoke_test", "name": "test"},
                "reference_labels": {"path": str(labels)},
                "models": [{"repository": "mock://deterministic", "revision": "mock-v1"}],
                "generation": {"repeats": 1, "temperature": 0.0, "seeds": [1], "max_tokens": 32},
                "outputs": {"root": str(root / "runs")},
                "analysis": {"enabled": True, "specification": str(specification)},
            }, config_path)
            from src.datasets import sha256_file
            paths = create_run(
                config,
                dataset_sha256=sha256_file(dataset),
                reference_labels_sha256=sha256_file(labels),
                run_id="analyzed",
            )
            store = ResultStore(paths)
            store.append_batch([response("D1", "NON_MEDICAL_DEVICE", 0)])
            update_model_run(paths, "mock://deterministic@mock-v1", status="completed", details={
                "integrity": {"completed": True}
            })
            freeze_raw_evidence(paths)
            outputs = analyze_run(paths)
            self.assertTrue(outputs.model_summary.is_file())
            with outputs.model_summary.open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["accuracy_all_generations"]), 1.0)
            metadata = json.loads(outputs.metadata.read_text(encoding="utf-8"))
            self.assertEqual(metadata["n_generations"], 1)

    def test_analysis_rejects_unverified_model_runs(self):
        with self.assertRaisesRegex(AnalysisError, "integrity-verified"):
            from src.analysis import _analysis_ready
            _analysis_ready({"models": [{"repository": "m", "revision": "r"}], "model_runs": {}})


if __name__ == "__main__":
    unittest.main()
