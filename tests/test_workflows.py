from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.config import load_config
from src.final_gate import FinalFreezeError
from src.inference import DeterministicMockEngine, GeneratedText
from src.integrity import IntegrityError
from src.model_worker import execute_model
from src.orchestrator import run_study
from src.preflight import ModelAssetPreflight
from src.results import ResultStore
from src.run_state import RunStateError, load_manifest, update_model_run
from src.paths import PROJECT_ROOT


def write_config(
    root: Path,
    *,
    phase: str = "development",
    analysis: bool = True,
    run_label: str = "workflow",
) -> Path:
    dataset = root / "devices.csv"
    labels = root / "labels.csv"
    specification = root / "analysis.yaml"
    dataset.write_text(
        "device_id,description\nD1,first development example\nD2,second development example\n",
        encoding="utf-8",
    )
    labels.write_text(
        "device_id,ground_truth\nD1,NON_MEDICAL_DEVICE\nD2,NON_MEDICAL_DEVICE\n",
        encoding="utf-8",
    )
    specification.write_text("version: primary_analysis_v1\n", encoding="utf-8")
    config_path = root / "study.yaml"
    config_path.write_text(yaml.safe_dump({
        "phase": phase,
        "study": {"name": "workflow-test", "run_label": run_label},
        "dataset": {"path": str(dataset), "phase": phase, "name": "workflow-data"},
        "reference_labels": {"path": str(labels)},
        "models": [{"repository": "test/local-instruct", "revision": "revision-v1"}],
        "generation": {"repeats": 2, "temperature": 0.0, "seeds": [41, 42], "max_tokens": 64},
        "outputs": {"root": str(root / "runs")},
        "analysis": {
            "enabled": analysis,
            "specification": str(specification) if analysis else None,
        },
        "failure_policy": {"continue_on_model_failure": False},
    }, sort_keys=False), encoding="utf-8")
    return config_path


def fake_assets(*args, **kwargs):
    return ModelAssetPreflight("test", True, True, 12, True, {"kind": "test"})


def in_process_model(config, paths, index):
    with patch("src.model_worker.model_asset_preflight", side_effect=fake_assets), patch(
        "src.model_worker.create_engine", return_value=DeterministicMockEngine()
    ):
        execute_model(config, paths, index)
    return 0


class MalformedEngine:
    def generate(self, messages, *, temperature, max_tokens, seed):
        return [GeneratedText("not json", 5, 2, "stop") for _ in messages]


class WorkflowTests(unittest.TestCase):
    def test_development_normal_completion_uses_real_pipeline_with_test_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = run_study(
                write_config(root), run_id="development-complete", worker_runner=in_process_model
            )
            manifest = load_manifest(paths)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["raw_evidence"]["generation_count"], 4)
            self.assertTrue((paths.analysis / "model_summary.csv").is_file())

    def test_interruption_then_resume_executes_only_missing_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_config(root)

            def interrupted(config, paths, index):
                store = ResultStore(paths)
                store.append_batch([{
                    "model": config.models[0].repository,
                    "model_revision": config.models[0].revision,
                    "device_id": "D1",
                    "run_index": 0,
                    "raw_response": "partial",
                    "parse_success": False,
                    "schema_valid": False,
                }])
                update_model_run(
                    paths,
                    "test/local-instruct@revision-v1",
                    status="partial",
                    details={"error_category": "recoverable_interruption"},
                )
                return 130

            interrupted_paths = run_study(
                config_path, run_id="resumable", worker_runner=interrupted
            )
            self.assertEqual(load_manifest(interrupted_paths)["status"], "partial")
            resumed = run_study(
                config_path, resume_run_id="resumable", worker_runner=in_process_model
            )
            records = resumed.responses.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(records), 4)
            self.assertEqual(load_manifest(resumed)["status"], "completed")

    def test_existing_run_collision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_config(root, analysis=False)
            run_study(config, run_id="collision", worker_runner=in_process_model)
            with self.assertRaisesRegex(RunStateError, "refusing overwrite"):
                run_study(config, run_id="collision", worker_runner=in_process_model)

    def test_malformed_output_is_evidence_not_execution_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = write_config(root)

            def malformed_worker(config, paths, index):
                with patch("src.model_worker.model_asset_preflight", side_effect=fake_assets), patch(
                    "src.model_worker.create_engine", return_value=MalformedEngine()
                ):
                    execute_model(config, paths, index)
                return 0

            paths = run_study(config_path, run_id="malformed", worker_runner=malformed_worker)
            manifest = load_manifest(paths)
            self.assertEqual(manifest["status"], "completed")
            records = [json.loads(line) for line in paths.responses.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(not record["parse_success"] for record in records))
            with (paths.analysis / "model_summary.csv").open(encoding="utf-8", newline="") as handle:
                summary = next(csv.DictReader(handle))
            self.assertEqual(float(summary["schema_valid_rate"]), 0.0)

    def test_model_failure_marks_run_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = run_study(
                write_config(root, analysis=False),
                run_id="model-failed",
                worker_runner=lambda config, paths, index: 1,
            )
            self.assertEqual(load_manifest(paths)["status"], "failed")
            self.assertFalse(paths.evidence_manifest.exists())

    def test_integrity_failure_is_never_marked_successful(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def integrity_failure_worker(config, paths, index):
                with patch("src.model_worker.model_asset_preflight", side_effect=fake_assets), patch(
                    "src.model_worker.create_engine", return_value=DeterministicMockEngine()
                ), patch(
                    "src.model_worker.validate_model_integrity",
                    side_effect=IntegrityError("deliberate test grid failure"),
                ):
                    try:
                        execute_model(config, paths, index)
                    except IntegrityError:
                        return 1
                return 0

            paths = run_study(
                write_config(root, analysis=False),
                run_id="integrity-failed",
                worker_runner=integrity_failure_worker,
            )
            manifest = load_manifest(paths)
            self.assertEqual(manifest["status"], "failed")
            model_state = manifest["model_runs"]["test/local-instruct@revision-v1"]
            self.assertEqual(model_state["error_category"], "integrity_failure")
            self.assertFalse(paths.evidence_manifest.exists())

    def test_final_dry_run_rejects_missing_freeze_before_creating_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = root / "development.csv"
            final = root / "final.csv"
            labels = root / "labels.csv"
            specification = root / "analysis.yaml"
            development.write_text("device_id,description\nD1,development\n", encoding="utf-8")
            final.write_text("device_id,description\nF1,final\n", encoding="utf-8")
            labels.write_text("device_id,ground_truth\nF1,CLASS_I\n", encoding="utf-8")
            specification.write_text("version: primary_analysis_v1\n", encoding="utf-8")
            config_path = root / "final.yaml"
            config_path.write_text(yaml.safe_dump({
                "phase": "final",
                "study": {"name": "final-dry", "run_label": "final"},
                "dataset": {
                    "path": str(final), "phase": "final", "name": "sealed",
                    "development_paths": [str(development)],
                },
                "reference_labels": {"path": str(labels)},
                "models": [{"repository": "example/model", "revision": "1" * 40}],
                "generation": {"repeats": 1, "temperature": 0.0, "seeds": [1], "max_tokens": 32},
                "outputs": {"root": str(root / "runs")},
                "analysis": {"enabled": True, "specification": str(specification)},
                "final_freeze": {},
            }, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(FinalFreezeError, "incomplete"):
                run_study(config_path, run_id="must-not-exist", worker_runner=in_process_model)
            self.assertFalse((root / "runs" / "must-not-exist").exists())

    def test_top_level_command_is_cwd_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = write_config(root, phase="smoke_test")
            raw = yaml.safe_load(config.read_text(encoding="utf-8"))
            raw["models"] = [{"repository": "mock://deterministic", "revision": "mock-v1"}]
            config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "run_study.py"),
                    "--config",
                    str(config),
                    "--run-id",
                    "foreign-cwd",
                ],
                cwd=PROJECT_ROOT.parent,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "runs" / "foreign-cwd" / "analysis" / "model_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
