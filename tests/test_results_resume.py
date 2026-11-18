from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import GenerationConfig, ModelConfig, validate_config_dict
from src.inference import DeviceInput, GeneratedText, run_model_incremental
from src.results import (
    GenerationKey,
    ResultStore,
    expected_generation_keys,
    missing_generation_keys,
    read_jsonl,
    freeze_raw_evidence,
    verify_frozen_raw_evidence,
)
from src.resume import validate_resume_compatibility
from src.run_state import RunStateError, create_run, load_manifest


class CountingEngine:
    def __init__(self):
        self.calls = 0

    def generate(self, messages, *, temperature, max_tokens, seed):
        self.calls += 1
        response = json.dumps({
            "classification": "CLASS_I",
            "rationale": "ok",
            "applicable_regulation": "MDR",
            "applicable_provision": "Annex VIII",
            "missing_information": [],
        })
        return [GeneratedText(response, 3, 4, "stop") for _ in messages]


def make_config(root: Path, source: Path):
    source.write_text("phase: smoke_test\n", encoding="utf-8")
    return validate_config_dict({
        "phase": "smoke_test",
        "study": {"name": "resume", "run_label": "resume"},
        "dataset": {"path": "devices.csv", "phase": "smoke_test", "name": "synthetic"},
        "reference_labels": {"path": None},
        "models": [{"repository": "mock/model", "revision": "revision1"}],
        "generation": {"repeats": 2, "temperature": 0.0, "seeds": [1, 2], "max_tokens": 16},
        "outputs": {"root": str(root)},
        "analysis": {"enabled": False},
    }, source)


class ResultResumeTests(unittest.TestCase):
    def test_incremental_resume_runs_only_missing_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root / "runs", root / "config.yaml")
            paths = create_run(config, dataset_sha256="d" * 64, reference_labels_sha256=None, run_id="resume")
            store = ResultStore(paths)
            model = config.models[0]
            devices = [DeviceInput("D1", "one"), DeviceInput("D2", "two")]
            engine = CountingEngine()

            first_repeat = GenerationConfig(1, 0.0, (1,), 16)
            self.assertEqual(run_model_incremental(engine, model, devices, first_repeat, store), 2)
            self.assertEqual(len(read_jsonl(paths.responses)), 2)

            resumed_store = ResultStore(paths)
            generated = run_model_incremental(engine, model, devices, config.generation, resumed_store)
            self.assertEqual(generated, 2)
            self.assertEqual(len(read_jsonl(paths.responses)), 4)
            self.assertEqual(engine.calls, 2)
            self.assertEqual(len(paths.answers.read_text(encoding="utf-8").splitlines()), 5)

    def test_duplicate_append_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root / "runs", root / "config.yaml")
            paths = create_run(config, dataset_sha256="d" * 64, reference_labels_sha256=None, run_id="duplicate")
            store = ResultStore(paths)
            record = {
                "model": "mock/model", "model_revision": "revision1", "device_id": "D1",
                "run_index": 0, "raw_response": "{}",
            }
            store.append(record)
            with self.assertRaisesRegex(RunStateError, "duplicate generation"):
                store.append(record)

    def test_missing_key_calculation(self):
        models = [ModelConfig("m", "r")]
        devices = [DeviceInput("D1", "one"), DeviceInput("D2", "two")]
        generation = GenerationConfig(2, 0.0, (1, 2), 16)
        expected = expected_generation_keys(models, devices, generation)
        completed = {GenerationKey("m", "r", "D1", 0)}
        self.assertEqual(len(missing_generation_keys(expected, completed)), 3)

    def test_incompatible_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root / "runs", root / "config.yaml")
            paths = create_run(config, dataset_sha256="d" * 64, reference_labels_sha256=None, run_id="compat")
            manifest = load_manifest(paths)
            changed = validate_config_dict({
                "phase": "smoke_test",
                "study": {"name": "resume", "run_label": "resume"},
                "dataset": {"path": "devices.csv", "phase": "smoke_test", "name": "synthetic"},
                "reference_labels": {"path": None},
                "models": [{"repository": "mock/model", "revision": "revision1"}],
                "generation": {"repeats": 2, "temperature": 0.1, "seeds": [1, 2], "max_tokens": 16},
                "outputs": {"root": str(root / "runs")},
                "analysis": {"enabled": False},
            }, root / "changed.yaml")
            with self.assertRaisesRegex(RunStateError, "incompatible"):
                validate_resume_compatibility(manifest, changed, dataset_sha256="d" * 64)

    def test_frozen_evidence_is_hash_verified_and_refuses_new_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root / "runs", root / "config.yaml")
            paths = create_run(config, dataset_sha256="d" * 64, reference_labels_sha256=None, run_id="frozen")
            store = ResultStore(paths)
            store.append_batch([{
                "model": "mock/model", "model_revision": "revision1", "device_id": "D1",
                "run_index": 0, "raw_response": "{}",
            }])
            frozen = freeze_raw_evidence(paths)
            self.assertEqual(frozen["generation_count"], 1)
            self.assertEqual(verify_frozen_raw_evidence(paths)["responses_sha256"], frozen["responses_sha256"])
            with self.assertRaisesRegex(RunStateError, "frozen"):
                ResultStore(paths)


if __name__ == "__main__":
    unittest.main()
