from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.config import ConfigError, GenerationConfig, ModelConfig, validate_config_dict
from src.inference import DeviceInput, GeneratedText, generate_repeat
from src.paths import PROJECT_ROOT, resolve_config_path, resolve_project_path
from src.prompt import SYSTEM_PROMPT


class FakeEngine:
    def generate(self, messages, *, temperature, max_tokens, seed):
        payload = json.dumps({
            "classification": "CLASS_I",
            "rationale": "test",
            "applicable_regulation": "MDR",
            "applicable_provision": "Annex VIII",
            "missing_information": [],
        })
        return [GeneratedText(payload, 10, 5, "stop") for _ in messages]


def valid_config_dict():
    return {
        "phase": "smoke_test",
        "study": {"name": "test", "run_label": "smoke"},
        "dataset": {"path": "data/devices.csv", "phase": "smoke_test", "name": "synthetic"},
        "reference_labels": {"path": "data/labels.csv"},
        "models": [{"repository": "example/model", "revision": "abc123"}],
        "generation": {"repeats": 2, "temperature": 0.0, "seeds": [10, 11], "max_tokens": 64},
        "outputs": {"root": "../runs"},
        "analysis": {"enabled": True, "specification": None},
    }


class PathTests(unittest.TestCase):
    def test_project_relative_path_is_cwd_independent(self):
        self.assertEqual(resolve_project_path("README.md"), (PROJECT_ROOT / "README.md").resolve())

    def test_config_relative_path_uses_config_parent(self):
        config = PROJECT_ROOT / "configs" / "example.yaml"
        self.assertEqual(
            resolve_config_path("../data/devices.csv", config),
            (PROJECT_ROOT / "data" / "devices.csv").resolve(),
        )


class ConfigTests(unittest.TestCase):
    def test_valid_config_resolves_paths(self):
        source = PROJECT_ROOT / "configs" / "study.yaml"
        config = validate_config_dict(valid_config_dict(), source)
        self.assertEqual(config.phase, "smoke_test")
        self.assertEqual(config.generation.seeds, (10, 11))
        self.assertEqual(config.dataset.path, (PROJECT_ROOT / "configs/data/devices.csv").resolve())

    def test_unknown_field_is_rejected(self):
        raw = valid_config_dict()
        raw["surprise"] = True
        with self.assertRaisesRegex(ConfigError, "Unknown field"):
            validate_config_dict(raw, Path("study.yaml").resolve())

    def test_seed_count_must_match_repeats(self):
        raw = valid_config_dict()
        raw["generation"]["seeds"] = [1]
        with self.assertRaisesRegex(ConfigError, "one integer per repeat"):
            validate_config_dict(raw, Path("study.yaml").resolve())

    def test_configuration_phase_must_match_dataset_phase(self):
        raw = valid_config_dict()
        raw["phase"] = "development"
        raw["dataset"]["phase"] = "final"
        with self.assertRaisesRegex(ConfigError, "cannot consume"):
            validate_config_dict(raw, Path("study.yaml").resolve())


class InferenceTests(unittest.TestCase):
    def test_shared_repeat_uses_configured_seed_and_parser(self):
        model = ModelConfig("example/model", "abc123")
        generation = GenerationConfig(2, 0.0, (10, 11), 64)
        records = generate_repeat(
            FakeEngine(), model, [DeviceInput("D1", "description")], generation, 1
        )
        self.assertEqual(records[0]["seed"], 11)
        self.assertEqual(records[0]["classification"], "CLASS_I")
        self.assertEqual(records[0]["raw_response"], records[0]["raw"])
        self.assertIn("Follow these rules", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
