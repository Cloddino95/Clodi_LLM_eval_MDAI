from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import validate_config_dict
from src.inference import DeterministicMockEngine
from src.preflight import (
    PREFLIGHT_DEVICE,
    PreflightError,
    configuration_preflight,
    dataset_preflight,
    model_asset_preflight,
    smoke_generation_preflight,
)


def write_config_source(path: Path):
    path.write_text("phase: smoke_test\n", encoding="utf-8")


def raw_config(root: Path, dataset: Path, labels: Path, phase: str = "smoke_test"):
    return {
        "phase": phase,
        "study": {"name": "preflight", "run_label": phase},
        "dataset": {"path": str(dataset), "phase": phase, "name": "test"},
        "reference_labels": {"path": str(labels)},
        "models": [{"repository": "mock://deterministic", "revision": "mock-v1"}],
        "generation": {"repeats": 1, "temperature": 0.0, "seeds": [1], "max_tokens": 32},
        "outputs": {"root": str(root / "runs")},
        "analysis": {"enabled": False},
    }


class PreflightTests(unittest.TestCase):
    def test_dataset_and_mock_model_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "devices.csv"
            labels = root / "labels.csv"
            source = root / "config.yaml"
            dataset.write_text("device_id,description\nD1,synthetic\n", encoding="utf-8")
            labels.write_text("device_id,ground_truth\nD1,CLASS_I\n", encoding="utf-8")
            write_config_source(source)
            config = validate_config_dict(raw_config(root, dataset, labels), source)
            configuration_preflight(config)
            checked = dataset_preflight(config)
            self.assertEqual(len(checked.devices), 1)
            assets = model_asset_preflight(
                config.models[0], checked.devices,
                generation_max_tokens=config.generation.max_tokens,
                phase=config.phase,
            )
            self.assertEqual(assets.gpu_memory_estimate["kind"], "mock")
            smoke = smoke_generation_preflight(DeterministicMockEngine(), config.models[0], config)
            self.assertEqual(smoke["device_id"], PREFLIGHT_DEVICE.device_id)
            self.assertTrue(smoke["nonempty_response"])

    def test_reference_ids_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "devices.csv"
            labels = root / "labels.csv"
            source = root / "config.yaml"
            dataset.write_text("device_id,description\nD1,synthetic\n", encoding="utf-8")
            labels.write_text("device_id,ground_truth\nOTHER,CLASS_I\n", encoding="utf-8")
            write_config_source(source)
            config = validate_config_dict(raw_config(root, dataset, labels), source)
            with self.assertRaisesRegex(PreflightError, "do not match"):
                dataset_preflight(config)

    def test_mock_is_forbidden_outside_smoke_test(self):
        from src.config import ModelConfig
        with self.assertRaisesRegex(PreflightError, "only for smoke_test"):
            model_asset_preflight(
                ModelConfig("mock://deterministic", "mock-v1"),
                (PREFLIGHT_DEVICE,),
                generation_max_tokens=32,
                phase="development",
            )


if __name__ == "__main__":
    unittest.main()
