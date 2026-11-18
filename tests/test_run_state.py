from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.config import validate_config_dict
from src.run_state import (
    RunStateError,
    create_run,
    generate_run_id,
    load_manifest,
    setup_run_logger,
    update_manifest,
)


def config_for(root: Path, source: Path):
    source.write_text("phase: smoke_test\n", encoding="utf-8")
    return validate_config_dict({
        "phase": "smoke_test",
        "study": {"name": "lifecycle", "run_label": "smoke test"},
        "dataset": {"path": "devices.csv", "phase": "smoke_test", "name": "synthetic"},
        "reference_labels": {"path": None},
        "models": [{"repository": "mock/model", "revision": "abc123"}],
        "generation": {"repeats": 1, "temperature": 0.0, "seeds": [7], "max_tokens": 32},
        "outputs": {"root": str(root)},
        "analysis": {"enabled": False},
    }, source)


class RunStateTests(unittest.TestCase):
    def test_run_id_is_human_readable_and_unique_component_present(self):
        now = datetime(2026, 8, 13, 15, 42, 12, tzinfo=timezone.utc)
        value = generate_run_id("final 01", now=now)
        self.assertRegex(value, r"^final_01_20260813_154212_[0-9a-f]{4}$")

    def test_manifest_lifecycle_logging_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.yaml"
            dataset = root / "devices.csv"
            dataset.write_text("device_id,description\nD1,test\n", encoding="utf-8")
            config = config_for(root / "runs", config_file)
            paths = create_run(
                config,
                dataset_sha256="a" * 64,
                reference_labels_sha256=None,
                run_id="fixed_run",
            )
            manifest = load_manifest(paths)
            self.assertEqual(manifest["status"], "created")
            self.assertTrue(paths.raw.is_dir())
            self.assertTrue(paths.figures.is_dir())
            logger = setup_run_logger(paths, console=False)
            logger.info("configuration loaded")
            for handler in logger.handlers:
                handler.flush()
            self.assertIn("configuration loaded", paths.log.read_text(encoding="utf-8"))
            updated = update_manifest(paths, status="running", event="execution_started")
            self.assertEqual(updated["status"], "running")
            with self.assertRaisesRegex(RunStateError, "refusing overwrite"):
                create_run(
                    config,
                    dataset_sha256="a" * 64,
                    reference_labels_sha256=None,
                    run_id="fixed_run",
                )
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
