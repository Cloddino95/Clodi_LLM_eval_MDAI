from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config import GenerationConfig, ModelConfig
from src.inference import DeviceInput
from src.integrity import IntegrityError, validate_model_integrity


def record(model, revision, device, repeat, parse_ok=True):
    return {
        "model": model,
        "model_revision": revision,
        "device_id": device,
        "run_index": repeat,
        "parse_ok": parse_ok,
        "raw_response": "malformed" if not parse_ok else "{}",
    }


class IntegrityTests(unittest.TestCase):
    def test_complete_grid_counts_parse_failures_as_generated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            rows = [record("m", "r", "D1", 0, False), record("m", "r", "D1", 1)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            report = validate_model_integrity(
                path,
                ModelConfig("m", "r"),
                [DeviceInput("D1", "description")],
                GenerationConfig(2, 0.0, (1, 2), 16),
            )
            self.assertTrue(report.completed)
            self.assertEqual(report.actual_count, 2)

    def test_missing_and_duplicate_keys_prevent_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            duplicate = record("m", "r", "D1", 0)
            path.write_text(json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(IntegrityError, "duplicates=1"):
                validate_model_integrity(
                    path,
                    ModelConfig("m", "r"),
                    [DeviceInput("D1", "description")],
                    GenerationConfig(2, 0.0, (1, 2), 16),
                    raise_on_failure=True,
                )


if __name__ == "__main__":
    unittest.main()

