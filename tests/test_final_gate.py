from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import validate_config_dict
from src.final_gate import (
    FinalFreezeError,
    current_final_freeze_values,
    final_freeze_preflight,
)
from src.preflight import dataset_preflight


def raw_final(root: Path, freeze=None):
    return {
        "phase": "final",
        "study": {"name": "final-dry-run", "run_label": "final"},
        "dataset": {
            "path": str(root / "final.csv"),
            "phase": "final",
            "name": "sealed",
            "development_paths": [str(root / "development.csv")],
        },
        "reference_labels": {"path": str(root / "labels.csv")},
        "models": [{"repository": "example/model", "revision": "1" * 40}],
        "generation": {"repeats": 2, "temperature": 0.0, "seeds": [1, 2], "max_tokens": 64},
        "outputs": {"root": str(root / "runs")},
        "analysis": {"enabled": True, "specification": str(root / "analysis.yaml")},
        "final_freeze": freeze or {},
    }


class FinalGateTests(unittest.TestCase):
    def prepare(self, root: Path):
        (root / "final.csv").write_text(
            "device_id,description\nF1,final unique description\n", encoding="utf-8"
        )
        (root / "development.csv").write_text(
            "device_id,description\nD1,development unique description\n", encoding="utf-8"
        )
        (root / "labels.csv").write_text(
            "device_id,ground_truth\nF1,CLASS_I\n", encoding="utf-8"
        )
        (root / "analysis.yaml").write_text(
            "version: primary_analysis_v1\n", encoding="utf-8"
        )
        source = root / "final.yaml"
        source.write_text("phase: final\n", encoding="utf-8")
        return source

    def test_final_mode_fails_closed_when_freeze_fields_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.prepare(root)
            config = validate_config_dict(raw_final(root), source)
            checked = dataset_preflight(config)
            with self.assertRaisesRegex(FinalFreezeError, "incomplete"):
                final_freeze_preflight(config, checked)

    def test_matching_final_freeze_passes_and_any_change_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.prepare(root)
            unfrozen = validate_config_dict(raw_final(root), source)
            checked = dataset_preflight(unfrozen)
            with patch("src.final_gate.git_commit", return_value="a" * 40):
                values = current_final_freeze_values(unfrozen, checked)
                frozen = validate_config_dict(raw_final(root, values), source)
                self.assertTrue(final_freeze_preflight(frozen, checked)["verified"])
                (root / "analysis.yaml").write_text(
                    "version: primary_analysis_v1\ndescription: changed\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(FinalFreezeError, "mismatch"):
                    final_freeze_preflight(frozen, checked)


if __name__ == "__main__":
    unittest.main()
