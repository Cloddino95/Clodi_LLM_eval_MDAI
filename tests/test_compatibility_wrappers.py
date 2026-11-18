from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CompatibilityWrapperTests(unittest.TestCase):
    def test_bash_wrapper_contains_no_study_settings(self):
        script = (PROJECT_ROOT / "03_evaluation" / "sweep_models.sh").read_text(encoding="utf-8")
        self.assertIn("run_study.py", script)
        self.assertNotIn("MODELS=", script)
        self.assertNotIn("REPEATS=", script)
        self.assertNotIn("TEMPERATURE=", script)

    def test_run_eval_is_config_driven_thin_worker(self):
        script_path = PROJECT_ROOT / "03_evaluation" / "run_eval.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=PROJECT_ROOT.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)
        source = script_path.read_text(encoding="utf-8")
        self.assertIn("execute_model", source)
        self.assertNotIn('.open("w")', source)

    def test_model_downloader_is_config_driven_and_revision_pinned(self):
        script_path = (
            PROJECT_ROOT
            / "01_general_information_and_requirements"
            / "scripts"
            / "download_models.py"
        )
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=PROJECT_ROOT.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)
        source = script_path.read_text(encoding="utf-8")
        self.assertIn("revision=model.revision", source)
        self.assertNotIn("MODEL_IDS", source)

    def test_prompt_diagnostic_requires_an_exact_revision(self):
        script_path = PROJECT_ROOT / "02_testing" / "diagnose_prompt.py"
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=PROJECT_ROOT.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--revision", result.stdout)
        source = script_path.read_text(encoding="utf-8")
        self.assertIn("revision=args.revision", source)


if __name__ == "__main__":
    unittest.main()
