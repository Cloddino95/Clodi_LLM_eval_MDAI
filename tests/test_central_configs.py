from __future__ import annotations

import unittest

from src.config import load_config
from src.paths import PROJECT_ROOT


class CentralConfigurationTests(unittest.TestCase):
    def test_all_central_configs_load_and_have_explicit_model_revisions(self):
        for name, phase in (
            ("smoke_test.yaml", "smoke_test"),
            ("development.yaml", "development"),
            ("final.yaml", "final"),
        ):
            with self.subTest(name=name):
                config = load_config(PROJECT_ROOT / "configs" / name)
                self.assertEqual(config.phase, phase)
                self.assertTrue(config.models)
                self.assertTrue(all(model.revision for model in config.models))

    def test_development_data_keeps_reference_labels_out_of_inference_file(self):
        config = load_config(PROJECT_ROOT / "configs" / "development.yaml")
        header = config.dataset.path.read_text(encoding="utf-8").splitlines()[0]
        labels_header = config.reference_labels.path.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header, "device_id,description")
        self.assertEqual(labels_header, "device_id,ground_truth")

    def test_final_config_points_to_sealed_not_development_data(self):
        config = load_config(PROJECT_ROOT / "configs" / "final.yaml")
        self.assertEqual(config.dataset.phase, "final")
        self.assertEqual(config.dataset.path.parent.name, "final")
        self.assertNotIn("development", config.dataset.path.parts)
        self.assertTrue(config.dataset.development_paths)


if __name__ == "__main__":
    unittest.main()
