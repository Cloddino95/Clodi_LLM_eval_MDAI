"""The model preflight must use a model's local_path folder when one is configured."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.preflight import PreflightError, _resolve_cached_model


def _model(local_path):
    return SimpleNamespace(
        repository="example-org/example-model",
        revision="0123456789abcdef0123456789abcdef01234567",
        local_path=str(local_path) if local_path is not None else None,
    )


class LocalPathPreflightTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_local_path_with_config_and_weights_is_used(self):
        (self.folder / "config.json").write_text("{}")
        (self.folder / "model-00001-of-00001.safetensors").write_bytes(b"x")
        self.assertEqual(_resolve_cached_model(_model(self.folder)), self.folder.resolve())

    def test_mistral_style_folder_with_params_json_is_accepted(self):
        (self.folder / "params.json").write_text("{}")
        (self.folder / "consolidated.safetensors").write_bytes(b"x")
        self.assertEqual(_resolve_cached_model(_model(self.folder)), self.folder.resolve())

    def test_missing_local_path_folder_fails(self):
        with self.assertRaisesRegex(PreflightError, "not a directory"):
            _resolve_cached_model(_model(self.folder / "does-not-exist"))

    def test_local_path_without_config_fails(self):
        (self.folder / "model.safetensors").write_bytes(b"x")
        with self.assertRaisesRegex(PreflightError, "no config.json"):
            _resolve_cached_model(_model(self.folder))

    def test_local_path_without_weights_fails(self):
        (self.folder / "config.json").write_text("{}")
        with self.assertRaisesRegex(PreflightError, "no weight files"):
            _resolve_cached_model(_model(self.folder))

    def test_without_local_path_the_cache_lookup_is_still_used(self):
        env = {"HF_HUB_OFFLINE": "1", "HF_HOME": str(self.folder / "empty-hf-home")}
        with mock.patch.dict(os.environ, env):
            with self.assertRaisesRegex(
                PreflightError,
                "not completely available in the local cache|huggingface_hub is not installed",
            ):
                _resolve_cached_model(_model(None))


if __name__ == "__main__":
    unittest.main()
