"""The preflight tokenizer check must match how vLLM inference builds prompts."""

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.preflight import PreflightError, _load_preflight_tokenizer, _prompt_token_count

MODEL = SimpleNamespace(repository="example-org/example-model", revision="0123456789abcdef")
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


class _Tok:
    def __init__(self, result=None, error=None):
        self._result, self._error = result, error

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if self._error:
            raise self._error
        return self._result


class PromptTokenCountTests(unittest.TestCase):
    def test_plain_token_list_is_counted(self):
        self.assertEqual(_prompt_token_count(_Tok([1, 2, 3]), MESSAGES, MODEL), 3)

    def test_dict_output_counts_input_ids_not_keys(self):
        packet = {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]}
        self.assertEqual(_prompt_token_count(_Tok(packet), MESSAGES, MODEL), 5)

    def test_mistral_style_tokenizer_without_chat_template_attribute_works(self):
        tok = _Tok(list(range(7)))
        self.assertFalse(hasattr(tok, "chat_template"))
        self.assertEqual(_prompt_token_count(tok, MESSAGES, MODEL), 7)

    def test_failing_template_becomes_preflight_error(self):
        with self.assertRaisesRegex(PreflightError, "Chat template could not be applied"):
            _prompt_token_count(_Tok(error=ValueError("no template")), MESSAGES, MODEL)

    def test_empty_template_output_becomes_preflight_error(self):
        with self.assertRaisesRegex(PreflightError, "produced no tokens"):
            _prompt_token_count(_Tok([]), MESSAGES, MODEL)


class LoadTokenizerTests(unittest.TestCase):
    def _fake_vllm(self, get_tokenizer):
        vllm = types.ModuleType("vllm")
        tokenizers = types.ModuleType("vllm.tokenizers")
        tokenizers.get_tokenizer = get_tokenizer
        vllm.tokenizers = tokenizers
        return {"vllm": vllm, "vllm.tokenizers": tokenizers}

    def test_tokenizer_is_loaded_through_vllm_from_the_model_folder(self):
        loader = mock.Mock(return_value="the-tokenizer")
        with mock.patch.dict(sys.modules, self._fake_vllm(loader)):
            result = _load_preflight_tokenizer(MODEL, Path("/models/example"))
        self.assertEqual(result, "the-tokenizer")
        loader.assert_called_once_with(str(Path("/models/example")), revision=MODEL.revision)

    def test_loading_failure_becomes_preflight_error(self):
        loader = mock.Mock(side_effect=OSError("missing tokenizer"))
        with mock.patch.dict(sys.modules, self._fake_vllm(loader)):
            with self.assertRaisesRegex(PreflightError, "Tokenizer preflight failed"):
                _load_preflight_tokenizer(MODEL, Path("/models/example"))


if __name__ == "__main__":
    unittest.main()
