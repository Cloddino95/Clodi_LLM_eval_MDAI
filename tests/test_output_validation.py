from __future__ import annotations

import json
import unittest

from src.config import GenerationConfig, ModelConfig
from src.inference import DeviceInput, GeneratedText, generate_repeat
from src.prompt import parse_response


def valid_payload(**updates):
    payload = {
        "classification": "CLASS_IIA",
        "rationale": "MDR reasoning",
        "applicable_regulation": "MDR",
        "applicable_provision": "Annex VIII, Rule 11",
        "missing_information": [],
    }
    payload.update(updates)
    return payload


class OneOutputEngine:
    def __init__(self, text, finish_reason="stop"):
        self.text = text
        self.finish_reason = finish_reason

    def generate(self, messages, *, temperature, max_tokens, seed):
        return [GeneratedText(self.text, 10, 10, self.finish_reason)]


class OutputValidationTests(unittest.TestCase):
    def test_valid_response_flags_are_explicit(self):
        parsed = parse_response(json.dumps(valid_payload()))
        self.assertTrue(parsed["parse_success"])
        self.assertTrue(parsed["schema_valid"])
        self.assertTrue(parsed["classification_valid"])
        self.assertTrue(parsed["citation_present"])

    def test_invalid_label_is_preserved_not_blank(self):
        parsed = parse_response(json.dumps(valid_payload(classification="CLASS_IV")))
        self.assertTrue(parsed["parse_success"])
        self.assertFalse(parsed["schema_valid"])
        self.assertFalse(parsed["classification_valid"])
        self.assertEqual(parsed["classification"], "CLASS_IV")
        self.assertIn("invalid_classification", parsed["validation_issues"])

    def test_wrong_missing_information_type_is_not_silently_coerced(self):
        parsed = parse_response(json.dumps(valid_payload(missing_information="needed")))
        self.assertFalse(parsed["missing_information_valid_type"])
        self.assertEqual(parsed["missing_information_raw"], "needed")
        self.assertEqual(parsed["missing_information"], "needed")

    def test_json_repair_and_extraneous_text_are_observable(self):
        text = "Here is the result: " + json.dumps(valid_payload())[:-1] + ",} trailing"
        parsed = parse_response(text)
        self.assertTrue(parsed["parse_success"])
        self.assertTrue(parsed["json_repaired"])
        self.assertTrue(parsed["extraneous_text_present"])

    def test_truncation_is_recorded_without_discarding_raw_text(self):
        raw = json.dumps(valid_payload())
        records = generate_repeat(
            OneOutputEngine(raw, "length"),
            ModelConfig("mock", "revision"),
            [DeviceInput("D1", "description")],
            GenerationConfig(1, 0.0, (1,), 32),
            0,
        )
        self.assertTrue(records[0]["response_truncated"])
        self.assertEqual(records[0]["raw_response"], raw)
        self.assertIn("response_truncated", records[0]["validation_issues"])


if __name__ == "__main__":
    unittest.main()

